# Compares BLAS backends on `mace_energy_forces!`: OpenBLAS (Julia's default), PureBLAS and,
# on macOS, Apple Accelerate. One thread. Run it periodically: it is the standing check of
# PureBLAS against OpenBLAS and Accelerate on PureMLIP's own matrix products.
#
# Environment: PureBLAS and AppleAccelerate are not dependencies of PureMLIP or of the bench
# project, so use a separate project and add them there. From the repository root:
#
#   julia --project=<dir> -e 'using Pkg; Pkg.develop(path = "PureMLIP"); Pkg.add(["Chairmarks", "JSON"])'
#   julia --project=<dir> -e 'using Pkg; Pkg.develop(path = "<PureBLAS checkout>"); Pkg.add("AppleAccelerate")'
#   julia -t 1 --project=<dir> PureMLIP/bench/blas_compare.jl
#
# A backend whose package is missing from the active project is skipped. AppleAccelerate is
# macOS-only. The PureBLAS commit is read from `git` in its checkout; it is "unknown" when
# PureBLAS is not a git checkout.
#
#   PUREMLIP_CHECKPOINT_DIR  directory with mace-mpa-0-medium.model
#   PUREMLIP_BENCH_CASES     cases as "N:precision,...", precision 64 or 32 (default "5:64,5:32")
#   PUREMLIP_BENCH_ROUNDS    interleaved rounds (default 5)
#   PUREMLIP_BENCH_OUT       JSON output path (default results/blas_compare_mac.json)
#
# Each round runs, for every case in turn, the backends in a fixed order (OpenBLAS, PureBLAS,
# Accelerate), with an untimed warm-up evaluation after each switch and three Chairmarks
# samples (one evaluation each), so that slow drift of the machine enters every backend alike.
# A round whose swap usage grew is discarded and repeated.
#
# The backend is switched through libblastrampoline: every switch first restores OpenBLAS alone
# (`lbt_forward(...; clear = true)`), then activates the backend. Loading AppleAccelerate
# already forwards BLAS to Accelerate, so the OpenBLAS arm is only valid after that reset;
# the active library is verified from the `dgemm` forward after each switch.
using PureMLIP, Chairmarks, JSON, LinearAlgebra, Printf, Statistics

BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("run with julia -t 1")
include(joinpath(@__DIR__, "..", "test", "reference", "structures.jl"))

const CK = joinpath(ENV["PUREMLIP_CHECKPOINT_DIR"], "mace-mpa-0-medium.model")
const SPECS = [
    (parse(Int, a), b == "64" ? Float64 : Float32) for (a, b) in
        split.(split(get(ENV, "PUREMLIP_BENCH_CASES", "5:64,5:32"), ','), ':')
]
const ROUNDS = parse(Int, get(ENV, "PUREMLIP_BENCH_ROUNDS", "5"))
const OUT = get(ENV, "PUREMLIP_BENCH_OUT", joinpath(@__DIR__, "results", "blas_compare_mac.json"))
const SAMPLES = 3
const ATTEMPTS = 3
# The largest inner dimension of any matrix product in the model.
const ROUNDOFF_N = 1280
const ROUNDOFF_C = 10

swap_used_mb() = parse(Float64, match(r"used = ([0-9.]+)M", read(`sysctl -n vm.swapusage`, String))[1])
libraries() = [basename(l.libname) for l in BLAS.get_config().loaded_libs]

# Path of the OpenBLAS that Julia loaded, taken before any other backend is loaded.
const OPENBLAS = only(l.libname for l in BLAS.get_config().loaded_libs if occursin("openblas", l.libname))

# Library file name that owns the `dgemm` forward; "unknown" for code outside a shared library.
function dgemm_owner()
    p = BLAS.lbt_get_forward("dgemm_", :ilp64)
    info = Ref{NTuple{4, Ptr{Cvoid}}}()
    ccall(:dladdr, Cint, (Ptr{Cvoid}, Ptr{NTuple{4, Ptr{Cvoid}}}), p, info) == 0 && return "unknown"
    return basename(unsafe_string(Ptr{UInt8}(info[][1])))
end

function git_commit(dir)
    try
        c = readchomp(`git -C $dir rev-parse --short=8 HEAD`)
        dirty = !isempty(readchomp(`git -C $dir status --porcelain`))
        return dirty ? c * "-dirty" : c
    catch
        return "unknown"
    end
end

# Backends available in the active project, in measurement order.
const MODULES = Dict{String, Module}()
for (label, pkg) in (("PureBLAS", :PureBLAS), ("Accelerate", :AppleAccelerate))
    (label == "Accelerate" && !Sys.isapple()) && continue
    Base.find_package(String(pkg)) === nothing && (println("skipping ", label, ": ", pkg, " is not in the active project"); continue)
    MODULES[label] = Base.require(Main, pkg)
end
const BACKENDS = ["OpenBLAS"; filter(in(keys(MODULES)), ["PureBLAS", "Accelerate"])]

openblas!() = (BLAS.lbt_forward(OPENBLAS; clear = true); BLAS.set_num_threads(1); nothing)

function switch!(backend)
    openblas!()
    if backend == "PureBLAS"
        Base.invokelatest(MODULES[backend].activate)
        Base.invokelatest(MODULES[backend].is_active) || error("PureBLAS did not become active")
    elseif backend == "Accelerate"
        Base.invokelatest(MODULES[backend].load_accelerate)
        Base.invokelatest(MODULES[backend].set_num_threads, 1)
        Base.invokelatest(MODULES[backend].get_num_threads) == 1 || error("Accelerate is not single-threaded")
    end
    owner = dgemm_owner()
    ok = backend == "OpenBLAS" ? occursin("openblas", owner) : backend == "Accelerate" ? owner == "libBLAS.dylib" : owner == "unknown"
    ok || error("backend ", backend, " requested but dgemm is owned by ", owner)
    return (; backend, dgemm_owner = owner, loaded_libraries = libraries())
end

function make_case(N, T, ck)
    cu = cu_bench(N)
    model = MACEModel{T}(ck)
    positions = Matrix{T}(cu.positions')
    cell = Matrix{T}(cu.cell')
    z = cu.atomic_numbers
    nl = neighbor_list(positions, cell, model.r_max; extra_capacity = 64)
    ws = MACEWorkspace{T}(model, length(z), nl.n_edges)
    res = MACEResult{T}(length(z))
    eval! = () -> mace_energy_forces!(res, ws, model, positions, cell, z, nl)
    return (; N, T, n = length(z), n_edges = nl.n_edges, res, eval!, key = "cu$(N)_$(T)")
end

snapshot(res) = (; energy = res.energy, forces = copy(res.forces), virial = copy(res.virial), stress = copy(res.stress))

# Differences of a backend's result from OpenBLAS's, against the rounding bound
# c·eps(T)·N times the scale of the quantity.
function agreement(T, ref, got)
    bound = ROUNDOFF_C * eps(T) * ROUNDOFF_N
    err(a, b) = maximum(abs, a .- b) / max(maximum(abs, a), floatmin(T))
    e_energy = abs(ref.energy - got.energy) / abs(ref.energy)
    e_forces, e_virial, e_stress = err(ref.forces, got.forces), err(ref.virial, got.virial), err(ref.stress, got.stress)
    return (; rel_energy_error = Float64(e_energy), rel_forces_error = Float64(e_forces), rel_virial_error = Float64(e_virial), rel_stress_error = Float64(e_stress), bound = Float64(bound), within_bound = max(e_energy, e_forces, e_virial, e_stress) <= bound)
end

function main()
    ck = load_mace_checkpoint(CK)
    cases = [make_case(N, T, ck) for (N, T) in SPECS]
    samples = Dict(c.key => Dict(b => Vector{Vector{Float64}}() for b in BACKENDS) for c in cases)
    discarded = Dict(c.key => Any[] for c in cases)
    swaps = Dict(c.key => Any[] for c in cases)
    snaps = Dict(c.key => Dict{String, Any}() for c in cases)
    active = Dict{String, Any}()
    for r in 1:ROUNDS, c in cases
        for attempt in 1:ATTEMPTS
            sw0 = swap_used_mb()
            round_samples = Dict{String, Vector{Float64}}()
            for b in BACKENDS
                active[b] = switch!(b)
                c.eval!()
                r == 1 && attempt == 1 && (snaps[c.key][b] = snapshot(c.res))
                round_samples[b] = [s.time for s in (@be c.eval!() evals = 1 samples = SAMPLES seconds = 600).samples]
            end
            sw1 = swap_used_mb()
            if sw1 > sw0 + 1
                println("swap grew ", sw0, " -> ", sw1, " MB during ", c.key, " round ", r, "; round discarded")
                push!(discarded[c.key], (; round = r, swap_start_mb = sw0, swap_end_mb = sw1, samples_s = round_samples))
                attempt == ATTEMPTS && error("swap keeps growing; the machine is paging")
                continue
            end
            for b in BACKENDS
                push!(samples[c.key][b], round_samples[b])
            end
            push!(swaps[c.key], (; round = r, swap_start_mb = sw0, swap_end_mb = sw1))
            @printf("round %d  %s  %s\n", r, c.key, join((@sprintf("%s %.4f s", b, median(round_samples[b])) for b in BACKENDS), "  "))
            break
        end
    end
    openblas!()

    entries = []
    failures = String[]
    for c in cases
        s = samples[c.key]
        med(b) = median(reduce(vcat, s[b]))
        backends = []
        for b in BACKENDS
            per_round = median.(s[b])
            ratios = per_round ./ median.(s["OpenBLAS"])
            push!(
                backends, (;
                    backend = b, active = active[b], samples_s = s[b], round_medians_s = per_round,
                    median_s = med(b), ratio_over_openblas = med(b) / med("OpenBLAS"),
                    round_ratios_over_openblas = ratios, ratio_min = minimum(ratios), ratio_max = maximum(ratios),
                    agreement_with_openblas = b == "OpenBLAS" ? nothing : agreement(c.T, snaps[c.key]["OpenBLAS"], snaps[c.key][b]),
                )
            )
            a = backends[end].agreement_with_openblas
            a !== nothing && !a.within_bound && push!(failures, "$(c.key) $b: $a")
        end
        push!(entries, (; system = "cu_bench($(c.N))", n_atoms = c.n, n_edges = c.n_edges, precision = string(c.T), backends, swaps = swaps[c.key], discarded_rounds = discarded[c.key]))
    end
    versions = Dict{String, Any}(
        "julia" => string(VERSION), "PureMLIP" => string(pkgversion(PureMLIP)),
        "macos" => Sys.isapple() ? strip(read(`sw_vers -productVersion`, String)) : "",
        "openblas" => basename(OPENBLAS),
    )
    haskey(MODULES, "PureBLAS") && (versions["PureBLAS"] = string(pkgversion(MODULES["PureBLAS"])); versions["PureBLAS_commit"] = git_commit(pkgdir(MODULES["PureBLAS"])))
    haskey(MODULES, "Accelerate") && (versions["AppleAccelerate"] = string(pkgversion(MODULES["Accelerate"])))
    doc = (;
        schema = "puremlip-blas-compare-v1",
        machine = Sys.isapple() ? strip(read(`sysctl -n machdep.cpu.brand_string`, String)) : string(Sys.CPU_NAME),
        versions, model = "mace-mpa-0-medium", threads = 1, blas_threads = BLAS.get_num_threads(),
        samples_per_round = SAMPLES, rounds = ROUNDS,
        timed = "mace_energy_forces! with a fixed neighbor list, one thread, backends interleaved within each round",
        agreement_bound = "c*eps(T)*N relative to the quantity's max magnitude, c = $ROUNDOFF_C, N = $ROUNDOFF_N",
        cases = entries,
    )
    open(io -> (JSON.json(io, doc; pretty = true); println(io)), OUT, "w")
    isempty(failures) || error("results differ from OpenBLAS beyond the rounding bound:\n", join(failures, "\n"))
    return
end
main()
