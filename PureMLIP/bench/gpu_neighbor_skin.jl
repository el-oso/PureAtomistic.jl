# The cost of rebuilding the device neighbor list against the cost of an evaluation, and the
# default skin of `device_neighbor_list` that follows from it, in Float32 with MACE-MPA-0.
#
#   GPU_BACKEND=Metal PUREMLIP_CHECKPOINT_DIR=<dir> julia --project=PureMLIP/bench/gpu/metal PureMLIP/bench/gpu_neighbor_skin.jl
#
#   GPU_BACKEND              Metal (default), CUDA or AMDGPU; the environment is bench/gpu/<backend in lower case>
#   PUREMLIP_CHECKPOINT_DIR  directory with mace-mpa-0-medium.model
#   SKIN_CASES               cu_bench sizes (default "5,14")
#   SKIN_MD                  "auto" (default): run the MD comparison only when the rule below needs it;
#                            "always" or "never"
#   PUREMLIP_BENCH_OUT       JSON output path (default results/gpu_neighbor_skin_<backend>.json)
#
# On macOS, export HWLOC_COMPONENTS=-darwin first (see the KernelAbstractions rule in AGENTS.md).
#
# Measured per case in five rounds, each call synchronized and timed as the median of 20 after two
# warm-up calls, and each figure the median over the rounds:
# `neighbor_list!` on a list without skin (every call rebuilds), `neighbor_list!` on a list with a
# 1 Å skin at unchanged positions (every call keeps the list: the skin criterion and the kernel
# launches that find nothing to do), and `mace_energy_forces!` with the list without skin.
#
# Decision rule: the default skin is 0 when a rebuild costs less than 2 % of an evaluation at every
# case. Otherwise it is the skin in {0.5, 1.0, 1.5} Å whose 100-step MD run is the shortest: velocity
# Verlet at dt = 2 fs from Maxwell-Boltzmann velocities at 1300 K (seeded), with `neighbor_list!`
# and `mace_energy_forces!` every step, timed from the first step to the synchronized end. The runs
# for skins 0, 0.5, 1.0 and 1.5 Å go in two rounds, the second in reverse order, and each skin's
# time is the faster of its two. The choice is made at the largest case, where the speed targets
# are judged; the smaller case is reported beside it.
using PureMLIP, KernelAbstractions, LinearAlgebra, Statistics, Random, JSON

const BACKEND_NAME = get(ENV, "GPU_BACKEND", "Metal")
BACKEND_NAME in ("Metal", "CUDA", "AMDGPU") || error("GPU_BACKEND must be Metal, CUDA or AMDGPU, got ", BACKEND_NAME)
@eval using $(Symbol(BACKEND_NAME))
const VENDOR = getfield(Main, Symbol(BACKEND_NAME))
VENDOR.functional() || error(BACKEND_NAME, " is not functional on this machine")
const backend = BACKEND_NAME == "Metal" ? VENDOR.MetalBackend() : BACKEND_NAME == "CUDA" ? VENDOR.CUDABackend() : VENDOR.ROCBackend()

include(joinpath(pkgdir(PureMLIP), "test", "reference", "structures.jl"))

const CASES = parse.(Int, split(get(ENV, "SKIN_CASES", "5,14"), ","))
const MD_MODE = get(ENV, "SKIN_MD", "auto")
MD_MODE in ("auto", "always", "never") || error("SKIN_MD must be auto, always or never, got ", MD_MODE)
const SKINS = (0.5, 1.0, 1.5)
const THRESHOLD = 0.02
const STEPS = 100
const ROUNDS = 5
const DT = 2.0f0                       # fs
const TEMPERATURE = 1300.0             # K
const KB = 8.617333262e-5              # eV/K
const ACCEL = 9.648533212e-3           # Å/fs² per eV/(Å·amu)
const MASS_CU = 63.546                 # amu

function todevice(A::AbstractArray)
    D = KernelAbstractions.allocate(backend, eltype(A), size(A))
    copyto!(D, A)
    return D
end

function median_ms(f; runs = 20)
    f()
    f()
    KernelAbstractions.synchronize(backend)
    t = Float64[]
    for _ in 1:runs
        t0 = time_ns()
        f()
        KernelAbstractions.synchronize(backend)
        push!(t, (time_ns() - t0) / 1.0e6)
    end
    return median(t)
end

function setup(N, ck)
    cu = cu_bench(N)
    T = Float32
    m = MACEModel{T}(ck)
    P, C = T.(permutedims(cu.positions)), T.(cu.cell)
    n = size(P, 2)
    dm = PureMLIP.device_model(m, backend; elements = [29])
    return (; N, n, P, C, m, dm, dP = todevice(P), dC = todevice(C), dZ = todevice(Int32.(cu.atomic_numbers)))
end

# Spare capacity for the MD runs: the edges at the start grow as the atoms move.
capacity_margin(nl) = cld(nl.capacity, 10)

function costs(s)
    r_max = Float64(s.m.r_max)
    nl0 = PureMLIP.device_neighbor_list(s.dP, s.dC, r_max)
    ws = PureMLIP.device_workspace(s.dm, s.n, nl0.capacity)
    res = PureMLIP.device_result(s.dm, s.n)
    nl1 = PureMLIP.device_neighbor_list(s.dP, s.dC, r_max; skin = 1.0)
    rounds = [
        (
            rebuild = median_ms(() -> neighbor_list!(nl0, s.dP, s.dC)),
            keep = median_ms(() -> neighbor_list!(nl1, s.dP, s.dC)),
            evaluation = median_ms(() -> mace_energy_forces!(res, ws, s.dm, s.dP, s.dC, s.dZ, nl0)),
        ) for _ in 1:ROUNDS
    ]
    Array(nl1.status)[1] == 0 || error("the list with a skin rebuilt at unchanged positions")
    rebuild, keep, evaluation = (median(getfield.(rounds, f)) for f in (:rebuild, :keep, :evaluation))
    return (;
        rebuild_ms = rebuild, keep_ms = keep, evaluation_ms = evaluation, rebuild_fraction = rebuild / evaluation, edges = nl0.capacity,
        rounds_rebuild_ms = getfield.(rounds, :rebuild), rounds_keep_ms = getfield.(rounds, :keep), rounds_evaluation_ms = getfield.(rounds, :evaluation),
    )
end

function md_run(s, skin)
    r_max = Float64(s.m.r_max)
    probe = PureMLIP.device_neighbor_list(s.dP, s.dC, r_max; skin)
    nl = PureMLIP.device_neighbor_list(s.dP, s.dC, r_max; skin, extra_capacity = capacity_margin(probe))
    ws = PureMLIP.device_workspace(s.dm, s.n, nl.capacity)
    res = PureMLIP.device_result(s.dm, s.n)
    rng = Xoshiro(20261107)
    σ = sqrt(KB * TEMPERATURE / MASS_CU * ACCEL)   # Å/fs
    v = todevice(Float32.(σ .* randn(rng, 3, s.n)))
    x = copy(s.dP)
    half = Float32(DT / 2 * ACCEL / MASS_CU)
    neighbor_list!(nl, x, s.dC)
    mace_energy_forces!(res, ws, s.dm, x, s.dC, s.dZ, nl)
    KernelAbstractions.synchronize(backend)
    rebuilds = 0
    t0 = time_ns()
    for _ in 1:STEPS
        v .+= half .* res.forces
        x .+= DT .* v
        neighbor_list!(nl, x, s.dC)
        rebuilds += nl.host_status[1]
        mace_energy_forces!(res, ws, s.dm, x, s.dC, s.dZ, nl)
        v .+= half .* res.forces
    end
    KernelAbstractions.synchronize(backend)
    seconds = (time_ns() - t0) / 1.0e9
    return (; skin, seconds, rebuilds = Int(rebuilds), capacity = nl.capacity)
end

function main()
    ck = load_mace_checkpoint(joinpath(ENV["PUREMLIP_CHECKPOINT_DIR"], "mace-mpa-0-medium.model"))
    record = Dict{String, Any}(
        "backend" => BACKEND_NAME, "precision" => "Float32", "model" => "mace-mpa-0-medium", "julia" => string(VERSION),
        "threshold" => THRESHOLD, "steps" => STEPS, "dt_fs" => DT, "temperature_K" => TEMPERATURE, "cases" => Any[],
    )
    setups = [setup(N, ck) for N in CASES]
    for s in setups
        c = costs(s)
        println(
            "cu_bench($(s.N)): rebuild $(round(c.rebuild_ms; digits = 3)) ms, keep $(round(c.keep_ms; digits = 3)) ms, ",
            "evaluation $(round(c.evaluation_ms; digits = 2)) ms, rebuild/evaluation $(round(100c.rebuild_fraction; digits = 2)) %"
        )
        flush(stdout)
        push!(record["cases"], Dict{String, Any}("cu_bench" => s.N, "atoms" => s.n, (string(k) => v for (k, v) in pairs(c))...))
    end
    under = all(c -> c["rebuild_fraction"] < THRESHOLD, record["cases"])
    run_md = MD_MODE == "always" || (MD_MODE == "auto" && !under)
    if run_md
        for (s, case) in zip(setups, record["cases"])
            runs = Dict{Float64, Vector{Any}}()
            for order in ((0.0, SKINS...), reverse((0.0, SKINS...)))
                for skin in order
                    r = md_run(s, skin)
                    println("cu_bench($(s.N)) skin $skin: $(round(r.seconds; digits = 3)) s, $(r.rebuilds) rebuilds")
                    flush(stdout)
                    push!(get!(runs, skin, Any[]), Dict{String, Any}((string(k) => v for (k, v) in pairs(r))...))
                end
            end
            case["md"] = [Dict{String, Any}("skin" => k, "runs" => v, "seconds" => minimum(r["seconds"] for r in v)) for (k, v) in sort(collect(runs); by = first)]
        end
    end
    default = if under
        0.0
    else
        # Judged at the largest case, as the speed targets are.
        largest = record["cases"][argmax([c["atoms"] for c in record["cases"]])]
        SKINS[argmin([only(md["seconds"] for md in largest["md"] if md["skin"] == skin) for skin in SKINS])]
    end
    record["rebuild_under_threshold"] = under
    record["default_skin"] = default
    out = get(ENV, "PUREMLIP_BENCH_OUT", joinpath(@__DIR__, "results", "gpu_neighbor_skin_$(lowercase(BACKEND_NAME)).json"))
    write(out, JSON.json(record; pretty = 2))
    println("default skin $default Å; wrote $out")
    return nothing
end

main()
