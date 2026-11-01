# CPU speed of PureMLIP's MACE evaluation against LAMMPS-MACE, one thread each.
#
#   julia -t 1 --project=PureMLIP/bench PureMLIP/bench/mace_cpu.jl
#
#   PUREMLIP_LAMMPS            the `lmp` binary (built with ML-MACE)
#   PUREMLIP_LAMMPS_MODEL_DIR  directory with mpa0_f64-lammps.pt and mpa0_f32-lammps.pt
#   PUREMLIP_CHECKPOINT_DIR    directory with mace-mpa-0-medium.model
#   PUREMLIP_BENCH_CASES       cases as "N:precision:rounds,...", precision 64 or 32
#                              (default "5:64:5,5:32:5,7:64:3,7:32:3,8:32:3")
#   PUREMLIP_BENCH_OUT         JSON output path (default results/mace_cpu_mac.json)
#   PUREMLIP_BENCH_STATE       samples file; rounds already in it are skipped, so a run
#                              cut short by PUREMLIP_BENCH_BUDGET_S (seconds, default
#                              unlimited, checked between measurements) resumes where it stopped
#
# Each round runs, for every case in turn, one Julia measurement and one LAMMPS run, so that
# slow drift of the machine enters both arms alike.
#
# Julia arm: `mace_energy_forces!` with a fixed neighbor list; `neighbor_list!` is timed
# separately. Three Chairmarks samples (one evaluation each) per round.
# LAMMPS arm: `run 0` (setup and first evaluation), then `run 20` with a zero timestep; the
# time is "Loop time" of `run 20` divided by 20, i.e. one force evaluation of the static
# configuration, setup excluded.
using PureMLIP, Chairmarks, JSON, LinearAlgebra, Printf, Statistics

BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("run with julia -t 1")
include(joinpath(@__DIR__, "..", "test", "reference", "structures.jl"))

const LMP = ENV["PUREMLIP_LAMMPS"]
const MODEL_DIR = ENV["PUREMLIP_LAMMPS_MODEL_DIR"]
const CK = joinpath(ENV["PUREMLIP_CHECKPOINT_DIR"], "mace-mpa-0-medium.model")
const SPECS = [
    (parse(Int, a), b == "64" ? Float64 : Float32, parse(Int, c)) for (a, b, c) in
        split.(split(get(ENV, "PUREMLIP_BENCH_CASES", "5:64:5,5:32:5,7:64:3,7:32:3,8:32:3"), ','), ':')
]
const OUT = get(ENV, "PUREMLIP_BENCH_OUT", joinpath(@__DIR__, "results", "mace_cpu_mac.json"))
const STATE = get(ENV, "PUREMLIP_BENCH_STATE", "")
const DEADLINE = time() + parse(Float64, get(ENV, "PUREMLIP_BENCH_BUDGET_S", "Inf"))
const JULIA_SAMPLES = 3

function write_data(path, cell, positions)
    return open(path, "w") do io
        println(io, "benchmark structure\n")
        println(io, size(positions, 1), " atoms\n1 atom types\n")
        println(io, "0.0 ", cell[1, 1], " xlo xhi\n0.0 ", cell[2, 2], " ylo yhi\n0.0 ", cell[3, 3], " zlo zhi\n")
        println(io, "Atoms # atomic\n")
        for n in axes(positions, 1)
            println(io, n, " 1 ", join(positions[n, :], " "))
        end
    end
end

const LOOP = r"Loop time of ([0-9.eE+-]+) on 1 procs for (\d+) steps"

# Swap in use (MB). Swap growing during a measurement means the machine was paging.
swap_used_mb() = parse(Float64, match(r"used = ([0-9.]+)M", read(`sysctl -n vm.swapusage`, String))[1])

function lammps_step_time(data, model)
    rssfile = tempname()
    cmd = setenv(
        `/usr/bin/time -l -o $rssfile $LMP -in $(joinpath(@__DIR__, "in.bench")) -log none -var data $data -var model $model -var symbols Cu`,
        merge(ENV, Dict("OMP_NUM_THREADS" => "1", "HWLOC_COMPONENTS" => "-darwin")),
    )
    out = IOBuffer()
    run(pipeline(cmd; stdout = out, stderr = out))
    text = String(take!(out))
    rss_mb = parse(Float64, match(r"(\d+)\s+maximum resident set size", read(rssfile, String))[1]) / 2^20
    m = collect(eachmatch(LOOP, text))
    (!isempty(m) && parse(Int, m[end][2]) == 20) || error("unexpected LAMMPS output:\n", text)
    return parse(Float64, m[end][1]) / 20, rss_mb
end

function make_case(N, T, rounds, ck, tmp)
    cu = cu_bench(N)
    model = MACEModel{T}(ck)
    positions = Matrix{T}(cu.positions')
    cell = Matrix{T}(cu.cell')
    z = cu.atomic_numbers
    n = length(z)
    nl = neighbor_list(positions, cell, model.r_max; extra_capacity = 64)
    ws = MACEWorkspace{T}(model, n, nl.n_edges)
    res = MACEResult{T}(n)
    data = joinpath(tmp, "cu$(N)_$(T).data")
    write_data(data, cu.cell, cu.positions)
    mfile = joinpath(MODEL_DIR, T === Float64 ? "mpa0_f64-lammps.pt" : "mpa0_f32-lammps.pt")
    eval! = () -> mace_energy_forces!(res, ws, model, positions, cell, z, nl)
    nlup! = () -> neighbor_list!(nl, positions, cell)
    eval!(); nlup!(); eval!()
    return (; N, T, rounds, n, n_edges = nl.n_edges, ws_bytes = Base.summarysize(ws), nl_bytes = Base.summarysize(nl), eval!, nlup!, data, mfile, key = "cu$(N)_$(T)")
end

load_state() = isfile(STATE) ? JSON.parsefile(STATE) : Dict{String, Any}()
save_state(st) = isempty(STATE) || open(io -> JSON.json(io, st), STATE, "w")

function main()
    tmp = mktempdir()
    ck = load_mace_checkpoint(CK)
    cases = [make_case(N, T, r, ck, tmp) for (N, T, r) in SPECS]
    state = load_state()
    for c in cases
        get!(state, c.key, Dict("julia" => Any[], "nl" => Any[], "lammps" => Any[]))
    end
    for r in 1:maximum(c.rounds for c in cases)
        for c in cases
            s = state[c.key]
            (r <= c.rounds && length(s["lammps"]) < r) || continue
            time() < DEADLINE || (println("budget reached; rerun to resume"); return)
            local t, tn, l, rss, sw0, sw1
            for attempt in 1:3
                sw0 = swap_used_mb()
                t = [b.time for b in (@be c.eval!() evals = 1 samples = JULIA_SAMPLES seconds = 600).samples]
                tn = [b.time for b in (@be c.nlup!() evals = 1 samples = JULIA_SAMPLES seconds = 600).samples]
                l, rss = lammps_step_time(c.data, c.mfile)
                sw1 = swap_used_mb()
                sw1 <= sw0 + 1 && break
                println("swap grew ", sw0, " -> ", sw1, " MB during ", c.key, " round ", r, "; round discarded")
                push!(get!(s, "discarded", Any[]), Dict("round" => r, "swap_start_mb" => sw0, "swap_end_mb" => sw1, "julia" => t, "lammps" => l))
                attempt == 3 && error("swap keeps growing; the machine is paging")
            end
            push!(s["julia"], t); push!(s["nl"], tn); push!(s["lammps"], l)
            for (k, v) in ("rss_mb" => rss, "swap_start_mb" => sw0, "swap_end_mb" => sw1)
                push!(get!(s, k, Any[]), v)
            end
            save_state(state)
            @printf("round %d  %6d atoms %-7s  julia %.4f s  neighbor_list! %.4f s  lammps %.4f s  ratio %.3f\n", r, c.n, c.T, median(t), median(tn), l, l / median(t))
        end
    end
    lammps_help = read(setenv(`$LMP -h`, merge(ENV, Dict("HWLOC_COMPONENTS" => "-darwin"))), String)
    lammps_version = strip(match(r"Large-scale Atomic/Molecular Massively Parallel Simulator - (.+)", lammps_help)[1])
    entries = []
    for c in cases
        s = state[c.key]
        J = [Float64.(x) for x in s["julia"]]; NL = [Float64.(x) for x in s["nl"]]; Lm = Float64.(s["lammps"])
        jall = reduce(vcat, J); nall = reduce(vcat, NL)
        ratios = Lm ./ median.(J)
        push!(
            entries, (;
                system = "cu_bench($(c.N))", n_atoms = c.n, n_edges = c.n_edges, precision = string(c.T),
                workspace_bytes = c.ws_bytes, neighbor_list_bytes = c.nl_bytes,
                julia_samples_s = J, neighbor_list_samples_s = NL, lammps_step_samples_s = Lm,
                julia_median_s = median(jall), neighbor_list_median_s = median(nall), lammps_median_s = median(Lm),
                round_ratios_lammps_over_julia = ratios,
                median_ratio_lammps_over_julia = median(Lm) / median(jall),
                ratio_min = minimum(ratios), ratio_max = maximum(ratios),
                swap_checked = length(get(s, "swap_end_mb", [])) == length(Lm),
lammps_peak_rss_mb = Float64.(get(s, "rss_mb", Float64[])),
swap_start_mb = Float64.(get(s, "swap_start_mb", Float64[])), swap_end_mb = Float64.(get(s, "swap_end_mb", Float64[])),
discarded_rounds = get(s, "discarded", []),
median_ratio_vs_julia_with_neighbor_list = median(Lm) / (median(jall) + median(nall)),
            )
        )
    end
    doc = (;
        schema = "puremlip-mace-cpu-bench-v1",
        machine = strip(read(`sysctl -n machdep.cpu.brand_string`, String)),
        julia_version = string(VERSION), lammps_version, model = "mace-mpa-0-medium",
        threads = 1, blas_threads = BLAS.get_num_threads(), julia_samples_per_round = JULIA_SAMPLES,
        julia_timed = "mace_energy_forces! with a fixed neighbor list; neighbor_list! timed separately",
        lammps_timed = "Loop time of 'run 20' (timestep 0.0, static atoms) divided by 20; setup and 'run 0' excluded",
        cases = entries,
    )
    open(io -> (JSON.json(io, doc; pretty = true); println(io)), OUT, "w")
    return
end
main()
