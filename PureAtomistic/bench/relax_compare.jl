# Geometry optimization of the two 256-site Cu cells with MACE-MPA-0: FIRE, FIRE2, TrustRegionBFGS
# and TrustRegionLBFGS (this package) against Optim's L-BFGS (relax_methods.jl).
#
#   PUREMLIP_CHECKPOINT_DIR=<dir> julia -t 1 --project=PureAtomistic/bench PureAtomistic/bench/relax_compare.jl
#
# Recorded per run: force evaluations, wall time, final energy, the largest atomic force of a fresh
# evaluation at the returned positions, and the size of the optimizer state. ROUNDS rounds run every
# method on every cell, in an order rotated by the round, so a drift of the host does not favor one
# method; the median wall time over rounds is kept. Speed-ups are against FIRE, as the ratio of
# median wall times (and of evaluation counts). One thread, BLAS on one thread. The peak memory of
# each method is measured in a process of its own by relax_single.jl.
#
# Writes PureAtomistic/bench/results/relax_compare_mac.json (or PUREATOMISTIC_RELAX_OUT).
using JSON, Statistics
include(joinpath(@__DIR__, "cells.jl"))
include(joinpath(@__DIR__, "relax_methods.jl"))

BLAS.set_num_threads(1)
const ROUNDS = parse(Int, get(ENV, "PUREATOMISTIC_RELAX_ROUNDS", "3"))

# PUREATOMISTIC_RELAX_SMOKE=1 runs the script end to end on PureMLIP's tiny checkpoint and the
# rattled 4-atom cell, to check it before a long run; its output is not a measurement.
const SMOKE = get(ENV, "PUREATOMISTIC_RELAX_SMOKE", "") == "1"
model = SMOKE ? MACEModel(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint())) : MACEModel(load_mace_checkpoint(mpa0_path()))
const RUN_CELLS = SMOKE ? (tiny = tiny,) : CELLS
for (_, f) in METHODS
    run_one(model, tiny, f)
end
runs = Dict{Tuple{String, String}, Vector{Any}}()
for r in 1:ROUNDS
    for (name, build) in pairs(RUN_CELLS)
        for (mname, f) in circshift(METHODS, r - 1)
            res = run_one(model, build, f)
            push!(get!(runs, (String(name), mname), Any[]), res)
            @printf(
                "round %d %-8s %-18s evals %4d  %7.1f s  E %.7f eV  fmax %.2e  %s\n",
                r, name, mname, res.evals, res.seconds, res.energy, res.fmax, string(res.extra)
            )
            flush(stdout)
        end
    end
end
cells = Dict{String, Any}[]
for name in keys(RUN_CELLS)
    rows = Dict{String, Any}[]
    ref = median(x.seconds for x in runs[(String(name), "FIRE")])
    ref_evals = median(x.evals for x in runs[(String(name), "FIRE")])
    for (mname, _) in METHODS
        rs = runs[(String(name), mname)]
        med = median(x.seconds for x in rs)
        ev = median(x.evals for x in rs)
        push!(
            rows, Dict(
                "method" => mname, "median_seconds" => med, "seconds" => [x.seconds for x in rs],
                "evals" => [x.evals for x in rs], "iterations" => [x.iterations for x in rs],
                "energy" => [x.energy for x in rs], "fmax" => [x.fmax for x in rs],
                "state_bytes" => first(rs).state_bytes, "extra" => [x.extra for x in rs],
                "converged" => all(x.fmax < FMAX for x in rs),
                "speedup_vs_FIRE" => ref / med, "evals_ratio_vs_FIRE" => ref_evals / ev,
            )
        )
        @printf("%-8s %-18s median %7.1f s  speed-up vs FIRE %.2f  evals %d\n", name, mname, med, ref / med, ev)
    end
    push!(cells, Dict("cell" => String(name), "methods" => rows))
end
out = Dict{String, Any}(
    "description" => "Relaxation of MPA-0 Cu cells to fmax = 1e-3 eV/Å; FIRE is the reference of the speed-ups",
    "fmax" => FMAX, "rounds" => ROUNDS, "julia" => string(VERSION), "cpu" => host_cpu(),
    "threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
    "optim" => string(pkgversion(Optim)), "cells" => cells,
)
path = get(ENV, "PUREATOMISTIC_RELAX_OUT", joinpath(@__DIR__, "results", "relax_compare_mac.json"))
open(io -> (JSON.json(io, out; pretty = true); println(io)), path, "w")
println("wrote ", path)
