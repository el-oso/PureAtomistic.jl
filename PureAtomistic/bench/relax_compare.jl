# Geometry optimization of the two Cu cells with MACE-MPA-0: FIRE and TrustRegionBFGS (this
# package) against Optim's L-BFGS, BFGS and conjugate gradient. Optim is a dependency of this
# bench environment only.
#
#   PUREMLIP_CHECKPOINT_DIR=<dir> julia -t 1 --project=PureAtomistic/bench PureAtomistic/bench/relax_compare.jl
#
# Every method relaxes from the same start to fmax = 1e-3 eV/Å (Optim: no gradient component above
# fmax/√3, which bounds every atomic force by fmax). Recorded per run: force evaluations (counted
# by a wrapper around the potential), wall time, final energy and the largest atomic force of a
# fresh evaluation at the returned positions. ROUNDS rounds run every method on every cell, in an
# order rotated by the round, so a drift of the host does not favor one method; the median wall
# time over rounds is kept. Speed-ups are against FIRE, the package default, as the ratio of
# median wall times (and of evaluation counts). One thread, BLAS on one thread.
#
# Writes PureAtomistic/bench/results/relax_compare_mac.json (or PUREATOMISTIC_RELAX_OUT).
using JSON, Printf, Statistics, Optim
include(joinpath(@__DIR__, "cells.jl"))
using PureAtomistic.Integrators: TrustRegionBFGS, RelaxResult

BLAS.set_num_threads(1)
const FMAX = 1.0e-3
const ROUNDS = parse(Int, get(ENV, "PUREATOMISTIC_RELAX_ROUNDS", "3"))

function optim_relax!(sys, potential, method)
    x0 = vec(copy(sys.positions))
    function eval!(G, x)
        copyto!(sys.positions, x)
        prepare!(potential, sys.positions, sys.cell)
        energy_forces!(potential, sys.positions, sys.cell)
        isnothing(G) || (G .= .-vec(potential.result.forces))
        return potential.result.energy
    end
    objective = OnceDifferentiable(x -> eval!(nothing, x), (G, x) -> (eval!(G, x); nothing), eval!, x0)
    options = Optim.Options(; g_abstol = FMAX / sqrt(3.0), iterations = 10_000)
    result = optimize(objective, x0, method, options)
    copyto!(sys.positions, Optim.minimizer(result))
    return Optim.iterations(result)
end

const METHODS = [
    "FIRE" => (sys, p) -> relax!(sys, p; fmax = FMAX).steps,
    "TrustRegionBFGS" => (sys, p) -> relax!(sys, p, TrustRegionBFGS(sys); fmax = FMAX).steps,
    "Optim.LBFGS" => (sys, p) -> optim_relax!(sys, p, LBFGS()),
    "Optim.BFGS" => (sys, p) -> optim_relax!(sys, p, BFGS()),
    "Optim.ConjugateGradient" => (sys, p) -> optim_relax!(sys, p, ConjugateGradient()),
]

function run_one(model, build, f)
    sys = build()
    p = Counting(mace(model, sys))
    t0 = time_ns()
    iters = f(sys, p)
    t = (time_ns() - t0) / 1.0e9
    fm, E = fresh_fmax(model, sys)
    return (; seconds = t, evals = p.evals, iterations = iters, energy = E, fmax = fm)
end

# PUREATOMISTIC_RELAX_SMOKE=1 runs the script end to end on PureMLIP's tiny checkpoint and the
# rattled 4-atom cell, to check it before a long run; its output is not a measurement.
const SMOKE = get(ENV, "PUREATOMISTIC_RELAX_SMOKE", "") == "1"
model = SMOKE ? MACEModel(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint())) : MACEModel(load_mace_checkpoint(mpa0_path()))
# Warm-up: compile every method on a rattled 4-atom cell (the same argument types as the runs).
function tiny()
    sys = fcc_cell(A0, 1, :Cu)
    sys.positions .+= 0.05 .* randn(Xoshiro(2), size(sys.positions))
    return sys
end
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
                "round %d %-8s %-24s evals %4d  %7.1f s  E %.6f eV  fmax %.2e\n",
                r, name, mname, res.evals, res.seconds, res.energy, res.fmax
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
                "converged" => all(x.fmax < FMAX for x in rs),
                "speedup_vs_FIRE" => ref / med, "evals_ratio_vs_FIRE" => ref_evals / ev,
            )
        )
        @printf("%-8s %-24s median %7.1f s  speed-up vs FIRE %.2f  evals %d\n", name, mname, med, ref / med, ev)
    end
    push!(cells, Dict("cell" => String(name), "methods" => rows))
end
out = Dict{String, Any}(
    "description" => "Relaxation of MPA-0 Cu cells to fmax = 1e-3 eV/Å; FIRE is the reference of the speed-ups",
    "fmax" => FMAX, "rounds" => ROUNDS, "julia" => string(VERSION), "cpu" => host_cpu(),
    "threads" => Threads.nthreads(), "optim" => string(pkgversion(Optim)), "cells" => cells,
)
path = get(ENV, "PUREATOMISTIC_RELAX_OUT", joinpath(@__DIR__, "results", "relax_compare_mac.json"))
open(io -> (JSON.json(io, out; pretty = true); println(io)), path, "w")
println("wrote ", path)
