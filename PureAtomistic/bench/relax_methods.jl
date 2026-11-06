# The relaxers compared by relax_compare.jl and relax_single.jl: FIRE, FIRE2, TrustRegionBFGS and
# TrustRegionLBFGS (this package, default settings) and Optim's L-BFGS. Optim is a dependency of
# this bench environment only.
#
# Every method relaxes from the same start to fmax = 1e-3 eV/Å (Optim: no gradient component above
# fmax/√3, which bounds every atomic force by fmax). Force evaluations are counted by a wrapper
# around the potential; for Optim the calls of the objective, of the gradient and of the combined
# function are counted separately as well, and each of them evaluates the potential once.
using Optim, Printf
using PureAtomistic.Integrators: FIRE, FIRE2, TrustRegionBFGS, TrustRegionLBFGS, RelaxResult

const FMAX = 1.0e-3

mutable struct OptimCalls
    f::Int
    g::Int
    fg::Int
end

function optim_relax!(sys, potential, method, calls::OptimCalls)
    x0 = vec(copy(sys.positions))
    function eval!(G, x)
        copyto!(sys.positions, x)
        prepare!(potential, sys.positions, sys.cell)
        energy_forces!(potential, sys.positions, sys.cell)
        isnothing(G) || (G .= .-vec(potential.result.forces))
        return potential.result.energy
    end
    f(x) = (calls.f += 1; eval!(nothing, x))
    g!(G, x) = (calls.g += 1; eval!(G, x); nothing)
    fg!(G, x) = (calls.fg += 1; eval!(G, x))
    objective = OnceDifferentiable(f, g!, fg!, x0)
    options = Optim.Options(; g_abstol = FMAX / sqrt(3.0), iterations = 10_000)
    result = optimize(objective, x0, method, options)
    copyto!(sys.positions, Optim.minimizer(result))
    return Optim.iterations(result)
end

# name => (sys, potential) -> (iterations, state bytes, extra), where `extra` records what the
# method reports beyond the common fields.
const METHODS = [
    "FIRE" => function (sys, p)
        opt = FIRE(sys)
        return relax!(sys, p, opt; fmax = FMAX).steps, Base.summarysize(opt), Dict{String, Any}()
    end,
    "FIRE2" => function (sys, p)
        opt = FIRE2(sys)
        return relax!(sys, p, opt; fmax = FMAX).steps, Base.summarysize(opt), Dict{String, Any}()
    end,
    "TrustRegionBFGS" => function (sys, p)
        opt = TrustRegionBFGS(sys)
        return relax!(sys, p, opt; fmax = FMAX).steps, Base.summarysize(opt), Dict{String, Any}()
    end,
    "TrustRegionLBFGS" => function (sys, p)
        opt = TrustRegionLBFGS(sys)
        r = relax!(sys, p, opt; fmax = FMAX)
        return r.steps, Base.summarysize(opt), Dict{String, Any}("boxed" => opt.boxed, "memory" => opt.memory)
    end,
    "Optim.LBFGS" => function (sys, p)
        calls = OptimCalls(0, 0, 0)
        iters = optim_relax!(sys, p, LBFGS(), calls)
        return iters, 0, Dict{String, Any}("f_calls" => calls.f, "g_calls" => calls.g, "fg_calls" => calls.fg)
    end,
]

method(name) = last(METHODS[findfirst(m -> first(m) == name, METHODS)])

function run_one(model, build, f)
    sys = build()
    p = Counting(mace(model, sys))
    t0 = time_ns()
    iters, state_bytes, extra = f(sys, p)
    t = (time_ns() - t0) / 1.0e9
    fm, E = fresh_fmax(model, sys)
    return (; seconds = t, evals = p.evals, iterations = iters, energy = E, fmax = fm, state_bytes, extra)
end

# A rattled 4-atom cell, for compiling every method with the argument types of the runs.
function tiny()
    sys = fcc_cell(A0, 1, :Cu)
    sys.positions .+= 0.05 .* randn(Xoshiro(2), size(sys.positions))
    return sys
end
