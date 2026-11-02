module PureAtomisticOptimExt

import Optim
import PureAtomistic
using PureAtomistic: AbstractPotential, RelaxResult, prepare!, energy_forces!, relax!
using PureAtomistic.Systems: System

# Evaluate the potential at the flat coordinate vector x (column-major 3 × n), writing the
# configuration into sys.positions, and return the energy; the gradient is minus the forces.
function _evaluate!(G, sys, potential, x)
    P = sys.positions
    copyto!(P, x)
    prepare!(potential, P, sys.cell)
    energy_forces!(potential, P, sys.cell)
    if !isnothing(G)
        copyto!(G, vec(potential.result.forces))
        G .*= -1
    end
    return potential.result.energy
end

"""
    relax!(sys, potential, Optim.LBFGS(); fmax = 1e-3, maxsteps = 10_000) -> RelaxResult

Minimize the energy of `potential` over the positions of `sys` at fixed cell with Optim's
L-BFGS, updating `sys.positions` in place. Optim stops when no gradient component exceeds
`fmax / √3`, which bounds the Euclidean norm of every atomic force by `fmax`; `maxsteps` caps
Optim's iterations. `converged` in the result is decided on the final forces, like the FIRE
method, and `steps` counts Optim iterations.
"""
function PureAtomistic.relax!(
        sys::System{T}, potential::AbstractPotential, method::Optim.LBFGS; fmax::Real = 1.0e-3, maxsteps::Integer = 10_000
    ) where {T}
    PureAtomistic.Integrators._check_relax_arguments(sys, potential, fmax, maxsteps)
    x0 = vec(copy(sys.positions))
    f(x) = _evaluate!(nothing, sys, potential, x)
    g!(G, x) = (_evaluate!(G, sys, potential, x); nothing)
    fg!(G, x) = _evaluate!(G, sys, potential, x)
    objective = Optim.OnceDifferentiable(f, g!, fg!, x0)
    options = Optim.Options(; g_abstol = Float64(fmax) / sqrt(3.0), iterations = Int(maxsteps))
    result = Optim.optimize(objective, x0, method, options)
    # Leave the potential's result at the returned minimizer.
    _evaluate!(nothing, sys, potential, Optim.minimizer(result))
    fm = PureAtomistic.Integrators._max_force(potential.result.forces)
    E = potential.result.energy
    return RelaxResult{T}(fm < fmax, Optim.iterations(result), E, fm)
end

end
