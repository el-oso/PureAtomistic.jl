module Integrators

using ..Systems: System, natoms, kick!, drift!, no_hook
using ..Potentials: AbstractPotential, n_atoms, prepare!, energy_forces!
using TypeContracts: @contract, @verify, Self
using LinearAlgebra: I, Diagonal, Symmetric, cholesky!, issuccess, dot, mul!, norm
import PureDAQP
using PureDAQP: ActiveSet

export AbstractIntegrator, VelocityVerlet, advance!
export FIRE, RelaxResult, relax!
export FIRE2, TrustRegionBFGS, TrustRegionLBFGS

include("verlet.jl")
include("../optimize/fire.jl")
include("../optimize/fire2.jl")
# lbfgs.jl first: the `relax!` of quasi_newton.jl takes either trust-region state.
include("../optimize/lbfgs.jl")
include("../optimize/quasi_newton.jl")

# Checked at precompile time: every integrator has the contract's method and it infers `Nothing`.
@verify VelocityVerlet trim_compat = true

end
