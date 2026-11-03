module Integrators

using ..Systems: System, AMU_IN_EV_FS2_PER_A2, natoms
using ..Potentials: AbstractPotential, n_atoms, prepare!, energy_forces!
using TypeContracts: @contract, @verify, Self
using LinearAlgebra: I, Diagonal, Symmetric, cholesky!, issuccess, dot, mul!
import PureDAQP
using PureDAQP: ActiveSet

export AbstractIntegrator, VelocityVerlet, advance!
export FIRE, RelaxResult, relax!
export TrustRegionBFGS

include("verlet.jl")
include("../optimize/fire.jl")
include("../optimize/quasi_newton.jl")

# Checked at precompile time: every integrator has the contract's method and it infers `Nothing`.
@verify VelocityVerlet trim_compat = true

end
