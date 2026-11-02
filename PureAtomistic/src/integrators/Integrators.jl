module Integrators

using ..Systems: System, AMU_IN_EV_FS2_PER_A2
using ..Potentials: AbstractPotential, prepare!, energy_forces!
using TypeContracts: @contract, @verify, Self

export AbstractIntegrator, VelocityVerlet, advance!

include("verlet.jl")

# Checked at precompile time: every integrator has the contract's method and it infers `Nothing`.
@verify VelocityVerlet trim_compat = true

end
