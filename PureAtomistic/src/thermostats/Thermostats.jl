module Thermostats

import Random
using ..Systems: System, AMU_IN_EV_FS2_PER_A2, KB_EV_PER_K, natoms, kinetic_energy, kick!, drift!, no_hook
using ..Potentials: AbstractPotential, prepare!, energy_forces!
using TypeContracts: @contract, @verify, Self

export AbstractThermostat, LangevinBAOAB, BussiThermostat, thermostat_step!, randgamma

"""
    AbstractThermostat

Supertype of the thermostats. A thermostat type `H` provides
`thermostat_step!(thermostat::H, sys, potential, dt, rng, integrate!)`, which advances `sys` by
one time step of `dt` fs at the thermostat's temperature and leaves the energy and the forces of
the new configuration in `potential.result`. `integrate!(sys, potential)` takes one step of the
simulation's integrator; a thermostat either calls it once, with its own velocity updates around
it, or replaces it by its own splitting. Random numbers come from `rng` only.
"""
abstract type AbstractThermostat end

"""
    thermostat_step!(thermostat, sys, potential, dt, rng, integrate!) -> nothing
    thermostat_step!(thermostat, sys, potential, dt, rng, integrate!, after_drift!) -> nothing

Advance `sys` by one time step of `dt` fs under `potential` coupled to `thermostat`, drawing
random numbers from `rng`; see [`AbstractThermostat`](@ref). `after_drift!(sys)` runs after
the last position update and before the force evaluation of the step. A thermostat that moves
the positions only through `integrate!` leaves the hook to `integrate!`, which then carries it:
the seven-argument fallback drops the hook and calls the six-argument method. A thermostat with
its own splitting implements the seven-argument method.
"""
function thermostat_step! end

thermostat_step!(th::AbstractThermostat, sys, potential, dt, rng, integrate!, after_drift!) =
    thermostat_step!(th, sys, potential, dt, rng, integrate!)

include("gamma_sampler.jl")
include("langevin.jl")
include("bussi.jl")

@contract AbstractThermostat "A thermostat coupled to the time step." begin
    thermostat_step!(::Self, ::System, ::AbstractPotential, ::Real, ::Random.AbstractRNG, ::Any)::Nothing => "advance the system by one thermostatted time step"
end

# Checked at precompile time: every thermostat has the contract's method and it infers `Nothing`.
@verify LangevinBAOAB trim_compat = true
@verify BussiThermostat trim_compat = true

end
