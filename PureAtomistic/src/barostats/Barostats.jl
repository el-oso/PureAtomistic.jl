module Barostats

using ..Systems: System, EV_PER_A3_IN_GPA, pressure
using ..Potentials: AbstractStressPotential, prepare!, energy_forces!, stress
using TypeContracts: @contract, @verify, Self

export AbstractBarostat, BerendsenBarostat, barostat_step!

"""
    AbstractBarostat

Supertype of the barostats. A barostat type `B` provides
`barostat_step!(barostat::B, sys, potential, dt)`, which the driver calls after each time step
of `dt` fs: it rescales the cell and the positions of `sys` toward the barostat's pressure and
leaves the energy, forces and stress of the rescaled configuration in `potential`. The potential
must be an [`AbstractStressPotential`](@ref).
"""
abstract type AbstractBarostat end

"""
    barostat_step!(barostat, sys, potential, dt) -> nothing

Couple `sys` to the pressure bath of `barostat` over one time step of `dt` fs; see
[`AbstractBarostat`](@ref).
"""
function barostat_step! end

include("berendsen.jl")

@contract AbstractBarostat "A barostat coupled to the time step." begin
    barostat_step!(::Self, ::System, ::AbstractStressPotential, ::Real)::Nothing => "rescale the cell toward the target pressure"
end

# Checked at precompile time: every barostat has the contract's method and it infers `Nothing`.
@verify BerendsenBarostat trim_compat = true

end
