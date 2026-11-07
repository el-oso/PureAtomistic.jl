module Barostats

using ..Systems: System, EV_PER_A3_IN_GPA, pressure
using ..Potentials: AbstractStressPotential, stress
using TypeContracts: @contract, @verify, Self

export AbstractBarostat, BerendsenBarostat, scale_factor, rescale!

"""
    AbstractBarostat

Supertype of the barostats. A barostat type `B` provides
`scale_factor(barostat::B, sys, potential, dt)`, the factor `μ` by which a time step of `dt` fs
scales the cell and the positions, computed at the start of the step from the stress in
`potential` (an [`AbstractStressPotential`](@ref) holding the result of the current
configuration) and the velocities of `sys`. The driver applies it with [`rescale!`](@ref)
after the drift of the step and before its force evaluation, so a barostatted step costs one
force evaluation and leaves `potential.result` at the rescaled configuration.
"""
abstract type AbstractBarostat end

"""
    scale_factor(barostat, sys, potential, dt) -> μ

The factor (of the element type of `sys`) by which one time step of `dt` fs scales the cell
and the positions of `sys`; see [`AbstractBarostat`](@ref). Nothing is changed. Throws when
the step would make the cell degenerate.
"""
function scale_factor end

"""
    rescale!(sys, μ) -> nothing

Multiply the cell and every position of `sys` by `μ`, leaving the fractional coordinates and
the velocities unchanged. The potential is not evaluated.
"""
function rescale!(sys::System, μ::Real)
    _scale!(sys.cell, μ)
    _scale!(sys.positions, μ)
    return nothing
end

# Multiply every entry of `A` by `μ` in place: one product per entry. `map!` runs on device
# arrays too; the in-place broadcast `A .*= μ` would do the same, but its aliasing check holds an
# allocating branch that the allocation gates of the barostat steps reject.
function _scale!(A::AbstractArray, μ)
    map!(x -> x * μ, A, A)
    return nothing
end

include("berendsen.jl")

@contract AbstractBarostat "A barostat coupled to the time step." begin
    scale_factor(::Self, ::System, ::AbstractStressPotential, ::Real)::AbstractFloat => "the cell scale factor of one time step"
end

# Checked at precompile time: every barostat has the contract's method and it infers a float.
@verify BerendsenBarostat trim_compat = true

end
