"""
    BerendsenBarostat(P0, τ_P, β_T)

Isotropic weak coupling to a pressure bath at `P0` GPa with coupling time `τ_P` fs and an
assumed isothermal compressibility `β_T` in 1/GPa. After each time step of `dt` fs the cell
and every position are multiplied by

    μ = [1 - (β_T dt / τ_P) (P0 - P)]^(1/3),

where `P` is the instantaneous pressure in GPa: the potential's part `-tr(stress)/3` plus the
kinetic part `2K/(3V)` (that is, `N k_B T / V` with `T` counted over `3N` degrees of
freedom), converted from eV/Å³ with [`EV_PER_A3_IN_GPA`](@ref). Velocities are not scaled.

The factor follows from the target `dP/dt = (P0 - P)/τ_P`: with `β_T = -(1/V) dV/dP` the volume
must change at `dV/dt = -β_T V (P0 - P)/τ_P`, so one step multiplies the volume by
`μ³ = 1 - (β_T dt/τ_P)(P0 - P)`. A pressure above `P0` expands the cell. If `β_T` equals the
system's true compressibility `1/B`, the pressure relaxes as `exp(-t/τ_P)`; in general at the
rate `β_T B/τ_P`.

The method is from H. J. C. Berendsen, J. P. M. Postma, W. F. van Gunsteren, A. DiNola and
J. R. Haak, J. Chem. Phys. 81, 3684 (1984). The update here is re-derived as above, not
transcribed from the paper; the tests verify it (relaxation rate of a Lennard-Jones solid, the
ideal-gas volume, and the zero-stress lattice constant of a MACE model).

Fractional coordinates are unchanged by the scaling, so unwrapped positions stay continuous.
The potential is evaluated again at the rescaled configuration, so the forces of the next
step's first kick and `potential.result` belong to it; a barostatted step costs two
evaluations. A step whose volume factor `μ³` is not positive and finite throws an
`ArgumentError` and leaves the system unchanged. The constructor throws an `ArgumentError`
unless `τ_P` and `β_T` are positive and finite and `P0` is finite.
"""
struct BerendsenBarostat <: AbstractBarostat
    P0::Float64
    τ_P::Float64
    β_T::Float64
    function BerendsenBarostat(P0::Real, τ_P::Real, β_T::Real)
        isfinite(P0) || throw(ArgumentError(lazy"pureatomistic: the target pressure must be finite, got $P0"))
        (isfinite(τ_P) && τ_P > 0) ||
            throw(ArgumentError(lazy"pureatomistic: the barostat coupling time must be positive and finite, got $τ_P"))
        (isfinite(β_T) && β_T > 0) ||
            throw(ArgumentError(lazy"pureatomistic: the barostat compressibility must be positive and finite, got $β_T"))
        return new(Float64(P0), Float64(τ_P), Float64(β_T))
    end
end

function barostat_step!(b::BerendsenBarostat, sys::System{T}, potential::AbstractStressPotential, dt::Real)::Nothing where {T}
    # The potential evaluation at the end needs 1-based arrays; checking first leaves `sys`
    # unchanged when it throws.
    Base.require_one_based_indexing(sys.positions, sys.cell)
    P = pressure(sys, stress(potential)) * T(EV_PER_A3_IN_GPA)
    μ3 = 1 - T(b.β_T * dt / b.τ_P) * (T(b.P0) - P)
    (isfinite(μ3) && μ3 > 0) ||
        throw(ArgumentError(lazy"pureatomistic: the Berendsen step would make the cell degenerate (volume factor $μ3 at pressure $P GPa)"))
    μ = cbrt(μ3)
    _scale!(sys.cell, μ)
    _scale!(sys.positions, μ)
    prepare!(potential, sys.positions, sys.cell)
    energy_forces!(potential, sys.positions, sys.cell)
    return nothing
end

# Multiply every entry of `A` by `μ` in place.
function _scale!(A::AbstractArray, μ)
    for i in eachindex(A)
        A[i] *= μ
    end
    return nothing
end
