"""
    BerendsenBarostat(P0, τ_P, β_T)

Isotropic weak coupling to a pressure bath at `P0` GPa with coupling time `τ_P` fs and an
assumed isothermal compressibility `β_T` in 1/GPa. Each time step of `dt` fs multiplies the
cell and every position by

    μ = [1 - (β_T dt / τ_P) (P0 - P)]^(1/3),

where `P` is the instantaneous pressure in GPa at the start of the step: the potential's part
`-tr(stress)/3` plus the kinetic part `2K/(3V)`, which equals `N_f k_B T/(3V)` for `T` counted
over `N_f` degrees of freedom; with the center-of-mass momentum removed,
`⟨2K/(3V)⟩ = (N - 1) k_B T / V`. `K` is the whole kinetic energy, so the kinetic energy of the
center-of-mass motion counts toward the pressure. The sum is converted from eV/Å³ with
[`EV_PER_A3_IN_GPA`](@ref). Velocities are not scaled.

The factor follows from the target `dP/dt = (P0 - P)/τ_P`: with `β_T = -(1/V) dV/dP` the volume
must change at `dV/dt = -β_T V (P0 - P)/τ_P`, so one step multiplies the volume by
`μ³ = 1 - (β_T dt/τ_P)(P0 - P)`. A pressure above `P0` expands the cell. If `β_T` equals the
system's true compressibility `1/B`, the pressure relaxes as `exp(-t/τ_P)`; in general at the
rate `β_T B/τ_P`.

The method is from H. J. C. Berendsen, J. P. M. Postma, W. F. van Gunsteren, A. DiNola and
J. R. Haak, J. Chem. Phys. 81, 3684 (1984). The update here is re-derived as above, not
transcribed from the paper; the tests verify it (relaxation rate of a Lennard-Jones solid, the
ideal-gas volume, the mean pressure of a thermostatted solid, and the zero-stress lattice
constant of a MACE model).

[`scale_factor`](@ref) computes `μ` in the element type `T` of the system; the driver applies
it with [`rescale!`](@ref) after the drift of the step and before its force evaluation, so the
step costs one force evaluation and `potential.result` belongs to the rescaled configuration.
Fractional coordinates are unchanged by the scaling, so unwrapped positions stay continuous.
Once `μ` rounds to 1 the cell stops relaxing: `x = (β_T dt/τ_P)(P - P0)` must reach `1.5 eps(T)`
for an expanding cell (`μ³ = 1 + x`, floats above 1 are `eps(T)` apart) and `0.75 eps(T)` for a
shrinking one (floats below 1 are `eps(T)/2` apart). In Float32 the pressure therefore settles
within `1.8e-7 τ_P/(β_T dt)` GPa above `P0` or `0.9e-7 τ_P/(β_T dt)` GPa below it; a static
Lennard-Jones crystal with `τ_P/(β_T dt) = 1000` stops at `P - P0 = 1.78e-4` and `-8.9e-5` GPa.

A step whose volume factor `μ³` is not positive and finite throws an `ArgumentError` before
anything moves. The constructor throws an `ArgumentError` unless `τ_P` and `β_T` are positive
and finite and `P0` is finite.
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

function scale_factor(b::BerendsenBarostat, sys::System{T}, potential::AbstractStressPotential, dt::Real)::AbstractFloat where {T}
    # The force evaluation after the rescaling needs a 1-based cell; checking here fails before
    # anything moves.
    Base.require_one_based_indexing(sys.cell)
    P = pressure(sys, stress(potential)) * T(EV_PER_A3_IN_GPA)
    μ3 = 1 - T(b.β_T * dt / b.τ_P) * (T(b.P0) - P)
    (isfinite(μ3) && μ3 > 0) ||
        throw(ArgumentError(lazy"pureatomistic: the Berendsen step would make the cell degenerate (volume factor $μ3 at pressure $P GPa)"))
    return cbrt(μ3)
end
