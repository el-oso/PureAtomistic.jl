"""
    AbstractIntegrator

Supertype of the time integrators. An integrator type `I` provides
`advance!(integrator::I, sys, potential)`, which moves `sys` by one time step and leaves the
energy and the forces of the new configuration in `potential.result`.
"""
abstract type AbstractIntegrator end

"""
    VelocityVerlet(dt)

The velocity Verlet integrator with time step `dt` in fs. One step is

    v ← v + (dt/2) F(x)/m,   x ← x + dt v,   F ← F(x),   v ← v + (dt/2) F(x)/m,

which is the kick-drift-kick form of the Störmer-Verlet method: it uses one force evaluation
per step, is time-reversible and symplectic, and has a global error of order `dt²`. Positions
are advanced in the cell without wrapping, so the unwrapped trajectory stays continuous.

Eliminating the half-step velocity gives the velocity form

    x(t + dt) = x(t) + dt v(t) + (dt²/2) a(t),
    v(t + dt) = v(t) + (dt/2) (a(t) + a(t + dt)),   a = F/m.

The positions-only form is due to L. Verlet, Phys. Rev. 159, 98 (1967); the velocity form is
from W. C. Swope, H. C. Andersen, P. H. Berens and K. R. Wilson, J. Chem. Phys. 76, 637
(1982).

Forces are in eV/Å and masses in amu, so the acceleration is `F / (m · AMU_IN_EV_FS2_PER_A2)`
in Å/fs².
"""
struct VelocityVerlet <: AbstractIntegrator
    dt::Float64
    function VelocityVerlet(dt::Real)
        (isfinite(dt) && dt > 0) ||
            throw(ArgumentError(lazy"pureatomistic: the time step must be positive and finite, got $dt"))
        return new(Float64(dt))
    end
end

"""
    advance!(integrator, sys, potential) -> nothing

Advance `sys` by one step of `integrator` under the forces of `potential`, which must hold the
forces of the current positions in `potential.result` (the state [`energy_forces!`](@ref)
leaves). On return `potential.result` holds the energy and forces of the new positions.
"""
function advance!(vv::VelocityVerlet, sys::System{T}, potential::AbstractPotential)::Nothing where {T}
    Base.require_one_based_indexing(sys.positions, sys.velocities, sys.masses)
    h = T(vv.dt)
    _kick!(sys.velocities, potential.result.forces, sys.masses, h / 2)
    _drift!(sys.positions, sys.velocities, h)
    prepare!(potential, sys.positions, sys.cell)
    energy_forces!(potential, sys.positions, sys.cell)
    _kick!(sys.velocities, potential.result.forces, sys.masses, h / 2)
    return nothing
end

# v ← v + τ F / m for every atom, with the mass in amu and the force in eV/Å.
function _kick!(V::AbstractMatrix{T}, F::AbstractMatrix, m::AbstractVector{T}, τ::T) where {T}
    axes(F) == axes(V) || throw(DimensionMismatch(lazy"pureatomistic: forces axes $(axes(F)) differ from velocities axes $(axes(V))"))
    c = T(AMU_IN_EV_FS2_PER_A2)
    for a in axes(V, 2)
        f = τ / (m[a] * c)
        for k in axes(V, 1)
            V[k, a] += f * F[k, a]
        end
    end
    return nothing
end

# x ← x + h v.
function _drift!(X::AbstractMatrix{T}, V::AbstractMatrix{T}, h::T) where {T}
    for a in axes(X, 2), k in axes(X, 1)
        X[k, a] += h * V[k, a]
    end
    return nothing
end

@contract AbstractIntegrator "A time integrator." begin
    advance!(::Self, ::System, ::AbstractPotential)::Nothing => "advance the system by one time step"
end
