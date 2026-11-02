"""
    LennardJones{T}(positions, cell; epsilon, sigma, r_cut, extra_capacity_fraction = 0.1)

The 12-6 Lennard-Jones pair potential `V(r) = 4ε((σ/r)¹² - (σ/r)⁶)`, cut at `r_cut` and
shifted so that it vanishes there: every pair closer than `r_cut` contributes
`V(r) - V(r_cut)`, and the forces are the exact gradient of that energy (the force jumps by
`-V'(r_cut)` at the cutoff). `epsilon` is in eV, `sigma` and `r_cut` in Å.
The neighbor list is built for `positions` and `cell` with room for `extra_capacity_fraction`
times as many edges again, and [`prepare!`](@ref) rebuilds it when a configuration needs more.
"""
mutable struct LennardJones{T <: AbstractFloat} <: AbstractPotential
    const epsilon::T
    const sigma::T
    const shift::T
    const extra_capacity_fraction::Float64
    nl::NeighborList{T}
    const result::PotentialResult{T}
    n_regrows::Int
end

function LennardJones{T}(
        positions::AbstractMatrix, cell::AbstractMatrix; epsilon::Real, sigma::Real, r_cut::Real,
        extra_capacity_fraction::Real = 0.1
    ) where {T <: AbstractFloat}
    n = size(positions, 2)
    _check_configuration(T, n, positions, cell)
    (isfinite(epsilon) && epsilon > 0 && isfinite(sigma) && sigma > 0) ||
        throw(ArgumentError(lazy"pureatomistic: epsilon and sigma must be positive and finite, got $epsilon and $sigma"))
    (isfinite(r_cut) && r_cut > 0) || throw(ArgumentError(lazy"pureatomistic: r_cut must be positive and finite, got $r_cut"))
    fraction = _check_capacity_fraction(extra_capacity_fraction)
    ε, σ = T(epsilon), T(sigma)
    nl = _sized_list(positions, cell, r_cut, fraction)
    shift = _lj(ε, σ, T(nl.r_cut))[1]
    return LennardJones{T}(ε, σ, shift, fraction, nl, PotentialResult{T}(n), 0)
end

function LennardJones(
        positions::AbstractMatrix{T}, cell::AbstractMatrix; epsilon::Real, sigma::Real, r_cut::Real,
        extra_capacity_fraction::Real = 0.1
    ) where {T <: AbstractFloat}
    return LennardJones{T}(positions, cell; epsilon, sigma, r_cut, extra_capacity_fraction)
end

n_atoms(p::LennardJones)::Int = p.nl.n_atoms
cutoff(p::LennardJones)::Float64 = p.nl.r_cut

function prepare!(p::LennardJones{T}, positions::AbstractMatrix, cell::AbstractMatrix)::Nothing where {T}
    _check_configuration(T, n_atoms(p), positions, cell)
    _refresh!(p.nl, positions, cell) || _regrow!(p, positions, cell)
    return nothing
end

# The only allocating path of `prepare!`; it runs when a configuration outgrows the list.
@noinline function _regrow!(p::LennardJones{T}, positions::AbstractMatrix{T}, cell::AbstractMatrix{T}) where {T}
    p.nl = _sized_list(positions, cell, p.nl.r_cut, p.extra_capacity_fraction)
    p.n_regrows += 1
    _log_regrow("LennardJones", length(p.nl.senders))
    return true
end

# The energy V(r) and its derivative V'(r) of the uncut potential.
function _lj(ε::T, σ::T, r::T) where {T}
    s6 = (σ / r)^6
    return (4 * ε * (s6 * s6 - s6), -24 * ε * (2 * s6 * s6 - s6) / r)
end

# Each pair appears as two directed edges, so each edge carries half the pair energy.
function energy_forces!(p::LennardJones{T}, positions::AbstractMatrix, cell::AbstractMatrix)::Nothing where {T}
    _check_configuration(T, n_atoms(p), positions, cell)
    nl = p.nl
    F = p.result.forces
    fill!(F, zero(T))
    E = zero(T)
    for e in 1:nl.n_edges
        s, r = nl.senders[e], nl.receivers[e]
        n1, n2, n3 = nl.shifts[1, e], nl.shifts[2, e], nl.shifts[3, e]
        r1 = positions[1, r] - positions[1, s] + (cell[1, 1] * n1 + cell[1, 2] * n2 + cell[1, 3] * n3)
        r2 = positions[2, r] - positions[2, s] + (cell[2, 1] * n1 + cell[2, 2] * n2 + cell[2, 3] * n3)
        r3 = positions[3, r] - positions[3, s] + (cell[3, 1] * n1 + cell[3, 2] * n2 + cell[3, 3] * n3)
        d = sqrt(r1 * r1 + r2 * r2 + r3 * r3)
        V, dV = _lj(p.epsilon, p.sigma, d)
        E += V - p.shift
        # ∂(V/2)/∂x_r = V'/2 · r/d; the sender gets the opposite.
        g = dV / (2 * d)
        F[1, r] -= g * r1
        F[2, r] -= g * r2
        F[3, r] -= g * r3
        F[1, s] += g * r1
        F[2, s] += g * r2
        F[3, s] += g * r3
    end
    p.result.energy = E / 2
    return nothing
end
