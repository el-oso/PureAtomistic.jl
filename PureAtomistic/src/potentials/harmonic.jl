"""
    HarmonicPotential{T}(sites, k)

Independent springs of stiffness `k` (eV/Å²) that tie atom `i` to `sites[:, i]` (Å):
`E = k/2 Σᵢ |dᵢ|²` and `Fᵢ = -k dᵢ`, where `dᵢ` is the displacement of atom `i` from its site,
taken to the periodic image whose fractional coordinates relative to the site lie in
`[-1/2, 1/2)`. Away from the boundary of that range the energy is smooth and the forces are its
exact gradient. The potential has no pair interactions: its cutoff is zero and [`prepare!`](@ref)
only checks the configuration.
"""
struct HarmonicPotential{T <: AbstractFloat} <: AbstractPotential
    sites::Matrix{T}
    k::T
    result::PotentialResult{T}
    function HarmonicPotential{T}(sites::AbstractMatrix, k::Real) where {T <: AbstractFloat}
        size(sites, 1) == 3 || throw(DimensionMismatch(lazy"pureatomistic: sites have size $(size(sites)), expected 3 rows"))
        (isfinite(k) && k > 0) || throw(ArgumentError(lazy"pureatomistic: the spring constant must be positive and finite, got $k"))
        n = size(sites, 2)
        return new{T}(Matrix{T}(sites), T(k), PotentialResult{T}(n))
    end
end

HarmonicPotential(sites::AbstractMatrix{T}, k::Real) where {T <: AbstractFloat} = HarmonicPotential{T}(sites, k)

n_atoms(p::HarmonicPotential)::Int = size(p.sites, 2)
cutoff(::HarmonicPotential)::Float64 = 0.0

function prepare!(p::HarmonicPotential{T}, positions::AbstractMatrix, cell::AbstractMatrix)::Nothing where {T}
    _check_configuration(T, n_atoms(p), positions, cell)
    return nothing
end

function energy_forces!(p::HarmonicPotential{T}, positions::AbstractMatrix, cell::AbstractMatrix)::Nothing where {T}
    _check_configuration(T, n_atoms(p), positions, cell)
    ci = _inverse3(cell)
    E = zero(T)
    F = p.result.forces
    for i in axes(positions, 2)
        d1, d2, d3 = _minimum_image(
            cell, ci, positions[1, i] - p.sites[1, i], positions[2, i] - p.sites[2, i], positions[3, i] - p.sites[3, i]
        )
        E += d1 * d1 + d2 * d2 + d3 * d3
        F[1, i] = -p.k * d1
        F[2, i] = -p.k * d2
        F[3, i] = -p.k * d3
    end
    p.result.energy = p.k * E / 2
    return nothing
end

# The inverse of the 3 × 3 matrix `C` as a tuple of rows, by the adjugate.
function _inverse3(C::AbstractMatrix{T}) where {T}
    c11, c21, c31 = C[1, 1], C[2, 1], C[3, 1]
    c12, c22, c32 = C[1, 2], C[2, 2], C[3, 2]
    c13, c23, c33 = C[1, 3], C[2, 3], C[3, 3]
    a11 = c22 * c33 - c23 * c32
    a12 = c13 * c32 - c12 * c33
    a13 = c12 * c23 - c13 * c22
    a21 = c23 * c31 - c21 * c33
    a22 = c11 * c33 - c13 * c31
    a23 = c13 * c21 - c11 * c23
    a31 = c21 * c32 - c22 * c31
    a32 = c12 * c31 - c11 * c32
    a33 = c11 * c22 - c12 * c21
    det = c11 * a11 + c12 * a21 + c13 * a31
    iszero(det) && throw(ArgumentError("pureatomistic: the cell is singular"))
    return ((a11 / det, a12 / det, a13 / det), (a21 / det, a22 / det, a23 / det), (a31 / det, a32 / det, a33 / det))
end

# The displacement `d` moved by a lattice vector so that its fractional coordinates lie in [-1/2, 1/2).
function _minimum_image(C::AbstractMatrix{T}, ci, d1::T, d2::T, d3::T) where {T}
    s1 = ci[1][1] * d1 + ci[1][2] * d2 + ci[1][3] * d3
    s2 = ci[2][1] * d1 + ci[2][2] * d2 + ci[2][3] * d3
    s3 = ci[3][1] * d1 + ci[3][2] * d2 + ci[3][3] * d3
    n1, n2, n3 = floor(s1 + T(0.5)), floor(s2 + T(0.5)), floor(s3 + T(0.5))
    return (
        d1 - (C[1, 1] * n1 + C[1, 2] * n2 + C[1, 3] * n3),
        d2 - (C[2, 1] * n1 + C[2, 2] * n2 + C[2, 3] * n3),
        d3 - (C[3, 1] * n1 + C[3, 2] * n2 + C[3, 3] * n3),
    )
end
