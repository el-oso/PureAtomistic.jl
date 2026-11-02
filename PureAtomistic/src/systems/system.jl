"""
    System{T}(positions, velocities, masses, species, cell)

The state of a periodic configuration. `positions` and `velocities` are `3 × n` (Å and Å/fs),
`masses` has length `n` (amu), `species` holds the `n` atomic numbers, and the columns of the
`3 × 3` `cell` are the lattice vectors (Å). The atom axes of the four per-atom arrays must be
equal; any `AbstractArray` with those axes is accepted, including views and `OffsetArray`s.
"""
struct System{
        T <: AbstractFloat, P <: AbstractMatrix{T}, V <: AbstractMatrix{T}, M <: AbstractVector{T},
        S <: AbstractVector{<:Integer}, C <: AbstractMatrix{T},
    }
    positions::P
    velocities::V
    masses::M
    species::S
    cell::C

    function System{T, P, V, M, S, C}(positions, velocities, masses, species, cell) where {T, P, V, M, S, C}
        size(positions, 1) == 3 ||
            throw(ArgumentError(lazy"pureatomistic: positions need 3 rows, got $(size(positions, 1))"))
        axes(velocities) == axes(positions) ||
            throw(ArgumentError(lazy"pureatomistic: velocities axes $(axes(velocities)) differ from positions axes $(axes(positions))"))
        axes(masses, 1) == axes(positions, 2) ||
            throw(ArgumentError(lazy"pureatomistic: masses axes $(axes(masses)) differ from the atom axis $(axes(positions, 2))"))
        axes(species, 1) == axes(positions, 2) ||
            throw(ArgumentError(lazy"pureatomistic: species axes $(axes(species)) differ from the atom axis $(axes(positions, 2))"))
        size(cell) == (3, 3) ||
            throw(ArgumentError(lazy"pureatomistic: the cell must be 3 × 3, got $(size(cell))"))
        all(>(0), masses) || throw(ArgumentError("pureatomistic: masses must be positive"))
        return new{T, P, V, M, S, C}(positions, velocities, masses, species, cell)
    end
end

function System{T}(positions, velocities, masses, species, cell) where {T}
    p = convert(AbstractMatrix{T}, positions)::AbstractMatrix{T}
    v = convert(AbstractMatrix{T}, velocities)::AbstractMatrix{T}
    m = convert(AbstractVector{T}, masses)::AbstractVector{T}
    c = convert(AbstractMatrix{T}, cell)::AbstractMatrix{T}
    return System{T, typeof(p), typeof(v), typeof(m), typeof(species), typeof(c)}(p, v, m, species, c)
end

System(positions, velocities, masses, species, cell) =
    System{float(eltype(positions))}(positions, velocities, masses, species, cell)

"""
    natoms(sys)

Number of atoms in `sys`.
"""
natoms(sys::System) = length(axes(sys.positions, 2))

# The first three indices of an axis: the 1-based position `k` of a 3-element axis maps to
# `_triple(r)[k]`, whatever the axis offset.
@inline _triple(r) = (first(r), first(r) + 1, first(r) + 2)

# The nine entries of a 3 × 3 matrix in column-major order, read through its own axes.
@inline function _entries(A)
    r, s = _triple(axes(A, 1)), _triple(axes(A, 2))
    return (
        A[r[1], s[1]], A[r[2], s[1]], A[r[3], s[1]],
        A[r[1], s[2]], A[r[2], s[2]], A[r[3], s[2]],
        A[r[1], s[3]], A[r[2], s[3]], A[r[3], s[3]],
    )
end

# Determinant of a column-major 3 × 3 given as a 9-tuple.
@inline function _det3(c)
    return c[1] * (c[5] * c[9] - c[8] * c[6]) - c[4] * (c[2] * c[9] - c[8] * c[3]) +
        c[7] * (c[2] * c[6] - c[5] * c[3])
end

# Inverse of a column-major 3 × 3 by cofactors; throws on a zero or non-finite determinant.
@inline function _inverse3(c)
    d = _det3(c)
    (isfinite(d) && !iszero(d)) || throw(ArgumentError("pureatomistic: the cell is singular"))
    return (
        (c[5] * c[9] - c[8] * c[6]) / d, (c[8] * c[3] - c[2] * c[9]) / d, (c[2] * c[6] - c[5] * c[3]) / d,
        (c[7] * c[6] - c[4] * c[9]) / d, (c[1] * c[9] - c[7] * c[3]) / d, (c[4] * c[3] - c[1] * c[6]) / d,
        (c[4] * c[8] - c[7] * c[5]) / d, (c[7] * c[2] - c[1] * c[8]) / d, (c[1] * c[5] - c[4] * c[2]) / d,
    )
end

"""
    volume(sys)

Cell volume `|det(cell)|` in Å³.
"""
volume(sys::System) = abs(_det3(_entries(sys.cell)))

"""
    wrap!(sys)

Move every atom into the unit cell by subtracting the integer multiples of the lattice vectors
that bring its fractional coordinates into `[0, 1)`. Works for triclinic cells. An atom that is
already inside keeps its position bit for bit. Throws if the cell is singular or a position is
not finite.
"""
function wrap!(sys::System{T}) where {T}
    c = _entries(sys.cell)
    ic = _inverse3(c)
    P = sys.positions
    r = _triple(axes(P, 1))
    for a in axes(P, 2)
        x1, x2, x3 = P[r[1], a], P[r[2], a], P[r[3], a]
        f1 = ic[1] * x1 + ic[4] * x2 + ic[7] * x3
        f2 = ic[2] * x1 + ic[5] * x2 + ic[8] * x3
        f3 = ic[3] * x1 + ic[6] * x2 + ic[9] * x3
        isfinite(f1 + f2 + f3) ||
            throw(ArgumentError(lazy"pureatomistic: atom $a has a non-finite position"))
        n1, n2, n3 = floor(f1), floor(f2), floor(f3)
        if !(iszero(n1) && iszero(n2) && iszero(n3))
            P[r[1], a] = x1 - (n1 * c[1] + n2 * c[4] + n3 * c[7])
            P[r[2], a] = x2 - (n1 * c[2] + n2 * c[5] + n3 * c[8])
            P[r[3], a] = x3 - (n1 * c[3] + n2 * c[6] + n3 * c[9])
        end
    end
    return sys
end

"""
    kinetic_energy(sys)

Kinetic energy `Σ mᵢ vᵢ² / 2` in eV, accumulated in the working type `T`.
"""
function kinetic_energy(sys::System{T}) where {T}
    V = sys.velocities
    r = _triple(axes(V, 1))
    s = zero(T)
    for a in axes(V, 2)
        s += sys.masses[a] * (V[r[1], a]^2 + V[r[2], a]^2 + V[r[3], a]^2)
    end
    return T(AMU_IN_EV_FS2_PER_A2 / 2) * s
end

"""
    total_momentum(sys)

Total momentum `Σ mᵢ vᵢ` as a 3-tuple in amu·Å/fs.
"""
function total_momentum(sys::System{T}) where {T}
    V = sys.velocities
    r = _triple(axes(V, 1))
    p1 = p2 = p3 = zero(T)
    for a in axes(V, 2)
        m = sys.masses[a]
        p1 += m * V[r[1], a]
        p2 += m * V[r[2], a]
        p3 += m * V[r[3], a]
    end
    return (p1, p2, p3)
end

"""
    temperature(sys; dof = 3 * natoms(sys) - 3)

Kinetic temperature `2 K / (dof · k_B)` in K. The default `dof` is `3n - 3`, the count with the
center-of-mass momentum removed. Throws when `dof < 1`.
"""
function temperature(sys::System{T}; dof::Integer = 3 * natoms(sys) - 3) where {T}
    dof >= 1 || throw(ArgumentError("pureatomistic: temperature needs at least one degree of freedom"))
    return 2 * kinetic_energy(sys) / (T(dof) * T(KB_EV_PER_K))
end

"""
    pressure(sys, stress)

Pressure in eV/Å³ from the 3 × 3 virial `stress` (eV/Å³, tension positive) and the kinetic
contribution: `P = -tr(stress) / 3 + 2 K / (3 V)`. Throws if the cell is singular.
"""
function pressure(sys::System{T}, stress::AbstractMatrix) where {T}
    size(stress) == (3, 3) ||
        throw(ArgumentError(lazy"pureatomistic: the stress must be 3 × 3, got $(size(stress))"))
    r, s = _triple(axes(stress, 1)), _triple(axes(stress, 2))
    tr = stress[r[1], s[1]] + stress[r[2], s[2]] + stress[r[3], s[3]]
    V = volume(sys)
    (isfinite(V) && !iszero(V)) || throw(ArgumentError("pureatomistic: the cell is singular"))
    return T(-tr / 3) + 2 * kinetic_energy(sys) / (3 * V)
end
