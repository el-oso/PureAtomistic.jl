# Elements with a built-in mass: atomic number, symbol and conventional atomic weight in amu
# (IUPAC abridged values).
const _ELEMENT_Z = [1, 6, 7, 8, 13, 14, 26, 28, 29, 47, 79]
const _ELEMENT_SYMBOL = [:H, :C, :N, :O, :Al, :Si, :Fe, :Ni, :Cu, :Ag, :Au]
const _ELEMENT_MASS = [1.008, 12.011, 14.007, 15.999, 26.982, 28.085, 55.845, 58.693, 63.546, 107.868, 196.967]

_element_index(z::Integer) = findfirst(==(z), _ELEMENT_Z)
_element_index(s::Symbol) = findfirst(==(s), _ELEMENT_SYMBOL)

function _element_index_checked(element)
    i = _element_index(element)
    isnothing(i) && throw(ArgumentError(lazy"pureatomistic: unknown element $element"))
    return i
end

"""
    atomic_number(element)

Atomic number of an element given by symbol (`:Cu`) or number. The built-in elements are
H, C, N, O, Al, Si, Fe, Ni, Cu, Ag and Au; others throw.
"""
atomic_number(element::Union{Symbol, Integer}) = _ELEMENT_Z[_element_index_checked(element)]

"""
    atomic_mass(element)

Conventional atomic weight in amu of an element given by symbol or atomic number (same set as
[`atomic_number`](@ref)).
"""
atomic_mass(element::Union{Symbol, Integer}) = _ELEMENT_MASS[_element_index_checked(element)]

"""
    maxwell_boltzmann!(sys, temp, rng; remove_com = true)

Draw velocities from the Maxwell–Boltzmann distribution at `temp` K, `vᵢ ~ N(0, k_B T / mᵢ)` per
component, in a fixed order (atom by atom, component by component). With `remove_com` the
center-of-mass momentum is then set to zero. The velocities are finally rescaled so that
[`temperature`](@ref) with the matching degrees of freedom (`3n - 3` with `remove_com`, `3n`
without) equals `temp` up to rounding. Throws for a negative `temp` or fewer than one degree of
freedom; `temp = 0` zeroes the velocities.
"""
function maxwell_boltzmann!(sys::System{T}, temp::Real, rng::Random.AbstractRNG; remove_com::Bool = true) where {T}
    temp >= 0 || throw(ArgumentError(lazy"pureatomistic: the target temperature must be nonnegative, got $temp"))
    dof = (remove_com ? 3 * natoms(sys) - 3 : 3 * natoms(sys))
    dof >= 1 || throw(ArgumentError("pureatomistic: temperature needs at least one degree of freedom"))
    V = sys.velocities
    r = _triple(axes(V, 1))
    if iszero(temp)
        fill!(V, zero(T))
        return sys
    end
    kT = T(KB_EV_PER_K) * T(temp)
    for a in axes(V, 2)
        σ = sqrt(kT / (sys.masses[a] * T(AMU_IN_EV_FS2_PER_A2)))
        for d in 1:3
            V[r[d], a] = σ * Random.randn(rng, T)
        end
    end
    if remove_com
        p = total_momentum(sys)
        mtot = sum(sys.masses)
        for a in axes(V, 2), d in 1:3
            V[r[d], a] -= p[d] / mtot
        end
    end
    current = temperature(sys; dof)
    current > 0 || throw(ArgumentError("pureatomistic: the sampled velocities carry no kinetic energy"))
    scale = sqrt(T(temp) / current)
    for a in axes(V, 2), d in 1:3
        V[r[d], a] *= scale
    end
    return sys
end

"""
    fcc_cell(a, nrep, element; mass = atomic_mass(element))

Face-centered cubic crystal of lattice constant `a` (Å) with `nrep` conventional cells along each
axis (an integer, or a 3-tuple), `4·prod(nrep)` atoms of `element` (symbol or atomic number) and
zero velocities. The cell is `a·diag(nrep)`. The element type follows `float(typeof(a))`.
"""
function fcc_cell(a::Real, nrep::NTuple{3, Integer}, element; mass::Real = atomic_mass(element))
    T = float(typeof(a))
    all(>=(1), nrep) || throw(ArgumentError(lazy"pureatomistic: nrep must be at least 1 in each direction, got $nrep"))
    a > 0 || throw(ArgumentError(lazy"pureatomistic: the lattice constant must be positive, got $a"))
    z = atomic_number(element)
    n = 4 * prod(nrep)
    basis = ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1))
    positions = Matrix{T}(undef, 3, n)
    idx = 0
    for k in 0:(nrep[3] - 1), j in 0:(nrep[2] - 1), i in 0:(nrep[1] - 1), b in basis
        idx += 1
        positions[1, idx] = a * (i + b[1] / 2)
        positions[2, idx] = a * (j + b[2] / 2)
        positions[3, idx] = a * (k + b[3] / 2)
    end
    cell = zeros(T, 3, 3)
    for d in 1:3
        cell[d, d] = a * nrep[d]
    end
    return System{T}(positions, zeros(T, 3, n), fill(T(mass), n), fill(z, n), cell)
end
fcc_cell(a::Real, nrep::Integer, element; kwargs...) = fcc_cell(a, (nrep, nrep, nrep), element; kwargs...)

"""
    remove_atom(sys, i)

A new [`System`](@ref) without atom `i` (an index of the atom axis), built from plain
arrays whose atom axis is `1:n-1`. The cell is shared with `sys`.
"""
function remove_atom(sys::System{T}, i::Integer) where {T}
    i in axes(sys.positions, 2) ||
        throw(ArgumentError(lazy"pureatomistic: atom $i is outside the atom axis $(axes(sys.positions, 2))"))
    keep = [a for a in axes(sys.positions, 2) if a != i]
    r = _triple(axes(sys.positions, 1))
    rv = _triple(axes(sys.velocities, 1))
    positions = Matrix{T}(undef, 3, length(keep))
    velocities = Matrix{T}(undef, 3, length(keep))
    for (k, a) in enumerate(keep), d in 1:3
        positions[d, k] = sys.positions[r[d], a]
        velocities[d, k] = sys.velocities[rv[d], a]
    end
    return System{T}(positions, velocities, sys.masses[keep], sys.species[keep], sys.cell)
end
