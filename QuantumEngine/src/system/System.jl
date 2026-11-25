"""
    QuantumEngine.System

Atoms, point charges and molecules, with the electron counts and the nuclear repulsion energy.
Positions are in bohr and masses in amu.
"""
module System

using LinearAlgebra: norm
using StaticArrays: SVector
using StrictMode: @strict_exempt, @strict_policy
using ..Units: Units

"""
    Atom{T<:Real}

An atom: its atomic number `Z` (0 for a ghost), the `symbol` as written, its `position` in bohr,
whether it is a `ghost` (it carries basis functions but no nucleus and no electrons) and its
`mass` in amu (0 for a ghost).

    Atom(symbol::AbstractString, position::AbstractVector; mass = nothing)

builds an atom with `T = float(eltype(position))` from a `position` of length 3. The `symbol` is
matched without regard to case: `El` is an atom of element `El`, and `El@`, `BqEl` and `Bq-El` are
ghosts of `El`. The default `mass` is that of the most abundant isotope. Throws an
`ArgumentError` for a position whose length is not 3, a bare `Bq`, a symbol that names no element,
and a `mass` given for a ghost.
"""
struct Atom{T <: Real}
    Z::Int
    symbol::String
    position::SVector{3, T}
    ghost::Bool
    mass::T
    Atom{T}(Z, symbol, position, ghost, mass) where {T} = new{T}(Z, symbol, position, ghost, mass)
end

function Atom(symbol::AbstractString, position::AbstractVector; mass = nothing)
    T = float(eltype(position))
    n = length(position)
    n == 3 || throw(ArgumentError(lazy"system: an atom position has 3 coordinates, got $(n)"))
    i = firstindex(position)
    p = SVector{3, T}(position[i], position[i + 1], position[i + 2])
    z, ghost = _parse_symbol(symbol)
    if ghost
        isnothing(mass) || throw(ArgumentError(lazy"system: a ghost atom has no mass, got $(mass)"))
        return Atom{T}(0, String(symbol), p, true, zero(T))
    end
    m = isnothing(mass) ? Units.element(z).mass_most_abundant : mass
    return Atom{T}(z, String(symbol), p, false, m)
end

"""
    element_number(symbol::AbstractString) -> Int

The atomic number of the element that `symbol` names under the spelling `Atom` reads (`El`, or
the ghosts `El@`, `BqEl` and `Bq-El`, in any case), or 0 when it names no element. Throws an
`ArgumentError` for a bare `Bq`, which is malformed rather than unknown, with the message `Atom`
throws for it.
"""
function element_number(symbol::AbstractString)
    lo, hi, ghost = _symbol_body(symbol)
    ghost && lo > hi && _is_bq_prefix(symbol) &&
        throw(ArgumentError("system: a Bq atom needs an element (BqO, O@); a point charge is `Bq x y z q`"))
    return _element_number(symbol, lo, hi)
end

# The element number named by `symbol` and whether it names a ghost; throws for a bare `Bq` and
# for a symbol that names no element.
function _parse_symbol(symbol::AbstractString)
    z = element_number(symbol)
    z == 0 && throw(ArgumentError(lazy"system: unknown element symbol \"$(symbol)\""))
    return z, _symbol_body(symbol)[3]
end

# The code units `lo:hi` of `symbol` that spell its element, and whether the symbol marks a
# ghost: `El@`, `BqEl` and `Bq-El` are ghosts of `El` (the prefix in any case). The range is
# returned instead of a `SubString`, which the allocation scan counts as an allocation.
function _symbol_body(symbol::AbstractString)
    n = ncodeunits(symbol)
    if n >= 1 && codeunit(symbol, n) == UInt8('@')
        return 1, n - 1, true
    elseif _is_bq_prefix(symbol)
        return (n >= 3 && codeunit(symbol, 3) == UInt8('-') ? 4 : 3), n, true
    end
    return 1, n, false
end

_is_bq_prefix(s::AbstractString) =
    ncodeunits(s) >= 2 && _lower(codeunit(s, 1)) == UInt8('b') && _lower(codeunit(s, 2)) == UInt8('q')

_lower(c::UInt8) = UInt8('A') <= c <= UInt8('Z') ? c + (UInt8('a') - UInt8('A')) : c

# The atomic number of the element whose symbol the code units `lo:hi` of `s` spell in any case,
# or 0 if none does.
function _element_number(s::AbstractString, lo::Int, hi::Int)
    for z in 1:Units.nelements()
        ref = Units.element(z).symbol
        hi - lo + 1 == ncodeunits(ref) || continue
        all(k -> _lower(codeunit(s, lo + k - 1)) == _lower(codeunit(ref, k)), 1:ncodeunits(ref)) && return z
    end
    return 0
end

"""
    PointCharge{T<:Real}(charge::T, position::SVector{3,T})

A point charge `charge` (in units of the elementary charge) at `position` (bohr).
"""
struct PointCharge{T <: Real}
    charge::T
    position::SVector{3, T}
end

"""
    Molecule{T<:Real, V<:AbstractVector{Atom{T}}}

The `atoms`, the total `charge`, the spin `multiplicity` (2S + 1) and the external
`point_charges` of a molecule.

    Molecule(atoms::AbstractVector{Atom{T}}; charge = 0, multiplicity = 1, point_charges = PointCharge{T}[])

Throws an `ArgumentError` for a multiplicity below 1.
"""
struct Molecule{T <: Real, V <: AbstractVector{Atom{T}}}
    atoms::V
    charge::Int
    multiplicity::Int
    point_charges::Vector{PointCharge{T}}
    function Molecule{T, V}(atoms, charge, multiplicity, point_charges) where {T, V}
        multiplicity >= 1 ||
            throw(ArgumentError(lazy"system: multiplicity must be at least 1, got $(multiplicity)"))
        return new{T, V}(atoms, charge, multiplicity, point_charges)
    end
end

Molecule(atoms::AbstractVector{Atom{T}}; charge = 0, multiplicity = 1, point_charges = PointCharge{T}[]) where {T} =
    Molecule{T, typeof(atoms)}(atoms, charge, multiplicity, point_charges)

"""
    basis_element(a::Atom) -> Int

The atomic number of the element whose basis functions `a` carries: its own for a real atom, and
the element a ghost stands for, read from its symbol. Throws an `ArgumentError` for a ghost whose
symbol names no element.
"""
function basis_element(a::Atom)
    a.ghost || return a.Z
    lo, hi, _ = _symbol_body(a.symbol)
    z = _element_number(a.symbol, lo, hi)
    z == 0 && throw(ArgumentError(lazy"system: ghost atom \"$(a.symbol)\" stands for no element"))
    return z
end

"""
    nelectrons(mol::Molecule) -> Int

The number of electrons: the sum of the atomic numbers minus the charge. Throws an
`ArgumentError` when the charge exceeds the nuclear charge.
"""
function nelectrons(mol::Molecule)
    atoms = mol.atoms
    z = 0
    for i in eachindex(atoms)
        z += atoms[i].Z
    end
    n = z - mol.charge
    n >= 0 || throw(ArgumentError(lazy"system: charge $(mol.charge) exceeds the nuclear charge $(z)"))
    return n
end

# The electron count and the number of unpaired electrons, mult − 1; throws when the electrons
# cannot be split into that many unpaired ones and pairs.
function _spin_split(mol::Molecule)
    n = nelectrons(mol)
    u = mol.multiplicity - 1
    d = n - u
    (d < 0 || isodd(d)) &&
        throw(ArgumentError(lazy"system: $(n) electrons cannot have multiplicity $(mol.multiplicity)"))
    return n, u
end

"""
    nalpha(mol::Molecule) -> Int

The number of α electrons, (n + mult − 1) / 2. Throws an `ArgumentError` when n − (mult − 1) is
odd or negative.
"""
function nalpha(mol::Molecule)
    n, u = _spin_split(mol)
    return Int((n + u) / 2)
end

"""
    nbeta(mol::Molecule) -> Int

The number of β electrons, (n − mult + 1) / 2. Throws an `ArgumentError` when n − (mult − 1) is
odd or negative.
"""
function nbeta(mol::Molecule)
    n, u = _spin_split(mol)
    return Int((n - u) / 2)
end

"""
    nuclear_repulsion(mol::Molecule{T}) -> T

The repulsion energy Σ_{i<j} Z_i Z_j / r_ij (hartree) of the atoms with Z > 0; ghosts carry no
nucleus. Throws an `ArgumentError` when two such atoms coincide, naming their indices in
`eachindex(mol.atoms)`. The point charges are not included: their interaction with each other and
with the nuclei is not part of this energy.
"""
function nuclear_repulsion(mol::Molecule{T}) where {T}
    atoms = mol.atoms
    e = zero(T)
    for i in eachindex(atoms), j in eachindex(atoms)
        i < j || continue
        Z_i = atoms[i].Z
        Z_j = atoms[j].Z
        (Z_i > 0 && Z_j > 0) || continue
        # `norm` rescales when the plain sum of squares overflows or underflows, so a distance of a
        # few hundred bohr stays finite in Float16.
        r = norm(atoms[i].position - atoms[j].position)
        iszero(r) && throw(ArgumentError(lazy"system: atoms $(i) and $(j) coincide"))
        e += Z_i * Z_j / r
    end
    return e
end

export Atom, PointCharge, Molecule, basis_element, nelectrons, nalpha, nbeta, nuclear_repulsion
public element_number

# The keyword body of `Atom` allocates the `String` of a `SubString` symbol, the token a reader
# passes; with a `String` symbol it allocates nothing and is held to `:noalloc`. So the exemption
# names the `SubString{String}` signature of the keyword body only, and holds whichever of the two
# is compiled first. The keyword body of `Molecule` allocates nothing (its default point-charge
# vector is built by the methods that call it), so `Molecule` has no exemption.
@strict_exempt :noalloc types = (Any, Type{Atom}, SubString{String}, AbstractVector) Atom reason = "entry point: keyword body of the Atom constructor"
# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
