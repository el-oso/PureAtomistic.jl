"""
    AbstractPotential

Supertype of every potential the MD driver evaluates. A potential owns its neighbor list and
its buffers, and holds the result of its last evaluation in `p.result`, whose fields `energy`
(eV) and `forces` (`3 × n`, eV/Å, minus the gradient of the energy) every potential provides.
Positions are `3 × n` (Å, one column per atom) and the cell is `3 × 3` with the lattice
vectors as columns; both must have the element type the potential was built with.

A step calls [`prepare!`](@ref) for the new positions and cell, then
[`energy_forces!`](@ref) with the same positions and cell.
"""
abstract type AbstractPotential end

"""
    AbstractStressPotential <: AbstractPotential

A potential that also provides [`stress`](@ref), which a barostat needs.
"""
abstract type AbstractStressPotential <: AbstractPotential end

"""
    n_atoms(p) -> Int

The number of atoms `p` evaluates.
"""
function n_atoms end

"""
    cutoff(p) -> Float64

The interaction range of `p` in Å; zero for a potential without pair interactions.
"""
function cutoff end

"""
    prepare!(p, positions, cell) -> nothing

Refresh the neighbor list of `p` for `positions` and `cell`. When the new configuration holds
more edges than the list has room for, `p` rebuilds its list and buffers with the extra
capacity it was built with, reports it on standard error and counts it in `p.n_regrows`.
"""
function prepare! end

"""
    energy_forces!(p, positions, cell) -> nothing

Evaluate `p` at `positions` and `cell`, which [`prepare!`](@ref) has seen last, and store the
energy and the forces in `p.result`.
"""
function energy_forces! end

"""
    stress(p) -> AbstractMatrix

The `3 × 3` stress of the last evaluation in eV/Å³, `∂E/∂ε / volume` for the symmetric
strain `ε`: positive in tension, so the potential's pressure is `-tr(stress) / 3`.
"""
function stress end

"""
    node_energies(p) -> AbstractVector

The per-atom energies of the last evaluation in eV; they sum to the energy.
"""
function node_energies end

@contract AbstractPotential "Energies and forces of an atomic configuration." begin
    n_atoms(::Self)::Int => "number of atoms"
    cutoff(::Self)::Float64 => "interaction range in Å"
    prepare!(::Self, ::AbstractMatrix, ::AbstractMatrix)::Nothing => "refresh the neighbor list for positions and cell"
    energy_forces!(::Self, ::AbstractMatrix, ::AbstractMatrix)::Nothing => "fill p.result with the energy and forces"
    :optional
    stress(::Self)::AbstractMatrix => "stress in eV/Å³, positive in tension"
    node_energies(::Self)::AbstractVector => "per-atom energies in eV"
end

@contract AbstractStressPotential "A potential that also provides the stress." begin
    stress(::Self)::AbstractMatrix => "stress in eV/Å³, positive in tension"
end

"""
    PotentialResult{T}(n_atoms)

The energy (eV) and the `3 × n_atoms` forces (eV/Å) of a potential that computes nothing else.
"""
mutable struct PotentialResult{T <: AbstractFloat}
    energy::T
    forces::Matrix{T}
    function PotentialResult{T}(n_atoms::Integer) where {T <: AbstractFloat}
        n_atoms >= 0 || throw(ArgumentError(lazy"pureatomistic: the number of atoms must not be negative, got $n_atoms"))
        return new{T}(zero(T), zeros(T, 3, Int(n_atoms)))
    end
end

# The arrays of a configuration must have the potential's element type `T`, be 1-based, and
# hold `n` atoms.
function _check_configuration(::Type{T}, n::Int, positions::AbstractMatrix, cell::AbstractMatrix) where {T}
    Base.require_one_based_indexing(positions, cell)
    (eltype(positions) === T && eltype(cell) === T) ||
        throw(ArgumentError(lazy"pureatomistic: positions and cell must have element type $T, got $(eltype(positions)) and $(eltype(cell))"))
    size(positions) == (3, n) ||
        throw(DimensionMismatch(lazy"pureatomistic: positions have size $(size(positions)), expected (3, $n)"))
    size(cell) == (3, 3) || throw(DimensionMismatch(lazy"pureatomistic: the cell has size $(size(cell)), expected (3, 3)"))
    return nothing
end

function _check_capacity_fraction(f::Real)
    (isfinite(f) && f >= 0) ||
        throw(ArgumentError(lazy"pureatomistic: extra_capacity_fraction must be finite and non-negative, got $f"))
    return Float64(f)
end

# A neighbor list for `positions` and `cell` with room for `fraction` more edges than it finds.
function _sized_list(positions::AbstractMatrix{T}, cell::AbstractMatrix{T}, r_cut::Real, fraction::Float64) where {T}
    m = neighbor_list(positions, cell, r_cut).n_edges
    return neighbor_list(positions, cell, r_cut; extra_capacity = ceil(Int, fraction * m))
end

# Refresh `nl` in place; false when the configuration holds more edges than `nl` has room for.
# `neighbor_list!` throws an `ArgumentError` for that and for an invalid configuration (atoms that
# coincide, positions that are not finite). The caller rebuilds the list with `neighbor_list`
# after a false, which throws the same error again for an invalid configuration, so only the
# capacity error ends in a rebuilt list.
function _refresh!(nl::NeighborList{T}, positions::AbstractMatrix{T}, cell::AbstractMatrix{T})::Bool where {T}
    try
        neighbor_list!(nl, positions, cell)
    catch err
        err isa ArgumentError || rethrow()
        return false
    end
    return true
end

# Core.stderr is a concrete stream, so the message stays resolvable in a trimmed program, where
# the logging macros are not.
function _log_regrow(name::String, capacity::Int)
    print(Core.stderr, "pureatomistic: ", name, " rebuilt its neighbor list and buffers with capacity ", capacity, " edges\n")
    return nothing
end
