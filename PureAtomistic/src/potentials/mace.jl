"""
    MACEPotential{T}(model, positions, cell, atomic_numbers; extra_capacity_fraction = 0.1)

A PureMLIP MACE model as a potential. It holds a `NeighborList{T}` of cutoff `model.r_max` built
for `positions` and `cell`, with room for `extra_capacity_fraction` times as many edges again as
that configuration has, a `MACEWorkspace{T}` that covers the list's capacity, and the
`MACEResult{T}` of the last evaluation in `result` (energy, node energies, forces, virial and
stress). `T` is the element type of `model`.
"""
mutable struct MACEPotential{T <: AbstractFloat} <: AbstractStressPotential
    const model::MACEModel{T}
    const atomic_numbers::Vector{Int}
    const extra_capacity_fraction::Float64
    nl::NeighborList{T}
    ws::MACEWorkspace{T}
    const result::MACEResult{T}
    n_regrows::Int
end

function MACEPotential{T}(
        model::MACEModel{T}, positions::AbstractMatrix, cell::AbstractMatrix, atomic_numbers::AbstractVector{<:Integer};
        extra_capacity_fraction::Real = 0.1
    ) where {T <: AbstractFloat}
    n = size(positions, 2)
    _check_configuration(T, n, positions, cell)
    Base.require_one_based_indexing(atomic_numbers)
    length(atomic_numbers) == n ||
        throw(DimensionMismatch(lazy"pureatomistic: $(length(atomic_numbers)) atomic numbers for $n atoms"))
    fraction = _check_capacity_fraction(extra_capacity_fraction)
    nl = _sized_list(positions, cell, model.r_max, fraction)
    ws = MACEWorkspace{T}(model, n, length(nl.senders))
    return MACEPotential{T}(model, Vector{Int}(atomic_numbers), fraction, nl, ws, MACEResult{T}(n), 0)
end

function MACEPotential(
        model::MACEModel{T}, positions::AbstractMatrix, cell::AbstractMatrix, atomic_numbers::AbstractVector{<:Integer};
        extra_capacity_fraction::Real = 0.1
    ) where {T <: AbstractFloat}
    return MACEPotential{T}(model, positions, cell, atomic_numbers; extra_capacity_fraction)
end

n_atoms(p::MACEPotential)::Int = p.nl.n_atoms
cutoff(p::MACEPotential)::Float64 = p.nl.r_cut
stress(p::MACEPotential)::Matrix = p.result.stress
node_energies(p::MACEPotential)::Vector = p.result.node_energies

function prepare!(p::MACEPotential{T}, positions::AbstractMatrix, cell::AbstractMatrix)::Nothing where {T}
    _check_configuration(T, n_atoms(p), positions, cell)
    _refresh!(p.nl, positions, cell) || _regrow!(p, positions, cell)
    return nothing
end

# The only allocating path of `prepare!`; it runs when a configuration outgrows the list.
@noinline function _regrow!(p::MACEPotential{T}, positions::AbstractMatrix{T}, cell::AbstractMatrix{T}) where {T}
    p.nl = _sized_list(positions, cell, p.nl.r_cut, p.extra_capacity_fraction)
    p.ws = MACEWorkspace{T}(p.model, n_atoms(p), length(p.nl.senders))
    p.n_regrows += 1
    _log_regrow("MACEPotential", length(p.nl.senders))
    return true
end

function energy_forces!(p::MACEPotential{T}, positions::AbstractMatrix, cell::AbstractMatrix)::Nothing where {T}
    _check_configuration(T, n_atoms(p), positions, cell)
    mace_energy_forces!(p.result, p.ws, p.model, positions, cell, p.atomic_numbers, p.nl)
    return nothing
end
