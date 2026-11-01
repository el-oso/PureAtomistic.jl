# Blocks of consecutive receivers. A neighbor list sorts its edges by receiver, so the edges that
# end at the atoms `a:b` are one contiguous range of edges; the per-atom layers of an interaction
# then run on one block of receivers at a time, in buffers of `block` atoms.

function _check_receivers(nl::NeighborList)
    receivers = view(nl.receivers, 1:nl.n_edges)
    issorted(receivers) || throw(ArgumentError("model: neighbor list edges must be sorted by receiver"))
    n = nl.n_atoms
    (isempty(receivers) || (first(receivers) >= 1 && last(receivers) <= n)) ||
        throw(ArgumentError(lazy"model: neighbor list receivers must lie in 1:$n"))
    return nothing
end

# Iterates `(atoms, edges)`: the atoms `atoms` in blocks of at most `block`, in order, with the
# range `edges` of the edges whose receiver is in `atoms`. `receivers` holds the receivers of the
# edges, sorted (`_check_receivers`).
struct ReceiverBlocks{V <: AbstractVector{Int}}
    receivers::V
    n_atoms::Int
    block::Int
end

ReceiverBlocks(nl::NeighborList, block::Int) = ReceiverBlocks(view(nl.receivers, 1:nl.n_edges), nl.n_atoms, block)

Base.length(b::ReceiverBlocks) = cld(b.n_atoms, b.block)
Base.eltype(::Type{<:ReceiverBlocks}) = Tuple{UnitRange{Int}, UnitRange{Int}}

function Base.iterate(b::ReceiverBlocks, (i, e) = (1, firstindex(b.receivers)))
    i > b.n_atoms && return nothing
    atoms = i:min(i + b.block - 1, b.n_atoms)
    stop = searchsortedlast(b.receivers, last(atoms))
    return (atoms, e:stop), (last(atoms) + 1, stop + 1)
end
