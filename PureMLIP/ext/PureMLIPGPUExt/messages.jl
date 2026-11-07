# The summed messages of each receiver on the device: the channel-wise tensor product of the sender
# features with the edge attributes and the radial-MLP weights, and the density of each receiver.

"""
    _messages!(ws, model, k, chunk, table) -> nothing

Adds the messages of interaction `k` of the edges of `chunk` to `ws.M[k]` and their densities to
`ws.density[:, k]`, per receiver and in edge order, from the sender features `ws.x[k]`, the
harmonics `ws.Y`, the radial-MLP outputs `ws.tpw3` and the radial features `ws.feats` of the chunk.
`table` is `_tp_table` of the tensor product of interaction `k`. Over the chunks in order, from
zero, each receiver's message and density are the sums over its edges in edge order, as in the
CPU's `_messages!`. One work item per channel and receiver (every `R`-th receiver of the chunk's
range) sums in registers, and no two work items write the same entry.
"""
function _messages!(ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, k::Int, chunk::EdgeChunk, table::Val) where {T}
    nb = _chunk_length(ws, chunk)
    nl = chunk.nl
    C = model.host.C
    R = _chunk_receivers(nl, nb)
    _launch_grouped!(
        _messages_kernel!, ws.backend, (C, R), (C, max(1, 256 ÷ C)), ws.M[k], view(ws.density, :, k), ws.x[k], ws.Y, ws.tpw3,
        ws.feats, model.density[k], nl.senders, nl.receivers, nl.row_ptr, nl.n_edges, table, first(chunk.edges) - 1, nb, R,
    )
    return nothing
end

@kernel function _messages_kernel!(
        M, density, @Const(x), @Const(Y), @Const(w), @Const(feats), @Const(dw), @Const(senders), @Const(receivers),
        @Const(row_ptr), @Const(n_edges), table, e0, nb, R
    )
    u, il = @index(Global, NTuple)
    _messages_work_item!(M, density, x, StoredHarmonics(Y), w, feats, dw, senders, receivers, row_ptr, n_edges[1], table, e0, nb, R, u, il)
end

# The harmonics of an edge as a tuple, read from column `col` of the chunk's harmonics `Y`.
struct StoredHarmonics{A}
    Y::A
end
@inline (h::StoredHarmonics)(e, col, ::Val{K}) where {K} = ntuple(j -> h.Y[j, col], Val(K))

# Work item `(u, il)`: channel `u` of receivers `lo + il - 1`, `lo + il - 1 + R`, … of the chunk's
# receiver range `lo:hi`, with `Y(e, col, Val(K))` the harmonics of edge `e` in chunk column `col`.
@inline function _messages_work_item!(M, density, x, Y, w, feats, dw, senders, receivers, row_ptr, n_edges, table, e0, nb, R, u, il)
    lo, hi = _receiver_range(receivers, n_edges, e0, nb)
    for i in (lo + il - 1):R:hi
        edges = _receiver_edges(row_ptr, i, e0, nb)
        _tp_receiver!(M, u, i, Y, w, x, senders, edges, e0, table)
        u == 1 && _density_receiver!(density, i, feats, dw, edges, e0)
    end
    return nothing
end

# Adds tanh(z²) of each edge of `edges` to `density[i]` in order, with z the radial features of
# column `e - e0` weighted by `dw / sqrt(n_basis)` (`density_edge`).
@inline function _density_receiver!(density, i, feats, dw, edges, e0)
    T = eltype(density)
    r = sqrt(T(size(feats, 1)))
    ρ = density[i]
    for e in edges
        j = e - e0
        z = zero(T)
        for b in axes(feats, 1)
            z += feats[b, j] * (dw[b] / r)
        end
        ρ += tanh(z * z)
    end
    density[i] = ρ
    return nothing
end
