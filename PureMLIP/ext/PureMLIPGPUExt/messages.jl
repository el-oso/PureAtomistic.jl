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

"""
    _messages_pullback!(ws, model, k, chunk, table) -> nothing

The pullback of the messages of interaction `k` for the edges of `chunk`, from the cotangents
`ws.dM[k]` of the summed messages: per edge, the cotangents of the radial-MLP outputs into
`ws.dtpw3` and of the harmonics into `ws.dYe`, from the chunk's harmonics `ws.Y`, the radial-MLP
outputs `ws.tpw3` and the sender features `ws.x[k]`. One work group per edge, one work item per
channel; the channels' terms of `ws.dYe` are added in channel order. Columns of slots past
`nl.n_edges[1]` get zero cotangents. The model may have at most `_PULLBACK_CHANNELS` channels.
"""
function _messages_pullback!(ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, k::Int, chunk::EdgeChunk, table::Val) where {T}
    nb = _chunk_length(ws, chunk)
    nl = chunk.nl
    C = _check_pullback_channels(model.host.C)
    _launch_grouped!(
        _edge_pullback_kernel!, ws.backend, (C, nb), (C, 1), ws.dtpw3, ws.dYe, ws.Y, ws.tpw3, ws.x[k], ws.dM[k],
        nl.senders, nl.receivers, nl.n_edges, table, C, first(chunk.edges) - 1,
    )
    return nothing
end

# The largest channel count of the message pullback: its group memory holds one column of terms
# per channel, and its size must be known when the kernel compiles.
const _PULLBACK_CHANNELS = 256

_check_pullback_channels(C::Int) =
    C <= _PULLBACK_CHANNELS ? C : throw(ArgumentError(lazy"gpu: the message pullback holds at most $_PULLBACK_CHANNELS channels, the model has $C"))

# The channels' terms of ∂E/∂Y meet in group memory, which only a kernel body can hold; the
# arithmetic is in `_tp_edge_pullback!` and `_channel_sum!`.
@kernel function _edge_pullback_kernel!(
        dw, dYe, @Const(Y), @Const(w), @Const(x), @Const(dM), @Const(senders), @Const(receivers), @Const(n_edges),
        table::Val{TABLE}, C, e0
    ) where {TABLE}
    u, col = @index(Global, NTuple)
    buf = @localmem eltype(dYe) (TABLE[2], _PULLBACK_CHANNELS)
    e = e0 + col
    if e <= n_edges[1]
        part = _tp_edge_pullback!(dw, u, e, col, StoredHarmonics(Y), w, x, dM, senders[e], receivers[e], table)
        _store_terms!(buf, part, u)
    else
        _zero_edge_pullback!(dw, buf, u, col)
    end
    @synchronize
    _channel_sum!(dYe, buf, u, col, C)
end

@inline function _store_terms!(buf, part, u)
    for j in eachindex(part)
        buf[j, u] = part[j]
    end
    return nothing
end

@inline function _zero_edge_pullback!(dw, buf, u, col)
    for p in axes(dw, 2)
        dw[u, p, col] = zero(eltype(dw))
    end
    for j in axes(buf, 1)
        buf[j, u] = zero(eltype(buf))
    end
    return nothing
end

# Work item `u` of `C` writes the entries `u, u + C, …` of column `col` of `dYe`: each the sum of
# the `C` channels' terms in `buf`, in channel order.
@inline function _channel_sum!(dYe, buf, u, col, C)
    for j in u:C:size(buf, 1)
        acc = zero(eltype(dYe))
        for t in 1:C
            acc += buf[j, t]
        end
        dYe[j, col] = acc
    end
    return nothing
end

"""
    _sender_gradient!(ws, model, chunk, table) -> nothing

Adds to `ws.dx` the cotangents of the sender features of interaction 2 from the edges of `chunk`,
per receiver of the chunk and in edge order: the term of the reverse edge of each edge
(`_tp_sender_receiver!`), from the chunk's harmonics `ws.Y`, the radial-MLP outputs `ws.tpw3` and
the cotangents `ws.dM[2]`. Over the chunks in order, from zero, `ws.dx[:, i, :]` is the sum over
the edges that atom `i` sends, gathered at `i` without atomics.
"""
function _sender_gradient!(ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, chunk::EdgeChunk, table::Val) where {T}
    nb = _chunk_length(ws, chunk)
    nl = chunk.nl
    C = model.host.C
    R = _chunk_receivers(nl, nb)
    _launch_grouped!(
        _sender_kernel!, ws.backend, (C, R), (C, max(1, 256 ÷ C)), ws.dx, ws.Y, ws.tpw3, ws.dM[2],
        nl.senders, nl.receivers, nl.row_ptr, nl.n_edges, table, first(chunk.edges) - 1, nb, R,
    )
    return nothing
end

@kernel function _sender_kernel!(dx, @Const(Y), @Const(w), @Const(dM), @Const(senders), @Const(receivers), @Const(row_ptr), @Const(n_edges), table, e0, nb, R)
    u, il = @index(Global, NTuple)
    _sender_work_item!(dx, StoredHarmonics(Y), w, dM, senders, receivers, row_ptr, n_edges[1], table, e0, nb, R, u, il)
end

# Work item `(u, il)`: channel `u` of receivers `lo + il - 1`, `lo + il - 1 + R`, … of the chunk's
# receiver range `lo:hi`.
@inline function _sender_work_item!(dx, Y, w, dM, senders, receivers, row_ptr, n_edges, table, e0, nb, R, u, il)
    lo, hi = _receiver_range(receivers, n_edges, e0, nb)
    for i in (lo + il - 1):R:hi
        _tp_sender_receiver!(dx, u, i, Y, w, dM, senders, _receiver_edges(row_ptr, i, e0, nb), e0, table)
    end
    return nothing
end

"""
    _edge_gradient!(ws, model, k, chunk) -> nothing

Adds to `ws.dEdr[:, e]`, for each edge `e` of `chunk`, the terms of interaction `k` of ∂E/∂r_e:
∂E/∂Y · ∂Y/∂r from `ws.dYe` and `ws.dY`, then (∂E/∂d) r_e / d with ∂E/∂d from the radial-feature
cotangents `ws.dfeats_chunk`, the density term of the receiver (`ws.ddensity[:, k]`, through
`density_edge`'s derivative) and the feature derivatives `ws.dfeats`.
"""
function _edge_gradient!(ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, k::Int, chunk::EdgeChunk) where {T}
    nb = _chunk_length(ws, chunk)
    nl = chunk.nl
    _launch!(
        _edge_gradient_kernel!, ws.backend, nb, ws.dEdr, ws.dYe, ws.dY, ws.dfeats_chunk, ws.feats, ws.dfeats, view(ws.ddensity, :, k),
        model.density[k], ws.r, ws.d, nl.receivers, nl.n_edges, first(chunk.edges) - 1,
    )
    return nothing
end

@kernel function _edge_gradient_kernel!(
        dEdr, @Const(dYe), @Const(dY), @Const(dfc), @Const(feats), @Const(dfeats), @Const(ddensity), @Const(wd), @Const(r), @Const(d),
        @Const(receivers), @Const(n_edges), e0
    )
    col = @index(Global)
    if e0 + col <= n_edges[1]
        _edge_gradient!(dEdr, dYe, dY, dfc, feats, dfeats, ddensity, wd, r, d, receivers, col, e0)
    end
end

# Edge e = e0 + col: dEdr[:, e] += Σ_j dYe[j] dY[j, :] (in that order, as the CPU's
# `_messages_pullback!`), then += dd r[:, e] with dd = Σ_b (dfc[b] + a wd[b]) dfeats[b] / d and
# a = ddensity[receiver] ∂ρ/∂(feats·wd/√nb), ρ = tanh(z²) the edge's density (`density_edge`).
@inline function _edge_gradient!(dEdr, dYe, dY, dfc, feats, dfeats, ddensity, wd, r, d, receivers, col, e0)
    T = eltype(dEdr)
    e = e0 + col
    for a in 1:3
        acc = zero(T)
        for j in axes(dYe, 1)
            acc += dYe[j, col] * dY[j, a, col]
        end
        dEdr[a, e] += acc
    end
    rb = sqrt(T(size(feats, 1)))
    z = zero(T)
    for b in axes(feats, 1)
        z += feats[b, col] * (wd[b] / rb)
    end
    ρ = tanh(z * z)
    a = ddensity[receivers[e]] * (2 * z * (one(T) - ρ * ρ) / rb)
    dd = zero(T)
    for b in axes(dfeats, 1)
        dd += (dfc[b, col] + a * wd[b]) * dfeats[b, col]
    end
    dd /= d[e]
    for c in 1:3
        dEdr[c, e] += dd * r[c, e]
    end
    return nothing
end
