"""
    DeviceChannelwiseTP{T, IM, V}

A `ChannelwiseTP{T}` on a device: entry `q` of `host.entries`, `(i, j, k, v)`, is
`index[:, q] == (i, j, k)` and `value[q] == v`. The entry range of each instruction is
`host.paths`.
"""
struct DeviceChannelwiseTP{T, IM <: AbstractMatrix{Int32}, V <: AbstractVector{T}}
    index::IM
    value::V
    host::ChannelwiseTP{T}
end

function DeviceChannelwiseTP(backend::Backend, tp::ChannelwiseTP{T}) where {T}
    index = Int32[e[a] for a in 1:3, e in tp.entries]
    value = T[e[4] for e in tp.entries]
    return DeviceChannelwiseTP(_upload(backend, index), _upload(backend, value), tp)
end

# The nonzero entries of `tp` as a value for a type parameter: `(K_out, K_in2, entries)` with
# entries `(i, j, k, p, v)` (components of x, Y and the output, instruction, coefficient) in the
# order of `tp.entries`. A kernel specialized on it holds the output components in registers.
_tp_table(tp::ChannelwiseTP) =
    Val((tp.K_out, tp.K_in2, Tuple((e[1], e[2], e[3], p, e[4]) for p in eachindex(tp.paths) for e in tp.entries[tp.paths[p]])))

"""
    _tp_receiver!(M, u, i, Y, w, x, senders, edges, e0, ::Val{table})

Channel `u` of the summed messages of receiver `i`: adds the message of each edge `e` of `edges`,
in order, to `M[u, i, :]` as [`tp_edge_add!`](@ref PureMLIP.Tensors.tp_edge_add!) does, with the
edge attributes `Y(e, e - e0, Val(K_in2))` (a tuple), the weights `w[u, :, e - e0]` and the sender
features `x[u, senders[e], :]` (node layout `(C, n, K)`). `table` is `_tp_table` of the tensor
product. The components of `M[u, i, :]` are read once, held in registers while the edges are
added, and written once.
"""
@generated function _tp_receiver!(M, u, i, Y, w, x, senders, edges, e0, ::Val{table}) where {table}
    K, K2, entries = table
    acc(k) = Symbol(:m, k)
    sym(s, a) = Symbol(s, a)
    ps = sort(unique(e[4] for e in entries))
    is = sort(unique(e[1] for e in entries))
    return quote
        $([:($(acc(k)) = M[u, i, $k]) for k in 1:K]...)
        for e in edges
            col = e - e0
            s = senders[e]
            y = Y(e, col, Val($K2))
            $([:($(sym(:w, p)) = w[u, $p, col]) for p in ps]...)
            $([:($(sym(:x, a)) = x[u, s, $a]) for a in is]...)
            $([:($(acc(k)) += (($v * y[$j]) * $(sym(:w, p))) * $(sym(:x, a))) for (a, j, k, p, v) in entries]...)
        end
        $([:(M[u, i, $k] = $(acc(k))) for k in 1:K]...)
        return nothing
    end
end

"""
    _tp_edge_pullback!(dw, u, e, col, Y, w, x, dM, s, r, ::Val{table}) -> NTuple{K_in2}

Channel `u` of the pullback of the message of edge `e` from atom `s` to atom `r`, held in chunk
column `col`, as [`tp_edge_pullback_add!`](@ref PureMLIP.Tensors.tp_edge_pullback_add!) computes
it with the cotangent `dM[u, r, :]` (node layout `(C, n, K)`): writes the weight cotangents
`dw[u, p, col]` of every instruction `p`, and returns this channel's terms of the cotangent of the
edge attributes `Y(e, col, Val(K_in2))`, which the caller adds over the channels.
"""
@generated function _tp_edge_pullback!(dw, u, e, col, Y, w, x, dM, s, r, ::Val{table}) where {table}
    K, K2, entries = table
    ps = sort(unique(e[4] for e in entries))
    is = sort(unique(e[1] for e in entries))
    ks = sort(unique(e[3] for e in entries))
    sym(s, a) = Symbol(s, a)
    return quote
        T = eltype(dw)
        y = Y(e, col, Val($K2))
        $([:($(sym(:w, p)) = w[u, $p, col]) for p in ps]...)
        $([:($(sym(:x, a)) = x[u, s, $a]) for a in is]...)
        $([:($(sym(:g, k)) = dM[u, r, $k]) for k in ks]...)
        $([:($(sym(:dw, p)) = zero(T)) for p in ps]...)
        $([:($(sym(:dy, j)) = zero(T)) for j in 1:K2]...)
        $(
            [
                quote
                    $(sym(:dw, p)) += (($v * y[$j]) * $(sym(:g, k))) * $(sym(:x, a))
                    $(sym(:dy, j)) += $v * (($(sym(:g, k)) * $(sym(:w, p))) * $(sym(:x, a)))
                end for (a, j, k, p, v) in entries
            ]...
        )
        $([:(dw[u, $p, col] = $(sym(:dw, p))) for p in ps]...)
        return ($([sym(:dy, j) for j in 1:K2]...),)
    end
end

"""
    _tp_sender_receiver!(dx, u, i, Y, w, dM, senders, edges, e0, ::Val{table})

Channel `u` of the cotangent of the sender features of atom `i` from the messages that `i`
sends: adds to `dx[u, i, :]`, for each edge `e` of `edges` (edges received by `i`, in order), the
term of [`tp_edge_pullback_add!`](@ref PureMLIP.Tensors.tp_edge_pullback_add!) of its reverse edge,
which runs from `i` to `senders[e]`. The reverse edge has the same length, so the same radial-MLP
weights `w[u, :, e - e0]`, and the opposite vector, so the harmonics `(-1)^l Y` of degree `l`;
the cotangent is that of its receiver, `dM[u, senders[e], :]`. Summing over the edges received
by `i` reaches every edge sent by `i` once, as the reverse map pairs them.
"""
@generated function _tp_sender_receiver!(dx, u, i, Y, w, dM, senders, edges, e0, ::Val{table}) where {table}
    K, K2, entries = table
    acc(a) = Symbol(:d, a)
    sym(s, a) = Symbol(s, a)
    ps = sort(unique(e[4] for e in entries))
    is = sort(unique(e[1] for e in entries))
    ks = sort(unique(e[3] for e in entries))
    # The harmonic in component j has degree l = isqrt(j - 1) and parity (-1)^l.
    parity(j) = iseven(isqrt(j - 1)) ? 1 : -1
    return quote
        $([:($(acc(a)) = dx[u, i, $a]) for a in is]...)
        for e in edges
            col = e - e0
            s = senders[e]
            y = Y(e, col, Val($K2))
            $([:($(sym(:w, p)) = w[u, $p, col]) for p in ps]...)
            $([:($(sym(:g, k)) = dM[u, s, $k]) for k in ks]...)
            $([:($(acc(a)) += (($(parity(j) * v) * y[$j]) * $(sym(:g, k))) * $(sym(:w, p))) for (a, j, k, p, v) in entries]...)
        end
        $([:(dx[u, i, $a] = $(acc(a))) for a in is]...)
        return nothing
    end
end
