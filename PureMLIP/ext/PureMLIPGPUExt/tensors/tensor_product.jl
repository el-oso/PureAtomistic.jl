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
