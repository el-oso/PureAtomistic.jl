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
