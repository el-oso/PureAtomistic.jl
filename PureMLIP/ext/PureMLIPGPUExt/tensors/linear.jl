# Device forms of `E3nnLinear` and `SpeciesLinear`: the weight blocks stacked into one device
# array, with the host map kept for its slots, offsets and sizes.

"""
    DeviceE3nnLinear{T, A3, V}

An `E3nnLinear{T}` on a device: `W[:, :, p] == host.W[p]` (`C_out × C_in` per instruction) and
`pw == host.pw`. The slots, offsets and sizes are read from `host`.
"""
struct DeviceE3nnLinear{T, A3 <: AbstractArray{T, 3}, V <: AbstractVector{T}}
    W::A3
    pw::V
    host::E3nnLinear{T}
end

function DeviceE3nnLinear(backend::Backend, lin::E3nnLinear{T}) where {T}
    W = zeros(T, lin.C_out, lin.C_in, length(lin.W))
    for p in eachindex(lin.W)
        W[:, :, p] = lin.W[p]
    end
    return DeviceE3nnLinear(_upload(backend, W), _upload(backend, lin.pw), lin)
end

"""
    DeviceSpeciesLinear{T, A4, V}

A `SpeciesLinear{T}` on a device: `W[:, :, s, p] == host.W[p][:, :, s]` (`C_out × C_in` per
species and instruction) and `c == host.c`.
"""
struct DeviceSpeciesLinear{T, A4 <: AbstractArray{T, 4}, V <: AbstractVector{T}}
    W::A4
    c::V
    host::SpeciesLinear{T}
end

function DeviceSpeciesLinear(backend::Backend, sl::SpeciesLinear{T}) where {T}
    W = zeros(T, sl.C_out, sl.C_in, sl.n_species, length(sl.W))
    for p in eachindex(sl.W)
        W[:, :, :, p] = sl.W[p]
    end
    return DeviceSpeciesLinear(_upload(backend, W), _upload(backend, sl.c), sl)
end
