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

# Slot `off + 1 : off + d` of the node features `x` (`(C, n, K)` layout) as one `C × n·d` matrix.
_slot(x::AbstractArray{T, 3}, off::Int, d::Int) where {T} = reshape(view(x, :, :, (off + 1):(off + d)), size(x, 1), size(x, 2) * d)

"""
    _linear!(y, dl::DeviceE3nnLinear, x) -> y

[`linear!`](@ref PureMLIP.Tensors.linear!) on node features in the `(C, n, K)` layout: for each
instruction, one matrix product of its weights with the input slot, as one `C × n(2l + 1)`
matrix, added to the output slot. Overwrites `y`.
"""
function _linear!(y::AbstractArray{T, 3}, dl::DeviceE3nnLinear{T}, x::AbstractArray{T, 3}) where {T}
    lin = dl.host
    (size(x, 1), size(x, 3)) == (lin.C_in, lin.K_in) && size(y) == (lin.C_out, size(x, 2), lin.K_out) ||
        throw(DimensionMismatch(lazy"gpu: linear map from $(size(x)) to $(size(y)), expected ($(lin.C_in), n, $(lin.K_in)) to ($(lin.C_out), n, $(lin.K_out))"))
    fill!(y, zero(T))
    for p in eachindex(lin.slots)
        i, o = lin.slots[p]
        d = 2 * lin.in_ls[i] + 1
        mul!(_slot(y, lin.out_off[o], d), view(dl.W, :, :, p), _slot(x, lin.in_off[i], d), lin.pw[p], true)
    end
    return y
end

"""
    _species_linear!(y, dsl::DeviceSpeciesLinear, x, species, list, scratch, backend) -> y

[`species_linear!`](@ref PureMLIP.Tensors.species_linear!) on node features in the `(C, n, K)`
layout, for nodes whose species index `species[i]` is in `list`. With one species, one matrix
product per instruction over every node; with several, per species the input with the other
species' nodes set to zero (in `scratch`, the size of `x`) is multiplied by that species' weights
and added, so each node receives exactly its own species' product. Overwrites `y`.
"""
function _species_linear!(
        y::AbstractArray{T, 3}, dsl::DeviceSpeciesLinear{T}, x::AbstractArray{T, 3}, species::AbstractVector{Int32},
        list::AbstractVector{Int}, scratch::AbstractArray{T, 3}, backend::Backend
    ) where {T}
    sl = dsl.host
    (size(x, 1), size(x, 3)) == (sl.C_in, sl.K) && size(y) == (sl.C_out, size(x, 2), sl.K) && size(scratch) == size(x) ||
        throw(DimensionMismatch(lazy"gpu: species linear map from $(size(x)) to $(size(y)), expected ($(sl.C_in), n, $(sl.K)) to ($(sl.C_out), n, $(sl.K))"))
    fill!(y, zero(T))
    if length(list) == 1
        _species_products!(y, dsl, x, only(list))
    else
        for s in list
            _launch!(_mask_kernel!, backend, size(x), scratch, x, species, s)
            _species_products!(y, dsl, scratch, s)
        end
    end
    return y
end

function _species_products!(y::AbstractArray{T, 3}, dsl::DeviceSpeciesLinear{T}, x::AbstractArray{T, 3}, s::Int) where {T}
    sl = dsl.host
    for p in eachindex(sl.slots)
        i, o = sl.slots[p]
        d = 2 * sl.ls[i] + 1
        mul!(_slot(y, sl.off[o], d), view(dsl.W, :, :, s, p), _slot(x, sl.off[i], d), sl.c[p], true)
    end
    return nothing
end

@kernel function _mask_kernel!(xs, @Const(x), @Const(species), s)
    u, i, k = @index(Global, NTuple)
    _mask!(xs, x, species, s, u, i, k)
end

@inline function _mask!(xs, x, species, s, u, i, k)
    xs[u, i, k] = species[i] == s ? x[u, i, k] : zero(eltype(xs))
    return nothing
end

"""
    _linear_pullback!(dx, dl::DeviceE3nnLinear, dy) -> dx

[`linear_pullback!`](@ref PureMLIP.Tensors.linear_pullback!) on node features in the `(C, n, K)`
layout: for each instruction, one matrix product of its transposed weights with the output slot's
cotangent, added to the input slot. Overwrites `dx`.
"""
function _linear_pullback!(dx::AbstractArray{T, 3}, dl::DeviceE3nnLinear{T}, dy::AbstractArray{T, 3}) where {T}
    lin = dl.host
    (size(dy, 1), size(dy, 3)) == (lin.C_out, lin.K_out) && size(dx) == (lin.C_in, size(dy, 2), lin.K_in) ||
        throw(DimensionMismatch(lazy"gpu: linear pullback from $(size(dy)) to $(size(dx)), expected ($(lin.C_out), n, $(lin.K_out)) to ($(lin.C_in), n, $(lin.K_in))"))
    fill!(dx, zero(T))
    for p in eachindex(lin.slots)
        i, o = lin.slots[p]
        d = 2 * lin.in_ls[i] + 1
        mul!(_slot(dx, lin.in_off[i], d), transpose(view(dl.W, :, :, p)), _slot(dy, lin.out_off[o], d), lin.pw[p], true)
    end
    return dx
end

"""
    _species_linear_pullback!(dx, dsl::DeviceSpeciesLinear, dy, species, list, scratch, backend) -> dx

[`species_linear_pullback!`](@ref PureMLIP.Tensors.species_linear_pullback!) on node features in
the `(C, n, K)` layout, for nodes whose species index is in `list`, as
[`_species_linear!`](@ref) computes the forward map: with several species, per species the
cotangent with the other species' nodes set to zero (in `scratch`, at least as long as `dy`) is
multiplied by that species' transposed weights and added. Overwrites `dx`.
"""
function _species_linear_pullback!(
        dx::AbstractArray{T, 3}, dsl::DeviceSpeciesLinear{T}, dy::AbstractArray{T, 3}, species::AbstractVector{Int32},
        list::AbstractVector{Int}, scratch::AbstractArray{T, 3}, backend::Backend
    ) where {T}
    sl = dsl.host
    (size(dy, 1), size(dy, 3)) == (sl.C_out, sl.K) && size(dx) == (sl.C_in, size(dy, 2), sl.K) ||
        throw(DimensionMismatch(lazy"gpu: species linear pullback from $(size(dy)) to $(size(dx)), expected ($(sl.C_out), n, $(sl.K)) to ($(sl.C_in), n, $(sl.K))"))
    fill!(dx, zero(T))
    if length(list) == 1
        _species_pullback_products!(dx, dsl, dy, only(list))
    else
        masked = _front(scratch, size(dy))
        for s in list
            _launch!(_mask_kernel!, backend, size(dy), masked, dy, species, s)
            _species_pullback_products!(dx, dsl, masked, s)
        end
    end
    return dx
end

function _species_pullback_products!(dx::AbstractArray{T, 3}, dsl::DeviceSpeciesLinear{T}, dy::AbstractArray{T, 3}, s::Int) where {T}
    sl = dsl.host
    for p in eachindex(sl.slots)
        i, o = sl.slots[p]
        d = 2 * sl.ls[i] + 1
        mul!(_slot(dx, sl.off[i], d), transpose(view(dsl.W, :, :, s, p)), _slot(dy, sl.off[o], d), sl.c[p], true)
    end
    return nothing
end

# The first `prod(dims)` entries of the buffer `A`, as an array of size `dims`.
function _front(A::AbstractArray, dims::Dims)
    len = prod(dims)
    len <= length(A) || throw(ArgumentError(lazy"gpu: scratch of $(length(A)) entries for an array of size $dims"))
    return reshape(view(vec(A), 1:len), dims)
end
