# Device arrays: uploads, the checks every device entry point applies to its array arguments, and
# buffers that share one block of device memory.

# A copy of `A` with element type `S` in the memory of `backend`.
function _upload(backend::Backend, ::Type{S}, A::AbstractArray) where {S}
    D = allocate(backend, S, size(A))
    copyto!(D, convert(Array{S}, A))
    return D
end
_upload(backend::Backend, A::AbstractArray{T}) where {T} = _upload(backend, T, A)

# A zero-filled array with element type `S` and size `dims` in the memory of `backend`.
_device_zeros(backend::Backend, ::Type{S}, dims) where {S} = KernelAbstractions.zeros(backend, S, dims...)

# `Base.require_one_based_indexing` with the extension's prefix on its message.
function _require_one_based(A::AbstractArray...)
    try
        Base.require_one_based_indexing(A...)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError(string("gpu: arrays must be 1-based; ", err.msg)))
    end
    return nothing
end

# The backend of `positions`, after checking that every array of `others` lives on it.
function _common_backend(positions::AbstractArray, others::AbstractArray...)
    backend = get_backend(positions)
    for A in others
        get_backend(A) == backend || throw(ArgumentError("gpu: all arrays must be on the backend of the positions"))
    end
    return backend
end

# The name of a backend type without its `Backend` suffix: `Metal` for `MetalBackend`.
_backend_name(backend::Backend) = replace(string(nameof(typeof(backend))), r"Backend$" => "")

# Element type `T` must be one the backend computes in.
function _check_precision(backend::Backend, ::Type{T}) where {T}
    T === Float64 && !KernelAbstractions.supports_float64(backend) &&
        throw(ArgumentError(lazy"gpu: $(_backend_name(backend)) has no Float64; build the model as MACEModel{Float32}"))
    return nothing
end

# Launches `kernel` on `backend` over `ndrange` with `args`; a launch with no work items is
# skipped.
function _launch!(kernel, backend::Backend, ndrange, args...)
    prod(ndrange) > 0 && kernel(backend)(args...; ndrange)
    return nothing
end

# Zero-filled arrays over one block of device memory, given as groups of sizes, with the layout
# of the core's workspace pools: the arrays of a group lie side by side from the block's first
# element, and the groups overlap each other. Each member has the type of a separately allocated
# array of its size.
function _device_pool(backend::Backend, ::Type{T}, groups::Tuple{Vararg{Tuple{Vararg{NTuple{3, Int}}}}}) where {T}
    len = maximum(map(g -> sum(map(prod, g); init = 0), groups); init = 0)
    mem = _pool_block(backend, T, len)
    fill!(mem, zero(T))
    function place(g)
        sizes = map(prod, g)
        return map((offset, dims) -> _pool_member(mem, offset, dims), map(-, accumulate(+, sizes; init = 0), sizes), g)
    end
    return map(place, groups)
end

# On the CPU backend the block is a `Memory`, and `Base.wrap` (documented in Base but not declared
# public) gives each member the type `Array` of a separate allocation; a contiguous view of a GPU
# array is already an array of the GPU type.
_pool_block(::CPU, ::Type{T}, len::Int) where {T} = Memory{T}(undef, len)
_pool_block(backend::Backend, ::Type{T}, len::Int) where {T} = allocate(backend, T, len)

_pool_member(mem::Memory, offset::Int, dims::NTuple{3, Int}) =
    Base.wrap(Array, iszero(offset) || iszero(prod(dims)) ? memoryref(mem) : memoryref(mem, offset + 1), dims)
_pool_member(mem::AbstractVector, offset::Int, dims::NTuple{3, Int}) = reshape(view(mem, (offset + 1):(offset + prod(dims))), dims)

# `_launch!` with work groups of size `workgroupsize`. The size is given at the launch, not to the
# kernel's constructor, whose type would then depend on its value and dispatch at run time.
function _launch_grouped!(kernel, backend::Backend, ndrange, workgroupsize, args...)
    prod(ndrange) > 0 && kernel(backend)(args...; ndrange, workgroupsize)
    return nothing
end
