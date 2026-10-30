# Tensors rebuilt from torch pickles. `data` is contiguous in torch (row-major) order;
# a 0-dimensional tensor has `shape == Int[]` and one element.
struct CheckpointTensor{T}
    shape::Vector{Int}
    data::Vector{T}
    requires_grad::Bool
end

numel(x::CheckpointTensor) = length(x.data)

"""
    torch_array(x::CheckpointTensor{T}, ::Val{N}) -> Array{T,N}

Array `A` with `A[i1, ..., iN] == torch_tensor[i1-1, ..., iN-1]`.
"""
function torch_array(x::CheckpointTensor{T}, ::Val{N}) where {T,N}
    length(x.shape) == N || throw(DimensionMismatch(lazy"tensor has $(length(x.shape)) dimensions, expected $N"))
    # Row-major data is column-major data of the reversed dimensions.
    rev = reshape(x.data, ntuple(d -> x.shape[N + 1 - d], Val(N)))
    return permutedims(rev, ntuple(d -> N + 1 - d, Val(N)))
end

# Little-endian element `k` (zero-based) of `bytes`.
function load_element(bytes::AbstractVector{UInt8}, ::Type{T}, k::Int) where {T<:Union{Float64,Int64}}
    base = firstindex(bytes) + k * sizeof(T)
    u = UInt64(0)
    for b in 0:(sizeof(T) - 1)
        u |= UInt64(bytes[base + b]) << (8b)
    end
    return reinterpret(T, u)
end

# Integers held by the PK_TUPLE (or PK_LIST) node `idx`.
function int_items(t::PickleTree, idx::Integer)
    n = t.nodes[idx]
    (n.kind == PK_TUPLE || n.kind == PK_LIST) || error("checkpoint: expected a tuple of integers")
    out = Int[]
    for j in n.items
        t.nodes[j].kind == PK_INT || error("checkpoint: expected a tuple of integers")
        push!(out, Int(t.nodes[j].int))
    end
    return out
end

function node_is_global(t::PickleTree, idx::Integer, name::String)
    n = t.nodes[idx]
    return n.kind == PK_GLOBAL && n.str == name
end

# Whether `idx` is a PK_CALL of the global `name`.
function node_is_call(t::PickleTree, idx::Integer, name::String)
    n = t.nodes[idx]
    return n.kind == PK_CALL && node_is_global(t, n.items[1], name)
end

function call_args(t::PickleTree, idx::Integer)
    a = t.nodes[t.nodes[idx].items[2]]
    a.kind == PK_TUPLE || error("checkpoint: call arguments are not a tuple")
    return a.items
end

# Reads the tensor rebuilt by the PK_CALL `idx` of `_rebuild_tensor_v2`; the element type
# follows the storage class.
function read_tensor(a::TorchArchive, t::PickleTree, idx::Integer, name::String, requires_grad::Bool)
    args = call_args(t, idx)
    (length(args) == 6 || length(args) == 7) ||
        error(lazy"checkpoint: tensor $name: _rebuild_tensor_v2 takes 6 or 7 arguments, got $(length(args))")
    st = t.nodes[args[1]]
    st.kind == PK_PERSID || error(lazy"checkpoint: tensor $name: storage is not a persistent id")
    pid = t.nodes[st.items[1]]
    (pid.kind == PK_TUPLE && length(pid.items) == 5) || error(lazy"checkpoint: tensor $name: malformed storage id")
    tag = t.nodes[pid.items[1]]
    cls = t.nodes[pid.items[2]]
    key = t.nodes[pid.items[3]]
    (tag.kind == PK_STR && tag.str == "storage" && cls.kind == PK_GLOBAL && key.kind == PK_STR) ||
        error(lazy"checkpoint: tensor $name: malformed storage id")
    t.nodes[args[2]].kind == PK_INT || error(lazy"checkpoint: tensor $name: storage offset is not an integer")
    offset = Int(t.nodes[args[2]].int)
    shape = int_items(t, args[3])
    stride = int_items(t, args[4])
    length(shape) == length(stride) || error(lazy"checkpoint: tensor $name: size and stride lengths differ")
    bytes = entry_data(a, "data/" * key.str)
    if cls.str == "torch.DoubleStorage"
        data = gather(bytes, Float64, offset, shape, stride, name)
        return CheckpointTensor{Float64}(shape, data, requires_grad)
    elseif cls.str == "torch.LongStorage"
        data = gather(bytes, Int64, offset, shape, stride, name)
        return CheckpointTensor{Int64}(shape, data, requires_grad)
    end
    error(lazy"checkpoint: tensor $name: unsupported storage type $(cls.str)")
end

# Elements of a strided view of `bytes` in row-major order of the view.
function gather(bytes::AbstractVector{UInt8}, ::Type{T}, offset::Int, shape::Vector{Int}, stride::Vector{Int}, name::String) where {T}
    nstorage = length(bytes) ÷ sizeof(T)
    n = prod(shape; init = 1)
    n >= 0 || error(lazy"checkpoint: tensor $name has a negative dimension")
    data = Vector{T}(undef, n)
    nd = length(shape)
    counter = zeros(Int, nd)
    lin = offset
    for k in eachindex(data)
        (0 <= lin < nstorage) || error(lazy"checkpoint: tensor $name reads outside its storage (element $(lin) of $nstorage)")
        data[k] = load_element(bytes, T, lin)
        # Advance the odometer, last dimension fastest.
        d = nd
        while d >= 1
            counter[d] += 1
            lin += stride[d]
            if counter[d] < shape[d]
                break
            end
            lin -= stride[d] * shape[d]
            counter[d] = 0
            d -= 1
        end
    end
    return data
end
