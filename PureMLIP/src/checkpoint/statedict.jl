# The tensors of a torch module tree, in `state_dict()` order.

struct StateDict
    f64::Dict{String, CheckpointTensor{Float64}}
    i64::Dict{String, CheckpointTensor{Int64}}
    names::Vector{String}
end

# Index of the value stored under `key` in the PK_DICT `idx`, or 0 when absent.
function dict_get(t::PickleTree, idx::Integer, key::String)
    n = t.nodes[idx]
    n.kind == PK_DICT || error(lazy"checkpoint: expected a dict while looking up $key")
    for j in 1:2:(length(n.items) - 1)
        k = t.nodes[n.items[j]]
        k.kind == PK_STR && k.str == key && return Int(n.items[j + 1])
    end
    return 0
end

# Entries of the PK_DICT `idx` as (key, value index) pairs in insertion order.
function dict_pairs(t::PickleTree, idx::Integer)
    n = t.nodes[idx]
    n.kind == PK_DICT || error("checkpoint: expected a dict")
    out = Pair{String, Int}[]
    for j in 1:2:(length(n.items) - 1)
        k = t.nodes[n.items[j]]
        k.kind == PK_STR || error("checkpoint: dict key is not a string")
        push!(out, k.str => Int(n.items[j + 1]))
    end
    return out
end

# How a module path reads in a message; the root module's path is empty.
describe_module(path::String) = isempty(path) ? "root module" : string("module ", path)

# One of the `_parameters`, `_buffers` or `_modules` dicts of the module state `st`.
function module_dict(t::PickleTree, st::Int, key::String, path::String)
    i = dict_get(t, st, key)
    iszero(i) && error(lazy"checkpoint: $(describe_module(path)) has no $key")
    return i
end

# The dict holding `_parameters`, `_buffers`, `_modules` for the module node `idx`.
function module_state(t::PickleTree, idx::Integer, path::String)
    n = t.nodes[idx]
    n.kind == PK_CALL || error(lazy"checkpoint: $(describe_module(path)) is not an object")
    if node_is_call(t, idx, "torch.fx.graph_module.reduce_graph_module")
        args = call_args(t, idx)
        isempty(args) && error(lazy"checkpoint: $(describe_module(path)) has no body")
        return Int(args[1])
    end
    length(n.items) == 3 || error(lazy"checkpoint: $(describe_module(path)) has no state")
    return Int(n.items[3])
end

# Names listed in `_non_persistent_buffers_set` of the module state `st`.
function non_persistent_names(t::PickleTree, st::Int)
    i = dict_get(t, st, "_non_persistent_buffers_set")
    names = String[]
    i == 0 && return names
    node_is_call(t, i, "__builtin__.set") || error("checkpoint: _non_persistent_buffers_set is not a set")
    args = call_args(t, i)
    for l in args
        t.nodes[l].kind == PK_LIST || error("checkpoint: _non_persistent_buffers_set holds a non-list")
        for s in t.nodes[l].items
            t.nodes[s].kind == PK_STR || error("checkpoint: _non_persistent_buffers_set holds a non-string")
            push!(names, t.nodes[s].str)
        end
    end
    return names
end

function add_tensor!(sd::StateDict, a::TorchArchive, t::PickleTree, idx::Int, name::String)
    node = t.nodes[idx]
    node.kind == PK_NONE && return nothing
    requires_grad = false
    if node_is_call(t, idx, "torch._utils._rebuild_parameter")
        args = call_args(t, idx)
        length(args) == 3 || error(lazy"checkpoint: parameter $name: _rebuild_parameter takes 3 arguments")
        t.nodes[args[2]].kind == PK_BOOL || error(lazy"checkpoint: parameter $name: requires_grad is not a bool")
        requires_grad = t.nodes[args[2]].int != 0
        idx = Int(args[1])
    end
    node_is_call(t, idx, "torch._utils._rebuild_tensor_v2") || error(lazy"checkpoint: $name is not a tensor")
    (name in sd.names) && error(lazy"checkpoint: duplicate state dict name $name")
    x = read_tensor(a, t, idx, name, requires_grad)
    if x isa CheckpointTensor{Float64}
        sd.f64[name] = x
    else
        sd.i64[name] = x
    end
    push!(sd.names, name)
    return nothing
end

function walk_module!(sd::StateDict, a::TorchArchive, t::PickleTree, idx::Int, prefix::String)
    st = module_state(t, idx, prefix)
    for key in ("_parameters", "_buffers", "_modules")
        module_dict(t, st, key, prefix)
    end
    skip = non_persistent_names(t, st)
    for (n, v) in dict_pairs(t, dict_get(t, st, "_parameters"))
        add_tensor!(sd, a, t, v, string(prefix, n))
    end
    for (n, v) in dict_pairs(t, dict_get(t, st, "_buffers"))
        n in skip || add_tensor!(sd, a, t, v, string(prefix, n))
    end
    for (n, v) in dict_pairs(t, dict_get(t, st, "_modules"))
        t.nodes[v].kind == PK_NONE && continue
        walk_module!(sd, a, t, v, string(prefix, n, "."))
    end
    return nothing
end

"""
    read_state_dict(a::TorchArchive, t::PickleTree) -> StateDict

Collect the parameters and persistent buffers of the module tree rooted at `t.root`,
depth first in torch's order (own parameters, own buffers, then submodules).
`None` entries are skipped; a duplicate name is an error.
"""
function read_state_dict(a::TorchArchive, t::PickleTree)
    bo = entry_data(a, "byteorder")
    String(bo[:]) == "little" || error("checkpoint: big-endian checkpoints are not supported")
    sd = StateDict(Dict{String, CheckpointTensor{Float64}}(), Dict{String, CheckpointTensor{Int64}}(), String[])
    walk_module!(sd, a, t, Int(t.root), "")
    return sd
end
