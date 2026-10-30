# MACE model settings read from the attributes and buffers of a pickled ScaleShiftMACE.

# One term `mul x (l, parity)` of an e3nn `Irreps`; `parity` is 1 (even) or -1 (odd).
struct IrrepMul
    mul::Int
    l::Int
    parity::Int
end

struct InteractionSpec
    class::String
    avg_num_neighbors::Float64
    radial_MLP::Vector{Int}
    node_feats_irreps::Vector{IrrepMul}
    edge_attrs_irreps::Vector{IrrepMul}
    edge_feats_irreps::Vector{IrrepMul}
    target_irreps::Vector{IrrepMul}
    hidden_irreps::Vector{IrrepMul}
end

# `hidden_irreps` is empty for a linear readout.
struct ReadoutSpec
    class::String
    hidden_irreps::Vector{IrrepMul}
end

struct MACEHyperparameters
    model_class::String
    r_max::Float64
    num_interactions::Int
    atomic_numbers::Vector{Int}
    heads::Vector{String}
    interactions::Vector{InteractionSpec}
    correlations::Vector{Int}
    readouts::Vector{ReadoutSpec}
end

const SCALE_SHIFT_MACE = "mace.modules.models.ScaleShiftMACE"

# Name of the class whose instance is the module node `idx`.
function module_class(t::PickleTree, idx::Integer)
    n = t.nodes[idx]
    n.kind == PK_CALL || error("checkpoint: expected a module object")
    c = t.nodes[n.items[1]]
    c.kind == PK_GLOBAL || error("checkpoint: module class is not a global")
    return c.str
end

# Attribute `key` of the module state `st`; errors when absent.
function attribute(t::PickleTree, st::Int, key::String, path::String)
    i = dict_get(t, st, key)
    iszero(i) && error(lazy"checkpoint: $(describe_module(path)) has no attribute $key")
    return i
end

# Submodule `key` of the module state `st`.
function submodule(t::PickleTree, st::Int, key::String, path::String)
    i = dict_get(t, module_dict(t, st, "_modules", path), key)
    (iszero(i) || t.nodes[i].kind == PK_NONE) && error(lazy"checkpoint: $(describe_module(path)) has no submodule $key")
    return i
end

# Submodules of the ModuleList (or other container) state `st`, in order.
submodules(t::PickleTree, st::Int, path::String) = [v for (_, v) in dict_pairs(t, module_dict(t, st, "_modules", path))]

# The tensor stored under `key` in the `_buffers` dict of the module state `st`.
function buffer_tensor(a::TorchArchive, t::PickleTree, st::Int, key::String, path::String)
    i = dict_get(t, module_dict(t, st, "_buffers", path), key)
    (iszero(i) || !node_is_call(t, i, "torch._utils._rebuild_tensor_v2")) && error(lazy"checkpoint: $(describe_module(path)) has no tensor buffer $key")
    return read_tensor(a, t, i, string(path, key), false)
end

function float_scalar(a::TorchArchive, t::PickleTree, st::Int, key::String, path::String)
    x = buffer_tensor(a, t, st, key, path)
    (x isa CheckpointTensor{Float64} && numel(x) == 1) || error(lazy"checkpoint: $path$key is not a Float64 scalar")
    return x.data[1]
end

function int_buffer(a::TorchArchive, t::PickleTree, st::Int, key::String, path::String)
    x = buffer_tensor(a, t, st, key, path)
    x isa CheckpointTensor{Int64} || error(lazy"checkpoint: $path$key is not an Int64 tensor")
    return Vector{Int}(x.data)
end

function float_value(t::PickleTree, idx::Integer, what::String)
    n = t.nodes[idx]
    n.kind == PK_FLOAT && return n.float
    n.kind == PK_INT && return Float64(n.int)
    error(lazy"checkpoint: $what is not a number")
end

function irrep_parity(c::Char, l::Int, s::AbstractString)
    c == 'e' && return 1
    c == 'o' && return -1
    c == 'y' && return iseven(l) ? 1 : -1
    error(lazy"checkpoint: irreps string \"$s\" has an unknown parity $c")
end

# Terms of an e3nn irreps string such as "128x0e + 64x1o"; a term without `x` has
# multiplicity 1.
function parse_irreps(s::AbstractString)
    out = IrrepMul[]
    for term in split(s, '+')
        term = strip(term)
        isempty(term) && continue
        parts = split(term, 'x')
        length(parts) <= 2 || error(lazy"checkpoint: malformed irreps string \"$s\"")
        mul = length(parts) == 2 ? tryparse(Int, parts[1]) : 1
        ir = parts[end]
        length(ir) >= 2 || error(lazy"checkpoint: malformed irreps string \"$s\"")
        l = tryparse(Int, ir[1:prevind(ir, lastindex(ir))])
        (isnothing(mul) || isnothing(l)) && error(lazy"checkpoint: malformed irreps string \"$s\"")
        push!(out, IrrepMul(mul, l, irrep_parity(ir[end], l, s)))
    end
    return out
end

# The single argument of the e3nn object `idx` (a tuple subclass pickled as cls((items...),)).
function e3nn_items(t::PickleTree, idx::Integer, cls::String, what::String)
    node_is_call(t, idx, cls) || error(lazy"checkpoint: $what is not an $cls")
    args = call_args(t, idx)
    (length(args) == 1 && t.nodes[args[1]].kind == PK_TUPLE) || error(lazy"checkpoint: $what: malformed $cls")
    return t.nodes[args[1]].items
end

# An `Irreps` attribute, stored either as an e3nn `Irreps` object or as its string form.
function irreps_value(t::PickleTree, idx::Integer, what::String)
    n = t.nodes[idx]
    n.kind == PK_STR && return parse_irreps(n.str)
    out = IrrepMul[]
    for m in e3nn_items(t, idx, "e3nn.o3._irreps.Irreps", what)
        mi = e3nn_items(t, m, "e3nn.o3._irreps._MulIr", what)
        length(mi) == 2 || error(lazy"checkpoint: $what: malformed _MulIr")
        lp = e3nn_items(t, mi[2], "e3nn.o3._irreps.Irrep", what)
        length(lp) == 2 || error(lazy"checkpoint: $what: malformed Irrep")
        mul, l, p = t.nodes[mi[1]], t.nodes[lp[1]], t.nodes[lp[2]]
        (mul.kind == PK_INT && l.kind == PK_INT && p.kind == PK_INT && (p.int == 1 || p.int == -1)) ||
            error(lazy"checkpoint: $what: malformed irreps term")
        push!(out, IrrepMul(Int(mul.int), Int(l.int), Int(p.int)))
    end
    return out
end

function irreps_attribute(t::PickleTree, st::Int, key::String, path::String)
    return irreps_value(t, attribute(t, st, key, path), string(path, key))
end

function read_interaction(a::TorchArchive, t::PickleTree, idx::Int, path::String)
    st = module_state(t, idx, path)
    # Newer MACE registers avg_num_neighbors as a buffer; older files keep a float attribute.
    avg = if !iszero(dict_get(t, module_dict(t, st, "_buffers", path), "avg_num_neighbors"))
        float_scalar(a, t, st, "avg_num_neighbors", path)
    else
        float_value(t, attribute(t, st, "avg_num_neighbors", path), string(path, "avg_num_neighbors"))
    end
    return InteractionSpec(
        module_class(t, idx), avg,
        int_items(t, attribute(t, st, "radial_MLP", path)),
        irreps_attribute(t, st, "node_feats_irreps", path),
        irreps_attribute(t, st, "edge_attrs_irreps", path),
        irreps_attribute(t, st, "edge_feats_irreps", path),
        irreps_attribute(t, st, "target_irreps", path),
        irreps_attribute(t, st, "hidden_irreps", path),
    )
end

# Body order of a product block: each contraction holds `correlation - 1` weights in
# its `weights` ParameterList; all contractions of one block must agree.
function read_correlation(t::PickleTree, idx::Int, path::String)
    st = module_state(t, idx, path)
    sc_path = string(path, "symmetric_contractions.")
    sc = module_state(t, submodule(t, st, "symmetric_contractions", path), sc_path)
    cs_path = string(sc_path, "contractions.")
    cs = module_state(t, submodule(t, sc, "contractions", sc_path), cs_path)
    correlation = 0
    for c in submodules(t, cs, cs_path)
        cst = module_state(t, c, cs_path)
        w = module_state(t, submodule(t, cst, "weights", cs_path), cs_path)
        k = length(dict_pairs(t, module_dict(t, w, "_parameters", cs_path))) + 1
        iszero(correlation) || k == correlation ||
            error(lazy"checkpoint: contractions of $path disagree on the correlation order ($correlation and $k)")
        correlation = k
    end
    iszero(correlation) && error(lazy"checkpoint: product block $path has no contractions")
    return correlation
end

function read_readout(t::PickleTree, idx::Int, path::String)
    st = module_state(t, idx, path)
    hidden = iszero(dict_get(t, st, "hidden_irreps")) ? IrrepMul[] : irreps_attribute(t, st, "hidden_irreps", path)
    return ReadoutSpec(module_class(t, idx), hidden)
end

function string_items(t::PickleTree, idx::Integer, what::String)
    n = t.nodes[idx]
    (n.kind == PK_LIST || n.kind == PK_TUPLE) || error(lazy"checkpoint: $what is not a list of strings")
    out = String[]
    for j in n.items
        t.nodes[j].kind == PK_STR || error(lazy"checkpoint: $what is not a list of strings")
        push!(out, t.nodes[j].str)
    end
    return out
end

"""
    read_hyperparameters(a::TorchArchive, t::PickleTree) -> MACEHyperparameters

Model settings of the ScaleShiftMACE pickled as `t`, whose tensors live in `a`.
"""
function read_hyperparameters(a::TorchArchive, t::PickleTree)
    root = Int(t.root)
    cls = module_class(t, root)
    cls == SCALE_SHIFT_MACE || error(lazy"unsupported model class $cls")
    st = module_state(t, root, "")
    num_interactions = int_buffer(a, t, st, "num_interactions", "")
    length(num_interactions) == 1 || error("checkpoint: num_interactions is not a scalar")
    n = num_interactions[1]
    interactions = InteractionSpec[]
    for (k, m) in enumerate(submodules(t, module_state(t, submodule(t, st, "interactions", ""), "interactions."), "interactions."))
        push!(interactions, read_interaction(a, t, m, string("interactions.", k - 1, ".")))
    end
    correlations = Int[]
    for (k, m) in enumerate(submodules(t, module_state(t, submodule(t, st, "products", ""), "products."), "products."))
        push!(correlations, read_correlation(t, m, string("products.", k - 1, ".")))
    end
    readouts = ReadoutSpec[]
    for (k, m) in enumerate(submodules(t, module_state(t, submodule(t, st, "readouts", ""), "readouts."), "readouts."))
        push!(readouts, read_readout(t, m, string("readouts.", k - 1, ".")))
    end
    (length(interactions) == n && length(correlations) == n && length(readouts) == n) ||
        error(lazy"checkpoint: num_interactions is $n but the model has $(length(interactions)) interactions, $(length(correlations)) products and $(length(readouts)) readouts")
    return MACEHyperparameters(
        cls, float_scalar(a, t, st, "r_max", ""), n, int_buffer(a, t, st, "atomic_numbers", ""),
        string_items(t, attribute(t, st, "heads", ""), "heads"), interactions, correlations, readouts,
    )
end
