# Constants pickled into the e3nn modules of a ScaleShiftMACE: the normalize2mom factors,
# the instruction lists of Linear and TensorProduct modules, and the root flags.

# `f` is the pickled global of the wrapped function, e.g. "torch.nn.functional.silu".
struct Normalize2Mom
    cst::Float64
    f::String
end

# Indices are 0-based as pickled.
struct LinearInstruction
    i_in::Int
    i_out::Int
    path_shape::Vector{Int}
    path_weight::Float64
end

struct LinearSpec
    irreps_in::Vector{IrrepMul}
    irreps_out::Vector{IrrepMul}
    instructions::Vector{LinearInstruction}
end

struct TPInstruction
    i_in1::Int
    i_in2::Int
    i_out::Int
    connection_mode::String
    has_weight::Bool
    path_weight::Float64
    path_shape::Vector{Int}
end

struct TPSpec
    irreps_in1::Vector{IrrepMul}
    irreps_in2::Vector{IrrepMul}
    irreps_out::Vector{IrrepMul}
    instructions::Vector{TPInstruction}
end

"""
    MACEConstants

Constants stored in the pickle tree of a checkpoint, keyed by module path (`.`-separated,
for example `"interactions.1.conv_tp"`). A module reached through several paths (layers
of one `FullyConnectedNet` share an activation) is recorded under each path.

  - `normalize2mom`: every `normalize2mom` module.
  - `linear`: every e3nn `Linear`.
  - `tensor_product`: every `TensorProduct` and `FullyConnectedTensorProduct`.
  - `pair_repulsion`: the root attribute; `false` when absent.
  - `distance_transform`: class of `radial_embedding.distance_transform`; `""` when absent.
  - `product_use_sc`: the `use_sc` flag of each product block.
"""
struct MACEConstants
    normalize2mom::Dict{String, Normalize2Mom}
    linear::Dict{String, LinearSpec}
    tensor_product::Dict{String, TPSpec}
    pair_repulsion::Bool
    distance_transform::String
    product_use_sc::Vector{Bool}
end

const NORMALIZE2MOM = "e3nn.math._normalize_activation.normalize2mom"
const E3NN_LINEAR = "e3nn.o3._linear.Linear"
const E3NN_LINEAR_INSTRUCTION = "e3nn.o3._linear.Instruction"
const E3NN_TP_INSTRUCTION = "e3nn.o3._tensor_product._instruction.Instruction"
const E3NN_TENSOR_PRODUCTS = (
    "e3nn.o3._tensor_product._tensor_product.TensorProduct",
    "e3nn.o3._tensor_product._sub.FullyConnectedTensorProduct",
)

child_path(path::String, name::String) = isempty(path) ? name : string(path, ".", name)

function bool_value(t::PickleTree, idx::Integer, path::String, what::String)
    t.nodes[idx].kind == PK_BOOL || error(lazy"checkpoint: $path: malformed $what")
    return t.nodes[idx].int != 0
end

function int_value(t::PickleTree, idx::Integer, path::String, what::String)
    t.nodes[idx].kind == PK_INT || error(lazy"checkpoint: $path: malformed $what")
    return Int(t.nodes[idx].int)
end

function float_item(t::PickleTree, idx::Integer, path::String, what::String)
    t.nodes[idx].kind == PK_FLOAT || error(lazy"checkpoint: $path: malformed $what")
    return t.nodes[idx].float
end

function shape_item(t::PickleTree, idx::Integer, path::String, what::String)
    n = t.nodes[idx]
    n.kind == PK_TUPLE || error(lazy"checkpoint: $path: malformed $what")
    return [int_value(t, j, path, what) for j in n.items]
end

# Arguments of the instruction object `idx`, a call of the global `cls` with `nargs` arguments.
function instruction_args(t::PickleTree, idx::Integer, cls::String, nargs::Int, path::String)
    node_is_call(t, idx, cls) || error(lazy"checkpoint: $path: malformed instruction")
    args = call_args(t, idx)
    length(args) == nargs || error(lazy"checkpoint: $path: malformed instruction")
    return args
end

# The list of instructions stored under `instructions` in the module state `st`.
function instruction_list(t::PickleTree, st::Int, path::String)
    i = attribute(t, st, "instructions", path)
    t.nodes[i].kind == PK_LIST || error(lazy"checkpoint: $path: malformed instructions")
    return t.nodes[i].items
end

function read_normalize2mom(t::PickleTree, st::Int, path::String)
    f = attribute(t, st, "f", path)
    t.nodes[f].kind == PK_GLOBAL || error(lazy"checkpoint: $path: malformed f")
    return Normalize2Mom(float_item(t, attribute(t, st, "cst", path), path, "cst"), t.nodes[f].str)
end

function read_linear(t::PickleTree, st::Int, path::String)
    instructions = LinearInstruction[]
    for j in instruction_list(t, st, path)
        a = instruction_args(t, j, E3NN_LINEAR_INSTRUCTION, 4, path)
        push!(
            instructions, LinearInstruction(
                int_value(t, a[1], path, "instruction"), int_value(t, a[2], path, "instruction"),
                shape_item(t, a[3], path, "instruction"), float_item(t, a[4], path, "instruction"),
            )
        )
    end
    return LinearSpec(
        irreps_attribute(t, st, "irreps_in", path), irreps_attribute(t, st, "irreps_out", path), instructions,
    )
end

function read_tensor_product(t::PickleTree, st::Int, path::String)
    instructions = TPInstruction[]
    for j in instruction_list(t, st, path)
        a = instruction_args(t, j, E3NN_TP_INSTRUCTION, 7, path)
        t.nodes[a[4]].kind == PK_STR || error(lazy"checkpoint: $path: malformed instruction")
        push!(
            instructions, TPInstruction(
                int_value(t, a[1], path, "instruction"), int_value(t, a[2], path, "instruction"),
                int_value(t, a[3], path, "instruction"), t.nodes[a[4]].str,
                bool_value(t, a[5], path, "instruction"), float_item(t, a[6], path, "instruction"),
                shape_item(t, a[7], path, "instruction"),
            )
        )
    end
    return TPSpec(
        irreps_attribute(t, st, "irreps_in1", path), irreps_attribute(t, st, "irreps_in2", path),
        irreps_attribute(t, st, "irreps_out", path), instructions,
    )
end

function walk_constants!(k::MACEConstants, t::PickleTree, idx::Int, path::String)
    st = module_state(t, idx, path)
    cls = module_class(t, idx)
    if cls == NORMALIZE2MOM
        k.normalize2mom[path] = read_normalize2mom(t, st, path)
    elseif cls == E3NN_LINEAR
        k.linear[path] = read_linear(t, st, path)
    elseif cls in E3NN_TENSOR_PRODUCTS
        k.tensor_product[path] = read_tensor_product(t, st, path)
    end
    for (name, v) in dict_pairs(t, module_dict(t, st, "_modules", path))
        t.nodes[v].kind == PK_NONE && continue
        walk_constants!(k, t, v, child_path(path, name))
    end
    return nothing
end

"""
    read_constants(t::PickleTree) -> MACEConstants

Collect the constants of every module in the tree rooted at `t.root` (see `MACEConstants`).
"""
function read_constants(t::PickleTree)
    root = Int(t.root)
    st = module_state(t, root, "")
    pr = dict_get(t, st, "pair_repulsion")
    pair_repulsion = iszero(pr) ? false : bool_value(t, pr, "", "pair_repulsion")
    distance_transform = ""
    re = dict_get(t, module_dict(t, st, "_modules", ""), "radial_embedding")
    if !iszero(re) && t.nodes[re].kind != PK_NONE
        dt = dict_get(t, module_dict(t, module_state(t, re, "radial_embedding"), "_modules", "radial_embedding"), "distance_transform")
        iszero(dt) || t.nodes[dt].kind == PK_NONE || (distance_transform = module_class(t, dt))
    end
    use_sc = Bool[]
    for (k, m) in enumerate(submodules(t, module_state(t, submodule(t, st, "products", ""), "products"), "products"))
        p = string("products.", k - 1)
        push!(use_sc, bool_value(t, attribute(t, module_state(t, m, p), "use_sc", p), p, "use_sc"))
    end
    k = MACEConstants(
        Dict{String, Normalize2Mom}(), Dict{String, LinearSpec}(), Dict{String, TPSpec}(),
        pair_repulsion, distance_transform, use_sc,
    )
    walk_constants!(k, t, root, "")
    return k
end
