# Restricted reader for pickle protocol 2 as written by torch.save. Only the opcodes
# and globals listed below are accepted; nothing is ever executed. The result is a
# flat tree of nodes that refer to each other by index.

@enum PickleKind::UInt8 PK_NONE PK_BOOL PK_INT PK_FLOAT PK_STR PK_BYTES PK_TUPLE PK_LIST PK_DICT PK_GLOBAL PK_CALL PK_PERSID

# `int` holds PK_INT values and PK_BOOL (0 or 1); `float` holds PK_FLOAT; `str` holds
# PK_STR, PK_GLOBAL ("module.name") and the bytes of PK_BYTES. `items` holds node
# indices: elements of PK_TUPLE/PK_LIST, k1, v1, k2, v2, ... of PK_DICT,
# [callable, args] (plus the BUILD state as a third item) of PK_CALL, and
# [persistent-id tuple] of PK_PERSID.
struct PickleNode
    kind::PickleKind
    int::Int64
    float::Float64
    str::String
    items::Vector{Int32}
end

PickleNode(kind::PickleKind; int::Integer = 0, float::Real = 0.0, str::AbstractString = "", items::AbstractVector{Int32} = Int32[]) =
    PickleNode(kind, Int64(int), Float64(float), String(str), Vector{Int32}(items))

struct PickleTree
    nodes::Vector{PickleNode}
    root::Int32
end

const ALLOWED_GLOBALS = Set{String}(
    [
        "__builtin__.set", "_codecs.encode", "collections.OrderedDict",
        "e3nn.math._normalize_activation.normalize2mom", "e3nn.nn._activation.Activation",
        "e3nn.nn._fc.FullyConnectedNet", "e3nn.nn._fc._Layer", "e3nn.o3._irreps.Irrep",
        "e3nn.o3._irreps.Irreps", "e3nn.o3._irreps._MulIr", "e3nn.o3._linear.Instruction",
        "e3nn.o3._linear.Linear", "e3nn.o3._spherical_harmonics.SphericalHarmonics",
        "e3nn.o3._tensor_product._instruction.Instruction",
        "e3nn.o3._tensor_product._sub.FullyConnectedTensorProduct",
        "e3nn.o3._tensor_product._tensor_product.TensorProduct",
        "mace.modules.blocks.AtomicEnergiesBlock", "mace.modules.blocks.EquivariantProductBasisBlock",
        "mace.modules.blocks.LinearNodeEmbeddingBlock", "mace.modules.blocks.LinearReadoutBlock",
        "mace.modules.blocks.NonLinearReadoutBlock", "mace.modules.blocks.RadialEmbeddingBlock",
        "mace.modules.blocks.RealAgnosticDensityInteractionBlock",
        "mace.modules.blocks.RealAgnosticDensityResidualInteractionBlock",
        "mace.modules.blocks.ScaleShiftBlock", "mace.modules.irreps_tools.reshape_irreps",
        "mace.modules.models.ScaleShiftMACE", "mace.modules.radial.AgnesiTransform",
        "mace.modules.radial.BesselBasis", "mace.modules.radial.PolynomialCutoff",
        "mace.modules.radial.ZBLBasis", "mace.modules.symmetric_contraction.Contraction",
        "mace.modules.symmetric_contraction.SymmetricContraction", "torch.DoubleStorage",
        "torch.LongStorage", "torch._utils._rebuild_parameter", "torch._utils._rebuild_tensor_v2",
        "torch.fx._symbolic_trace.Tracer", "torch.fx.graph_module.reduce_graph_module",
        "torch.nn.functional.silu", "torch.nn.modules.container.ModuleList",
        "torch.nn.modules.container.ParameterList",
    ]
)

# Opcodes this reader accepts (31 in total).
const OP_PROTO = 0x80
const OP_GLOBAL = UInt8('c')
const OP_BINPUT = UInt8('q')
const OP_LONG_BINPUT = UInt8('r')
const OP_BINGET = UInt8('h')
const OP_LONG_BINGET = UInt8('j')
const OP_BININT = UInt8('J')
const OP_BININT1 = UInt8('K')
const OP_BININT2 = UInt8('M')
const OP_BINFLOAT = UInt8('G')
const OP_BINUNICODE = UInt8('X')
const OP_MARK = UInt8('(')
const OP_EMPTY_TUPLE = UInt8(')')
const OP_TUPLE = UInt8('t')
const OP_TUPLE1 = 0x85
const OP_TUPLE2 = 0x86
const OP_TUPLE3 = 0x87
const OP_EMPTY_LIST = UInt8(']')
const OP_APPEND = UInt8('a')
const OP_APPENDS = UInt8('e')
const OP_EMPTY_DICT = UInt8('}')
const OP_SETITEM = UInt8('s')
const OP_SETITEMS = UInt8('u')
const OP_NONE = UInt8('N')
const OP_NEWTRUE = 0x88
const OP_NEWFALSE = 0x89
const OP_REDUCE = UInt8('R')
const OP_NEWOBJ = 0x81
const OP_BUILD = UInt8('b')
const OP_BINPERSID = UInt8('Q')
const OP_STOP = UInt8('.')

mutable struct Unpickler{V <: AbstractVector{UInt8}}
    data::V
    pos::Int                    # index of the next unread byte
    nodes::Vector{PickleNode}
    stack::Vector{Int32}        # node indices
    marks::Vector{Int}          # stack lengths at each MARK
    memo::Dict{Int, Int32}
end

function take_byte!(u::Unpickler)
    u.pos <= lastindex(u.data) || error("unpickle: unexpected end of data")
    b = u.data[u.pos]
    u.pos += 1
    return b
end

# Unsigned little-endian integer of type T.
function take_le!(u::Unpickler, ::Type{T}) where {T <: Unsigned}
    u.pos + sizeof(T) - 1 <= lastindex(u.data) || error("unpickle: unexpected end of data")
    x = zero(T)
    for k in 0:(sizeof(T) - 1)
        x |= T(u.data[u.pos + k]) << (8k)
    end
    u.pos += sizeof(T)
    return x
end

function take_string!(u::Unpickler, n::Integer)
    n >= 0 && u.pos + n - 1 <= lastindex(u.data) || error("unpickle: unexpected end of data")
    s = String(u.data[u.pos:(u.pos + n - 1)])
    u.pos += n
    return s
end

# Text up to the next newline (consumed), as written by GLOBAL.
function take_line!(u::Unpickler)
    start = u.pos
    while take_byte!(u) != UInt8('\n')
    end
    return String(u.data[start:(u.pos - 2)])
end

function push_node!(u::Unpickler, node::PickleNode)
    push!(u.nodes, node)
    idx = Int32(length(u.nodes))
    push!(u.stack, idx)
    return idx
end

function pop_node!(u::Unpickler, at::Int)
    isempty(u.stack) && error(lazy"unpickle: stack underflow at byte $at")
    # A MARK must stay below every value an opcode consumes.
    (isempty(u.marks) || length(u.stack) > u.marks[end]) || error(lazy"unpickle: stack underflow at byte $at")
    return pop!(u.stack)
end

function pop_mark!(u::Unpickler, at::Int)
    isempty(u.marks) && error(lazy"unpickle: missing MARK at byte $at")
    m = pop!(u.marks)
    items = u.stack[(m + 1):end]
    resize!(u.stack, m)
    return items
end

function pop_n!(u::Unpickler, n::Int, at::Int)
    items = Vector{Int32}(undef, n)
    for k in n:-1:1
        items[k] = pop_node!(u, at)
    end
    return items
end

function opcode_name(op::UInt8)
    op == OP_SETITEM && return "SETITEM"
    op == OP_SETITEMS && return "SETITEMS"
    op == OP_APPEND && return "APPEND"
    op == OP_APPENDS && return "APPENDS"
    return "BUILD"
end

# Confirms `u.nodes[idx]` has kind `kind` for opcode `op` at byte `at`.
function require_kind(u::Unpickler, idx::Int32, kind::PickleKind, what::String, op::UInt8, at::Int)
    u.nodes[idx].kind == kind || error(lazy"unpickle: $(opcode_name(op)) at byte $at requires a $what target")
    return nothing
end

function lookup_memo(u::Unpickler, key::Integer, op::UInt8, at::Int)
    haskey(u.memo, Int(key)) || error(lazy"unpickle: BINGET of unset memo index $key at byte $at")
    push!(u.stack, u.memo[Int(key)])
    return nothing
end

function reduce_call!(u::Unpickler, at::Int)
    args = pop_node!(u, at)
    callable = pop_node!(u, at)
    cn = u.nodes[callable]
    an = u.nodes[args]
    cn.kind == PK_GLOBAL || error(lazy"unpickle: REDUCE/NEWOBJ at byte $at needs a global callable")
    an.kind == PK_TUPLE || error(lazy"unpickle: REDUCE/NEWOBJ at byte $at needs a tuple of arguments")
    if cn.str == "collections.OrderedDict"
        isempty(an.items) || error(lazy"unpickle: collections.OrderedDict called with arguments at byte $at")
        push_node!(u, PickleNode(PK_DICT))
    elseif cn.str == "_codecs.encode"
        (
            length(an.items) == 2 && u.nodes[an.items[1]].kind == PK_STR &&
                u.nodes[an.items[2]].kind == PK_STR && u.nodes[an.items[2]].str == "latin1"
        ) ||
            error(lazy"unpickle: _codecs.encode at byte $at must be called as (str, \"latin1\")")
        bytes = UInt8[]
        for c in u.nodes[an.items[1]].str
            UInt32(c) < 0x0100 || error(lazy"unpickle: _codecs.encode string at byte $at holds a code point above 255")
            push!(bytes, UInt8(c))
        end
        push_node!(u, PickleNode(PK_BYTES; str = String(bytes)))
    else
        push_node!(u, PickleNode(PK_CALL; items = Int32[callable, args]))
    end
    return nothing
end

"""
    unpickle(bytes::AbstractVector{UInt8}) -> PickleTree

Parse a protocol-2 pickle without executing it. Only the opcodes torch.save emits for
MACE checkpoints and the globals in `ALLOWED_GLOBALS` are accepted; anything else
throws an error naming the offending byte (1-based position of the opcode).
"""
function unpickle(bytes::AbstractVector{UInt8})
    Base.require_one_based_indexing(bytes)
    u = Unpickler{typeof(bytes)}(bytes, firstindex(bytes), PickleNode[], Int32[], Int[], Dict{Int, Int32}())
    while true
        at = u.pos
        op = take_byte!(u)
        if op == OP_PROTO
            v = take_byte!(u)
            v == 0x02 || error(lazy"unpickle: unsupported pickle protocol $v")
        elseif op == OP_GLOBAL
            mod = take_line!(u)
            name = take_line!(u)
            full = string(mod, ".", name)
            full in ALLOWED_GLOBALS || error(lazy"unpickle: global $full is not allowed (byte $at)")
            push_node!(u, PickleNode(PK_GLOBAL; str = full))
        elseif op == OP_BINPUT
            isempty(u.stack) && error(lazy"unpickle: stack underflow at byte $at")
            u.memo[Int(take_byte!(u))] = u.stack[end]
        elseif op == OP_LONG_BINPUT
            isempty(u.stack) && error(lazy"unpickle: stack underflow at byte $at")
            u.memo[Int(take_le!(u, UInt32))] = u.stack[end]
        elseif op == OP_BINGET
            lookup_memo(u, take_byte!(u), op, at)
        elseif op == OP_LONG_BINGET
            lookup_memo(u, take_le!(u, UInt32), op, at)
        elseif op == OP_BININT
            push_node!(u, PickleNode(PK_INT; int = reinterpret(Int32, take_le!(u, UInt32))))
        elseif op == OP_BININT1
            push_node!(u, PickleNode(PK_INT; int = take_byte!(u)))
        elseif op == OP_BININT2
            push_node!(u, PickleNode(PK_INT; int = take_le!(u, UInt16)))
        elseif op == OP_BINFLOAT
            # The only big-endian operand in the format.
            x = UInt64(0)
            for _ in 1:8
                x = (x << 8) | UInt64(take_byte!(u))
            end
            push_node!(u, PickleNode(PK_FLOAT; float = reinterpret(Float64, x)))
        elseif op == OP_BINUNICODE
            s = take_string!(u, take_le!(u, UInt32))
            isvalid(s) || error(lazy"unpickle: string at byte $at is not valid UTF-8")
            push_node!(u, PickleNode(PK_STR; str = s))
        elseif op == OP_MARK
            push!(u.marks, length(u.stack))
        elseif op == OP_EMPTY_TUPLE
            push_node!(u, PickleNode(PK_TUPLE))
        elseif op == OP_TUPLE
            push_node!(u, PickleNode(PK_TUPLE; items = pop_mark!(u, at)))
        elseif op == OP_TUPLE1 || op == OP_TUPLE2 || op == OP_TUPLE3
            push_node!(u, PickleNode(PK_TUPLE; items = pop_n!(u, Int(op - OP_TUPLE1) + 1, at)))
        elseif op == OP_EMPTY_LIST
            push_node!(u, PickleNode(PK_LIST))
        elseif op == OP_EMPTY_DICT
            push_node!(u, PickleNode(PK_DICT))
        elseif op == OP_APPEND
            v = pop_node!(u, at)
            target = pop_node!(u, at)
            require_kind(u, target, PK_LIST, "list", op, at)
            push!(u.nodes[target].items, v)
            push!(u.stack, target)
        elseif op == OP_APPENDS
            vs = pop_mark!(u, at)
            target = pop_node!(u, at)
            require_kind(u, target, PK_LIST, "list", op, at)
            append!(u.nodes[target].items, vs)
            push!(u.stack, target)
        elseif op == OP_SETITEM
            v = pop_node!(u, at)
            k = pop_node!(u, at)
            target = pop_node!(u, at)
            require_kind(u, target, PK_DICT, "dict", op, at)
            push!(u.nodes[target].items, k, v)
            push!(u.stack, target)
        elseif op == OP_SETITEMS
            kvs = pop_mark!(u, at)
            iseven(length(kvs)) || error(lazy"unpickle: SETITEMS at byte $at has an odd number of items")
            target = pop_node!(u, at)
            require_kind(u, target, PK_DICT, "dict", op, at)
            append!(u.nodes[target].items, kvs)
            push!(u.stack, target)
        elseif op == OP_NONE
            push_node!(u, PickleNode(PK_NONE))
        elseif op == OP_NEWTRUE
            push_node!(u, PickleNode(PK_BOOL; int = 1))
        elseif op == OP_NEWFALSE
            push_node!(u, PickleNode(PK_BOOL; int = 0))
        elseif op == OP_REDUCE || op == OP_NEWOBJ
            reduce_call!(u, at)
        elseif op == OP_BUILD
            state = pop_node!(u, at)
            target = pop_node!(u, at)
            require_kind(u, target, PK_CALL, "call", op, at)
            length(u.nodes[target].items) == 2 || error(lazy"unpickle: second BUILD at byte $at on one object")
            push!(u.nodes[target].items, state)
            push!(u.stack, target)
        elseif op == OP_BINPERSID
            pid = pop_node!(u, at)
            push_node!(u, PickleNode(PK_PERSID; items = Int32[pid]))
        elseif op == OP_STOP
            root = pop_node!(u, at)
            isempty(u.stack) || error(lazy"unpickle: stack not empty after STOP at byte $at")
            u.pos > lastindex(u.data) || error(lazy"unpickle: trailing data after STOP at byte $(u.pos)")
            return PickleTree(u.nodes, root)
        else
            error(lazy"unpickle: unsupported opcode 0x$(string(op; base = 16, pad = 2)) at byte $at")
        end
    end
    return
end
