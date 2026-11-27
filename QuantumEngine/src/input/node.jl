"""
    NodeKind

The kind of a structured-input value: `NullNode`, `BoolNode`, `IntNode`, `FloatNode`,
`StringNode`, `ArrayNode` or `MapNode`.
"""
@enum NodeKind NullNode BoolNode IntNode FloatNode StringNode ArrayNode MapNode

@doc "`NullNode::NodeKind`: a null value." NullNode
@doc "`BoolNode::NodeKind`: a `Bool`, in the field `b`." BoolNode
@doc "`IntNode::NodeKind`: an `Int`, in the field `i`." IntNode
@doc "`FloatNode::NodeKind`: a `Float64`, in the field `f`." FloatNode
@doc "`StringNode::NodeKind`: a `String`, in the field `s`." StringNode
@doc "`ArrayNode::NodeKind`: a list, in the field `items`." ArrayNode
@doc "`MapNode::NodeKind`: a map, with its keys in `keys` and its values in `items`." MapNode

"""
    Node(nothing)
    Node(b::Bool)
    Node(i::Int)
    Node(f::Float64)
    Node(s::String)
    Node(items::Vector{Node})
    Node(keys::Vector{String}, items::Vector{Node})

One value of a structured input (JSON, or a YAML file or any `AbstractDict` once converted), held
in one concrete tree type. `kind` says which field holds the value: `b`, `i`, `f` or `s` for a
scalar, `items` for a list, and `keys` with `items` for a map, which holds `keys[k] => items[k]`
in the order given. The fields a kind does not use hold `false`, `0`, `0.0`, `""` and empty
vectors. A map with a repeated key is an `ArgumentError`, `input: duplicate key "a"`.
"""
struct Node
    kind::NodeKind
    b::Bool
    i::Int
    f::Float64
    s::String
    items::Vector{Node}
    keys::Vector{String}
    Node(kind::NodeKind, b::Bool, i::Int, f::Float64, s::String, items::Vector{Node}, keys::Vector{String}) =
        new(kind, b, i, f, s, items, keys)
end

Node(::Nothing) = Node(NullNode, false, 0, 0.0, "", Node[], String[])
Node(b::Bool) = Node(BoolNode, b, 0, 0.0, "", Node[], String[])
Node(i::Int) = Node(IntNode, false, i, 0.0, "", Node[], String[])
Node(f::Float64) = Node(FloatNode, false, 0, f, "", Node[], String[])
Node(s::String) = Node(StringNode, false, 0, 0.0, s, Node[], String[])
Node(items::Vector{Node}) = Node(ArrayNode, false, 0, 0.0, "", items, String[])

function Node(keys::Vector{String}, items::Vector{Node})
    length(keys) == length(items) ||
        throw(ArgumentError(lazy"input: a map has $(length(keys)) keys and $(length(items)) values"))
    for j in eachindex(keys), k in firstindex(keys):(j - 1)
        keys[k] == keys[j] && throw(ArgumentError(lazy"input: duplicate key $(repr(keys[j]))"))
    end
    return Node(MapNode, false, 0, 0.0, "", items, keys)
end
