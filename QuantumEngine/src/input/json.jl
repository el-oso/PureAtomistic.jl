# JSON text to `Node`. `JSON.parse` builds each object as a `_JSONObject`, whose `setindex!` turns
# the value into a `Node` at once and throws on a key already present: JSON.jl keeps the last of
# two equal keys silently, and the oracle's reader rejects them. Arrays arrive as JSON.jl's
# `Vector{Any}` and scalars as `Int64`, `Float64`, `String`, `Bool` or `nothing` (and as `BigInt`
# or `BigFloat` beyond the 64-bit ranges), each converted by `_json_node`.

struct _JSONObject <: AbstractDict{String, Any}
    keys::Vector{String}
    items::Vector{Node}
end

_JSONObject() = _JSONObject(String[], Node[])

# JSON.jl builds an object with `dicttype()` and `setindex!` alone, so `_JSONObject` defines no other
# method of the dictionary interface. JSON.jl's values reach `setindex!` and the arrays as `Any`;
# `_json_node` tells their types apart with `isa`, so no call on them dispatches at run time.
function Base.setindex!(d::_JSONObject, @nospecialize(v), k::AbstractString)
    key = String(k)
    key in d.keys && throw(ArgumentError(lazy"input: duplicate key $(repr(key))"))
    push!(d.keys, key)
    push!(d.items, _json_node(v))
    return d
end

# A JSON.jl value as a `Node`. JSON.jl reads an integer beyond the `Int64` range as a `BigInt` and
# a number beyond the `Float64` range as a `BigFloat`; the integer is an error, and the number is
# ±Inf, as Rust's `f64` parser, which the oracle uses, reads it.
function _json_node(@nospecialize(x))
    x isa Nothing && return Node(nothing)
    x isa Bool && return Node(x)
    x isa Int && return Node(x)
    x isa Float64 && return Node(x)
    x isa String && return Node(x)
    x isa _JSONObject && return Node(x.keys, x.items)
    if x isa Vector{Any}
        items = Node[]
        for y in x
            push!(items, _json_node(y))
        end
        return Node(items)
    end
    x isa BigInt && throw(ArgumentError(lazy"input: the integer $(string(x)) is outside the Int64 range"))
    x isa BigFloat && return Node(x > 0 ? Inf : -Inf)
    throw(ArgumentError(lazy"input: JSON.jl returned a value of type $(string(typeof(x)))"))
end

"""
    json_node(text::AbstractString) -> Node

The `Node` tree of the JSON document `text`, read by JSON.jl. A JSON integer is an `IntNode`
and a number with a fraction or an exponent a `FloatNode`; a number beyond the `Float64` range
is ±Inf. A repeated key in any object is an `ArgumentError`, `input: duplicate key "a"`, as is
an integer outside the `Int64` range; text that is not JSON is JSON.jl's error.
"""
json_node(text::AbstractString) = _json_node(JSON.parse(text; dicttype = _JSONObject))

"""
    parse_json(text::AbstractString; source = "<string>", lenient = false) -> ParsedInput

The JSON input file `text`, read by `parse_structured(json_node(text); source, lenient)`.
"""
parse_json(text::AbstractString; source::AbstractString = "<string>", lenient::Bool = false) =
    parse_structured(json_node(text); source, lenient)
