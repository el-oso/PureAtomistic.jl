# JSON text to `Node`. `JSON.parse` builds each object as a `_JSONObject`, which keeps its keys and
# values in the order of the text, a repeated key included: JSON.jl's own dictionary would keep
# the last of two equal keys silently, and the oracle's reader rejects them. The tree is then
# converted by the dictionary reader's `_dict_node`, which names the key path in its errors: a
# repeated key, an integer beyond the `Int64` range (JSON.jl reads it as a `BigInt`) and a value of
# a type no JSON text yields. A number beyond the `Float64` range, which JSON.jl reads as a
# `BigFloat`, converts to ±Inf, as Rust's `f64` parser, which the oracle uses, reads it.

struct _JSONObject <: AbstractDict{String, Any}
    keys::Vector{String}
    values::Vector{Any}
end

_JSONObject() = _JSONObject(String[], Any[])

# JSON.jl builds an object with `dicttype()` and `setindex!` alone, and `_dict_map` iterates it, so
# `_JSONObject` defines no other method of the dictionary interface.
function Base.setindex!(d::_JSONObject, @nospecialize(v), k::AbstractString)
    push!(d.keys, String(k))
    push!(d.values, v)
    return d
end

Base.iterate(d::_JSONObject, j::Int = firstindex(d.keys)) =
    j > lastindex(d.keys) ? nothing : (d.keys[j] => d.values[j], j + 1)

"""
    json_node(text::AbstractString; source::AbstractString = "<string>") -> Node

The `Node` tree of the JSON document `text`, read by JSON.jl. A JSON integer is an `IntNode`
and a number with a fraction or an exponent a `FloatNode`; a number beyond the `Float64` range
is ±Inf. Text that is not JSON is an `InputError` of `source`, line 0,
`not valid JSON (<JSON.jl's message>)`. A key repeated within one object and an integer outside
the `Int64` range are `InputError`s naming the key path, as `node` gives them
(`input: <string>: scf: duplicate key "a"`).
"""
function json_node(text::AbstractString; source::AbstractString = "<string>")
    src = String(source)
    x = try
        JSON.parse(text; dicttype = _JSONObject)
    catch e
        e isa ArgumentError || rethrow()
        throw(InputError(src, 0, lazy"not valid JSON ($(chomp(e.msg)))"))
    end
    return _dict_node(x, "", src)
end

"""
    parse_json(text::AbstractString; source = "<string>", lenient = false) -> ParsedInput

The JSON input file `text`, read by `parse_structured(json_node(text; source); source, lenient)`.
"""
parse_json(text::AbstractString; source::AbstractString = "<string>", lenient::Bool = false) =
    parse_structured(json_node(text; source); source, lenient)
