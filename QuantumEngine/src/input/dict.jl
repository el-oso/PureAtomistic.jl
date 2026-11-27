# Any `AbstractDict` to `Node`, as YAML.jl, JSON.jl and other readers build one. The values of such
# a dictionary are untyped, so `_dict_node` tells their kinds apart with `isa`; the reader decides
# the scalar types, and the schema walker's type checks judge them.

"""
    node(d::AbstractDict; source::AbstractString = "<dict>") -> Node

The `Node` tree of the dictionary `d`, converted recursively: an `AbstractDict` is a map, in the
dictionary's iteration order, with its keys converted with `String`; an `AbstractVector` is a list;
a `Bool`, an `Integer` in the `Int` range, an `AbstractFloat` (converted to `Float64`) and an
`AbstractString` are scalars; `nothing` and `missing` are null. A key that is not an
`AbstractString`, a key repeated within one map, an integer outside the `Int` range and a value of
any other type are an `InputError` of `source`, line 0, naming the key path
(`input: <dict>: geometry[1].position[2]: unsupported value of type Char`).
"""
node(d::AbstractDict; source::AbstractString = "<dict>") = _dict_map(d, "", String(source))

_path_prefix(path::String) = isempty(path) ? "" : path * ": "

function _dict_node(@nospecialize(x), path::String, source::String)
    x isa Nothing && return Node(nothing)
    x isa Missing && return Node(nothing)
    x isa Bool && return Node(x)
    x isa Int && return Node(x)
    if x isa Integer
        typemin(Int) <= x <= typemax(Int) ||
            throw(InputError(source, 0, lazy"$(_path_prefix(path))the integer $(string(x)) is outside the Int64 range"))
        return Node(Int(x))
    end
    x isa AbstractFloat && return Node(Float64(x))
    x isa AbstractString && return Node(String(x))
    x isa AbstractDict && return _dict_map(x, path, source)
    if x isa AbstractVector
        items = Node[]
        for (k, y) in enumerate(x)
            push!(items, _dict_node(y, _list_path(path, k), source))
        end
        return Node(items)
    end
    throw(InputError(source, 0, lazy"$(_path_prefix(path))unsupported value of type $(string(typeof(x)))"))
end

function _dict_map(@nospecialize(d::AbstractDict), path::String, source::String)
    keys = String[]
    items = Node[]
    for (k, v) in d
        k isa AbstractString ||
            throw(InputError(source, 0, lazy"$(_path_prefix(path))the key $(repr(k)) of type $(string(typeof(k))) is not a string"))
        key = String(k)
        key in keys && throw(InputError(source, 0, lazy"$(_path_prefix(path))duplicate key $(repr(key))"))
        push!(keys, key)
        push!(items, _dict_node(v, _child(path, key), source))
    end
    return Node(keys, items)
end

"""
    parse_dict(d::AbstractDict; source::AbstractString, lenient::Bool = false) -> ParsedInput

The structured input `d`, read by `parse_structured(node(d; source); source, lenient)`.
"""
parse_dict(d::AbstractDict; source::AbstractString, lenient::Bool = false) =
    parse_structured(node(d; source); source, lenient)

"""
    parse_yaml(text::AbstractString; source::AbstractString = "<string>", lenient::Bool = false) -> ParsedInput

The YAML input file `text`, read by YAML.jl into a dictionary and then by `parse_dict`. The
function has a method only when YAML.jl is loaded (`using YAML`, which loads the extension
`QuantumEngineYAMLExt`).

YAML.jl decides the scalar types, and the Input page of the manual lists where they differ from the
oracle's reader. A key repeated within one mapping is an `InputError` in both modes. A document that
is empty is an error in both modes. A top-level value that is not a mapping is an error naming the
mismatch in strict mode and the error of an input with no atoms in lenient mode. Text that YAML.jl's
parser or scanner rejects is an `InputError` `not valid YAML (<YAML.jl's message>)` in both modes;
YAML.jl's other errors propagate unchanged. That message and the not-a-mapping message end in
`; this file looks like .inp syntax` when the first line that is neither blank nor a `#` comment is
a `keyword value` line. A merge key (`<<`) is an `InputError`
`YAML merge keys (<<) are not supported` in both modes.
"""
function parse_yaml end

# The error of a YAML input read without YAML.jl loaded.
_yaml_unavailable(source::AbstractString) =
    InputError(source, 0, "reading YAML needs `using YAML` (QuantumEngineYAMLExt)")
