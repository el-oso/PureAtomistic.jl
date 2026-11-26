"""
    QuantumEngineYAMLExt

YAML input for QuantumEngine: the method of `QuantumEngine.Input.parse_yaml`, which reads YAML text
with YAML.jl into a dictionary and calls `parse_dict`. It loads with YAML.jl.
"""
module QuantumEngineYAMLExt

using QuantumEngine: InputError, Input
using QuantumEngine.Input: Node, parse_dict, parse_structured
using YAML: YAML

# A YAML mapping as written: its keys and values in the order of the file, a repeated key included,
# which `parse_dict` then rejects; a `Dict` would keep only the last value of a repeated key, in hash
# order. YAML.jl builds each mapping with `_YAMLMap()` and `setindex!` (and, before storing a key,
# `haskey`, logging an error when it is already present), and `parse_dict` iterates it, so it defines
# no other method of the dictionary interface.
struct _YAMLMap <: AbstractDict{Any, Any}
    keys::Vector{Any}
    values::Vector{Any}
end

_YAMLMap() = _YAMLMap(Any[], Any[])

Base.iterate(m::_YAMLMap, j::Int = firstindex(m.keys)) =
    j > lastindex(m.keys) ? nothing : (m.keys[j] => m.values[j], j + 1)
Base.haskey(m::_YAMLMap, k) = any(isequal(k), m.keys)

function Base.setindex!(m::_YAMLMap, v, k)
    push!(m.keys, k)
    push!(m.values, v)
    return m
end

# Whether `text` reads as `.inp` syntax: its first line that is neither blank nor a `#` comment is a
# `keyword value` line, a word that starts with a letter and does not end in `:`, then a value.
function _looks_like_inp(text::AbstractString)
    for l in eachline(IOBuffer(text))
        s = lstrip(l)
        (isempty(s) || startswith(s, '#')) && continue
        return occursin(r"^[A-Za-z][A-Za-z0-9_-]*[ \t]+[^ \t]", s)
    end
    return false
end

_inp_hint(text::AbstractString) = _looks_like_inp(text) ? "; this file looks like .inp syntax" : ""

function Input.parse_yaml(text::AbstractString; source::AbstractString = "<string>", lenient::Bool = false)
    doc = try
        YAML.load(text; dicttype = _YAMLMap)
    catch e
        # A parser error, in either mode; YAML.jl's other errors propagate unchanged.
        e isa YAML.ParserError || rethrow()
        throw(InputError(source, 0, "not valid YAML ($(sprint(showerror, e)))$(_inp_hint(text))"))
    end
    doc isa AbstractDict && return parse_dict(doc; source, lenient)
    doc === nothing && return parse_structured(Node(nothing); source, lenient)
    # The oracle reads a document that is not a mapping as one without any key, so it has no atoms.
    lenient && return parse_dict(_YAMLMap(); source, lenient)
    throw(InputError(source, 0, "top-level YAML document is not a mapping$(_inp_hint(text))"))
end

end
