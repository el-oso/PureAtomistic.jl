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

function Input.parse_yaml(text::AbstractString; source::AbstractString = "<string>", lenient::Bool = false)
    doc = YAML.load(text; dicttype = _YAMLMap)
    doc isa AbstractDict && return parse_dict(doc; source, lenient)
    doc === nothing && return parse_structured(Node(nothing); source, lenient)
    # The oracle reads a document that is not a mapping as one without any key, so it has no atoms.
    lenient && return parse_dict(_YAMLMap(); source, lenient)
    doc isa AbstractString &&
        throw(InputError(source, 0, "top-level YAML document is not a mapping; this file looks like .inp syntax"))
    throw(InputError(source, 0, "top-level YAML document is not a mapping"))
end

end
