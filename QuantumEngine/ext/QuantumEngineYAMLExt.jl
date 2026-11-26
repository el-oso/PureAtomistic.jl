"""
    QuantumEngineYAMLExt

YAML input for QuantumEngine: the method of `QuantumEngine.Input.parse_yaml`, which reads YAML text
with YAML.jl into a dictionary and calls `parse_dict`. It loads with YAML.jl.
"""
module QuantumEngineYAMLExt

using QuantumEngine: InputError, Input
using QuantumEngine.Input: Node, parse_dict, parse_structured, parse_yaml
using StrictMode: @strict_exempt, @strict_policy
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

# A mapping with a merge key (`<<`), which `_yaml_map` rejects.
struct _MergeKey <: Exception end

# YAML.jl's constructor of every mapping. YAML.jl merges the mapping of a merge key into the mapping
# that holds it and lets an explicit key override a merged one; the oracle's reader (yaml-rust2)
# has no merge keys and reads `<<` as an ordinary key. A merge key is therefore rejected before
# YAML.jl merges it: the key nodes still carry YAML.jl's merge tag here.
function _yaml_map(constructor, node)
    for (key, _) in node.value
        key.tag == "tag:yaml.org,2002:merge" && throw(_MergeKey())
    end
    return YAML.construct_mapping(_YAMLMap, constructor, node)
end

# Whether `text` reads as `.inp` syntax: its first line that is neither blank nor a `#` comment is a
# `keyword value` line, a word that starts with a letter and does not end in `:`, then a value. A
# leading UTF-8 byte order mark is skipped, as YAML.jl skips it.
function _looks_like_inp(text::AbstractString)
    for l in eachline(IOBuffer(chopprefix(text, "﻿")))
        s = lstrip(l)
        (isempty(s) || startswith(s, '#')) && continue
        return occursin(r"^[A-Za-z][A-Za-z0-9_-]*[ \t]+[^ \t]", s)
    end
    return false
end

_inp_hint(text::AbstractString) = _looks_like_inp(text) ? "; this file looks like .inp syntax" : ""

function Input.parse_yaml(text::AbstractString; source::AbstractString = "<string>", lenient::Bool = false)
    doc = try
        YAML.load(text, Dict{String, Function}("tag:yaml.org,2002:map" => _yaml_map))
    catch e
        # A merge key, a parser error or a scanner error, in either mode; YAML.jl's other errors
        # propagate unchanged.
        e isa _MergeKey && throw(InputError(source, 0, "YAML merge keys (<<) are not supported"))
        e isa YAML.ParserError || e isa YAML.ScannerError || rethrow()
        throw(InputError(source, 0, "not valid YAML ($(sprint(showerror, e)))$(_inp_hint(text))"))
    end
    doc isa AbstractDict && return parse_dict(doc; source, lenient)
    doc === nothing && return parse_structured(Node(nothing); source, lenient)
    # The oracle reads a document that is not a mapping as one without any key, so it has no atoms.
    lenient && return parse_dict(_YAMLMap(); source, lenient)
    throw(InputError(source, 0, "top-level YAML document is not a mapping$(_inp_hint(text))"))
end

@strict_exempt :typestable :noalloc :noboxing types = (AbstractString,) parse_yaml reason = "reads YAML.jl's untyped values (YAML.load returns Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (AbstractString, Bool, typeof(parse_yaml), AbstractString) parse_yaml reason = "reads YAML.jl's untyped values (YAML.load returns Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (Any, Any) _yaml_map reason = "reads YAML.jl's untyped values (YAML.load returns Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (_YAMLMap,) iterate reason = "reads YAML.jl's untyped values (YAML.load returns Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (_YAMLMap, Int) iterate reason = "reads YAML.jl's untyped values (YAML.load returns Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (_YAMLMap, Any) haskey reason = "reads YAML.jl's untyped values (YAML.load returns Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (_YAMLMap, Any, Any) setindex! reason = "reads YAML.jl's untyped values (YAML.load returns Any); not on the trimmed path"
@strict_exempt :noalloc _YAMLMap reason = "entry point: YAML mapping constructor"
@strict_exempt :noalloc _looks_like_inp reason = "entry point helper: .inp hint of parse_yaml"
@strict_exempt :noalloc _inp_hint reason = "entry point helper: .inp hint of parse_yaml"

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
