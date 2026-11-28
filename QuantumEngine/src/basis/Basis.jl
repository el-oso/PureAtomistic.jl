"""
    QuantumEngine.Basis

Gaussian basis sets: the shell geometry (the Cartesian component order, the primitive
normalization and the contraction renormalization), the types of the basis data as a file gives
it (`Primitive`, `ShellSpec`, `ECPChannel`, `ECPPotential`, `ElementBasis`), the Gaussian94
`.gbs` reader (`parse_gbs`, `read_element_basis`), the reader of the oracle's ECP JSON format
(`parse_ecp_json`, `read_ecp_json`), the resolution of a basis name to its file
(`normalize_basis_name`, `candidate_stems`, `basis_search_paths`, `locate_basis_file`) and the
molecular basis (`Shell`, `BasisSet`) with its AO ranges and shell pairs.
"""
module Basis

using JSON: JSON
using StaticArrays: SVector
using StrictMode: @strict_exempt, @strict_policy
using ..Errors: BasisError
using ..IO: input_line, is_decimal_int, parse_coordinate
using ..Units: element, ascii_lowercase, symbol_number

include("geometry.jl")
include("types.jl")
include("gbs.jl")
include("ecp_json.jl")
include("names.jl")
include("search.jl")

export ncart, cartesian_component, primitive_norm, contraction_renorm
export Primitive, ShellSpec, ECPChannel, ECPPotential, ElementBasis, Shell, BasisSet
export ao_range, shell_pairs, ShellPairs
export parse_gbs, read_element_basis
export parse_ecp_json, read_ecp_json
export normalize_basis_name, candidate_stems, basis_search_paths, locate_basis_file

# Storing an `ECPPotential` in the field `ecp::Union{Nothing,ECPPotential{T}}` allocates (48 bytes
# per construction in Float64 and Float32, measured); given `nothing`, and for the other type
# constructors, nothing is allocated. The sweep judges no constructor; the item "basis: constructor
# signatures" proves both.
@strict_exempt :noalloc ElementBasis reason = "entry point: element basis constructor, allocates its ECP field"
@strict_exempt :noalloc parse_gbs reason = "entry point: gbs reader"
@strict_exempt :noalloc read_element_basis reason = "entry point: gbs reader"
@strict_exempt :noalloc _gbs_error reason = "entry point: gbs reader"
@strict_exempt :noalloc _unknown_element reason = "entry point: gbs reader"
@strict_exempt :noalloc _gbs_number reason = "entry point: gbs reader"
@strict_exempt :noalloc _gbs_shell reason = "entry point: gbs reader"
@strict_exempt :noalloc _ecp_title reason = "entry point: gbs reader"
@strict_exempt :noalloc _is_title reason = "entry point: gbs reader"
@strict_exempt :noalloc _capitalized reason = "entry point: gbs reader"
@strict_exempt :noalloc _shell_ends reason = "entry point: gbs reader"
@strict_exempt :noalloc _channel_ends reason = "entry point: gbs reader"
@strict_exempt :noalloc _too_few_channels reason = "entry point: gbs reader"
# JSON.jl's `parse` dispatches at run time inside itself and hands `setindex!` values typed `Any`
# (arrays as `Vector{Any}`); `_ecp_value` turns each into an `_ECPValue` once, and the ECP walker
# reads concrete fields only.
@strict_exempt :typestable :noalloc :noboxing types = (Base.IO,) parse_ecp_json reason = "reads JSON.jl's untyped values (JSON.parse returns Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (AbstractString, typeof(parse_ecp_json), Base.IO) parse_ecp_json reason = "reads JSON.jl's untyped values (JSON.parse returns Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (AbstractString,) read_ecp_json reason = "reads JSON.jl's untyped values (JSON.parse returns Any); not on the trimmed path"
@strict_exempt :noalloc :noboxing types = (Any,) _ecp_value reason = "reads JSON.jl's untyped values (JSON.parse returns Any); not on the trimmed path"
@strict_exempt :noalloc :noboxing types = (_ECPObject, Any, AbstractString) setindex! reason = "reads JSON.jl's untyped values (JSON.parse returns Any); not on the trimmed path"
@strict_exempt :noalloc _ECPObject reason = "entry point: ECP JSON reader (called by parse_ecp_json)"
@strict_exempt :noalloc _ECPValue reason = "entry point: ECP JSON reader (called by parse_ecp_json)"
@strict_exempt :noalloc _ecp_scalar reason = "entry point: ECP JSON reader (called by parse_ecp_json)"
@strict_exempt :noalloc _ecp_error reason = "entry point: ECP JSON reader (called by parse_ecp_json)"
@strict_exempt :noalloc _ecp_index reason = "entry point: ECP JSON reader (called by parse_ecp_json)"
@strict_exempt :noalloc _ecp_channel reason = "entry point: ECP JSON reader (called by parse_ecp_json)"
@strict_exempt :noalloc normalize_basis_name reason = "entry point: basis name resolution"
@strict_exempt :noalloc candidate_stems reason = "entry point: basis name resolution"
@strict_exempt :noalloc stem_variants reason = "entry point: basis name resolution (called by candidate_stems)"
@strict_exempt :noalloc :noboxing types = () basis_search_paths reason = "entry point: basis search path list; Base's pkgdir indexes a tuple at run time (tuple.jl:33), which boxes"
@strict_exempt :noalloc :noboxing types = (AbstractDict, Union{Nothing, AbstractString}, typeof(basis_search_paths)) basis_search_paths reason = "entry point: basis search path list; Base's pkgdir indexes a tuple at run time (tuple.jl:33), which boxes"
@strict_exempt :noalloc :noboxing types = (AbstractString,) locate_basis_file reason = "entry point: basis file lookup; Base's abspath (through normpath) and the default basis_search_paths() index a tuple at run time (tuple.jl:33), which boxes"
@strict_exempt :noalloc :noboxing types = (AbstractVector{<:AbstractString}, typeof(locate_basis_file), AbstractString) locate_basis_file reason = "entry point: basis file lookup; Base's abspath (through normpath) indexes a tuple at run time (tuple.jl:33), which boxes"
@strict_exempt :noalloc _comma_list reason = "entry point: basis file lookup (called by locate_basis_file)"

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
