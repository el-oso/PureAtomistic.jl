"""
    QuantumEngine.Basis

Gaussian basis sets: the shell geometry (the Cartesian component order, the primitive
normalization and the contraction renormalization), the types of the basis data as a file gives
it (`Primitive`, `ShellSpec`, `ECPChannel`, `ECPPotential`, `ElementBasis`), the Gaussian94
`.gbs` reader (`parse_gbs`, `read_element_basis`) and the molecular basis (`Shell`, `BasisSet`)
with its AO ranges and shell pairs.
"""
module Basis

using StaticArrays: SVector
using StrictMode: @strict_exempt, @strict_policy
using ..Errors: BasisError
using ..IO: input_line, is_decimal_int, parse_coordinate
using ..Units: element, ascii_lowercase, symbol_number

include("geometry.jl")
include("types.jl")
include("gbs.jl")

export ncart, cartesian_component, primitive_norm, contraction_renorm
export Primitive, ShellSpec, ECPChannel, ECPPotential, ElementBasis, Shell, BasisSet
export ao_range, shell_pairs, ShellPairs
export parse_gbs, read_element_basis

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

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
