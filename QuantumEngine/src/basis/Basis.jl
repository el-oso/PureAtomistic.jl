"""
    QuantumEngine.Basis

Gaussian basis sets: the shell geometry (the Cartesian component order, the primitive
normalization and the contraction renormalization), the types of the basis data as a file gives
it (`Primitive`, `ShellSpec`, `ECPChannel`, `ECPPotential`, `ElementBasis`) and the molecular
basis (`Shell`, `BasisSet`) with its AO ranges and shell pairs.
"""
module Basis

using StaticArrays: SVector
using StrictMode: @strict_exempt, @strict_policy

include("geometry.jl")
include("types.jl")

export ncart, cartesian_component, primitive_norm, contraction_renorm
export Primitive, ShellSpec, ECPChannel, ECPPotential, ElementBasis, Shell, BasisSet
export ao_range, shell_pairs, ShellPairs

# Storing an `ECPPotential` in the field `ecp::Union{Nothing,ECPPotential{T}}` allocates (48 bytes
# per construction in Float64 and Float32, measured); given `nothing`, and for the other type
# constructors, nothing is allocated. The sweep judges no constructor; the item "basis: constructor
# signatures" proves both.
@strict_exempt :noalloc ElementBasis reason = "entry point: element basis constructor, allocates its ECP field"

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
