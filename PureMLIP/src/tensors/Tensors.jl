"""
    PureMLIP.Tensors

Equivariant tensor kernels on node and edge features: the channel-wise tensor product with
sparse e3nn coupling tensors, e3nn `Linear` and species-indexed linear maps, and the MACE
symmetric contraction, each with its vector-Jacobian product. It depends only on
`PureMLIP.Equivariant` (for the coupling tensors) and `LinearAlgebra`, and holds no code that
reads checkpoints, so it can be used on its own.
"""
module Tensors

using ..Equivariant, LinearAlgebra

export ChannelwiseTP, tp_edge_add!, tp_edge_pullback_add!
export E3nnLinear, linear!, linear_pullback!, SpeciesLinear, species_linear!, species_linear_pullback!
export Contraction, SymmetricContraction, symmetric_contraction!, symmetric_contraction_pullback!
export symmetric_contraction_scratch_size

public checked_eachindex, n_outputs

include("eachindex.jl")
include("linear.jl")
include("tensor_product.jl")
include("symmetric_contraction.jl")

end
