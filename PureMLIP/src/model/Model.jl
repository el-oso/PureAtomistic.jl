module Model

using ..Checkpoint, ..Equivariant, LinearAlgebra

# `eachindex(A, B...)` with the axes check written out. Base throws a mismatch error whose
# message builder is a dynamic call, which `juliac --trim` rejects once a caller is precompiled.
@noinline _axes_mismatch(A, B) = throw(DimensionMismatch(lazy"model: arrays must share axes, got $(axes(A)) and $(axes(B))"))

@inline function _eachindex(A::AbstractArray, B::AbstractArray)
    axes(A) == axes(B) || _axes_mismatch(A, B)
    return eachindex(IndexStyle(A, B), A)
end

@inline function _eachindex(A::AbstractArray, B::AbstractArray, C::AbstractArray)
    axes(A) == axes(B) || _axes_mismatch(A, B)
    axes(A) == axes(C) || _axes_mismatch(A, C)
    return eachindex(IndexStyle(A, B, C), A)
end

include("neighbors.jl")
include("radial.jl")
include("linear.jl")
include("tensor_product.jl")
include("radial_mlp.jl")
include("symmetric_contraction.jl")
include("mace_model.jl")
include("evaluate.jl")

export NeighborList, neighbor_list, neighbor_list!
export MACEModel, MACEResult, MACEWorkspace, mace_energy!, mace_energy_forces!, mace_energy_forces

end
