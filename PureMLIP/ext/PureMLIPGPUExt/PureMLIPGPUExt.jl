module PureMLIPGPUExt

using KernelAbstractions: KernelAbstractions, Backend, CPU, allocate, get_backend, @kernel, @index, @Const, @localmem, @synchronize
using LinearAlgebra: mul!
using PureMLIP: PureMLIP, MACEModel, NeighborList, neighbor_list
using PureMLIP.Tensors: E3nnLinear, SpeciesLinear, ChannelwiseTP, Contraction, SymmetricContraction
using PureMLIP.Tensors: ContractionMatrices, contraction_matrices!

# The device forms of the PureMLIP.Tensors types and their kernels live in `tensors/`, which
# mirrors `src/tensors/`.
include("tensors/arrays.jl")
include("tensors/linear.jl")
include("tensors/tensor_product.jl")
include("tensors/symmetric_contraction.jl")
include("tensors/contraction.jl")
include("device.jl")
include("edges.jl")
include("radial.jl")
include("messages.jl")
include("readout.jl")
include("reduce.jl")
include("evaluate.jl")
include("backward.jl")

end
