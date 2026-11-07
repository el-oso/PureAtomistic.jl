module PureMLIPGPUExt

using KernelAbstractions: KernelAbstractions, Backend, CPU, allocate, get_backend
using PureMLIP: PureMLIP, MACEModel, NeighborList, neighbor_list
using PureMLIP.Tensors: E3nnLinear, SpeciesLinear, ChannelwiseTP, Contraction, SymmetricContraction
using PureMLIP.Tensors: ContractionMatrices, contraction_matrices!

# The device forms of the PureMLIP.Tensors types live in `tensors/`, which mirrors `src/tensors/`.
include("tensors/arrays.jl")
include("tensors/linear.jl")
include("tensors/tensor_product.jl")
include("tensors/symmetric_contraction.jl")
include("device.jl")

end
