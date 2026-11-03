# Tensor kernels

`PureMLIP.Tensors` is a public submodule that holds the equivariant tensor kernels of the MACE
model: the channel-wise tensor product, e3nn linear maps, species-indexed linear maps and the
symmetric contraction, each with its pullback. `PureMLIP` does not re-export its names; reach
them as `PureMLIP.Tensors.name`, or with `using PureMLIP.Tensors`. The symmetric contraction also
runs as matrix products over groups of atoms of one species, with per-species matrices
(`ContractionMatrices`) and per-group buffers (`ContractionBuffers`). The kernels' docstrings state
what each computes as einsums; the
[spherical harmonics and tensor kernels example](examples/tensors.md) lists them.

## Reference

```@docs
PureMLIP.Tensors
PureMLIP.Tensors.ChannelwiseTP
PureMLIP.Tensors.tp_edge_add!
PureMLIP.Tensors.tp_edge_pullback_add!
PureMLIP.Tensors.E3nnLinear
PureMLIP.Tensors.linear!
PureMLIP.Tensors.linear_pullback!
PureMLIP.Tensors.SpeciesLinear
PureMLIP.Tensors.species_linear!
PureMLIP.Tensors.species_linear_pullback!
PureMLIP.Tensors.Contraction
PureMLIP.Tensors.SymmetricContraction
PureMLIP.Tensors.symmetric_contraction!(::AbstractArray{S, 3}, ::PureMLIP.Tensors.SymmetricContraction, ::AbstractArray{S, 3}, ::AbstractVector{Int}, ::AbstractArray{S}) where {S}
PureMLIP.Tensors.symmetric_contraction_pullback!(::AbstractArray{S, 3}, ::PureMLIP.Tensors.SymmetricContraction, ::AbstractArray{S, 3}, ::AbstractArray{S, 3}, ::AbstractVector{Int}, ::AbstractArray{S}) where {S}
PureMLIP.Tensors.symmetric_contraction_scratch_size
PureMLIP.Tensors.ContractionMatrices
PureMLIP.Tensors.contraction_matrices!
PureMLIP.Tensors.ContractionBuffers
PureMLIP.Tensors.symmetric_contraction!(::AbstractArray{S, 3}, ::PureMLIP.Tensors.ContractionMatrices{S}, ::AbstractArray{S, 3}, ::AbstractVector{Int}, ::PureMLIP.Tensors.ContractionBuffers{S}) where {S}
PureMLIP.Tensors.symmetric_contraction_pullback!(::AbstractArray{S, 3}, ::PureMLIP.Tensors.ContractionMatrices{S}, ::AbstractArray{S, 3}, ::AbstractArray{S, 3}, ::AbstractVector{Int}, ::PureMLIP.Tensors.ContractionBuffers{S}) where {S}
PureMLIP.Tensors.checked_eachindex
PureMLIP.Tensors.n_outputs
```
