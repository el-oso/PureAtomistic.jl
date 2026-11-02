# Tensor kernels

`PureMLIP.Tensors` is a public submodule that holds the equivariant tensor kernels of the MACE
model: the channel-wise tensor product, e3nn linear maps, species-indexed linear maps and the
symmetric contraction, each with its pullback. `PureMLIP` does not re-export its names; reach
them as `PureMLIP.Tensors.name`, or with `using PureMLIP.Tensors`. The kernels' docstrings state
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
PureMLIP.Tensors.symmetric_contraction!
PureMLIP.Tensors.symmetric_contraction_pullback!
PureMLIP.Tensors.symmetric_contraction_scratch_size
PureMLIP.Tensors.checked_eachindex
PureMLIP.Tensors.n_outputs
```
