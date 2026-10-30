# PureMLIP.jl

Pure-Julia machine-learned interatomic potentials. The package has no binary
artifacts and is built to be compatible with `juliac --trim`.

## Reading MACE checkpoints

```julia
using PureMLIP
ck = load_mace_checkpoint("mace-mp-0b3-medium.model")
```

`load_mace_checkpoint` has two methods: one takes the file contents as a
`Vector{UInt8}`, the other a path. Both return a `MACECheckpoint` with two fields:

- `hyperparameters`, a `MACEHyperparameters` (model class, `r_max`, number of
  interactions, atomic numbers, heads, per-interaction irreps and radial MLP
  sizes, correlation orders, readouts);
- `state`, every parameter and persistent buffer of the module tree, keyed by
  its `state_dict()` name. Each entry is a `CheckpointTensor`; `torch_array(x, Val(N))`
  gives it as an `N`-dimensional `Array` indexed one-based, with the same layout as
  the torch tensor.

Supported files are `torch.save`d `ScaleShiftMACE` models, such as the MACE-MP-0b3 and
MACE-MPA-0 releases, whose pickles use only an allowlisted set of classes. Anything else
(other model classes, unlisted pickle globals) raises an error naming the offending item
rather than loading partially.
