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
- `state`, the tensors stored in the file, keyed by their `state_dict()` names. It is a
  `StateDict` with three fields: `f64` and `i64` map a name to a
  `CheckpointTensor{Float64}` or `CheckpointTensor{Int64}`, and `names` lists every
  name in torch's `state_dict()` order. `torch_array(x, Val(N))` gives a tensor as an
  `N`-dimensional `Array` indexed one-based, with the same layout as the torch tensor:

  ```julia
  z = torch_array(ck.state.i64["atomic_numbers"], Val(1))
  ```

  Buffers that e3nn rebuilds when the model is constructed (the Wigner-3j buffers of the
  compiled tensor products) are not stored in the file and so are not in `state`: for
  MACE-MP-0b3, 89 tensors are stored against 93 in torch's in-memory `state_dict()`.

Supported files are `torch.save`d `ScaleShiftMACE` models, such as the MACE-MP-0b3 and
MACE-MPA-0 releases, whose pickles use only an allowlisted set of classes. Anything else
(other model classes, unlisted pickle globals) raises an error naming the offending item
rather than loading partially.
