# PureAtomistic.jl

A monorepo for pure-Julia atomistic simulation packages.

- `PureMLIP/` holds machine-learned interatomic potentials, starting with a
  reader for mace-torch `.model` checkpoints.
- `PureAtomistic/` is the simulation layer built on PureMLIP: molecular dynamics with thermostats and
  barostats, structure relaxation, trajectory and structure I/O, and analysis.

Goals for every package here:

- No binary artifacts: no `_jll` dependencies, only Julia code.
- Compatible with `juliac --trim`, so programs can be compiled to small
  standalone executables or shared libraries.

Licensed under the MIT license.

## MACE checkpoint reader

`PureMLIP.load_mace_checkpoint` reads mace-torch `.model` files (`ScaleShiftMACE`
checkpoints such as MACE-MP-0b3) into hyperparameters and plain Julia arrays, using only
Julia code. `PureMLIP/juliac/build.jl` compiles the example program `mace_inspect` with
`juliac --trim=safe`; running `PureMLIP/juliac/build/bin/mace_inspect model.model` prints
the model class, `r_max`, number of interactions, number of state tensors and number of
trainable parameters.
