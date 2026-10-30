# PureAtomistic.jl

A monorepo for pure-Julia atomistic simulation packages.

- `PureMLIP/` holds machine-learned interatomic potentials, starting with a
  reader for mace-torch `.model` checkpoints.
- `PureAtomistic` (the simulation layer built on PureMLIP) will be added later.

Goals for every package here:

- No binary artifacts: no `_jll` dependencies, only Julia code.
- Compatible with `juliac --trim`, so programs can be compiled to small
  standalone executables or shared libraries.

Licensed under the MIT license.
