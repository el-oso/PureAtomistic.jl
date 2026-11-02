# Example gallery

Each page below is generated from a script in `PureMLIP/docs/examples/`, which the
documentation build runs. Every script also runs on its own:

```
julia --project=PureMLIP/docs PureMLIP/docs/examples/vacancy.jl
```

The scripts download the MACE-MPA-0 checkpoint on first use into `PUREMLIP_CHECKPOINT_DIR`
(default: a scratch directory of the first Julia depot) and check its SHA-256. Their packages
(Literate, GLMakie, JSON) belong to the documentation environment, not to PureMLIP.

| Page | Shows |
|:-----|:------|
| [Loading a model](examples/load_model.md) | MACE-MPA-0's cutoff, its radial features, and the energy of a Cu dimer as the atoms separate |
| [Energy, forces and stress](examples/energy_forces.md) | a 32-atom Cu cell rotated through a full turn: constant energy, forces and stress rotating with it |
| [A vacancy in Cu](examples/vacancy.md) | the vacancy formation energy, and a neighbor moved in a straight line into the vacancy |
| [Lattice constant of Cu](examples/lattice_constant.md) | energy against lattice constant and the minimum from a parabola fit |
| [Energies of a large cell](examples/energy_workspace.md) | a 4,000-atom cell evaluated with `MACEEnergyWorkspace`, and the memory per atom of both workspaces |
| [Speed with PureBLAS](examples/pureblas.md) | speed-up with PureBLAS active, read from the committed benchmark records |
| [Spherical harmonics and tensor kernels](examples/tensors.md) | real spherical harmonics up to `l = 3`, and the einsum of each `PureMLIP.Tensors` kernel |
