# The MACE model

`PureMLIP.Model` evaluates a ScaleShiftMACE on the CPU: energies, forces, virial and stress
of a fully periodic structure.

## Loading a model

```julia
using PureMLIP
ck = load_mace_checkpoint("mace-mp-0b3-medium.model")
model = MACEModel(ck)            # Float64 parameters
model32 = MACEModel{Float32}(ck) # Float32 parameters
```

`MACEModel{T}(ck)` reads the channel, element, basis and radial-MLP sizes from the
checkpoint and checks every other architectural choice against the supported architecture
below. A difference in an architectural choice throws an `ArgumentError` that begins
`model: unsupported checkpoint` and names the item found and the item expected; a tensor of
the wrong shape throws an `ArgumentError` or `DimensionMismatch` with a `model:` prefix.

Public fields: `z_table` (the atomic numbers of the model's elements), `r_max` (cutoff, Å),
`e0` (reference energy of each element, eV), `scale` and `shift`.

The supported architecture is the one of MACE-MP-0b3 and MACE-MPA-0: a `ScaleShiftMACE` with
one head, two density-normalized interactions (`RealAgnosticDensityInteractionBlock`, then
`RealAgnosticDensityResidualInteractionBlock`), hidden irreps `Cx0e + Cx1o`, edge attributes
`1x0e + 1x1o + 1x2e + 1x3o`, correlation order 3 in both product blocks, a linear and a
nonlinear readout, ZBL pair repulsion, the Agnesi distance transform on the Bessel radial
basis, and `normalize2mom(silu)` activations.

## Evaluating

```julia
r = mace_energy_forces(model, positions, cell, atomic_numbers)
r.energy, r.forces, r.virial, r.stress
```

- `positions` is `3 × n`, one column per atom, in Å. Atoms need not lie inside the cell.
- `cell` is `3 × 3` with the lattice vectors as its columns, in Å.
- `atomic_numbers` holds one integer per atom; each must be an element of `model.z_table`.
- The structure is periodic in all three directions. There is no non-periodic mode.
- Energies are in eV, forces in eV/Å, the virial in eV and the stress in eV/Å³.
  `node_energies` holds the energy of each atom with its reference energy included.

The virial is `-∂E/∂ε` for the symmetric strain `ε`, and the stress is `∂E/∂ε / |det cell|`:
positive when the cell is under tension, so the stress is `-virial / |det cell|`. Both are
symmetric `3 × 3` matrices. Their Voigt components `(xx, yy, zz, yz, xz, xy)` are the entries
`[1,1]`, `[2,2]`, `[3,3]`, `[2,3]`, `[1,3]` and `[1,2]`.

`mace_energy_forces` allocates its neighbor list, workspace and result. For repeated
evaluations, allocate them once and call the in-place forms, which do not allocate:

```julia
nl = neighbor_list(positions, cell, model.r_max; extra_capacity = 64)
ws = MACEWorkspace{Float64}(model, size(positions, 2), length(nl.senders))
res = MACEResult{Float64}(size(positions, 2))
neighbor_list!(nl, positions, cell)
mace_energy_forces!(res, ws, model, positions, cell, atomic_numbers, nl)  # or mace_energy!
```

When only energies are needed, a `MACEEnergyWorkspace` takes the place of the `MACEWorkspace` in
`mace_energy!`. It holds no derivatives and nothing for a backward pass, and runs the per-atom
layers of both interactions on blocks of receivers, so it is smaller. Measured with
`Base.summarysize` for MACE-MPA-0 in Float64 at the default `edge_chunk = 4096` and
`block = 256`, with 78 edges per atom: at 10,976 atoms it takes 14,402 bytes per atom against
88,258 for a `MACEWorkspace`. Each workspace also holds buffers whose size does not depend on the
number of atoms or edges, about 63 MB for a `MACEEnergyWorkspace` and 118 MB for a
`MACEWorkspace` at those settings. They dominate small cells: at 256 atoms the two workspaces take
about 255 kB and 537 kB per atom. A smaller `edge_chunk` and `block` shrink that part. The
energies are those of `mace_energy_forces!` up to rounding; the forces, virial and stress of the
result are left as they were.

```julia
ws_e = MACEEnergyWorkspace{Float64}(model, size(positions, 2), length(nl.senders))
mace_energy!(res, ws_e, model, positions, cell, atomic_numbers, nl)
```

## The neighbor list

`neighbor_list(positions, cell, r_cut)` returns the directed edges `(sender, receiver, shift)`
with `0 < ‖positions[:, receiver] - positions[:, sender] + cell * shift‖ < r_cut`, sorted by
receiver. The model requires `r_cut >= model.r_max` and edges sorted by receiver, as
`neighbor_list` returns them, and throws an `ArgumentError` otherwise. Its capacity is the
number of edges found plus `extra_capacity`; `neighbor_list!` rebuilds the list for new
positions in place and throws when the new edge count exceeds the capacity, in which case the
list must be rebuilt with `neighbor_list`. A workspace is sized for a number of atoms and
edges and is used with any structure within them; for repeated evaluation it must cover the
list's capacity, `length(nl.senders)`, not only the current edge count `nl.n_edges`.

## Float32

`MACEModel{Float32}` computes in Float32 end to end. The arguments of the evaluation functions
must have the model's element type. Against LAMMPS-MACE on the reference structures, the
largest Float32 energy difference is 1.6e-5 eV on an energy of -235 eV, and the largest force
difference is 9.4e-6 eV/Å.

## Using PureBLAS

The matrix products of the model go through `LinearAlgebra.mul!`, so PureBLAS can carry them
without any change to PureMLIP:

```julia
using PureBLAS
PureBLAS.activate()
```

Every PureBLAS speed measurement so far comes from one Apple M6 machine on one thread; there is
no measurement on another machine yet. The [PureBLAS example](examples/pureblas.md) shows the
measured speed-ups, read from the committed benchmark records.

PureBLAS requires Julia 1.13 and is not registered. The tests use commit `c38f273f`; add that
commit with
`Pkg.add(url = "https://github.com/el-oso/PureBLAS.jl", rev = "c38f273fd75807811e647e5dc69ae85dbaceeaf6")`.
`PureBLAS.activate()` reroutes the BLAS calls of the whole process, not only PureMLIP's, and
`PureBLAS.deactivate()` restores the previous libraries. PureMLIP does not depend on PureBLAS.
The tests in `PureMLIP/test/pureblas` run with PureBLAS active. They check the reference
energies, forces and virials, computed with `mace_energy_forces`, and that `mace_energy!` and
`mace_energy_forces!` with a `MACEWorkspace` do not allocate; `mace_energy!` with a
`MACEEnergyWorkspace` is not covered there.

## Reference

```@docs
NeighborList
neighbor_list
neighbor_list!
MACEModel
MACEResult
MACEWorkspace
MACEEnergyWorkspace
mace_energy!
mace_energy_forces!
mace_energy_forces
```
