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
ws = MACEWorkspace{Float64}(model, size(positions, 2), nl.n_edges)
res = MACEResult{Float64}(size(positions, 2))
neighbor_list!(nl, positions, cell)
mace_energy_forces!(res, ws, model, positions, cell, atomic_numbers, nl)  # or mace_energy!
```

## The neighbor list

`neighbor_list(positions, cell, r_cut)` returns the directed edges `(sender, receiver, shift)`
with `0 < ‖positions[:, receiver] - positions[:, sender] + cell * shift‖ < r_cut`, sorted by
receiver. The model requires `r_cut >= model.r_max` and throws otherwise. Its capacity is the
number of edges found plus `extra_capacity`; `neighbor_list!` rebuilds the list for new
positions in place and throws when the new edge count exceeds the capacity, in which case the
list must be rebuilt with `neighbor_list`. A workspace is sized for a number of atoms and
edges and is used with any structure within them.

## Float32

`MACEModel{Float32}` computes in Float32 end to end. The arguments of the evaluation functions
must have the model's element type. Against LAMMPS-MACE on the reference structures, the
largest Float32 energy difference is 1.6e-5 eV on an energy of -235 eV, and the largest force
difference is 9.4e-6 eV/Å.
