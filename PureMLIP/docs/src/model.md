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
ws = MACEWorkspace{Float64}(model, size(positions, 2), length(nl.senders); elements = atomic_numbers)
res = MACEResult{Float64}(size(positions, 2))
neighbor_list!(nl, positions, cell)
mace_energy_forces!(res, ws, model, positions, cell, atomic_numbers, nl)  # or mace_energy!
```

When only energies are needed, a `MACEEnergyWorkspace` takes the place of the `MACEWorkspace` in
`mace_energy!`. It holds no derivatives and nothing for a backward pass, and runs the per-atom
layers of both interactions on blocks of receivers, so it is smaller. Measured with
`Base.summarysize` for MACE-MPA-0 in Float64 at the default `edge_chunk = 4096` and
`block = 256`, with 78 edges per atom: at 10,976 atoms of one element it takes 14,402 bytes per
atom against 91,613 for a `MACEWorkspace` (89,557 without its contraction matrices). Each
workspace also holds buffers whose size does not depend on the number of atoms or edges, about
63 MB for a `MACEEnergyWorkspace` and 132 MB for a `MACEWorkspace` at those settings, plus the
`MACEWorkspace`'s contraction matrices, 22.6 MB per element. They dominate small cells. A smaller
`edge_chunk` and `block` shrink the fixed part. The energies are those of `mace_energy_forces!` up
to rounding; the forces, virial and stress of the result are left as they were.

```julia
ws_e = MACEEnergyWorkspace{Float64}(model, size(positions, 2), length(nl.senders))
mace_energy!(res, ws_e, model, positions, cell, atomic_numbers, nl)
```

`elements` is a required keyword: the atomic numbers the workspace can evaluate. The workspace
holds the symmetric-contraction matrices only for the listed elements (for MACE-MPA-0, 22.6 MB per
element in Float64 and 11.3 MB in Float32), built by the constructor; an evaluation on an atom of
any other element throws an `ArgumentError`. With these matrices the first product block and the
pullbacks of both run as matrix products over groups of 64 atoms of one element, through `mul!`,
so an activated PureBLAS runs them.

`MACEWorkspace{T}(model, n_atoms, n_edges; elements, nchunks)` splits the atoms of an evaluation
into `nchunks` receiver chunks, contiguous ranges of nearly equal size, each evaluated with its own
buffers. Results for different `nchunks` agree up to rounding; for a given `nchunks` they are the
same on every evaluation. With several chunks `edge_chunk` defaults to 1024 edges instead of
4096, since every chunk holds its own edge-chunk buffers. Each chunk after the first then adds,
for MACE-MPA-0 in Float64 at the default `block`, about 55 MB (25.5 MB of edge-chunk buffers, 15
MB of block buffers and 15 MB of symmetric-contraction scratch) plus 4,096 bytes per atom:
112,029 bytes per atom with 4 chunks and 148,560 with 8 at 10,976 atoms of one element, against
91,613 with one; the chunks share the contraction matrices.

Passing `ThreadedChunks()` as the last argument of `mace_energy!` or `mace_energy_forces!` runs
the receiver chunks of the forward and the backward pass in parallel on the threads Julia was
started with: chunks 1 to `nchunks - 1` on worker tasks that the workspace keeps from one
evaluation to the next, the last on the calling task. The sums over the edges that give the
forces and the virial run on the calling task. The results are the same bits as without `ThreadedChunks()` for the same `nchunks`, on any
number of threads. The speed-ups in `PureMLIP/bench/results/` (`thread_scaling_*`,
`ab_threaded_*`) were measured under `julia -t n,0 --gcthreads=1,0` (no interactive thread, one
garbage-collection thread) with `nchunks = n` at 2, 4, 6 and 8 threads on an Apple M6 (2 + 4 fast
cores and 6 efficiency cores); past 6 threads the extra tasks run on efficiency cores. Each task
runs its own matrix products, so a multithreaded BLAS should be limited to one thread
(`BLAS.set_num_threads(1)` with OpenBLAS; PureBLAS runs one thread until
`PureBLAS.set_num_threads` is called).

The first evaluation with `ThreadedChunks()` on a workspace starts the workers, and with PureBLAS each worker
allocates PureBLAS's task-local matrix-product scratch once. The bytes that first evaluation
allocates, measured with `@allocated` inside a function on cu_bench(2) (32 atoms) with four
threads on Apple arm64 (macOS), Julia 1.12.7 and 1.13.3, are, for both precisions and both
`mace_energy!` and `mace_energy_forces!`:

| `nchunks` | OpenBLAS, first evaluation | OpenBLAS, first evaluation after `close(ws)` |
|---|---|---|
| 2 | 5,664 | 384 |
| 4 | 7,808 | 1,152 |
| 8 | 12,432 | 2,688 |

They are the tasks, channels, locks and vectors of the workers, and the same in three fresh
processes on each Julia version. The tests gate
these figures exactly on macOS arm64 and as an upper bound of twice the figure on every other
platform, where they have not been measured. With PureBLAS the first
evaluation allocates megabytes, because each worker's scratch of matrix-product buffers is
retained: 11.1 MB (1 worker), 21.1 MB (3 workers) and 27.5 MB (7 workers) in Float64, and 5.6,
10.8 and 13.8 MB in Float32, the same in every process. The total allocated is larger than the
retained scratch and varies between processes for `mace_energy_forces!` (Float64: 23.4
to 37.6 MB with 4 chunks and 31.4 to 45.1 MB with 8, over eighteen processes; Float32, 4 chunks: 11.9 to 17.9 MB), so the tests gate a range
there: per precision and chunk count, from 512 KiB to 1.4 times the largest total seen, on
macOS arm64 only; on other platforms, where PureBLAS's blocking differs and the totals have not
been measured, they gate an upper bound of twice that total. The range is temporary: it stands
in for an exact figure while PureBLAS keeps its scratch per task. What the extra, unretained
bytes are has not been traced. With `mace_energy!` the totals were
the same in all three processes: 1.65, 4.21 and 8.45 MB in Float64 and 0.83, 2.14 and 4.20 MB in
Float32 for 2, 4 and 8 chunks. The allocation profiler lists the small objects of the workers
(tasks, channels, locks) but not the scratch vectors, whose data buffers it does not record. The tests show that later evaluations
allocate nothing, with OpenBLAS and with PureBLAS: the fewest bytes of five evaluations is 0 (a
single evaluation can allocate up to 192 bytes more when a thread first runs a function in a
task, an entry in the runtime's method cache). Idle workers wait without running. They stop
when the workspace is no longer reachable, or at once with `close(ws)`; a later threaded
evaluation starts them again. The neighbor list of a threaded evaluation must have the element
type of the model. A workspace serves one evaluation at a time: a threaded phase started on `ws`
while another runs, or `close(ws)` during one, throws.

## Evaluation on a GPU

With KernelAbstractions loaded (and Metal, CUDA or AMDGPU for a GPU), the `PureMLIPGPUExt`
extension evaluates a model on the device that holds its arrays:

```julia
using KernelAbstractions, Metal  # KernelAbstractions loads the extension
backend = MetalBackend()
P, C = Float32.(positions), Float32.(cell)
dm = PureMLIP.device_model(model32, backend; elements = unique(atomic_numbers))
dP, dC = MtlArray(P), MtlArray(C)
dnl = PureMLIP.device_neighbor_list(dP, dC, model32.r_max; extra_capacity = 1000)  # built on the device
n = size(P, 2)
ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
res = PureMLIP.device_result(dm, n)
mace_energy_forces!(res, ws, dm, dP, dC, MtlArray(Int32.(atomic_numbers)), dnl)
neighbor_list!(dnl, dP, dC)  # after the positions have changed, on the device
```

`device_neighbor_list(positions, cell, r_cut; skin, extra_capacity)` builds the list on the
device with the grid of the CPU's `neighbor_list`, so its edges, their order and the map from
each edge to its reverse equal those of `neighbor_list(positions, cell, r_cut + skin)` in the same
precision; `PureMLIP.device_neighbor_list(nl, backend)` copies a host list instead.
`neighbor_list!` rebuilds it on the device when an atom has moved by more than half the skin since
the last build or the cell has changed (every call when the skin is 0, the default), and reads back
only the list's status word. When the edges exceed the capacity it throws, and the list holds no
edges until it is built anew with `device_neighbor_list`; an evaluation with such a list throws
the same error. The default skin is 0 because a rebuild costs under 1 % of an evaluation on Metal
(`bench/results/gpu_neighbor_skin_metal.json`).

The results stay on the device; `Array(res.forces)` copies them to the host. The forces are
gathered per atom over the edges it receives, through the list's map from each edge to its
reverse, and the energy and virial are fixed-shape tree sums, so an evaluation uses no atomic
operations and repeats bit for bit. Metal has no Float64, so a model on Metal is a
`MACEModel{Float32}`. The device backward pass supports at most 256 channels;
`device_model` throws an `ArgumentError` for a model with more, before any launch.

The device `mace_energy!`, `mace_energy_forces!` and `neighbor_list!` are the second exception to
"`!` functions do not allocate": each call allocates host memory for its kernel launches, its
matrix products and the one read of the device's status word. Measured on cu_bench(2) (32 Cu atoms,
MACE-MPA-0, one element, the workspace defaults; for `neighbor_list!` a 6 Å cutoff, a call that
rebuilds and one that keeps the list), as the fewest bytes of five calls:

| backend | `mace_energy!` | `mace_energy_forces!` | `neighbor_list!` (rebuild / keep) |
|---|---|---|---|
| KernelAbstractions CPU (Julia 1.12 and 1.13, Float32 and Float64) | 1,168 | 2,400 | 0 / 0 |
| Metal, Apple M6 (Julia 1.13, Float32) | 991,408 to 991,504 | 2,007,888 to 2,009,344 | 10,048 / 11,760 |

On Metal one `mul!` allocates 3,008 bytes on the host, and the symmetric contraction runs one
matrix product per channel, so most of the figure grows with the channels, the contraction
blocks and the elements; more edge chunks add launches too. The tests gate these figures as
upper bounds per backend. The device memory does not grow: on Metal the allocated device memory
reads the same after 10 and after 210 evaluations.

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
`mace_energy_forces!` with a `MACEWorkspace` do not allocate, also with `ThreadedChunks()` on four
threads once the workers have started, where they give the bits of the serial evaluation;
`mace_energy!` with a `MACEEnergyWorkspace` is not covered there.

## Reference

```@docs
NeighborList
neighbor_list
neighbor_list!
MACEModel
MACEResult
MACEWorkspace
MACEEnergyWorkspace
ThreadedChunks
close(::MACEWorkspace)
mace_energy!
mace_energy_forces!
mace_energy_forces
```
