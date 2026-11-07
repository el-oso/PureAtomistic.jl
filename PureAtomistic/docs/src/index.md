# PureAtomistic.jl

Molecular dynamics on top of PureMLIP potentials, in pure Julia.

The package is organized in layers: `Systems` holds the atomic configuration, `Potentials`
evaluates energies and forces, `Integrators`, `Thermostats` and `Barostats` advance it, `Analysis`
reads the results, and `IO` reads and writes configurations.

The [example gallery](gallery.md) shows the validation runs of the molecular-dynamics driver:
energy conservation, canonical sampling with the Bussi thermostat, and a vacancy at 1300 K.

## Molecular dynamics on a GPU

With KernelAbstractions loaded (and Metal, CUDA or AMDGPU for a GPU), the `PureAtomisticGPUExt`
extension runs the whole velocity Verlet step on the device that holds the system's arrays:

```julia
using PureAtomistic, PureMLIP, Metal, Random
model = MACEModel{Float32}(load_mace_checkpoint(path))
host = PureAtomistic.Systems.fcc_cell(3.615f0, 5, :Cu)
sys = PureAtomistic.to_device(host, MetalBackend())
p = PureAtomistic.device_potential(model, sys; skin = 0)
sim = Simulation(sys, p, VelocityVerlet(2.0); rng = Xoshiro(0))
run!(sim, 1000; every = 100, observer = s -> (copy_state!(host, s.system); println(total_energy(s))))
```

The kicks and the drift are kernels with the arithmetic of the CPU step, so for the same forces
they give the CPU's bits. A step copies no array between host and device: the host reads back the
status word of the neighbor list once (`p.readbacks` counts every read), and `kinetic_energy`,
`total_energy` and `temperature` of the simulation each read one scalar summed on the device.
`copy_state!` copies the positions, velocities and cell to a host system at the output interval.

When a step packs more edges than the device list has room for, the potential builds a new list
and workspace with `extra_capacity_fraction` spare edges (default 0.05), evaluates again, counts it
in `p.n_regrows` and reports it on standard error. That step reads the status word twice (both
counted in `p.readbacks`), and the new list reads its edge count and its status word back while it
is built (not counted). The old list and workspace stay allocated until the garbage collector frees
them. Spare edges cost evaluation time: on Metal at 10,976 atoms, 10 % costs 5.6 % per evaluation,
so the default 5 % costs about 2.8 %. In 2,000 steps of 2 fs of 500 Cu atoms near 1300 K at
constant volume the edge count stayed within 3.5 % below and 0.005 % above that of the first step,
and a rebuild took 8 ms against 68 ms for an evaluation
(`bench/results/gpu_capacity_margin_metal.json`), so a rare regrowth is cheap.

`BussiThermostat`, `LangevinBAOAB` and `BerendsenBarostat` run on the device too. Their random
numbers come from the counter-based generator Philox4x32-10 (J. K. Salmon et al., SC'11), drawn
in the kernels: each step takes one key from `sim.rng`, and each normal number has its own counter
(atom and component for the Langevin noise), so a seeded run repeats bit for bit on one device and a
regrowth does not move the generator. Normal numbers come from the Box–Muller transform and the
gamma variate of the Bussi step from the Marsaglia–Tsang method, as on the CPU. Device runs match
CPU runs in their statistics, not in their random numbers. The Bussi step sums the kinetic energy
and computes its scale factor on the device, and the BAOAB step runs its O step as a kernel, so a
thermostatted step reads back only the status word. The barostat computes the pressure on the
device and reads it back as one scalar, so a barostatted step reads twice; `rescale!` scales the
cell and the positions on the device. Under the barostat the edge count follows the volume: in
2,000 steps of 2 fs of 500 Cu atoms near 1300 K at 0, 10 and 30 GPa, the volume ranged from 0.906
to 1.106 times its first value and the edge count from 12.5 % below to 3.8 % above, and the default
5 % spare edges never regrew the list (`bench/results/gpu_barostat_regrow_metal.json`).

A step allocates host memory for its kernel launches and the status read, an exception to the
rule that `!` functions do not allocate: 2,464 bytes on the KernelAbstractions CPU backend and
about 457,000 bytes on Metal for the tiny test checkpoint, both gated by the GPU test suite, which
also checks that the device memory in use stays the same over 1,000 steps. With MACE-MPA-0 on 32
Cu atoms a step allocates 2,400 bytes on the CPU backend and about 2.02 MB on Metal, almost all of
it the evaluation's matrix products.
