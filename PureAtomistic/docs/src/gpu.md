# Molecular dynamics on a GPU

With KernelAbstractions loaded (and Metal, CUDA or AMDGPU for a GPU), the `PureAtomisticGPUExt`
extension runs the whole velocity Verlet step on the device that holds the system's arrays.

## Usage


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

## What stays on the device

The positions, velocities, cell, masses and atomic numbers, the neighbor list, the model's
workspace, the energy, forces and stress, the kinetic-energy sums, the thermostat and barostat
arithmetic and the random numbers all live on the device. Per step the host does three things: it
launches the kernels, draws one 64-bit key from `sim.rng` when a thermostat is present, and reads
back the status word of the neighbor list (plus one scalar for the barostat's pressure). The
observer copies the state to a host system with `copy_state!` only at the output interval.

## Validation on Metal

`PureAtomistic/bench/md_validation.jl` runs the validations of the CPU driver through this path
with `PUREATOMISTIC_DEVICE=Metal` (Float32, MACE-MPA-0, skin 0), and stores the records as
`md_*_f32_metal.json` in `PureAtomistic/bench/results/`. Each record carries the verdicts of
`criteria` and `compare_to_baseline` against the Float32 records of the CPU. The thermostat noise
of the device comes from Philox and that of the CPU from Xoshiro, so the device runs are
statistically independent of the baseline.

| Run | Metal | CPU Float32 baseline | Verdict |
|:----|:------|:---------------------|:--------|
| NVE, dt 1 to 24 fs: drift, RMS (eV/atom/ps, eV/atom) | stable at each dt; dt 24 fs: 5.8e-5, 4.7e-4 | dt 24 fs: 5.9e-5, 4.7e-4 | stable where the baseline is stable |
| NVE, dt 32 fs | unstable (RMS 1.2e-3 above the 5e-4 bound) | unstable (1.2e-3) | same boundary as the baseline |
| NVT, Bussi 1200 K: mean temperature | 1205.5 K, mean z 1.15, variance z -0.51 | 1206.9 K, mean z 0.95, variance z -0.22 | mean difference within 3 combined SEs |
| Vacancy at 1300 K, 440 ps: hops | 56 | 64 | exact conditional binomial test p = 0.52 (threshold 0.01) |

The records hold every number: `bench/results/md_nve_conservation_f32_metal.json`,
`md_thermostat_check_f32_metal.json` and `md_vacancy_hop_f32_metal.json`. The vacancy run has the
simulated length of the CPU Float32 baseline, 440 ps, and the neighbor list never regrew in any run.
The 32 fs run is the edge of stability, so its failing absolute criterion is the expected result and
what is compared is that it fails where the baseline fails.

The absence of regrowth under the barostat (0, 10 and 30 GPa, see above) is specific to the 6 Å
cutoff of MACE-MPA-0 lying between the fifth and sixth fcc shells. At a lattice constant `a` those
shells lie at `a √(5/2)` and `a √3`, so the cutoff stays between them for 3.464 Å < `a` < 3.795 Å,
that is volumes from 0.88 to 1.15 times that of `a = 3.615 Å`. A larger change of the volume, or
another cutoff, carries a shell across the cutoff and changes the edge count abruptly.

## Failures

A thermostat that fails on the device (a gamma sampler at its cap, or a Bussi step at zero kinetic
energy) sets a flag in the status word of the neighbor list, and the next status read throws. The
flag is sticky: every later status read of that potential throws, and the scalar reads of the energy
queries and of the barostat do not look at it. Unlike the CPU, which throws before it changes
anything, the Bussi step at zero kinetic energy has made its first half kick and its drift when the
exception is raised, and `sim.step_count` has not advanced.
