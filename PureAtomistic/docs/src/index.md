# PureAtomistic.jl

Molecular dynamics on top of PureMLIP potentials, in pure Julia.

The package is organized in layers: `Systems` holds the atomic configuration, `Potentials`
evaluates energies and forces, `Integrators`, `Thermostats` and `Barostats` advance it, `Analysis`
reads the results, and `IO` reads and writes configurations.

The [example gallery](gallery.md) shows the validation runs of the molecular-dynamics driver:
energy conservation, canonical sampling with the Bussi thermostat, and a vacancy at 1300 K.

[Trajectory analysis](analysis.md) covers `g(r)`, coordination numbers, the mean-square
displacement and diffusion coefficient, and RMSD, with the atomic diffusion of a vacancy at
1200 K as an example.

## Molecular dynamics on a GPU

With KernelAbstractions and a vendor package loaded, the velocity Verlet step, the Bussi and
Langevin thermostats and the Berendsen barostat run on the device; see [GPU](gpu.md).
