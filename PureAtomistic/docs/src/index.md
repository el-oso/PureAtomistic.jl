# PureAtomistic.jl

Molecular dynamics on top of PureMLIP potentials, in pure Julia.

The package is organized in layers: `Systems` holds the atomic configuration, `Potentials`
evaluates energies and forces, `Integrators`, `Thermostats` and `Barostats` advance it, `Analysis`
reads the results, and `IO` reads and writes configurations.

The [example gallery](gallery.md) shows the validation runs of the molecular-dynamics driver:
energy conservation, canonical sampling with the Bussi thermostat, and a vacancy at 1300 K.
