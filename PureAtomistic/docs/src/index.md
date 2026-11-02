# PureAtomistic.jl

Molecular dynamics on top of PureMLIP potentials, in pure Julia.

The package is organized in layers: `Systems` holds the atomic configuration, `Potentials`
evaluates energies and forces, `Integrators`, `Thermostats` and `Barostats` advance it, `Analysis`
reads the results, and `IO` reads and writes configurations.
