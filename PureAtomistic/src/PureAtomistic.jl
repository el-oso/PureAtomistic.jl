module PureAtomistic

# Dependencies the package source will use; imported here so each one is a declared, used dependency.
import LinearAlgebra, Random, PrecompileTools, PureMLIP, StrictMode, TypeContracts

include("systems/Systems.jl")
include("potentials/Potentials.jl")
include("integrators/Integrators.jl")
include("thermostats/Thermostats.jl")
include("barostats/Barostats.jl")
include("io/IO.jl")
include("analysis/Analysis.jl")
include("viz.jl")

using .Systems: System, natoms, kinetic_energy
using .Potentials: AbstractPotential, AbstractStressPotential, PotentialResult, n_atoms, cutoff, prepare!,
    energy_forces!, stress, node_energies, MACEPotential, HarmonicPotential, LennardJones
using .Integrators: AbstractIntegrator, VelocityVerlet, advance!
export AbstractPotential, AbstractStressPotential, PotentialResult, n_atoms, cutoff, prepare!,
    energy_forces!, stress, node_energies, MACEPotential, HarmonicPotential, LennardJones
export AbstractIntegrator, VelocityVerlet, Simulation, step!, run!, total_energy

include("driver.jl")

export animate_trajectory, plot_vacancy_track, plot_arrhenius, plot_energy_conservation

using .Integrators: FIRE, RelaxResult, relax!
export FIRE, RelaxResult, relax!

public Potentials, Systems, Integrators, Thermostats, Barostats, IO, Analysis

end
