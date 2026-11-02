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

using .Potentials: AbstractPotential, AbstractStressPotential, PotentialResult, n_atoms, cutoff, prepare!,
    energy_forces!, stress, node_energies, MACEPotential, HarmonicPotential, LennardJones
export AbstractPotential, AbstractStressPotential, PotentialResult, n_atoms, cutoff, prepare!,
    energy_forces!, stress, node_energies, MACEPotential, HarmonicPotential, LennardJones

export animate_trajectory, plot_vacancy_track, plot_arrhenius, plot_energy_conservation

public Potentials, Systems, Integrators, Thermostats, Barostats, IO, Analysis

end
