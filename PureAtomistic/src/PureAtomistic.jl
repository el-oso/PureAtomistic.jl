module PureAtomistic

# Dependencies the package source will use; imported here so each one is a declared, used dependency.
import LinearAlgebra, Random, PrecompileTools, PureDAQP, PureMLIP, StrictMode, TOML, TypeContracts

include("systems/Systems.jl")
include("potentials/Potentials.jl")
include("integrators/Integrators.jl")
include("thermostats/Thermostats.jl")
include("barostats/Barostats.jl")
include("io/IO.jl")
include("analysis/Analysis.jl")
include("viz.jl")

using .Systems: System, natoms, kinetic_energy, copy_state!
export copy_state!
using .Potentials: AbstractPotential, AbstractStressPotential, PotentialResult, n_atoms, cutoff, prepare!,
    energy_forces!, stress, node_energies, MACEPotential, HarmonicPotential, LennardJones
using .Integrators: AbstractIntegrator, VelocityVerlet, advance!
export AbstractPotential, AbstractStressPotential, PotentialResult, n_atoms, cutoff, prepare!,
    energy_forces!, stress, node_energies, MACEPotential, HarmonicPotential, LennardJones
export AbstractIntegrator, VelocityVerlet, Simulation, step!, run!, total_energy
using .Thermostats: AbstractThermostat, LangevinBAOAB, BussiThermostat, thermostat_step!
export AbstractThermostat, LangevinBAOAB, BussiThermostat
using .Barostats: AbstractBarostat, BerendsenBarostat, scale_factor, rescale!
export AbstractBarostat, BerendsenBarostat

include("driver.jl")
include("md_run.jl")
include("device.jl")

export animate_trajectory, plot_vacancy_track, plot_arrhenius, plot_energy_conservation

using .Integrators: FIRE, FIRE2, RelaxResult, relax!, TrustRegionBFGS, TrustRegionLBFGS
export FIRE, FIRE2, RelaxResult, relax!, TrustRegionBFGS, TrustRegionLBFGS

public device_potential, to_device, run_md, Potentials, Systems, Integrators, Thermostats, Barostats, IO, Analysis

end
