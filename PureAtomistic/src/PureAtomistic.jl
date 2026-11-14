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

using .Integrators: FIRE, FIRE2, RelaxResult, relax!, TrustRegionBFGS, TrustRegionLBFGS, NEBResult, neb
export FIRE, FIRE2, RelaxResult, relax!, TrustRegionBFGS, TrustRegionLBFGS, NEBResult, neb

public device_potential, to_device, run_md, Potentials, Systems, Integrators, Thermostats, Barostats, IO, Analysis

StrictMode.@strict_exempt :noalloc Simulation reason = "entry point: simulation constructor"
StrictMode.@strict_exempt :noalloc run_md reason = "entry point: runs a simulation from a config file"
StrictMode.@strict_exempt :noalloc _simulate reason = "entry point: builds the simulation of `run_md` and writes its trajectory"
StrictMode.@strict_exempt :noalloc _simulate_b reason = "entry point: picks the barostat of the `run_md` simulation"
StrictMode.@strict_exempt :noalloc _read_config_file reason = "entry point: reads the config file of `run_md`"
StrictMode.@strict_exempt :noalloc _read_config reason = "entry point: checks the config table of `run_md`"
StrictMode.@strict_exempt :noalloc _read_thermostat reason = "entry point: reads the thermostat table of the `run_md` config"
StrictMode.@strict_exempt :noalloc _read_barostat reason = "entry point: reads the barostat table of the `run_md` config"
StrictMode.@strict_exempt :noalloc _require reason = "entry point: checks that a key of the `run_md` config is present"
StrictMode.@strict_exempt :noalloc _key reason = "entry point: names a key of the `run_md` config in an error"
StrictMode.@strict_exempt :noalloc _show reason = "entry point: shows a value of the `run_md` config in an error"
StrictMode.@strict_exempt :noalloc _show_value reason = "entry point: shows a value of the `run_md` config in an error"
StrictMode.@strict_exempt :noalloc _table reason = "entry point: reads a table of the `run_md` config"
StrictMode.@strict_exempt :noalloc _string reason = "entry point: reads a string of the `run_md` config"
StrictMode.@strict_exempt :noalloc _bool reason = "entry point: reads a Boolean of the `run_md` config"
StrictMode.@strict_exempt :noalloc _int reason = "entry point: reads an integer of the `run_md` config"
StrictMode.@strict_exempt :noalloc _float reason = "entry point: reads a number of the `run_md` config"
StrictMode.@strict_exempt :noalloc _size reason = "entry point: reads the lattice size of the `run_md` config"

end
