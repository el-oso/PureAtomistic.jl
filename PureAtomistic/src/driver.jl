"""
    Simulation(system, potential, integrator; thermostat = nothing, barostat = nothing, rng)

A molecular-dynamics run: the [`System`](@ref) `system` evolves under the forces of `potential`
with `integrator`. The constructor evaluates the potential at the initial positions, so the
forces and the energy of step 0 are in `potential.result`. `rng` is the random number generator
that stochastic couplings draw from; it is required so that a run never depends on a hidden
seed. `sim.step_count` counts the completed steps.

`thermostat` is `nothing` (constant energy) or an [`AbstractThermostat`](@ref), which needs the
[`VelocityVerlet`](@ref) integrator and takes its time step. Only `barostat = nothing` is
accepted: no barostat is available, and any other value throws.
"""
mutable struct Simulation{
        T <: AbstractFloat, S <: System{T}, P <: AbstractPotential, I <: AbstractIntegrator,
        H <: Union{Nothing, AbstractThermostat}, R <: Random.AbstractRNG,
    }
    const system::S
    const potential::P
    const integrator::I
    const thermostat::H
    const rng::R
    step_count::Int
end

function Simulation(
        system::System{T}, potential::AbstractPotential, integrator::AbstractIntegrator;
        thermostat = nothing, barostat = nothing, rng::Random.AbstractRNG
    ) where {T}
    isnothing(barostat) || throw(ArgumentError("pureatomistic: the driver supports only barostat = nothing"))
    isnothing(thermostat) || thermostat isa AbstractThermostat ||
        throw(ArgumentError(lazy"pureatomistic: thermostat must be an AbstractThermostat or nothing, got $(typeof(thermostat))"))
    isnothing(thermostat) || integrator isa VelocityVerlet ||
        throw(ArgumentError(lazy"pureatomistic: a thermostat needs the VelocityVerlet integrator, got $(typeof(integrator))"))
    n_atoms(potential) == natoms(system) ||
        throw(DimensionMismatch(lazy"pureatomistic: the potential has $(n_atoms(potential)) atoms, the system has $(natoms(system))"))
    prepare!(potential, system.positions, system.cell)
    energy_forces!(potential, system.positions, system.cell)
    return Simulation{T, typeof(system), typeof(potential), typeof(integrator), typeof(thermostat), typeof(rng)}(
        system, potential, integrator, thermostat, rng, 0
    )
end

"""
    total_energy(sim)

Kinetic plus potential energy of `sim` in eV, with the kinetic energy accumulated in the
working type of the system.
"""
total_energy(sim::Simulation{T}) where {T} = kinetic_energy(sim.system) + T(sim.potential.result.energy)

"""
    step!(sim) -> nothing

Advance `sim` by one time step: a step of its integrator, or of its thermostat when it has one.
"""
function step!(sim::Simulation)::Nothing
    _step!(sim.thermostat, sim)
    sim.step_count += 1
    return nothing
end

_step!(::Nothing, sim::Simulation) = advance!(sim.integrator, sim.system, sim.potential)

function _step!(th::AbstractThermostat, sim::Simulation)
    integrator = sim.integrator::VelocityVerlet
    return thermostat_step!(th, sim.system, sim.potential, integrator.dt, sim.rng, _IntegratorStep(integrator))
end

# One step of `integrator`, the plain step a thermostat calls inside its own.
struct _IntegratorStep{I <: AbstractIntegrator}
    integrator::I
end
(s::_IntegratorStep)(sys, potential) = advance!(s.integrator, sys, potential)

"""
    run!(sim, nsteps; every = 1, observer = nothing) -> nothing

Take `nsteps` steps. When `observer` is given it is called as `observer(sim)` after every
`every`-th step (steps `every`, `2every`, … of this call).
"""
function run!(sim::Simulation, nsteps::Integer; every::Integer = 1, observer = nothing)::Nothing
    nsteps >= 0 || throw(ArgumentError(lazy"pureatomistic: nsteps must not be negative, got $nsteps"))
    every >= 1 || throw(ArgumentError(lazy"pureatomistic: every must be at least 1, got $every"))
    for i in 1:nsteps
        step!(sim)
        if !isnothing(observer) && i % every == 0
            observer(sim)
        end
    end
    return nothing
end
