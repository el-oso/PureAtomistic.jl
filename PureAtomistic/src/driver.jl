"""
    Simulation(system, potential, integrator; thermostat = nothing, barostat = nothing, rng)

A molecular-dynamics run: the [`System`](@ref) `system` evolves under the forces of `potential`
with `integrator`. The constructor evaluates the potential at the initial positions, so the
forces and the energy of step 0 are in `potential.result`. `rng` is the random number generator
that stochastic couplings draw from; it is required so that a run never depends on a hidden
seed. `sim.step_count` counts the completed steps.

`thermostat` is `nothing` (constant energy) or an [`AbstractThermostat`](@ref), which needs the
[`VelocityVerlet`](@ref) integrator and takes its time step. `barostat` is `nothing` (constant
volume) or an [`AbstractBarostat`](@ref), which needs the [`VelocityVerlet`](@ref) integrator
and an [`AbstractStressPotential`](@ref); it rescales the cell within each step (see
[`step!`](@ref)).
"""
mutable struct Simulation{
        T <: AbstractFloat, S <: System{T}, P <: AbstractPotential, I <: AbstractIntegrator,
        H <: Union{Nothing, AbstractThermostat}, R <: Random.AbstractRNG, B <: Union{Nothing, AbstractBarostat},
    }
    const system::S
    const potential::P
    const integrator::I
    const thermostat::H
    const rng::R
    const barostat::B
    step_count::Int
end

function Simulation(
        system::System{T}, potential::AbstractPotential, integrator::AbstractIntegrator;
        thermostat = nothing, barostat = nothing, rng::Random.AbstractRNG
    ) where {T}
    isnothing(barostat) || barostat isa AbstractBarostat ||
        throw(ArgumentError(lazy"pureatomistic: barostat must be an AbstractBarostat or nothing, got $(typeof(barostat))"))
    isnothing(barostat) || integrator isa VelocityVerlet ||
        throw(ArgumentError(lazy"pureatomistic: a barostat needs the VelocityVerlet integrator, got $(typeof(integrator))"))
    isnothing(barostat) || potential isa AbstractStressPotential ||
        throw(ArgumentError(lazy"pureatomistic: a barostat needs an AbstractStressPotential, got $(typeof(potential))"))
    isnothing(thermostat) || thermostat isa AbstractThermostat ||
        throw(ArgumentError(lazy"pureatomistic: thermostat must be an AbstractThermostat or nothing, got $(typeof(thermostat))"))
    isnothing(thermostat) || integrator isa VelocityVerlet ||
        throw(ArgumentError(lazy"pureatomistic: a thermostat needs the VelocityVerlet integrator, got $(typeof(integrator))"))
    n_atoms(potential) == natoms(system) ||
        throw(DimensionMismatch(lazy"pureatomistic: the potential has $(n_atoms(potential)) atoms, the system has $(natoms(system))"))
    prepare!(potential, system.positions, system.cell)
    energy_forces!(potential, system.positions, system.cell)
    return Simulation{
        T, typeof(system), typeof(potential), typeof(integrator), typeof(thermostat), typeof(rng), typeof(barostat),
    }(system, potential, integrator, thermostat, rng, barostat, 0)
end

"""
    total_energy(sim)

Kinetic plus potential energy of `sim` in eV, with the kinetic energy accumulated in the
working type of the system.
"""
total_energy(sim::Simulation{T}) where {T} = kinetic_energy(sim.system) + T(sim.potential.result.energy)

"""
    kinetic_energy(sim::Simulation)

The kinetic energy of `sim.system` in eV.
"""
Systems.kinetic_energy(sim::Simulation) = kinetic_energy(sim.system)

"""
    temperature(sim::Simulation; dof = 3 * natoms(sim.system) - 3)

The kinetic temperature of `sim.system` in K.
"""
Systems.temperature(sim::Simulation; dof::Integer = 3 * natoms(sim.system) - 3) = Systems.temperature(sim.system; dof)

"""
    step!(sim) -> nothing

Advance `sim` by one time step: a step of its integrator, or of its thermostat when it has one.
With a barostat, its scale factor is computed from the state at the start of the step and the
cell and positions are rescaled after the drift, before the step's single force evaluation.
"""
function step!(sim::Simulation)::Nothing
    _step!(sim.thermostat, sim, _rescaling(sim.barostat, sim))
    sim.step_count += 1
    return nothing
end

# The `after_drift!` hook of the step: `nothing` without a barostat, otherwise the rescaling by
# the barostat's factor for this step.
_rescaling(::Nothing, ::Simulation) = nothing

function _rescaling(b::AbstractBarostat, sim::Simulation)
    integrator = sim.integrator::VelocityVerlet
    potential = sim.potential::AbstractStressPotential
    return _Rescale(scale_factor(b, sim.system, potential, integrator.dt))
end

struct _Rescale{T <: AbstractFloat}
    μ::T
end
(r::_Rescale)(sys) = rescale!(sys, r.μ)

_step!(::Nothing, sim::Simulation, ::Nothing) = advance!(sim.integrator, sim.system, sim.potential)
_step!(::Nothing, sim::Simulation, hook) = advance!(sim.integrator::VelocityVerlet, sim.system, sim.potential, hook)

function _step!(th::AbstractThermostat, sim::Simulation, hook)
    integrator = sim.integrator::VelocityVerlet
    return _thermostat_step!(th, sim, integrator.dt, _IntegratorStep(integrator, hook), hook)
end

_thermostat_step!(th, sim, dt, integrate!, ::Nothing) =
    thermostat_step!(th, sim.system, sim.potential, dt, sim.rng, integrate!)
_thermostat_step!(th, sim, dt, integrate!, hook) =
    thermostat_step!(th, sim.system, sim.potential, dt, sim.rng, integrate!, hook)

# One step of `integrator` with the step's `after_drift!` hook (`nothing` for none), the plain
# step a thermostat calls inside its own.
struct _IntegratorStep{I <: AbstractIntegrator, H}
    integrator::I
    hook::H
end
(s::_IntegratorStep{<:AbstractIntegrator, Nothing})(sys, potential) = advance!(s.integrator, sys, potential)
(s::_IntegratorStep)(sys, potential) = advance!(s.integrator::VelocityVerlet, sys, potential, s.hook)

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
