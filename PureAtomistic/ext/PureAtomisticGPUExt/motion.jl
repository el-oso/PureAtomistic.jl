# The velocity Verlet step and the energy queries of a simulation on the device. A step launches
# its kernels on the backend of the potential and reads back only the status word of the
# evaluation.

"""
    advance!(vv::VelocityVerlet, sys::System, p::DeviceMACEPotential, after_drift! = no_hook) -> nothing

The velocity Verlet step of [`advance!`](@ref) on the device: the half kicks and the drift are
kernels with the arithmetic of `kick!` and `drift!`, `after_drift!(sys)` runs after the drift, and
the forces come from `prepare!` and `energy_forces!` of `p`, which read back the status word once.
The arrays of `sys` must live on the backend of `p`.
"""
function Integrators.advance!(vv::VelocityVerlet, sys::System{T}, p::DeviceMACEPotential{T}, after_drift! = no_hook)::Nothing where {T <: AbstractFloat}
    _check_motion(sys, p)
    h = T(vv.dt)
    _kick!(sys, p, h / 2)
    _drift!(sys, p, h)
    after_drift!(sys)
    Potentials.prepare!(p, sys.positions, sys.cell)
    Potentials.energy_forces!(p, sys.positions, sys.cell)
    _kick!(sys, p, h / 2)
    return nothing
end

# Velocities and masses must be 1-based arrays of the potential's sizes on its backend; the
# positions and the cell are checked by `prepare!`.
function _check_motion(sys::System, p::DeviceMACEPotential)
    Base.require_one_based_indexing(sys.velocities, sys.masses)
    n = Potentials.n_atoms(p)
    (size(sys.velocities) == (3, n) && length(sys.masses) == n) ||
        throw(DimensionMismatch(lazy"pureatomistic: the system has $(natoms(sys)) atoms, the potential $n"))
    (get_backend(sys.velocities) == p.nl.backend && get_backend(sys.masses) == p.nl.backend) ||
        throw(ArgumentError("pureatomistic: velocities and masses must live on the backend of the potential"))
    return nothing
end

# Launches `kernel` on `backend` over `ndrange`; a launch with no work items is skipped.
function _launch!(kernel, backend::Backend, ndrange, args...)
    prod(ndrange) > 0 && kernel(backend)(args...; ndrange)
    return nothing
end

_kick!(sys::System{T}, p::DeviceMACEPotential{T}, τ::T) where {T} = _kick!(p.nl.backend, sys.velocities, p.result.forces, sys.masses, τ)
_drift!(sys::System{T}, p::DeviceMACEPotential{T}, h::T) where {T} = _drift!(p.nl.backend, sys.positions, sys.velocities, h)

# The kick and the drift on the 3 × n arrays of one backend.
_kick!(backend::Backend, V::AbstractMatrix{T}, F, m, τ::T) where {T} =
    _launch!(_kick_kernel!, backend, size(V), V, F, m, τ, T(AMU_IN_EV_FS2_PER_A2))
_drift!(backend::Backend, X::AbstractMatrix{T}, V, h::T) where {T} = _launch!(_drift_kernel!, backend, size(X), X, V, h)

@kernel function _kick_kernel!(V, @Const(F), @Const(m), τ, c)
    k, a = @index(Global, NTuple)
    V[k, a] = _kicked(V[k, a], F[k, a], m[a], τ, c)
end

# `kick!`'s update of one velocity component: v + τ F / m, with the mass converted by `c`.
@inline _kicked(v, f, m, τ, c) = v + τ / (m * c) * f

@kernel function _drift_kernel!(X, @Const(V), h)
    k, a = @index(Global, NTuple)
    X[k, a] = _drifted(X[k, a], V[k, a], h)
end

# `drift!`'s update of one position component.
@inline _drifted(x, v, h) = x + h * v

# A simulation on the device. The element type is left open here, since a method with `T` named in
# the signature would be ambiguous with the driver's methods on `Simulation`.
const _DeviceSimulation = Simulation{<:AbstractFloat, <:System, <:DeviceMACEPotential}

"""
    kinetic_energy(sim::Simulation) with a DeviceMACEPotential

The kinetic energy in eV, summed on the device in a fixed-shape tree and read back as one scalar.
"""
Systems.kinetic_energy(sim::_DeviceSimulation) = _energy_readback!(sim, false)

"""
    total_energy(sim::Simulation) with a DeviceMACEPotential

Kinetic plus potential energy in eV, added on the device and read back as one scalar.
"""
PureAtomistic.total_energy(sim::_DeviceSimulation) = _energy_readback!(sim, true)

"""
    temperature(sim::Simulation; dof = 3 * natoms(sim.system) - 3) with a DeviceMACEPotential

The kinetic temperature in K from the kinetic energy read back by `kinetic_energy(sim)`.
"""
function Systems.temperature(sim::_DeviceSimulation; dof::Integer = 3 * natoms(sim.system) - 3)
    dof >= 1 || throw(ArgumentError("pureatomistic: temperature needs at least one degree of freedom"))
    K = Systems.kinetic_energy(sim)
    T = typeof(K)
    return 2 * K / (T(dof) * T(KB_EV_PER_K))
end

# The kinetic energy, plus the potential energy of the last evaluation when `with_potential`, in
# the arithmetic of the CPU `kinetic_energy` and `total_energy` with the sum over the atoms taken
# in the tree order of PureMLIP's device sums.
function _energy_readback!(sim::Simulation{T}, with_potential::Bool) where {T}
    sys, p = sim.system, sim.potential
    _check_motion(sys, p)
    _kinetic_sum!(sys, p)
    _launch!(_energy_kernel!, p.nl.backend, 1, p.scalar, p.kinetic_sum, p.result.energy, T(AMU_IN_EV_FS2_PER_A2 / 2), with_potential)
    return _read_status!(p, p.host_scalar, p.scalar)[1]
end

# `p.kinetic_sum[1] = Σᵢ mᵢ |vᵢ|²` on the device, the sum of `kinetic_energy` before its factor.
function _kinetic_sum!(sys::System, p::DeviceMACEPotential)
    n = Potentials.n_atoms(p)
    _launch!(_kinetic_kernel!, p.nl.backend, n, p.kinetic, sys.velocities, sys.masses)
    PureMLIP.device_sum!(p.kinetic_sum, p.kinetic, n)
    return nothing
end

@kernel function _kinetic_kernel!(out, @Const(V), @Const(m))
    a = @index(Global)
    out[a] = _kinetic_term(V, m, a)
end

# The term of atom `a` in `kinetic_energy`'s sum.
@inline _kinetic_term(V, m, a) = m[a] * (V[1, a]^2 + V[2, a]^2 + V[3, a]^2)

@kernel function _energy_kernel!(out, @Const(total), @Const(energy), half_amu, with_potential)
    out[1] = _energy_value(total[1], energy[1], half_amu, with_potential)
end

@inline _energy_value(s, e, half_amu, with_potential) = with_potential ? half_amu * s + e : half_amu * s
