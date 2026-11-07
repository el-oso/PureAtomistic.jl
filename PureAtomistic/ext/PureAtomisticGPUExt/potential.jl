# `DeviceMACEPotential`: a MACE potential whose model, neighbor list, workspace and result live on
# a KernelAbstractions backend, and `to_device`, which copies a system there.

# The module of PureMLIP's GPU extension, reached through a device type that it defines. An
# extension cannot name another package's extension when it loads, since that one may load after
# it; the module of a concrete type is a constant to inference, so calls through it dispatch
# statically.
_mlip(x) = parentmodule(typeof(x))

"""
    DeviceMACEPotential{T, M, L, W, R, VI, V} <: AbstractStressPotential

A PureMLIP MACE model as a potential on a KernelAbstractions backend; build it with
[`device_potential`](@ref PureAtomistic.device_potential). It holds the device model `model`,
the device neighbor list `nl` of cutoff `r_max` with skin `skin`, the device workspace `ws`, the
device result `result` (`result.energy` is a device vector of length 1, `result.forces` a device
`3 × n` matrix), the atomic numbers `numbers` as a device `Int32` vector, and the scratch of the
device reductions of the kinetic energy. The arrays of a configuration passed to it must live on
its backend.

[`prepare!`](@ref) rebuilds the list on the device and reads nothing back.
[`energy_forces!`](@ref) evaluates on the device and then reads the status word of the list once.
When the configuration holds more edges than the list has room for, the evaluation is redone
with a list and a workspace built with `extra_capacity_fraction` times as many edges again as the
configuration holds, reported on standard error and counted in `n_regrows`. Every copy from
device to host memory that the potential makes after it is built (each status read, and each
scalar read of the energy queries of a `Simulation`) is counted in `readbacks`; the construction
of a neighbor list (by `device_potential` and at a regrowth) reads back on its own and is not
counted.
"""
mutable struct DeviceMACEPotential{T <: AbstractFloat, M, L, W, R, VI <: AbstractVector{Int32}, V <: AbstractVector{T}} <: AbstractStressPotential
    const model::M
    const numbers::VI
    const skin::Float64
    const extra_capacity_fraction::Float64
    nl::L
    ws::W
    const result::R
    n_regrows::Int
    readbacks::Int
    const kinetic::V          # n: mᵢ |vᵢ|² of each atom
    const kinetic_sum::V      # 1: the sum of `kinetic`
    const scalar::V           # 1: the energy that a query reads back
    const host_scalar::Vector{T}
end

function PureAtomistic.device_potential(
        model::MACEModel{T}, sys::System{T}; skin, extra_capacity_fraction = 0.1
    ) where {T <: AbstractFloat}
    backend = _system_backend(sys)
    (isfinite(skin) && skin >= 0) ||
        throw(ArgumentError(lazy"pureatomistic: the skin must be finite and non-negative, got $skin"))
    (isfinite(extra_capacity_fraction) && extra_capacity_fraction >= 0) ||
        throw(ArgumentError(lazy"pureatomistic: extra_capacity_fraction must be finite and non-negative, got $extra_capacity_fraction"))
    fraction = Float64(extra_capacity_fraction)
    n = natoms(sys)
    species = Array(sys.species)
    dm = PureMLIP.device_model(model, backend; elements = sort!(unique(species)))
    numbers = allocate(backend, Int32, n)
    copyto!(numbers, Int32.(species))
    count = PureMLIP.device_neighbor_list(sys.positions, sys.cell, model.r_max; skin).capacity
    nl = _sized_list(sys.positions, sys.cell, model.r_max, skin, fraction, count)
    ws = PureMLIP.device_workspace(dm, n, nl.capacity)
    res = PureMLIP.device_result(dm, n)
    z(k) = KernelAbstractions.zeros(backend, T, k)
    kinetic = z(n)
    return DeviceMACEPotential{T, typeof(dm), typeof(nl), typeof(ws), typeof(res), typeof(numbers), typeof(kinetic)}(
        dm, numbers, Float64(skin), fraction, nl, ws, res, 0, 0, kinetic, z(1), z(1), zeros(T, 1)
    )
end

# The backend of the arrays of `sys`, which must all live on it and be 1-based.
function _system_backend(sys::System)
    Base.require_one_based_indexing(sys.positions, sys.velocities, sys.masses, sys.species, sys.cell)
    backend = get_backend(sys.positions)
    all(A -> get_backend(A) == backend, (sys.velocities, sys.masses, sys.species, sys.cell)) ||
        throw(ArgumentError("pureatomistic: the arrays of the system must all live on one backend"))
    return backend
end

# A device list for `count` edges with room for `fraction` more.
_sized_list(positions, cell, r_cut, skin, fraction::Float64, count::Int) =
    PureMLIP.device_neighbor_list(positions, cell, r_cut; skin, extra_capacity = ceil(Int, fraction * count))

"""
    to_device(sys::System, backend::KernelAbstractions.Backend) -> System

A copy of `sys` whose positions, velocities, masses, atomic numbers and cell are 1-based arrays
on `backend`, with the element types of `sys`.
"""
function PureAtomistic.to_device(sys::System{T}, backend::Backend) where {T <: AbstractFloat}
    function up(A)
        H = Array{eltype(A)}(undef, size(A))
        copyto!(H, A)
        D = allocate(backend, eltype(A), size(A))
        copyto!(D, H)
        return D
    end
    return System{T}(up(sys.positions), up(sys.velocities), up(sys.masses), up(sys.species), up(sys.cell))
end

Potentials.n_atoms(p::DeviceMACEPotential)::Int = p.nl.n_atoms
Potentials.cutoff(p::DeviceMACEPotential)::Float64 = Float64(p.nl.r_cut)
Potentials.stress(p::DeviceMACEPotential)::AbstractMatrix = p.result.stress
Potentials.node_energies(p::DeviceMACEPotential)::AbstractVector = p.result.node_energies

# Positions and cell must be 1-based arrays with the potential's element type and sizes, on its backend.
function _check_configuration(p::DeviceMACEPotential{T}, positions::AbstractMatrix, cell::AbstractMatrix) where {T}
    Base.require_one_based_indexing(positions, cell)
    (eltype(positions) === T && eltype(cell) === T) ||
        throw(ArgumentError(lazy"pureatomistic: positions and cell must have element type $T, got $(eltype(positions)) and $(eltype(cell))"))
    n = Potentials.n_atoms(p)
    size(positions) == (3, n) ||
        throw(DimensionMismatch(lazy"pureatomistic: positions have size $(size(positions)), expected (3, $n)"))
    size(cell) == (3, 3) || throw(DimensionMismatch(lazy"pureatomistic: the cell has size $(size(cell)), expected (3, 3)"))
    (get_backend(positions) == p.nl.backend && get_backend(cell) == p.nl.backend) ||
        throw(ArgumentError("pureatomistic: positions and cell must live on the backend of the potential"))
    return nothing
end

# Rebuilds the list on the device without reading anything back: an overflow of its capacity, and
# every other flag of the build, is acted on by the status read of `energy_forces!`.
function Potentials.prepare!(p::DeviceMACEPotential, positions::AbstractMatrix, cell::AbstractMatrix)::Nothing
    _check_configuration(p, positions, cell)
    _mlip(p.nl)._rebuild!(p.nl, positions, cell, false)
    return nothing
end

# One status read after the evaluation. An overflow empties the list, so the evaluation it read
# saw no edges; the list and workspace are rebuilt and the evaluation is redone before any flag is
# trusted. The other flags throw.
function Potentials.energy_forces!(p::DeviceMACEPotential, positions::AbstractMatrix, cell::AbstractMatrix)::Nothing
    _check_configuration(p, positions, cell)
    ext = _mlip(p.nl)
    _evaluate!(p, positions, cell)
    if _read_status!(p)[ext.STATUS_OVERFLOW] > 0
        _regrow!(p, positions, cell)
        _evaluate!(p, positions, cell)
        _read_status!(p)
    end
    ext._check_status(p.ws.status, p.model, p.numbers, p.ws, p.nl)
    return nothing
end

_evaluate!(p::DeviceMACEPotential, positions, cell) =
    _mlip(p.nl)._energy_forces_unread!(p.result, p.ws, p.model, positions, cell, p.numbers, p.nl)

# The only copy from device to host memory after construction; `p.readbacks` counts them.
function _read_status!(p::DeviceMACEPotential, host::Vector, device::AbstractVector)
    copyto!(host, device)
    p.readbacks += 1
    return host
end
_read_status!(p::DeviceMACEPotential) = _read_status!(p, p.ws.status, p.nl.status)

# The only allocating path of `energy_forces!`: a list and a workspace for the edge count that the
# overflowed build found, with the potential's spare capacity.
@noinline function _regrow!(p::DeviceMACEPotential, positions, cell)
    count = Int(p.ws.status[_mlip(p.nl).STATUS_OVERFLOW])
    p.nl = _sized_list(positions, cell, p.nl.r_cut, p.skin, p.extra_capacity_fraction, count)
    p.ws = PureMLIP.device_workspace(p.model, Potentials.n_atoms(p), p.nl.capacity)
    p.n_regrows += 1
    print(Core.stderr, "pureatomistic: DeviceMACEPotential rebuilt its neighbor list and buffers with capacity ", p.nl.capacity, " edges\n")
    return nothing
end
