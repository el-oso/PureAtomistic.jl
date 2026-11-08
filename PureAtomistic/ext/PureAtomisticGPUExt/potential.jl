# `DeviceMACEPotential`: a MACE potential whose model, neighbor list, workspace and result live on
# a KernelAbstractions backend, and `to_device`, which copies a system there.

"""
    DeviceMACEPotential{T, M, L, W, R, VI, V} <: AbstractStressPotential

A PureMLIP MACE model as a potential on a KernelAbstractions backend; build it with
[`device_potential`](@ref PureAtomistic.device_potential). It holds the device model `model`,
the device neighbor list `nl` of cutoff `r_max` with skin `skin`, the device workspace `ws`, the
device result `result` (`result.energy` is a device vector of length 1, `result.forces` a device
`3 × n` matrix), the atomic numbers `numbers` as a device `Int32` vector, the scratch of the
device reductions of the kinetic energy, and that of the device Bussi step. The arrays of a
configuration passed to it must live on its backend.

[`prepare!`](@ref) rebuilds the list on the device and reads nothing back.
[`energy_forces!`](@ref) evaluates on the device and then reads the status word of the list once.
When the configuration holds more edges than the list has room for, the list and the workspace
are built anew for the configuration with `extra_capacity_fraction` times as many edges again as
it holds (default 0.05), the evaluation is redone, and the regrowth is reported on standard error
and counted in `n_regrows`. Every copy from device to host memory that the potential makes after it
is built (each status read, each scalar read of the energy queries of a `Simulation`, and the
pressure read of the Berendsen barostat) is counted in `readbacks`: one per step, one more with
the barostat, and one more in a step that regrows, whose new list also reads its edge count
and its status word back inside `device_neighbor_list` (two copies that are not counted). The
list of `device_potential` reads back the same two. A status read also throws for the flag
that the device thermostats set in the list's status word (a gamma sampler at its cap, or a
Bussi step at zero kinetic energy). That flag is sticky: nothing clears it, so after a thermostat
step has set it, every later status read of the potential throws, and the simulation cannot
continue. The scalar reads of the energy queries and of the barostat do not read the status word
and never throw for it.

Spare edges cost evaluation time, since the per-edge kernels run over every slot: on Metal at
cu_bench(14), 10 % spare slots cost 5.6 % per evaluation and 50 % cost 23.6 %, so the default 5 %
costs about 2.8 %. In 2,000 steps of 2 fs of 500 Cu atoms near 1300 K at constant volume the edge
count stayed within 3.5 % below and 0.005 % above that of the first step, and a regrowth took 8 ms
against 68 ms for an evaluation (`bench/results/gpu_capacity_margin_metal.json`): regrowth is
cheap, and it is logged.
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
    const draws::V            # 5: R₁ and S of the two Bussi half-steps of a step, then the factor α
    const host_scalar::Vector{T}
end

function PureAtomistic.device_potential(
        model::MACEModel{T}, sys::System{T}; skin, extra_capacity_fraction = 0.05
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
    nl = PureMLIP.device_neighbor_list(sys.positions, sys.cell, model.r_max; skin, extra_capacity_fraction = fraction)
    ws = PureMLIP.device_workspace(dm, n, nl.capacity)
    res = PureMLIP.device_result(dm, n)
    z(k) = KernelAbstractions.zeros(backend, T, k)
    kinetic = z(n)
    return DeviceMACEPotential{T, typeof(dm), typeof(nl), typeof(ws), typeof(res), typeof(numbers), typeof(kinetic)}(
        dm, numbers, Float64(skin), fraction, nl, ws, res, 0, 0, kinetic, z(1), z(1), z(5), zeros(T, 1)
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
    PureMLIP.device_rebuild!(p.nl, positions, cell)
    return nothing
end

# One status read after the evaluation. An overflow empties the list, so the evaluation it read
# saw no edges; the list and workspace are rebuilt and the evaluation is redone before any result
# is used. The other flags throw.
function Potentials.energy_forces!(p::DeviceMACEPotential, positions::AbstractMatrix, cell::AbstractMatrix)::Nothing
    _check_configuration(p, positions, cell)
    _evaluate!(p, positions, cell)
    _read_status!(p)
    if PureMLIP.check_device_status(p.ws, p.model, p.numbers, p.nl) > 0
        _regrow!(p, positions, cell)
        _evaluate!(p, positions, cell)
        _read_status!(p)
        # A list built for this configuration holds its edges; an overflow here is a defect.
        m = PureMLIP.check_device_status(p.ws, p.model, p.numbers, p.nl)
        iszero(m) || throw(ArgumentError(lazy"pureatomistic: the rebuilt neighbor list overflowed its capacity with $m edges"))
    end
    return nothing
end

_evaluate!(p::DeviceMACEPotential, positions, cell) =
    PureMLIP.device_energy_forces!(p.result, p.ws, p.model, positions, cell, p.numbers, p.nl)

# Every copy from device to host memory after construction goes through `_read_status!`, which
# counts it in `p.readbacks`: the status word of the list, or a scalar of the energy queries or of
# the barostat.
function _read_status!(p::DeviceMACEPotential)
    status = PureMLIP.read_device_status!(p.ws, p.nl)
    p.readbacks += 1
    _check_thermostat_flag(status)
    return nothing
end

# A scalar read does not look at the status word, so it never throws for a flag.
function _read_status!(p::DeviceMACEPotential, host::Vector, device::AbstractVector)
    copyto!(host, device)
    p.readbacks += 1
    return host
end

# The only allocating path of `energy_forces!`: a list built for the configuration (which reads
# the edge count and then the status word back, two copies that `readbacks` does not count) and a
# workspace for its capacity. The old list and workspace stay allocated until the garbage
# collector frees them, so a regrowth briefly holds both.
@noinline function _regrow!(p::DeviceMACEPotential, positions, cell)
    p.nl = PureMLIP.device_neighbor_list(positions, cell, p.nl.r_cut; skin = p.skin, extra_capacity_fraction = p.extra_capacity_fraction)
    p.ws = PureMLIP.device_workspace(p.model, Potentials.n_atoms(p), p.nl.capacity)
    p.n_regrows += 1
    print(Core.stderr, "pureatomistic: DeviceMACEPotential rebuilt its neighbor list and buffers with capacity ", p.nl.capacity, " edges\n")
    return nothing
end
