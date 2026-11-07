# `mace_energy!` on the device: the forward pass of `mace_energy!` with every stage on the backend
# of the model, and one read of the status word at the end.

# Out of line, for the reason of the core's `_check_inputs`: the cutoff comparison widens a Float32
# radius to the list's cutoff type.
@noinline function _check_evaluation(
        res::DeviceMACEResult, ws::DeviceMACEWorkspace, model::DeviceMACEModel, positions, cell, atomic_numbers, nl::DeviceNeighborList
    )
    _require_one_based(positions, cell, atomic_numbers)
    _common_backend(positions, cell, atomic_numbers, ws.r, nl.senders, res.energy, model.e0)
    n = nl.n_atoms
    length(atomic_numbers) == n || throw(DimensionMismatch(lazy"gpu: $(length(atomic_numbers)) atomic numbers for $n atoms"))
    length(res.node_energies) == n ||
        throw(DimensionMismatch(lazy"gpu: the result holds $(length(res.node_energies)) atoms, the neighbor list has $n"))
    ws.n_atoms >= n || throw(ArgumentError(lazy"gpu: the workspace holds $(ws.n_atoms) atoms, the neighbor list has $n"))
    rc, rm = nl.r_cut, model.host.r_max
    rc >= rm || throw(ArgumentError(lazy"gpu: the neighbor list cutoff $rc is below r_max $rm"))
    return nothing
end

"""
    mace_energy!(res::DeviceMACEResult, ws::DeviceMACEWorkspace, model::DeviceMACEModel, positions, cell, atomic_numbers, nl::DeviceNeighborList) -> res

[`mace_energy!`](@ref) on the backend of `model`: `positions`, `cell` and `atomic_numbers` are
arrays on that backend, and `res.energy[1]` and `res.node_energies` are filled. The per-edge stages
run in chunks of `ws.edge_chunk` edge slots, the node stages over every atom, and the energy is a
fixed-shape tree sum of the node energies, so an evaluation repeats bit for bit. Afterwards the
host reads the status word of `nl` once and throws if an atomic number is not among the model's
`elements` or two atoms are at one place. After a throw the contents of `res` and of the
workspace are undefined.
"""
function PureMLIP.mace_energy!(
        res::DeviceMACEResult{T}, ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, positions::AbstractMatrix,
        cell::AbstractMatrix, atomic_numbers::AbstractVector{<:Integer}, nl::DeviceNeighborList
    ) where {T}
    _energy_unread!(res, ws, model, positions, cell, atomic_numbers, nl)
    PureMLIP.read_device_status!(ws, nl)
    _check_status(ws, model, atomic_numbers, nl)
    return res
end

# The device `mace_energy!` without its read of the status word, the forward pass of
# `device_energy_forces!`: `res` is valid only once `read_device_status!` has read the status word
# afterwards and `check_device_status` has returned 0.
function _energy_unread!(
        res::DeviceMACEResult{T}, ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, positions::AbstractMatrix,
        cell::AbstractMatrix, atomic_numbers::AbstractVector{<:Integer}, nl::DeviceNeighborList
    ) where {T}
    _check_evaluation(res, ws, model, positions, cell, atomic_numbers, nl)
    n = nl.n_atoms
    numbers = view(ws.numbers, 1:n)
    _prepare!(ws, model, positions, cell, atomic_numbers, nl)
    _interaction!(ws, model, 1, nl, numbers, model.tables[1])
    _first_nodes!(ws, model, n)
    _interaction!(ws, model, 2, nl, numbers, model.tables[2])
    _second_nodes!(res, ws, model, n)
    return res
end

# The species of each atom (checked against the model's elements), the edge vectors and lengths,
# and the embedding with the sender features of interaction 0.
function _prepare!(ws::DeviceMACEWorkspace, model::DeviceMACEModel, positions, cell, atomic_numbers, nl::DeviceNeighborList)
    backend, m = ws.backend, model.host
    n = nl.n_atoms
    fill!(view(nl.status, STATUS_UNKNOWN_ELEMENT:STATUS_ZERO_LENGTH), zero(Int32))
    z1 = model.elements[1]
    _launch!(_species_kernel!, backend, n, ws.species, ws.numbers, nl.status, atomic_numbers, model.species, Int32(z1), Int32(m.species[z1 + 1]))
    _edges!(ws, positions, cell, nl)
    _launch!(_embedding_kernel!, backend, (m.C, n), ws.h0, model.embedding, ws.species)
    _linear!(ws.x[1], model.up[1], ws.h0)
    return nothing
end

# Product 0 and the inputs of interaction 1, from the messages of interaction 0.
function _first_nodes!(ws::DeviceMACEWorkspace, model::DeviceMACEModel, n::Int)
    backend, m = ws.backend, model.host
    C = m.C
    list = model.contraction[1].species
    _linear!(ws.L0, model.linear[1], ws.M[1])
    _launch!(_normalize_kernel!, backend, (C, n, size(ws.L0, 3)), ws.L0, view(ws.density, :, 1))
    _species_linear!(ws.X1, model.skip[1], ws.L0, ws.species, list, ws.dX1, backend)
    Zf = _contraction_buffer(ws.Zf, model.contraction[1], 1, ws.contraction_block, C)
    Yc = _contraction_buffer(ws.Yc, model.contraction[1], 2, ws.contraction_block, C)
    _symmetric_contraction!(ws.B[1], model.contraction[1], ws.X1, ws.species, n, Zf, Yc, backend)
    _linear!(ws.h1, model.product_linear[1], ws.B[1])
    K = m.skip[2].K
    _species_linear!(ws.sc, model.skip[2], view(ws.h1, :, :, 1:K), ws.species, list, view(ws.dX1, :, :, 1:K), backend)
    _linear!(ws.x[2], model.up[2], ws.h1)
    _linear!(ws.e1, model.readout0, ws.h1)
    return nothing
end

# Product 1 with the skip connection, the readout, the energy of each atom and their sum, from the
# messages of interaction 1.
function _second_nodes!(res::DeviceMACEResult, ws::DeviceMACEWorkspace, model::DeviceMACEModel, n::Int)
    backend, m = ws.backend, model.host
    C = m.C
    _linear!(ws.X2, model.linear[2], ws.M[2])
    _launch!(_normalize_kernel!, backend, (C, n, size(ws.X2, 3)), ws.X2, view(ws.density, :, 2))
    Zf = _contraction_buffer(ws.Zf, model.contraction[2], 1, ws.contraction_block, C)
    Yc = _contraction_buffer(ws.Yc, model.contraction[2], 2, ws.contraction_block, C)
    _symmetric_contraction!(ws.B[2], model.contraction[2], ws.X2, ws.species, n, Zf, Yc, backend)
    _linear!(ws.h2, model.product_linear[2], ws.B[2])
    _launch!(_add_kernel!, backend, length(ws.h2), ws.h2, ws.sc)
    _linear!(ws.gpre, model.readout1[1], ws.h2)
    _launch!(_readout_silu_kernel!, backend, length(ws.g), ws.g, ws.gpre, m.readout_cst)
    _linear!(ws.e2, model.readout1[2], ws.g)
    _launch!(_node_energy_kernel!, backend, n, res.node_energies, model.e0, ws.species, ws.pair, ws.e1, ws.e2, m.scale, m.shift)
    _sum!(res.energy, res.node_energies, n, backend)
    return nothing
end

# The per-edge pass of interaction `k` over the edge chunks in order: radial features and
# harmonics, the radial MLP and the messages; the second interaction also sums the ZBL pair terms.
function _interaction!(ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, k::Int, nl::DeviceNeighborList, numbers, table::Val) where {T}
    fill!(ws.M[k], zero(T))
    fill!(view(ws.density, :, k), zero(T))
    k == 2 && fill!(ws.pair, zero(T))
    for c in 1:_n_edge_chunks(nl, ws.edge_chunk)
        chunk = _edge_chunk(nl, numbers, ws.edge_chunk, c)
        _edge_features!(ws, model, k, chunk)
        _radial_mlp!(ws, model, k, chunk)
        _messages!(ws, model, k, chunk, table)
        k == 2 && _zbl_edges!(ws, model, chunk)
    end
    return nothing
end

# Buffer `which` (1: features, 2: products) of the contraction `dcm`, as a features × block ×
# channels array over the front of the workspace's flat buffer `flat`.
function _contraction_buffer(flat::AbstractVector, dcm::DeviceContractionMatrices, which::Int, block::Int, C::Int)
    rows = size(dcm.F, which)
    return reshape(view(flat, 1:(rows * block * C)), rows, block, C)
end

@kernel function _species_kernel!(species, numbers, status, @Const(atomic_numbers), @Const(table), z1, s1)
    i = @index(Global)
    _species!(species, numbers, status, atomic_numbers, table, z1, s1, i)
end

# The species index of atom `i` from `table[z + 1]` (0 for an element the device model does not
# hold), and its atomic number. An unknown element sets the status flag and takes the first
# element `z1` (species `s1`), so that every later kernel indexes within its tables; the host
# throws before the result is used.
@inline function _species!(species, numbers, status, atomic_numbers, table, z1, s1, i)
    z = atomic_numbers[i]
    s = 0 <= z < length(table) ? table[z + 1] : zero(eltype(table))
    if iszero(s)
        status[STATUS_UNKNOWN_ELEMENT] = one(eltype(status))
        species[i], numbers[i] = s1, z1
    else
        species[i], numbers[i] = s, z
    end
    return nothing
end

PureMLIP.read_device_status!(ws::DeviceMACEWorkspace, nl::DeviceNeighborList) = copyto!(ws.status, nl.status)

# Throws for every flag that a build of `nl` and an evaluation set, the overflow of the capacity
# included.
function _check_status(ws::DeviceMACEWorkspace, model::DeviceMACEModel, atomic_numbers, nl::DeviceNeighborList)
    m = PureMLIP.check_device_status(ws, model, atomic_numbers, nl)
    m > 0 && throw(ArgumentError(lazy"gpu: $m edges exceed the capacity $(nl.capacity)"))
    return nothing
end

# The arrays are copied to the host only to name the culprit of a flag.
function PureMLIP.check_device_status(ws::DeviceMACEWorkspace, model::DeviceMACEModel, atomic_numbers, nl::DeviceNeighborList)::Int
    status = ws.status
    _check_list_flags(status)
    m = Int(status[STATUS_OVERFLOW])
    m > 0 && return m
    if !iszero(status[STATUS_UNKNOWN_ELEMENT])
        z = first(z for z in Array(atomic_numbers) if !(z in model.elements))
        throw(ArgumentError(lazy"gpu: atomic number $z is not among the device model's elements $(model.elements)"))
    end
    if !iszero(status[STATUS_ZERO_LENGTH])
        e = findfirst(iszero, Array(view(ws.d, 1:nl.capacity)))
        s, q = Array(nl.senders)[e], Array(nl.receivers)[e]
        throw(ArgumentError(lazy"gpu: edge $e from atom $s to atom $q has length zero; the two atoms are at one place"))
    end
    return 0
end
