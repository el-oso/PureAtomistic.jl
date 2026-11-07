# The device model, neighbor list, workspace and result, and the entry points that build them.

"""
    DeviceMACEModel{T, B, V, M, A3, A4, VI, IM, TB}

A `MACEModel{T}` with every parameter array copied to the backend `backend` (the array types are
`V`, `M`, `A3` and `A4` by rank, `VI` and `IM` for `Int32` vectors and matrices). `host` is the
model it was built from, for its sizes and scalars. The device model evaluates only the atomic
numbers in `elements`: `species[z + 1]` is the model's species index of `z` when `z` is in
`elements`, and 0 otherwise. The contraction matrices of both product blocks are built on the
host for these species and uploaded. `tables` holds the coupling table of each convolution as a `Val`
(`_tp_table`), so that the kernels specialized on it dispatch statically. Build it with `device_model`.
"""
struct DeviceMACEModel{T, B <: Backend, V, M, A3, A4, VI, IM, TB <: NTuple{2, Val}}
    backend::B
    host::MACEModel{T}
    elements::Vector{Int}
    species::VI
    e0::V
    embedding::M
    bessel_weights::V
    radial_radii::V                   # radial_embedding covalent radii, indexed by atomic number + 1
    zbl_radii::V                      # pair repulsion covalent radii, indexed by atomic number + 1
    up::NTuple{2, DeviceE3nnLinear{T, A3, V}}
    mlp::NTuple{2, NTuple{4, M}}
    density::NTuple{2, V}
    conv::NTuple{2, DeviceChannelwiseTP{T, IM, V}}
    linear::NTuple{2, DeviceE3nnLinear{T, A3, V}}
    skip::NTuple{2, DeviceSpeciesLinear{T, A4, V}}
    product::NTuple{2, DeviceSymmetricContraction{T, IM, V, A3}}
    contraction::NTuple{2, DeviceContractionMatrices{T, A4, IM}}
    product_linear::NTuple{2, DeviceE3nnLinear{T, A3, V}}
    readout0::DeviceE3nnLinear{T, A3, V}
    readout1::NTuple{2, DeviceE3nnLinear{T, A3, V}}
    tables::TB
end

function PureMLIP.device_model(model::MACEModel{T}, backend::Backend; elements) where {T <: AbstractFloat}
    _check_precision(backend, T)
    isempty(elements) && throw(ArgumentError("gpu: elements must name at least one atomic number"))
    species = Int[]
    for z in elements
        s = 0 <= z < length(model.species) ? model.species[z + 1] : 0
        s > 0 || throw(ArgumentError(lazy"gpu: element $z is not in the model's element table"))
        s in species || push!(species, s)
    end
    zs = Int[z for z in elements]
    map_host = Int32[s in species ? s : 0 for s in model.species]
    up(A) = _upload(backend, A)
    lin(l) = DeviceE3nnLinear(backend, l)
    e0 = up(model.e0)
    emb = up(model.embedding)
    cm = ntuple(k -> DeviceContractionMatrices(backend, model.product[k], species), Val(2))
    sp = up(map_host)
    up_lin = map(lin, model.up)
    tables = map(_tp_table, model.conv)
    return DeviceMACEModel{T, typeof(backend), typeof(e0), typeof(emb), typeof(up_lin[1].W), typeof(cm[1].F), typeof(sp), typeof(cm[1].pairs), typeof(tables)}(
        backend, model, unique(zs), sp, e0, emb,
        up(model.radial.bessel_weights), up(model.radial.covalent_radii), up(model.zbl.covalent_radii),
        up_lin, map(m -> map(up, m.W), model.mlp), map(up, model.density),
        map(tp -> DeviceChannelwiseTP(backend, tp), model.conv), map(lin, model.linear),
        map(sl -> DeviceSpeciesLinear(backend, sl), model.skip),
        map(sc -> DeviceSymmetricContraction(backend, sc), model.product), cm,
        map(lin, model.product_linear), lin(model.readout0), map(lin, model.readout1),
        tables,
    )
end

# The entries of the status word of a device neighbor list.
const STATUS_REBUILT = 1
const STATUS_OVERFLOW = 2
const STATUS_SAMPLER_CAP = 3
const STATUS_UNKNOWN_ELEMENT = 4
const STATUS_ZERO_LENGTH = 5
const STATUS_LENGTH = 5

"""
    DeviceNeighborList{T, B, VI, MI}

A neighbor list on the backend `backend`, with the edges of `NeighborList` sorted by receiver and
stored as `Int32`: edge `e` runs from `senders[e]` to `receivers[e]` with lattice translation
`shifts[:, e]`, and the edges of receiver `i` are `row_ptr[i]:row_ptr[i + 1] - 1`. `reverse[e]` is
the edge from `receivers[e]` to `senders[e]` with translation `-shifts[:, e]`. The arrays have
`capacity` columns; `n_edges[1]` of them are valid. The edges are those within `r_cut + skin`.
`status` holds five flags written on the device, at the indices named by the `STATUS_*`
constants: rebuilt, capacity overflow, sampler cap reached, unknown element and zero-length edge.
"""
struct DeviceNeighborList{T, B <: Backend, VI <: AbstractVector{Int32}, MI <: AbstractMatrix{Int32}}
    backend::B
    r_cut::T
    skin::T
    n_atoms::Int
    capacity::Int
    row_ptr::VI
    receivers::VI
    senders::VI
    shifts::MI
    reverse::VI
    n_edges::VI
    status::VI
end

# The receiver CSR of the first `n_edges` edges of `nl` and the index of each edge's reverse.
function _csr_and_reverse(nl::NeighborList)
    n, E = nl.n_atoms, nl.n_edges
    receivers = view(nl.receivers, 1:E)
    issorted(receivers) || throw(ArgumentError("gpu: neighbor list edges must be sorted by receiver"))
    row_ptr = zeros(Int32, n + 1)
    row_ptr[1] = 1
    for r in receivers
        1 <= r <= n || throw(ArgumentError(lazy"gpu: neighbor list receivers must lie in 1:$n"))
        row_ptr[r + 1] += 1
    end
    for i in 1:n
        row_ptr[i + 1] += row_ptr[i]
    end
    key(e) = (nl.receivers[e], nl.senders[e], nl.shifts[1, e], nl.shifts[2, e], nl.shifts[3, e])
    index = Dict{NTuple{5, Int}, Int}(key(e) => e for e in 1:E)
    reverse = zeros(Int32, E)
    for e in 1:E
        rev = get(index, (nl.senders[e], nl.receivers[e], -nl.shifts[1, e], -nl.shifts[2, e], -nl.shifts[3, e]), 0)
        rev > 0 || throw(ArgumentError(lazy"gpu: neighbor list edge $e has no reverse edge"))
        reverse[e] = rev
    end
    return row_ptr, reverse
end

function _device_neighbor_list(nl::NeighborList{T}, backend::Backend, r_cut::T, skin::T) where {T}
    row_ptr, reverse = _csr_and_reverse(nl)
    E, capacity = nl.n_edges, length(nl.senders)
    padded(v) = (w = zeros(Int32, capacity); w[1:E] = view(v, 1:E); w)
    up(A) = _upload(backend, A)
    return DeviceNeighborList(
        backend, r_cut, skin, nl.n_atoms, capacity, up(row_ptr), up(padded(nl.receivers)), up(padded(nl.senders)),
        up(Int32.(nl.shifts)), up(padded(reverse)), up(Int32[E]), _device_zeros(backend, Int32, (STATUS_LENGTH,)),
    )
end

PureMLIP.device_neighbor_list(nl::NeighborList{T}, backend::Backend) where {T <: AbstractFloat} =
    _device_neighbor_list(nl, backend, T(nl.r_cut), zero(T))

# The edges within `r_cut + skin` are found on the host from copies of `positions` and `cell`.
# Each call copies the arrays to the host and back, so it is not for use inside an MD step.
function PureMLIP.device_neighbor_list(
        positions::AbstractMatrix, cell::AbstractMatrix, r_cut::Real; skin::Real = 0, extra_capacity::Integer = 0
    )
    _require_one_based(positions, cell)
    backend = _common_backend(positions, cell)
    T = eltype(positions)
    eltype(cell) === T || throw(ArgumentError(lazy"gpu: positions and cell must share one element type, got $T and $(eltype(cell))"))
    T <: AbstractFloat || throw(ArgumentError(lazy"gpu: positions must hold floating-point numbers, got $T"))
    skin >= 0 || throw(ArgumentError(lazy"gpu: the skin must not be negative, got $skin"))
    nl = neighbor_list(Array(positions), Array(cell), r_cut + skin; extra_capacity)
    return _device_neighbor_list(nl, backend, T(r_cut), T(skin))
end

"""
    DeviceMACEWorkspace{T, B, V, M, A3, VI}

The buffers of an evaluation of a `DeviceMACEModel` on up to `n_atoms` atoms and `n_edges`
directed edges, on the backend `backend`. Per edge it holds the edge vector `r`, its length `d`
and the energy gradient `dEdr`. The harmonics `Y` and their gradients `dY`, the radial features
and their length derivatives, the radial-MLP outputs and their cotangents, its hidden layers and
the ZBL terms are held for `edge_chunk` edges at a time; a chunk is a range of `edge_chunk`
consecutive edge slots of the neighbor list (fewer in the last), and a receiver's edges may span
two chunks. The per-node buffers hold the quantities of the `MACEWorkspace` buffers of the same
names, in a `(C, n, K)` layout (channel, atom, e3nn component) in which each slot of degree `l` is
one contiguous `C × n(2l + 1)` matrix for `mul!`; the summed messages `M`, their cotangents `dM`
and the first linear output `L0` are held for every atom. They share memory in the pools of
`MACEWorkspace`. The symmetric contraction builds its features `Zf` and matrix products `Yc` for
`contraction_block` atoms at a time. `species` and `numbers` hold each atom's species index and
atomic number as the species kernel checked them, and `status` is the host copy of the neighbor
list's status word that an evaluation reads back.
"""
struct DeviceMACEWorkspace{T, B <: Backend, V <: AbstractVector{T}, M <: AbstractMatrix{T}, A3 <: AbstractArray{T, 3}, VI <: AbstractVector{Int32}}
    backend::B
    n_atoms::Int
    n_edges::Int
    edge_chunk::Int
    contraction_block::Int
    species::VI
    numbers::VI
    status::Vector{Int32}
    Zf::V                    # features × contraction_block × C, for the largest feature count
    Yc::V                    # columns × contraction_block × C, for the largest column count
    # per edge
    r::M                     # 3 × E
    d::V
    dEdr::M                  # 3 × E
    # per edge of one edge chunk
    Y::M                     # 16 × edge_chunk
    dY::A3                   # 16 × 3 × edge_chunk, ∂Y/∂r
    feats::M                 # n_basis × edge_chunk
    dfeats::M
    tpw::M                   # W × edge_chunk
    tpw3::A3                 # C × W/C × edge_chunk, the same memory as tpw
    pre::A3                  # hidden × edge_chunk × 3
    act::A3
    dtpw::M
    dtpw3::A3                # the same memory as dtpw
    mlp_scratch::A3          # hidden × edge_chunk × 2
    dfeats_chunk::M
    zbl::V                   # edge energies
    dzbl::V                  # and their length derivatives
    # per node; pools as in MACEWorkspace, plus `messages` (M[1], M[2], then dM[2], dM[1]) and
    # `linear` (L0, then dL0)
    h0::A3                   # pool `small`
    x::NTuple{2, A3}
    density::M               # n × 2
    M::NTuple{2, A3}         # pool `messages`
    dM::NTuple{2, A3}        # pool `messages`
    L0::A3                   # pool `linear`
    dL0::A3                  # pool `linear`
    X1::A3
    X2::A3
    B::NTuple{2, A3}         # pool `small`
    h1::A3                   # pool `small`
    sc::A3                   # pool `small`
    h2::A3                   # pool `small`
    gpre::A3
    g::A3
    e1::A3
    e2::A3
    pair::V
    de1::A3
    de2::A3
    dg::A3
    dh2::A3                  # pool `small`
    dh1::A3                  # pool `small`
    dh1_part::A3             # pool `small`
    dB::NTuple{2, A3}        # pool `small`
    dX1::A3                  # pool `mid`
    dX2::A3                  # pool `mid`
    dx::A3                   # ∂E/∂x[2] per sender atom; pool `dx`
    dx0::A3                  # ∂E/∂x[1], never written; pool `dx`
    ddensity::M              # n × 2
end

# The default `edge_chunk` of `device_workspace`: the smallest edge chunk whose per-edge pass on
# cu_bench(14) is within 5 % of the fastest and whose edge-chunk buffers take at most 1/8 of the
# device memory (bench/results/gpu_stage_profile_metal.json).
const DEFAULT_EDGE_CHUNK = 2^14

# The default `contraction_block` of `device_workspace`, in atoms. On cu_bench(14) (Float32, Metal)
# the forward pass takes 360 ms with blocks of 4096 atoms, 356 ms with one block of every atom (1 %
# less) and 393 ms with blocks of 1024 atoms (9 % more); one block of every atom takes 1.2 GB of
# contraction buffers in place of 455 MB (bench/results/gpu_stage_profile_metal.json).
const DEFAULT_CONTRACTION_BLOCK = 4096

function PureMLIP.device_workspace(
        model::DeviceMACEModel{T}, n_atoms::Integer, n_edges::Integer; edge_chunk::Integer = DEFAULT_EDGE_CHUNK,
        contraction_block::Integer = DEFAULT_CONTRACTION_BLOCK
    ) where {T}
    (n_atoms >= 0 && n_edges >= 0) ||
        throw(ArgumentError(lazy"gpu: workspace sizes must not be negative, got $n_atoms atoms and $n_edges edges"))
    edge_chunk > 0 || throw(ArgumentError(lazy"gpu: the edge chunk must be positive, got $edge_chunk"))
    contraction_block > 0 || throw(ArgumentError(lazy"gpu: the contraction block must be positive, got $contraction_block"))
    backend, m = model.backend, model.host
    n, E = Int(n_atoms), Int(n_edges)
    chunk = min(Int(edge_chunk), max(E, 1))
    cblock = min(Int(contraction_block), max(n, 1))
    C = m.C
    nb = length(m.radial.bessel_weights)
    W = max(size(m.mlp[1].W[4], 2), size(m.mlp[2].W[4], 2))
    H = max(size(m.mlp[1].W[1], 2), size(m.mlp[2].W[1], 2))
    nf = maximum(cm -> size(cm.F, 1), model.contraction)
    ncol = maximum(cm -> size(cm.F, 2), model.contraction)
    z(dims...) = _device_zeros(backend, T, dims)
    dims(lin::E3nnLinear) = (lin.C_out, n, lin.K_out)
    node(lin::E3nnLinear) = z(dims(lin)...)
    products(k) = (C, n, sum(ct -> 2ct.l + 1, m.product[k].contractions))
    messages(k) = (C, n, m.conv[k].K_out)
    ((dX2,), (dX1,)) = _device_pool(backend, T, ((dims(m.linear[2]),), ((C, n, m.skip[1].K),)))
    ((h0, h1, B0, B1, sc, h2), (dh2, dh1, dh1_part, dB0, dB1)) = _device_pool(
        backend, T, (
            ((C, n, 1), dims(m.product_linear[1]), products(1), products(2), (C, n, m.skip[2].K), dims(m.product_linear[2])),
            (dims(m.product_linear[2]), dims(m.product_linear[1]), dims(m.product_linear[1]), products(1), products(2)),
        )
    )
    ((dx,), (dx0,)) = _device_pool(backend, T, ((dims(m.up[2]),), (dims(m.up[1]),)))
    ((M0,), (M1,), (dM1,), (dM0,)) = _device_pool(backend, T, ((messages(1),), (messages(2),), (messages(2),), (messages(1),)))
    ((L0,), (dL0,)) = _device_pool(backend, T, ((dims(m.linear[1]),), (dims(m.linear[1]),)))
    tpw, dtpw = z(W, chunk), z(W, chunk)
    r = z(3, E)
    return DeviceMACEWorkspace{T, typeof(backend), typeof(z(0)), typeof(r), typeof(h0), typeof(model.species)}(
        backend, n, E, chunk, cblock, _device_zeros(backend, Int32, (n,)), _device_zeros(backend, Int32, (n,)),
        zeros(Int32, STATUS_LENGTH), z(nf * cblock * C), z(ncol * cblock * C),
        r, z(E), z(3, E),
        z(16, chunk), z(16, 3, chunk), z(nb, chunk), z(nb, chunk), tpw, reshape(tpw, C, W ÷ C, chunk),
        z(H, chunk, 3), z(H, chunk, 3), dtpw, reshape(dtpw, C, W ÷ C, chunk), z(H, chunk, 2), z(nb, chunk), z(chunk), z(chunk),
        h0, (node(m.up[1]), node(m.up[2])), z(n, 2), (M0, M1), (dM0, dM1), L0, dL0,
        z(C, n, m.skip[1].K), node(m.linear[2]), (B0, B1), h1, sc, h2,
        node(m.readout1[1]), node(m.readout1[1]), node(m.readout0), node(m.readout1[2]), z(n),
        node(m.readout0), node(m.readout1[2]), node(m.readout1[1]), dh2, dh1, dh1_part, (dB0, dB1),
        dX1, dX2, dx, dx0, z(n, 2),
    )
end

"""
    DeviceMACEResult{T, B, V, M}

The outputs of a device evaluation, on the backend `backend`: `energy` (a vector of length 1,
eV), `node_energies` (eV per atom), `forces` (`3 × n_atoms`, eV/Å), `virial` and `stress`
(`3 × 3`).
"""
struct DeviceMACEResult{T, B <: Backend, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    backend::B
    energy::V
    node_energies::V
    forces::M
    virial::M
    stress::M
end

function PureMLIP.device_result(model::DeviceMACEModel{T}, n_atoms::Integer) where {T}
    n_atoms >= 0 || throw(ArgumentError(lazy"gpu: the number of atoms must not be negative, got $n_atoms"))
    backend = model.backend
    z(dims...) = _device_zeros(backend, T, dims)
    return DeviceMACEResult(backend, z(1), z(Int(n_atoms)), z(3, Int(n_atoms)), z(3, 3), z(3, 3))
end
