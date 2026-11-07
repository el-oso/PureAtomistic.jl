# The per-edge geometry on the device: edge vectors and lengths of every edge, the edge chunks that
# the per-edge buffers hold one at a time, and e3nn's spherical harmonics with their gradient.

"""
    EdgeChunk{L, VZ}

The edge slots `edges` of the device neighbor list `nl`, a range of consecutive slots that the
workspace's per-edge-chunk buffers hold: column `j` holds slot `first(edges) + j - 1`. Slots past
`nl.n_edges[1]` hold no edge; the kernels read that count on the device and give such a column zero
features. A receiver's edges may lie in two or more chunks. `atomic_numbers` are those of every atom
of the list, on its backend.
"""
struct EdgeChunk{L <: DeviceNeighborList, VZ <: AbstractVector{<:Integer}}
    nl::L
    atomic_numbers::VZ
    edges::UnitRange{Int}
end

# The number of edge chunks of `edge_chunk` slots that cover the slots of `nl`.
_n_edge_chunks(nl::DeviceNeighborList, edge_chunk::Int) = cld(nl.capacity, edge_chunk)

# Chunk `c` of the chunks of `edge_chunk` slots of `nl`.
_edge_chunk(nl::DeviceNeighborList, atomic_numbers::AbstractVector{<:Integer}, edge_chunk::Int, c::Int) =
    EdgeChunk(nl, atomic_numbers, ((c - 1) * edge_chunk + 1):min(c * edge_chunk, nl.capacity))

"""
    _edge_chunks(nl, atomic_numbers, edge_chunk) -> Vector{EdgeChunk}

The chunks of `edge_chunk` consecutive edge slots that cover the `nl.capacity` slots of `nl`. They
depend only on sizes known on the host, so building them reads nothing from the device.
"""
function _edge_chunks(nl::DeviceNeighborList, atomic_numbers::AbstractVector{<:Integer}, edge_chunk::Integer)
    _require_one_based(atomic_numbers)
    _common_backend(nl.senders, atomic_numbers)
    edge_chunk > 0 || throw(ArgumentError(lazy"gpu: the edge chunk must be positive, got $edge_chunk"))
    length(atomic_numbers) == nl.n_atoms ||
        throw(DimensionMismatch(lazy"gpu: $(length(atomic_numbers)) atomic numbers for $(nl.n_atoms) atoms"))
    return [_edge_chunk(nl, atomic_numbers, Int(edge_chunk), c) for c in 1:_n_edge_chunks(nl, Int(edge_chunk))]
end

# The number of edge slots of `chunk`, after checking that the chunk fits the workspace.
function _chunk_length(ws::DeviceMACEWorkspace, chunk::EdgeChunk)
    _common_backend(ws.r, chunk.nl.senders, chunk.atomic_numbers)
    nb = length(chunk.edges)
    nb <= ws.edge_chunk || throw(ArgumentError(lazy"gpu: the chunk has $nb edges, the workspace holds $(ws.edge_chunk)"))
    last(chunk.edges) <= min(ws.n_edges, chunk.nl.capacity) ||
        throw(ArgumentError(lazy"gpu: the chunk reaches edge $(last(chunk.edges)), the workspace holds $(ws.n_edges) and the list $(chunk.nl.capacity)"))
    chunk.nl.n_atoms <= ws.n_atoms || throw(ArgumentError(lazy"gpu: the list has $(chunk.nl.n_atoms) atoms, the workspace holds $(ws.n_atoms)"))
    return nb
end

# The number of receivers that the per-receiver kernels of a chunk of `nb` slots launch work items
# for: about the receivers of `nb` edges at the list's mean degree per slot. Each work item handles
# every `R`-th receiver of the chunk's receiver range, so any `R ≥ 1` covers the range.
_chunk_receivers(nl::DeviceNeighborList, nb::Int) = max(1, min(nl.n_atoms, cld(nb * nl.n_atoms, max(nl.capacity, 1)) + 2))

# The first and last receiver with an edge in slots `e0 + 1 : e0 + nb`, with `n_edges` valid
# slots; `(1, 0)` when the slots hold no edge.
@inline function _receiver_range(receivers, n_edges, e0, nb)
    stop = min(e0 + nb, Int(n_edges))
    stop >= e0 + 1 || return 1, 0
    return Int(receivers[e0 + 1]), Int(receivers[stop])
end

# The edges of receiver `i` among slots `e0 + 1 : e0 + nb`.
@inline _receiver_edges(row_ptr, i, e0, nb) = max(Int(row_ptr[i]), e0 + 1):min(Int(row_ptr[i + 1]) - 1, e0 + nb)

"""
    _edges!(ws, positions, cell, nl) -> nothing

Edge vectors `ws.r[:, e] = positions[:, receiver] - positions[:, sender] + cell · shift` and their
lengths `ws.d[e]` for the first `nl.n_edges[1]` edges, read on the device. The workspace holds
every edge the list can hold. An edge of length zero (two atoms at one place) sets
`nl.status[STATUS_ZERO_LENGTH]`; its harmonics are undefined.
"""
function _edges!(ws::DeviceMACEWorkspace{T}, positions::AbstractMatrix, cell::AbstractMatrix, nl::DeviceNeighborList) where {T}
    _require_one_based(positions, cell)
    _common_backend(positions, cell, ws.r, nl.senders)
    (eltype(positions) === T && eltype(cell) === T) ||
        throw(ArgumentError(lazy"gpu: positions and cell must have the workspace's element type $T, got $(eltype(positions)) and $(eltype(cell))"))
    size(positions) == (3, nl.n_atoms) ||
        throw(DimensionMismatch(lazy"gpu: positions have size $(size(positions)), the neighbor list has $(nl.n_atoms) atoms"))
    size(cell) == (3, 3) || throw(DimensionMismatch(lazy"gpu: the cell has size $(size(cell)), expected (3, 3)"))
    ws.n_edges >= nl.capacity ||
        throw(ArgumentError(lazy"gpu: the workspace holds $(ws.n_edges) edges, the neighbor list up to $(nl.capacity)"))
    _launch!(_edges_kernel!, ws.backend, nl.capacity, ws.r, ws.d, nl.status, positions, cell, nl.senders, nl.receivers, nl.shifts, nl.n_edges)
    return nothing
end

# Convention for every kernel of the extension: the kernel body only indexes and calls one
# `@inline` function that holds the arithmetic. A kernel compiles through the backend's own
# compiler, which the Float32 conversion sweep does not read; the tests check the `@inline`
# functions on the host instead, so arithmetic written in a kernel body would go unchecked.
#
# One work item per edge slot of the list; slots past `n_edges[1]` hold no edge.
@kernel function _edges_kernel!(r, d, status, @Const(positions), @Const(cell), @Const(senders), @Const(receivers), @Const(shifts), @Const(n_edges))
    e = @index(Global)
    if e <= n_edges[1]
        _edge!(r, d, e, positions, cell, senders[e], receivers[e], shifts)
        _flag_zero_length!(status, d, e)
    end
end

# Every work item that finds a zero length writes the same value, so the order of the writes does
# not matter.
@inline function _flag_zero_length!(status, d, e)
    iszero(d[e]) && (status[STATUS_ZERO_LENGTH] = one(eltype(status)))
    return nothing
end

# The vector and length of edge `e` from atom `s` to atom `q`.
@inline function _edge!(r, d, e, positions, cell, s, q, shifts)
    T = eltype(r)
    n1, n2, n3 = T(shifts[1, e]), T(shifts[2, e]), T(shifts[3, e])
    for a in 1:3
        r[a, e] = positions[a, q] - positions[a, s] + (cell[a, 1] * n1 + cell[a, 2] * n2 + cell[a, 3] * n3)
    end
    d[e] = sqrt(r[1, e] * r[1, e] + r[2, e] * r[2, e] + r[3, e] * r[3, e])
    return nothing
end

"""
    _edge_features!(ws, model, k, chunk) -> nothing

For the edges of `chunk`, in columns `1:length(chunk.edges)`: e3nn's spherical harmonics `ws.Y`
of the edge vectors and their gradients `ws.dY[:, i, j] = ∂Y/∂r_i`, and the radial features
`ws.feats` with their length derivatives `ws.dfeats`. Edges with `d ≥ r_max` get zero features and
derivatives, and so do the columns of slots past `nl.n_edges[1]`, whose harmonics are zero. The
features are the same for both interactions; `k` names the interaction, as for every per-edge
stage.
"""
function _edge_features!(ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, k::Integer, chunk::EdgeChunk) where {T}
    1 <= k <= 2 || throw(ArgumentError(lazy"gpu: the interaction must be 1 or 2, got $k"))
    nb = _chunk_length(ws, chunk)
    e0 = first(chunk.edges) - 1
    nl = chunk.nl
    _launch!(_harmonics_kernel!, ws.backend, nb, ws.Y, ws.dY, ws.r, nl.n_edges, isqrt(size(ws.Y, 1)) - 1, e0)
    _launch!(
        _radial_features_kernel!, ws.backend, nb, ws.feats, ws.dfeats, ws.d, nl.senders, nl.receivers, nl.n_edges, chunk.atomic_numbers,
        model.radial_radii, model.bessel_weights, _radial_params(model.host.radial), e0,
    )
    return nothing
end

@kernel function _harmonics_kernel!(Y, dY, @Const(r), @Const(n_edges), L, e0)
    j = @index(Global)
    e = e0 + j
    if e <= n_edges[1]
        _e3nn_harmonics_column!(Y, dY, j, L, r[1, e], r[2, e], r[3, e])
    else
        _zero_column!(Y, dY, j)
    end
end

@inline function _zero_column!(Y, dY, col)
    for k in axes(Y, 1)
        Y[k, col] = zero(eltype(Y))
        for i in 1:3
            dY[k, i, col] = zero(eltype(dY))
        end
    end
    return nothing
end

# e3nn's harmonics (`e3nn_spherical_harmonics_and_gradient!`) of the vector (v1, v2, v3) up to
# degree L, into column `col` of `Y` and `dY[:, :, col]`: the native
# harmonics of (v3, v1, v2) times sqrt(4π). The recurrence for the unnormalized polynomials Q_lm,
# the prefactors F_lm and the gradient identities are ported from SpheriCart (MIT),
# github.com/lab-cosmo/sphericart at commit 5db6baecce938cd971ee4f8b6eea45d0a1099287,
# julia/src/ka_kernels.jl and generated_kernels.jl. The Q_lm (m ≥ 0) are first stored in the Y
# slots of (l, m); the second pass, by descending l, overwrites row l while row l - 1 still holds
# Q_{l-1,·}, which the gradient reads.
@inline function _e3nn_harmonics_column!(Y, dY, col, L, v1, v2, v3)
    T = eltype(Y)
    vx, vy, vz = v3, v1, v2
    r = sqrt(vx * vx + vy * vy + vz * vz)
    x, y, z = vx / r, vy / r, vz / r
    Y[_sh(0, 0), col] = one(T)
    if L >= 1
        Y[_sh(1, 1), col] = -one(T)
        Y[_sh(1, 0), col] = z
    end
    for l in 2:L
        qll = Y[_sh(l - 1, l - 1), col]
        Y[_sh(l, l), col] = -(2l - 1) * qll
        Y[_sh(l, l - 1), col] = (2l - 1) * z * qll
        for m in (l - 2):-1:0
            Y[_sh(l, m), col] = ((2l - 1) * z * Y[_sh(l - 1, m), col] - (l + m - 1) * Y[_sh(l - 2, m), col]) / (l - m)
        end
    end
    c = sqrt(4 * T(π))
    rt2 = sqrt(T(2))
    for l in L:-1:0
        # F_l0 = sqrt((2l+1)/(2π)); F_lm = -F_l,m-1 / sqrt((l+m)(l+1-m)).
        F = sqrt(T(2l + 1) / (2 * T(π)))
        s, cm = zero(T), one(T)             # sin(mφ), cos(mφ) times sin^m θ, at m = 0
        sp, cp = zero(T), zero(T)           # the same at m - 1
        for m in 0:l
            if m > 0
                F = -F / sqrt(T((l + m) * (l + 1 - m)))
                sp, cp = s, cm
                s, cm = sp * x + cp * y, cp * x - sp * y
            end
            Qlm = Y[_sh(l, m), col]
            # ∂Q_lm/∂(x, y, z) = (x Q_{l-1,m+1}, y Q_{l-1,m+1}, (l+m) Q_{l-1,m}), with Q_{l-1,k} = 0
            # for k > l - 1; ∂(c_m, s_m)/∂x = m (c_{m-1}, s_{m-1}), ∂(c_m, s_m)/∂y = m (-s_{m-1}, c_{m-1}).
            qa = m + 1 <= l - 1 ? Y[_sh(l - 1, m + 1), col] : zero(T)
            qb = m <= l - 1 ? (l + m) * Y[_sh(l - 1, m), col] : zero(T)
            if iszero(m)
                Fm = F / rt2
                _e3nn_grad_row!(dY, _sh(l, 0), col, Fm * x * qa, Fm * y * qa, Fm * qb, x, y, z, r, c)
                Y[_sh(l, 0), col] = F / rt2 * Qlm * c
            else
                _e3nn_grad_row!(dY, _sh(l, -m), col, F * (Qlm * m * sp + x * qa * s), F * (Qlm * m * cp + y * qa * s), F * qb * s, x, y, z, r, c)
                _e3nn_grad_row!(dY, _sh(l, m), col, F * (Qlm * m * cp + x * qa * cm), F * (-Qlm * m * sp + y * qa * cm), F * qb * cm, x, y, z, r, c)
                Y[_sh(l, -m), col] = F * Qlm * s * c
                Y[_sh(l, m), col] = F * Qlm * cm * c
            end
        end
    end
    return nothing
end

# Position of Y_lm in a column of harmonics, l^2 + l + m + 1.
@inline _sh(l, m) = l * l + l + m + 1

# Row `k` of the gradient with respect to the raw vector, sqrt(4π) (I - u uᵀ) g / ‖v‖, of a
# function whose gradient with respect to the unit vector u = (x, y, z) = (v3, v1, v2)/‖v‖ is
# g = (gx, gy, gz), stored in e3nn's column order (∂/∂v1, ∂/∂v2, ∂/∂v3).
@inline function _e3nn_grad_row!(dY, k, col, gx, gy, gz, x, y, z, r, c)
    p = x * gx + y * gy + z * gz
    dY[k, 3, col] = c * ((gx - p * x) / r)
    dY[k, 1, col] = c * ((gy - p * y) / r)
    dY[k, 2, col] = c * ((gz - p * z) / r)
    return nothing
end
