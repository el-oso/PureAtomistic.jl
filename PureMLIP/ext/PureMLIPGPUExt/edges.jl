# The per-edge geometry on the device: edge vectors and lengths of every edge, the receiver chunks
# that the per-edge buffers hold one at a time, and e3nn's spherical harmonics with their gradient.

"""
    EdgeChunk{L, VZ}

The receivers `atoms` of the device neighbor list `nl` and their edges `edges`, a contiguous range
because the edges are sorted by receiver; `atomic_numbers` are those of every atom of the list, on
its backend. Column `j` of the workspace's per-edge-chunk buffers holds edge `first(edges) + j - 1`.
"""
struct EdgeChunk{L <: DeviceNeighborList, VZ <: AbstractVector{<:Integer}}
    nl::L
    atomic_numbers::VZ
    atoms::UnitRange{Int}
    edges::UnitRange{Int}
end

"""
    _edge_chunks(nl, atomic_numbers, row_ptr, edge_chunk) -> Vector{EdgeChunk}

Consecutive ranges of whole receivers of `nl` that cover every atom, each with at most
`edge_chunk` edges and as many receivers as fit. `row_ptr` is a host copy of `nl.row_ptr`.
"""
function _edge_chunks(nl::DeviceNeighborList, atomic_numbers::AbstractVector{<:Integer}, row_ptr::AbstractVector{<:Integer}, edge_chunk::Integer)
    _require_one_based(row_ptr, atomic_numbers)
    _common_backend(nl.senders, atomic_numbers)
    n = nl.n_atoms
    length(row_ptr) == n + 1 || throw(DimensionMismatch(lazy"gpu: row_ptr has length $(length(row_ptr)), expected $(n + 1)"))
    length(atomic_numbers) == n || throw(DimensionMismatch(lazy"gpu: $(length(atomic_numbers)) atomic numbers for $n atoms"))
    chunks = EdgeChunk{typeof(nl), typeof(atomic_numbers)}[]
    a = 1
    while a <= n
        degree = row_ptr[a + 1] - row_ptr[a]
        degree <= edge_chunk || throw(ArgumentError(lazy"gpu: receiver $a has $degree edges, more than the edge chunk $edge_chunk"))
        b = a
        while b < n && row_ptr[b + 2] - row_ptr[a] <= edge_chunk
            b += 1
        end
        push!(chunks, EdgeChunk(nl, atomic_numbers, a:b, Int(row_ptr[a]):Int(row_ptr[b + 1] - 1)))
        a = b + 1
    end
    return chunks
end

# The number of edges of `chunk`, after checking that the chunk fits the workspace.
function _chunk_length(ws::DeviceMACEWorkspace, chunk::EdgeChunk)
    _common_backend(ws.r, chunk.nl.senders, chunk.atomic_numbers)
    nb = length(chunk.edges)
    nb <= ws.edge_chunk || throw(ArgumentError(lazy"gpu: the chunk has $nb edges, the workspace holds $(ws.edge_chunk)"))
    last(chunk.edges) <= ws.n_edges || throw(ArgumentError(lazy"gpu: the chunk reaches edge $(last(chunk.edges)), the workspace holds $(ws.n_edges)"))
    last(chunk.atoms) <= ws.n_atoms || throw(ArgumentError(lazy"gpu: the chunk reaches atom $(last(chunk.atoms)), the workspace holds $(ws.n_atoms)"))
    return nb
end

"""
    _edges!(ws, positions, cell, nl) -> nothing

Edge vectors `ws.r[:, e] = positions[:, receiver] - positions[:, sender] + cell · shift` and their
lengths `ws.d[e]` for the first `nl.n_edges[1]` edges, read on the device. The workspace holds
every edge the list can hold.
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
    _launch!(_edges_kernel!, ws.backend, nl.capacity, ws.r, ws.d, positions, cell, nl.senders, nl.receivers, nl.shifts, nl.n_edges)
    return nothing
end

# One work item per edge slot of the list; slots past `n_edges[1]` hold no edge. Each kernel of
# the extension only indexes and calls one `@inline` function that holds its arithmetic, so that
# the arithmetic also compiles as host code, where the Float32 conversion sweep of the tests sees
# it.
@kernel function _edges_kernel!(r, d, @Const(positions), @Const(cell), @Const(senders), @Const(receivers), @Const(shifts), @Const(n_edges))
    e = @index(Global)
    e <= n_edges[1] && _edge!(r, d, e, positions, cell, senders[e], receivers[e], shifts)
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
derivatives. The features are the same for both interactions; `k` names the interaction, as for
every per-edge stage.
"""
function _edge_features!(ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, k::Integer, chunk::EdgeChunk) where {T}
    1 <= k <= 2 || throw(ArgumentError(lazy"gpu: the interaction must be 1 or 2, got $k"))
    nb = _chunk_length(ws, chunk)
    e0 = first(chunk.edges) - 1
    _launch!(_harmonics_kernel!, ws.backend, nb, ws.Y, ws.dY, ws.r, isqrt(size(ws.Y, 1)) - 1, e0)
    nl = chunk.nl
    _launch!(
        _radial_features_kernel!, ws.backend, nb, ws.feats, ws.dfeats, ws.d, nl.senders, nl.receivers, chunk.atomic_numbers,
        model.radial_radii, model.bessel_weights, _radial_params(model.host.radial), e0,
    )
    return nothing
end

@kernel function _harmonics_kernel!(Y, dY, @Const(r), L, e0)
    j = @index(Global)
    e = e0 + j
    _e3nn_harmonics_column!(Y, dY, j, L, r[1, e], r[2, e], r[3, e])
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
