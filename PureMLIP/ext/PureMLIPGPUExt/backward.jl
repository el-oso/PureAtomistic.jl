# `mace_energy_forces!` on the device: the forward pass of `mace_energy!`, then the backward pass
# of the CPU's `mace_energy_forces!` with every stage on the backend of the model, and the forces,
# virial and stress gathered per atom without atomics.

"""
    mace_energy_forces!(res::DeviceMACEResult, ws::DeviceMACEWorkspace, model::DeviceMACEModel, positions, cell, atomic_numbers, nl::DeviceNeighborList) -> res

[`mace_energy_forces!`](@ref) on the backend of `model`: as the device [`mace_energy!`](@ref), and
also fill `res.forces`, `res.virial` and `res.stress`. With `g_e = ∂E/∂r_e`, the force on atom
`i` is gathered over the edges it receives, `F_i = Σ_{e: receiver(e) = i} (g_{reverse(e)} - g_e)`,
since the reverse of an edge received by `i` is an edge sent by `i`; the cotangents of the sender
features are gathered the same way. The virial sums `r_e g_eᵀ` per receiving atom, then over the
atoms in a fixed-shape tree, and the stress divides it by the volume `|det cell|`, computed on the
device. No sum depends on how the work items are scheduled, so an evaluation repeats bit for bit.
The model may have at most 256 channels (`device_model` throws for more), since the message
pullback sums the channels' terms in group memory of that size. After a throw the contents of
`res` and of the workspace are undefined.
"""
function PureMLIP.mace_energy_forces!(
        res::DeviceMACEResult{T}, ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, positions::AbstractMatrix,
        cell::AbstractMatrix, atomic_numbers::AbstractVector{<:Integer}, nl::DeviceNeighborList
    ) where {T}
    _energy_forces_unread!(res, ws, model, positions, cell, atomic_numbers, nl)
    copyto!(ws.status, nl.status)
    _check_status(ws.status, model, atomic_numbers, ws, nl)
    return res
end

# The device `mace_energy_forces!` without its read of the status word, as `_energy_unread!`.
function _energy_forces_unread!(
        res::DeviceMACEResult{T}, ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, positions::AbstractMatrix,
        cell::AbstractMatrix, atomic_numbers::AbstractVector{<:Integer}, nl::DeviceNeighborList
    ) where {T}
    n = nl.n_atoms
    size(res.forces) == (3, n) ||
        throw(DimensionMismatch(lazy"gpu: the result forces have size $(size(res.forces)), expected (3, $n)"))
    (size(res.virial) == (3, 3) && size(res.stress) == (3, 3)) ||
        throw(DimensionMismatch(lazy"gpu: the result virial and stress must be 3 × 3, got $(size(res.virial)) and $(size(res.stress))"))
    _energy_unread!(res, ws, model, positions, cell, atomic_numbers, nl)
    numbers = view(ws.numbers, 1:n)
    _second_nodes_pullback!(ws, model, n)
    _interaction_pullback!(ws, model, 2, nl, numbers, model.tables[2])
    _first_nodes_pullback!(ws, model, n)
    _interaction_pullback!(ws, model, 1, nl, numbers, model.tables[1])
    _forces!(res, ws, cell, nl)
    return res
end

# The scratch of the species-linear pullbacks: the pool of the message cotangents, which is free
# before each of them writes `dM[2]` or `dM[1]`.
_pullback_scratch(ws::DeviceMACEWorkspace) = length(ws.dM[1]) >= length(ws.dM[2]) ? ws.dM[1] : ws.dM[2]

# The backward pass from the energy to the message cotangents of interaction 2: the readouts,
# the skip connection of interaction 2, product 1 and the normalization of interaction 2 (pullback
# of `L = linear(M) / (density + 1)`).
function _second_nodes_pullback!(ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, n::Int) where {T}
    backend, m = ws.backend, model.host
    C = m.C
    list = model.contraction[1].species
    # Readouts: ∂E/∂e1 = ∂E/∂e2 = scale per atom.
    fill!(ws.de1, m.scale)
    fill!(ws.de2, m.scale)
    _linear_pullback!(ws.dg, model.readout1[2], ws.de2)
    _launch!(_readout_slope_kernel!, backend, length(ws.dg), ws.dg, ws.gpre, m.readout_cst)
    _linear_pullback!(ws.dh2, model.readout1[1], ws.dg)
    _linear_pullback!(ws.dh1, model.readout0, ws.de1)
    # h2 = product_linear(B1) + skip(h1[:, :, 1:K]).
    K = m.skip[2].K
    part = view(ws.dh1_part, :, :, 1:K)
    _species_linear_pullback!(part, model.skip[2], ws.dh2, ws.species, list, _pullback_scratch(ws), backend)
    _launch!(_add_kernel!, backend, length(part), view(ws.dh1, :, :, 1:K), part)
    _linear_pullback!(ws.dB[2], model.product_linear[2], ws.dh2)
    Zf = _contraction_buffer(ws.Zf, model.contraction[2], 1, ws.contraction_block, C)
    Yc = _contraction_buffer(ws.Yc, model.contraction[2], 2, ws.contraction_block, C)
    _symmetric_contraction_pullback!(ws.dX2, model.contraction[2], ws.dB[2], ws.X2, ws.species, n, Zf, Yc, backend)
    _launch!(_normalization_pullback_kernel!, backend, n, view(ws.ddensity, :, 2), ws.dX2, ws.dX2, ws.X2, view(ws.density, :, 2))
    _linear_pullback!(ws.dM[2], model.linear[2], ws.dX2)
    return nothing
end

# The backward pass from the sender-feature cotangents of interaction 2 to the message cotangents
# of interaction 0: the sender linear of interaction 2, product 0, the skip connection of
# interaction 1 and its normalization. X1 = skip(L0) is a linear map per atom and dL0 its
# pullback of dX1, so ⟨dL0, L0⟩ = ⟨dX1, X1⟩, and L0 (which shares memory with dL0) is not read.
function _first_nodes_pullback!(ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, n::Int) where {T}
    backend, m = ws.backend, model.host
    C = m.C
    _linear_pullback!(ws.dh1_part, model.up[2], ws.dx)
    _launch!(_add_kernel!, backend, length(ws.dh1), ws.dh1, ws.dh1_part)
    _linear_pullback!(ws.dB[1], model.product_linear[1], ws.dh1)
    Zf = _contraction_buffer(ws.Zf, model.contraction[1], 1, ws.contraction_block, C)
    Yc = _contraction_buffer(ws.Yc, model.contraction[1], 2, ws.contraction_block, C)
    _symmetric_contraction_pullback!(ws.dX1, model.contraction[1], ws.dB[1], ws.X1, ws.species, n, Zf, Yc, backend)
    _species_linear_pullback!(ws.dL0, model.skip[1], ws.dX1, ws.species, model.contraction[1].species, _pullback_scratch(ws), backend)
    _launch!(_normalization_pullback_kernel!, backend, n, view(ws.ddensity, :, 1), ws.dL0, ws.dX1, ws.X1, view(ws.density, :, 1))
    _linear_pullback!(ws.dM[1], model.linear[1], ws.dL0)
    return nothing
end

# The per-edge backward pass of interaction `k` over the edge chunks in order: the harmonics,
# radial features and radial MLP are recomputed per chunk, then the message pullback, the
# radial-MLP pullback and each edge's ∂E/∂r. The pass of interaction 2 starts ∂E/∂r with the ZBL
# terms and gathers the sender-feature cotangents `ws.dx`.
function _interaction_pullback!(ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, k::Int, nl::DeviceNeighborList, numbers, table::Val) where {T}
    if k == 2
        _zbl_gradient!(ws, model, nl, numbers)
        fill!(ws.dx, zero(T))
    end
    for c in 1:_n_edge_chunks(nl, ws.edge_chunk)
        chunk = _edge_chunk(nl, numbers, ws.edge_chunk, c)
        _edge_features!(ws, model, k, chunk)
        _radial_mlp!(ws, model, k, chunk)
        _messages_pullback!(ws, model, k, chunk, table)
        _radial_mlp_pullback!(ws, model, k, chunk)
        _edge_gradient!(ws, model, k, chunk)
        k == 2 && _sender_gradient!(ws, model, chunk, table)
    end
    return nothing
end

@kernel function _readout_slope_kernel!(dg, @Const(gpre), cst)
    q = @index(Global)
    _readout_slope!(dg, gpre, cst, q)
end

# dg[q] *= the derivative of cst·z·σ(z) at z = gpre[q] (`normalized_silu`).
@inline function _readout_slope!(dg, gpre, cst, q)
    z = gpre[q]
    σ = inv(one(z) + exp(-z))
    s = z * σ
    dg[q] *= cst * (σ + s * (one(z) - σ))
    return nothing
end

@kernel function _normalization_pullback_kernel!(ddensity, dL, @Const(dY), @Const(Y), @Const(density))
    i = @index(Global)
    _normalization_pullback!(ddensity, dL, dY, Y, density, i)
end

# The density part of the pullback of L = linear(M) / q, q = density + 1, at node `i`:
# ddensity[i] = -⟨dY, Y⟩ / q over the node's entries, then dL[:, i, :] /= q (`dL` may be `dY`).
@inline function _normalization_pullback!(ddensity, dL, dY, Y, density, i)
    T = eltype(dL)
    q = density[i] + one(T)
    s = zero(T)
    for k in axes(dY, 3), u in axes(dY, 1)
        s += dY[u, i, k] * Y[u, i, k]
    end
    ddensity[i] = -s / q
    for k in axes(dL, 3), u in axes(dL, 1)
        dL[u, i, k] /= q
    end
    return nothing
end

# The forces, virial and stress from ∂E/∂r of every edge in `ws.dEdr`.
function _forces!(res::DeviceMACEResult{T}, ws::DeviceMACEWorkspace{T}, cell::AbstractMatrix, nl::DeviceNeighborList) where {T}
    n, backend = nl.n_atoms, ws.backend
    _launch!(_forces_kernel!, backend, n, res.forces, ws.virial_parts, ws.dEdr, ws.r, nl.row_ptr, nl.reverse)
    for q in 1:9
        _sum!(view(ws.virial_sum, q:q), view(ws.virial_parts, :, q), n, backend)
    end
    _launch!(_virial_stress_kernel!, backend, 1, res.virial, res.stress, ws.virial_sum, cell)
    return nothing
end

@kernel function _forces_kernel!(F, parts, @Const(dEdr), @Const(r), @Const(row_ptr), @Const(reverse))
    i = @index(Global)
    _forces!(F, parts, dEdr, r, row_ptr, reverse, i)
end

# Atom `i`: F[:, i] = Σ_e (dEdr[:, reverse(e)] - dEdr[:, e]) and parts[i, (a - 1)·3 + b] =
# Σ_e r[b, e] dEdr[a, e], over the edges `e` that `i` receives, in edge order.
@inline function _forces!(F, parts, dEdr, r, row_ptr, reverse, i)
    T = eltype(F)
    edges = Int(row_ptr[i]):(Int(row_ptr[i + 1]) - 1)
    for a in 1:3
        acc = zero(T)
        for e in edges
            acc += dEdr[a, reverse[e]] - dEdr[a, e]
        end
        F[a, i] = acc
    end
    for a in 1:3, b in 1:3
        acc = zero(T)
        for e in edges
            acc += r[b, e] * dEdr[a, e]
        end
        parts[i, (a - 1) * 3 + b] = acc
    end
    return nothing
end

@kernel function _virial_stress_kernel!(virial, stress, @Const(A), @Const(cell))
    _virial_stress!(virial, stress, A, cell)
end

# With A[(a - 1)·3 + b] = Σ_e r[b, e] ∂E/∂r[a, e]: virial = -sym(A) and stress = sym(A) / |det cell|.
@inline function _virial_stress!(virial, stress, A, cell)
    V = abs(_det3(cell))
    for a in 1:3, b in 1:3
        w = -(A[(a - 1) * 3 + b] + A[(b - 1) * 3 + a]) / 2
        virial[b, a] = w
        stress[b, a] = -w / V
    end
    return nothing
end

_det3(C) = C[1, 1] * (C[2, 2] * C[3, 3] - C[2, 3] * C[3, 2]) -
    C[1, 2] * (C[2, 1] * C[3, 3] - C[2, 3] * C[3, 1]) +
    C[1, 3] * (C[2, 1] * C[3, 2] - C[2, 2] * C[3, 1])
