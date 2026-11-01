# Evaluation of a MACEModel on a periodic structure. Node features are `(C, K, n)` arrays
# (channel, e3nn component, atom), as the layers expect.

"""
    MACEResult{T}(n_atoms)

Outputs of one evaluation: `energy` (eV), `node_energies` (eV per atom, `E0` included),
`forces` (`3 × n_atoms`, eV/Å), `virial` and `stress` (`3 × 3`).
"""
mutable struct MACEResult{T}
    energy::T
    node_energies::Vector{T}
    forces::Matrix{T}
    virial::Matrix{T}
    stress::Matrix{T}
    function MACEResult{T}(n_atoms::Integer) where {T}
        n_atoms >= 0 || throw(ArgumentError(lazy"model: the number of atoms must not be negative, got $n_atoms"))
        n = Int(n_atoms)
        return new{T}(zero(T), zeros(T, n), zeros(T, 3, n), zeros(T, 3, 3), zeros(T, 3, 3))
    end
end

"""
    MACEWorkspace{T}(model, n_atoms, n_edges; edge_chunk = 4096)

The buffers of an evaluation of `model` on up to `n_atoms` atoms and `n_edges` directed edges.
Per edge it holds the edge vector, its length and the energy gradient with respect to the edge
vector; the spherical harmonics, the radial features and the radial-MLP outputs (the
tensor-product weights) are recomputed from them `edge_chunk` edges at a time. The workspace
also holds the energy derivatives of the backward pass of [`mace_energy_forces!`](@ref). The
evaluation functions require `T` to be the element type of the model; the constructor reads only
sizes from `model`, so `T` may also be a number type the layers accept, such as a dual number.
"""
struct MACEWorkspace{T}
    n_atoms::Int
    n_edges::Int
    chunk::Int
    species::Vector{Int}
    # per edge
    r::Matrix{T}             # 3 × E, edge vectors
    d::Vector{T}             # edge lengths
    dEdr::Matrix{T}          # 3 × E, ∂E/∂r per edge, accumulated by the backward pass
    # per chunk, recomputed from r and d in each pass over the chunk
    Y::Matrix{T}             # 16 × chunk, e3nn spherical harmonics
    feats::Matrix{T}         # n_basis × chunk, radial features
    dfeats::Matrix{T}        # n_basis × chunk, their derivatives with respect to the length
    tpw::Matrix{T}           # W × chunk, radial-MLP outputs
    tpw3::Array{T, 3}        # C × W/C × chunk, the same memory as tpw
    pre::Array{T, 3}         # hidden × chunk × 3
    act::Array{T, 3}
    dtpw::Matrix{T}          # W × chunk, ∂E/∂tpw
    dtpw3::Array{T, 3}       # the same memory as dtpw
    mlp_scratch::Array{T, 3} # hidden × chunk × 2
    dfeats_chunk::Matrix{T}  # n_basis × chunk, ∂E/∂feats through the radial MLP
    # per edge, reused: ∂E/∂Y of one edge and the gradient of its harmonics
    dYe::Vector{T}
    dYdr::Matrix{T}          # 16 × 3, ∂Y/∂r
    # Per node. The fields marked with a pool share memory with the other members of that pool;
    # the constructor lists the order in which each pool's members are used.
    h0::Array{T, 3}                  # pool `small`
    x::NTuple{2, Array{T, 3}}        # linear_up outputs
    density::Matrix{T}               # n × 2
    M::NTuple{2, Array{T, 3}}        # summed messages; pool `big`
    L::NTuple{2, Array{T, 3}}        # interaction linear outputs, density-normalized; L[1] in pool `mid`
    X1::Array{T, 3}                  # interaction 0 output (after skip_tp)
    B::NTuple{2, Array{T, 3}}        # symmetric-contraction outputs; pool `small`
    h1::Array{T, 3}                  # pool `small`
    sc::Array{T, 3}                  # pool `small`
    h2::Array{T, 3}                  # pool `small`
    gpre::Array{T, 3}                # nonlinear readout, before the activation
    g::Array{T, 3}                   # and after it
    e1::Array{T, 3}
    e2::Array{T, 3}
    pair::Vector{T}
    scratch::Vector{T}
    # per node, derivatives
    de1::Array{T, 3}
    de2::Array{T, 3}
    dg::Array{T, 3}
    dh2::Array{T, 3}
    dh1::Array{T, 3}
    dh1_part::Array{T, 3}            # one contribution to dh1 before it is added
    dB::NTuple{2, Array{T, 3}}
    dX1::Array{T, 3}                 # pool `mid`
    dL::NTuple{2, Array{T, 3}}       # dL[1] in pool `small`, dL[2] in pool `mid`
    dM::NTuple{2, Array{T, 3}}       # pool `big`
    dx::NTuple{2, Array{T, 3}}       # ∂E/∂x[2] in dx[2]; dx[1] (pool `dx`) is never written
    ddensity::Matrix{T}              # n × 2
    function MACEWorkspace{T}(model::MACEModel, n_atoms::Integer, n_edges::Integer; edge_chunk::Integer = 4096) where {T}
        (n_atoms >= 0 && n_edges >= 0) ||
            throw(ArgumentError(lazy"model: workspace sizes must not be negative, got $n_atoms atoms and $n_edges edges"))
        edge_chunk > 0 || throw(ArgumentError(lazy"model: the edge chunk must be positive, got $edge_chunk"))
        n, E = Int(n_atoms), Int(n_edges)
        chunk = min(Int(edge_chunk), max(E, 1))
        C = model.C
        nb = length(model.radial.bessel_weights)
        W = max(size(model.mlp[1].W[4], 2), size(model.mlp[2].W[4], 2))
        H = max(size(model.mlp[1].W[1], 2), size(model.mlp[2].W[1], 2))
        tpw = zeros(T, W, chunk)
        dtpw = zeros(T, W, chunk)
        node(lin::E3nnLinear) = zeros(T, lin.C_out, lin.K_out, n)
        dims(lin::E3nnLinear) = (lin.C_out, lin.K_out, n)
        messages(k) = (C, model.conv[k].K_out, n)
        products(k) = (C, n_outputs(model.product[k]), n)
        scratch = zeros(T, max(symmetric_contraction_scratch_size(model.product[1]), symmetric_contraction_scratch_size(model.product[2])))
        # Pools of buffers whose lifetimes do not overlap, in the order of use in
        # `mace_energy_forces!`; each member is dead before the next one is written.
        # - big: M[1], M[2] (forward), dM[2], dM[1] (backward).
        # - mid: L[1] (read only by skip[1]), dL[2], dX1.
        # - small: h0, h1, B[1], B[2], sc and h2 side by side (forward only; the backward reads
        #   none of them), then dL[1] over all of them.
        # - dx: dx[2], then dx[1], which interaction 0 passes with `need_dx = false`, so it is
        #   neither read nor written; it holds no memory of its own.
        (), (M0, M1, dM0, dM1) = _pool(T, (), (messages(1), messages(2), messages(1), messages(2)))
        (), (L0, dL1, dX1) = _pool(T, (), (dims(model.linear[1]), dims(model.linear[2]), (C, model.skip[1].K, n)))
        (h0, h1, B0, B1, sc, h2), (dL0,) = _pool(
            T, ((C, 1, n), dims(model.product_linear[1]), products(1), products(2), (C, model.skip[2].K, n), dims(model.product_linear[2])),
            (dims(model.linear[1]),),
        )
        (), (dx1, dx0) = _pool(T, (), (dims(model.up[2]), dims(model.up[1])))
        return new{T}(
            n, E, chunk, zeros(Int, n),
            zeros(T, 3, E), zeros(T, E), zeros(T, 3, E),
            zeros(T, 16, chunk), zeros(T, nb, chunk), zeros(T, nb, chunk),
            tpw, reshape(tpw, C, W ÷ C, chunk), zeros(T, H, chunk, 3), zeros(T, H, chunk, 3),
            dtpw, reshape(dtpw, C, W ÷ C, chunk), zeros(T, H, chunk, 2), zeros(T, nb, chunk),
            zeros(T, 16), zeros(T, 16, 3),
            h0, (node(model.up[1]), node(model.up[2])), zeros(T, n, 2),
            (M0, M1), (L0, node(model.linear[2])), zeros(T, C, model.skip[1].K, n),
            (B0, B1), h1, sc, h2,
            node(model.readout1[1]), node(model.readout1[1]), node(model.readout0), node(model.readout1[2]), zeros(T, n), scratch,
            node(model.readout0), node(model.readout1[2]), node(model.readout1[1]), node(model.product_linear[2]),
            node(model.product_linear[1]), node(model.product_linear[1]), (zeros(T, products(1)), zeros(T, products(2))),
            dX1, (dL0, dL1), (dM0, dM1), (dx0, dx1), zeros(T, n, 2),
        )
    end
end

# Zero-filled arrays of the sizes in `packed` and `shared` over one block of memory: the `packed`
# arrays lie side by side from the block's first element, and every `shared` array starts at the
# first element too, so it overlaps them and the other `shared` arrays. `Base.wrap` makes each an
# `Array` over the block, so every member keeps the type `Array{T, 3}` of a separately allocated
# buffer. `Base.wrap` is documented in Base but not declared public.
function _pool(::Type{T}, packed::Tuple{Vararg{NTuple{3, Int}}}, shared::Tuple{Vararg{NTuple{3, Int}}}) where {T}
    plen = map(prod, packed)
    mem = Memory{T}(undef, max(sum(plen; init = 0), maximum(map(prod, shared); init = 0)))
    fill!(mem, zero(T))
    at(offset, dims) = Base.wrap(Array, iszero(offset) || iszero(prod(dims)) ? memoryref(mem) : memoryref(mem, offset + 1), dims)
    return map(at, map(-, accumulate(+, plen; init = 0), plen), packed), map(d -> at(0, d), shared)
end

# Out of line: the cutoff comparison widens a Float32 model radius to the Float64 `r_cut`, and a
# conversion inlined into `mace_energy!` would hide behind it from the precision check.
@noinline function _check_inputs(res::MACEResult, ws::MACEWorkspace, model::MACEModel, positions, cell, atomic_numbers, nl::NeighborList)
    Base.require_one_based_indexing(positions, cell, atomic_numbers)
    n, m = nl.n_atoms, nl.n_edges
    size(positions) == (3, n) ||
        throw(DimensionMismatch(lazy"model: positions have size $(size(positions)), the neighbor list has $n atoms"))
    size(cell) == (3, 3) || throw(DimensionMismatch(lazy"model: the cell has size $(size(cell)), expected (3, 3)"))
    length(atomic_numbers) == n ||
        throw(DimensionMismatch(lazy"model: $(length(atomic_numbers)) atomic numbers for $n atoms"))
    length(res.node_energies) == n ||
        throw(DimensionMismatch(lazy"model: the result holds $(length(res.node_energies)) atoms, the neighbor list has $n"))
    rc, rm = nl.r_cut, model.r_max
    rc >= rm || throw(ArgumentError(lazy"model: the neighbor list cutoff $rc is below r_max $rm"))
    ws.n_atoms >= n || throw(ArgumentError(lazy"model: the workspace holds $(ws.n_atoms) atoms, the neighbor list has $n"))
    ws.n_edges >= m || throw(ArgumentError(lazy"model: the workspace holds $(ws.n_edges) edges, the neighbor list has $m"))
    return nothing
end

# Edge vectors and lengths of every edge.
function _edges!(ws::MACEWorkspace{T}, positions, cell, nl::NeighborList) where {T}
    for e in 1:nl.n_edges
        s, r = nl.senders[e], nl.receivers[e]
        n1, n2, n3 = nl.shifts[1, e], nl.shifts[2, e], nl.shifts[3, e]
        for a in 1:3
            ws.r[a, e] = positions[a, r] - positions[a, s] + (cell[a, 1] * n1 + cell[a, 2] * n2 + cell[a, 3] * n3)
        end
        ws.d[e] = sqrt(ws.r[1, e]^2 + ws.r[2, e]^2 + ws.r[3, e]^2)
    end
    return nothing
end

# Radial features and their length derivatives of edges `start:stop`, in columns `1:stop-start+1`
# of `ws.feats` and `ws.dfeats`.
function _chunk_radial!(ws::MACEWorkspace{T}, model::MACEModel, atomic_numbers, nl::NeighborList, start::Int, stop::Int) where {T}
    for e in start:stop
        s, r = nl.senders[e], nl.receivers[e]
        j = e - start + 1
        radial_embedding!(view(ws.feats, :, j), view(ws.dfeats, :, j), model.radial, ws.d[e], Int(atomic_numbers[s]), Int(atomic_numbers[r]))
    end
    return nothing
end

# Messages of interaction `k` summed at the receivers, and the density of each receiver. The
# harmonics and radial features are recomputed per chunk of `ws.chunk` edges from `ws.r` and `ws.d`.
function _messages!(
        M::AbstractArray{T, 3}, density::AbstractVector{T}, ws::MACEWorkspace{T}, model::MACEModel, k::Int,
        x::AbstractArray{T, 3}, atomic_numbers, nl::NeighborList
    ) where {T}
    fill!(M, zero(T))
    fill!(density, zero(T))
    mlp, tp = model.mlp[k], model.conv[k]
    W = size(mlp.W[4], 2)
    P = length(tp.paths)
    for start in 1:ws.chunk:nl.n_edges
        stop = min(start + ws.chunk - 1, nl.n_edges)
        nb = stop - start + 1
        _chunk_radial!(ws, model, atomic_numbers, nl, start, stop)
        radial_mlp!(view(ws.tpw, 1:W, 1:nb), view(ws.pre, :, 1:nb, :), view(ws.act, :, 1:nb, :), mlp, view(ws.feats, :, 1:nb))
        for e in start:stop
            s, r = nl.senders[e], nl.receivers[e]
            j = e - start + 1
            Y = view(ws.Y, :, j)
            e3nn_spherical_harmonics!(Y, view(ws.r, :, e), 3)
            tp_edge_add!(view(M, :, :, r), tp, view(x, :, :, s), Y, view(ws.tpw3, :, 1:P, j))
            density[r] += density_edge(view(ws.feats, :, j), model.density[k])[1]
        end
    end
    return nothing
end

# The divisor of the summed messages of a node with density ρ; the forward pass and its pullback
# both use it.
_density_norm(ρ) = ρ + one(ρ)

# L = linear(M) / (density + 1), per node.
function _normalized_linear!(L::AbstractArray{T, 3}, lin::E3nnLinear, M::AbstractArray{T, 3}, density::AbstractVector{T}) where {T}
    linear!(L, lin, M)
    for i in axes(L, 3)
        q = _density_norm(density[i])
        for c in axes(L, 2), u in axes(L, 1)
            L[u, c, i] /= q
        end
    end
    return L
end

"""
    mace_energy!(res, ws, model, positions, cell, atomic_numbers, nl) -> res

Evaluate `model` on the periodic structure with `positions` (`3 × n`, Å), lattice vectors as the
columns of `cell`, and `atomic_numbers`, over the edges of the neighbor list `nl` (built for
these positions with a cutoff of at least `model.r_max`). Fills `res.energy` and
`res.node_energies`: `E0` of the element plus `scale · (pair + readouts) + shift` per atom.
"""
function mace_energy!(
        res::MACEResult{T}, ws::MACEWorkspace{T}, model::MACEModel{T}, positions::AbstractMatrix{T},
        cell::AbstractMatrix{T}, atomic_numbers::AbstractVector{<:Integer}, nl::NeighborList
    ) where {T}
    _check_inputs(res, ws, model, positions, cell, atomic_numbers, nl)
    n = nl.n_atoms
    atoms = 1:n
    species = view(ws.species, atoms)
    for i in atoms
        z = atomic_numbers[i]
        s = 0 <= z < length(model.species) ? model.species[z + 1] : 0
        s > 0 || throw(ArgumentError(lazy"model: element $z is not in the model's element table"))
        species[i] = s
    end
    _edges!(ws, positions, cell, nl)

    # Embedding and interaction 0.
    h0 = view(ws.h0, :, :, atoms)
    for i in atoms, w in axes(h0, 1)
        h0[w, 1, i] = model.embedding[w, species[i]]
    end
    x0 = view(ws.x[1], :, :, atoms)
    linear!(x0, model.up[1], h0)
    M0, dens0 = view(ws.M[1], :, :, atoms), view(ws.density, atoms, 1)
    _messages!(M0, dens0, ws, model, 1, x0, atomic_numbers, nl)
    L0 = _normalized_linear!(view(ws.L[1], :, :, atoms), model.linear[1], M0, dens0)
    X1 = view(ws.X1, :, :, atoms)
    species_linear!(X1, model.skip[1], L0, species)

    # Product 0.
    B0 = view(ws.B[1], :, :, atoms)
    symmetric_contraction!(B0, model.product[1], X1, species, ws.scratch)
    h1 = view(ws.h1, :, :, atoms)
    linear!(h1, model.product_linear[1], B0)

    # Interaction 1: the skip connection acts on the block input.
    sc = view(ws.sc, :, :, atoms)
    species_linear!(sc, model.skip[2], view(h1, :, 1:model.skip[2].K, :), species)
    x1 = view(ws.x[2], :, :, atoms)
    linear!(x1, model.up[2], h1)
    M1, dens1 = view(ws.M[2], :, :, atoms), view(ws.density, atoms, 2)
    _messages!(M1, dens1, ws, model, 2, x1, atomic_numbers, nl)
    X2 = _normalized_linear!(view(ws.L[2], :, :, atoms), model.linear[2], M1, dens1)

    # Product 1 with the skip connection.
    B1 = view(ws.B[2], :, :, atoms)
    symmetric_contraction!(B1, model.product[2], X2, species, ws.scratch)
    h2 = view(ws.h2, :, :, atoms)
    linear!(h2, model.product_linear[2], B1)
    for q in eachindex(h2, sc)
        h2[q] += sc[q]
    end

    # Readouts.
    e1 = view(ws.e1, :, :, atoms)
    linear!(e1, model.readout0, h1)
    gpre, g = view(ws.gpre, :, :, atoms), view(ws.g, :, :, atoms)
    linear!(gpre, model.readout1[1], h2)
    for q in eachindex(g, gpre)
        g[q] = normalized_silu(gpre[q], model.readout_cst)[1]
    end
    e2 = view(ws.e2, :, :, atoms)
    linear!(e2, model.readout1[2], g)

    # ZBL pair repulsion, half of each directed edge credited to its receiver.
    pair = view(ws.pair, atoms)
    fill!(pair, zero(T))
    for e in 1:nl.n_edges
        s, r = nl.senders[e], nl.receivers[e]
        pair[r] += zbl_edge_energy(ws.d[e], Int(atomic_numbers[s]), Int(atomic_numbers[r]), model.zbl)[1]
    end

    E = zero(T)
    for i in atoms
        ei = model.e0[species[i]] + (model.scale * ((pair[i] + e1[1, 1, i]) + e2[1, 1, i]) + model.shift)
        res.node_energies[i] = ei
        E += ei
    end
    res.energy = E
    return res
end

# The density part of the pullback of L = linear(M) / q, q = density + 1: overwrites `ddensity`
# with -⟨dL, L⟩ / q per node and scales `dL` by 1 / q in place. The inner product is taken as
# ⟨dY, Y⟩, which equals ⟨dL, L⟩ when Y = A L and dY is its cotangent with dL = Aᵀ dY for a
# per-node linear map A (with A = identity, `dY` and `Y` are `dL` and `L`).
function _normalization_pullback!(
        ddensity::AbstractVector{T}, dL::AbstractArray{T, 3}, dY::AbstractArray{T, 3}, Y::AbstractArray{T, 3},
        density::AbstractVector{T}
    ) where {T}
    for i in axes(dL, 3)
        q = _density_norm(density[i])
        s = zero(T)
        for c in axes(dY, 2), u in axes(dY, 1)
            s += dY[u, c, i] * Y[u, c, i]
        end
        ddensity[i] = -s / q
        for c in axes(dL, 2), u in axes(dL, 1)
            dL[u, c, i] /= q
        end
    end
    return nothing
end

# Pullback of L = linear(M) / (density + 1): overwrites `dM` and `ddensity` from `dL`, the
# derivative with respect to L. `dL` is scaled by 1 / (density + 1) in place.
function _normalized_linear_pullback!(
        dM::AbstractArray{T, 3}, ddensity::AbstractVector{T}, lin::E3nnLinear, dL::AbstractArray{T, 3},
        L::AbstractArray{T, 3}, density::AbstractVector{T}
    ) where {T}
    _normalization_pullback!(ddensity, dL, dL, L, density)
    linear_pullback!(dM, lin, dL)
    return nothing
end

# Pullback of `_messages!` for interaction `k`: overwrites `dx` (the derivative with respect to
# the sender features) and adds to `ws.dEdr` per edge, from `dM` and `ddensity`. The harmonics,
# radial features and radial MLP are recomputed over the same chunks as in `_messages!`. Each
# edge's ∂E/∂Y and ∂E/∂d enter `ws.dEdr` through ∂Y/∂r and ∂d/∂r = r / d. With
# `need_dx = false`, `dx` is neither read nor written; only its size is checked.
function _messages_pullback!(
        dx::AbstractArray{T, 3}, ws::MACEWorkspace{T}, model::MACEModel{T}, k::Int, x::AbstractArray{T, 3},
        dM::AbstractArray{T, 3}, ddensity::AbstractVector{T}, atomic_numbers, nl::NeighborList;
        need_dx::Bool = true
    ) where {T}
    need_dx && fill!(dx, zero(T))
    mlp, tp, wd = model.mlp[k], model.conv[k], model.density[k]
    W = size(mlp.W[4], 2)
    P = length(tp.paths)
    dYe, dYdr = ws.dYe, ws.dYdr
    for start in 1:ws.chunk:nl.n_edges
        stop = min(start + ws.chunk - 1, nl.n_edges)
        nb = stop - start + 1
        _chunk_radial!(ws, model, atomic_numbers, nl, start, stop)
        pre, act = view(ws.pre, :, 1:nb, :), view(ws.act, :, 1:nb, :)
        radial_mlp!(view(ws.tpw, 1:W, 1:nb), pre, act, mlp, view(ws.feats, :, 1:nb))
        dw = view(ws.dtpw, 1:W, 1:nb)
        fill!(dw, zero(T))
        for e in start:stop
            s, r = nl.senders[e], nl.receivers[e]
            j = e - start + 1
            Y = view(ws.Y, :, j)
            e3nn_spherical_harmonics_and_gradient!(Y, dYdr, view(ws.r, :, e), 3)
            fill!(dYe, zero(T))
            tp_edge_pullback_add!(
                view(dx, :, :, s), dYe, view(ws.dtpw3, :, 1:P, j), tp,
                view(dM, :, :, r), view(x, :, :, s), Y, view(ws.tpw3, :, 1:P, j);
                need_dx
            )
            for a in 1:3
                acc = zero(T)
                for i in axes(dYdr, 1)
                    acc += dYe[i] * dYdr[i, a]
                end
                ws.dEdr[a, e] += acc
            end
        end
        dfc = view(ws.dfeats_chunk, :, 1:nb)
        radial_mlp_pullback!(dfc, view(ws.mlp_scratch, :, 1:nb, :), mlp, dw, pre, act)
        for e in start:stop
            j = e - start + 1
            # ∂ρ/∂feats = c .* wd for the edge density ρ added to its receiver.
            a = ddensity[nl.receivers[e]] * density_edge(view(ws.feats, :, j), wd)[2]
            dd = zero(T)
            for b in axes(dfc, 1)
                dd += (dfc[b, j] + a * wd[b]) * ws.dfeats[b, j]
            end
            dd /= ws.d[e]
            for c in 1:3
                ws.dEdr[c, e] += dd * ws.r[c, e]
            end
        end
    end
    return nothing
end

"""
    mace_energy_forces!(res, ws, model, positions, cell, atomic_numbers, nl) -> res

As [`mace_energy!`](@ref), and also fill `res.forces` (`3 × n`, eV/Å, `-∂E/∂positions`),
`res.virial` (`-∂E/∂ε` for the symmetric strain `ε`, eV) and `res.stress`
(`∂E/∂ε / |det cell|`, eV/Å³, positive when tensile). The derivatives come from a backward
pass through the layers: with `g_e = ∂E/∂r_e` for edge vector `r_e`, the sender of `e` gets
`+g_e` and the receiver `-g_e` in the forces, and `∂E/∂ε = sym(Σ_e r_e g_eᵀ)`.
"""
function mace_energy_forces!(
        res::MACEResult{T}, ws::MACEWorkspace{T}, model::MACEModel{T}, positions::AbstractMatrix{T},
        cell::AbstractMatrix{T}, atomic_numbers::AbstractVector{<:Integer}, nl::NeighborList
    ) where {T}
    n = nl.n_atoms
    size(res.forces) == (3, n) ||
        throw(DimensionMismatch(lazy"model: the result forces have size $(size(res.forces)), expected (3, $n)"))
    (size(res.virial) == (3, 3) && size(res.stress) == (3, 3)) ||
        throw(DimensionMismatch(lazy"model: the result virial and stress must be 3 × 3, got $(size(res.virial)) and $(size(res.stress))"))
    mace_energy!(res, ws, model, positions, cell, atomic_numbers, nl)
    atoms = 1:n
    E = nl.n_edges
    species = view(ws.species, atoms)
    # ∂E/∂r per edge starts from the ZBL pair term; both message pullbacks add to it.
    for e in 1:E
        s, r = nl.senders[e], nl.receivers[e]
        d = ws.d[e]
        dd = model.scale * zbl_edge_energy(d, Int(atomic_numbers[s]), Int(atomic_numbers[r]), model.zbl)[2] / d
        for a in 1:3
            ws.dEdr[a, e] = dd * ws.r[a, e]
        end
    end

    # Readouts: ∂E/∂e1 = ∂E/∂e2 = scale per atom.
    de1, de2 = view(ws.de1, :, :, atoms), view(ws.de2, :, :, atoms)
    fill!(de1, model.scale)
    fill!(de2, model.scale)
    dg, gpre = view(ws.dg, :, :, atoms), view(ws.gpre, :, :, atoms)
    linear_pullback!(dg, model.readout1[2], de2)
    for q in eachindex(dg, gpre)
        dg[q] *= normalized_silu(gpre[q], model.readout_cst)[2]
    end
    dh2 = view(ws.dh2, :, :, atoms)
    linear_pullback!(dh2, model.readout1[1], dg)
    dh1 = view(ws.dh1, :, :, atoms)
    linear_pullback!(dh1, model.readout0, de1)

    # Product 1 and the skip connection of interaction 1 (h2 = product_linear(B1) + sc).
    dB1 = view(ws.dB[2], :, :, atoms)
    linear_pullback!(dB1, model.product_linear[2], dh2)
    dX2 = view(ws.dL[2], :, :, atoms)
    symmetric_contraction_pullback!(dX2, model.product[2], dB1, view(ws.L[2], :, :, atoms), species, ws.scratch)
    part = view(ws.dh1_part, :, :, atoms)
    K = model.skip[2].K
    species_linear_pullback!(view(part, :, 1:K, :), model.skip[2], dh2, species)
    for i in atoms, c in 1:K, u in axes(dh1, 1)
        dh1[u, c, i] += part[u, c, i]
    end

    # Interaction 1.
    dM1, ddens1 = view(ws.dM[2], :, :, atoms), view(ws.ddensity, atoms, 2)
    _normalized_linear_pullback!(dM1, ddens1, model.linear[2], dX2, view(ws.L[2], :, :, atoms), view(ws.density, atoms, 2))
    dx1 = view(ws.dx[2], :, :, atoms)
    _messages_pullback!(dx1, ws, model, 2, view(ws.x[2], :, :, atoms), dM1, ddens1, atomic_numbers, nl)
    linear_pullback!(part, model.up[2], dx1)
    for q in eachindex(dh1, part)
        dh1[q] += part[q]
    end

    # Product 0 and interaction 0. The embedding h0 does not depend on positions, so the derivative
    # with respect to the interaction-0 sender features is not needed.
    dB0 = view(ws.dB[1], :, :, atoms)
    linear_pullback!(dB0, model.product_linear[1], dh1)
    dX1, X1 = view(ws.dX1, :, :, atoms), view(ws.X1, :, :, atoms)
    symmetric_contraction_pullback!(dX1, model.product[1], dB0, X1, species, ws.scratch)
    dL0 = view(ws.dL[1], :, :, atoms)
    species_linear_pullback!(dL0, model.skip[1], dX1, species)
    dM0, ddens0 = view(ws.dM[1], :, :, atoms), view(ws.ddensity, atoms, 1)
    # L[1] shares memory with dL[2] and dX1 and no longer holds L0. X1 = skip[1](L0) is a linear
    # map per atom and dL0 is its pullback of dX1, so ⟨dL0, L0⟩ = ⟨dX1, X1⟩.
    _normalization_pullback!(ddens0, dL0, dX1, X1, view(ws.density, atoms, 1))
    linear_pullback!(dM0, model.linear[1], dL0)
    # With `need_dx = false` the sender derivative dx[1] is neither read nor written.
    _messages_pullback!(view(ws.dx[1], :, :, atoms), ws, model, 1, view(ws.x[1], :, :, atoms), dM0, ddens0, atomic_numbers, nl; need_dx = false)

    # Edge vectors: g_e = ∂E/∂r_e.
    F = res.forces
    fill!(F, zero(T))
    A = res.virial
    fill!(A, zero(T))
    for e in 1:E
        s, r = nl.senders[e], nl.receivers[e]
        for a in 1:3
            ga = ws.dEdr[a, e]
            F[a, s] += ga
            F[a, r] -= ga
            for b in 1:3
                A[b, a] += ws.r[b, e] * ga
            end
        end
    end
    # A = Σ_e r_e g_eᵀ; virial = -sym(A), stress = sym(A) / volume.
    V = abs(_det3(cell))
    for a in 1:3, b in 1:a
        w = -(A[a, b] + A[b, a]) / 2
        A[a, b] = w
        A[b, a] = w
    end
    for q in eachindex(res.stress, A)
        res.stress[q] = -A[q] / V
    end
    return res
end

_det3(C) = C[1, 1] * (C[2, 2] * C[3, 3] - C[2, 3] * C[3, 2]) -
    C[1, 2] * (C[2, 1] * C[3, 3] - C[2, 3] * C[3, 1]) +
    C[1, 3] * (C[2, 1] * C[3, 2] - C[2, 2] * C[3, 1])

"""
    mace_energy_forces(model, positions, cell, atomic_numbers) -> MACEResult

[`mace_energy_forces!`](@ref) with a neighbor list of cutoff `model.r_max`, a workspace and a
result allocated for this structure.
"""
function mace_energy_forces(
        model::MACEModel{T}, positions::AbstractMatrix{T}, cell::AbstractMatrix{T}, atomic_numbers::AbstractVector{<:Integer}
    ) where {T}
    nl = neighbor_list(positions, cell, model.r_max)
    n = nl.n_atoms
    ws = MACEWorkspace{T}(model, n, nl.n_edges)
    return mace_energy_forces!(MACEResult{T}(n), ws, model, positions, cell, atomic_numbers, nl)
end
