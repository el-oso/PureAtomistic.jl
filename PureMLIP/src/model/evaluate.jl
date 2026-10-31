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
Per-edge geometry is held for every edge; the radial-MLP outputs (the tensor-product weights)
are computed `edge_chunk` edges at a time.
"""
struct MACEWorkspace{T}
    n_atoms::Int
    n_edges::Int
    chunk::Int
    species::Vector{Int}
    # per edge
    r::Matrix{T}             # 3 × E, edge vectors
    d::Vector{T}             # edge lengths
    Y::Matrix{T}             # 16 × E, e3nn spherical harmonics
    feats::Matrix{T}         # n_basis × E, radial features
    dfeats::Matrix{T}        # n_basis × E, their derivatives with respect to the length
    # per chunk
    tpw::Matrix{T}           # W × chunk, radial-MLP outputs
    tpw3::Array{T, 3}        # C × W/C × chunk, the same memory as tpw
    pre::Array{T, 3}         # hidden × chunk × 3
    act::Array{T, 3}
    # per node
    h0::Array{T, 3}
    x::NTuple{2, Array{T, 3}}        # linear_up outputs
    density::Matrix{T}               # n × 2
    M::NTuple{2, Array{T, 3}}        # summed messages
    L::NTuple{2, Array{T, 3}}        # interaction linear outputs, density-normalized
    X1::Array{T, 3}                  # interaction 0 output (after skip_tp)
    B::NTuple{2, Array{T, 3}}        # symmetric-contraction outputs
    h1::Array{T, 3}
    sc::Array{T, 3}
    h2::Array{T, 3}
    g::Array{T, 3}
    e1::Array{T, 3}
    e2::Array{T, 3}
    pair::Vector{T}
    scratch::Vector{T}
    function MACEWorkspace{T}(model::MACEModel{T}, n_atoms::Integer, n_edges::Integer; edge_chunk::Integer = 4096) where {T}
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
        node(lin::E3nnLinear) = zeros(T, lin.C_out, lin.K_out, n)
        scratch = zeros(T, max(symmetric_contraction_scratch_size(model.product[1]), symmetric_contraction_scratch_size(model.product[2])))
        return new{T}(
            n, E, chunk, zeros(Int, n),
            zeros(T, 3, E), zeros(T, E), zeros(T, 16, E), zeros(T, nb, E), zeros(T, nb, E),
            tpw, reshape(tpw, C, W ÷ C, chunk), zeros(T, H, chunk, 3), zeros(T, H, chunk, 3),
            zeros(T, C, 1, n), (node(model.up[1]), node(model.up[2])), zeros(T, n, 2),
            (zeros(T, C, model.conv[1].K_out, n), zeros(T, C, model.conv[2].K_out, n)),
            (node(model.linear[1]), node(model.linear[2])), zeros(T, C, model.skip[1].K, n),
            (zeros(T, C, _n_out(model.product[1]), n), zeros(T, C, _n_out(model.product[2]), n)),
            node(model.product_linear[1]), zeros(T, C, model.skip[2].K, n), node(model.product_linear[2]),
            node(model.readout1[1]), node(model.readout0), node(model.readout1[2]), zeros(T, n), scratch,
        )
    end
end

function _check_inputs(res::MACEResult, ws::MACEWorkspace, model::MACEModel, positions, cell, atomic_numbers, nl::NeighborList)
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

# Edge vectors, lengths, spherical harmonics and radial features of every edge.
function _edges!(ws::MACEWorkspace{T}, model::MACEModel{T}, positions, cell, atomic_numbers, nl::NeighborList) where {T}
    for e in 1:nl.n_edges
        s, r = nl.senders[e], nl.receivers[e]
        n1, n2, n3 = nl.shifts[1, e], nl.shifts[2, e], nl.shifts[3, e]
        for a in 1:3
            ws.r[a, e] = positions[a, r] - positions[a, s] + (cell[a, 1] * n1 + cell[a, 2] * n2 + cell[a, 3] * n3)
        end
        d = sqrt(ws.r[1, e]^2 + ws.r[2, e]^2 + ws.r[3, e]^2)
        ws.d[e] = d
        e3nn_spherical_harmonics!(view(ws.Y, :, e), view(ws.r, :, e), 3)
        radial_embedding!(view(ws.feats, :, e), view(ws.dfeats, :, e), model.radial, d, Int(atomic_numbers[s]), Int(atomic_numbers[r]))
    end
    return nothing
end

# Messages of interaction `k` summed at the receivers, and the density of each receiver.
function _messages!(M::AbstractArray{T, 3}, density::AbstractVector{T}, ws::MACEWorkspace{T}, model::MACEModel{T}, k::Int, x::AbstractArray{T, 3}, nl::NeighborList) where {T}
    fill!(M, zero(T))
    fill!(density, zero(T))
    mlp, tp = model.mlp[k], model.conv[k]
    W = size(mlp.W[4], 2)
    P = length(tp.paths)
    for start in 1:ws.chunk:nl.n_edges
        stop = min(start + ws.chunk - 1, nl.n_edges)
        nb = stop - start + 1
        radial_mlp!(view(ws.tpw, 1:W, 1:nb), view(ws.pre, :, 1:nb, :), view(ws.act, :, 1:nb, :), mlp, view(ws.feats, :, start:stop))
        for e in start:stop
            s, r = nl.senders[e], nl.receivers[e]
            tp_edge_add!(view(M, :, :, r), tp, view(x, :, :, s), view(ws.Y, :, e), view(ws.tpw3, :, 1:P, e - start + 1))
            density[r] += density_edge(view(ws.feats, :, e), model.density[k])[1]
        end
    end
    return nothing
end

# L = linear(M) / (density + 1), per node.
function _normalized_linear!(L::AbstractArray{T, 3}, lin::E3nnLinear, M::AbstractArray{T, 3}, density::AbstractVector{T}) where {T}
    linear!(L, lin, M)
    for i in axes(L, 3)
        q = density[i] + 1
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
    _edges!(ws, model, positions, cell, atomic_numbers, nl)

    # Embedding and interaction 0.
    h0 = view(ws.h0, :, :, atoms)
    for i in atoms, w in axes(h0, 1)
        h0[w, 1, i] = model.embedding[w, species[i]]
    end
    x0 = view(ws.x[1], :, :, atoms)
    linear!(x0, model.up[1], h0)
    M0, dens0 = view(ws.M[1], :, :, atoms), view(ws.density, atoms, 1)
    _messages!(M0, dens0, ws, model, 1, x0, nl)
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
    _messages!(M1, dens1, ws, model, 2, x1, nl)
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
    g = view(ws.g, :, :, atoms)
    linear!(g, model.readout1[1], h2)
    for q in eachindex(g)
        g[q] = normalized_silu(g[q], model.readout_cst)[1]
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
