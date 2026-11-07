# The per-node stages of the forward pass on the device: the embedding, the density normalization,
# the skip connection, the nonlinear readout's activation and the energy of each atom.

@kernel function _embedding_kernel!(h0, @Const(embedding), @Const(species))
    w, i = @index(Global, NTuple)
    _embed!(h0, embedding, species, w, i)
end

# h0[w, i, 1] = embedding[w, species[i]]: the one-hot species embedding.
@inline function _embed!(h0, embedding, species, w, i)
    h0[w, i, 1] = embedding[w, species[i]]
    return nothing
end

@kernel function _normalize_kernel!(L, @Const(density))
    u, i, k = @index(Global, NTuple)
    _normalize!(L, density, u, i, k)
end

# L[u, i, k] /= density[i] + 1 (`_normalized_linear!`).
@inline function _normalize!(L, density, u, i, k)
    L[u, i, k] /= density[i] + one(eltype(L))
    return nothing
end

@kernel function _add_kernel!(y, @Const(x))
    q = @index(Global)
    _add!(y, x, q)
end

@inline function _add!(y, x, q)
    y[q] += x[q]
    return nothing
end

@kernel function _readout_silu_kernel!(g, @Const(gpre), cst)
    q = @index(Global)
    _readout_silu!(g, gpre, cst, q)
end

# g = cst·z·σ(z) of z = gpre[q] (`normalized_silu`).
@inline function _readout_silu!(g, gpre, cst, q)
    z = gpre[q]
    σ = inv(one(z) + exp(-z))
    g[q] = z * σ * cst
    return nothing
end

@kernel function _node_energy_kernel!(energies, @Const(e0), @Const(species), @Const(pair), @Const(e1), @Const(e2), scale, shift)
    i = @index(Global)
    _node_energy!(energies, e0, species, pair, e1, e2, scale, shift, i)
end

# E0 of the atom's element plus scale·((pair + e1) + e2) + shift (`_node_energy`).
@inline function _node_energy!(energies, e0, species, pair, e1, e2, scale, shift, i)
    energies[i] = e0[species[i]] + (scale * ((pair[i] + e1[1, i, 1]) + e2[1, i, 1]) + shift)
    return nothing
end
