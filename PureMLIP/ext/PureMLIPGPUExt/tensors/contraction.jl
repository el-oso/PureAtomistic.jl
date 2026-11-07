# The symmetric contraction on the device as matrix products, in the `(C, n, K)` node layout.

"""
    _symmetric_contraction!(B, dcm::DeviceContractionMatrices, X, species, n, Zf, Yc, backend) -> B

The matrix-product [`symmetric_contraction!`](@ref PureMLIP.Tensors.symmetric_contraction!) of
the first `n` nodes of `X` (`C × n × L`) into `B` (`C × n × K_out`), in blocks of `size(Zf, 2)`
nodes: the features of every node and channel of a block (`Zf`, features × nodes × channels), then
per species of `dcm` and channel one matrix product with the forward matrix into `Yc`, and the
contraction with `X` for the block's nodes of that species. With one species every node takes it.
"""
function _symmetric_contraction!(
        B::AbstractArray{T, 3}, dcm::DeviceContractionMatrices{T}, X::AbstractArray{T, 3}, species::AbstractVector{Int32}, n::Int,
        Zf::AbstractArray{T, 3}, Yc::AbstractArray{T, 3}, backend::Backend
    ) where {T}
    C, L, K = dcm.C, dcm.L, dcm.K_out
    nf, ncol = size(dcm.F, 1), size(dcm.F, 2)
    blk = size(Zf, 2)
    (size(X, 1), size(X, 3)) == (C, L) && size(B) == (C, size(X, 2), K) && n <= size(X, 2) ||
        throw(DimensionMismatch(lazy"gpu: symmetric contraction from $(size(X)) to $(size(B)) for $n nodes, expected ($C, n, $L) to ($C, n, $K)"))
    size(Zf) == (nf, blk, C) && size(Yc) == (ncol, blk, C) ||
        throw(DimensionMismatch(lazy"gpu: contraction buffers of sizes $(size(Zf)) and $(size(Yc)), expected ($nf, $blk, $C) and ($ncol, $blk, $C)"))
    single = length(dcm.species) == 1
    for a0 in 0:blk:(n - 1)
        nblk = min(blk, n - a0)
        _launch!(_features_kernel!, backend, (nblk, C), Zf, X, dcm.pairs, L, a0)
        for (j, s) in pairs(dcm.species)
            for c in 1:C
                mul!(view(Yc, :, 1:nblk, c), transpose(view(dcm.F, :, :, c, j)), view(Zf, :, 1:nblk, c))
            end
            _launch!(_contract_kernel!, backend, (C, K, nblk), B, Yc, X, species, single ? 0 : s, L, a0)
        end
    end
    return B
end

@kernel function _features_kernel!(Zf, @Const(X), @Const(pairs), L, a0)
    r, c = @index(Global, NTuple)
    _features!(Zf, X, pairs, L, r, c, a0 + r)
end

# The features of node `a` and channel `c` into `Zf[:, r, c]`: the products X[c, a, x] X[c, a, i]
# of the pairs, the values X[c, a, x], and 1 (`ContractionMatrices`).
@inline function _features!(Zf, X, pairs, L, r, c, a)
    T = eltype(Zf)
    np = size(pairs, 2)
    for q in 1:np
        Zf[q, r, c] = X[c, a, pairs[1, q]] * X[c, a, pairs[2, q]]
    end
    for x in 1:L
        Zf[np + x, r, c] = X[c, a, x]
    end
    Zf[np + L + 1, r, c] = one(T)
    return nothing
end

@kernel function _contract_kernel!(B, @Const(Yc), @Const(X), @Const(species), s, L, a0)
    c, j, r = @index(Global, NTuple)
    _contract!(B, Yc, X, species, s, L, c, j, r, a0 + r)
end

# B[c, a, j] = Σ_w Yc[(j - 1)L + w, r, c] X[c, a, w] for node `a` (block column `r`) when its
# species is `s`, or for every node when `s` is 0.
@inline function _contract!(B, Yc, X, species, s, L, c, j, r, a)
    (iszero(s) || species[a] == s) || return nothing
    acc = zero(eltype(B))
    for w in 1:L
        acc += Yc[(j - 1) * L + w, r, c] * X[c, a, w]
    end
    B[c, a, j] = acc
    return nothing
end
