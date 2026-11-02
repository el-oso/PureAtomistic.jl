"""
    Contraction{T}(l, U1, U2, U3, Wmax, W0, W1)

One output irrep `l` (`2l + 1` components) of a MACE symmetric contraction of correlation 3.
`U1`, `U2`, `U3` are the stored `U_matrix_1..3` buffers in torch index order, with a leading
`m` index of length `2l + 1` when `l > 0` and none when `l = 0`; `Wmax`, `W0`, `W1` are
`weights_max`, `weights.0`, `weights.1` in torch order `[n_species, K, C]`. The U arrays are kept
as coordinate lists of their nonzero entries, `((m, w, x, i, k), value)` for `U3`, with `m = 1`
when `l = 0`; the weights are kept as `[C, K, n_species]` so the channel index is innermost.
"""
struct Contraction{T}
    l::Int
    U1::Vector{Tuple{NTuple{3, Int}, T}}   # (m, w, k)
    U2::Vector{Tuple{NTuple{4, Int}, T}}   # (m, w, x, k)
    U3::Vector{Tuple{NTuple{5, Int}, T}}   # (m, w, x, i, k)
    Wmax::Array{T, 3}
    W0::Array{T, 3}
    W1::Array{T, 3}
    n_ell::Int
    function Contraction{T}(
            l::Int, U1::AbstractArray{<:Real}, U2::AbstractArray{<:Real}, U3::AbstractArray{<:Real},
            Wmax::AbstractArray{<:Real, 3}, W0::AbstractArray{<:Real, 3}, W1::AbstractArray{<:Real, 3}
        ) where {T}
        l >= 0 || throw(ArgumentError(lazy"tensors: symmetric contraction: l must be non-negative, got $l"))
        Base.require_one_based_indexing(U1, U2, U3, Wmax, W0, W1)
        nm = 2l + 1
        lead = l > 0
        if lead && !(ndims(U1) == 3 && ndims(U2) == 4 && ndims(U3) == 5 && size(U1, 1) == size(U2, 1) == size(U3, 1) == nm)
            throw(ArgumentError(lazy"tensors: symmetric contraction: l = $l needs U arrays with a leading index of length $nm"))
        end
        if !lead && !(ndims(U1) == 2 && ndims(U2) == 3 && ndims(U3) == 4)
            throw(ArgumentError("tensors: symmetric contraction: l = 0 needs U arrays of 2, 3 and 4 dimensions"))
        end
        s1, s2, s3 = _tail(size(U1), lead), _tail(size(U2), lead), _tail(size(U3), lead)
        L = s1[1]
        m_ = lead ? string(nm, ", ") : ""
        s2[1] == s2[2] == L || throw(DimensionMismatch(lazy"tensors: symmetric contraction: U_matrix_2 has size $(size(U2)), expected ($m_$L, $L, K)"))
        s3[1] == s3[2] == s3[3] == L || throw(DimensionMismatch(lazy"tensors: symmetric contraction: U_matrix_3 has size $(size(U3)), expected ($m_$L, $L, $L, K)"))
        _check_weights(U3, "U_matrix_3", s3[4], Wmax, "weights_max")
        _check_weights(U2, "U_matrix_2", s2[3], W0, "weights.0")
        _check_weights(U1, "U_matrix_1", s1[2], W1, "weights.1")
        for (W, name) in ((W0, "weights.0"), (W1, "weights.1"))
            (size(W, 1), size(W, 3)) == (size(Wmax, 1), size(Wmax, 3)) ||
                throw(DimensionMismatch(lazy"tensors: symmetric contraction: $name has size $(size(W)), expected $((size(Wmax, 1), size(W, 2), size(Wmax, 3)))"))
        end
        return new{T}(
            l, _nonzeros(T, U1, Val(3)), _nonzeros(T, U2, Val(4)), _nonzeros(T, U3, Val(5)),
            permutedims(Wmax, (3, 2, 1)), permutedims(W0, (3, 2, 1)), permutedims(W1, (3, 2, 1)), L
        )
    end
end

_tail(sz::Tuple, lead::Bool) = lead ? Base.tail(sz) : sz

function _check_weights(U, uname::String, K::Int, W, wname::String)
    size(W, 2) == K || throw(DimensionMismatch(lazy"tensors: symmetric contraction: $uname has size $(size(U)), $wname has size $(size(W))"))
    return nothing
end

# Nonzero entries of `U` as `(coordinates, value)` with `K` coordinates; a `U` of `K - 1`
# dimensions has no leading `m` index and gets coordinate `m = 1`.
function _nonzeros(::Type{T}, U::AbstractArray{<:Real}, ::Val{K}) where {T, K}
    out = Tuple{NTuple{K, Int}, T}[]
    for I in CartesianIndices(U)
        v = U[I]
        iszero(v) && continue
        push!(out, (_lead_m(Tuple(I), Val(K)), T(v)))
    end
    return out
end

_lead_m(t::NTuple{K, Int}, ::Val{K}) where {K} = t
_lead_m(t::Tuple{Vararg{Int}}, ::Val{K}) where {K} = (1, t...)::NTuple{K, Int}

"""
    SymmetricContraction{T}(contractions)

The contractions of one MACE product block, in output order. Every contraction has the same
channel count `C`, ell count `L` and species count. The output `B` has `C × K_out` entries per
node, `K_out = Σ (2l + 1)`, with the components of contraction `j` following those of `j - 1`.
"""
struct SymmetricContraction{T}
    contractions::Vector{Contraction{T}}
    function SymmetricContraction{T}(contractions::Vector{Contraction{T}}) where {T}
        isempty(contractions) && throw(ArgumentError("tensors: symmetric contraction needs at least one contraction"))
        c1 = first(contractions)
        for ct in contractions
            (size(ct.Wmax, 1), ct.n_ell, size(ct.Wmax, 3)) == (size(c1.Wmax, 1), c1.n_ell, size(c1.Wmax, 3)) ||
                throw(DimensionMismatch("tensors: symmetric contraction: contractions see different channel, ell or species counts"))
        end
        return new{T}(contractions)
    end
end

"""
    n_outputs(sc) -> Int

The size of the second dimension of the output `B` of [`symmetric_contraction!`](@ref):
`2l + 1` summed over the contractions of `sc`.
"""
n_outputs(sc::SymmetricContraction) = sum(ct -> 2ct.l + 1, sc.contractions)

"""
    symmetric_contraction_scratch_size(sc) -> Int

Length of the `scratch` vector that [`symmetric_contraction!`](@ref) and
[`symmetric_contraction_pullback!`](@ref) need: the per-node `C2` and `C1` buffers of the
largest contraction.
"""
function symmetric_contraction_scratch_size(sc::SymmetricContraction)
    n = 0
    for ct in sc.contractions
        C, M, L = size(ct.Wmax, 1), 2ct.l + 1, ct.n_ell
        n = max(n, C * M * L * L + C * M * L)
    end
    return n
end

function _check_args(sc::SymmetricContraction, B, bname::String, X, species, scratch)
    Base.require_one_based_indexing(B, X, species, scratch)
    ct = first(sc.contractions)
    C, L, n = size(ct.Wmax, 1), ct.n_ell, size(X, 3)
    (size(X, 1), size(X, 2)) == (C, L) ||
        throw(DimensionMismatch(lazy"tensors: symmetric contraction expects X of size ($C, $L, n), got $(size(X))"))
    size(B) == (C, n_outputs(sc), n) ||
        throw(DimensionMismatch(lazy"tensors: symmetric contraction expects $bname of size $((C, n_outputs(sc), n)), got $(size(B))"))
    length(species) == n || throw(DimensionMismatch(lazy"tensors: symmetric contraction expects $n species, got $(length(species))"))
    need = symmetric_contraction_scratch_size(sc)
    length(scratch) >= need ||
        throw(DimensionMismatch(lazy"tensors: symmetric contraction scratch has $(length(scratch)) elements, needs $need"))
    nspecies = size(ct.Wmax, 3)
    for s in species
        1 <= s <= nspecies || throw(ArgumentError(lazy"tensors: species index $s is outside 1:$nspecies"))
    end
    return nothing
end

# Fills `scratch` with the node-`a` buffers of contraction `ct` for species `s`:
# `C2[c, m, w, x]` at `I2[c, m, w, x]`, then `C1[c, m, w]` at `o1 + I1[c, m, w]`.
function _c2_c1!(scratch::AbstractArray{S}, ct::Contraction, X::AbstractArray{S, 3}, a::Int, s::Int) where {S}
    C, M, L = size(ct.Wmax, 1), 2ct.l + 1, ct.n_ell
    I2 = LinearIndices((C, M, L, L))
    I1 = LinearIndices((C, M, L))
    o1 = length(I2)
    for j in 1:(o1 + length(I1))
        scratch[j] = zero(S)
    end
    for ((m, w, x, k), v) in ct.U2, c in 1:C
        scratch[I2[c, m, w, x]] += v * ct.W0[c, k, s]
    end
    for ((m, w, x, i, k), v) in ct.U3
        C2 = view(scratch, I2[1, m, w, x]:I2[C, m, w, x])
        Wk, Xi = view(ct.Wmax, :, k, s), view(X, :, i, a)
        # In range: c comes from `checked_eachindex` over exactly the arrays indexed, each a checked view.
        @inbounds for c in checked_eachindex(C2, Wk, Xi)
            C2[c] += v * Wk[c] * Xi[c]
        end
    end
    for ((m, w, k), v) in ct.U1, c in 1:C
        scratch[o1 + I1[c, m, w]] += v * ct.W1[c, k, s]
    end
    # A2 contracts the last ell index of C2 with X. B alone would not change if it contracted
    # the first (Σ_{w,x} C2[w, x] X[w] X[x] is symmetric), but the pullback uses C1 itself.
    for x in 1:L, w in 1:L, m in 1:M
        C1 = view(scratch, (o1 + I1[1, m, w]):(o1 + I1[C, m, w]))
        C2wx = view(scratch, I2[1, m, w, x]:I2[C, m, w, x])
        Xx = view(X, :, x, a)
        # In range: c comes from `checked_eachindex` over exactly the arrays indexed, each a checked view.
        @inbounds for c in checked_eachindex(C1, C2wx, Xx)
            C1[c] += C2wx[c] * Xx[c]
        end
    end
    return I2, I1, o1
end

"""
    symmetric_contraction!(B, sc, X, species, scratch) -> B

MACE symmetric contraction of correlation 3. `X` is `C × L × n` (channel, e3nn ell index
`l² + m`, node), `species` the 1-based species index of each node, `B` is `C × K_out × n`. For
contraction `ct` of order `l`, with `s = species[a]` and the U and W of `ct`:

    C2[c, m, w, x] = Σ_k U2[m, w, x, k] W0[s, k, c] + Σ_{i,k} U3[m, w, x, i, k] Wmax[s, k, c] X[c, i, a]
    C1[c, m, w]    = Σ_k U1[m, w, k] W1[s, k, c] + Σ_x C2[c, m, w, x] X[c, x, a]
    B[c, o + m, a] = Σ_w C1[c, m, w] X[c, w, a]

where `o` is the number of components of the contractions before `ct`. `scratch` holds at least
[`symmetric_contraction_scratch_size`](@ref)`(sc)` elements of the type of `X`.

As einsums, with `z[a, s] = (s == species[a])`, `s` the species, `k` the coupling-tensor component and
`m, w, x, i` the indices of the `U` arrays (`m` is 1 for `l = 0`): `B[c, m, a]` is the sum of
`ein"mwk,skc,as,cwa->cma"(U1, W1, z, X)`, `ein"mwxk,skc,as,cxa,cwa->cma"(U2, W0, z, X, X)` and
`ein"mwxik,skc,as,cia,cxa,cwa->cma"(U3, Wmax, z, X, X, X)`. The `U` arrays are stored as nonzero entries.
"""
function symmetric_contraction!(
        B::AbstractArray{S, 3}, sc::SymmetricContraction, X::AbstractArray{S, 3},
        species::AbstractVector{Int}, scratch::AbstractArray{S}
    ) where {S}
    _check_args(sc, B, "B", X, species, scratch)
    for a in axes(X, 3)
        s = species[a]
        o = 0
        for ct in sc.contractions
            _, I1, o1 = _c2_c1!(scratch, ct, X, a, s)
            C, M, L = size(ct.Wmax, 1), 2ct.l + 1, ct.n_ell
            for m in 1:M, c in 1:C
                B[c, o + m, a] = zero(S)
            end
            for w in 1:L, m in 1:M, c in 1:C
                B[c, o + m, a] += scratch[o1 + I1[c, m, w]] * X[c, w, a]
            end
            o += M
        end
    end
    return B
end

"""
    symmetric_contraction_pullback!(dX, sc, dB, X, species, scratch) -> dX

Overwrites `dX` with `Σ dB .* ∂B/∂X` for `B` of [`symmetric_contraction!`](@ref) at `X`.
`C2` and `C1` are recomputed per node in `scratch`.

As an einsum, `dX` is the sum, over each `X` factor of each of the three einsums of
[`symmetric_contraction!`](@ref), of that einsum with the factor dropped, `dB[c, m, a]` (labels `cma`)
as an extra operand, and the factor's own labels (`cwa`, `cxa` or `cia`) as the output.
"""
function symmetric_contraction_pullback!(
        dX::AbstractArray{S, 3}, sc::SymmetricContraction, dB::AbstractArray{S, 3}, X::AbstractArray{S, 3},
        species::AbstractVector{Int}, scratch::AbstractArray{S}
    ) where {S}
    _check_args(sc, dB, "dB", X, species, scratch)
    Base.require_one_based_indexing(dX)
    size(dX) == size(X) ||
        throw(DimensionMismatch(lazy"tensors: symmetric contraction expects dX of size $(size(X)), got $(size(dX))"))
    fill!(dX, zero(S))
    for a in axes(X, 3)
        s = species[a]
        o = 0
        for ct in sc.contractions
            I2, I1, o1 = _c2_c1!(scratch, ct, X, a, s)
            C, M, L = size(ct.Wmax, 1), 2ct.l + 1, ct.n_ell
            # B = Σ_w C1 X[w]:  dX[w] += dB C1[w], and dC1[m, w] = dB[m] X[w].
            for w in 1:L, m in 1:M
                dXw, dBm = view(dX, :, w, a), view(dB, :, o + m, a)
                C1 = view(scratch, (o1 + I1[1, m, w]):(o1 + I1[C, m, w]))
                for c in checked_eachindex(dXw, dBm, C1)
                    dXw[c] += dBm[c] * C1[c]
                end
            end
            # C1 ∋ Σ_x C2[w, x] X[x]:  dX[x] += dC1[m, w] C2[m, w, x], and dC2[m, w, x] = dC1[m, w] X[x].
            for x in 1:L, w in 1:L, m in 1:M
                dXx, dBm, Xw = view(dX, :, x, a), view(dB, :, o + m, a), view(X, :, w, a)
                C2 = view(scratch, I2[1, m, w, x]:I2[C, m, w, x])
                for c in checked_eachindex(dXx, dBm, Xw, C2)
                    dXx[c] += dBm[c] * Xw[c] * C2[c]
                end
            end
            # C2 ∋ U3 Wmax X[i]:  dX[i] += U3 Wmax dC2.
            for ((m, w, x, i, k), v) in ct.U3
                dXi, Wk, dBm = view(dX, :, i, a), view(ct.Wmax, :, k, s), view(dB, :, o + m, a)
                Xw, Xx = view(X, :, w, a), view(X, :, x, a)
                # In range: c comes from `checked_eachindex` over exactly the arrays indexed, each a checked view.
                @inbounds for c in checked_eachindex(dXi, Wk, dBm, Xw, Xx)
                    dXi[c] += v * Wk[c] * dBm[c] * Xw[c] * Xx[c]
                end
            end
            o += M
        end
    end
    return dX
end

"""
    ContractionMatrices{T}(sc; forward = true)

A [`SymmetricContraction`](@ref) `sc` as dense matrices per species and channel, for the forms of
[`symmetric_contraction!`](@ref) and [`symmetric_contraction_pullback!`](@ref) that run as matrix
products over groups of atoms. The matrices of species `s` are built by
[`contraction_matrices!`](@ref); a new value holds none. With `forward = false` only the pullback
matrices are built.

For one atom of species `s` and one channel `c`, with `L` ell indices and `K_out` outputs, the
`nf = L(L + 1)/2 + L + 1` features are the products `X[c, x] X[c, i]` for `x ≤ i` (in the order
`i = 1..L`, `x = 1..i`), the values `X[c, x]`, and 1. Column `(j - 1)L + w` belongs to output `j`
and ell index `w`. The forward matrix `F[:, :, c]` (`nf × K_out·L`) folds `U3·Wmax`, `U2·W0`
and `U1·W1`, so that with `y = F[:, :, c]ᵀ z` for the features `z`

    B[c, j] = Σ_w y[(j - 1)L + w] X[c, w]

The pullback matrix `G[:, :, c]` folds the three terms of `∂B/∂X` (one per `X` factor), so that
with `g = G[:, :, c]ᵀ z`

    dX[c, w] = Σ_j dB[c, j] g[(j - 1)L + w]

Each matrix of a species holds `nf · K_out·L · C` elements; for MACE-MPA-0 about one tenth of
them are nonzero.
"""
struct ContractionMatrices{T}
    C::Int
    L::Int
    K_out::Int
    n_species::Int
    forward::Bool
    pairs::Vector{NTuple{2, Int}}    # (x, i) of feature q, x ≤ i
    F::Vector{Array{T, 3}}           # per species: nf × K_out·L × C, or 0 × 0 × 0 until built
    G::Vector{Array{T, 3}}
    function ContractionMatrices{T}(sc::SymmetricContraction; forward::Bool = true) where {T}
        ct = first(sc.contractions)
        C, L, ns = size(ct.Wmax, 1), ct.n_ell, size(ct.Wmax, 3)
        pairs = [(x, i) for i in 1:L for x in 1:i]
        empty() = [Array{T, 3}(undef, 0, 0, 0) for _ in 1:ns]
        return new{T}(C, L, n_outputs(sc), ns, forward, pairs, empty(), empty())
    end
end

# Feature count and column count of the matrices.
_n_features(cm::ContractionMatrices) = length(cm.pairs) + cm.L + 1
_n_columns(cm::ContractionMatrices) = cm.K_out * cm.L

# Feature row of the product X[x] X[i]; the pairs are ordered by i, then x.
_pair_row(x::Int, i::Int) = x <= i ? div(i * (i - 1), 2) + x : div(x * (x - 1), 2) + i

"""
    contraction_matrices!(cm, sc, s) -> cm

Builds the matrices of species `s` in `cm` from the contraction `sc` that `cm` was made from, unless
they are built already. Building allocates the matrices; a call for a built species does not
allocate.
"""
function contraction_matrices!(cm::ContractionMatrices{T}, sc::SymmetricContraction{T}, s::Int) where {T}
    1 <= s <= cm.n_species || throw(ArgumentError(lazy"tensors: species index $s is outside 1:$(cm.n_species)"))
    isempty(cm.G[s]) || return cm
    ct1 = first(sc.contractions)
    (size(ct1.Wmax, 1), ct1.n_ell, n_outputs(sc), size(ct1.Wmax, 3)) == (cm.C, cm.L, cm.K_out, cm.n_species) ||
        throw(DimensionMismatch("tensors: contraction matrices and symmetric contraction differ in channels, ells, outputs or species"))
    C, L = cm.C, cm.L
    np, nf = length(cm.pairs), _n_features(cm)
    F = zeros(T, cm.forward ? nf : 0, cm.forward ? _n_columns(cm) : 0, cm.forward ? C : 0)
    G = zeros(T, nf, _n_columns(cm), C)
    col(j, w) = (j - 1) * L + w
    o = 0
    for ct in sc.contractions
        for ((m, w, x, i, k), v) in ct.U3, c in 1:C
            u = v * ct.Wmax[c, k, s]
            cm.forward && (F[_pair_row(x, i), col(o + m, w), c] += u)
            G[_pair_row(x, i), col(o + m, w), c] += u
            G[_pair_row(w, i), col(o + m, x), c] += u
            G[_pair_row(w, x), col(o + m, i), c] += u
        end
        for ((m, w, x, k), v) in ct.U2, c in 1:C
            u = v * ct.W0[c, k, s]
            cm.forward && (F[np + x, col(o + m, w), c] += u)
            G[np + x, col(o + m, w), c] += u
            G[np + w, col(o + m, x), c] += u
        end
        for ((m, w, k), v) in ct.U1, c in 1:C
            u = v * ct.W1[c, k, s]
            cm.forward && (F[nf, col(o + m, w), c] += u)
            G[nf, col(o + m, w), c] += u
        end
        o += 2ct.l + 1
    end
    cm.F[s] = F
    cm.G[s] = G
    return cm
end

"""
    ContractionBuffers{T}(cms...; atoms = 64)

Scratch for the matrix-product forms of [`symmetric_contraction!`](@ref) and
[`symmetric_contraction_pullback!`](@ref) with any of the [`ContractionMatrices`](@ref) `cms`:
the features of up to `atoms` atoms for every channel, the matrix-product outputs, and the atom
indices of one group.
"""
struct ContractionBuffers{T}
    Z::Array{T, 3}                   # features × channels × atoms
    Y::Array{T, 3}                   # columns × channels × atoms
    atoms::Vector{Int}
    function ContractionBuffers{T}(cms::ContractionMatrices...; atoms::Integer = 64) where {T}
        atoms > 0 || throw(ArgumentError(lazy"tensors: contraction buffers need a positive atom count, got $atoms"))
        nf = nc = C = 0
        for cm in cms
            nf, nc, C = max(nf, _n_features(cm)), max(nc, _n_columns(cm)), max(C, cm.C)
        end
        return new{T}(zeros(T, nf, C, atoms), zeros(T, nc, C, atoms), zeros(Int, atoms))
    end
end

# The argument checks of the matrix-product kernels.
function _check_matrix_args(cm::ContractionMatrices, B, bname::String, X, species, buf::ContractionBuffers)
    Base.require_one_based_indexing(B, X, species)
    C, L, n = cm.C, cm.L, size(X, 3)
    (size(X, 1), size(X, 2)) == (C, L) ||
        throw(DimensionMismatch(lazy"tensors: symmetric contraction expects X of size ($C, $L, n), got $(size(X))"))
    size(B) == (C, cm.K_out, n) ||
        throw(DimensionMismatch(lazy"tensors: symmetric contraction expects $bname of size $((C, cm.K_out, n)), got $(size(B))"))
    length(species) == n || throw(DimensionMismatch(lazy"tensors: symmetric contraction expects $n species, got $(length(species))"))
    (size(buf.Z, 1) >= _n_features(cm) && size(buf.Y, 1) >= _n_columns(cm) && size(buf.Z, 2) >= C && size(buf.Y, 2) >= C) ||
        throw(DimensionMismatch("tensors: contraction buffers are smaller than the contraction matrices"))
    for s in species
        1 <= s <= cm.n_species || throw(ArgumentError(lazy"tensors: species index $s is outside 1:$(cm.n_species)"))
        isempty(cm.G[s]) && throw(ArgumentError(lazy"tensors: species $s has no contraction matrices"))
    end
    return nothing
end

# The features of the atoms of one group, for every channel, into Z[:, c, r].
function _features!(Z::AbstractArray{S, 3}, cm::ContractionMatrices, X::AbstractArray{S, 3}, atoms::AbstractVector{Int}) where {S}
    np = length(cm.pairs)
    nf = _n_features(cm)
    C = cm.C
    for (r, a) in enumerate(atoms)
        for (q, (x, i)) in enumerate(cm.pairs)
            for c in 1:C
                Z[q, c, r] = X[c, x, a] * X[c, i, a]
            end
        end
        for x in 1:cm.L, c in 1:C
            Z[np + x, c, r] = X[c, x, a]
        end
        for c in 1:C
            Z[nf, c, r] = one(S)
        end
    end
    return nothing
end

# Y[:, c, r] = M[:, :, c]ᵀ Z[:, c, r] for each channel c over the first n atoms: one matrix product
# per channel on the strided channel slices of the buffers.
function _channel_products!(Y::AbstractArray{S, 3}, M::AbstractArray{S, 3}, Z::AbstractArray{S, 3}, cm::ContractionMatrices, n::Int) where {S}
    nf, nc = _n_features(cm), _n_columns(cm)
    for c in 1:cm.C
        mul!(view(Y, 1:nc, c, 1:n), transpose(view(M, :, :, c)), view(Z, 1:nf, c, 1:n))
    end
    return nothing
end

# B of the n atoms in `buf.atoms` (all of species s) from their features and the forward matrices.
function _forward_group!(B::AbstractArray{S, 3}, cm::ContractionMatrices{S}, X::AbstractArray{S, 3}, buf::ContractionBuffers{S}, s::Int, n::Int) where {S}
    C, L = cm.C, cm.L
    atoms = view(buf.atoms, 1:n)
    _features!(buf.Z, cm, X, atoms)
    _channel_products!(buf.Y, cm.F[s], buf.Z, cm, n)
    for (r, a) in enumerate(atoms), j in 1:cm.K_out
        for c in 1:C
            B[c, j, a] = zero(S)
        end
        for w in 1:L, c in 1:C
            B[c, j, a] += buf.Y[(j - 1) * L + w, c, r] * X[c, w, a]
        end
    end
    return nothing
end

# dX of the n atoms in `buf.atoms` (all of species s) from their features and the pullback matrices.
function _pullback_group!(
        dX::AbstractArray{S, 3}, cm::ContractionMatrices{S}, dB::AbstractArray{S, 3}, X::AbstractArray{S, 3},
        buf::ContractionBuffers{S}, s::Int, n::Int
    ) where {S}
    C, L = cm.C, cm.L
    atoms = view(buf.atoms, 1:n)
    _features!(buf.Z, cm, X, atoms)
    _channel_products!(buf.Y, cm.G[s], buf.Z, cm, n)
    for (r, a) in enumerate(atoms), w in 1:L
        for c in 1:C
            dX[c, w, a] = zero(S)
        end
        for j in 1:cm.K_out, c in 1:C
            dX[c, w, a] += dB[c, j, a] * buf.Y[(j - 1) * L + w, c, r]
        end
    end
    return nothing
end

"""
    symmetric_contraction!(B, cm::ContractionMatrices, X, species, buf::ContractionBuffers) -> B

[`symmetric_contraction!`](@ref) as matrix products: the atoms of each species, in order and in
groups of up to `length(buf.atoms)`, have their features built for every channel, multiplied by
the forward matrix of each channel, and contracted with `X`. Every species in `species` must have
its matrices built in `cm`, which must hold forward matrices. The results agree with the per-atom
form up to rounding.
"""
function symmetric_contraction!(
        B::AbstractArray{S, 3}, cm::ContractionMatrices{S}, X::AbstractArray{S, 3},
        species::AbstractVector{Int}, buf::ContractionBuffers{S}
    ) where {S}
    cm.forward || throw(ArgumentError("tensors: these contraction matrices hold no forward matrices"))
    _check_matrix_args(cm, B, "B", X, species, buf)
    nb = length(buf.atoms)
    for s in 1:cm.n_species
        isempty(cm.G[s]) && continue
        n = 0
        for a in eachindex(species)
            species[a] == s || continue
            n += 1
            buf.atoms[n] = a
            if n == nb
                _forward_group!(B, cm, X, buf, s, n)
                n = 0
            end
        end
        n > 0 && _forward_group!(B, cm, X, buf, s, n)
    end
    return B
end

"""
    symmetric_contraction_pullback!(dX, cm::ContractionMatrices, dB, X, species, buf::ContractionBuffers) -> dX

[`symmetric_contraction_pullback!`](@ref) as matrix products with the pullback matrices of `cm`, in
the groups of the matrix-product [`symmetric_contraction!`](@ref). It does not recompute the
forward pass. Every species in `species` must have its matrices built in `cm`.
"""
function symmetric_contraction_pullback!(
        dX::AbstractArray{S, 3}, cm::ContractionMatrices{S}, dB::AbstractArray{S, 3}, X::AbstractArray{S, 3},
        species::AbstractVector{Int}, buf::ContractionBuffers{S}
    ) where {S}
    _check_matrix_args(cm, dB, "dB", X, species, buf)
    Base.require_one_based_indexing(dX)
    size(dX) == size(X) ||
        throw(DimensionMismatch(lazy"tensors: symmetric contraction expects dX of size $(size(X)), got $(size(dX))"))
    nb = length(buf.atoms)
    for s in 1:cm.n_species
        isempty(cm.G[s]) && continue
        n = 0
        for a in eachindex(species)
            species[a] == s || continue
            n += 1
            buf.atoms[n] = a
            if n == nb
                _pullback_group!(dX, cm, dB, X, buf, s, n)
                n = 0
            end
        end
        n > 0 && _pullback_group!(dX, cm, dB, X, buf, s, n)
    end
    return dX
end
