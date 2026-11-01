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
