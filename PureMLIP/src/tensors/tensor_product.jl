"""
    ChannelwiseTP{T}(in1_ls, in2_ls, out_ls, instructions)

e3nn `TensorProduct` in `uvu` mode with one weight per channel and instruction: channel `u` of
the first input couples only to channel `u` of the output. Slots have degrees `in1_ls`,
`in2_ls` and `out_ls`; `instructions` lists `(in1 slot, in2 slot, out slot, path weight)`,
1-based. The coupling tensor of an instruction is `path weight · e3nn_wigner_3j(l1, l2, l3)`,
stored as its nonzero entries.
"""
struct ChannelwiseTP{T}
    paths::Vector{UnitRange{Int}}             # entries of instruction p
    entries::Vector{Tuple{Int, Int, Int, T}}  # (i, j, k, pw·C[i, j, k]), 1-based components of x, Y and the output
    K_in1::Int
    K_in2::Int
    K_out::Int
    function ChannelwiseTP{T}(
            in1_ls::Vector{Int}, in2_ls::Vector{Int}, out_ls::Vector{Int},
            instructions::Vector{Tuple{Int, Int, Int, Float64}}
        ) where {T}
        off1, off2, off3 = _slot_offsets(in1_ls), _slot_offsets(in2_ls), _slot_offsets(out_ls)
        paths = UnitRange{Int}[]
        entries = Tuple{Int, Int, Int, T}[]
        for (a, b, c, pw) in instructions
            (a in eachindex(in1_ls) && b in eachindex(in2_ls) && c in eachindex(out_ls)) ||
                throw(ArgumentError(lazy"tensors: tensor-product instruction ($a, $b, $c) is outside the $(length(in1_ls)), $(length(in2_ls)) and $(length(out_ls)) slots"))
            C = e3nn_wigner_3j(in1_ls[a], in2_ls[b], out_ls[c])
            start = length(entries) + 1
            for I in CartesianIndices(C)
                iszero(C[I]) && continue
                i, j, k = Tuple(I)
                push!(entries, (off1[a] + i, off2[b] + j, off3[c] + k, T(pw * C[I])))
            end
            push!(paths, start:length(entries))
        end
        K(ls) = sum(l -> 2l + 1, ls; init = 0)
        return new{T}(paths, entries, K(in1_ls), K(in2_ls), K(out_ls))
    end
end

function _check_tp(tp::ChannelwiseTP, M::AbstractMatrix, x::AbstractMatrix, Y::AbstractVector, w::AbstractMatrix, names::NTuple{4, String})
    Base.require_one_based_indexing(M, x, Y, w)
    C = size(M, 1)
    size(x) == (C, tp.K_in1) ||
        throw(DimensionMismatch(lazy"tensors: tensor product with $C channels: $(names[1]) has size $(size(x)), expected $((C, tp.K_in1))"))
    length(Y) == tp.K_in2 ||
        throw(DimensionMismatch(lazy"tensors: tensor product with $C channels: $(names[2]) has length $(length(Y)), expected $(tp.K_in2)"))
    size(w) == (C, length(tp.paths)) ||
        throw(DimensionMismatch(lazy"tensors: tensor product with $C channels: $(names[3]) has size $(size(w)), expected $((C, length(tp.paths)))"))
    size(M, 2) == tp.K_out ||
        throw(DimensionMismatch(lazy"tensors: tensor product with $C channels: $(names[4]) has size $(size(M)), expected $((C, tp.K_out))"))
    return nothing
end

"""
    tp_edge_add!(M, tp::ChannelwiseTP, x, Y, w) -> M

Adds the message of one edge to `M` (`C × K_out`):
`M[u, k] += Σ_p w[u, p] Σ_{i,j} pw_p C_p[i, j, k] x[u, i] Y[j]`, with `x` (`C × K_in1`) the
sender features, `Y` (`K_in2`) the edge attributes and `w` (`C × n_instructions`) the weights.

As an einsum: `ein"up,ui,j,pijk->uk"(w, x, Y, Cpw)`, with `u` the channel, `p` the instruction and
`i`, `j`, `k` the components of `x`, `Y` and `M`; `Cpw[p, i, j, k]` is the dense form of the stored
entries (zero outside the slots of instruction `p`).
"""
function tp_edge_add!(M::AbstractMatrix{S}, tp::ChannelwiseTP, x::AbstractMatrix{S}, Y::AbstractVector{S}, w::AbstractMatrix{S}) where {S}
    _check_tp(tp, M, x, Y, w, ("x", "Y", "w", "the output"))
    for p in eachindex(tp.paths)
        wp = view(w, :, p)
        for q in tp.paths[p]
            i, j, k, v = tp.entries[q]
            vy = v * Y[j]
            Mk, xi = view(M, :, k), view(x, :, i)
            # In range: u comes from `checked_eachindex` over exactly the arrays indexed, each a checked view.
            @inbounds for u in checked_eachindex(Mk, wp, xi)
                Mk[u] += vy * wp[u] * xi[u]
            end
        end
    end
    return M
end

"""
    tp_edge_pullback_add!(dx, dY, dw, tp::ChannelwiseTP, dM, x, Y, w; need_dx = true) -> nothing

Adds the vector-Jacobian products of [`tp_edge_add!`](@ref) with the cotangent `dM` to `dx`,
`dY` and `dw`. With `need_dx = false`, `dx` is neither read nor written.

As einsums, with `Cpw` and the labels of [`tp_edge_add!`](@ref):
`dx += ein"up,uk,pijk,j->ui"(w, dM, Cpw, Y)`, `dY += ein"uk,up,ui,pijk->j"(dM, w, x, Cpw)` and
`dw += ein"uk,ui,pijk,j->up"(dM, x, Cpw, Y)`.
"""
function tp_edge_pullback_add!(
        dx::AbstractMatrix{S}, dY::AbstractVector{S}, dw::AbstractMatrix{S}, tp::ChannelwiseTP,
        dM::AbstractMatrix{S}, x::AbstractMatrix{S}, Y::AbstractVector{S}, w::AbstractMatrix{S};
        need_dx::Bool = true
    ) where {S}
    _check_tp(tp, dM, x, Y, w, ("x", "Y", "w", "the output"))
    _check_tp(tp, dM, dx, dY, dw, ("dx", "dY", "dw", "the output"))
    for p in eachindex(tp.paths)
        wp, dwp = view(w, :, p), view(dw, :, p)
        for q in tp.paths[p]
            i, j, k, v = tp.entries[q]
            vy = v * Y[j]
            dMk, xi = view(dM, :, k), view(x, :, i)
            acc = zero(S)
            if need_dx
                dxi = view(dx, :, i)
                @simd for u in checked_eachindex(dMk, wp, xi, dxi, dwp)
                    g = dMk[u]
                    dxi[u] += vy * g * wp[u]
                    dwp[u] += vy * g * xi[u]
                    acc += g * wp[u] * xi[u]
                end
            else
                @simd for u in checked_eachindex(dMk, wp, xi, dwp)
                    g = dMk[u]
                    dwp[u] += vy * g * xi[u]
                    acc += g * wp[u] * xi[u]
                end
            end
            dY[j] += v * acc
        end
    end
    return nothing
end
