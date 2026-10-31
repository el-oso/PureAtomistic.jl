# e3nn `Linear` and the species-indexed linear map of MACE's `skip_tp`, on node features stored
# as `(C, K, n)` arrays: channel fastest, then e3nn component, then node. A slot of degree `l`
# occupies `2l+1` consecutive components; slots follow each other in order.

# 0-based component offset of each slot.
_slot_offsets(ls::Vector{Int}) = cumsum([0; [2l + 1 for l in ls[1:(end - 1)]]])

"""
    E3nnLinear{T}(weight, in_ls, out_ls, instructions, C_in, C_out)

e3nn `Linear` between slots of degrees `in_ls` and `out_ls`, `C_in` and `C_out` channels per
slot. `instructions` lists `(input slot, output slot, path weight)`, 1-based, in weight order;
block `p` of `weight` is the torch `[C_in, C_out]` row-major segment `p`. Instructions sharing an
output slot add up; an output slot without an instruction is zero, and an input slot without
one is ignored.
"""
struct E3nnLinear{T}
    W::Vector{Matrix{T}}         # W[p][w, u] = torch block p [u, w]
    pw::Vector{T}
    slots::Vector{Tuple{Int, Int}}
    in_ls::Vector{Int}
    in_off::Vector{Int}
    out_off::Vector{Int}
    C_in::Int
    C_out::Int
    K_in::Int
    K_out::Int
    function E3nnLinear{T}(
            weight::AbstractVector{<:Real}, in_ls::Vector{Int}, out_ls::Vector{Int},
            instructions::Vector{Tuple{Int, Int, Float64}}, C_in::Int, C_out::Int
        ) where {T}
        Base.require_one_based_indexing(weight)
        blk = C_in * C_out
        length(weight) == length(instructions) * blk ||
            throw(ArgumentError(lazy"model: linear weight has $(length(weight)) entries, expected $(length(instructions) * blk)"))
        for (i, o, _) in instructions
            (i in eachindex(in_ls) && o in eachindex(out_ls)) ||
                throw(ArgumentError(lazy"model: linear instruction ($i, $o) is outside the $(length(in_ls)) input and $(length(out_ls)) output slots"))
            in_ls[i] == out_ls[o] ||
                throw(ArgumentError(lazy"model: linear instruction ($i, $o) couples l = $(in_ls[i]) to l = $(out_ls[o])"))
        end
        W = [Matrix{T}(reshape(weight[((p - 1) * blk + 1):(p * blk)], C_out, C_in)) for p in eachindex(instructions)]
        pw = T[T(pwp) for (_, _, pwp) in instructions]
        slots = [(i, o) for (i, o, _) in instructions]
        K_in = sum(l -> 2l + 1, in_ls; init = 0)
        K_out = sum(l -> 2l + 1, out_ls; init = 0)
        return new{T}(W, pw, slots, copy(in_ls), _slot_offsets(in_ls), _slot_offsets(out_ls), C_in, C_out, K_in, K_out)
    end
end

function _check_linear(lin::E3nnLinear, y::AbstractArray, x::AbstractArray)
    Base.require_one_based_indexing(y, x)
    (size(x, 1) == lin.C_in && size(x, 2) == lin.K_in) ||
        throw(DimensionMismatch(lazy"model: linear input has size $(size(x)), expected ($(lin.C_in), $(lin.K_in), n)"))
    size(y) == (lin.C_out, lin.K_out, size(x, 3)) ||
        throw(DimensionMismatch(lazy"model: linear output has size $(size(y)), expected $((lin.C_out, lin.K_out, size(x, 3)))"))
    return nothing
end

"""
    linear!(y, lin::E3nnLinear, x) -> y

`y[w, k_out, i] = Σ_p pw_p Σ_u W_p[u, w] x[u, k_in, i]` over the instructions `p` that map the
input slot holding component `k_in` to the output slot holding `k_out`, with matching `m`.
Overwrites `y`.
"""
function linear!(y::AbstractArray{S, 3}, lin::E3nnLinear, x::AbstractArray{S, 3}) where {S}
    _check_linear(lin, y, x)
    fill!(y, zero(S))
    for p in eachindex(lin.W, lin.pw, lin.slots)
        i, o = lin.slots[p]
        for m in 1:(2 * lin.in_ls[i] + 1)
            mul!(view(y, :, lin.out_off[o] + m, :), lin.W[p], view(x, :, lin.in_off[i] + m, :), lin.pw[p], true)
        end
    end
    return y
end

"""
    linear_pullback!(dx, lin::E3nnLinear, dy) -> dx

The vector-Jacobian product of [`linear!`](@ref) with the cotangent `dy`. Overwrites `dx`.
"""
function linear_pullback!(dx::AbstractArray{S, 3}, lin::E3nnLinear, dy::AbstractArray{S, 3}) where {S}
    _check_linear(lin, dy, dx)
    fill!(dx, zero(S))
    for p in eachindex(lin.W, lin.pw, lin.slots)
        i, o = lin.slots[p]
        for m in 1:(2 * lin.in_ls[i] + 1)
            mul!(view(dx, :, lin.in_off[i] + m, :), transpose(lin.W[p]), view(dy, :, lin.out_off[o] + m, :), lin.pw[p], true)
        end
    end
    return dx
end

"""
    SpeciesLinear{T}(weight, ls, instructions, C_in, n_species, C_out)

e3nn `FullyConnectedTensorProduct` of node features (slots of degrees `ls`) with a one-hot
species vector: a linear map per species. `instructions` lists `(input slot, output slot, path
weight)`, 1-based; block `p` of `weight` is the torch `[C_in, n_species, C_out]` row-major
segment `p`. `c[p] = pw · (1 / sqrt(2l+1))`, the path weight times e3nn's `(l, 0, l)` coupling;
this product order reproduces e3nn's folded constants bit for bit.
"""
struct SpeciesLinear{T}
    W::Vector{Array{T, 3}}       # W[p][w, u, s] = torch block p [u, s, w]
    c::Vector{T}
    slots::Vector{Tuple{Int, Int}}
    ls::Vector{Int}
    off::Vector{Int}
    C_in::Int
    C_out::Int
    K::Int
    n_species::Int
    function SpeciesLinear{T}(
            weight::AbstractVector{<:Real}, ls::Vector{Int}, instructions::Vector{Tuple{Int, Int, Float64}},
            C_in::Int, n_species::Int, C_out::Int
        ) where {T}
        Base.require_one_based_indexing(weight)
        blk = C_in * n_species * C_out
        length(weight) == length(instructions) * blk ||
            throw(ArgumentError(lazy"model: species linear weight has $(length(weight)) entries, expected $(length(instructions) * blk)"))
        for (i, o, _) in instructions
            (i in eachindex(ls) && o in eachindex(ls)) ||
                throw(ArgumentError(lazy"model: species linear instruction ($i, $o) is outside the $(length(ls)) slots"))
            ls[i] == ls[o] ||
                throw(ArgumentError(lazy"model: species linear instruction ($i, $o) couples l = $(ls[i]) to l = $(ls[o])"))
        end
        W = [
            Array{T, 3}(permutedims(reshape(weight[((p - 1) * blk + 1):(p * blk)], C_out, n_species, C_in), (1, 3, 2)))
                for p in eachindex(instructions)
        ]
        c = T[T(pw * (1 / sqrt(2 * ls[i] + 1))) for (i, _, pw) in instructions]
        slots = [(i, o) for (i, o, _) in instructions]
        K = sum(l -> 2l + 1, ls; init = 0)
        return new{T}(W, c, slots, copy(ls), _slot_offsets(ls), C_in, C_out, K, n_species)
    end
end

function _check_species_linear(sl::SpeciesLinear, y::AbstractArray, x::AbstractArray, species::AbstractVector{Int})
    Base.require_one_based_indexing(y, x, species)
    (size(x, 1) == sl.C_in && size(x, 2) == sl.K) ||
        throw(DimensionMismatch(lazy"model: species linear input has size $(size(x)), expected ($(sl.C_in), $(sl.K), n)"))
    size(y) == (sl.C_out, sl.K, size(x, 3)) ||
        throw(DimensionMismatch(lazy"model: species linear output has size $(size(y)), expected $((sl.C_out, sl.K, size(x, 3)))"))
    length(species) == size(x, 3) ||
        throw(DimensionMismatch(lazy"model: $(length(species)) species for $(size(x, 3)) nodes"))
    for s in species
        1 <= s <= sl.n_species || throw(ArgumentError(lazy"model: species index $s is outside 1:$(sl.n_species)"))
    end
    return nothing
end

"""
    species_linear!(y, sl::SpeciesLinear, x, species) -> y

`y[w, k, i] = c_p Σ_u W_p[u, species[i], w] x[u, k, i]` for each instruction `p`, with `k` over
the components of its slots. Overwrites `y`.
"""
function species_linear!(y::AbstractArray{S, 3}, sl::SpeciesLinear, x::AbstractArray{S, 3}, species::AbstractVector{Int}) where {S}
    _check_species_linear(sl, y, x, species)
    fill!(y, zero(S))
    for p in eachindex(sl.W, sl.c, sl.slots)
        i, o = sl.slots[p]
        d = 2 * sl.ls[i] + 1
        ri, ro = (sl.off[i] + 1):(sl.off[i] + d), (sl.off[o] + 1):(sl.off[o] + d)
        for n in axes(x, 3)
            mul!(view(y, :, ro, n), view(sl.W[p], :, :, species[n]), view(x, :, ri, n), sl.c[p], true)
        end
    end
    return y
end

"""
    species_linear_pullback!(dx, sl::SpeciesLinear, dy, species) -> dx

The vector-Jacobian product of [`species_linear!`](@ref) with the cotangent `dy`. Overwrites `dx`.
"""
function species_linear_pullback!(dx::AbstractArray{S, 3}, sl::SpeciesLinear, dy::AbstractArray{S, 3}, species::AbstractVector{Int}) where {S}
    _check_species_linear(sl, dy, dx, species)
    fill!(dx, zero(S))
    for p in eachindex(sl.W, sl.c, sl.slots)
        i, o = sl.slots[p]
        d = 2 * sl.ls[i] + 1
        ri, ro = (sl.off[i] + 1):(sl.off[i] + d), (sl.off[o] + 1):(sl.off[o] + d)
        for n in axes(dx, 3)
            mul!(view(dx, :, ri, n), transpose(view(sl.W[p], :, :, species[n])), view(dy, :, ro, n), sl.c[p], true)
        end
    end
    return dx
end
