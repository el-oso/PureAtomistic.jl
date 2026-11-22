"""
    Taylor{N,K,M,T,L}

Truncated Taylor polynomial in `N` variables up to total degree `K`, with coefficients of type `T`.
Only the variables in the bit mask `M` (bit `i-1` set means variable `i`) carry nonzero higher-order
terms. `c` holds `c_α = (∂^α f)/α!` for every multi-index `α` of `multi_indices(Val(N), Val(K))`
supported on those variables, in that order, so `c[1]` is the value and `L == binomial(count_ones(M) + K, K)`.
`N`, `K` and `M` are `Int`s with `1 ≤ N ≤ 63` (the mask is an `Int`), `K ≥ 0` and `0 ≤ M < 2^N`.
"""
struct Taylor{N, K, M, T, L} <: Real
    c::NTuple{L, T}
    function Taylor{N, K, M, T, L}(c::Tuple) where {N, K, M, T, L}
        N isa Int || throw(ArgumentError(lazy"PureTaylor: N = $N must be an Int, got $(typeof(N))"))
        1 <= N <= 63 || throw(ArgumentError(lazy"PureTaylor: N = $N must satisfy 1 ≤ N ≤ 63 (the mask is an Int)"))
        K isa Int && K >= 0 || throw(ArgumentError(lazy"PureTaylor: K = $K must be an Int ≥ 0"))
        M isa Int || throw(ArgumentError(lazy"PureTaylor: mask M = $M must be an Int, got $(typeof(M))"))
        0 <= M < UInt128(1) << N || throw(ArgumentError(lazy"PureTaylor: mask M = $M must satisfy 0 ≤ M < 2^N for N = $N"))
        L == binomial(count_ones(M) + K, K) ||
            throw(ArgumentError(lazy"PureTaylor: L = $L must equal binomial(count_ones(M) + K, K) = $(binomial(count_ones(M) + K, K)) for M = $M, K = $K"))
        return new{N, K, M, T, L}(c)
    end
end

"""
    variable(Val(N), Val(K), Val(i), x, s = one(x))

The number `x + s·uᵢ`, a series in `uᵢ`, for `1 ≤ i ≤ N`. With `s = x` every coefficient stays of
order 1 in size; `derivatives` undoes the scaling.
"""
function variable(::Val{N}, ::Val{K}, ::Val{i}, x::T, s::T = one(T)) where {N, K, i, T}
    1 <= i <= N || throw(ArgumentError(lazy"PureTaylor: variable index i = $i must satisfy 1 ≤ i ≤ N = $N"))
    c = ntuple(j -> j == 1 ? x : j == 2 ? s : zero(T), Val(K + 1))
    return Taylor{N, K, 1 << (i - 1), T, K + 1}(c)
end

"""
    constant(Val(N), Val(K), x)

The constant `x` as a `Taylor` with mask 0.
"""
constant(::Val{N}, ::Val{K}, x::T) where {N, K, T} = Taylor{N, K, 0, T, 1}((x,))

"""
    value(a)

The value of a `Taylor` (its coefficient of order 0); a non-`Taylor` number is its own value.
"""
value(a::Taylor) = a.c[1]
value(x::Real) = x

# The unrolled body of `coefficient`: a chain of comparisons in the order of the stored indices.
function _coefficient_unrolled_body(N, K, M)
    idx = stored_indices(Val(N), Val(K), Val(M))
    body = :(return zero(T))
    for j in length(idx):-1:1
        body = :(α == $(idx[j]) ? (return a.c[$j]) : $body)
    end
    return body
end

"""
    coefficient(a, α)

The coefficient `c_α = (∂^α f)/α!` of `a` (in the seed variables `u`), or zero when `α` is not stored.
"""
@generated function coefficient(a::Taylor{N, K, M, T, L}, α::NTuple{N, Int}) where {N, K, M, T, L}
    _use_loop(L) && return :(_coefficient_loop(a, α))
    return _coefficient_unrolled_body(N, K, M)
end

# The unrolled form at any size; the tests compare it with the loop form.
@generated _coefficient_unrolled(a::Taylor{N, K, M, T, L}, α::NTuple{N, Int}) where {N, K, M, T, L} =
    _coefficient_unrolled_body(N, K, M)

"""
    derivatives(a, invscale)

All partial derivatives `∂^α f = α! · c_α · Π invscale_i^α_i` in `multi_indices(Val(N), Val(K))`
order, with `invscale_i = 1/s_i` for the seed scales `s_i`; zero where `α` is not stored. The entries
of `invscale` are converted to the coefficient type of `a`.

Each derivative is formed as `c_α` times the factors `j·invscale_i`, `j = 1, …, α_i`, variable by
variable, so neither `α!·c_α` nor `invscale_i^α_i` has to be representable. Within one variable the
factors increase in magnitude with `j`, so for a derivative in one variable no partial product exceeds the larger
of `|c_α|` and the result in magnitude.
"""
derivatives(a::Taylor{N, K, M, T}, invscale::NTuple{N, Real}) where {N, K, M, T} =
    _derivatives(a, ntuple(i -> convert(T, invscale[i]), Val(N)))

# The unrolled body of `_derivatives`: the factors j·invscale_i are formed once each, in the type T,
# and applied left to right.
function _derivatives_unrolled_body(N, K, M, T)
    pos = Dict(α => j for (j, α) in enumerate(stored_indices(Val(N), Val(K), Val(M))))
    body = Expr(:block)
    factor(i, j) = j == 1 ? :(invscale[$i]) : Symbol(:f_, i, :_, j)
    for i in 1:N, j in 2:K
        push!(body.args, :($(factor(i, j)) = $(T(j)) * invscale[$i]))
    end
    terms = map(multi_indices(Val(N), Val(K))) do α
        haskey(pos, α) || return :(zero($T))
        scale = Expr(:call, :*, :(a.c[$(pos[α])]))
        for i in 1:N, j in 1:α[i]
            push!(scale.args, factor(i, j))
        end
        return length(scale.args) == 2 ? scale.args[2] : scale
    end
    push!(body.args, Expr(:tuple, terms...))
    return body
end

# Unrolled up to `_UNROLL_MAX` derivatives, `_derivatives_loop` above.
@generated function _derivatives(a::Taylor{N, K, M, T, L}, invscale::NTuple{N, T}) where {N, K, M, T, L}
    _use_loop(length(multi_indices(Val(N), Val(K)))) && return :(_derivatives_loop(a, invscale))
    return _derivatives_unrolled_body(N, K, M, T)
end

# The unrolled form at any size; the tests compare it with the loop form.
@generated _derivatives_unrolled(a::Taylor{N, K, M, T, L}, invscale::NTuple{N, T}) where {N, K, M, T, L} =
    _derivatives_unrolled_body(N, K, M, T)

# `coefficient` by a loop over the stored multi-indices, in their order.
@noinline function _coefficient_loop(a::Taylor{N, K, M, T}, α::NTuple{N, Int}) where {N, K, M, T}
    idx = stored_indices(Val(N), Val(K), Val(M))
    for j in eachindex(idx)
        α == idx[j] && return a.c[j]
    end
    return zero(T)
end

# The position in `a` (0 where not stored) of each multi-index of `multi_indices(Val(N), Val(K))`.
@generated function _derivatives_table(::Val{N}, ::Val{K}, ::Val{M}) where {N, K, M}
    pos = Dict(α => j for (j, α) in enumerate(stored_indices(Val(N), Val(K), Val(M))))
    return _int32_tuple(get(pos, α, 0) for α in multi_indices(Val(N), Val(K)))
end

# `_derivatives` by a loop: the same factors j·invscale_i, recomputed where they are used, applied
# left to right in the same order.
@noinline function _derivatives_loop(a::Taylor{N, K, M, T}, invscale::NTuple{N, T}) where {N, K, M, T}
    idx = multi_indices(Val(N), Val(K))
    pos = _derivatives_table(Val(N), Val(K), Val(M))
    return _loop_tuple(q -> _derivative_entry(a, invscale, pos[q], idx[q]), T, Val(length(idx)))
end

# One entry of `_derivatives_loop`: the coefficient at position `p` (0 where not stored) for the
# multi-index α.
@inline function _derivative_entry(a::Taylor{N, K, M, T}, invscale::NTuple{N, T}, p, α) where {N, K, M, T}
    iszero(p) && return zero(T)
    v = a.c[p]
    for i in 1:N, j in 1:α[i]
        v *= isone(j) ? invscale[i] : T(j) * invscale[i]
    end
    return v
end
