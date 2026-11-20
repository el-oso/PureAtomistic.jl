"""
    Taylor{N,K,M,T,L}

Truncated Taylor polynomial in `N` variables up to total degree `K`, with coefficients of type `T`.
Only the variables in the bit mask `M` (bit `i-1` set means variable `i`) carry nonzero higher-order
terms. `c` holds `c_α = (∂^α f)/α!` for every multi-index `α` of `multi_indices(N, K)` supported on
those variables, in that order, so `c[1]` is the value and `L == binomial(count_ones(M) + K, K)`.
"""
struct Taylor{N,K,M,T,L} <: Real
    c::NTuple{L,T}
    function Taylor{N,K,M,T,L}(c::Tuple) where {N,K,M,T,L}
        1 <= N <= 64 || throw(ArgumentError("PureTaylor: N = $N must satisfy 1 ≤ N ≤ 64"))
        0 <= M < UInt128(1) << N || throw(ArgumentError("PureTaylor: mask M = $M must satisfy 0 ≤ M < 2^N for N = $N"))
        L == binomial(count_ones(M) + K, K) ||
            throw(ArgumentError("PureTaylor: L = $L must equal binomial(count_ones(M) + K, K) = $(binomial(count_ones(M) + K, K)) for M = $M, K = $K"))
        return new{N,K,M,T,L}(c)
    end
end

"""
    variable(Val(N), Val(K), Val(i), x, s = one(x))

The number `x + s·uᵢ`, a series in `uᵢ`. With `s = x` every coefficient stays of order 1 in size;
`derivatives` undoes the scaling.
"""
function variable(::Val{N}, ::Val{K}, ::Val{i}, x::T, s::T = one(T)) where {N,K,i,T}
    c = ntuple(j -> j == 1 ? x : j == 2 ? s : zero(T), Val(K + 1))
    return Taylor{N,K,1 << (i - 1),T,K + 1}(c)
end

"""
    constant(Val(N), Val(K), x)

The constant `x` as a `Taylor` with mask 0.
"""
constant(::Val{N}, ::Val{K}, x::T) where {N,K,T} = Taylor{N,K,0,T,1}((x,))

"""
    value(a)

The value of a `Taylor` (its coefficient of order 0); a non-`Taylor` number is its own value.
"""
value(a::Taylor) = a.c[1]
value(x::Real) = x

"""
    coefficient(a, α)

The coefficient `c_α = (∂^α f)/α!` of `a` (in the seed variables `u`), or zero when `α` is not stored.
"""
@generated function coefficient(a::Taylor{N,K,M,T,L}, α::NTuple{N,Int}) where {N,K,M,T,L}
    idx = stored_indices(N, K, M)
    body = :(return zero(T))
    for j in length(idx):-1:1
        body = :(α == $(idx[j]) ? (return a.c[$j]) : $body)
    end
    return body
end

"""
    derivatives(a, invscale)

All partial derivatives `∂^α f = α! · c_α · Π invscale_i^α_i` in `multi_indices(N, K)` order, with
`invscale_i = 1/s_i` for the seed scales `s_i`; zero where `α` is not stored.
"""
@generated function derivatives(a::Taylor{N,K,M,T,L}, invscale::NTuple{N,T}) where {N,K,M,T,L}
    pos = Dict(α => j for (j, α) in enumerate(stored_indices(N, K, M)))
    terms = map(multi_indices(N, K)) do α
        haskey(pos, α) || return :(zero($T))
        scale = Expr(:call, :*, :($T($(prod(factorial, α)))), :(a.c[$(pos[α])]))
        for i in 1:N, _ in 1:α[i]
            push!(scale.args, :(invscale[$i]))
        end
        return scale
    end
    return Expr(:tuple, terms...)
end
