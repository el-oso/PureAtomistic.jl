"""
    substitute(E::Taylor{N,K}, u)

Change of variables. `E` is a series whose `N` variables are increments of `N` quantities; `u[i]` is the
`i`-th quantity as a `Taylor{N2,K}` over new variables, or a plain number for a quantity that does not
change. At least one `u[i]` is a `Taylor`, and all of them have the same `N2`. The result is
`Σ_α c_α · Π_i (u[i] - value(u[i]))^α_i` over the coefficients `c_α` of `E`, a `Taylor{N2,K}` over
the new variables. The values of `u` are not used: `E` is taken to be expanded around them. When variable
`i` of `E` was seeded with a scale `s` (`variable(…, x, s)`), it is the increment divided by `s`, so
pass `u[i] / s`.
"""
@generated function substitute(
        E::Taylor{N, K, M, T, L}, u::Tuple{Vararg{Union{Taylor{<:Any, K, <:Any, T}, T}, N}}
    ) where {N, K, M, T, L}
    # The number of new variables, from the Taylor elements of `u`; they must agree.
    news = unique(P.parameters[1] for P in u.parameters if P <: Taylor)
    length(news) == 1 ||
        return :(throw(ArgumentError($("PureTaylor: substitute needs at least one Taylor number in u, all over the same number of variables; got $(Tuple(news))"))))
    N2 = only(news)
    body = Expr(:block, Expr(:meta, :inline))
    # h[i]: the nilpotent part of u[i], or nothing for a plain number (its increment is zero).
    h = map(1:N) do i
        u.parameters[i] <: Taylor && isodd(M >> (i - 1)) || return nothing
        s = Symbol(:h, i)
        push!(body.args, :($s = u[$i] - value(u[$i])))
        return s
    end
    # Each monomial Π hᵢ^αᵢ is built once: a power of one variable from the next lower power, any
    # other monomial as the monomial without its last variable times that variable's power.
    monomials = Dict{NTuple{N, Int}, Symbol}()
    function monomial(α)
        haskey(monomials, α) && return monomials[α]
        i = findlast(>(0), α)
        power = ntuple(k -> k == i ? α[i] : 0, N)
        if α == power
            α[i] == 1 && return monomials[α] = h[i]
            rhs = :($(monomial(power .- ntuple(k -> k == i, N))) * $(h[i]))
        else
            rhs = :($(monomial(α .- power)) * $(monomial(power)))
        end
        s = Symbol(:m_, join(α, '_'))
        push!(body.args, :($s = $rhs))
        return monomials[α] = s
    end
    # The sum, one fresh name per partial sum so that each has a single type.
    acc = nothing
    for (j, α) in enumerate(stored_indices(Val(N), Val(K), Val(M)))
        j == 1 && continue
        any(i -> α[i] > 0 && isnothing(h[i]), 1:N) && continue
        term = :(_scale_from($(monomial(α)), E.c[$j], Val($(sum(α)))))
        s = Symbol(:s_, j)
        push!(body.args, :($s = $(isnothing(acc) ? term : :($acc + $term))))
        acc = s
    end
    push!(body.args, isnothing(acc) ? :(constant(Val($N2), Val(K), E.c[1])) : :($acc + E.c[1]))
    return body
end

# r·m for a series m with no terms below degree D, such as a monomial of degree D in increments:
# the coefficients below degree D are zero without a product, so an infinite r (a coefficient whose
# value is beyond the range of T) does not reach them as 0·Inf.
@generated function _scale_from(m::Taylor{N, K, M, T, L}, r::T, ::Val{D}) where {N, K, M, T, L, D}
    terms = map(enumerate(stored_indices(Val(N), Val(K), Val(M)))) do (j, α)
        sum(α) < D ? :(zero(T)) : :(m.c[$j] * r)
    end
    return quote
        $(Expr(:meta, :inline))
        Taylor{N, K, M, T, L}($(Expr(:tuple, terms...)))
    end
end
