import Base: sqrt, cbrt, exp, expm1, log, log1p

# s·(a - value(a)): the part of `a` without constant term, scaled by `s`.
function _scaled_nilpotent(a::Taylor{N,K,M,T,L}, s::T) where {N,K,M,T,L}
    return Taylor{N,K,M,T,L}(ntuple(i -> i == 1 ? zero(T) : s * a.c[i], Val(L)))
end

# The constant `x` in the type of `a`.
_constant_like(a::Taylor{N,K,M,T,L}, x::T) where {N,K,M,T,L} = Taylor{N,K,M,T,L}(ntuple(i -> i == 1 ? x : zero(T), Val(L)))

# Σₖ d[k+1]·(s·h)ᵏ for k = 0..K, with h = a - value(a) the part of `a` without constant term. `d`
# holds the normalized derivatives fᵏ(x)/k! of a univariate f at x = value(a), in the variable
# t = s·h. Horner's rule; the partial sum multiplied by tʲ in the end is needed only to degree K - j,
# so each step is a degree-limited product.
@generated function compose(a::Taylor{N,K,M,T,L}, d::NTuple{K1,T}, s::T) where {N,K,M,T,L,K1}
    K1 == K + 1 || return :(throw(ArgumentError($("PureTaylor: compose needs K + 1 = $(K + 1) derivatives, got $K1"))))
    body = quote
        t = _scaled_nilpotent(a, s)
        p = _constant_like(a, d[$(K + 1)])
    end
    for j in (K - 1):-1:0
        push!(body.args, :(p = mul_truncated(t, p, Val(true), Val($(K - j))) + d[$(j + 1)]))
    end
    push!(body.args, :p)
    return body
end

# (b₀, g(b₀, 1), g(g(b₀, 1), 2), …): the K + 1 terms of the recurrence bₖ = g(bₖ₋₁, k), unrolled.
@generated function _recurrence(g, b0, ::Val{K}) where {K}
    b = [Symbol(:b, k) for k in 0:K]
    body = Expr(:block, Expr(:meta, :inline), :($(b[1]) = b0))
    for k in 1:K
        push!(body.args, :($(b[k + 1]) = g($(b[k]), $k)))
    end
    push!(body.args, Expr(:tuple, b...))
    return body
end

# (x + h)ᵖ = xᵖ·(1 + h/x)ᵖ = Σₖ xᵖ·binomial(p, k)·(h/x)ᵏ, with the generalized binomial
# coefficient of real p built by its recurrence binomial(p, k) = binomial(p, k-1)·(p - k + 1)/k.
function _power_series(a::Taylor{N,K,M,T}, v::T, p::T) where {N,K,M,T}
    d = _recurrence((b, k) -> b * (p - T(k - 1)) / T(k), v, Val(K))
    return compose(a, d, inv(value(a)))
end

"""
    sqrt(a::Taylor)

The series of `√a`, with its value from Base's `sqrt` (a negative value throws `DomainError`). At value
0 the derivatives are singular: every coefficient but the value is `NaN`.
"""
sqrt(a::Taylor{N,K,M,T}) where {N,K,M,T} = _power_series(a, sqrt(value(a)), inv(T(2)))

"""
    cbrt(a::Taylor)

The series of `∛a`, with its value from Base's `cbrt`. At value 0 the derivatives are singular: every
coefficient but the value is `NaN`.
"""
cbrt(a::Taylor{N,K,M,T}) where {N,K,M,T} = _power_series(a, cbrt(value(a)), inv(T(3)))

# Base defines `^(::Number, ::Rational)`; a `Rational` method of its own keeps the `Real` one
# unambiguous. An integer-valued exponent of magnitude at most 2^62 takes the integer-power path
# (products only), which stays finite at value 0.
function _real_power(a::Taylor{N,K,M,T}, p::Real) where {N,K,M,T}
    q = convert(T, p)
    isinteger(q) && abs(q) <= T(2)^62 && return a^trunc(Int, q)
    return _power_series(a, value(a)^q, q)
end

"""
    a^p

The series of `aᵖ` for a real (or rational) `p`, converted to the coefficient type of `a`. An
integer-valued `p` is the integer power. Otherwise the value comes from Base's `^` (a negative value
throws `DomainError`), and at value 0 the derivatives are singular: every coefficient but the value is
`NaN`.
"""
^(a::Taylor, p::Real) = _real_power(a, p)
^(a::Taylor, p::Rational) = _real_power(a, p)

# exp(x + h) = Σₖ exp(x)/k!·hᵏ; expm1 differs only in the value.
function exp(a::Taylor{N,K,M,T}) where {N,K,M,T}
    v = exp(value(a))
    return compose(a, _recurrence((b, k) -> b / T(k), v, Val(K)), one(T))
end
function expm1(a::Taylor{N,K,M,T}) where {N,K,M,T}
    x = value(a)
    d = _recurrence((b, k) -> b / T(k), exp(x), Val(K))
    return compose(a, ntuple(i -> i == 1 ? expm1(x) : d[i], Val(K + 1)), one(T))
end

# log(y + h) = log(y) + Σₖ₌₁ (-1)ᵏ⁻¹/k·(h/y)ᵏ, at y = x for log and y = 1 + x for log1p.
function _log_series(a::Taylor{N,K,M,T}, v::T, y::T) where {N,K,M,T}
    d = ntuple(i -> i == 1 ? v : (iseven(i) ? one(T) : -one(T)) / T(i - 1), Val(K + 1))
    return compose(a, d, inv(y))
end
log(a::Taylor) = _log_series(a, log(value(a)), value(a))
log1p(a::Taylor{N,K,M,T}) where {N,K,M,T} = _log_series(a, log1p(value(a)), one(T) + value(a))
