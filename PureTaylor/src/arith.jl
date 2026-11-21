import Base: +, -, *, /, ^, inv, literal_pow, zero, one, isfinite, isnan

# Coefficient-wise sum or difference over the union mask. The generated body is marked inline so
# the coefficient tuples stay in registers.
@generated function _addsub(a::Taylor{N,K,M1,T,L1}, b::Taylor{N,K,M2,T,L2}, ::Val{sgn}) where {N,K,M1,M2,T,L1,L2,sgn}
    M = M1 | M2
    pos1 = Dict(α => j for (j, α) in enumerate(stored_indices(Val(N), Val(K), Val(M1))))
    pos2 = Dict(α => j for (j, α) in enumerate(stored_indices(Val(N), Val(K), Val(M2))))
    terms = map(stored_indices(Val(N), Val(K), Val(M))) do α
        in1, in2 = haskey(pos1, α), haskey(pos2, α)
        x = in1 ? :(a.c[$(pos1[α])]) : nothing
        y = in2 ? :(b.c[$(pos2[α])]) : nothing
        x === nothing && y === nothing ? :(zero(T)) :
            x === nothing ? (sgn == 1 ? y : :(-$y)) : y === nothing ? x : Expr(:call, sgn == 1 ? :+ : :-, x, y)
    end
    L = length(terms)
    return quote
        $(Expr(:meta, :inline))
        Taylor{N,K,$M,T,$L}($(Expr(:tuple, terms...)))
    end
end

+(a::Taylor, b::Taylor) = _addsub(a, b, Val(1))
-(a::Taylor, b::Taylor) = _addsub(a, b, Val(-1))
-(a::Taylor{N,K,M,T,L}) where {N,K,M,T,L} = Taylor{N,K,M,T,L}(ntuple(i -> -a.c[i], Val(L)))

# Scalar sums change only the value. The tuples are built with `ntuple(f, Val(L))`: `map` over a
# tuple longer than 32 allocates, and AllocCheck cannot prove that splatting one does not.
function +(a::Taylor{N,K,M,T,L}, r::Real) where {N,K,M,T,L}
    s = convert(T, r)
    return Taylor{N,K,M,T,L}(ntuple(i -> i == 1 ? a.c[1] + s : a.c[i], Val(L)))
end
+(r::Real, a::Taylor) = a + r
function -(a::Taylor{N,K,M,T,L}, r::Real) where {N,K,M,T,L}
    s = convert(T, r)
    return Taylor{N,K,M,T,L}(ntuple(i -> i == 1 ? a.c[1] - s : a.c[i], Val(L)))
end
function -(r::Real, a::Taylor{N,K,M,T,L}) where {N,K,M,T,L}
    s = convert(T, r)
    return Taylor{N,K,M,T,L}(ntuple(i -> i == 1 ? s - a.c[1] : -a.c[i], Val(L)))
end

zero(a::Taylor) = zero(typeof(a))
zero(::Type{Taylor{N,K,M,T,L}}) where {N,K,M,T,L} = Taylor{N,K,M,T,L}(ntuple(_ -> zero(T), Val(L)))
one(a::Taylor) = one(typeof(a))
one(::Type{Taylor{N,K,M,T,L}}) where {N,K,M,T,L} = Taylor{N,K,M,T,L}(ntuple(i -> i == 1 ? one(T) : zero(T), Val(L)))

isfinite(a::Taylor) = all(isfinite, a.c)
isnan(a::Taylor) = any(isnan, a.c)

# The product `a·b` restricted to total degree ≤ D (higher coefficients are zero), over the union
# mask; `a`'s constant term is left out when `skip0`. Each output coefficient is one straight-line
# `muladd` chain over the pairs of stored indices that sum to it, inlined so the tuples stay in
# registers.
@generated function mul_truncated(
        a::Taylor{N,K,M1,T,L1}, b::Taylor{N,K,M2,T,L2}, ::Val{skip0}, ::Val{D}
    ) where {N,K,M1,M2,T,L1,L2,skip0,D}
    M = M1 | M2
    idx1 = stored_indices(Val(N), Val(K), Val(M1))
    pos2 = Dict(α => j for (j, α) in enumerate(stored_indices(Val(N), Val(K), Val(M2))))
    terms = map(stored_indices(Val(N), Val(K), Val(M))) do α
        acc = nothing
        sum(α) <= D || return :(zero(T))
        for (i, β) in enumerate(idx1)
            skip0 && i == 1 && continue
            all(β .<= α) || continue
            γ = α .- β
            haskey(pos2, γ) || continue
            x, y = :(a.c[$i]), :(b.c[$(pos2[γ])])
            acc = acc === nothing ? :($x * $y) : :(muladd($x, $y, $acc))
        end
        return acc === nothing ? :(zero(T)) : acc
    end
    L = length(terms)
    return quote
        $(Expr(:meta, :inline))
        Taylor{N,K,$M,T,$L}($(Expr(:tuple, terms...)))
    end
end

*(a::Taylor{N,K}, b::Taylor) where {N,K} = mul_truncated(a, b, Val(false), Val(K))
function *(a::Taylor{N,K,M,T,L}, r::Real) where {N,K,M,T,L}
    s = convert(T, r)
    return Taylor{N,K,M,T,L}(ntuple(i -> a.c[i] * s, Val(L)))
end
*(r::Real, a::Taylor) = a * r

# 1/(x + h) = Σₖ (-1)ᵏ (h/x)ᵏ / x: the series in t = h/x has coefficients ±1/x.
function inv(a::Taylor{N,K,M,T,L}) where {N,K,M,T,L}
    r = inv(value(a))
    return compose(a, ntuple(k -> isodd(k) ? r : -r, Val(K + 1)), r)
end

/(a::Taylor, b::Taylor) = a * inv(b)
/(a::Taylor{N,K,M,T}, r::Real) where {N,K,M,T} = a * inv(convert(T, r))
/(r::Real, a::Taylor) = r * inv(a)

# Product of `|n|` copies of `a` by repeated squaring; `isodd` and `÷ 2` act on the magnitude, so a
# negative `n` (including `typemin`) is handled without negating it.
function _pow_abs(a::Taylor, n::Integer)
    r, x = one(a), a
    while !iszero(n)
        isodd(n) && (r *= x)
        n = n ÷ 2
        iszero(n) || (x *= x)
    end
    return r
end

^(a::Taylor, n::Integer) = n < 0 ? inv(_pow_abs(a, n)) : _pow_abs(a, n)

@generated function literal_pow(::typeof(^), a::Taylor, ::Val{n}) where {n}
    # Julia lowers only integer literals to literal_pow; any other exponent is the general power.
    n isa Integer || return :(a ^ $n)
    n == 0 && return :(one(a))
    m = abs(n)
    # Unrolled repeated squaring of the literal exponent.
    body = Expr(:block, :(x = a))
    r = nothing
    while true
        if isodd(m)
            push!(body.args, r === nothing ? :(r = x) : :(r = r * x))
            r = :r
        end
        m ÷= 2
        m == 0 && break
        push!(body.args, :(x = x * x))
    end
    push!(body.args, n < 0 ? :(inv(r)) : :r)
    return body
end
