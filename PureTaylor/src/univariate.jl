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
