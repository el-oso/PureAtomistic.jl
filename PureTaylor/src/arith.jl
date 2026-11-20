import Base: +, -, zero, one, isfinite, isnan

# Coefficient-wise sum or difference over the union mask. The generated body is marked inline so
# the coefficient tuples stay in registers.
@generated function _addsub(a::Taylor{N,K,M1,T,L1}, b::Taylor{N,K,M2,T,L2}, ::Val{sgn}) where {N,K,M1,M2,T,L1,L2,sgn}
    M = M1 | M2
    pos1 = Dict(α => j for (j, α) in enumerate(stored_indices(N, K, M1)))
    pos2 = Dict(α => j for (j, α) in enumerate(stored_indices(N, K, M2)))
    terms = map(stored_indices(N, K, M)) do α
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
-(a::Taylor{N,K,M,T,L}) where {N,K,M,T,L} = Taylor{N,K,M,T,L}(map(-, a.c))

# The higher-order coefficients of `a`, unchanged.
_rest(a::Taylor{N,K,M,T,L}) where {N,K,M,T,L} = ntuple(i -> a.c[i + 1], Val(L - 1))

function +(a::Taylor{N,K,M,T,L}, r::Real) where {N,K,M,T,L}
    return Taylor{N,K,M,T,L}((a.c[1] + convert(T, r), _rest(a)...))
end
+(r::Real, a::Taylor) = a + r
function -(a::Taylor{N,K,M,T,L}, r::Real) where {N,K,M,T,L}
    return Taylor{N,K,M,T,L}((a.c[1] - convert(T, r), _rest(a)...))
end
function -(r::Real, a::Taylor{N,K,M,T,L}) where {N,K,M,T,L}
    return Taylor{N,K,M,T,L}((convert(T, r) - a.c[1], map(-, _rest(a))...))
end

zero(a::Taylor) = zero(typeof(a))
zero(::Type{Taylor{N,K,M,T,L}}) where {N,K,M,T,L} = Taylor{N,K,M,T,L}(ntuple(_ -> zero(T), Val(L)))
one(a::Taylor) = one(typeof(a))
one(::Type{Taylor{N,K,M,T,L}}) where {N,K,M,T,L} = Taylor{N,K,M,T,L}(ntuple(i -> i == 1 ? one(T) : zero(T), Val(L)))

isfinite(a::Taylor) = all(isfinite, a.c)
isnan(a::Taylor) = any(isnan, a.c)
