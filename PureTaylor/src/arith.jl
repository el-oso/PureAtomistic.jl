import Base: +, -, *, /, ^, inv, literal_pow, zero, one, isfinite, isnan

# The unrolled body of `_addsub`.
function _addsub_unrolled_body(N, K, M1, M2, sgn)
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
        Taylor{N, K, $M, T, $L}($(Expr(:tuple, terms...)))
    end
end

# Coefficient-wise sum or difference over the union mask. The generated body is marked inline so
# the coefficient tuples stay in registers; above `_UNROLL_MAX` coefficients it is `_addsub_loop`.
@generated function _addsub(a::Taylor{N, K, M1, T, L1}, b::Taylor{N, K, M2, T, L2}, ::Val{sgn}) where {N, K, M1, M2, T, L1, L2, sgn}
    _use_loop(length(stored_indices(Val(N), Val(K), Val(M1 | M2)))) && return :(_addsub_loop(a, b, Val($sgn)))
    return _addsub_unrolled_body(N, K, M1, M2, sgn)
end

# The unrolled form at any size; the tests compare it with the loop form.
@generated _addsub_unrolled(a::Taylor{N, K, M1, T, L1}, b::Taylor{N, K, M2, T, L2}, ::Val{sgn}) where {N, K, M1, M2, T, L1, L2, sgn} =
    _addsub_unrolled_body(N, K, M1, M2, sgn)

# Positions in `a` and `b` (0 where not stored) of each multi-index of the union mask.
@generated function _addsub_table(::Val{N}, ::Val{K}, ::Val{M1}, ::Val{M2}) where {N, K, M1, M2}
    pos1 = Dict(α => j for (j, α) in enumerate(stored_indices(Val(N), Val(K), Val(M1))))
    pos2 = Dict(α => j for (j, α) in enumerate(stored_indices(Val(N), Val(K), Val(M2))))
    idx = stored_indices(Val(N), Val(K), Val(M1 | M2))
    return (_int32_tuple(get(pos1, α, 0) for α in idx), _int32_tuple(get(pos2, α, 0) for α in idx))
end

# `_addsub` by a loop over `_addsub_table`.
@noinline function _addsub_loop(a::Taylor{N, K, M1, T}, b::Taylor{N, K, M2, T}, ::Val{sgn}) where {N, K, M1, M2, T, sgn}
    pos1, pos2 = _addsub_table(Val(N), Val(K), Val(M1), Val(M2))
    L = length(pos1)
    c = _loop_tuple(j -> _addsub_coefficient(a, b, pos1[j], pos2[j], Val(sgn)), T, Val(L))
    return Taylor{N, K, M1 | M2, T, L}(c)
end

# One coefficient of `_addsub_loop`, from the positions `i` in `a` and `k` in `b` (0 where not stored).
@inline function _addsub_coefficient(a::Taylor{N, K, M1, T}, b::Taylor{N, K, M2, T}, i, k, ::Val{sgn}) where {N, K, M1, M2, T, sgn}
    iszero(i) && iszero(k) && return zero(T)
    iszero(i) && return sgn == 1 ? b.c[k] : -b.c[k]
    iszero(k) && return a.c[i]
    return sgn == 1 ? a.c[i] + b.c[k] : a.c[i] - b.c[k]
end

+(a::Taylor, b::Taylor) = _addsub(a, b, Val(1))
-(a::Taylor, b::Taylor) = _addsub(a, b, Val(-1))
-(a::Taylor{N, K, M, T, L}) where {N, K, M, T, L} = Taylor{N, K, M, T, L}(ntuple(i -> -a.c[i], Val(L)))

# Scalar sums change only the value. The tuples are built with `ntuple(f, Val(L))`: `map` over a
# tuple longer than 32 allocates, and AllocCheck cannot prove that splatting one does not.
function +(a::Taylor{N, K, M, T, L}, r::Real) where {N, K, M, T, L}
    s = convert(T, r)
    return Taylor{N, K, M, T, L}(ntuple(i -> i == 1 ? a.c[1] + s : a.c[i], Val(L)))
end
+(r::Real, a::Taylor) = a + r
function -(a::Taylor{N, K, M, T, L}, r::Real) where {N, K, M, T, L}
    s = convert(T, r)
    return Taylor{N, K, M, T, L}(ntuple(i -> i == 1 ? a.c[1] - s : a.c[i], Val(L)))
end
function -(r::Real, a::Taylor{N, K, M, T, L}) where {N, K, M, T, L}
    s = convert(T, r)
    return Taylor{N, K, M, T, L}(ntuple(i -> i == 1 ? s - a.c[1] : -a.c[i], Val(L)))
end

zero(a::Taylor) = zero(typeof(a))
zero(::Type{Taylor{N, K, M, T, L}}) where {N, K, M, T, L} = Taylor{N, K, M, T, L}(ntuple(_ -> zero(T), Val(L)))
one(a::Taylor) = one(typeof(a))
one(::Type{Taylor{N, K, M, T, L}}) where {N, K, M, T, L} = Taylor{N, K, M, T, L}(ntuple(i -> i == 1 ? one(T) : zero(T), Val(L)))

isfinite(a::Taylor) = all(isfinite, a.c)
isnan(a::Taylor) = any(isnan, a.c)

# The unrolled body of `mul_truncated`: each output coefficient is one straight-line `_fma` chain
# over the pairs of stored indices that sum to it, inlined so the tuples stay in registers.
function _mul_unrolled_body(N, K, M1, M2, skip0, D)
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
            acc = acc === nothing ? :($x * $y) : :(_fma($x, $y, $acc))
        end
        return acc === nothing ? :(zero(T)) : acc
    end
    L = length(terms)
    return quote
        $(Expr(:meta, :inline))
        Taylor{N, K, $M, T, $L}($(Expr(:tuple, terms...)))
    end
end

# The product `a·b` restricted to total degree ≤ D (higher coefficients are zero), over the union
# mask; `a`'s constant term is left out when `skip0`. Unrolled up to `_UNROLL_MAX` coefficients,
# `_mul_loop` above.
@generated function mul_truncated(
        a::Taylor{N, K, M1, T, L1}, b::Taylor{N, K, M2, T, L2}, ::Val{skip0}, ::Val{D}
    ) where {N, K, M1, M2, T, L1, L2, skip0, D}
    _use_loop(length(stored_indices(Val(N), Val(K), Val(M1 | M2)))) && return :(_mul_loop(a, b, $skip0, $D))
    return _mul_unrolled_body(N, K, M1, M2, skip0, D)
end

# The unrolled form at any size; the tests compare it with the loop form.
@generated _mul_unrolled(
    a::Taylor{N, K, M1, T, L1}, b::Taylor{N, K, M2, T, L2}, ::Val{skip0}, ::Val{D}
) where {N, K, M1, M2, T, L1, L2, skip0, D} = _mul_unrolled_body(N, K, M1, M2, skip0, D)

# For each multi-index α of the union mask, the pairs (i, k) of positions in `a` and `b` whose
# multi-indices sum to α, in increasing i: those of output j are entries `offsets[j]:offsets[j+1]-1`
# of `ia` and `ib`. `degree[j]` is the total degree of α.
@generated function _mul_table(::Val{N}, ::Val{K}, ::Val{M1}, ::Val{M2}) where {N, K, M1, M2}
    idx1 = stored_indices(Val(N), Val(K), Val(M1))
    pos2 = Dict(γ => k for (k, γ) in enumerate(stored_indices(Val(N), Val(K), Val(M2))))
    idx = stored_indices(Val(N), Val(K), Val(M1 | M2))
    offsets, ia, ib = [1], Int[], Int[]
    for α in idx
        for (i, β) in enumerate(idx1)
            k = all(β .<= α) ? get(pos2, α .- β, 0) : 0
            iszero(k) || (push!(ia, i); push!(ib, k))
        end
        push!(offsets, length(ia) + 1)
    end
    return (_int32_tuple(offsets), _int32_tuple(ia), _int32_tuple(ib), _int32_tuple(sum.(idx)))
end

# `mul_truncated` by a loop over `_mul_table`, with the same `_fma` chain per coefficient.
@noinline function _mul_loop(a::Taylor{N, K, M1, T}, b::Taylor{N, K, M2, T}, skip0::Bool, D::Int) where {N, K, M1, M2, T}
    offsets, ia, ib, degree = _mul_table(Val(N), Val(K), Val(M1), Val(M2))
    L = length(degree)
    c = _loop_tuple(j -> _mul_coefficient(a, b, offsets, ia, ib, degree, skip0, D, j), T, Val(L))
    return Taylor{N, K, M1 | M2, T, L}(c)
end

# Coefficient j of `_mul_loop`: the `_fma` chain over the pairs of `_mul_table` for output j.
@inline function _mul_coefficient(a::Taylor{N, K, M1, T}, b::Taylor{N, K, M2, T}, offsets, ia, ib, degree, skip0, D, j) where {N, K, M1, M2, T}
    acc = zero(T)
    degree[j] <= D || return acc
    first = true
    for q in offsets[j]:(offsets[j + 1] - 1)
        i = ia[q]
        skip0 && isone(i) && continue
        x, y = a.c[i], b.c[ib[q]]
        acc = first ? x * y : _fma(x, y, acc)
        first = false
    end
    return acc
end

# The Taylor operations (`*`, `inv`, `compose`, `power_series` and the `fast_*` series methods) are
# inlined into their callers when their series have at most `_INLINE_MAX` coefficients. Left to the
# compiler, the `_fma` chains make each of them exceed the inlining threshold; the out-of-line calls
# then cost up to 2.1 times at order 2 in a caller that evaluates a series functional over many
# points, and keep its loop from vectorizing. Above `_INLINE_MAX` the unrolled kernels are not forced
# inline: their code grows with the square of the number of coefficients, and forcing them into every
# caller made the compile time of polarized `lda_c_pw` at order 10 grow from 5 s to 104 s.
@inline *(a::Taylor{N, K, M, T, L}, b::Taylor) where {N, K, M, T, L} =
    _mul_operation(Val(max(L, length(b.c)) <= _INLINE_MAX), a, b)
@inline _mul_operation(::Val{true}, a::Taylor{N, K}, b) where {N, K} = mul_truncated(a, b, Val(false), Val(K))
_mul_operation(::Val{false}, a::Taylor{N, K}, b) where {N, K} = mul_truncated(a, b, Val(false), Val(K))
function *(a::Taylor{N, K, M, T, L}, r::Real) where {N, K, M, T, L}
    s = convert(T, r)
    return Taylor{N, K, M, T, L}(ntuple(i -> a.c[i] * s, Val(L)))
end
*(r::Real, a::Taylor) = a * r

# Division a/b solves b·c = a degree by degree: c_α = (a_α − Σ_{0<β≤α} b_β c_{α−β}) / b₀, in the
# order of the stored multi-indices of the union mask (graded by total degree, so each c_{α−β} is
# formed before c_α), with the sum one `_fma` chain over the pairs of `b` and `c` with β ≠ 0. The
# recurrence is used rather than a·inv(b) with inv(b) the geometric series 1/(b₀(1 + u)),
# u = (b − b₀)/b₀, composed by Horner's rule: where the coefficients of u exceed 1 in size, the
# powers of u are much larger than their sum. Measured on a denominator of pbe_c in PureXC, whose
# first coefficient of u is 2.4: the terms of the composed series exceed the order-8 coefficient of
# the quotient by 5060 times, those of the recurrence by 4.5 times.

# The error for a coefficient type whose quotient has another type, as for integer coefficients.
_div_type_error(T, Q) = :(throw(ArgumentError($("PureTaylor: the series of a Taylor with coefficient type $T needs coefficients of type $Q; seed the variables with floating-point or Rational values"))))

# The unrolled body of `_div`.
function _div_unrolled_body(N, K, M1, M2)
    M = M1 | M2
    pos1 = Dict(α => j for (j, α) in enumerate(stored_indices(Val(N), Val(K), Val(M1))))
    idx2 = stored_indices(Val(N), Val(K), Val(M2))
    idx = stored_indices(Val(N), Val(K), Val(M))
    posc = Dict(α => j for (j, α) in enumerate(idx))
    c = [Symbol(:c_, j) for j in eachindex(idx)]
    body = Expr(:block, Expr(:meta, :inline))
    for (j, α) in enumerate(idx)
        acc = nothing
        for (i, β) in enumerate(idx2)
            i == 1 && continue
            all(β .<= α) || continue
            k = get(posc, α .- β, 0)
            iszero(k) && continue
            x, y = :(b.c[$i]), c[k]
            acc = acc === nothing ? :($x * $y) : :(_fma($x, $y, $acc))
        end
        num = haskey(pos1, α) ? :(a.c[$(pos1[α])]) : :(zero(T))
        push!(body.args, :($(c[j]) = $(acc === nothing ? num : :($num - $acc)) / b.c[1]))
    end
    push!(body.args, :(Taylor{N, K, $M, T, $(length(idx))}($(Expr(:tuple, c...)))))
    return body
end

# a/b over the union mask, by the recurrence above: unrolled up to `_UNROLL_MAX` coefficients,
# `_div_loop` above.
@generated function _div(a::Taylor{N, K, M1, T, L1}, b::Taylor{N, K, M2, T, L2}) where {N, K, M1, M2, T, L1, L2}
    Q = Base.promote_op(/, T, T)
    isconcretetype(T) && Q !== T && return _div_type_error(T, Q)
    _use_loop(length(stored_indices(Val(N), Val(K), Val(M1 | M2)))) && return :(_div_loop(a, b))
    return _div_unrolled_body(N, K, M1, M2)
end

# The unrolled form at any size; the tests compare it with the loop form.
@generated _div_unrolled(a::Taylor{N, K, M1, T, L1}, b::Taylor{N, K, M2, T, L2}) where {N, K, M1, M2, T, L1, L2} =
    _div_unrolled_body(N, K, M1, M2)

# `_div` by a loop: the pairs of `_mul_table` for b times the result, without those of b's
# constant term, and the positions of a in the union mask from `_addsub_table`.
@noinline function _div_loop(a::Taylor{N, K, M1, T}, b::Taylor{N, K, M2, T}) where {N, K, M1, M2, T}
    M = M1 | M2
    offsets, ia, ib, degree = _mul_table(Val(N), Val(K), Val(M2), Val(M))
    pos1, _ = _addsub_table(Val(N), Val(K), Val(M1), Val(M2))
    L = length(degree)
    vl = Val(L)
    c = _with_scratch(T, vl) do s
        for j in 1:L
            _store!(s, j, _div_coefficient(a, b, s, offsets, ia, ib, pos1[j], j))
        end
        return _scratch_tuple(s, T, 0, vl)
    end
    return Taylor{N, K, M, T, L}(c)
end

# Coefficient j of `_div_loop`, from the coefficients before it in the scratch array `s`.
@inline function _div_coefficient(a::Taylor{N, K, M1, T}, b::Taylor{N, K, M2, T}, s, offsets, ia, ib, p, j) where {N, K, M1, M2, T}
    acc = zero(T)
    first = true
    for q in offsets[j]:(offsets[j + 1] - 1)
        i = ia[q]
        isone(i) && continue
        x, y = b.c[i], _load(s, ib[q])
        acc = first ? x * y : _fma(x, y, acc)
        first = false
    end
    num = iszero(p) ? zero(T) : a.c[p]
    return (first ? num : num - acc) / b.c[1]
end

# Inlined, like `*`.
@inline inv(a::Taylor{N, K, M, T, L}) where {N, K, M, T, L} = _inv_operation(Val(L <= _INLINE_MAX), a)
@inline _inv_operation(::Val{true}, a) = _div(one(a), a)
_inv_operation(::Val{false}, a) = _div(one(a), a)

/(a::Taylor, b::Taylor) = _div(a, b)
/(a::Taylor{N, K, M, T}, r::Real) where {N, K, M, T} = a * inv(convert(T, r))
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

# The unrolled body of `literal_pow` for an integer `n ≠ 0`: repeated squaring of the literal
# exponent.
function _literal_pow_unrolled_body(n)
    m = abs(n)
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

@generated function literal_pow(::typeof(^), a::Taylor{N, K, M, T, L}, ::Val{n}) where {N, K, M, T, L, n}
    # Julia lowers only integer literals to literal_pow; any other exponent is the general power.
    n isa Integer || return :(a^$n)
    n == 0 && return :(one(a))
    _use_loop(L) && return :(_literal_pow_loop(a, Val($n)))
    return _literal_pow_unrolled_body(n)
end

# The unrolled form at any size, for an integer `n ≠ 0`; the tests compare it with the loop form.
@generated _literal_pow_unrolled(::typeof(^), a::Taylor, ::Val{n}) where {n} = _literal_pow_unrolled_body(n)

# `literal_pow` by a loop over the bits of `|n|`, for an integer `n ≠ 0`, with the products of the
# unrolled form in the same order: `r` starts as the first factor rather than as one.
function _literal_pow_loop(a::Taylor, ::Val{n}) where {n}
    m = abs(n)
    x, r, started = a, a, false
    while true
        if isodd(m)
            r = started ? r * x : x
            started = true
        end
        m ÷= 2
        iszero(m) && break
        x = x * x
    end
    return n < 0 ? inv(r) : r
end
