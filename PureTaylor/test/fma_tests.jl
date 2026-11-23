# The multiply-adds of the kernels go through `PureTaylor._fma`: Base's `fma` for the IEEE floats and
# BigFloat, `x * y + z` for every other coefficient type.

# Results of every kernel that multiply-adds, on fixed series, for the reference checksums in
# `fma_reference.toml`. Those checksums were computed with Base's `fma` at every multiply-add of the
# kernels, so for Float32, Float64 and BigFloat the results must match them bit for bit.
@testmodule FmaReference begin
    using PureTaylor
    const P = PureTaylor

    # A series over the variables of `mask`, with coefficients of mixed sign and size, value > 0.
    function series(::Val{N}, ::Val{K}, ::Type{T}, mask, shift) where {N, K, T}
        s = constant(Val(N), Val(K), T(shift))
        for i in 1:N
            isodd(mask >> (i - 1)) || continue
            s = s + variable(Val(N), Val(K), Val(i), T(3 + i) / 10, T(-1)^i * T(5 + i) / 9)
        end
        return exp(s) + inv(s + 2)
    end

    # (N, K, mask of a, mask of b): L = 7, 20, 21 and 35 unrolled, 165 in the loop forms.
    const CASES = ((1, 6, 1, 1), (3, 3, 7, 5), (5, 2, 31, 7), (3, 4, 7, 6), (3, 8, 7, 3))

    # name => coefficients, for the series operations of PureTaylor and the scalar `fast_*` kernels.
    function results(::Type{T}) where {T}
        out = Pair{String, Vector{T}}[]
        put!(name, x::Taylor) = push!(out, name => collect(x.c))
        for (N, K, m1, m2) in CASES
            a = series(Val(N), Val(K), T, m1, 0.4); b = series(Val(N), Val(K), T, m2, -0.3)
            tag = "N$(N)_K$(K)_$(m1)_$(m2)"
            put!("mul_$tag", a * b)
            put!("mul_truncated_$tag", P.mul_truncated(a, b, Val(true), Val(max(K - 1, 0))))
            put!("div_$tag", a / b)
            put!("inv_$tag", inv(b))
            put!("compose_$tag", P.compose(a, ntuple(i -> T(-1)^i * T(7) / (i + 2), Val(K + 1)), T(-0.8)))
            for (fname, f) in (
                    ("fast_sqrt", fast_sqrt), ("fast_cbrt", fast_cbrt), ("fast_invcbrt", fast_invcbrt),
                    ("fast_log", fast_log), ("fast_log1p", fast_log1p), ("fast_exp", fast_exp), ("fast_expm1", fast_expm1),
                )
                put!("$(fname)_$tag", f(a))
            end
        end
        for K in (2, 4)
            x = variable(Val(3), Val(K), Val(1), T(0.7)); y = variable(Val(3), Val(K), Val(2), T(0.2))
            z = variable(Val(3), Val(K), Val(3), T(1.3))
            E = exp(x * y) + log(z) * x - y * y * z
            v = ntuple(i -> variable(Val(5), Val(K), Val(i), T(i) / 4 + T(0.1)), Val(5))
            put!("substitute_K$K", substitute(E, (v[1] + v[2], (v[1] - v[2]) / (v[1] + v[2]), v[3] + 2 * v[4] + v[5])))
            put!("substitute_loop_K$K", P._substitute_loop(E, (v[1] * v[5], T(0.3), v[2] + v[3])))
        end
        if T <: Union{Float32, Float64}
            xs = T.(range(0.05, 3.0; length = 64))
            for (fname, f) in (
                    ("fast_cbrt", fast_cbrt), ("fast_invcbrt", fast_invcbrt), ("fast_log", fast_log),
                    ("fast_log1p", fast_log1p), ("fast_exp", fast_exp), ("fast_expm1", fast_expm1),
                )
                push!(out, "scalar_$fname" => map(f, xs))
            end
        end
        return out
    end

    # FNV-1a (64 bit) over the exact bit patterns of the coefficients, as a hex string.
    function _fnv(h::UInt64, bytes)
        for b in bytes
            h = (h ⊻ UInt64(b)) * 0x00000100000001b3
        end
        return h
    end
    _bytes(x::Float32) = reinterpret(NTuple{4, UInt8}, x)
    _bytes(x::Float64) = reinterpret(NTuple{8, UInt8}, x)
    # The sign, binary exponent and significand of a finite BigFloat, as text; NaN and ±Inf by name.
    function _bytes(x::BigFloat)
        isfinite(x) || return codeunits(string(x))
        iszero(x) && return codeunits(signbit(x) ? "-0" : "0")
        m, e = frexp(x)
        return codeunits(string(BigInt(ldexp(m, precision(x))); base = 16) * "p" * string(e) * "/" * string(precision(x)))
    end
    checksum(v::AbstractVector) = string(foldl((h, x) -> _fnv(h, _bytes(x)), v; init = 0xcbf29ce484222325); base = 16, pad = 16)

    # `T => name => checksum` for the three float types, BigFloat at 256 bits.
    reference() = Dict(
        string(T) => Dict(name => checksum(v) for (name, v) in setprecision(() -> results(T), BigFloat, 256))
            for T in (Float32, Float64, BigFloat)
    )
end

@testitem "Float32, Float64 and BigFloat results match the reference checksums" setup = [FmaReference] begin
    using TOML
    using .FmaReference: reference
    stored = TOML.parsefile(joinpath(@__DIR__, "fma_reference.toml"))
    got = reference()
    @test keys(got) == keys(stored)
    for T in keys(stored)
        @test keys(got[T]) == keys(stored[T])
        for name in keys(stored[T])
            @test (T, name, got[T][name]) == (T, name, stored[T][name])
        end
    end
end

# A Taylor in u whose coefficients are Taylors in v (Rational{BigInt}, so every check is exact). The
# oracle of the product is the truncated two-dimensional convolution of the coefficient arrays.
@testitem "a Taylor with Taylor coefficients (exact)" begin
    using PureTaylor
    using PureTaylor: compose
    R = Rational{BigInt}
    inner(cs...) = Taylor{1, 3, 1, R, 4}(R.(cs))
    I = typeof(inner(0, 0, 0, 0))
    outer(cs...) = Taylor{1, 2, 1, I, 3}(cs)
    X = outer(inner(3 // 2, 1, 0, 0), inner(1, -1 // 3, 2, 0), inner(0, 1, 0, 5 // 7))
    Y = outer(inner(2, 1 // 2, -1, 1), inner(-1 // 4, 3, 0, 1), inner(1, 0, 2, -1))
    # coefficient [k, j] of uᵏ⁻¹vʲ⁻¹
    grid(Z) = [Z.c[k].c[j] for k in 1:3, j in 1:4]
    function conv(A, B)
        C = zeros(R, 3, 4)
        for k1 in 1:3, j1 in 1:4, k2 in 1:(4 - k1), j2 in 1:(5 - j1)
            C[k1 + k2 - 1, j1 + j2 - 1] += A[k1, j1] * B[k2, j2]
        end
        return C
    end
    unit = [k == j == 1 ? one(R) : zero(R) for k in 1:3, j in 1:4]
    @test X * Y isa typeof(X)
    @test grid(X * Y) == conv(grid(X), grid(Y))
    @test grid(X / Y) isa Matrix{R}
    @test conv(grid(X / Y), grid(Y)) == grid(X)
    @test conv(grid(inv(Y)), grid(Y)) == unit
    # `compose` adds a coefficient (a Taylor) to a series, which dispatches to the Taylor + Taylor
    # method and has no method for these types.
    @test_broken compose(X, Y.c, X.c[2]) isa typeof(X)
end

# A Real type of the tests, defined after PureTaylor: exact rational arithmetic, no `fma` method of
# its own.
@testmodule ExactReal begin
    struct Q <: Real
        v::Rational{BigInt}
    end
    Q(x::Integer) = Q(Rational{BigInt}(x))
    Base.:+(a::Q, b::Q) = Q(a.v + b.v)
    Base.:-(a::Q, b::Q) = Q(a.v - b.v)
    Base.:-(a::Q) = Q(-a.v)
    Base.:*(a::Q, b::Q) = Q(a.v * b.v)
    Base.:/(a::Q, b::Q) = Q(a.v / b.v)
    Base.inv(a::Q) = Q(inv(a.v))
    Base.zero(::Type{Q}) = Q(0)
    Base.one(::Type{Q}) = Q(1)
    # The square root of a rational square; any other argument is an error.
    function Base.sqrt(a::Q)
        n, d = isqrt(numerator(a.v)), isqrt(denominator(a.v))
        n^2 == numerator(a.v) && d^2 == denominator(a.v) || throw(DomainError(a.v, "not a rational square"))
        return Q(n // d)
    end
end

# Oracle: the same series in Rational{BigInt}, exact; for `fast_sqrt` the binomial series.
@testitem "a Taylor with a custom Real coefficient type (exact)" setup = [ExactReal] begin
    using PureTaylor
    using PureTaylor: compose
    using .ExactReal: Q
    R = Rational{BigInt}
    x0, a0, b0 = R(9, 4), R(-1, 3), R(5, 2)
    q = variable(Val(2), Val(3), Val(1), Q(x0)) + variable(Val(2), Val(3), Val(2), Q(a0), Q(b0))
    r = variable(Val(2), Val(3), Val(1), x0) + variable(Val(2), Val(3), Val(2), a0, b0)
    vals(z) = map(c -> c.v, z.c)
    @test vals(q * q * q) == (r * r * r).c
    d = (R(1), R(2), R(-3), R(4))
    @test vals(compose(q, Q.(d), Q(R(1, 3)))) == compose(r, d, R(1, 3)).c
    # √(x0 + u): coefficient k is binomial(1/2, k)·x0^(1/2 - k).
    u = variable(Val(1), Val(3), Val(1), Q(x0))
    binom(p, k) = prod((p - i + 1) / i for i in 1:k; init = one(R))
    @test vals(fast_sqrt(u)) == ntuple(k -> binom(R(1, 2), k - 1) * R(3, 2) / x0^(k - 1), 4)
    # `_div` takes the quotient type from `promote_op` when its method is generated, which does not
    # see the methods of a type defined after PureTaylor.
    @test_broken vals(q / (q + q)) == (r / (r + r)).c
    @test_broken vals(inv(q)) == inv(r).c
end
