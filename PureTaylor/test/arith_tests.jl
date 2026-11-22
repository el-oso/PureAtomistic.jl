@testmodule DenseOracle begin
    using PureTaylor
    using PureTaylor: multi_indices, stored_indices, coefficient

    # A Taylor{N,K,M} with random Rational{BigInt} coefficients (nonzero value, so it is invertible).
    function random_taylor(N, K, M, rng = 0)
        idx = stored_indices(Val(N), Val(K), Val(M))
        c = ntuple(j -> Rational{BigInt}(mod(7j + 3rng + 1, 11) - 4 + (j == 1 ? 9 : 0), 1 + mod(5j + rng, 6)), length(idx))
        return Taylor{N, K, M, Rational{BigInt}, length(idx)}(c)
    end

    # The truncated product by brute force over all pairs of multi-indices: every pair whose sum has
    # total degree ≤ D contributes, and `a`'s constant term is left out when `skip0`.
    function dense_product(a::Taylor{N, K}, b::Taylor{N, K}; D = K, skip0 = false) where {N, K}
        out = Dict(α => zero(Rational{BigInt}) for α in multi_indices(Val(N), Val(K)))
        for β in multi_indices(Val(N), Val(K)), γ in multi_indices(Val(N), Val(K))
            skip0 && sum(β) == 0 && continue
            α = β .+ γ
            sum(α) <= D || continue
            out[α] += coefficient(a, β) * coefficient(b, γ)
        end
        return out
    end

    matches(c::Taylor{N, K}, ref) where {N, K} = all(coefficient(c, α) == ref[α] for α in multi_indices(Val(N), Val(K)))
end

@testitem "product equals the dense product (exact)" setup = [DenseOracle] begin
    using .DenseOracle: random_taylor, dense_product, matches
    for (N, K, M1, M2) in ((1, 4, 1, 1), (2, 3, 1, 2), (3, 3, 3, 6), (3, 2, 7, 0), (2, 0, 1, 2))
        a = random_taylor(N, K, M1, 1); b = random_taylor(N, K, M2, 2)
        c = a * b
        @test c isa Taylor{N, K, M1 | M2}
        @test matches(c, dense_product(a, b))                     # exact: Rational{BigInt}
    end
end

@testitem "degree-limited product (exact)" setup = [DenseOracle] begin
    using .DenseOracle: random_taylor, dense_product, matches
    using PureTaylor: mul_truncated
    a = random_taylor(3, 3, 3, 1); b = random_taylor(3, 3, 6, 2)
    for D in 0:3, skip0 in (false, true)
        c = mul_truncated(a, b, Val(skip0), Val(D))
        @test c isa Taylor{3, 3, 7}
        @test matches(c, dense_product(a, b; D, skip0))         # exact: Rational{BigInt}
    end
end

@testitem "scalar product and division (exact)" setup = [DenseOracle] begin
    using .DenseOracle: random_taylor
    a = random_taylor(2, 3, 3, 1)
    @test (a * 3).c == map(x -> 3x, a.c)
    @test (3 * a).c == map(x -> 3x, a.c)
    @test (a / 3).c == map(x -> x / 3, a.c)
    @test (a / 3) isa typeof(a)
end

@testitem "inverse and division (exact)" setup = [DenseOracle] begin
    using .DenseOracle: random_taylor
    # `==` on a Taylor is `===`, so the exact comparisons compare the coefficient tuples.
    for (N, K, M) in ((1, 4, 1), (2, 3, 3), (3, 3, 5), (2, 2, 0), (2, 0, 1))
        a = random_taylor(N, K, M, 1)
        @test inv(a) isa typeof(a)
        @test (inv(a) * a).c == one(a).c                                 # exact: Rational{BigInt}
    end
    a = random_taylor(2, 3, 1, 1); b = random_taylor(2, 3, 2, 2)
    @test (a / b).c == (a * inv(b)).c
    @test (2 / b).c == (2 * inv(b)).c
    @test ((2 / b) * b).c == (2 * one(b)).c
end

@testitem "powers (exact)" setup = [DenseOracle] begin
    using .DenseOracle: random_taylor
    a = random_taylor(2, 3, 3, 1)
    @test (a^0).c == one(a).c
    @test (a^1).c == a.c
    @test (a^2).c == (a * a).c
    @test (a^3).c == (a * a * a).c
    @test (a^5).c == (a * a * a * a * a).c
    @test (a^-1).c == inv(a).c
    @test (a^-3).c == inv(a * a * a).c
    for n in -5:6                                                  # runtime exponent
        ref = n >= 0 ? prod(_ -> a, 1:n; init = one(a)) : inv(prod(_ -> a, 1:(-n); init = one(a)))
        @test (a^n).c == ref.c
        @test (a^BigInt(n)).c == ref.c
    end
end

# Julia lowers only integer literals to literal_pow; a hand-written call with another exponent is the
# general power.
@testitem "literal_pow with a non-integer exponent" begin
    a = variable(Val(1), Val(3), Val(1), 2.0)
    @test Base.literal_pow(^, a, Val(2.5)).c == (a^2.5).c           # exact: the same call
end

@testitem "series of a Taylor with integer coefficients" begin
    a = variable(Val(1), Val(2), Val(1), 2)
    msg = "PureTaylor: the series of a Taylor with coefficient type Int64 needs coefficients of type Float64"
    @test_throws msg inv(a)
    @test_throws msg sqrt(a)
    @test_throws msg exp(a)
    @test_throws msg log(a)
end

@testitem "inverse against nested ForwardDiff" begin
    using ForwardDiff
    using PureTaylor: multi_indices
    x0, y0 = 0.7, 1.3
    x = variable(Val(2), Val(3), Val(1), x0); y = variable(Val(2), Val(3), Val(2), y0)
    got = derivatives(inv(x * y + 2), (1.0, 1.0))
    ∂x(g) = (x, y) -> ForwardDiff.derivative(t -> g(t, y), x)
    ∂y(g) = (x, y) -> ForwardDiff.derivative(t -> g(x, t), y)
    function partial(α)
        g = (x, y) -> inv(x * y + 2)
        for _ in 1:α[1]
            g = ∂x(g)
        end
        for _ in 1:α[2]
            g = ∂y(g)
        end
        return g(big(x0), big(y0))
    end
    idx = multi_indices(Val(2), Val(3))
    ref = map(partial, idx)
    for j in eachindex(idx)
        # Rounding bound c·eps(Float64)·N with c = 2 (one rounding per multiply and per add) and
        # N = 40 roundings per coefficient: x·y (at most 3 products), + 2 (1), and the division
        # recurrence, through K = 3 degrees of at most 9 products (one rounding each in the fma
        # chain), one subtraction and one division each (33); relative to the largest reference
        # derivative of the same total degree. Largest measured: 1.1·eps·scale.
        scale = maximum(abs(ref[i]) for i in eachindex(idx) if sum(idx[i]) == sum(idx[j]))
        @test abs(got[j] - ref[j]) <= 2 * eps(Float64) * 40 * scale
    end
end

@testitem "order 0: power and division keep only the value" begin
    a = variable(Val(1), Val(0), Val(1), 2.0)^3 / 4
    @test a isa Taylor{1, 0, 1, Float64, 1}
    @test a.c == (2.0,)                                            # exact: 8/4 in binary
end

@testitem "Float32 stays Float32 through products, inverse and division" begin
    x = variable(Val(2), Val(2), Val(1), 1.5f0); y = variable(Val(2), Val(2), Val(2), 2.0f0)
    @test (x * y) isa Taylor{2, 2, 3, Float32}
    @test inv(x) isa Taylor{2, 2, 1, Float32}
    @test (x / 0.1) isa Taylor{2, 2, 1, Float32}
    @test (0.1 / x) isa Taylor{2, 2, 1, Float32}
    @test (x * 0.1) isa Taylor{2, 2, 1, Float32}
    @test x^-2 isa Taylor{2, 2, 1, Float32}
    @test_throws MethodError x * variable(Val(2), Val(2), Val(1), 1.0)
    @test_throws MethodError x * variable(Val(2), Val(3), Val(1), 1.0f0)
end

@testitem "products, inverse and substitution allocate nothing at L = 56" begin
    using StrictModeTest, StrictMode
    @noinline mul(a, b) = a * b
    @noinline inverse(a) = inv(a)
    @noinline neg(a) = -a
    @noinline rsub(r, a) = r - a
    @noinline subst(E, u) = substitute(E, u)
    for T in (Float64, Float32)
        v = ntuple(i -> variable(Val(5), Val(3), Val(i), T(i) / 4), 5)
        a = sum(v) + 1; b = a * v[1] + 2
        @test a isa Taylor{5, 3, 31, T, 56}
        u = (a, b, v[2] * v[3], a - v[4], v[5] + 3)
        mul(a, b); inverse(a); neg(a); rsub(T(1), a); subst(b, u)
        @test_noalloc mul(a, b)
        @test_noalloc inverse(a)
        @test_noalloc neg(a)
        @test_noalloc rsub(T(1), a)
        @test_noalloc subst(b, u)
    end
end
