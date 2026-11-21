@testitem "compose equals the sum of powers (exact)" begin
    using PureTaylor: compose
    for (N, K, M) in ((1, 4, 1), (2, 3, 3), (3, 2, 5), (2, 0, 1), (2, 3, 0))
        idx = PureTaylor.stored_indices(Val(N), Val(K), Val(M))
        L = length(idx)
        a = Taylor{N,K,M,Rational{BigInt},L}(ntuple(j -> Rational{BigInt}(mod(5j + 2, 9) - 3, 1 + mod(3j, 4)), L))
        d = ntuple(k -> Rational{BigInt}(k^2 - 3, k + 1), K + 1)
        s = Rational{BigInt}(-2, 3)
        t = s * (a - value(a))
        ref = sum(d[k + 1] * t^k for k in 0:K)
        c = compose(a, d, s)
        @test c isa typeof(a)
        @test c.c == ref.c                                          # exact: Rational{BigInt}
    end
end

@testitem "compose rejects a wrong number of derivatives" begin
    using PureTaylor: compose
    a = variable(Val(1), Val(3), Val(1), 2.0)
    @test_throws "PureTaylor: compose needs K + 1 = 4 derivatives, got 3" compose(a, (1.0, 2.0, 3.0), 1.0)
    @test_throws "PureTaylor: compose needs K + 1 = 4 derivatives, got 5" compose(a, (1.0, 2.0, 3.0, 4.0, 5.0), 1.0)
end

@testitem "elementary functions against closed-form derivatives (BigFloat)" begin
    setprecision(BigFloat, 256) do
        x0 = big"0.7"
        p = big(4) / 3
        # f⁽ᵏ⁾(x) of xʳ: r(r-1)⋯(r-k+1)·x^(r-k).
        falling(r, k) = prod((r - j for j in 0:(k - 1)); init = one(BigFloat))
        pow_ref(r) = (x, k) -> falling(r, k) * x^(r - k)
        log_ref(x, k) = k == 0 ? log(x) : (-1)^(k - 1) * factorial(big(k - 1)) / x^k
        cases = (
            (sqrt, pow_ref(big(1) / 2)),
            (cbrt, pow_ref(big(1) / 3)),
            (a -> a^p, pow_ref(p)),
            (exp, (x, k) -> exp(x)),
            (expm1, (x, k) -> k == 0 ? expm1(x) : exp(x)),
            (log, log_ref),
            (log1p, (x, k) -> k == 0 ? log1p(x) : log_ref(1 + x, k)),
        )
        for (f, ref) in cases, s in (one(BigFloat), x0)
            # The seed x0 + s·u: derivatives in x are recovered with invscale 1/s.
            got = derivatives(f(variable(Val(1), Val(4), Val(1), x0, s)), (inv(s),))
            for k in 0:4
                r = ref(x0, k)
                # Rounding bound c·eps(BigFloat)·N relative to the reference, with eps = 1.7e-77 at
                # 256 bits, c = 2 (one rounding per multiply and per add) and N = 100 (K = 4 Horner
                # steps, the coefficient recurrence, the seed unscaling and Base's value);
                # largest error measured: 1.8·eps.
                @test abs(got[k + 1] - r) <= 2 * eps(BigFloat) * 100 * abs(r)
            end
        end
        # sqrt(x)² recovers x through the series: a check independent of the closed forms. Same
        # bound as above, taken as absolute since every coefficient is at most 1 in magnitude.
        a = variable(Val(1), Val(4), Val(1), x0)
        @test maximum(abs.((sqrt(a) * sqrt(a) - a).c)) <= 2 * eps(BigFloat) * 100
    end
end

@testitem "composite of elementary functions against nested ForwardDiff" begin
    using ForwardDiff
    using PureTaylor: multi_indices
    f(x, y) = log1p(exp(-x) * y^(4 / 3)) / sqrt(x + cbrt(y))
    x0, y0 = 0.7, 1.3
    x = variable(Val(2), Val(3), Val(1), x0); y = variable(Val(2), Val(3), Val(2), y0)
    got = derivatives(f(x, y), (1.0, 1.0))
    ∂x(g) = (x, y) -> ForwardDiff.derivative(t -> g(t, y), x)
    ∂y(g) = (x, y) -> ForwardDiff.derivative(t -> g(x, t), y)
    function partial(α)
        g = f
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
        # N = 300: ten series operations (exp, ^, cbrt, sqrt, log1p, inv and four products or
        # sums), each at most K = 3 Horner steps over sums of at most L = 10 products, relative
        # to the largest reference derivative of the same total degree.
        scale = maximum(abs(ref[i]) for i in eachindex(idx) if sum(idx[i]) == sum(idx[j]))
        @test abs(got[j] - ref[j]) <= 2 * eps(Float64) * 300 * scale
    end
end

@testitem "powers with a Taylor exponent against nested ForwardDiff" begin
    using ForwardDiff
    using PureTaylor: multi_indices
    x0, y0 = 0.7, 1.3
    x = variable(Val(2), Val(3), Val(1), x0); y = variable(Val(2), Val(3), Val(2), y0)
    got = derivatives(x^y, (1.0, 1.0))
    ∂x(g) = (x, y) -> ForwardDiff.derivative(t -> g(t, y), x)
    ∂y(g) = (x, y) -> ForwardDiff.derivative(t -> g(x, t), y)
    function partial(α)
        g = (x, y) -> x^y
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
        # N = 62 roundings per coefficient: log(x) (Base's value 1, the scale 1/x 1, the
        # coefficients ±1/k 3, s·h 1 and K = 3 Horner steps of at most 3 products and 1 add 12:
        # 18), the product y·log(x) (at most 2 products and 1 add: 3), exp (Base's value 1, the
        # recurrence e/k 3 and 3 Horner steps of at most 10 products and 1 add 33: 37) and the
        # unscaling (α! and at most 3 invscale factors: 4); relative to the largest reference
        # derivative of the same total degree. Largest measured: 2.25·eps·scale.
        scale = maximum(abs(ref[i]) for i in eachindex(idx) if sum(idx[i]) == sum(idx[j]))
        @test abs(got[j] - ref[j]) <= 2 * eps(Float64) * 62 * scale
    end
end

@testitem "powers with a Taylor exponent: real and ℯ bases" begin
    y = variable(Val(2), Val(3), Val(2), 1.3) + variable(Val(2), Val(3), Val(1), 0.0)
    @test (2.0^y).c == exp(y * log(2.0)).c                          # exact: the same calls
    @test (2^y).c == exp(y * log(2.0)).c                            # exact: 2 converts to 2.0
    @test (ℯ^y).c == exp(y).c                                       # exact: the same call
    yf = variable(Val(1), Val(2), Val(1), 1.5f0)
    @test (2^yf) isa Taylor{1,2,1,Float32,3}
    @test (yf^yf) isa Taylor{1,2,1,Float32,3}
end

@testitem "log at value 0 and log1p at value -1 are -Inf" begin
    l = log(variable(Val(1), Val(3), Val(1), 0.0))
    @test value(l) == -Inf && !isfinite(l)                          # exact: Base's log(0.0)
    l1 = log1p(variable(Val(1), Val(3), Val(1), -1.0))
    @test value(l1) == -Inf && !isfinite(l1)                        # exact: Base's log1p(-1.0)
end

@testitem "elementary functions keep Base's domain errors" begin
    a = variable(Val(1), Val(2), Val(1), -1.0)
    @test_throws DomainError log(a)
    @test_throws DomainError sqrt(a)
    @test_throws DomainError a^0.5
    @test_throws DomainError log1p(a - 1)
    @test_throws DomainError a^variable(Val(1), Val(2), Val(1), 0.5)
end

@testitem "order 0: elementary functions give the value" begin
    a = exp(constant(Val(1), Val(0), 1.0))
    @test a isa Taylor{1,0,0,Float64,1}
    @test value(a) == exp(1.0)                                      # exact: the value is Base's exp
end

@testitem "rational and real exponents agree" begin
    a = variable(Val(2), Val(3), Val(1), 1.7)
    @test (a^(1 // 2)).c == (a^0.5).c                               # exact: one conversion to 0.5
    @test value(a^(1 // 3)) == 1.7^(1 / 3)                          # exact: the value is Base's ^
end

@testitem "Float32 stays Float32 through the elementary functions" begin
    x = variable(Val(2), Val(2), Val(1), 1.5f0) + variable(Val(2), Val(2), Val(2), 0.25f0)
    for f in (sqrt, cbrt, exp, expm1, log, log1p, a -> a^0.3, a -> a^(1 // 3))
        @test f(x) isa Taylor{2,2,3,Float32,6}
    end
end

@testitem "powers at value 0" begin
    a = variable(Val(2), Val(3), Val(1), 0.0) + variable(Val(2), Val(3), Val(2), 0.0)
    @test (a^2.0).c == (a * a).c                                    # exact: the same products
    @test (a^(3 // 1)).c == (a * a * a).c                           # exact: the same products
    # √ is singular at 0: the value is 0 and every higher coefficient is NaN, as documented.
    s = sqrt(a)
    @test value(s) == 0.0
    @test all(isnan, s.c[2:end])
    @test all(isnan, cbrt(a).c[2:end]) && all(isnan, (a^0.5).c[2:end])
end
