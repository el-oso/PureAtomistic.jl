# Oracle: the Base-named methods on Taylor numbers, which use Base's value functions and the same
# derivative formulas. The value is held to the scalar kernels' accuracy tolerance, 4 ulps (bound (c) of
# fast_tests.jl), here measured against Base's value. The value enters every other coefficient
# linearly (for log and log1p not at all), so the rest is bounded by the same 4 ulps relative plus the
# rounding of the shared formulas, taken twice (once per side): c = 2 (one rounding per multiply and
# per add), N = 30 (K = 3 Horner steps, each a sum of at most L = 10 products per coefficient),
# relative to the largest coefficient of the oracle. Largest measured: value 1 ulp, rest 1.2·eps·scale
# (Float64); 0 for Float32 at these points.
@testitem "fast functions on Taylor numbers agree with the Base methods" begin
    cases = (
        (fast_sqrt, sqrt), (fast_cbrt, cbrt), (fast_invcbrt, a -> a^(-1 // 3)), (fast_log, log),
        (fast_log1p, log1p), (fast_exp, exp), (fast_expm1, expm1),
    )
    for T in (Float32, Float64), x0 in (T(0.3), T(1.7)), (f, g) in cases
        a = variable(Val(2), Val(3), Val(1), x0, x0) + variable(Val(2), Val(3), Val(2), T(0.4), T(0.2))
        got, ref = f(a), g(a)
        @test got isa typeof(ref)
        @test value(got) == f(value(a))                             # exact: the scalar kernel's value
        @test abs(value(got) - value(ref)) <= 4 * eps(value(ref))
        scale = maximum(abs, ref.c)
        @test all(abs(got.c[j] - ref.c[j]) <= (4 + 2 * 2 * 30) * eps(T) * scale for j in 2:length(ref.c))
    end
end

@testitem "fast functions on Taylor numbers skip the domain check" begin
    a = variable(Val(1), Val(2), Val(1), -1.0)
    for f in (fast_sqrt, fast_log, fast_log1p ∘ (b -> b - 1))
        @test f(a) isa Taylor{1,2,1,Float64,3}
    end
end

@testitem "fast functions pass through nested ForwardDiff" begin
    using ForwardDiff
    f_fast(x) = fast_cbrt(x) * fast_log1p(x^2)
    f_base(x) = cbrt(x) * log1p(x^2)
    d(g) = x -> ForwardDiff.derivative(g, x)
    for (gf, gb) in ((f_fast, f_base), (d(f_fast), d(f_base)), (d(d(f_fast)), d(d(f_base))))
        @test d(gf)(0.7) == d(gb)(0.7)                              # exact: the same Base calls
    end
end
