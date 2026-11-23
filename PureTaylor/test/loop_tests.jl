# The loop forms of the kernels against their unrolled forms, on both sides of `_UNROLL_MAX`. Each
# kernel's unrolled form is reachable at any size as `_<kernel>_unrolled` (the same generated
# expression the kernel uses at or below the threshold) and its loop form as `_<kernel>_loop`.
#
# Every kernel accumulates with `_fma`, which rounds once for the IEEE floats and BigFloat and is
# the plain `x * y + z` for other coefficient types. A loop form and its unrolled form perform the
# same operations in the same order and agree exactly (`==`).
@testmodule LoopCases begin
    using PureTaylor
    const P = PureTaylor

    # A series over the variables of `mask`, with coefficients of mixed sign and size.
    function series(::Val{N}, ::Val{K}, ::Type{T}, mask, shift) where {N, K, T}
        s = constant(Val(N), Val(K), T(shift))
        for i in 1:N
            isodd(mask >> (i - 1)) || continue
            s = s + variable(Val(N), Val(K), Val(i), T(3 + i) / 10, T(-1)^i * T(5 + i) / 9)
        end
        return exp(s) - inv(s + 2)
    end

    # (N, K, mask of a, mask of b): L = 7, 35, 56, 84 and 120 at or below the threshold (126); 165,
    # 252 and 286 above it.
    const CASES = (
        (1, 6, 1, 1), (3, 4, 3, 6), (5, 3, 31, 7), (3, 6, 7, 7), (3, 7, 5, 7),
        (3, 8, 7, 3), (5, 5, 31, 31), (3, 10, 6, 7),
    )

end

@testitem "loop kernels equal the unrolled kernels" setup = [LoopCases] begin
    using PureTaylor
    using .LoopCases: P, series, CASES
    for T in (Float64, Float32), (N, K, m1, m2) in CASES
        a = series(Val(N), Val(K), T, m1, 0.4); b = series(Val(N), Val(K), T, m2, -0.3)
        for sgn in (1, -1)
            x, y = P._addsub_loop(a, b, Val(sgn)), P._addsub_unrolled(a, b, Val(sgn))
            @test typeof(x) == typeof(y)
            @test x.c == y.c                                         # exact: the same operations
        end
        for D in (0, 1, K), r in (T(-2.5), T(Inf))
            # isequal: an infinite r gives NaN where a coefficient is zero, in both forms alike.
            @test isequal(P._scale_from_loop(a, r, D).c, P._scale_from_unrolled(a, r, Val(D)).c)
        end
        invscale = ntuple(i -> T(i + 1) / 3, Val(N))
        @test P._derivatives_loop(a, invscale) == P._derivatives_unrolled(a, invscale)  # exact
        for α in P.multi_indices(Val(N), Val(K))
            @test P._coefficient_loop(a, α) == P._coefficient_unrolled(a, α)              # exact: a copy
        end
        @test P._coefficient_loop(a, ntuple(_ -> K + 1, Val(N))) == zero(T)
        # The loop forms of `literal_pow` and of `compose` without products (K = 0 below) call the
        # same product kernel as their unrolled forms where they are used, above the threshold.
        if P._use_loop(length(a.c))
            for n in (2, 3, 5, -3)
                @test P._literal_pow_loop(a, Val(n)).c == P._literal_pow_unrolled(^, a, Val(n)).c  # exact
            end
        end
    end
    # The recurrence, below and above the threshold (K + 1 = 7 and 151).
    for T in (Float64, Float32), K in (6, 150)
        g = (b, k) -> b * (T(0.3) - T(k - 1)) / T(k)
        @test P._loop_recurrence(g, T(1.3), Val(K + 1)) == P._recurrence_unrolled(g, T(1.3), Val(K))
    end
    # compose at K = 0 has no product.
    a0 = constant(Val(3), Val(0), 2.0)
    @test P._compose_loop(a0, (5.0,), 0.5) == P._compose_unrolled(a0, (5.0,), 0.5)
end

# Oracle: the unrolled product, exact.
@testitem "loop product equals the unrolled product" setup = [LoopCases] begin
    using PureTaylor
    using .LoopCases: P, series, CASES
    for T in (Float64, Float32), (N, K, m1, m2) in CASES
        a = series(Val(N), Val(K), T, m1, 0.4); b = series(Val(N), Val(K), T, m2, -0.3)
        for (skip0, D) in ((false, K), (true, K), (true, max(K - 2, 0)))
            x = P._mul_loop(a, b, skip0, D)
            y = P._mul_unrolled(a, b, Val(skip0), Val(D))
            @test typeof(x) == typeof(y)
            @test x.c == y.c
        end
    end
end

# Oracle: the unrolled compose, exact.
@testitem "loop compose equals the unrolled compose" setup = [LoopCases] begin
    using PureTaylor
    using .LoopCases: P, series, CASES
    for T in (Float64, Float32), (N, K, m1, _) in CASES
        a = series(Val(N), Val(K), T, m1, 0.4)
        d = ntuple(i -> T(-1)^i * T(7) / (i + 2), Val(K + 1))
        s = T(-0.8)
        x, y = P._compose_loop(a, d, s), P._compose_unrolled(a, d, s)
        @test typeof(x) == typeof(y)
        @test x.c == y.c
    end
end

# Oracle: the unrolled substitute, exact.
@testitem "loop substitute equals the unrolled substitute" setup = [LoopCases] begin
    using PureTaylor
    using .LoopCases: P
    for T in (Float64, Float32), K in (2, 3, 4, 6)
        x = variable(Val(3), Val(K), Val(1), T(0.7)); y = variable(Val(3), Val(K), Val(2), T(0.2))
        z = variable(Val(3), Val(K), Val(3), T(1.3))
        E = exp(x * y) + log(z) * x - y * y * z
        v = ntuple(i -> variable(Val(5), Val(K), Val(i), T(i) / 4 + T(0.1)), Val(5))
        # u over the masks 0b00011, 0b00011 and 0b11100, as in polarized pbe_c; then with a plain
        # number and a u element whose variable E does not use.
        for u in ((v[1] + v[2], (v[1] - v[2]) / (v[1] + v[2]), v[3] + 2 * v[4] + v[5]), (v[1] * v[5], T(0.3), v[2] + v[3]))
            r1, r2 = P._substitute_loop(E, u), P._substitute_unrolled(E, u)
            @test typeof(r1) == typeof(r2)
            @test r1.c == r2.c
        end
    end
end

# Both forms with a coefficient type that is not isbits, above the threshold: the loop forms then
# build their tuples through a Vector. Exact in Rational{BigInt}.
@testitem "loop kernels with Rational{BigInt} coefficients (exact)" setup = [LoopCases] begin
    using PureTaylor
    using .LoopCases: P
    R = Rational{BigInt}
    x = variable(Val(3), Val(8), Val(1), R(1, 3)); y = variable(Val(3), Val(8), Val(2), R(2)); z = variable(Val(3), Val(8), Val(3), R(-1, 2))
    a = (x + y * z + 1)^2; b = x * z - y + R(5, 7)
    @test P._use_loop(length(a.c))
    # All exact: Rational{BigInt}.
    @test (a * b).c == P._mul_unrolled(a, b, Val(false), Val(8)).c
    @test (a + b).c == P._addsub_unrolled(a, b, Val(1)).c
    d = ntuple(k -> isodd(k) ? inv(value(b)) : -inv(value(b)), Val(9))
    @test inv(b).c == P._compose_unrolled(b, d, inv(value(b))).c
    @test (a^3).c == P._literal_pow_unrolled(^, a, Val(3)).c
    @test derivatives(a, (R(1), R(2), R(3))) == P._derivatives_unrolled(a, (R(1), R(2), R(3)))
    @test P.coefficient(a, (1, 1, 1)) == P._coefficient_unrolled(a, (1, 1, 1))
    @test P._scale_from(a, R(3), Val(2)).c == P._scale_from_unrolled(a, R(3), Val(2)).c
    g = (q, k) -> q * (R(1, 3) - (k - 1)) / k
    @test P._recurrence(g, R(1), Val(139)) == P._recurrence_unrolled(g, R(1), Val(139))
    u = (x * y + z, R(2), y - z)
    @test substitute(b * a, u).c == P._substitute_unrolled(b * a, u).c
end

# Below the threshold each kernel is its unrolled form: the same lowered code. Above it, the kernel
# is its loop form, and `substitute` is not inlined.
@testitem "kernels switch to the loop form above the threshold" setup = [LoopCases] begin
    using PureTaylor
    using .LoopCases: P, series
    lowered(f, types) = only(code_lowered(f, types))
    @test P._UNROLL_MAX == 126
    small = series(Val(5), Val(2), Float64, 31, 0.4)                 # L = 21, the largest PureXC
    big = series(Val(5), Val(4), Float64, 31, 0.4)                   # uses at order 2; and L = 126
    above = series(Val(3), Val(8), Float64, 7, 0.4)                  # L = 165
    same(f, g, types, gtypes = types) = lowered(f, types).code == lowered(g, gtypes).code
    for (a, K) in ((small, 2), (big, 4))
        A = typeof(a)
        @test same(P.mul_truncated, P._mul_unrolled, (A, A, Val{false}, Val{2}))
        @test same(P._addsub, P._addsub_unrolled, (A, A, Val{1}))
        @test same(P.compose, P._compose_unrolled, (A, NTuple{K + 1, Float64}, Float64))
        @test same(Base.literal_pow, P._literal_pow_unrolled, (typeof(^), A, Val{3}))
        @test same(P.coefficient, P._coefficient_unrolled, (A, NTuple{5, Int}))
        @test same(P._derivatives, P._derivatives_unrolled, (A, NTuple{5, Float64}))
        @test same(P._scale_from, P._scale_from_unrolled, (A, Float64, Val{2}))
        @test same(P._div, P._div_unrolled, (A, A))
        U = Tuple{A, A, A, A, A}
        @test same(substitute, P._substitute_unrolled, (A, U))
        @test lowered(substitute, (A, U)).inlining == 1                           # inline
    end
    @test same(P._recurrence, P._recurrence_unrolled, (typeof(+), Float64, Val{125}))
    # Above the threshold: the same call as the loop form, so exact.
    @test (above * above).c == P._mul_loop(above, above, false, 8).c
    @test (above + above).c == P._addsub_loop(above, above, Val(1)).c
    r = inv(value(above))
    d = ntuple(k -> isodd(k) ? r : -r, Val(9))
    @test P.compose(above, d, r).c == P._compose_loop(above, d, r).c
    @test inv(above).c == P._div_loop(one(above), above).c
    @test (above / (above + 1)).c == P._div_loop(above, above + 1).c
    @test (above^3).c == P._literal_pow_loop(above, Val(3)).c
    E = series(Val(2), Val(8), Float64, 3, 0.1)
    u = (above, above)
    @test substitute(E, u).c == P._substitute_loop(E, u).c
    @test lowered(substitute, (typeof(E), typeof(u))).inlining == 2                 # noinline
    g = (q, k) -> q / k
    @test P._recurrence(g, 1.0, Val(126)) == P._loop_recurrence(g, 1.0, Val(127))
end

# Oracle: the unrolled division, exact.
@testitem "loop division equals the unrolled division" setup = [LoopCases] begin
    using PureTaylor
    using .LoopCases: P, series, CASES
    for T in (Float64, Float32), (N, K, m1, m2) in CASES
        a = series(Val(N), Val(K), T, m1, 0.4); b = series(Val(N), Val(K), T, m2, -0.3)
        x, y = P._div_loop(a, b), P._div_unrolled(a, b)
        @test typeof(x) == typeof(y)
        @test x.c == y.c
    end
end
