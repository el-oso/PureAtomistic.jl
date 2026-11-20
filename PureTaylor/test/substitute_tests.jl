@testmodule ChangeOfVariables begin
    using PureTaylor

    g(p, q) = cbrt(p) * exp(q) + log1p(p * q^2)

    # p = a + b and q = (a - b)/(a + b) as Taylor numbers of order 3 in (a, b).
    function pq(a0::T, b0::T) where {T}
        a = variable(Val(2), Val(3), Val(1), a0); b = variable(Val(2), Val(3), Val(2), b0)
        return a + b, (a - b) / (a + b)
    end
end

# Oracle: g computed directly in (a, b) by the Taylor type. Rounding bound c·eps(T)·N with c = 2 (one
# rounding per multiply and per add) and N = 300: on each side about ten series operations (cbrt, exp,
# log1p, the square, three products or sums, and p and q themselves), each at most K = 3 Horner steps
# over sums of at most L = 10 products, plus on the substituted side the sum over L = 10 coefficients
# of E; relative to the largest reference derivative of the same total degree. Largest measured:
# 0.97·eps·scale (Float64), 1.5·eps·scale (Float32).
@testitem "substitute changes variables" setup = [ChangeOfVariables] begin
    using .ChangeOfVariables: g, pq
    using PureTaylor: multi_indices
    for T in (Float64, Float32)
        a0, b0 = T(0.9), T(0.4)
        p, q = pq(a0, b0)
        E = g(variable(Val(2), Val(3), Val(1), value(p)), variable(Val(2), Val(3), Val(2), value(q)))
        S = substitute(E, (p, q))
        D = g(p, q)
        @test S isa typeof(D)
        got = derivatives(S, (one(T), one(T)))
        ref = derivatives(D, (one(T), one(T)))
        idx = multi_indices(Val(2), Val(3))
        for j in eachindex(idx)
            scale = maximum(abs(ref[i]) for i in eachindex(idx) if sum(idx[i]) == sum(idx[j]))
            @test abs(got[j] - ref[j]) <= 2 * eps(T) * 300 * scale
        end
    end
end

# A variable given as a plain number does not change: its increment is zero. Same oracle and bound.
# Largest measured: 0.07·eps·scale.
@testitem "substitute holds a plain-number variable fixed" setup = [ChangeOfVariables] begin
    using .ChangeOfVariables: g, pq
    p, q = pq(0.9, 0.4)
    E = g(variable(Val(2), Val(3), Val(1), value(p)), variable(Val(2), Val(3), Val(2), value(q)))
    S = substitute(E, (p, value(q)))
    D = g(p, value(q))
    @test S isa typeof(D)
    scale = maximum(abs, D.c)
    @test all(abs(S.c[j] - D.c[j]) <= 2 * eps(Float64) * 300 * scale for j in eachindex(D.c))
end

# The variables of E are the increments of the seeds' u: with scaled seeds x + s·u, dividing the new
# variables by s gives E's variable. Polynomial E, so the result is exact in Rational{BigInt}.
@testitem "substitute of a polynomial (exact)" begin
    R = Rational{BigInt}
    x = variable(Val(2), Val(3), Val(1), R(2), R(1, 2)); y = variable(Val(2), Val(3), Val(2), R(3))
    E = x^2 * y - 3x + y^3
    a = variable(Val(3), Val(3), Val(1), R(1)); b = variable(Val(3), Val(3), Val(3), R(5))
    u = (a * b / R(1, 2), a - b)
    S = substitute(E, u)
    # E in x and y, with x - 2 = a·b - 5 and y - 3 = a - b + 4.
    X = a * b - 5 + 2; Y = a - b + 4 + 3
    @test S.c == (X^2 * Y - 3X + Y^3).c                            # exact: Rational{BigInt}
    @test S isa Taylor{3,3,5,R}
end

@testitem "substitute at order 0 gives the value" begin
    E = exp(constant(Val(2), Val(0), 1.0))
    S = substitute(E, (variable(Val(3), Val(0), Val(1), 2.0), 3.0))
    @test S isa Taylor{3,0,0,Float64,1}
    @test value(S) == exp(1.0)                                      # exact: copied
end

@testitem "substitute rejects u without a common number of new variables" begin
    E = variable(Val(2), Val(2), Val(1), 1.0)
    @test_throws "PureTaylor: substitute needs at least one Taylor number in u, all over the same number of variables; got ()" substitute(E, (1.0, 2.0))
    @test_throws "PureTaylor: substitute needs at least one Taylor number in u, all over the same number of variables; got (3, 2)" substitute(
        E, (variable(Val(3), Val(2), Val(1), 1.0), variable(Val(2), Val(2), Val(1), 1.0))
    )
end
