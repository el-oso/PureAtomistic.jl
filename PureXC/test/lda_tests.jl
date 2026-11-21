# Reference rows of the specs' "Reference values" tables (Libxc 7.0.0, Float64): the input, zk and
# vrho. Restricted rows give n; polarized rows give (n_α, n_β) and vrho = (vrho_α, vrho_β).
@testmodule LDAReference begin
    using PureXC

    const LDA_X = (
        (1.0e-6, -0.0073855876638202235, -0.009847450218426965),
        (0.01, -0.15911766269205826, -0.212156883589411),
        (1.0, -0.7385587663820223, -0.9847450218426964),
        (100.0, -3.4280861230056243, -4.570781497340832),
        ((0.3, 0.1), -0.575171388289353, (-0.8305661184154147, -0.5758823822969721)),
        ((1.0, 0.001), -0.929689099822912, (-1.2407009817987997, -0.1240700981798798)),
        ((0.02, 0.02), -0.2525835451440944, (-0.33677806019212586, -0.33677806019212586)),
    )
    const LDA_C_PW = (
        (1.0e-6, -0.00477162315902391, -0.0060858858656315734),
        (0.01, -0.03769770328922326, -0.04387606205358233),
        (1.0, -0.07120031359839032, -0.0794572203196884),
        (100.0, -0.1125111809948515, -0.12205070626815523),
        ((0.3, 0.1), -0.05824958052852496, (-0.05385591326647189, -0.10055877615182211)),
        ((1.0, 0.001), -0.03768321214182925, (-0.04168267033646179, -0.2870635770635308)),
        ((0.02, 0.02), -0.04675320448391642, (-0.0536287333531124, -0.0536287333531124)),
    )
    const LDA_C_PW_MOD = (
        (1.0e-6, -0.004771623035336743, -0.006085885635601924),
        (0.01, -0.03769764282445714, -0.04387597615794007),
        (1.0, -0.07120005886619186, -0.07945690779111174),
        (100.0, -0.11251060623211576, -0.12205005230913385),
        ((0.3, 0.1), -0.05824939474999474, (-0.05385574005674182, -0.10055836084415391)),
        ((1.0, 0.001), -0.037683529511408645, (-0.04168306420165874, -0.2870566027090974)),
        ((0.02, 0.02), -0.046753101645079384, (-0.05362859476471246, -0.05362859476471246)),
    )

    const FUNCTIONALS = ((lda_x, LDA_X), (lda_c_pw, LDA_C_PW), (lda_c_pw_mod, LDA_C_PW_MOD))

    total(ρ::Real) = ρ
    total(ρ::Tuple) = ρ[1] + ρ[2]

    # The relative difference of every entry of `got` from `want`, entry by entry.
    reldiff(got::Real, want::Real) = abs(got - want) / abs(want)
    reldiff(got::Tuple, want::Tuple) = maximum(map(reldiff, got, want))

    # The specs' grid: n log-spaced over [1e-6, 1e3]; polarized at ζ ∈ {0, 0.25, −0.6, 0.95} with
    # n_α = n(1 + ζ)/2 and n_β = n(1 − ζ)/2 rounded to Float64.
    const GRID_N = exp10.(range(-6, 3; length = 19))
    const GRID_ζ = (0.0, 0.25, -0.6, 0.95)
    const GRID_POLARIZED = [(n * (1 + ζ) / 2, n * (1 - ζ) / 2) for n in GRID_N for ζ in GRID_ζ]

    # The largest error of the entries of one output, relative to its largest BigFloat entry.
    function scaled_error(got, want)
        scale = maximum(abs, want)
        iszero(scale) && return all(iszero, got) ? 0.0 : Inf
        return Float64(maximum(abs.(big.(got) .- want)) / scale)
    end
end

# Oracle: the specs' Libxc 7.0.0 reference rows. Tolerance (a): 1e-6 relative, the user's floor;
# each spec measured the difference of its equations in BigFloat from these rows at ≤ 9e-15.
@testitem "reference values of the LDA functionals" setup = [LDAReference] begin
    using .LDAReference: FUNCTIONALS, total, reldiff
    for (build, rows) in FUNCTIONALS
        f = build()
        for (ρ, zk, vrho) in rows
            r = xc_derivatives(f, Val(1), ρ)
            @test reldiff(r.e / total(ρ), zk) <= 1.0e-6
            @test reldiff(r.vrho, vrho) <= 1.0e-6
            # Exact: the value is formed by the same operations at every order.
            @test energy_density(f, ρ) == r.e
        end
    end
end

# Oracle: the same functional built for BigFloat (256 bits), on the same Float64 inputs converted
# exactly. Bound (b) c·eps·N, relative to the largest BigFloat entry of the same output: c = 2 (one
# rounding per multiply and per add, eps/2 each, doubled for the fast_* kernels, which are within
# 2 ulps) and N = 1500. N counts the roundings that reach one output: about 40 series operations
# (the three fits (E1) of 8 operations each, f(ζ) (E3), the combination (E4), the formation of n
# and ζ and the change of variables of the polarized case), each a Horner sum of at most K + 1 = 4
# steps over sums of at most L = 10 products, rounded once each: 40·(4 + 10)·2 ≈ 1100, rounded up.
# lda_x takes the same N, which bounds its 4 operations by a wide margin.
@testitem "LDA functionals agree with BigFloat" setup = [LDAReference] begin
    using .LDAReference: FUNCTIONALS, GRID_N, GRID_POLARIZED, scaled_error
    tol = 2 * eps(Float64) * 1500
    worst = Ref(0.0)
    for (build, _) in FUNCTIONALS
        f, fbig = build(), build(BigFloat)
        for ρ in Iterators.flatten((GRID_N, GRID_POLARIZED))
            got = xc_derivatives(f, Val(3), ρ)
            want = xc_derivatives(fbig, Val(3), map(big, ρ))
            @test keys(got) == keys(want)
            for name in keys(want)
                err = scaled_error(getfield(got, name), getfield(want, name))
                worst[] = max(worst[], err)
                @test err <= tol
            end
        end
    end
    # Largest measured: 44·eps (lda_c_pw, v3rho3, polarized).
    @test 0 < worst[] <= tol
end

# Spec lda_c_pw, section 5: f''(0) = 1.709921 as printed in [PW92]. Spec lda_c_pw_mod, section 5:
# the exact value 4/(9(2^(1/3) − 1)), which is 1.7099209341613657 in Float64.
@testitem "PW92 constants" begin
    @test lda_c_pw().fz20 == 1.709921
    @test lda_c_pw(Float32).fz20 == 1.709921f0
    @test lda_c_pw_mod().fz20 == 1.7099209341613657
    @test lda_c_pw_mod(Float32).fz20 == Float32(4 / (9 * (cbrt(big(2)) - 1)))
    @test lda_c_pw_mod(BigFloat).fz20 == 4 / (9 * (cbrt(big(2)) - 1))
    @test lda_c_pw().params == (
        (0.031091, 0.2137, 7.5957, 3.5876, 1.6382, 0.49294),
        (0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
        (0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671),
    )
    @test map(p -> p[1], lda_c_pw_mod().params) == (0.0310907, 0.01554535, 0.0168869)
    @test map(p -> p[2:end], lda_c_pw_mod().params) == map(p -> p[2:end], lda_c_pw().params)
    for build in (lda_x, lda_c_pw, lda_c_pw_mod)
        t = build().thresholds
        @test t.density == 1.0e-15
        @test t.zeta == 2.220446049250313e-16
        @test build(Float32).thresholds.zeta == eps(Float32)
    end
    @test kind(lda_x()) == :exchange
    @test kind(lda_c_pw()) == :correlation
    @test family(lda_c_pw_mod()) == :lda
end

# Spec section 7, "Libxc behavior", rule 1: the total density below n_thr is cut off (strict).
@testitem "LDA cut-off at the density threshold" begin
    for f in (lda_x(), lda_c_pw(), lda_c_pw_mod())
        for ρ in (0.999e-15, (0.999e-15, 0.0), (0.4995e-15, 0.4995e-15))
            r = xc_derivatives(f, Val(3), ρ)
            @test all(x -> all(iszero, x), values(r))
        end
    end
    # n = 1.0e-15 is not cut off. Oracle: the spec's printed zk, 5 and 8 significant digits.
    # Tolerance (a): half a unit in the last printed digit, relative.
    @test abs(xc_derivatives(lda_c_pw(), Val(0), 1.0e-15).e / 1.0e-15 + 6.896e-6) <= 0.5e-9
    @test abs(xc_derivatives(lda_c_pw_mod(), Val(0), 1.0e-15).e / 1.0e-15 + 6.8960262e-6) <= 0.5e-13
    # Polarized, the test is on n_α + n_β.
    for f in (lda_c_pw(), lda_c_pw_mod())
        @test xc_derivatives(f, Val(0), (1.0e-15, 0.0)).e < 0
    end
end

# Spec lda_x, section 7: a channel with n_σ ≤ n_thr contributes zero (rule 2), so the restricted
# energy is zero for n ≤ 2 n_thr (rule 3).
@testitem "lda_x per-channel rule" begin
    f = lda_x()
    for n in (1.0, 1.001, 1.999, 2.0) .* 1.0e-15
        @test all(iszero, values(xc_derivatives(f, Val(3), n)))
    end
    # Oracle: the spec's zk = −9.3068e-6 at n = 2.001e-15. Tolerance (a): half a unit in the last
    # printed digit.
    @test abs(xc_derivatives(f, Val(0), 2.001e-15).e / 2.001e-15 + 9.3068e-6) <= 0.5e-10
    @test all(x -> all(iszero, x), values(xc_derivatives(f, Val(3), (0.9995e-15, 0.9995e-15))))
    @test xc_derivatives(f, Val(0), (1.0005e-15, 1.0005e-15)).e < 0
    # The β channel contributes an exact zero to every output at n_β ≤ n_thr.
    ref = xc_derivatives(f, Val(3), (0.01, 0.0))
    for ρβ in (0.0, 0.5e-15, 1.0e-15)
        r = xc_derivatives(f, Val(3), (0.01, ρβ))
        # Oracle: the spec's zk. Tolerance (a): 1e-6 relative, the user's floor.
        @test abs(r.e / 0.01 + 0.20047569263577633) <= 1.0e-6 * 0.20047569263577633
        @test iszero(r.vrho[2])
        @test iszero(r.v2rho2[2]) && iszero(r.v2rho2[3])
        @test all(iszero, r.v3rho3[2:4])
        # Oracle: the result at n_β = 0. Bound (b): the α seed scale is n_α + n_β, which differs
        # by at most 1e-13 relative between these points; each output is the same expression
        # scaled by s^k·(1/s)^k, k ≤ 3, with at most 8 roundings of eps/2: 4·eps relative.
        for name in keys(ref)
            @test all(abs.(getfield(r, name) .- getfield(ref, name)) .<= 4 * eps() .* abs.(getfield(ref, name)))
        end
    end
    # Oracle: the spec's vrho_β = −1.42e-5 at n_β = 1.5e-15. Tolerance (a): half a unit in the
    # last printed digit.
    @test abs(xc_derivatives(f, Val(1), (0.01, 1.5e-15)).vrho[2] + 1.42e-5) <= 0.5e-7
end

# Spec lda_c_pw and lda_c_pw_mod, section 7, rule 2: each channel is raised to n_thr before r_s and
# ζ are formed, and the derivative with respect to a raised channel is the one at n_thr.
@testitem "PW92 channels raised to the density threshold" begin
    # Bound (b) for the same derivatives at different raw inputs: the raised densities are the same,
    # so the computation differs only in the seed scales s, which follow the raw total density.
    # Each output is formed with s^k·(1/s)^k, k ≤ 2, through the operations counted in "LDA
    # functionals agree with BigFloat": 2·eps·1500 relative to the largest entry of the output.
    close(g, w) = all(abs.(g .- w) .<= 2 * eps() * 1500 .* maximum(abs, w))
    # Oracle: the spec's values. Tolerance (a): 1e-6 relative, the user's floor.
    for (f, zk, vβ, v2ββ) in (
            (lda_c_pw(), -0.020296494557675415, -0.15101284284991515, 1.740992732755484e9),
            (lda_c_pw_mod(), -0.020296576673299066, -0.15101107190123725, nothing),
        )
        ref = xc_derivatives(f, Val(2), (0.01, 0.0))
        for ρβ in (0.0, 0.5e-15, 1.0e-15)
            n = 0.01 + ρβ
            r = xc_derivatives(f, Val(2), (0.01, ρβ))
            @test abs(r.e / n - zk) <= 1.0e-6 * abs(zk)
            @test abs(r.vrho[2] - vβ) <= 1.0e-6 * abs(vβ)
            isnothing(v2ββ) || @test abs(r.v2rho2[3] - v2ββ) <= 1.0e-6 * abs(v2ββ)
            # e = n·ε_c with the caller's n and the common ε_c: bound (b) of the derivatives,
            # plus one rounding of the product and one of the quotient.
            @test abs(r.e / n - ref.e / 0.01) <= (2 * eps() * 1500 + 2 * eps()) * abs(zk)
            for name in keys(ref)[2:end]
                @test close(getfield(r, name), getfield(ref, name))
            end
        end
    end
    # Both channels raised to n_thr. Oracle: the spec's lda_c_pw values. Tolerance (a): 1e-6.
    for (f, zk, v) in ((lda_c_pw(), -8.674368563248956e-6, -1.1544342326421484e-5), (lda_c_pw_mod(), nothing, nothing))
        ref = xc_derivatives(f, Val(2), (1.0e-15, 1.0e-15))
        for x in (0.5005e-15, 0.9995e-15, 1.0e-15)
            r = xc_derivatives(f, Val(2), (x, x))
            if !isnothing(zk)
                @test abs(r.e / 2x - zk) <= 1.0e-6 * abs(zk)
                @test all(abs.(r.vrho .- v) .<= 1.0e-6 * abs(v))
            end
            # e is the caller's n times the common ε_c; bound (b) as above.
            @test abs(r.e / 2x - ref.e / 2.0e-15) <= (2 * eps() * 1500 + 2 * eps()) * abs(ref.e / 2.0e-15)
            for name in keys(ref)[2:end]
                @test close(getfield(r, name), getfield(ref, name))
            end
        end
    end
end

# Spec lda_c_pw, section 7, rule 3: a factor (1 ± ζ)^(4/3) with 1 ± ζ ≤ ε_ζ is the constant
# ε_ζ^(4/3), which has no derivative. At n_α = 1000 and n_β = 1e-14, 1 − ζ rounds below ε_ζ, so
# v2rho2_ββ is small (Libxc 2.6e-3); at n_β = 1e-13 the factor is kept and v2rho2_ββ is large
# (Libxc 4.6e6). Only the qualitative effect is checked, as in the spec.
@testitem "PW92 ζ-factor replacement" begin
    for f in (lda_c_pw(), lda_c_pw_mod())
        small = xc_derivatives(f, Val(2), (1000.0, 1.0e-14)).v2rho2[3]
        large = xc_derivatives(f, Val(2), (1000.0, 1.0e-13)).v2rho2[3]
        @test isfinite(small) && abs(small) < 1
        @test large > 1.0e6
    end
end

# Review focus 1: a fully polarized point gives finite values at every order, equal to the rule's
# floor behavior.
@testitem "fully polarized LDA points" begin
    for f in (lda_x(), lda_c_pw(), lda_c_pw_mod()), K in 0:4, ρ in ((0.3, 0.0), (0.0, 0.3), (1.0e-6, 0.0))
        r = xc_derivatives(f, Val(K), ρ)
        @test all(x -> all(isfinite, x), values(r))
        # Oracle: the channel at n_thr. Bound (b): the energy e = n·ε_c uses the caller's n, which
        # changes by n_thr/n ≤ 1e-9; the derivatives are those at the same raised densities, so
        # they differ by the seed scales alone, the rounding bound 2·eps·1500.
        floor = map(x -> iszero(x) ? 1.0e-15 : x, ρ)
        w = xc_derivatives(f, Val(K), floor)
        for name in keys(r)
            g, h = getfield(r, name), getfield(w, name)
            @test all(abs.(g .- h) .<= (1.0e-9 + 2 * eps() * 1500) .* maximum(abs, h))
        end
    end
end

# Review focus 2: ρ = 0 and ρ = −0.0 give exact zeros at every order.
@testitem "zero densities give exact zeros" begin
    for f in (lda_x(), lda_c_pw(), lda_c_pw_mod(), lda_x(Float32), lda_c_pw(Float32)), K in 0:4
        for ρ in (0.0, -0.0, (0.0, 0.0), (-0.0, 0.0), (-0.0, -0.0), (0.0f0, -0.0f0))
            r = xc_derivatives(f, Val(K), ρ)
            @test all(x -> all(iszero, x), values(r))
        end
    end
end

@testitem "LDA result types" begin
    for build in (lda_x, lda_c_pw, lda_c_pw_mod)
        r = @inferred xc_derivatives(build(Float32), Val(3), (0.3f0, 0.1f0))
        @test all(x -> x isa Tuple{Vararg{Float32}}, values(r)[2:end]) && r.e isa Float32
        @test (@inferred xc_derivatives(build(Float32), Val(2), 0.3f0)).v2rho2 isa Float32
        @test (@inferred xc_derivatives(build(), Val(2), 0.3f0)).v2rho2 isa Float64
        @test (@inferred xc_derivatives(build(Float32), Val(1), (0.3, 0.1))).vrho isa NTuple{2, Float64}
    end
end

@testitem "LDA functionals allocate nothing" begin
    using StrictModeTest
    for T in (Float32, Float64), build in (lda_x, lda_c_pw, lda_c_pw_mod)
        f = build(T)
        ρ, ρp = T(0.3), (T(0.3), T(0.1))
        xc_derivatives(f, Val(2), ρ)
        xc_derivatives(f, Val(2), ρp)
        # @test_noalloc throws a StrictViolation unless AllocCheck proves the call allocation-free.
        @test (@test_noalloc xc_derivatives(f, Val(2), ρ)) isa NamedTuple
        @test (@test_noalloc xc_derivatives(f, Val(2), ρp)) isa NamedTuple
    end
end
