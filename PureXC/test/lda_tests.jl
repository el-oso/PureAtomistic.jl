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

    # The functional `f` with its parameters and thresholds converted exactly to BigFloat, so that
    # it is the same function as `f`. For a Float32 functional this keeps ε_ζ = eps(Float32):
    # `build(BigFloat)` has ε_ζ = eps(Float64), so its ζ-factor rule would replace a factor at
    # other points than `f` does.
    big_thresholds(t) = PureXC.Thresholds{BigFloat}(big(t.density), big(t.sigma), big(t.zeta), big(t.sigma_floor))
    exact_big(f::PureXC.LDAExchange) = PureXC.LDAExchange{BigFloat}(big_thresholds(f.thresholds))
    exact_big(f::PureXC.PW92Correlation) = PureXC.PW92Correlation{BigFloat}(
        map(p -> big.(p), f.params), big(f.fz20), big_thresholds(f.thresholds)
    )

    # The largest error of the outputs `got`, in the type T, against `want`, in units of eps(T): each
    # output relative to max(its largest entry of magnitude at most floatmax(T), floatmin(T)). An
    # entry whose exact value exceeds floatmax(T) must be the infinity of its sign, and otherwise
    # the error is Inf.
    function error_eps(got, want, T)
        worst = 0.0
        for name in keys(want)
            g, w = [getfield(got, name)...], [getfield(want, name)...]
            fits = abs.(w) .<= floatmax(T)
            all(@. ifelse(fits, isfinite(g), isinf(g) & (sign(g) == sign(w)))) || return Inf
            any(fits) || continue
            scale = max(maximum(abs, w[fits]), big(floatmin(T)))
            worst = max(worst, Float64(maximum(abs.(big.(g[fits]) .- w[fits])) / scale / eps(T)))
        end
        return worst
    end

    # κ = 2/(1 − |ζ|), the total of the channels raised to n_thr over the smaller one: 1 ± ζ is
    # formed from ζ (spec lda_c_pw, section 8), so its relative error is of order eps·κ.
    function zeta_conditioning(f, ρ)
        a, b = map(x -> big(max(x, f.thresholds.density)), ρ)
        return Float64((a + b) / min(a, b))
    end

    # The bound, in eps(T) relative to the largest entry of each output, of the outputs at order K
    # of the functional g at the input x, against `exact_big`:
    # - Float32 above `_float32_max_order`, evaluated in Float64 and rounded: 50, bound (c), measured
    #   at most 0.5.
    # - where nothing is ill-conditioned (lda_x; PW92 restricted, at the symmetric split, or where
    #   the factor (1 ± ζ)^(4/3) of the smaller channel is the constant of rule 3): at K ≤ 3, 2100,
    #   bound (b) of "LDA functionals agree with BigFloat"; at K ≥ 4, 1000, bound (c), measured at
    #   most 21, 29, 37, 51, 52, 59 and 66 at K = 4 to 10.
    # - PW92 where that factor is kept: the same plus 8κ, κ = 2/(1 − |ζ|), the conditioning of
    #   1 ± ζ formed from ζ (spec lda_c_pw, section 8), bound (c), measured at most 3.1·κ. Where κ
    #   is large this admits errors of order 1 relative: that is the documented conditioning.
    function point_bound(g, x, T, K)
        T === Float32 && K > PureXC._float32_max_order(typeof(g)) && return 50.0
        base = K <= 3 ? 2100.0 : 1000.0
        g isa PureXC.PW92Correlation && x isa Tuple || return base
        t = g.thresholds.density
        a, b = max(x[1], t), max(x[2], t)
        ζ = (a - b) / (a + b)
        κ = zeta_conditioning(g, x)
        return min(1 + ζ, 1 - ζ) < g.thresholds.zeta || κ <= 2 ? base : base + 8κ
    end

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
# exactly. Bound (b) c·eps·N, relative to the largest BigFloat entry of the same output. N counts
# the roundings of eps/2 that reach one output coefficient on the longest path, polarized PW92:
# - 73 series operations, each a coefficient that is a sum of at most L = 10 products, so at most
#   20 roundings: ζ and n_r (5: a − b, a + b, the inverse's product, the product, n_r/s), r_s (2),
#   √r_s (1), the three fits (E1) (14 each: 7 for the Horner sum, 2A·q, 1 + α₁r_s, two scalings,
#   the product, and the series parts of the inverse and the log1p counted below), f(ζ) (E3) (9),
#   ζ⁴ and 1 − ζ⁴ (3), the combination (E4) (8), the change of variables, the product with n_r
#   and the scaling of `derivatives` (3), 5 + 2 + 1 + 42 + 9 + 3 + 8 + 3 = 73: 73·20 = 1460;
# - 10 univariate functions (fast_invcbrt, fast_sqrt, three inverses and three fast_log1p in
#   (E1), two fast_cbrt in (E3); the inverse for ζ is in the 73), each K = 3 Horner steps of a
#   series product: 10·3·20 = 600;
# - the scalar kernels of those 10 functions, within 2 ulps = 4 roundings each: 40.
# N = 1460 + 600 + 40 = 2100. c = 1: eps/2 per rounding, doubled for the propagation of an error
# through the later operations to first order. lda_x takes the same bound, which covers its 5 operations widely.
# Largest measured: 44·eps relative to the largest entry (lda_c_pw, v3rho3, polarized); entry by
# entry, each against itself, 1385·eps.
@testitem "LDA functionals agree with BigFloat" setup = [LDAReference] begin
    using .LDAReference: FUNCTIONALS, GRID_N, GRID_POLARIZED, scaled_error
    tol = 2100 * eps(Float64)
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
        # Oracle: the result at n_β = 0. Exact: each channel is seeded at its own scale, n_α and
        # n_thr at all three points, so the α channel is the same computation and the β channel
        # contributes exact zeros.
        @test r == ref
    end
    # Oracle: the spec's vrho_β = −1.42e-5 at n_β = 1.5e-15. Tolerance (a): half a unit in the
    # last printed digit.
    @test abs(xc_derivatives(f, Val(1), (0.01, 1.5e-15)).vrho[2] + 1.42e-5) <= 0.5e-7
end

# Spec lda_c_pw and lda_c_pw_mod, section 7, rule 2: each channel is raised to n_thr before r_s and
# ζ are formed, and the derivative with respect to a raised channel is the one at n_thr.
@testitem "PW92 channels raised to the density threshold" begin
    # The derivatives at different raw inputs are exact: the raised densities are the same, and both
    # are seeded at max(n_r, m) of the raised total, so every series is the same computation.
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
            # e = n·ε_c with the caller's n and the same ε_c. Bound (b): e/n carries one rounding
            # of the product and one of the quotient on each side, 4 roundings of eps/2.
            @test abs(r.e / n - ref.e / 0.01) <= 2 * eps() * abs(zk)
            for name in keys(ref)[2:end]
                @test getfield(r, name) == getfield(ref, name)
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
            # e is the caller's n times the same ε_c; bound (b) as above.
            @test abs(r.e / 2x - ref.e / 2.0e-15) <= 2 * eps() * abs(ref.e / 2.0e-15)
            for name in keys(ref)[2:end]
                @test getfield(r, name) == getfield(ref, name)
            end
        end
    end
end

# Spec lda_c_pw, section 7, rule 3: a factor (1 ± ζ)^(4/3) with the rounded 1 ± ζ < ε_ζ is the
# constant ε_ζ^(4/3), which has no derivative. At n_α = 1000 and n_β = 1e-14, 1 − ζ rounds below
# ε_ζ, so v2rho2_ββ is small (Libxc 2.6e-3); at n_β = 1e-13 the factor is kept and v2rho2_ββ is
# large (Libxc 4.6e6). Only the qualitative effect is checked, as in the spec.
@testitem "PW92 ζ-factor replacement" begin
    for f in (lda_c_pw(), lda_c_pw_mod())
        small = xc_derivatives(f, Val(2), (1000.0, 1.0e-14)).v2rho2[3]
        large = xc_derivatives(f, Val(2), (1000.0, 1.0e-13)).v2rho2[3]
        @test isfinite(small) && abs(small) < 1
        @test large > 1.0e6
    end
end

# Rule 3 is strict on the rounded 1 − ζ, with ζ = (n_α − n_β)/n as formed in `_energy_density`: at
# (1000, 8e-14) the rounded 1 − ζ equals ε_ζ and the factor is kept, so v2rho2_ββ is large; at
# (1000, 5e-14) it rounds to 0 and the factor is replaced, so v2rho2_ββ is small. Only the
# qualitative effect is checked, as above.
@testitem "PW92 ζ-factor at the rounded 1 − ζ = ε_ζ" begin
    for f in (lda_c_pw(), lda_c_pw_mod())
        a = 1000.0
        ζ(b) = (a - b) / (a + b)
        @test 1 - ζ(8.0e-14) == f.thresholds.zeta
        @test iszero(1 - ζ(5.0e-14))
        @test xc_derivatives(f, Val(2), (a, 8.0e-14)).v2rho2[3] > 1.0e6
        small = xc_derivatives(f, Val(2), (a, 5.0e-14)).v2rho2[3]
        @test isfinite(small) && abs(small) < 1
    end
end

# Review focus 1: a fully polarized point gives finite values at every order, equal to the rule's
# floor behavior.
@testitem "fully polarized LDA points" begin
    for f in (lda_x(), lda_c_pw(), lda_c_pw_mod()), K in 0:4, ρ in ((0.3, 0.0), (0.0, 0.3), (1.0e-6, 0.0))
        r = xc_derivatives(f, Val(K), ρ)
        @test all(x -> all(isfinite, x), values(r))
        # Oracle: the channel at n_thr. The derivatives are exact: they are those at the same raised
        # densities with the same seed scales. Bound (b) for the energy e = n·ε_c, which uses the
        # caller's n: it changes by n_thr/n ≤ 1e-9, plus two roundings.
        floor = map(x -> iszero(x) ? 1.0e-15 : x, ρ)
        w = xc_derivatives(f, Val(K), floor)
        @test abs(r.e - w.e) <= (1.0e-9 + 2 * eps()) * abs(w.e)
        for name in keys(r)[2:end]
            @test getfield(r, name) == getfield(w, name)
        end
    end
end

# Review focus 2: ρ = 0 and ρ = −0.0 give exact zeros at every order.
@testitem "zero densities give exact zeros" begin
    for f in (lda_x(), lda_c_pw(), lda_c_pw_mod(), lda_x(Float32), lda_c_pw(Float32), lda_c_pw_mod(Float32)), K in 0:4
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
    # Float32 at the last order evaluated in Float32, `_float32_max_order`, and the first evaluated
    # in Float64.
    for build in (lda_x, lda_c_pw, lda_c_pw_mod), ρ in (0.3f0, (0.3f0, 0.1f0))
        f = build(Float32)
        k32 = PureXC._float32_max_order(typeof(f))
        @test k32 == (build === lda_x ? 10 : 2)
        for K in (k32, k32 + 1)
            xc_derivatives(f, Val(K), ρ)
            @test (@test_noalloc xc_derivatives(f, Val(K), ρ)) isa NamedTuple
        end
    end
end

# _pw92_eps promotes the value types of r_s and ζ, so a plain ζ of another type is accepted.
@testitem "_pw92_eps with a plain ζ" begin
    using PureTaylor
    pw = lda_c_pw()
    rs = variable(Val(1), Val(2), Val(1), 2.0)
    # Exact: ζ = 0 and ζ = 0.0 give the same operations on the same values.
    @test PureXC._pw92_eps(pw, rs, 0) == PureXC._pw92_eps(pw, rs, 0.0)
    @test PureXC._pw92_eps(pw, 2.0, 0) == PureXC._pw92_eps(pw, 2.0, 0.0)
    @test PureXC._pw92_eps(pw, 2.0f0, 0.25) isa Float64
end

# The domain limit n_max of an LDA. Oracle: the same functional in BigFloat. Bound (b) of "LDA
# functionals agree with BigFloat", which covers K ≤ 3: 2100·eps(T) relative to the largest entry of
# each output, with floatmin(T) as the smallest scale, so that an output whose exact entries all
# underflow may be zero. Orders 4 to 10 at n_max are checked by "LDA functionals at orders 0 to 10".
@testitem "LDA functionals at the domain limit" begin
    function within(f, fbig, K, ρ, T)
        r = xc_derivatives(f, Val(K), ρ)
        b = xc_derivatives(fbig, Val(K), map(big, ρ))
        return all(keys(r)) do name
            g, w = getfield(r, name), getfield(b, name)
            all(isfinite, g) && maximum(abs.(big.(g) .- w)) <= 2100 * eps(T) * max(maximum(abs, w), big(floatmin(T)))
        end
    end
    # Exact: the measured limits, powers of two.
    measured = Dict(Float64 => (767, 767, 767, 767, 767, 767, 767, 731, 679, 627, 575), Float32 => ntuple(_ -> 95, 11))
    for T in (Float32, Float64), K in 0:10
        n = PureXC._lda_density_max(T, Val(K))
        @test n == ldexp(one(T), measured[T][K + 1])
        for build in (lda_x, lda_c_pw, lda_c_pw_mod)
            f, fbig = build(T), build(BigFloat)
            for ρ in (n, (n, zero(T)), (zero(T), n), (n / 2, n / 2), (n / 4 * 3, n / 4))
                K <= 3 && @test within(f, fbig, K, ρ, T)
            end
            for ρ in (2n, nextfloat(n), (nextfloat(n), zero(T)), (zero(T), nextfloat(n)), (n / 2, n), (floatmax(T), floatmax(T)))
                @test_throws ArgumentError xc_derivatives(f, Val(K), ρ)
                @test_throws "xc: total density" xc_derivatives(f, Val(K), ρ)
            end
        end
    end
    # The limit is that of the type the functional evaluates in.
    @test_throws "xc: " xc_derivatives(lda_x(Float32), Val(1), 2.0f0^96)
    @test isfinite(xc_derivatives(lda_x(Float32), Val(1), 2.0^96).e)
    @test isfinite(xc_derivatives(lda_x(), Val(1), 2.0f0^96).e)
    @test_throws "xc: " energy_density(lda_x(), 2.0^768)
    @test isfinite(energy_density(lda_x(), 2.0^767))
end

# The seed scales: lda_x seeds each channel at its own value, PW92 both channels at the raised
# total. Exact: each scale is a maximum or a sum of the given numbers.
@testitem "seed scales of the LDA functionals" begin
    @test PureXC._seed_scales(lda_x(), Val(2), (0.25, 0.0)) == (0.25, 1.0e-15)
    @test PureXC._seed_scales(lda_x(Float32), Val(4), (1.0f0, 0.0f0)) == (1.0f0, 1.0f-15)
    # The scales do not depend on the order.
    @test PureXC._seed_scales(lda_x(), Val(12), (0.25, 0.0)) == (0.25, 1.0e-15)
    @test PureXC._seed_scales(lda_x(), Val(3), 0.0) == (1.0e-15,)
    for f in (lda_c_pw(), lda_c_pw_mod())
        @test PureXC._seed_scales(f, Val(2), (0.25, 0.0)) == (0.25 + 1.0e-15, 0.25 + 1.0e-15)
        @test PureXC._seed_scales(f, Val(2), (2.0e-15, 0.5e-15)) == (2.0e-15 + 1.0e-15, 2.0e-15 + 1.0e-15)
        @test PureXC._seed_scales(f, Val(1), 0.5) == (0.5,)
    end
    @test PureXC._seed_scales(lda_c_pw(Float32), Val(4), (0.0f0, -0.0f0)) == (1.0f-15 + 1.0f-15, 1.0f-15 + 1.0f-15)
end

# A nearly empty channel next to a large one: (2^p, ρβ) and (ρβ, 2^p) for 2^p up to n_max, at
# orders 1 to 3 (higher orders are in "LDA functionals at orders 0 to 10"). Oracle: the same
# functional in BigFloat, with its parameters and thresholds converted exactly (`exact_big`; it
# keeps the Float32 ε_ζ). Bounds: `point_bound`, with an infinity required exactly where the
# BigFloat entry exceeds floatmax(T). Measured where nothing is ill-conditioned: at most 3.5
# (lda_x) and 11 (PW92).
@testitem "LDA functionals at uneven splits" setup = [LDAReference] begin
    using .LDAReference: exact_big, error_eps, point_bound
    # Float32 lda_x: the series of the empty channel stays within range. Measured: 1.2 at both
    # points; the order-4 point takes bound (c), 100.
    f = lda_x(Float32)
    fb = exact_big(f)
    ρ = (2.0f0, 2.0f-15)
    @test error_eps(xc_derivatives(f, Val(3), ρ), xc_derivatives(fb, Val(3), big.(ρ)), Float32) <= 2100
    ρ = (1.0f0, 2.0f-15)
    r, b = xc_derivatives(f, Val(4), ρ), xc_derivatives(fb, Val(4), big.(ρ))
    # Exact: v4rho4_ββββ ≈ −7.2e38 is below −floatmax(Float32).
    @test b.v4rho4[5] < -floatmax(Float32) && r.v4rho4[5] == -Inf32
    @test error_eps(r, b, Float32) <= 100
    for T in (Float32, Float64), K in 1:3, build in (lda_x, lda_c_pw, lda_c_pw_mod)
        g = build(T)
        gb = exact_big(g)
        P = exponent(PureXC._lda_density_max(T, Val(K)))
        for p in unique([0:8:P; 5; P]), ρβ in (2.0e-15, 1.0e-6, 1.0), flip in (false, true)
            x = (ldexp(one(T), p), T(ρβ))
            x = flip ? reverse(x) : x
            err = error_eps(xc_derivatives(g, Val(K), x), xc_derivatives(gb, Val(K), big.(x)), T)
            @test err <= point_bound(g, x, T, K)
        end
    end
end

# Float32 at order 6. lda_x at (1, 2e-15), evaluated in Float32: the β channel is seeded at its own
# value, its series coefficient of order 6 is representable, and the output −1.8e69 is beyond
# floatmax, so it is −Inf. PW92 at (2^23, 1), evaluated in Float64 (order 6 is above its Float32
# order 2): the coefficient of order 6 is 6.2e36, 720 times which exceeds floatmax(Float32), and the
# output is 0.0128. Oracle: `exact_big`. Bound (c), in eps(Float32) relative to the largest entry of
# each output: 100 for lda_x (measured 1.9) and 30 for PW92 (measured 0.38).
@testitem "Float32 LDA derivatives at order 6" setup = [LDAReference] begin
    using .LDAReference: exact_big, error_eps
    f = lda_x(Float32)
    r = xc_derivatives(f, Val(6), (1.0f0, 2.0f-15))
    b = xc_derivatives(exact_big(f), Val(6), (big(1.0f0), big(2.0f-15)))
    # Exact: the entry is beyond floatmax(Float32), so it must be −Inf.
    @test b.v6rho6[7] < -floatmax(Float32) && r.v6rho6[7] == -Inf32
    @test error_eps(r, b, Float32) <= 100
    for build in (lda_c_pw, lda_c_pw_mod)
        g = build(Float32)
        ρ = (2.0f0^23, 1.0f0)
        @test error_eps(xc_derivatives(g, Val(6), ρ), xc_derivatives(exact_big(g), Val(6), big.(ρ)), Float32) <= 30
    end
end

# Orders 0 to 10 at the n_max splits, the uneven splits and (n/2, r·n/2) and its mirror for four
# ratios r near ε_ζ/2 and ε_ζ, for 2^p ≤ n_max with p ∈ 0:16:P, p = 5 and p = P. A Float32 call evaluates in Float64
# above `_float32_max_order` and rounds each output. Oracle:
# `exact_big`. Bounds: `point_bound`, with an infinity of the right sign required exactly where the
# BigFloat entry exceeds floatmax(T), so that no NaN passes.
@testitem "LDA functionals at orders 0 to 10" setup = [LDAReference] begin
    using .LDAReference: exact_big, error_eps, point_bound
    for T in (Float32, Float64), K in 0:10, build in (lda_x, lda_c_pw, lda_c_pw_mod)
        g = build(T)
        gb = exact_big(g)
        P = exponent(PureXC._lda_density_max(T, Val(K)))
        for p in unique([0:16:P; 5; P])
            n = ldexp(one(T), p)
            splits = Any[n, (n, zero(T)), (zero(T), n), (n / 2, n / 2), (n / 4 * 3, n / 4)]
            for x in (T(2.0e-15), T(1.0e-6), one(T))
                push!(splits, (n, x), (x, n))
            end
            # Near full polarization, where 1 − ζ is near ε_ζ: the ratios that set n_max, below, at
            # and just above ε_ζ/2, and near ε_ζ.
            for r in (T === Float32 ? (5.0e-8, 6.3e-8, 1.0e-7, 2.0e-7) : (1.1e-16, 1.122e-16, 1.15e-16, 2.25e-16))
                push!(splits, (n / 2, T(r * n / 2)), (T(r * n / 2), n / 2))
            end
            for x in splits
                err = error_eps(xc_derivatives(g, Val(K), x), xc_derivatives(gb, Val(K), map(big, x)), T)
                @test err <= point_bound(g, x, T, K)
            end
        end
    end
end

# Float32 at orders 8 and 9. lda_x at (1, 2e-15), K = 8, evaluated in Float32: the ββ entries of
# orders 6 to 8 are −1.8e69, 4.1e84 and −1.2e100, beyond floatmax(Float32). lda_c_pw at (1, 1e-6),
# K = 9, evaluated in Float64: the order-9 entries range from 19.8 to −1.4e48. Oracle as in "LDA
# functionals at orders 0 to 10". Bound (c), in eps(Float32): 100 for lda_x (measured 1.9) and 30
# for lda_c_pw (measured 0.39, and 0.32 for the entry 19.8 on its own).
@testitem "Float32 LDA derivatives at orders 8 and 9" setup = [LDAReference] begin
    using .LDAReference: exact_big, error_eps
    f = lda_x(Float32)
    r = @inferred xc_derivatives(f, Val(8), (1.0f0, 2.0f-15))
    b = xc_derivatives(exact_big(f), Val(8), (big(1.0f0), big(2.0f-15)))
    # Exact: infinities where the exact entries exceed floatmax(Float32).
    @test (r.v6rho6[7], r.v7rho7[8], r.v8rho8[9]) == (-Inf32, Inf32, -Inf32)
    @test b.v8rho8[9] < -floatmax(Float32)
    @test error_eps(r, b, Float32) <= 100
    g = lda_c_pw(Float32)
    ρ = (1.0f0, 1.0f-6)
    r = @inferred xc_derivatives(g, Val(9), ρ)
    b = xc_derivatives(exact_big(g), Val(9), big.(ρ))
    @test all(x -> !any(isnan, x), values(r))
    @test r.v9rho9[end] == -Inf32 && b.v9rho9[end] < -floatmax(Float32)
    @test isfinite(r.v9rho9[1]) && abs(r.v9rho9[1] - b.v9rho9[1]) <= 30 * eps(Float32) * abs(b.v9rho9[1])
    @test error_eps(r, b, Float32) <= 30
end

# BigFloat at order 12: 256 bits against 512 bits. Bounds (c) in eps of 256 bits, relative to the
# largest entry of each output: lda_x, 500, measured at most 5.3; PW92, 200 + 8κ (κ = 0 restricted),
# with 8κ the 1 ± ζ conditioning of `point_bound` (ε_ζ = eps(Float64) keeps the factor at these
# points), measured at most 108 restricted, 11 at (0.3, 0.1) and 2.6·κ at the other points.
@testitem "BigFloat LDA derivatives at order 12" setup = [LDAReference] begin
    using .LDAReference: zeta_conditioning
    for build in (lda_x, lda_c_pw, lda_c_pw_mod), ρ in (1.0, (1.0, 2.0e-15), (1.0, 1.0e-6), (0.3, 0.1), (1.0e-6, 0.0))
        lo = setprecision(() -> xc_derivatives(build(BigFloat), Val(12), map(big, ρ)), BigFloat, 256)
        hi = setprecision(() -> xc_derivatives(build(BigFloat), Val(12), map(big, ρ)), BigFloat, 512)
        tol = build === lda_x ? 500.0 : ρ isa Real ? 200.0 : 200 + 8 * zeta_conditioning(build(), ρ)
        for name in keys(hi)
            g, w = [getfield(lo, name)...], [getfield(hi, name)...]
            @test all(isfinite, g)
            scale = maximum(abs, w)
            @test maximum(abs.(g .- w)) <= tol * eps(BigFloat(1; precision = 256)) * scale
        end
    end
end

# A Float32 call above `_float32_max_order` converts the functional with `_with_number_type`, so
# every concrete functional of the package needs a method.
@testitem "every functional converts its number type" begin
    using InteractiveUtils: subtypes
    concrete(T) = isabstracttype(T) ? reduce(vcat, map(concrete, subtypes(T)); init = Any[]) : Any[T]
    types = filter(T -> parentmodule(T) === PureXC, concrete(PureXC.AbstractFunctional))
    @test !isempty(types)
    for T in types
        @test hasmethod(PureXC._with_number_type, Tuple{T, Type{Float64}})
    end
end

# The order above which a Float32 call evaluates in Float64, per functional: 10 for lda_x, 2 for
# PW92 (see `_float32_max_order`). Other types are never widened.
@testitem "Float32 evaluation order of the LDA functionals" begin
    E = PureXC._evaluation_type
    @test (E(lda_x(Float32), Float32, Val(10)), E(lda_x(Float32), Float32, Val(11))) === (Float32, Float64)
    for build in (lda_c_pw, lda_c_pw_mod)
        @test (E(build(Float32), Float32, Val(2)), E(build(Float32), Float32, Val(3))) === (Float32, Float64)
    end
    @test E(lda_c_pw(), Float64, Val(12)) === Float64
    @test E(lda_c_pw(Float32), Float64, Val(3)) === Float64
    @test PureXC._float32_max_order(PureXC.AbstractFunctional) == 2
end
