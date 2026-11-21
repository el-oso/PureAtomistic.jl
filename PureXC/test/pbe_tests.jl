# Reference rows of the specs' "Reference values" tables (Libxc 7.0.0, Float64): ρ, σ, zk, vrho and
# vsigma. Restricted rows give numbers; polarized rows give (n_α, n_β), (σ_αα, σ_αβ, σ_ββ) and tuples.
@testmodule PBEReference begin
    using PureXC

    const PBE_X = (
        (1.0e-6, 1.0e-14, -0.009857529291178524, -0.0092956540460543, -144289.43783188995),
        (0.01, 0.001, -0.23661249554580363, -0.2340118340980689, -0.30551809986125983),
        (1.0, 0.5, -0.7406686863550702, -0.9819517873661594, -0.004204845830450659),
        (100.0, 1.0e4, -3.4289981999409207, -4.5695661996270385, -9.117751103206427e-6),
        (
            (0.3, 0.1), (0.02, 0.005, 0.01), -0.5777797708644895,
            (-0.8290883168542295, -0.5668106267843886), (-0.01666238899320948, 0.0, -0.06948722425183211),
        ),
        (
            (1.0, 0.001), (0.5, 1.0e-4, 1.0e-6), -0.9313874433395687,
            (-1.238475205039433, -0.11231928807573745), (-0.003346181950688029, 0.0, -16.002793294311036),
        ),
    )
    const PBE_C = (
        (1.0e-6, 1.0e-14, -0.0018251719785951246, -0.006603549782069997, 163445.18311941568),
        (0.01, 0.001, -0.002082858384624374, -0.011104703381614515, 0.03379190245692333),
        (1.0, 0.5, -0.06915172038977144, -0.08203337875329407, 0.003964180823079662),
        (100.0, 1.0e4, -0.11161135888782478, -0.12321899201646803, 8.863660356898457e-6),
        (
            (0.3, 0.1), (0.02, 0.005, 0.01), -0.05688857138579849,
            (-0.05568950866256309, -0.10201373855798285),
            (0.013285863471859305, 0.026571726943718614, 0.013285863471859305),
        ),
        (
            (1.0, 0.001), (0.5, 1.0e-4, 1.0e-6), -0.036074487176590946,
            (-0.043655184565981234, -0.27853534969333865),
            (0.0030625050689373832, 0.0061250101378747665, 0.0030625050689373832),
        ),
    )
    const FUNCTIONALS = ((pbe_x, PBE_X), (pbe_c, PBE_C))

    total(ρ::Real) = ρ
    total(ρ::Tuple) = ρ[1] + ρ[2]

    # The relative difference of each entry of `got` from `want`; an exact zero must stay zero.
    reldiff(got::Real, want::Real) = iszero(want) ? (iszero(got) ? 0.0 : Inf) : abs(got - want) / abs(want)
    reldiff(got::Tuple, want::Tuple) = maximum(map(reldiff, got, want))

    # The specs' grid: n log-spaced over [1e-6, 1e3], s ∈ {0} ∪ log-spaced [1e-3, 10] with s the
    # exchange reduced gradient, polarized at ζ ∈ {0, 0.25, −0.6, 0.95} with σ_αβ = ½√(σ_αα σ_ββ),
    # each σ_σσ at the same s for its channel. Values are rounded to Float64.
    const GRID_N = exp10.(range(-6, 3; length = 19))
    const GRID_S = [0.0; exp10.(range(-3, 1; length = 9))]
    const GRID_ζ = (0.0, 0.25, -0.6, 0.95)
    const C3 = 4 * cbrt(3 * big(π)^2)^2
    const C6 = 4 * cbrt(6 * big(π)^2)^2
    const GRID_RESTRICTED = [(n, Float64(big(s)^2 * C3 * big(n)^(8 // 3))) for n in GRID_N for s in GRID_S]
    const GRID_POLARIZED = map([(n, ζ, s) for n in GRID_N for ζ in GRID_ζ for s in GRID_S]) do (n, ζ, s)
        a, b = n * (1 + ζ) / 2, n * (1 - ζ) / 2
        saa, sbb = big(s)^2 * C6 * big(a)^(8 // 3), big(s)^2 * C6 * big(b)^(8 // 3)
        ((a, b), (Float64(saa), Float64(sqrt(saa * sbb) / 2), Float64(sbb)))
    end

    # The functional `f` with its parameters and thresholds converted exactly to BigFloat, so that it
    # is the same function as `f`.
    big_thresholds(t) = PureXC.Thresholds{BigFloat}(big(t.density), big(t.sigma), big(t.zeta), big(t.sigma_floor))
    exact_big(f::PureXC.PW92Correlation) = PureXC.PW92Correlation{BigFloat}(
        map(p -> big.(p), f.params), big(f.fz20), big_thresholds(f.thresholds)
    )
    exact_big(f::PureXC.PBEExchange) = PureXC.PBEExchange{BigFloat}(big(f.κ), big(f.μ), big_thresholds(f.thresholds))
    exact_big(f::PureXC.PBECorrelation) = PureXC.PBECorrelation{BigFloat}(
        big(f.β), big(f.γ), big(f.B), exact_big(f.pw), big_thresholds(f.thresholds)
    )

    # n·(a·ε_c + b·H) of pbe_c, the terms of its energy, for the scale of the cancellation in ε_c + H.
    struct LinComb
        a::BigFloat
        b::BigFloat
    end
    (c::LinComb)(ε, H) = c.a * ε + c.b * H
    # ε_c with each series coefficient multiplied by 1 ± scale, the sign a fixed pseudo-random
    # function of `seed` and the coefficient's position; scale 0 leaves ε_c unchanged.
    struct Perturb
        seed::Int
        scale::BigFloat
    end
    perturb_sign(seed, i) = isodd((seed * 2654435761 + i * 40503) >> 7) ? 1 : -1
    (p::Perturb)(ε::PureXC.Taylor{N, K, M, T, L}) where {N, K, M, T, L} =
        PureXC.Taylor{N, K, M, T, L}(ntuple(i -> ε.c[i] * (1 + p.scale * perturb_sign(p.seed, i)), Val(L)))
    struct Term <: PureXC.AbstractGGA{BigFloat}
        f::PureXC.PBECorrelation{BigFloat}
        comb::LinComb
        adjust::Perturb
        thresholds::PureXC.Thresholds{BigFloat}
    end
    Term(f, c, p = Perturb(0, 0)) = Term(f, c, p, f.thresholds)
    PureXC._energy_density(t::Term, ρ, σ) = PureXC._pbe_c_energy(t.f, ρ, σ, t.comb, t.adjust)
    PureXC._energy_density(t::Term, ρ::Tuple, σ::Tuple) = PureXC._pbe_c_energy(t.f, ρ, σ, t.comb, t.adjust)
    PureXC._seed_scales(t::Term, k::Val, ρ::NTuple{2, R}, σ::NTuple{3, R}) where {R} = PureXC._seed_scales(t.f, k, ρ, σ)
    PureXC._seed_scales(t::Term, k::Val, ρ::R, σ::R) where {R} = PureXC._seed_scales(t.f, k, ρ, σ)
    PureXC._cutoff(t::Term, ρ, σ) = PureXC._cutoff(t.f, ρ, σ)
    PureXC._with_number_type(t::Term, ::Type{BigFloat}) = t

    U = PureXC._xc_derivatives_unchecked

    # The order and the multi-indices α of the entries of each output field of `xc_derivatives`,
    # in its layout, for nρ density and nσ σ inputs to order K (computed once per argument set).
    const LAYOUTS = Dict{NTuple{3, Int}, Any}()
    layout(nρ, nσ, K) = get!(() -> make_layout(nρ, nσ, K), LAYOUTS, (nρ, nσ, K))
    function make_layout(nρ, nσ, K)
        ms(n, m) = m == 0 ? [()] : [(i, r...) for i in 1:n for r in ms(n, m - 1) if isempty(r) || r[1] >= i]
        cnt(s, n) = ntuple(i -> count(==(i), s), n)
        out = Any[(0, [ntuple(_ -> 0, nρ + nσ)])]
        for j in 1:K, a in j:-1:0
            push!(out, (j, [(cnt(r, nρ)..., cnt(s, nσ)...) for r in ms(nρ, a) for s in ms(nσ, j - a)]))
        end
        return out
    end

    # The seed scales s of `f` at the point, in the type S it is evaluated in, and for each entry
    # of each output the factor ∏ s_i^α_i / α_i! that turns the derivative ∂^α e into its normalized
    # Taylor coefficient in the seed variables.
    function normalizers(f, K, ρ, σ, S, lay)
        fS = PureXC._functional_in(f, PureXC._result_type(f, ρ, σ), S)
        s = map(big, PureXC._seed_scales(fS, Val(K), map(x -> convert(S, x), ρ), map(x -> convert(S, x), σ)))
        pow = [[s[i]^k / factorial(big(k)) for k in 0:K] for i in eachindex(s)]
        return [[prod(pow[i][α[i] + 1] for i in eachindex(α)) for α in αs] for (_, αs) in lay]
    end

    # The scale of each derivative order j = 0…K at the point, in normalized Taylor coefficients:
    # the largest entry of order j of |want| (pbe_x) or of |D(n ε_c)| + |D(n H)| (pbe_c, the size of
    # the terms whose sum cancels at large t, spec gga_c_pbe section 7), at least floatmin(T). Errors
    # are measured against this per-order scale and not entry by entry or output by output: next to
    # a zero of an entry (an entry changing sign with t², say) the relative error of that entry has
    # no upper limit, while its error against the other coefficients of the same order stays at the
    # rounding level.
    function scales(f, K, ρ, σ, want, T, nrm, lay)
        vals = if f isa PureXC.PBECorrelation
            fb = exact_big(f)
            x, y = map(big, ρ), map(big, σ)
            a = U(Term(fb, LinComb(1, 0)), Val(K), x, y)
            b = U(Term(fb, LinComb(0, 1)), Val(K), x, y)
            [abs.([getfield(a, n)...]) .+ abs.([getfield(b, n)...]) for n in keys(want)]
        else
            [abs.([getfield(want, n)...]) for n in keys(want)]
        end
        sc = fill(big(floatmin(T)), K + 1)
        for (k, (j, _)) in enumerate(lay)
            sc[j + 1] = max(sc[j + 1], maximum(vals[k] .* nrm[k]))
        end
        return sc
    end

    # The largest error of the outputs `got`, in the type T, against `want`, in units of eps(T): for
    # each entry, its normalized error (|got − want| − floatmin(T))·∏s^α/α! less `less` (normalized,
    # at least zero), over the scale of its order. floatmin(T) is the absolute accuracy to which an
    # output of type T can hold a value: below it, an exact entry underflows. A NaN fails (Inf). An entry whose exact value fits in T must be
    # finite; one whose exact value exceeds floatmax(T) must be infinite, and with the exact sign
    # where its normalized size, less `less`, is above `bound` (eps(T) of its order's scale): below
    # that the sign is not determined to the working accuracy.
    function error_eps(got, want, nrm, lay, sc, T; less = nothing, bound = Inf)
        worst = 0.0
        for (k, name) in enumerate(keys(want))
            j = lay[k][1]
            g, w = [getfield(got, name)...], [getfield(want, name)...]
            for i in eachindex(w)
                isnan(g[i]) && return Inf
                d = isnothing(less) ? big(0) : less[k][i]
                if abs(w[i]) <= floatmax(T)
                    isfinite(g[i]) || return Inf
                    e = max(abs(big(g[i]) - w[i]) - floatmin(T), 0) * nrm[k][i]
                    worst = max(worst, Float64(max(e - d, 0) / sc[j + 1] / eps(T)))
                else
                    isinf(g[i]) || return Inf
                    sign(g[i]) == sign(w[i]) || abs(w[i]) * nrm[k][i] - d <= bound * eps(T) * sc[j + 1] || return Inf
                end
            end
        end
        return worst
    end

    # The sensitivity of pbe_c's outputs to the rounding of ε_c: for each entry, the largest change,
    # in BigFloat, when every series coefficient of ε_c is changed by one ulp of the evaluation type
    # S (relative eps(S), two fixed sign patterns). Near full polarization, where z = −ε_c/(γφ³)
    # changes by about 91 per unit of ζ, and at y = A·t² ≈ 1, this is far larger than the rounding of
    # the other terms; measured, the Float64 errors there are of its size.
    function sensitivity(f, K, ρ, σ, want, S)
        fb = exact_big(f)
        x, y = map(big, ρ), map(big, σ)
        outs = [U(Term(fb, LinComb(1, 1), Perturb(seed, big(eps(S)))), Val(K), x, y) for seed in (1, 2)]
        d(o, n) = abs.([getfield(o, n)...] .- [getfield(want, n)...])
        return [max.(d(outs[1], n), d(outs[2], n)) for n in keys(want)]
    end

    # The constant c of the sensitivity term: the bound of a pbe_c point grows by c times the
    # entry's sensitivity (`sensitivity`), bound (c). Measured at the points near full polarization
    # and y ≈ 1 (orders 3 to 6, Float64): with errors per entry against the scale of its order, none
    # exceeds the plain bound, so the term changes no result there.
    const SENSITIVITY_C = 8

    # The error of `f` at order K at (ρ, σ) against `exact_big(f)`, in eps(T) of the result type,
    # per entry against the scale of its order (`error_eps`). `allow` is the bound it will be
    # compared with. For pbe_c, where the plain error exceeds it, each entry's error is first reduced
    # by SENSITIVITY_C times its sensitivity (at most to zero), so that the comparison allows the
    # change from rounding ε_c by one ulp.
    function point_error(f, K, ρ, σ; allow = Inf)
        T = PureXC._result_type(f, ρ, σ)
        got = xc_derivatives(f, Val(K), ρ, σ)
        want = U(exact_big(f), Val(K), map(big, ρ), map(big, σ))
        S = PureXC._evaluation_type(f, T, Val(K))
        lay = layout(ρ isa Tuple ? 2 : 1, ρ isa Tuple ? 3 : 1, K)
        nrm = normalizers(f, K, ρ, σ, S, lay)
        sc = scales(f, K, ρ, σ, want, T, nrm, lay)
        err = error_eps(got, want, nrm, lay, sc, T; bound = allow)
        (err <= allow || !(f isa PureXC.PBECorrelation) || isnan(err)) && return err
        Δ = sensitivity(f, K, ρ, σ, want, S)
        return error_eps(got, want, nrm, lay, sc, T; less = [SENSITIVITY_C .* Δ[k] .* nrm[k] for k in eachindex(nrm)], bound = allow)
    end

    # κ_ζ = 2/(1 − |ζ|) at the raised channels where pbe_c keeps the factors (1 ± ζ)^p of the smaller
    # channel (rule 4 replaces them below ε_ζ), and 0 otherwise: 1 ± ζ is formed from ζ (spec
    # lda_c_pw, section 8), so its relative error is of order eps·κ_ζ.
    function zeta_conditioning(f, ρ)
        f isa PureXC.PBECorrelation && ρ isa Tuple || return 0.0
        t = f.thresholds.density
        a, b = map(x -> big(max(x, t)), ρ)
        ζ = (a - b) / (a + b)
        min(1 + ζ, 1 - ζ) <= f.thresholds.zeta && return 0.0
        return Float64((a + b) / min(a, b))
    end
end

# Oracle: the specs' Libxc 7.0.0 reference rows. Tolerance (a): 1e-6 relative, the user's floor;
# the specs measured the difference of their equations in BigFloat from these rows at ≤ 1.2e-13
# (gga_x_pbe) and ≤ 1.2e-14 (gga_c_pbe). Libxc's exact zeros (vsigma_αβ of exchange) are exact.
@testitem "reference values of the PBE functionals" setup = [PBEReference] begin
    using .PBEReference: FUNCTIONALS, total, reldiff
    for (build, rows) in FUNCTIONALS
        f = build()
        for (ρ, σ, zk, vrho, vsigma) in rows
            r = xc_derivatives(f, Val(1), ρ, σ)
            @test reldiff(r.e / total(ρ), zk) <= 1.0e-6
            @test reldiff(r.vrho, vrho) <= 1.0e-6
            @test reldiff(r.vsigma, vsigma) <= 1.0e-6
            # Exact: the value is formed by the same operations at every order.
            @test energy_density(f, ρ, σ) == r.e
        end
    end
    # γ = (1 − ln 2)/π², the double nearest the exact value (spec gga_c_pbe, section 5), and μ as
    # Libxc reports it (spec gga_x_pbe, section 5). Exact: constants.
    @test pbe_c().γ == 0.031090690869654894
    @test pbe_c(Float32).γ == Float32((1 - log(big(2))) / big(π)^2)
    @test (pbe_c().β, pbe_c().B) == (0.06672455060314922, 1.0)
    @test (pbe_x().κ, pbe_x().μ) == (0.804, 0.2195149727645171)
    @test pbe_c().pw == lda_c_pw_mod()
    @test (kind(pbe_x()), kind(pbe_c()), family(pbe_x())) == (:exchange, :correlation, :gga)
    for (f, n, s) in ((pbe_x(), 1.0e-15, 1.0000000000000027e-20), (pbe_c(), 1.0e-12, 1.0000000000000021e-16))
        @test (f.thresholds.density, f.thresholds.sigma, f.thresholds.zeta) == (n, s, 2.220446049250313e-16)
        @test f.thresholds.sigma_floor == s^2
    end
end

# Spec gga_x_pbe, section 7, rules 1 and 2: restricted input is cut off for n ≤ 2 n_thr, a channel
# at or below n_thr contributes exact zeros.
@testitem "pbe_x cut-off and per-channel rule" begin
    f = pbe_x()
    for n in (0.999, 1.0, 1.001, 1.999, 2.0) .* 1.0e-15
        @test all(iszero, values(xc_derivatives(f, Val(3), n, 0.0)))
    end
    # Oracle: the spec's zk = −9.3152e-6 at n = 2.001e-15, σ = 0. Tolerance (a): half a unit in the
    # last printed digit.
    @test abs(xc_derivatives(f, Val(0), 2.001e-15, 0.0).e / 2.001e-15 + 9.3152e-6) <= 0.5e-10
    ref = xc_derivatives(f, Val(3), (0.01, 0.0), (1.0e-3, 0.0, 0.0))
    for ρβ in (0.0, 5.0e-16, 1.0e-15)
        r = xc_derivatives(f, Val(3), (0.01, ρβ), (1.0e-3, 0.0, 0.0))
        # Oracle: the spec's zk. Tolerance (a): 1e-6 relative, the user's floor.
        @test abs(r.e / (0.01 + ρβ) + 0.27975390539018097) <= 1.0e-6 * 0.27975390539018097
        # Exact: the β channel contributes exact zeros (Libxc reports vrho_β ≈ 7e-17 from its
        # channel raised to n_thr), and the α channel is the same computation at all three points.
        @test iszero(r.vrho[2]) && iszero(r.vsigma[3])
        @test r == ref
    end
end

# Spec gga_x_pbe, section 7, rule 3: σ is raised to σ_thr², and vsigma at a raised σ is the derivative
# at the floor.
@testitem "pbe_x σ floor" begin
    f = pbe_x()
    floor = 1.0000000000000027e-20^2
    @test f.thresholds.sigma_floor == floor
    ref = xc_derivatives(f, Val(2), 3.0e-15, 0.0)
    # Oracle: the spec's values. Tolerance (a): 1e-6 relative, the user's floor.
    @test abs(ref.e / 3.0e-15 + 1.0655121955534234e-5) <= 1.0e-6 * 1.0655121955534234e-5
    @test abs(ref.vsigma + 9.78023690313513e16) <= 1.0e-6 * 9.78023690313513e16
    for x in (0.5, 1.0)
        # Exact: the raised σ is the floor, and the seed scale is r(n), the same at all three points.
        @test xc_derivatives(f, Val(2), 3.0e-15, x * floor) == ref
    end
    @test xc_derivatives(f, Val(0), 3.0e-15, 1.5 * floor).e != ref.e
    # Per channel.
    r0 = xc_derivatives(f, Val(2), (0.01, 3.0e-15), (1.0e-3, 0.0, 0.0))
    @test xc_derivatives(f, Val(2), (0.01, 3.0e-15), (1.0e-3, 0.0, floor)) == r0
    @test xc_derivatives(f, Val(2), (0.01, 3.0e-15), (1.0e-3, 0.0, 1.5 * floor)).vsigma[3] != r0.vsigma[3]
    # Spec section 7, "Limits": as a channel empties its second derivatives diverge. Oracle: the
    # spec's Libxc values at (0.01, 1.5e-15), σ = (1e-3, 0, 0), printed to 3 significant digits.
    # Tolerance (a): half a unit in the last printed digit.
    r = xc_derivatives(f, Val(2), (0.01, 1.5e-15), (1.0e-3, 0.0, 0.0))
    @test abs(r.v2rho2[3] + 3.18e9) <= 0.005e9
    @test abs(r.v2sigma2[6] - 5.9e54) <= 0.05e54
end

# Spec gga_c_pbe, section 7, rule 1: one cut-off, the total density below n_thr (strict), restricted
# and polarized alike.
@testitem "pbe_c cut-off at the density threshold" begin
    f = pbe_c()
    for (ρ, σ) in ((0.999e-12, 1.0e-30), ((0.999e-12, 0.0), (1.0e-30, 0.0, 0.0)), ((0.4995e-12, 0.4995e-12), (0.0, 0.0, 0.0)))
        @test all(x -> all(iszero, x), values(xc_derivatives(f, Val(3), ρ, σ)))
    end
    # Oracle: the spec's zk = −6.6604e-5 at n = 1e-12 (σ = 0). Tolerance (a): half a unit in the
    # last printed digit.
    @test abs(xc_derivatives(f, Val(0), 1.0e-12, 0.0).e / 1.0e-12 + 6.6604e-5) <= 0.5e-9
    for n in (1.999e-12, 2.001e-12)
        @test xc_derivatives(f, Val(0), n, 0.0).e < 0
    end
end

# Spec gga_c_pbe, section 7, rule 2: each channel is raised to n_thr, the derivatives are those at
# the raised densities, and e = n·(ε_c + H) with the caller's n.
@testitem "pbe_c channels raised to the density threshold" begin
    f = pbe_c()
    # Oracle: the spec's values. Tolerance (a): 1e-6 relative, the user's floor.
    for (σ, zk, vβ, v2ββ) in (
            ((1.0e-3, 0.0, 0.0), -0.0006198385727128573, 3.818163979061827, -1.2791281029994136e12),
            ((1.0e-4, 0.0, 0.0), -0.010227071671042167, 16.23626753063028, nothing),
        )
        ref = xc_derivatives(f, Val(2), (0.01, 0.0), σ)
        for ρβ in (0.0, 0.5e-12, 1.0e-12)
            n = 0.01 + ρβ
            r = xc_derivatives(f, Val(2), (0.01, ρβ), σ)
            @test abs(r.e / n - zk) <= 1.0e-6 * abs(zk)
            @test abs(r.vrho[2] - vβ) <= 1.0e-6 * abs(vβ)
            isnothing(v2ββ) || @test abs(r.v2rho2[3] - v2ββ) <= 1.0e-6 * abs(v2ββ)
            # e/n is the same ε_c + H; bound (b): one rounding of the product and one of the
            # quotient on each side, 4 roundings of eps/2.
            @test abs(r.e / n - ref.e / 0.01) <= 2 * eps() * abs(zk)
            # Exact: the same raised densities, seeded at the same scales.
            for name in keys(ref)[2:end]
                @test getfield(r, name) == getfield(ref, name)
            end
        end
    end
    ref = xc_derivatives(f, Val(2), (1.0e-12, 1.0e-12), (1.0e-30, 0.0, 1.0e-30))
    for x in (0.5005e-12, 0.9995e-12)
        r = xc_derivatives(f, Val(2), (x, x), (1.0e-30, 0.0, 1.0e-30))
        @test abs(r.e / 2x - ref.e / 2.0e-12) <= 2 * eps() * abs(ref.e / 2.0e-12)
        for name in keys(ref)[2:end]
            @test getfield(r, name) == getfield(ref, name)
        end
    end
end

# Spec gga_c_pbe, section 7, rules 3 and 5: each σ_σσ (restricted: σ) and the total σ are raised to
# σ_thr², and vsigma at a raised σ is the derivative at the floor.
@testitem "pbe_c σ floor" begin
    f = pbe_c()
    floor = 1.0000000000000021e-16^2
    @test f.thresholds.sigma_floor == floor
    ref = xc_derivatives(f, Val(2), 1.0e-11, 0.0)
    # Oracle: the spec's values. Tolerance (a): 1e-6 relative, the user's floor.
    @test abs(ref.e / 1.0e-11 + 0.00014165595883248967) <= 1.0e-6 * 0.00014165595883248967
    @test abs(ref.vsigma - 1.9656605414274226e12) <= 1.0e-6 * 1.9656605414274226e12
    for x in (0.5, 1.0)
        # Exact: the raised σ is the floor, and the seed scale is the t-scale at n.
        @test xc_derivatives(f, Val(2), 1.0e-11, x * floor) == ref
    end
    @test xc_derivatives(f, Val(0), 1.0e-11, 1.5 * floor).e != ref.e
    # Rule 5: σ_αα + 2σ_αβ + σ_ββ = 0 is raised to the floor. Exact: the total is the floor at both
    # points and the series of the total is the same sum of the seeds.
    a = xc_derivatives(f, Val(3), (0.5, 0.5), (1.0e7, -1.0e7, 1.0e7))
    b = xc_derivatives(f, Val(3), (0.5, 0.5), (2.0, -2.0, 2.0))
    @test all(x -> all(isfinite, x), values(a))
    @test a == b
    # A total below zero by rounding, inside the σαβ bound, is raised too.
    c = 0.3^2
    r = xc_derivatives(f, Val(2), (0.5, 0.5), (c, -nextfloat(c), c))
    @test all(x -> all(isfinite, x), values(r)) && r == xc_derivatives(f, Val(2), (0.5, 0.5), (c, -c, c))
    # Beyond the bound the input is a caller error; Libxc clamps σαβ instead (spec section 7).
    @test_throws "xc: " xc_derivatives(f, Val(1), (0.1, 0.05), (1.0e-2, 6.0e-3, 1.0e-4))
    @test isfinite(xc_derivatives(f, Val(1), (0.1, 0.05), (1.0e-2, 9.0e-4, 1.0e-4)).e)
end

# Spec gga_c_pbe, section 7, rule 4: a factor (1 ± ζ)^p with 1 ± ζ ≤ ε_ζ, in f(ζ) and φ(ζ), is the
# constant ε_ζ^p, which has no derivative. At (1e5, 0) the channel raised to n_thr gives
# 1 − ζ = 2e-17 < ε_ζ; at (1e3, 1e-6) the factors are kept. Only the qualitative effect is checked:
# the second derivative in the empty channel is moderate where the factors are constant and large
# where they are kept (spec section 7, "Limits").
@testitem "pbe_c ζ-factor replacement" begin
    f = pbe_c()
    small = xc_derivatives(f, Val(2), (1.0e5, 0.0), (1.0, 0.0, 0.0)).v2rho2[3]
    large = xc_derivatives(f, Val(2), (1.0e3, 1.0e-6), (1.0, 0.0, 0.0)).v2rho2[3]
    @test isfinite(small) && abs(small) < 1
    @test abs(large) > 100
    # Oracle: Base cbrt. Bound (b): fast_cbrt within 2 ulps, squared and summed, 8 roundings of eps/2.
    @test abs(PureXC._pbe_phi(1.0, eps()) - (cbrt(2.0)^2 + cbrt(eps())^2) / 2) <= 4 * eps()
end

# Review focus 1: a fully polarized point gives finite values at every order, equal to the rule's
# floor behavior: the empty channel at n_thr.
@testitem "fully polarized PBE points" begin
    for f in (pbe_x(), pbe_c()), K in 0:4
        t = f.thresholds.density
        for (ρ, σ) in (((0.3, 0.0), (0.02, 0.0, 0.0)), ((0.0, 0.3), (0.0, 0.0, 0.02)), ((1.0e-6, 0.0), (1.0e-14, 0.0, 0.0)))
            r = xc_derivatives(f, Val(K), ρ, σ)
            @test all(x -> all(isfinite, x), values(r))
            # Oracle: the empty channel at n_thr. The derivatives are exact: the same raised
            # densities and σ's, seeded at the same scales. pbe_x: the channel at n_thr contributes
            # exact zeros, so e is exact too. pbe_c: e = n(ε_c + H) with the caller's n, which
            # changes by n_thr/n ≤ 1e-6, plus two roundings (bound (b)).
            w = xc_derivatives(f, Val(K), map(x -> iszero(x) ? t : x, ρ), σ)
            f isa PureXC.PBEExchange ? (@test r.e == w.e) : (@test abs(r.e - w.e) <= (1.0e-6 + 2 * eps()) * abs(w.e))
            for name in keys(r)[2:end]
                @test getfield(r, name) == getfield(w, name)
            end
        end
    end
end

# Review focus 2: ρ = 0 and ρ = −0.0 give exact zeros at every order.
@testitem "zero densities give exact PBE zeros" begin
    for build in (pbe_x, pbe_c), T in (Float32, Float64), K in 0:4
        f = build(T)
        for (ρ, σ) in ((0.0, 0.0), (-0.0, 0.0), (-0.0, 1.0), (0.0f0, 1.0f0))
            @test all(iszero, values(xc_derivatives(f, Val(K), ρ, σ)))
        end
        for (ρ, σ) in (((0.0, 0.0), (0.0, 0.0, 0.0)), ((-0.0, 0.0), (1.0, -0.5, 1.0)), ((-0.0, -0.0), (0.0, -0.0, 0.0)), ((0.0f0, -0.0f0), (1.0f0, 0.0f0, 0.0f0)))
            @test all(x -> all(iszero, x), values(xc_derivatives(f, Val(K), ρ, σ)))
        end
    end
end

# Review focus 3: at σ = 0 every derivative is finite and vsigma is its limit, the derivative at the
# floor, which at n = 1 is the derivative at σ = 0 to within the floor's s² ≈ 3e-42. The seed scale
# of σ is positive. Oracle: the limits of spec gga_x_pbe and gga_c_pbe, section 7. Bound (b):
# 16·eps relative, for at most 8 roundings on either side.
@testitem "PBE at σ = 0" begin
    n = 1.0
    c3, c6 = 4 * cbrt(3 * big(π)^2)^2, 4 * cbrt(6 * big(π)^2)^2
    r = xc_derivatives(pbe_x(), Val(4), n, 0.0)
    @test all(isfinite, values(r))
    lim = -(3 / big(4)) * cbrt(3 / big(π)) * big(n)^(4 // 3) * big(pbe_x().μ) / (c3 * big(n)^(8 // 3))
    @test abs(r.vsigma - lim) <= 16 * eps() * abs(lim)
    ρ = (0.75, 0.25)
    rp = xc_derivatives(pbe_x(), Val(4), ρ, (0.0, 0.0, 0.0))
    @test all(x -> all(isfinite, x), values(rp))
    for (i, x) in ((1, ρ[1]), (3, ρ[2]))
        lim = -(3 / big(4)) * cbrt(6 / big(π)) * big(x)^(4 // 3) * big(pbe_x().μ) / (c6 * big(x)^(8 // 3))
        @test abs(rp.vsigma[i] - lim) <= 16 * eps() * abs(lim)
    end
    @test iszero(rp.vsigma[2])
    # pbe_c: ∂e/∂σ → φ·βπ/(16(3π²)^(1/3) n^(4/3)), the same for σ_αα and σ_ββ and twice that for σ_αβ.
    β = big(pbe_c().β)
    k = β * big(π) / (16 * cbrt(3 * big(π)^2))
    r = xc_derivatives(pbe_c(), Val(4), n, 0.0)
    @test all(isfinite, values(r))
    @test abs(r.vsigma - k) <= 16 * eps() * k
    rp = xc_derivatives(pbe_c(), Val(4), ρ, (0.0, 0.0, 0.0))
    @test all(x -> all(isfinite, x), values(rp))
    φ = (cbrt(big(1.5))^2 + cbrt(big(0.5))^2) / 2
    @test all(abs.(rp.vsigma .- (φ * k, 2φ * k, φ * k)) .<= 16 * eps() * 2φ * k)
    for f in (pbe_x(), pbe_c()), x in ((n, 0.0), ((1.0, 0.0), (0.0, 0.0, 0.0)))
        @test all(>(0), PureXC._seed_scales(f, Val(4), x...))
    end
end

# Review focus 4: a Float64 functional promotes Float32 inputs; a Float32 functional with Float32
# inputs stays Float32.
@testitem "PBE result types" begin
    for build in (pbe_x, pbe_c)
        r = @inferred xc_derivatives(build(Float32), Val(3), (0.3f0, 0.1f0), (0.02f0, 0.005f0, 0.01f0))
        @test r.e isa Float32 && all(x -> x isa Tuple{Vararg{Float32}}, values(r)[2:end])
        @test (@inferred xc_derivatives(build(Float32), Val(2), 0.3f0, 0.02f0)).v2sigma2 isa Float32
        @test (@inferred xc_derivatives(build(), Val(2), 0.3f0, 0.02f0)).v2sigma2 isa Float64
        @test (@inferred xc_derivatives(build(), Val(1), (0.3f0, 0.1f0), (0.02f0, 0.0f0, 0.01f0))).vsigma isa NTuple{3, Float64}
        @test (@inferred energy_density(build(Float32), 0.3f0, 0.02f0)) isa Float32
    end
end

# The parameter gradient: κ as a Taylor variable gives ∂e/∂κ. Oracle: the central difference of the
# same functional in BigFloat (256 bits) with step h = 1e-20, whose truncation error is of order
# h²·∂³e/∂κ³ ≈ 1e-40 relative. Bound (c): 16·eps relative, measured at most 2.8·eps.
@testitem "pbe_x gradient in κ" setup = [PBEReference] begin
    using PureTaylor
    using .PBEReference: big_thresholds
    f0 = pbe_x()
    κ = variable(Val(1), Val(1), Val(1), f0.κ)
    f = PureXC.PBEExchange{typeof(κ)}(κ, f0.μ, f0.thresholds)
    @test f.thresholds === f0.thresholds && f.μ isa typeof(κ)
    h = big(1.0e-20)
    worst = Ref(0.0)
    for (ρ, σ) in ((1.0, 0.5), (1.0e-6, 1.0e-14), (100.0, 1.0e4), ((0.3, 0.1), (0.02, 0.005, 0.01)), ((1.0, 0.0), (0.5, 0.0, 0.0)))
        e = energy_density(f, ρ, σ)
        @test e isa Taylor
        d = derivatives(e, (1.0,))
        # Exact: the value is the same operations on the same numbers.
        @test d[1] == energy_density(f0, ρ, σ)
        fb(k) = PureXC.PBEExchange{BigFloat}(big(f0.κ) + k, big(f0.μ), big_thresholds(f0.thresholds))
        want = (energy_density(fb(h), map(big, ρ), map(big, σ)) - energy_density(fb(-h), map(big, ρ), map(big, σ))) / 2h
        err = Float64(abs(d[2] - want) / abs(want) / eps())
        worst[] = max(worst[], err)
        @test err <= 16
    end
    @test 0 <= worst[] <= 16
    @test_throws "xc: " xc_derivatives(f, Val(1), 1.0, 0.5)
end

# Mixed ρ–σ partials. Oracle: central differences in BigFloat (256 bits) of the next lower order,
# with step h = 1e-20·x: truncation error of order h² ≈ 1e-40 relative, rounding 1e-77/1e-20. This checks that the
# mixed fields hold the partials of the named variables, independently of the series code. Bound
# (c): 5e-41 relative to the largest entry of the field, measured at most 8.2e-43.
@testitem "mixed ρ–σ partials of the PBE functionals" setup = [PBEReference] begin
    using .PBEReference: exact_big
    for build in (pbe_x, pbe_c)
        fb = exact_big(build())
        ρ, σ = big.((0.3, 0.1)), big.((0.02, 0.005, 0.01))
        r = xc_derivatives(fb, Val(3), ρ, σ)
        for j in 1:3
            h = σ[j] * big(1.0e-20)
            up = xc_derivatives(fb, Val(2), ρ, ntuple(i -> i == j ? σ[i] + h : σ[i], 3))
            dn = xc_derivatives(fb, Val(2), ρ, ntuple(i -> i == j ? σ[i] - h : σ[i], 3))
            fd(name) = (getfield(up, name) .- getfield(dn, name)) ./ 2h
            # ∂vrho_i/∂σ_j = v2rhosigma[i, j]; ∂v2rho2_ab/∂σ_j = v3rho2sigma[ab, j].
            for (lower, upper, nρ) in ((:vrho, :v2rhosigma, 2), (:v2rho2, :v3rho2sigma, 3))
                want = fd(lower)
                got = getfield(r, upper)[j:3:(3 * nρ)]
                @test maximum(abs.(got .- want)) <= big(5.0e-41) * maximum(abs, getfield(r, upper))
            end
        end
    end
end

@testitem "PBE functionals allocate nothing" begin
    using StrictModeTest, PureTaylor
    for T in (Float32, Float64), build in (pbe_x, pbe_c)
        f = build(T)
        ρ, σ = T(0.3), T(0.02)
        ρp, σp = (T(0.3), T(0.1)), (T(0.02), T(0.005), T(0.01))
        xc_derivatives(f, Val(2), ρ, σ)
        xc_derivatives(f, Val(2), ρp, σp)
        # @test_noalloc throws a StrictViolation unless AllocCheck proves the call allocation-free.
        @test (@test_noalloc xc_derivatives(f, Val(2), ρ, σ)) isa NamedTuple
        @test (@test_noalloc xc_derivatives(f, Val(2), ρp, σp)) isa NamedTuple
    end
    # Float32 at the last order evaluated in Float32, `_float32_max_order`, and the first evaluated
    # in Float64.
    for build in (pbe_x, pbe_c), (ρ, σ) in ((0.3f0, 0.02f0), ((0.3f0, 0.1f0), (0.02f0, 0.005f0, 0.01f0)))
        f = build(Float32)
        k32 = PureXC._float32_max_order(typeof(f))
        for K in (k32, k32 + 1)
            xc_derivatives(f, Val(K), ρ, σ)
            @test (@test_noalloc xc_derivatives(f, Val(K), ρ, σ)) isa NamedTuple
        end
    end
    # The energy density with κ as a series variable.
    f0 = pbe_x()
    κ = variable(Val(1), Val(1), Val(1), f0.κ)
    f = PureXC.PBEExchange{typeof(κ)}(κ, f0.μ, f0.thresholds)
    energy_density(f, 0.3, 0.02)
    energy_density(f, (0.3, 0.1), (0.02, 0.005, 0.01))
    @test (@test_noalloc energy_density(f, 0.3, 0.02)) isa Taylor
    @test (@test_noalloc energy_density(f, (0.3, 0.1), (0.02, 0.005, 0.01))) isa Taylor
end

# Oracle: the same functional built for BigFloat (256 bits), on the same Float64 inputs converted
# exactly (`exact_big`). Each entry's error is in normalized Taylor coefficients, against the largest
# coefficient of its order (`scales`, `error_eps`): for pbe_c the size of the terms n·ε_c and n·H,
# whose sum cancels at large t (spec gga_c_pbe, section 7). Bound
# (b) c·eps·N, with N the roundings of eps/2 that reach one output coefficient on the longest path,
# polarized pbe_c, counted as in "LDA functionals agree with BigFloat":
# - its PW92 part, 2100;
# - H: 30 series operations (φ: 2 + 3; φ², φ³: 2; q and t²: 4; −ε_c/(γφ³) and A: 4; y: 2; the two
#   forms of the argument: 8; γφ³·log1p: 2; the sum with ε_c and the product with n: 2; the change
#   of variables in three variables: 1), each a coefficient that is a sum of at most 20 products,
#   so at most 40 roundings: 1200;
# - 6 univariate functions (two (1 ± ζ)^(2/3), x^(−7/6), fast_expm1, two inverses, fast_log1p),
#   each K = 3 Horner steps of a series product, 6·3·40 = 720, and their scalar kernels within
#   2 ulps, 28;
# N = 2100 + 1200 + 720 + 28 ≈ 4100 and c = 1. pbe_x takes the same bound. Where pbe_c keeps the
# factors (1 ± ζ)^p of the smaller channel the bound adds 8κ_ζ, κ_ζ = 2/(1 − |ζ|), the conditioning
# of 1 ± ζ formed from ζ (spec lda_c_pw, section 8), as for PW92. Largest measured at the points
# without the 8κ_ζ term: pbe_x 13, pbe_c 12.
@testitem "PBE functionals agree with BigFloat" setup = [PBEReference] begin
    using .PBEReference: GRID_RESTRICTED, GRID_POLARIZED, point_error, zeta_conditioning
    worst = Dict(pbe_x => Ref(0.0), pbe_c => Ref(0.0))
    for build in (pbe_x, pbe_c)
        f = build()
        for (ρ, σ) in Iterators.flatten((GRID_RESTRICTED, GRID_POLARIZED))
            b = 4100 + 8 * zeta_conditioning(f, ρ)
            err = point_error(f, 3, ρ, σ; allow = b)
            worst[build][] = max(worst[build][], err)
            @test err <= b
        end
    end
    @test 0 < worst[pbe_x][] && 0 < worst[pbe_c][]
end

# The domain limits: n_max of the GGA family and the reduced-gradient limits s²_max (pbe_x, per
# channel) and t²_max (pbe_c). Exact: the measured limits are powers of two. At each limit the
# outputs are within the bound of "PBE functionals agree with BigFloat" (orders 0 to 3; higher orders
# are in "PBE functionals at orders 0 to 10"); above it the input is a caller error.
@testitem "PBE domain limits" setup = [PBEReference] begin
    using .PBEReference: point_error, zeta_conditioning
    @test PureXC._gga_density_max(Float64, Val(3)) == 2.0^381
    @test PureXC._gga_density_max(Float32, Val(3)) == 2.0f0^45
    @test PureXC._gga_density_max(BigFloat, Val(7)) == big(2.0)^381
    @test Tuple(map(K -> exponent(PureXC._pbe_x_s2_max(Float32, Val(K))), 0:10)) == (127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127)
    @test Tuple(map(K -> exponent(PureXC._pbe_x_s2_max(Float64, Val(K))), 0:10)) == (1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023)
    @test Tuple(map(K -> exponent(PureXC._pbe_c_t2_max(Float32, Val(K))), 0:10)) == (19, 19, 19, 19, 127, 127, 16, 16, 16, 14, 13)
    @test Tuple(map(K -> exponent(PureXC._pbe_c_t2_max(Float64, Val(K))), 0:10)) == (131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131)
    c3 = 4 * cbrt(3 * big(π)^2)^2
    tscale(n) = PureXC._pbe_t_scale(big(n), big(1.0))
    for T in (Float32, Float64), K in 0:3
        nmax = PureXC._gga_density_max(T, Val(K))
        for build in (pbe_x, pbe_c)
            f = build(T)
            # n_max, restricted and split, at σ = 0 and at s² = t² = 1.
            for (ρ, σ) in ((nmax, zero(T)), ((nmax / 2, nmax / 2), (zero(T), zero(T), zero(T))), ((nmax, zero(T)), (T(1), zero(T), zero(T))))
                b = 4100 + 8 * zeta_conditioning(f, ρ)
                @test point_error(f, K, ρ, σ; allow = b) <= b
            end
            for (ρ, σ) in ((nextfloat(nmax), zero(T)), ((nmax, nmax), (zero(T), zero(T), zero(T))), (floatmax(T), zero(T)))
                @test_throws "xc: total density" xc_derivatives(f, Val(K), ρ, σ)
            end
            # The gradient limit at n = 2^-20, just inside (1 − 2^-10 of it) and twice above.
            n = T(2.0^-20)
            m = build === pbe_x ? big(PureXC._pbe_x_s2_max(T, Val(K))) * c3 * big(n)^(8 // 3) :
                big(PureXC._pbe_c_t2_max(T, Val(K))) * tscale(n)
            inside = T(m * (1 - big(2.0)^-10))
            if isfinite(inside)
                @test point_error(f, K, n, inside; allow = 4100) <= 4100
                @test_throws "xc: reduced gradient" xc_derivatives(f, Val(K), n, T(2 * m))
            end
        end
    end
    # Polarized: the limit applies to each channel of pbe_x and to the total σ of pbe_c.
    c6 = 4 * cbrt(6 * big(π)^2)^2
    # s² of the channel at 1e-12 is about 1e330, above s²_max (and formed as Inf).
    @test big(1.0e300) / (c6 * big(1.0e-12)^(8 // 3)) > big(PureXC._pbe_x_s2_max(Float64, Val(2)))
    @test_throws "xc: reduced gradient s²_β" xc_derivatives(pbe_x(), Val(2), (1.0, 1.0e-12), (1.0, 0.0, 1.0e300))
    @test_throws "xc: reduced gradient s²_α" xc_derivatives(pbe_x(), Val(2), (1.0e-12, 1.0), (1.0e300, 0.0, 1.0))
    t2 = big(PureXC._pbe_c_t2_max(Float64, Val(2)))
    g = Float64(2 * t2 * PureXC._pbe_t_scale(big(2.0), big(1.0)))
    @test_throws "xc: reduced gradient t²" xc_derivatives(pbe_c(), Val(2), (1.0, 1.0), (g / 4, g / 4, g / 4))
    # A channel at or below n_thr of pbe_x is not limited: it contributes nothing.
    @test iszero(xc_derivatives(pbe_x(), Val(2), (1.0, 0.0), (1.0, 0.0, 1.0e300)).vsigma[3])
end

# Orders 0 to 10 at 2^p for p ∈ {−20} ∪ 0:16:P ∪ {5, P} up to n_max = 2^P, and at the smallest
# densities not cut off, over the splits of `_lda_density_max`, each at s² (pbe_x, per channel) or
# t² (pbe_c) of 0, 1 and the limit of order K. A Float32 call evaluates in Float64 above
# `_float32_max_order` and rounds each output. Oracle: `exact_big`. Bounds (`point_bound`), per entry
# against the largest normalized coefficient of its order (`error_eps`); an entry whose exact value
# exceeds floatmax(T) must be infinite, with its sign where it is above the bound, and no NaN
# passes:
# - Float32 evaluated in Float64 and rounded: 45, bound (c), measured at most 0.47;
# - K ≤ 3: 4100, bound (b) of "PBE functionals agree with BigFloat", measured at most 898;
# - 4 ≤ K ≤ 7: 10^4, bound (c), measured at most 898;
# - K ≥ 8: 10^5, bound (c), measured at most 71;
# - pbe_c where the factors (1 ± ζ)^p of the smaller channel are kept: plus 8κ_ζ, as for PW92.
# Polarized pbe_c is checked here to order 6 (`kmax_polarized_c`); see the comment there.
@testitem "PBE functionals at orders 0 to 10" setup = [PBEReference] begin
    using .PBEReference: point_error, zeta_conditioning
    # Polarized pbe_c is a series in five coupled variables whose compilation grows quickly with the
    # order (measured at order 8: 153 s in Float64 and 26 min in BigFloat), so order 7 is measured
    # outside the suite; orders 9 and 10 overflow the stack.
    kmax_polarized_c = 6
    function point_bound(f, ρ, T, K)
        T === Float32 && K > PureXC._float32_max_order(typeof(f)) && return 45.0
        return (K <= 3 ? 4100.0 : K <= 7 ? 1.0e4 : 1.0e5) + 8 * zeta_conditioning(f, ρ)
    end
    c3, c6 = 4 * cbrt(3 * big(π)^2)^2, 4 * cbrt(6 * big(π)^2)^2
    # σ at the reduced gradient g (s² or t²), from the exact values, rounded to T.
    function gradient(f, T, ρ, g)
        if f isa PureXC.PBEExchange
            ρ isa Tuple || return T(g * c3 * big(ρ)^(8 // 3))
            a, b = g * c6 * big(ρ[1])^(8 // 3), g * c6 * big(ρ[2])^(8 // 3)
            return (T(a), T(sqrt(a * b) / 2), T(b))
        end
        t = big(f.thresholds.density)
        ρ isa Tuple || return T(g * PureXC._pbe_t_scale(max(big(ρ), t), big(1.0)))
        a, b = max(big(ρ[1]), t), max(big(ρ[2]), t)
        s = g * PureXC._pbe_t_scale(a + b, PureXC._pbe_phi((a - b) / (a + b), big(f.thresholds.zeta))) / 4
        return (T(s), T(s), T(s))
    end
    for T in (Float32, Float64), build in (pbe_x, pbe_c), K in 0:10
        f = build(T)
        P = exponent(PureXC._gga_density_max(T, Val(K)))
        lim = build === pbe_x ? PureXC._pbe_x_s2_max(T, Val(K)) : PureXC._pbe_c_t2_max(T, Val(K))
        ns = [ldexp(one(T), p) for p in unique([-20; 0:16:P; 5; P])]
        push!(ns, build === pbe_x ? T(2.5e-15) : T(1.0e-12))
        for n in ns
            splits = Any[n, (n, zero(T)), (zero(T), n), (n / 2, n / 2), (n / 4 * 3, n / 4)]
            for x in (T(2.0e-15), T(1.0e-6), one(T))
                push!(splits, (n, x), (x, n))
            end
            for r in (T === Float32 ? (5.0e-8, 1.0e-7, 2.0e-7) : (1.1e-16, 1.15e-16, 2.25e-16))
                push!(splits, (n / 2, T(r * n / 2)), (T(r * n / 2), n / 2))
            end
            for ρ in splits
                build === pbe_c && ρ isa Tuple && K > kmax_polarized_c && continue
                for g in (big(0), big(1), big(lim) * (1 - big(2.0)^-10))
                    σ = gradient(f, T, ρ, g)
                    all(isfinite, σ) || continue
                    b = point_bound(f, ρ, T, K)
                    @test point_error(f, K, ρ, σ; allow = b) <= b
                end
            end
        end
    end
end

# The multiplier B of spec gga_c_pbe (E2): the bracket of H is (1 + B·A·t²)/(1 + A·t² + B·A²t⁴).
# Oracle: the spec's Libxc 7.0.0 zk at B = 0, 0.5 and 2. Tolerance (a): 1e-6 relative, the user's
# floor; measured at most 5.6e-15.
@testitem "pbe_c with the multiplier B" begin
    f0 = pbe_c()
    with_B(B) = PureXC.PBECorrelation{Float64}(f0.β, f0.γ, B, f0.pw, f0.thresholds)
    rows = (
        (0.1, 0.05, -0.030048344929995998, -0.028185598834314613, -0.0238211968403012),
        (1.0, 3.0, -0.06094730572071168, -0.060755891958294214, -0.060190890487385275),
        (0.01, 0.001, -0.006592785715365489, -0.0031645952863285365, -0.001237283120988128),
    )
    for (n, σ, z0, zh, z2) in rows, (B, zk) in ((0.0, z0), (0.5, zh), (2.0, z2))
        @test abs(xc_derivatives(with_B(B), Val(0), n, σ).e / n - zk) <= 1.0e-6 * abs(zk)
    end
    # Large y = A·t², where the forms of H differ from those at small y. Oracle: (E1)–(E3) written
    # out in BigFloat (256 bits), and their central difference in σ with step 1e-25·σ (truncation
    # of order 1e-50). Bound (b): 8·eps relative to n·|ε_c| for e (whose value cancels to order
    # 1/y); bound (c) for vsigma: 32·eps relative; measured at most 1.6·eps and 8.7·eps.
    bt(t) = PureXC.Thresholds{BigFloat}(big(t.density), big(t.sigma), big(t.zeta), big(t.sigma_floor))
    pw = PureXC.PW92Correlation{BigFloat}(map(p -> big.(p), f0.pw.params), big(f0.pw.fz20), bt(f0.pw.thresholds))
    β, γ = big(f0.β), big(f0.γ)
    function direct(B, n, σ)
        ε = PureXC._pw92_eps(pw, PureXC._seitz_radius(n))
        t2 = big(π) * σ / (16 * cbrt(3 * big(π)^2 * n) * n^2)
        A = β / γ / expm1(-ε / γ)
        return n * (ε + γ * log1p(β / γ * t2 * (1 + B * A * t2) / (1 + A * t2 + B * A^2 * t2^2))), n * ε
    end
    for B in (0.0, 0.5, 2.0), (n, σ) in ((1.0, 3.0), (1.0, 3.0e4), (1.0e-6, 1.0e-8), (0.01, 1.0e6), (100.0, 1.0e10))
        r = xc_derivatives(with_B(B), Val(1), n, σ)
        e, nε = direct(B, big(n), big(σ))
        h = big(σ) * big(1.0e-25)
        vσ = (first(direct(B, big(n), big(σ) + h)) - first(direct(B, big(n), big(σ) - h))) / 2h
        @test abs(r.e - e) <= 8 * eps() * abs(nε)
        @test abs(r.vsigma - vσ) <= 32 * eps() * abs(vσ)
    end
end
