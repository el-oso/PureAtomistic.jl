# Test-only functionals. Their thresholds are far below every test value, so no floor moves a
# test point.
@testmodule TestFunctionals begin
    using PureXC

    const TINY = PureXC.Thresholds{Float64}(1.0e-200, 1.0e-100, eps(Float64))

    # e = ρ³; polarized, with ρ = ρα + ρβ.
    struct _TestLDA{T} <: PureXC.AbstractLDA{T}
        thresholds::PureXC.Thresholds{T}
    end
    _TestLDA() = _TestLDA{Float64}(TINY)
    PureXC._energy_density(::_TestLDA, ρ) = ρ^3
    PureXC._energy_density(::_TestLDA, ρ::Tuple) = (ρ[1] + ρ[2])^3

    # e = ρ²σ³ + ρσ; polarized, with ρ = ρα + ρβ and σ = σαα + 2σαβ + σββ.
    struct _TestGGA{T} <: PureXC.AbstractGGA{T}
        thresholds::PureXC.Thresholds{T}
    end
    _TestGGA() = _TestGGA{Float64}(TINY)
    _TestGGA(::Type{T}) where {T} = _TestGGA{T}(PureXC.default_thresholds(T, 1.0e-200, 1.0e-100, eps(Float64)))
    _gga(ρ, σ) = ρ^2 * σ^3 + ρ * σ
    PureXC._energy_density(::_TestGGA, ρ, σ) = _gga(ρ, σ)
    PureXC._energy_density(::_TestGGA, ρ::Tuple, σ::Tuple) = _gga(ρ[1] + ρ[2], σ[1] + 2 * σ[2] + σ[3])

    # Layout probes: e = Σ_α w(α) Π (x_i - 1)^α_i over every multi-index α up to degree 3, so at
    # x = (1, …, 1) the partial ∂^α e is D(α) = α!·w(α). The variables are (ρα, ρβ) for the LDA
    # and (ρα, ρβ, σαα, σαβ, σββ) for the GGA.
    struct _LayoutLDA <: PureXC.AbstractLDA{Float64}
        thresholds::PureXC.Thresholds{Float64}
    end
    struct _LayoutGGA <: PureXC.AbstractGGA{Float64}
        thresholds::PureXC.Thresholds{Float64}
    end

    # Every exponent tuple of length n with total degree ≤ 3.
    exponents(n) = [Tuple(a) for a in Iterators.product(ntuple(_ -> 0:3, n)...) if sum(a) <= 3]
    w(α) = 1 + sum(α[i] * 10^(i - 1) for i in eachindex(α))
    D(α) = prod(factorial, α) * w(α)

    function _layout(x::Tuple)
        h = map(xi -> xi - 1, x)
        acc = sum(h) * 0
        for α in exponents(length(x))
            acc += w(α) * prod(h[i]^α[i] for i in eachindex(α))
        end
        return acc
    end
    PureXC._energy_density(::_LayoutLDA, ρ::Tuple) = _layout(ρ)
    PureXC._energy_density(::_LayoutGGA, ρ::Tuple, σ::Tuple) = _layout((ρ..., σ...))

    # Multisets of size m over 1:n as nondecreasing index tuples, in lexicographic order.
    multisets(n, m) = m == 0 ? [()] : [(i, r...) for i in 1:n for r in multisets(n, m - 1) if isempty(r) || r[1] >= i]
    counts(seq, n) = ntuple(i -> count(==(i), seq), n)
end

@testitem "_raise moves the value to the floor and keeps the series" begin
    using PureTaylor
    r = PureXC._raise(variable(Val(1), Val(2), Val(1), 1.0e-20, 1.0e-15), 1.0e-15)
    # Exact by construction: only the stored value is replaced.
    @test value(r) == 1.0e-15
    @test PureTaylor.coefficient(r, (1,)) == 1.0e-15
    @test PureTaylor.coefficient(r, (2,)) == 0.0
    a = PureXC._raise(variable(Val(1), Val(2), Val(1), 2.0, 3.0), 1.0e-15)
    @test a === variable(Val(1), Val(2), Val(1), 2.0, 3.0)
    @test PureXC._raise(1.0e-20, 1.0e-15) == 1.0e-15
    @test PureXC._raise(0.5, 1.0e-15) == 0.5
    @test PureXC._raise(-0.0, 1.0e-15) == 1.0e-15
    @test PureXC._raise(1.0f-20, 1.0f-15) isa Float32
end

@testitem "Thresholds and their Float32 defaults" begin
    t = PureXC.Thresholds{Float64}(1.0e-15, 1.0000000000000027e-20, eps(Float64))
    @test t.density == 1.0e-15
    @test t.sigma == 1.0000000000000027e-20
    @test t.zeta == eps(Float64)
    # Exact: the floor is the square of the reported threshold, rounded once.
    @test t.sigma_floor == 1.0000000000000027e-20^2
    @test t.sigma_floor == 1.0000000000000052e-40
    d = PureXC.default_thresholds(Float64, 1.0e-15, 1.0000000000000027e-20, eps(Float64))
    @test d === t
    f = PureXC.default_thresholds(Float32, 1.0e-15, 1.0000000000000027e-20, eps(Float64))
    @test f isa PureXC.Thresholds{Float32}
    @test f.density == Float32(1.0e-15)
    @test f.sigma == Float32(1.0000000000000027e-20)
    @test f.zeta == eps(Float32)
    # Float32(1e-20)^2 is below floatmin(Float32), so the floor is raised to floatmin.
    @test f.sigma_floor == floatmin(Float32)
    g = PureXC.default_thresholds(Float32, 1.0e-12, 1.0000000000000021e-16, eps(Float64))
    # Float32(1e-16)^2 is normal in Float32, so it is kept.
    @test g.sigma_floor == Float32(1.0000000000000021e-16)^2
    @test PureXC.default_thresholds(Float32, 1.0e-50, 1.0e-3, eps(Float64)).density == floatmin(Float32)
    @test PureXC.default_thresholds(Float32, 1.0e-15, 1.0e-3, eps(Float64)).sigma_floor == 1.0f-3^2
end

@testitem "caller errors throw with the xc prefix" setup = [TestFunctionals] begin
    using .TestFunctionals: _TestLDA, _TestGGA
    lda, gga = _TestLDA(), _TestGGA()
    bad = (NaN, Inf, -Inf)
    for x in bad
        @test_throws "xc: " xc_derivatives(lda, Val(1), x)
        @test_throws "xc: " xc_derivatives(gga, Val(1), x, 1.0)
        @test_throws "xc: " xc_derivatives(gga, Val(1), 1.0, x)
        @test_throws "xc: " energy_density(gga, 1.0, x)
        for i in 1:2
            ρ = ntuple(j -> j == i ? x : 1.0, 2)
            @test_throws "xc: " xc_derivatives(lda, Val(1), ρ)
            @test_throws "xc: " xc_derivatives(gga, Val(1), ρ, (1.0, 0.0, 1.0))
        end
        for i in 1:3
            σ = ntuple(j -> j == i ? x : 1.0, 3)
            @test_throws "xc: " xc_derivatives(gga, Val(1), (1.0, 1.0), σ)
        end
    end
    @test_throws ArgumentError xc_derivatives(lda, Val(0), -1.0e-300)
    @test_throws "xc: " xc_derivatives(lda, Val(0), -1.0e-300)
    @test_throws "xc: " xc_derivatives(lda, Val(0), (-1.0e-300, 1.0))
    @test_throws "xc: " xc_derivatives(lda, Val(0), (1.0, -1.0e-300))
    @test_throws "xc: " xc_derivatives(gga, Val(0), -1.0e-300, 1.0)
    @test_throws "xc: " xc_derivatives(gga, Val(0), 1.0, -1.0e-300)
    @test_throws "xc: " xc_derivatives(gga, Val(0), (-1.0e-300, 1.0), (1.0, 0.0, 1.0))
    @test_throws "xc: " xc_derivatives(gga, Val(0), (1.0, -1.0e-300), (1.0, 0.0, 1.0))
    @test_throws "xc: " xc_derivatives(gga, Val(0), (1.0, 1.0), (-1.0e-300, 0.0, 1.0))
    @test_throws "xc: " xc_derivatives(gga, Val(0), (1.0, 1.0), (1.0, 0.0, -1.0e-300))
    @test_throws "xc: " energy_density(gga, (1.0, 1.0), (1.0, 0.0, -1.0e-300))
    # σαβ may be negative.
    @test xc_derivatives(gga, Val(0), (1.0, 1.0), (1.0, -1.0e-300, 1.0)).e isa Float64
    @test PureXC._check_inputs(gga, (1.0, 1.0), (1.0, -1.0e-300, 1.0)) === nothing
end

@testitem "the σαβ bound" setup = [TestFunctionals] begin
    using .TestFunctionals: _TestGGA
    gga = _TestGGA()
    σαα, σββ = 0.1, 0.2
    p = σαα * σββ
    bound = p * (1 + 16 * eps(Float64))
    # σαβ = −√(σαα σββ) in Float64 is accepted.
    edge = -sqrt(σαα * σββ)
    @test isfinite(xc_derivatives(gga, Val(1), (1.0, 1.0), (σαα, edge, σββ)).e)
    @test isfinite(xc_derivatives(gga, Val(1), (1.0, 1.0), (σαα, -edge, σββ)).e)
    # Inside the bound by a few eps: accepted.
    inside = -sqrt(p) * (1 + 4 * eps(Float64))
    @test inside^2 <= bound
    @test isfinite(xc_derivatives(gga, Val(0), (1.0, 1.0), (σαα, inside, σββ)).e)
    # Just beyond the bound: rejected, for either sign.
    beyond = -sqrt(bound) * (1 + 4 * eps(Float64))
    @test beyond^2 > bound
    @test_throws "xc: " xc_derivatives(gga, Val(0), (1.0, 1.0), (σαα, beyond, σββ))
    @test_throws "xc: " xc_derivatives(gga, Val(0), (1.0, 1.0), (σαα, -beyond, σββ))
    @test_throws "xc: " xc_derivatives(gga, Val(0), (1.0, 1.0), (0.0, 1.0e-3, 1.0))
end

@testitem "ρ = −0.0 is accepted as zero" setup = [TestFunctionals] begin
    using .TestFunctionals: _TestLDA, _TestGGA
    # Exact: every term of the test polynomials vanishes at ρ = 0.
    @test xc_derivatives(_TestLDA(), Val(1), -0.0) == (e = 0.0, vrho = 0.0)
    @test xc_derivatives(_TestGGA(), Val(0), -0.0, 3.0).e == 0.0
    @test xc_derivatives(_TestLDA(), Val(1), (-0.0, -0.0)).vrho == (0.0, 0.0)
    @test energy_density(_TestGGA(), (-0.0, 1.0), (1.0, 0.0, 1.0)) == 10.0
end

@testitem "field names and restricted values" setup = [TestFunctionals] begin
    using .TestFunctionals: _TestLDA, _TestGGA
    gga = _TestGGA()
    names = (
        (:e,),
        (:e, :vrho, :vsigma),
        (:e, :vrho, :vsigma, :v2rho2, :v2rhosigma, :v2sigma2),
        (:e, :vrho, :vsigma, :v2rho2, :v2rhosigma, :v2sigma2, :v3rho3, :v3rho2sigma, :v3rhosigma2, :v3sigma3),
    )
    # e = ρ²σ³ + ρσ at ρ = 2, σ = 3, differentiated by hand. Exact in Float64: every value and
    # every intermediate is a small integer or a product with the seed scales 1/2 and 1/3 that
    # rounds back to the integer.
    want = (
        e = 114.0, vrho = 111.0, vsigma = 110.0, v2rho2 = 54.0, v2rhosigma = 109.0, v2sigma2 = 72.0,
        v3rho3 = 0.0, v3rho2sigma = 54.0, v3rhosigma2 = 72.0, v3sigma3 = 24.0,
    )
    for (k, n) in zip(0:3, names)
        r = xc_derivatives(gga, Val(k), 2.0, 3.0)
        @test keys(r) == n
        @test r == NamedTuple{n}(want)
        p = xc_derivatives(gga, Val(k), (1.0, 1.0), (1.0, 0.25, 1.0))
        @test keys(p) == n
        @test all(x -> x isa Float64, p.e)
        lengths = (
            e = 1, vrho = 2, vsigma = 3, v2rho2 = 3, v2rhosigma = 6, v2sigma2 = 6,
            v3rho3 = 4, v3rho2sigma = 9, v3rhosigma2 = 12, v3sigma3 = 10,
        )
        for name in n[2:end]
            @test getfield(p, name) isa NTuple{getfield(lengths, name), Float64}
        end
    end
    @test energy_density(gga, 2.0, 3.0) == 114.0
    # e = ρ³ at ρ = 2.
    lda = _TestLDA()
    @test xc_derivatives(lda, Val(3), 2.0) == (e = 8.0, vrho = 12.0, v2rho2 = 12.0, v3rho3 = 6.0)
    @test keys(xc_derivatives(lda, Val(3), (1.0, 1.0))) == (:e, :vrho, :v2rho2, :v3rho3)
    @test xc_derivatives(lda, Val(2), (1.0, 1.0)) == (e = 8.0, vrho = (12.0, 12.0), v2rho2 = (12.0, 12.0, 12.0))
    @test energy_density(lda, 2.0) == 8.0
    @test PureXC._xc_derivatives_unchecked(gga, Val(2), 2.0, 3.0) == xc_derivatives(gga, Val(2), 2.0, 3.0)
end

@testitem "polarized layout" setup = [TestFunctionals] begin
    using .TestFunctionals: _LayoutLDA, _LayoutGGA, TINY, D, exponents, multisets, counts
    # Every partial up to order 3 has its own value, so a misplaced entry is detected.
    @test allunique(D.(exponents(5)))
    @test allunique(D.(exponents(2)))
    # Exact: the seeds are 1 + u_i, every coefficient is an integer and the scales are 1.
    r = xc_derivatives(_LayoutGGA(TINY), Val(3), (1.0, 1.0), (1.0, 1.0, 1.0))
    # Order 1 and 2 written out from the layout rule.
    @test r.e == D((0, 0, 0, 0, 0))
    @test r.vrho == D.(((1, 0, 0, 0, 0), (0, 1, 0, 0, 0)))
    @test r.vsigma == D.(((0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)))
    @test r.v2rho2 == D.(((2, 0, 0, 0, 0), (1, 1, 0, 0, 0), (0, 2, 0, 0, 0)))
    # (α·αα, α·αβ, α·ββ, β·αα, β·αβ, β·ββ)
    @test r.v2rhosigma == D.(
        (
            (1, 0, 1, 0, 0), (1, 0, 0, 1, 0), (1, 0, 0, 0, 1),
            (0, 1, 1, 0, 0), (0, 1, 0, 1, 0), (0, 1, 0, 0, 1),
        )
    )
    # (αα·αα, αα·αβ, αα·ββ, αβ·αβ, αβ·ββ, ββ·ββ)
    @test r.v2sigma2 == D.(
        (
            (0, 0, 2, 0, 0), (0, 0, 1, 1, 0), (0, 0, 1, 0, 1),
            (0, 0, 0, 2, 0), (0, 0, 0, 1, 1), (0, 0, 0, 0, 2),
        )
    )
    # Order 3 from the rule: ρ multisets outer, σ multisets inner, each in lexicographic order.
    for (name, a) in ((:v3rho3, 3), (:v3rho2sigma, 2), (:v3rhosigma2, 1), (:v3sigma3, 0))
        want = Tuple(
            D((counts(ρs, 2)..., counts(σs, 3)...)) for ρs in multisets(2, a) for σs in multisets(3, 3 - a)
        )
        @test getfield(r, name) == want
    end
    l = xc_derivatives(_LayoutLDA(TINY), Val(3), (1.0, 1.0))
    @test l.e == D((0, 0))
    @test l.vrho == D.(((1, 0), (0, 1)))
    @test l.v2rho2 == D.(((2, 0), (1, 1), (0, 2)))
    @test l.v3rho3 == D.(((3, 0), (2, 1), (1, 2), (0, 3)))
end

@testitem "result types follow the inputs" setup = [TestFunctionals] begin
    using .TestFunctionals: _TestGGA, _TestLDA
    @test (@inferred xc_derivatives(_TestGGA(), Val(2), 2.0f0, 3.0f0)).v2sigma2 isa Float64
    r = @inferred xc_derivatives(_TestGGA(Float32), Val(3), 2.0f0, 3.0f0)
    @test all(x -> x isa Float32, values(r))
    p = @inferred xc_derivatives(_TestGGA(Float32), Val(2), (1.0f0, 2.0f0), (0.1f0, 0.0f0, 0.2f0))
    @test p.v2rhosigma isa NTuple{6, Float32}
    @test (@inferred xc_derivatives(_TestGGA(Float32), Val(1), 2.0, 3.0f0)).vrho isa Float64
    @test (@inferred xc_derivatives(_TestLDA(), Val(1), (1, 2))).vrho isa NTuple{2, Float64}
    @test (@inferred energy_density(_TestLDA(), 2.0f0)) isa Float64
end

# A functional whose cut-off rule holds below a total density gives an all-zero result of the
# full type there, whatever its energy expression gives.
@testitem "cut-off points give exact zeros" setup = [TestFunctionals] begin
    using PureXC
    struct _CutGGA <: PureXC.AbstractGGA{Float64}
        thresholds::PureXC.Thresholds{Float64}
    end
    PureXC._cutoff(f::_CutGGA, ρ, σ) = ρ < f.thresholds.density
    PureXC._cutoff(f::_CutGGA, ρ::Tuple, σ::Tuple) = ρ[1] + ρ[2] < f.thresholds.density
    PureXC._energy_density(::_CutGGA, ρ, σ) = σ / ρ
    PureXC._energy_density(::_CutGGA, ρ::Tuple, σ::Tuple) = σ[1] / ρ[1] + σ[3] / ρ[2]
    f = _CutGGA(PureXC.Thresholds{Float64}(1.0e-10, 1.0e-20, eps(Float64)))
    z = xc_derivatives(f, Val(3), 0.0, 1.0)
    @test all(iszero, values(z))
    @test z isa typeof(xc_derivatives(f, Val(3), 1.0, 1.0))
    zp = xc_derivatives(f, Val(2), (0.0, -0.0), (1.0, 0.0, 1.0))
    @test all(x -> all(iszero, x), values(zp))
    @test zp isa typeof(xc_derivatives(f, Val(2), (1.0, 1.0), (1.0, 0.0, 1.0)))
    # Exact: 1/1 = 1 and its derivatives in ρ are integers.
    @test xc_derivatives(f, Val(1), 1.0, 1.0) == (e = 1.0, vrho = -1.0, vsigma = 1.0)
end

@testitem "xc_derivatives allocates nothing" setup = [TestFunctionals] begin
    using StrictModeTest
    using .TestFunctionals: _TestGGA
    f = _TestGGA()
    xc_derivatives(f, Val(2), (1.0, 2.0), (0.1, 0.0, 0.2))
    # @test_noalloc throws a StrictViolation unless AllocCheck proves the call allocation-free.
    @test (@test_noalloc xc_derivatives(f, Val(2), (1.0, 2.0), (0.1, 0.0, 0.2))) isa NamedTuple
    xc_derivatives(f, Val(3), 2.0, 3.0)
    @test (@test_noalloc xc_derivatives(f, Val(3), 2.0, 3.0)) isa NamedTuple
end
