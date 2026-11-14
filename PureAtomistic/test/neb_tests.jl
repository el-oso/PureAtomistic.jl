@testmodule NEBFixtures begin
    using PureAtomistic
    using PureAtomistic.Potentials
    using LinearAlgebra

    # The Müller–Brown surface (Müller and Brown, Theor. Chim. Acta 53, 75, 1979): its energy and
    # gradient at (x, y).
    const MA = (-200.0, -100.0, -170.0, 15.0)
    const Ma = (-1.0, -1.0, -6.5, 0.7)
    const Mb = (0.0, 0.0, 11.0, 0.6)
    const Mc = (-10.0, -10.0, -6.5, 0.7)
    const Mx = (1.0, 0.0, -0.5, -1.0)
    const My = (0.0, 0.5, 1.5, 1.0)
    function muller_brown(x::T, y::T) where {T}
        E, gx, gy = zero(T), zero(T), zero(T)
        for k in 1:4
            dx, dy = x - T(Mx[k]), y - T(My[k])
            e = T(MA[k]) * exp(T(Ma[k]) * dx^2 + T(Mb[k]) * dx * dy + T(Mc[k]) * dy^2)
            E += e
            gx += e * (2 * T(Ma[k]) * dx + T(Mb[k]) * dy)
            gy += e * (T(Mb[k]) * dx + 2 * T(Mc[k]) * dy)
        end
        return E, gx, gy
    end

    # Atom 1 moves on the Müller–Brown surface in (x, y). With `extra`, a second atom feels the
    # constant force -(3, 3, 3) and atom 1 the constant force -5 along z; both terms vanish at the
    # origin, where the bands of the tests start them.
    struct MullerBrown{T} <: AbstractPotential
        result::PotentialResult{T}
        extra::Bool
    end
    MullerBrown{T}(extra::Bool = false) where {T} = MullerBrown{T}(PotentialResult{T}(extra ? 2 : 1), extra)
    Potentials.n_atoms(p::MullerBrown) = p.extra ? 2 : 1
    Potentials.cutoff(::MullerBrown) = 0.0
    Potentials.prepare!(::MullerBrown, ::AbstractMatrix, ::AbstractMatrix) = nothing
    function Potentials.energy_forces!(p::MullerBrown{T}, x::AbstractMatrix, ::AbstractMatrix) where {T}
        E, gx, gy = muller_brown(x[1, 1], x[2, 1])
        F = p.result.forces
        F .= 0
        F[1, 1], F[2, 1] = -gx, -gy
        if p.extra
            E += 5 * x[3, 1] + 3 * (x[1, 2] + x[2, 2] + x[3, 2])
            F[3, 1] = -5
            F[:, 2] .= -3
        end
        p.result.energy = E
        return nothing
    end

    # The stationary point of the surface near p by Newton's method on the analytic gradient,
    # with a central-difference Hessian, and that Hessian at the point.
    function hessian(p)
        h = 1.0e-5
        H = zeros(2, 2)
        for j in 1:2
            q = copy(p)
            q[j] += h
            _, a1, a2 = muller_brown(q...)
            q[j] -= 2h
            _, b1, b2 = muller_brown(q...)
            H[:, j] = ([a1, a2] - [b1, b2]) / 2h
        end
        return Symmetric((H + H') / 2)
    end
    function stationary(p)
        for _ in 1:50
            _, gx, gy = muller_brown(p...)
            p = p - hessian(p) \ [gx, gy]
        end
        return p
    end

    const A = stationary([-0.558, 1.442])    # deepest minimum
    const B = stationary([0.623, 0.028])     # second minimum
    const S1 = stationary([-0.822, 0.624])   # the higher saddle on the path from A to B

    # M + 2 images on the straight line from minimum A to minimum B, atom 1 at z = 0 and, with
    # `extra`, atom 2 at the origin.
    function band(::Type{T}, M; extra = false) where {T}
        return [
            T.(hcat([A[1] + t * (B[1] - A[1]), A[2] + t * (B[2] - A[2]), 0.0], (extra ? (zeros(3),) : ())...))
                for t in range(0, 1; length = M + 2)
        ]
    end

    # FIRE with steps suited to the surface: its curvature reaches about 10⁴, so dt stays near
    # 2/√10⁴, and its features span tenths of a unit, so no move exceeds 0.01.
    fire(::Type{T}, n) where {T} = FIRE{T}(n; dt = 0.002, dtmax = 0.02, maxstep = 0.01)
    fire2(::Type{T}, n) where {T} = FIRE2{T}(n; dt = 0.002, dtmax = 0.02, dtmin = 4.0e-5, maxstep = 0.01)

    const CELL = Matrix(10.0I, 3, 3)
    distances(r) = [norm(r.images[j + 1] - r.images[j]) for j in 1:(length(r.images) - 1)]
end

@testitem "neb oracle: the Müller–Brown stationary points" setup = [NEBFixtures] begin
    using LinearAlgebra
    N = NEBFixtures
    # Oracle: the values Müller and Brown tabulate, to the three decimals they give.
    @test maximum(abs, N.A - [-0.558, 1.442]) < 1.0e-3
    @test maximum(abs, N.B - [0.623, 0.028]) < 1.0e-3
    @test maximum(abs, N.S1 - [-0.822, 0.624]) < 1.0e-3
    @test abs(N.muller_brown(N.S1...)[1] - -40.665) < 1.0e-3
    λ = eigvals(N.hessian(N.S1))
    @test λ[1] < 0 < λ[2]
end

@testitem "neb climbing image finds the Müller–Brown saddle point" setup = [NEBFixtures] begin
    using LinearAlgebra
    N = NEBFixtures
    fmax = 1.0e-3
    k = 50.0
    ES = N.muller_brown(N.S1...)[1]
    # Near the saddle the climbing force is the gradient reflected along the band, of the same
    # norm, so |∇V| < fmax puts the image within fmax / min |λ| of S1 to first order; the factor 2
    # covers the second-order term.
    λ = abs.(eigvals(N.hessian(N.S1)))
    bound = 2 * fmax / minimum(λ)
    for M in (7, 9), make in (N.fire, N.fire2)
        imgs = N.band(Float64, M)
        before = deepcopy(imgs)
        r = neb(imgs, N.CELL, N.MullerBrown{Float64}(); spring = k, fmax, maxsteps = 5000, opt = make(Float64, M))
        @test imgs == before
        @test r.converged
        @test r.max_force < fmax
        @test 0 < r.steps < 5000
        @test r.climbing == argmax(r.energies)
        @test r.images[1] == imgs[1] && r.images[end] == imgs[end]
        c = r.images[r.climbing]
        @test norm(c[1:2, 1] - N.S1) < bound
        @test c[3, 1] == 0
        # Energy: second order in the distance, ½ max |λ| bound².
        @test abs(r.energies[r.climbing] - ES) < maximum(λ) * bound^2 / 2
        # Each energy is the potential's at its image.
        @test r.energies == [N.muller_brown(x[1, 1], x[2, 1])[1] for x in r.images]
        # Along the tangent a non-climbing image's band force is the spring force alone, so
        # |k (d₊ - d₋)| ≤ |F| < fmax: neighbors on each side of the climbing image are equally spaced.
        d = N.distances(r)
        for j in 2:(M + 1)
            j == r.climbing && continue
            @test abs(d[j] - d[j - 1]) < fmax / k
        end
    end
end

@testitem "neb without climbing spaces the images equally below the saddle" setup = [NEBFixtures] begin
    N = NEBFixtures
    fmax = 1.0e-3
    k = 50.0
    for T in (Float64, Float32)
        # Float32: the surface's energies of order 100 round at 100 eps(Float32) ≈ 1e-5 and its
        # gradients of order 10³ at 1e-4, well below fmax = 1e-2.
        f = T === Float64 ? fmax : 1.0e-2
        r = neb(N.band(T, 9), T.(N.CELL), N.MullerBrown{T}(); spring = k, climb = false, fmax = f, maxsteps = 5000, opt = N.fire(T, 9))
        @test r.converged
        @test r.climbing == 0
        @test r isa NEBResult{T}
        # Without a climbing image every interior image carries a spring (see the climbing item).
        d = N.distances(r)
        for j in 2:length(d)
            @test abs(d[j] - d[j - 1]) < f / k
        end
        # The band's highest image lies below the saddle point: what the climbing image corrects.
        # Measured: 2.27 below in Float64 and 2.32 in Float32; the bound 0.1 is far above the
        # Float32 rounding of 1e-5.
        @test maximum(r.energies) < N.muller_brown(N.S1...)[1] - 0.1
    end
end

@testitem "neb tangent weights" begin
    w = PureAtomistic.Integrators._tangent_weights
    # Rising and falling: the uphill neighbor alone.
    @test w(1.0, 2.0, 3.0) == (1.0, 0.0)
    @test w(3.0, 2.0, 1.0) == (0.0, 1.0)
    # A maximum or minimum: the larger energy difference weighs the higher neighbor.
    @test w(1.0, 4.0, 2.0) == (3.0, 2.0)
    @test w(2.0, 4.0, 1.0) == (2.0, 3.0)
    @test w(3.0, 1.0, 2.0) == (1.0, 2.0)
    @test w(2.0, 1.0, 3.0) == (2.0, 1.0)
    # An equal neighbor: not strictly rising, so the extremum rule, which here weighs only the
    # higher neighbor.
    @test w(1.0, 2.0, 2.0) == (1.0, 0.0)
    # Flat: both differences equally.
    @test w(1.0, 1.0, 1.0) == (1.0, 1.0)
end

@testitem "neb keeps fixed atoms and frozen directions" setup = [NEBFixtures] begin
    N = NEBFixtures
    M = 7
    ref = neb(N.band(Float64, M), N.CELL, N.MullerBrown{Float64}(); spring = 50.0, fmax = 1.0e-3, maxsteps = 5000, opt = N.fire(Float64, M))
    imgs = N.band(Float64, M; extra = true)
    p = N.MullerBrown{Float64}(true)
    r = neb(imgs, N.CELL, p; spring = 50.0, fmax = 1.0e-3, maxsteps = 5000, fixed = [2], frozen = [1 => (0, 0, 2)], opt = N.fire(Float64, 2M))
    @test r.converged
    # The constrained components are zero and add exact zeros to every sum, so the band of atom 1
    # is the unconstrained one bit for bit.
    @test [x[:, 1] for x in r.images] == [x[:, 1] for x in ref.images]
    @test r.energies == ref.energies
    @test all(x -> x[:, 2] == zeros(3), r.images)
    # Unconstrained, the constant forces never balance.
    free = neb(imgs, N.CELL, p; spring = 50.0, fmax = 1.0e-3, maxsteps = 50, opt = N.fire(Float64, 2M))
    @test !free.converged && free.steps == 50
    @test free.max_force > 3
end

@testitem "neb limits and the FIRE2 uphill stop" setup = [NEBFixtures] begin
    N = NEBFixtures
    imgs = N.band(Float64, 3)
    p = N.MullerBrown{Float64}()
    r0 = neb(imgs, N.CELL, p; spring = 50.0, maxsteps = 0)
    @test !r0.converged && r0.steps == 0 && r0.images == imgs
    @test r0.max_force > 1
    # The default optimizer is FIRE with its defaults: the same as passing it.
    @test neb(imgs, N.CELL, p; spring = 50.0, maxsteps = 3).images ==
        neb(imgs, N.CELL, p; spring = 50.0, maxsteps = 3, opt = FIRE{Float64}(3)).images
    # The first FIRE2 step has zero power, which exceeds nneg_max = 0 and stops the band unmoved.
    r = neb(imgs, N.CELL, p; spring = 50.0, opt = FIRE2{Float64}(3; nneg_max = 0))
    @test !r.converged && r.steps == 0 && r.images == imgs
    # The state is reset by every call.
    opt = N.fire(Float64, 3)
    a = neb(imgs, N.CELL, p; spring = 50.0, maxsteps = 20, opt)
    b = neb(imgs, N.CELL, p; spring = 50.0, maxsteps = 20, opt)
    @test a.images == b.images && a.energies == b.energies
end

@testitem "neb rejects invalid input" setup = [NEBFixtures, RelaxFixtures] begin
    N = NEBFixtures
    imgs = N.band(Float64, 3)
    p = N.MullerBrown{Float64}()
    c = N.CELL
    @test_throws "pureatomistic: neb needs at least three images (two endpoints and one to move), got 2" neb(imgs[1:2], c, p)
    @test_throws "pureatomistic: every image must have size (3, 1), got (3, 2)" neb([imgs[1], zeros(3, 2), imgs[3]], c, p)
    @test_throws "pureatomistic: the potential has 2 atoms, the images 1" neb(imgs, c, N.MullerBrown{Float64}(true))
    @test_throws "pureatomistic: the spring constant must be positive and finite, got 0.0" neb(imgs, c, p; spring = 0.0)
    @test_throws "pureatomistic: fmax must be positive and finite, got -1.0" neb(imgs, c, p; fmax = -1.0)
    @test_throws "pureatomistic: maxsteps must not be negative, got -1" neb(imgs, c, p; maxsteps = -1)
    @test_throws "pureatomistic: the optimizer state has 2 atoms, the band moves 3" neb(imgs, c, p; opt = FIRE{Float64}(2))
    @test_throws "pureatomistic: fixed atom 2 is outside 1:1" neb(imgs, c, p; fixed = [2])
    @test_throws "pureatomistic: atom 1 is both fixed and frozen" neb(imgs, c, p; fixed = [1], frozen = [1 => (1, 0, 0)])
    same = [copy(imgs[1]) for _ in 1:3]
    @test_throws "pureatomistic: neb image 2 has a zero tangent: its neighbors coincide with it" neb(same, c, p)
    nanp = RelaxFixtures.NaNPotential(PotentialResult{Float64}(1))
    @test_throws "pureatomistic: neb met a non-finite energy or force at image 1 after 0 steps" neb(imgs, c, nanp)
end

@testitem "neb on views and OffsetArrays" setup = [NEBFixtures] begin
    using OffsetArrays
    N = NEBFixtures
    imgs = N.band(Float64, 5)
    p = N.MullerBrown{Float64}()
    kw = (; spring = 50.0, fmax = 1.0e-3, maxsteps = 5000)
    ref = neb(imgs, N.CELL, p; kw..., opt = N.fire(Float64, 5))
    # Views of a larger parent and an offset vector of images: the same arithmetic, so equality
    # is exact.
    parent = zeros(3, 2, length(imgs))
    for j in eachindex(imgs)
        parent[:, 1, j] .= imgs[j]
    end
    views = [view(parent, :, 1:1, j) for j in eachindex(imgs)]
    @test neb(views, N.CELL, p; kw..., opt = N.fire(Float64, 5)).images == ref.images
    @test parent[:, 2, :] == zeros(3, length(imgs))
    @test neb(OffsetArray(imgs, -3), N.CELL, p; kw..., opt = N.fire(Float64, 5)).images == ref.images
    off = [OffsetArray(x, 1:3, 0:0) for x in imgs]
    @test_throws "offset arrays are not supported" neb(off, N.CELL, p)
end

@testitem "TrimCheck: neb" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureAtomistic, PureMLIP
            const PP = PureAtomistic.Potentials
            const IN = PureAtomistic.Integrators
            neb64(i, c, p) = IN.neb(i, c, p; fmax = 1.0e-3, maxsteps = 5)
            neb32(i, c, p) = IN.neb(i, c, p; fmax = 1.0e-3, maxsteps = 5)
            neb2_64(i, c, p, o) = IN.neb(i, c, p; climb = false, fixed = [1], opt = o)
            neb2_32(i, c, p, o) = IN.neb(i, c, p; climb = false, fixed = [1], opt = o)
        end,
        neb64(Vector{Matrix{Float64}}, Matrix{Float64}, PP.LennardJones{Float64}),
        neb32(Vector{Matrix{Float32}}, Matrix{Float32}, PP.LennardJones{Float32}),
        neb64(Vector{Matrix{Float64}}, Matrix{Float64}, PP.MACEPotential{Float64}),
        neb32(Vector{Matrix{Float32}}, Matrix{Float32}, PP.MACEPotential{Float32}),
        neb2_64(Vector{Matrix{Float64}}, Matrix{Float64}, PP.MACEPotential{Float64}, IN.FIRE2{Float64}),
        neb2_32(Vector{Matrix{Float32}}, Matrix{Float32}, PP.MACEPotential{Float32}, IN.FIRE2{Float32}),
    )
end

@testitem "neb return types are concrete" tags = [:checks] setup = [RelaxFixtures] begin
    F = RelaxFixtures
    for T in (Float64, Float32)
        sys, _, p = F.lj_crystal(T)
        I = Vector{Matrix{T}}
        @test Base.return_types(neb, (I, typeof(sys.cell), typeof(p)))[1] === NEBResult{T}
        f(i, c, p, o) = neb(i, c, p; opt = o)
        @test Base.return_types(f, (I, typeof(sys.cell), typeof(p), FIRE2{T}))[1] === NEBResult{T}
        @test isconcretetype(NEBResult{T})
    end
end

# The band loop evaluates every interior image per step; after the band is built it allocates
# nothing, the potential's evaluation included. The potential's neighbor-list rebuild is a
# workspace path that a fixed band never takes (checked by `n_regrows`).
@testitem "neb band loop allocates nothing" tags = [:checks] setup = [RelaxFixtures] begin
    using StrictModeTest, StrictMode
    using PureAtomistic.Integrators: FIRE, FIRE2, _NEBBand, _neb!, _fixed_mask, _frozen_rows
    register_alloc_barrier!(PureAtomistic.Potentials._regrow!)
    @noinline loop!(b, cell, p, opt, tol) = _neb!(b, cell, p, opt, tol, 3)
    for T in (Float64, Float32)
        sys, perfect, lj = RelaxFixtures.lj_crystal(T)
        n, m = size(perfect, 2), 3
        X = Matrix{T}(undef, 3, n * (m + 2))
        for j in 0:(m + 1)
            X[:, (j * n + 1):((j + 1) * n)] .= perfect .+ (T(j) / (m + 1)) .* (sys.positions .- perfect)
        end
        isfixed = _fixed_mask(n, Int[])
        atoms, dirs = _frozen_rows(T, n, isfixed, Pair{Int, NTuple{3, Float64}}[])
        for opt in (FIRE{T}(n * m), FIRE2{T}(n * m))
            band = _NEBBand{T}(X, n, m, T(0.1), true, isfixed, atoms, dirs)
            loop!(band, sys.cell, lj, opt, T(1.0e-6))
            @test_noalloc loop!(band, sys.cell, lj, opt, T(1.0e-6))
            @test lj.n_regrows == 0
        end
    end
end
