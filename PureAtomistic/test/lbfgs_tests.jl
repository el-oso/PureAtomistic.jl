@testmodule LBFGSFixtures begin
    using LinearAlgebra, Random

    # The inverse BFGS matrix of the pairs (columns of S and Y, oldest first) from H₀ = γI, by the
    # dense update H ← (I - ρ s yᵀ) H (I - ρ y sᵀ) + ρ s sᵀ, in Float64.
    function dense_inverse_bfgs(S, Y, γ)
        n = size(S, 1)
        H = Matrix{Float64}(γ * I, n, n)
        for k in axes(S, 2)
            s, y = Float64.(S[:, k]), Float64.(Y[:, k])
            ρ = 1 / dot(s, y)
            V = I - ρ * y * s'
            H = V' * H * V + ρ * s * s'
        end
        return Symmetric(H)
    end

    # Curvature pairs y = A s of a symmetric positive definite A with eigenvalues in [lo, hi], with
    # the components on `frozen` (a list of coordinates) removed from both.
    function pairs_of(::Type{T}, n, k, lo, hi; seed = 1, zero_rows = Int[]) where {T}
        rng = Random.Xoshiro(seed)
        Q = Matrix(qr(randn(rng, n, n)).Q)
        A = Q * Diagonal(range(lo, hi; length = n)) * Q'
        S = randn(rng, n, k)
        S[zero_rows, :] .= 0
        Y = A * S
        Y[zero_rows, :] .= 0
        return T.(S), T.(Y)
    end
end

@testitem "TrustRegionLBFGS two-loop recursion equals the dense inverse BFGS matrix" setup = [LBFGSFixtures] begin
    using LinearAlgebra
    using PureAtomistic.Integrators: TrustRegionLBFGS, _two_loop!, _curvature_update!
    F = LBFGSFixtures
    for T in (Float64, Float32)
        n_at = 4
        n = 3n_at
        opt = TrustRegionLBFGS{T}(n_at; memory = 3)
        S, Y = F.pairs_of(T, n, 5, 1.0, 10.0)
        for k in 1:5
            @test _curvature_update!(opt, S[:, k], Y[:, k])
        end
        # Five pairs into a memory of three: the oldest two are gone, and γ is that of the newest.
        @test opt.npairs == 3
        @test opt.gamma == dot(S[:, 5], Y[:, 5]) / dot(Y[:, 5], Y[:, 5])
        H = F.dense_inverse_bfgs(S[:, 3:5], Y[:, 3:5], Float64(opt.gamma))
        v = T.(sin.(1.0:n))
        r = similar(v)
        _two_loop!(r, opt, v)
        # Rounding bound: c = 10, N = n · m · κ(H), with κ(H) from the dense matrix (3.4 here).
        # Measured relative error: 2.3e-16 (Float64), 1.1e-7 (Float32).
        ref = H * Float64.(v)
        κ = cond(Matrix(H))
        @test norm(r - ref) <= 10 * eps(T) * n * 3 * κ * norm(ref)
        # A pair failing the curvature test is not stored.
        @test !_curvature_update!(opt, S[:, 1], -Y[:, 1])
        @test opt.npairs == 3
    end
end

@testitem "TrustRegionLBFGS step inside the box is the two-loop step" setup = [LBFGSFixtures] begin
    using LinearAlgebra
    using PureAtomistic.Integrators: TrustRegionLBFGS, _trust_step!, _curvature_update!
    F = LBFGSFixtures
    for T in (Float64, Float32)
        n_at = 4
        n = 3n_at
        opt = TrustRegionLBFGS{T}(n_at; memory = 4, radius = 10.0, max_radius = 10.0)
        S, Y = F.pairs_of(T, n, 4, 1.0, 10.0; seed = 3)
        for k in 1:4
            _curvature_update!(opt, S[:, k], Y[:, k])
        end
        opt.grad .= reshape(T.(cos.(1.0:n)), 3, n_at)
        pred = _trust_step!(opt)
        H = F.dense_inverse_bfgs(S, Y, Float64(opt.gamma))
        g = Float64.(vec(opt.grad))
        ref = -(H * g)
        # Bound as for the two-loop recursion: c = 10, N = n · m · κ(H). Measured relative error
        # 3.5e-16 (Float64), 8.7e-8 (Float32).
        κ = cond(Matrix(H))
        tol = 10 * eps(T) * n * 4 * κ
        @test norm(vec(opt.step) - ref) <= tol * norm(ref)
        @test opt.boxed == 0
        # The model's decrease at -Hg is ½ gᵀHg, evaluated as -½ gᵀp: its error is at most
        # |g| |p - p*| ≤ tol |g| |Hg|, and gᵀHg ≥ |g| |Hg| / κ(H), so the relative bound is tol κ(H).
        # Measured 1.4e-16 (Float64), 5.1e-8 (Float32).
        @test pred ≈ dot(g, H * g) / 2 rtol = tol * κ
    end
end

@testitem "TrustRegionLBFGS box-bound step solves the subspace problem (KKT oracle)" setup = [LBFGSFixtures] begin
    using LinearAlgebra
    using PureDAQP: ActiveSet
    using PureAtomistic.Integrators: TrustRegionLBFGS, _trust_step!, _curvature_update!
    F = LBFGSFixtures
    for T in (Float64, Float32), solver in (ActiveSet(; working_set = :gram), ActiveSet(; working_set = :rows))
        n_at = 8
        n = 3n_at
        Δ = T(0.05)
        opt = TrustRegionLBFGS{T}(n_at; memory = 3, radius = Δ, fixed = [3], solver)
        fixedrows = 7:9
        S, Y = F.pairs_of(T, n, 3, 1.0, 10.0; seed = 5, zero_rows = fixedrows)
        for k in 1:3
            _curvature_update!(opt, S[:, k], Y[:, k])
        end
        g = T.(sin.(1.0:n) .* 2)
        g[fixedrows] .= 0
        opt.grad .= reshape(g, 3, n_at)
        pred = _trust_step!(opt)
        @test opt.boxed == 1
        p = Float64.(vec(opt.step))
        g64 = Float64.(g)
        B = inv(F.dense_inverse_bfgs(S, Y, Float64(opt.gamma)))
        # The subspace V = span{g, s, y}, orthonormalized in Float64 independently of the relaxer.
        Q = Matrix(qr(Float64.([g S Y])).Q)
        # Rounding bound for residuals: c = 10, N = n · κ(B), on gradients of size 2. Measured
        # worst: 1.2e-8 off V and 2.4e-15 in the stationarity residual (Float32).
        tol = 10 * eps(T) * n * cond(Matrix(B)) * 2
        @test all(abs.(p) .<= Δ * (1 + 100eps(T)))
        @test all(iszero, p[fixedrows])
        # p lies in V.
        @test norm(p - Q * (Q' * p)) <= tol
        # Stationarity within V: Qᵀ(g + Bp + μ) = 0 with μ supported on the coordinates at the
        # box, μᵢ ≥ 0 at the upper bound and ≤ 0 at the lower one.
        active = findall(i -> abs(p[i]) >= Δ * (1 - 100eps(T)), 1:n)
        @test !isempty(active) && length(active) < n - 3
        r = Q' * (g64 + B * p)
        μ = (Q[active, :]') \ (-r)
        @test norm(Q[active, :]' * μ + r) <= tol
        @test all(sign.(μ) .* sign.(p[active]) .>= 0 .|| abs.(μ) .<= tol)
        # The model decrease, evaluated as -(qᵀw + ½ wᵀw) in the solver's variables: the relative
        # bound derived for TrustRegionBFGS's step, c = 10, N = n κ(B) (Σ|qⱼwⱼ| + wᵀw) / |pred|.
        # Measured 4.1e-16 (Float64), 3.7e-7 (Float32).
        model(x) = dot(g64, x) + dot(x, B * x) / 2
        q, w = Float64.(opt.q), Float64.(opt.qp.x)
        amp = (sum(abs.(q .* w)) + sum(abs2, w)) / abs(pred)
        @test pred ≈ -model(p) rtol = 10 * eps(T) * n * cond(Matrix(B)) * amp
        # At least the decrease of the steepest-descent point inside the box (measured 1.09 against
        # 1.02 for this draw).
        t = min(dot(g64, g64) / dot(g64, B * g64), Float64(Δ) / maximum(abs, g64))
        @test -model(p) >= -model(-t * g64) * (1 - tol)
    end
end

@testitem "TrustRegionLBFGS box step over a one-dimensional span" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: TrustRegionLBFGS, _trust_step!, _curvature_update!
    # g, s and y all along x: the span is one-dimensional, the two dependent vectors are dropped.
    # H = 1/2 on that line (γ = sᵀy / yᵀy = 1/2), so the model is p + p², whose minimizer -1/2 the
    # box clamps to the radius 0.01: p = -0.01, decrease 0.01 - 0.0001.
    opt = TrustRegionLBFGS{Float64}(1; radius = 0.01)
    @test _curvature_update!(opt, [0.1, 0.0, 0.0], [0.2, 0.0, 0.0])
    opt.grad .= reshape([1.0, 0.0, 0.0], 3, 1)
    pred = _trust_step!(opt)
    @test opt.boxed == 1 && opt.rank == 1
    # Rounding bound: a few operations on values of order one, c = 8, N = 1.
    @test opt.step ≈ reshape([-0.01, 0.0, 0.0], 3, 1) rtol = 8 * eps()
    @test pred ≈ 0.0099 rtol = 8 * eps()
end

@testitem "TrustRegionLBFGS relaxes a Lennard-Jones crystal" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: TrustRegionLBFGS
    F = RelaxFixtures
    # Oracle as for FIRE: the undisplaced lattice is an equilibrium of the fixed cell, and a fresh
    # potential gives the energy and forces at the returned positions. Measured at fmax = 1e-3:
    # 8 steps against FIRE's 41 in both precisions; E - E_lattice = 9.5e-5 eV (Float64) and
    # -9.5e-5 eV (Float32); the bound 1e-3 on either side is 10.5 times that.
    for T in (Float64, Float32)
        sys, perfect, p = F.lj_crystal(T)
        E0 = F.energy_and_forces(F.lj_crystal(T)[3], perfect, sys.cell)[1]
        info = relax!(sys, p, TrustRegionLBFGS(sys); fmax = 1.0e-3)
        @test info.converged
        @test info isa RelaxResult{T}
        Eref, Fref = F.energy_and_forces(F.lj_crystal(T)[3], sys.positions, sys.cell)
        @test F.max_force(Fref) < 1.0e-3
        @test info.max_force ≈ F.max_force(Fref) rtol = 8 * eps(T)
        @test info.energy == p.result.energy
        @test info.energy ≈ Eref rtol = 4 * eps(T)
        @test -1.0e-3 < info.energy - E0 < 1.0e-3
        fire_sys, _, fire_p = F.lj_crystal(T)
        @test info.steps < relax!(fire_sys, fire_p, FIRE(fire_sys); fmax = 1.0e-3).steps
    end
end

@testitem "TrustRegionLBFGS keeps fixed atoms and frozen directions" setup = [RelaxFixtures] begin
    using LinearAlgebra
    using PureAtomistic.Integrators: TrustRegionLBFGS
    F = RelaxFixtures
    sys, _, p = F.lj_crystal(Float64)
    start = copy(sys.positions)
    d = [1.0, 2.0, 2.0] / 3
    opt = TrustRegionLBFGS(sys; memory = 5, fixed = [1, 7], frozen = [4 => d])
    info = relax!(sys, p, opt; fmax = 1.0e-3)
    @test info.converged
    @test sys.positions[:, [1, 7]] == start[:, [1, 7]]
    # Each step's component along d is removed after it is formed, so what remains is the rounding
    # of adding the steps to coordinates of size |x| ≤ 3 Å (atom 4): c = 4, N = steps · |x|.
    @test abs(dot(sys.positions[:, 4] - start[:, 4], d)) < 4 * eps() * info.steps * 3
    Fref = F.energy_and_forces(F.lj_crystal(Float64)[3], sys.positions, sys.cell)[2]
    free = Fref[:, setdiff(1:size(Fref, 2), [1, 7])]
    free[:, 4 - 1] .-= dot(Fref[:, 4], d) .* d
    @test F.max_force(free) < 1.0e-3
    @test info.max_force ≈ F.max_force(free) rtol = 1.0e-10
    @test F.max_force(Fref[:, [1, 7]]) > 1.0e-3
    # The box bound on the first steps, so the box solve with the constraints ran too.
    @test opt.boxed > 0
end

@testitem "TrustRegionLBFGS iteration count, limits and memory" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: TrustRegionLBFGS
    F = RelaxFixtures
    sys, perfect, p = F.lj_crystal(Float64)
    x0 = copy(sys.positions)
    opt = TrustRegionLBFGS(sys)
    full = relax!(sys, p, opt; fmax = 1.0e-3)
    xfull = copy(sys.positions)
    # The state is reset by every call.
    again, _, p2 = F.lj_crystal(Float64)
    @test relax!(again, p2, opt; fmax = 1.0e-3) == full && again.positions == xfull
    part, _, p3 = F.lj_crystal(Float64)
    r = relax!(part, p3, TrustRegionLBFGS(part); fmax = 1.0e-3, maxsteps = 3)
    @test !r.converged && r.steps == 3
    Eref = F.energy_and_forces(F.lj_crystal(Float64)[3], part.positions, part.cell)[1]
    @test r.energy == p3.result.energy
    @test r.energy ≈ Eref rtol = 4 * eps()
    edge, _, p4 = F.lj_crystal(Float64)
    last = relax!(edge, p4, TrustRegionLBFGS(edge); fmax = 1.0e-3, maxsteps = full.steps)
    @test last.converged && last.steps == full.steps
    z0, _, p5 = F.lj_crystal(Float64)
    @test relax!(z0, p5, TrustRegionLBFGS(z0); maxsteps = 0).steps == 0 && z0.positions == x0
    # A memory of one pair still converges, and holds one pair.
    one, _, p6 = F.lj_crystal(Float64)
    o1 = TrustRegionLBFGS(one; memory = 1)
    @test relax!(one, p6, o1; fmax = 1.0e-3).converged
    @test o1.npairs == 1
end

@testitem "relax! defaults to TrustRegionLBFGS" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: TrustRegionLBFGS
    F = RelaxFixtures
    # The keywords other than fmax and maxsteps build the state; the same arithmetic runs, so the
    # results are equal.
    a, _, pa = F.lj_crystal(Float64)
    b, _, pb = F.lj_crystal(Float64)
    @test relax!(a, pa; fmax = 1.0e-3, memory = 3, fixed = [2]) ==
        relax!(b, pb, TrustRegionLBFGS(b; memory = 3, fixed = [2]); fmax = 1.0e-3)
    @test a.positions == b.positions
end

@testitem "TrustRegionLBFGS rejects a step that raises the energy" setup = [QNFixtures] begin
    using PureAtomistic.Integrators: TrustRegionLBFGS
    F = QNFixtures
    # As for TrustRegionBFGS: H₀ = I/0.1 makes the model step -50 d₀, clamped by the box (the
    # one-dimensional box solve) to the radius 2: x moves from 0.5 to -1.5, the energy rises, the
    # step is undone and the radius quartered.
    sys, site, p = F.spring(Float64, (0.5, 0.0, 0.0), 1.0)
    start = copy(sys.positions)
    opt = TrustRegionLBFGS(sys; alpha = 0.1, radius = 2.0, max_radius = 2.0)
    r = relax!(sys, p, opt; fmax = 1.0e-8, maxsteps = 1)
    @test r.steps == 1 && !r.converged
    @test opt.boxed == 1
    @test sys.positions == start
    @test opt.radius == 0.5
    @test p.result.energy == r.energy == 0.125
    sys, site, p = F.spring(Float64, (0.5, 0.0, 0.0), 1.0)
    @test relax!(sys, p, TrustRegionLBFGS(sys; alpha = 0.1, radius = 2.0, max_radius = 2.0); fmax = 1.0e-8).converged
end

@testitem "Trust-region relaxers undo a rise the model cannot resolve" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: TrustRegionBFGS, TrustRegionLBFGS
    F = RelaxFixtures
    # A force of 1e-9 eV/Å predicts a decrease of order 1e-20 eV, below the rounding of an energy
    # near 1 eV, while the energy rises by the distance moved: the step is undone and the radius
    # quartered, as for a rejected step whose decrease the model does resolve.
    for make in (TrustRegionBFGS, TrustRegionLBFGS)
        sys, p = F.jump_system()
        start = copy(sys.positions)
        opt = make(sys; radius = 0.2, max_radius = 0.2)
        r = relax!(sys, p, opt; fmax = 1.0e-12, maxsteps = 1)
        @test r.steps == 1 && !r.converged
        @test sys.positions == start
        @test opt.radius == 0.05
    end
end

@testitem "TrustRegionLBFGS with a MACE potential" setup = [RelaxFixtures] begin
    using PureMLIP
    using PureAtomistic.Systems
    using PureAtomistic.Integrators: TrustRegionLBFGS
    F = RelaxFixtures
    for T in (Float64, Float32)
        model = F.tiny_model(T)
        sys = PureAtomistic.Systems.fcc_cell(T(3.6), 2, 29)
        sys.positions .+= T(0.05) .* T.(reshape(sin.(1.0:(3 * natoms(sys))), 3, natoms(sys)))
        p = MACEPotential(model, sys.positions, sys.cell, sys.species)
        E_start = F.energy_and_forces(MACEPotential(model, sys.positions, sys.cell, sys.species), sys.positions, sys.cell)[1]
        info = relax!(sys, p, TrustRegionLBFGS(sys); fmax = 1.0e-4, maxsteps = 2000)
        Eref, Fref = F.energy_and_forces(MACEPotential(model, sys.positions, sys.cell, sys.species), sys.positions, sys.cell)
        @test info.converged
        @test F.max_force(Fref) < 1.0e-4
        @test info.energy < E_start
        @test info.energy ≈ Eref rtol = 100 * eps(T)
    end
end

@testitem "TrustRegionLBFGS rejects invalid input" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: TrustRegionLBFGS
    using PureAtomistic.Systems
    F = RelaxFixtures
    sys, _, p = F.lj_crystal(Float64)
    small = fcc_cell(5.26, 1, :Ag)
    @test_throws "pureatomistic: the TrustRegionLBFGS state has 4 atoms, the system 108" relax!(sys, p, TrustRegionLBFGS(small))
    @test_throws "pureatomistic: TrustRegionLBFGS needs at least one atom, got 0" TrustRegionLBFGS{Float64}(0)
    @test_throws "pureatomistic: TrustRegionLBFGS memory must be at least 1, got 0" TrustRegionLBFGS(small; memory = 0)
    @test_throws "pureatomistic: TrustRegionLBFGS radius must be positive and finite, got 0.0" TrustRegionLBFGS(small; radius = 0.0)
    @test_throws "pureatomistic: TrustRegionLBFGS max_radius must be finite and at least radius = 0.1, got 0.05" TrustRegionLBFGS(small; max_radius = 0.05)
    @test_throws "pureatomistic: TrustRegionLBFGS alpha must be positive and finite, got -1.0" TrustRegionLBFGS(small; alpha = -1.0)
    @test_throws "pureatomistic: TrustRegionLBFGS eta must lie in [0, 1/4), got 0.5" TrustRegionLBFGS(small; eta = 0.5)
    @test_throws "pureatomistic: fixed atom 5 is outside 1:4" TrustRegionLBFGS(small; fixed = [5])
    @test_throws "pureatomistic: atom 2 is both fixed and frozen" TrustRegionLBFGS(small; fixed = [2], frozen = [2 => (1, 0, 0)])
    nanp = F.NaNPotential(PotentialResult{Float64}(natoms(sys)))
    @test_throws "pureatomistic: relax! met a non-finite energy or force after 0 steps" relax!(sys, nanp, TrustRegionLBFGS(sys))
    @test_throws "pureatomistic: a curvature pair of length 3 for 12 coordinates" PureAtomistic.Integrators._curvature_update!(TrustRegionLBFGS(small), ones(3), ones(3))
    # A subspace on which the stored curvature is not positive definite is reported.
    # γ = -1 with the pair s = y = (1, …, 1) gives H = -I off the direction of the pair.
    opt = TrustRegionLBFGS(small; radius = 1.0e-3)
    PureAtomistic.Integrators._curvature_update!(opt, ones(12), ones(12))
    opt.gamma = -1.0
    opt.grad .= 0
    opt.grad[1, 1] = 1
    @test_throws "pureatomistic: the limited-memory inverse Hessian is not positive definite on the step subspace" PureAtomistic.Integrators._trust_step!(opt)
end

@testitem "TrustRegionLBFGS on views and OffsetArrays" setup = [RelaxFixtures] begin
    using OffsetArrays
    using PureAtomistic.Integrators: TrustRegionLBFGS
    using PureAtomistic.Systems
    F = RelaxFixtures
    sys, _, p = F.lj_crystal(Float64)
    n = natoms(sys)
    base = copy(sys.positions)
    ref = relax!(sys, p, TrustRegionLBFGS(sys); fmax = 1.0e-3)
    parent = zeros(3, n + 5)
    parent[:, 1:n] .= base
    vsys = System{Float64}(view(parent, :, 1:n), zeros(3, n), copy(sys.masses), copy(sys.species), copy(sys.cell))
    vp = LennardJones(vsys.positions, vsys.cell; epsilon = 0.0104, sigma = 3.4, r_cut = 8.5)
    # A view of a larger parent: the same arithmetic, so equality is exact.
    @test relax!(vsys, vp, TrustRegionLBFGS(vsys); fmax = 1.0e-3) == ref
    @test vsys.positions == sys.positions
    @test parent[:, (n + 1):end] == zeros(3, 5)
    osys = System{Float64}(
        OffsetArray(copy(base), 1:3, 0:(n - 1)), OffsetArray(zeros(3, n), 1:3, 0:(n - 1)),
        OffsetArray(copy(sys.masses), 0:(n - 1)), OffsetArray(copy(sys.species), 0:(n - 1)), copy(sys.cell)
    )
    @test_throws "offset arrays are not supported" relax!(osys, p, TrustRegionLBFGS(sys))
end

@testitem "TrustRegionLBFGS storage is linear in the number of atoms" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: TrustRegionLBFGS
    # The solver holds the row matrix the step rewrites, with unit row scales, and nothing of size
    # n²: doubling the atoms doubles the state.
    for T in (Float64, Float32)
        opt = TrustRegionLBFGS{T}(4; memory = 3)
        @test opt.qp.prob.A === opt.M
        @test opt.qp.red.ws.M.A === opt.M
        @test all(isone, opt.qp.red.scale)
        @test size(opt.M) == (12, 7)
        a = Base.summarysize(TrustRegionLBFGS{T}(500; memory = 5))
        b = Base.summarysize(TrustRegionLBFGS{T}(1000; memory = 5))
        @test 1.9 < b / a < 2.1
    end
end

@testitem "TrustRegionLBFGS does not allocate" tags = [:checks] setup = [RelaxFixtures] begin
    using PureMLIP
    using PureAtomistic.Systems
    using StrictModeTest, StrictMode
    using PureAtomistic.Integrators: TrustRegionLBFGS, _trust_step!, _two_loop!, _curvature_update!,
        _subspace_problem!, _subspace_step!, _gradient!
    F = RelaxFixtures
    register_alloc_barrier!(PureAtomistic.Potentials._regrow!)
    @noinline step!(opt) = _trust_step!(opt)
    @noinline twoloop!(r, opt, v) = _two_loop!(r, opt, v)
    @noinline update!(opt, s, y) = _curvature_update!(opt, s, y)
    @noinline problem!(opt) = _subspace_problem!(opt)
    @noinline recover!(opt, w) = _subspace_step!(opt, w)
    @noinline gradient!(G, F, opt) = _gradient!(G, F, opt)
    @noinline run!(sys, p, opt) = relax!(sys, p, opt; fmax = 1.0e-3, maxsteps = 5)
    for T in (Float64, Float32)
        sys, _, lj = F.lj_crystal(T)
        msys = PureAtomistic.Systems.fcc_cell(T(3.6), 2, 29)
        msys.positions .+= T(0.05) .* T.(reshape(sin.(1.0:(3 * natoms(msys))), 3, natoms(msys)))
        mace = MACEPotential(F.tiny_model(T), msys.positions, msys.cell, msys.species)
        for (s, p) in ((sys, lj), (msys, mace))
            opt = TrustRegionLBFGS(s; memory = 3, radius = 1.0e-3, fixed = [1], frozen = [2 => (0, 0, 1)])
            n = length(opt.gradv)
            sv = T.(sin.(1.0:n))
            y2 = 2 .* sv
            update!(opt, sv, y2)
            opt.grad .= T(0.1)
            opt.grad[:, 1] .= 0
            # PureDAQP reads the clock inside `update!`/`solve!` (a `jl_hrtime` foreign call that
            # AllocCheck reports as an allocation), so the step and the relaxation are measured; the
            # code of this package around them is proven allocation-free on every path.
            step!(opt)
            @test opt.boxed == 1
            @test_noalloc twoloop!(opt.hv, opt, opt.gradv)
            @test_noalloc update!(opt, sv, y2)
            @test_noalloc problem!(opt)
            @test_noalloc recover!(opt, opt.qp.x)
            @test_noalloc gradient!(opt.trial, p.result.forces, opt)
            @test_steady_state step!(opt)
            run!(s, p, opt)
            @test_steady_state run!(s, p, opt)
            @test p.n_regrows == 0
        end
    end
end

@testitem "TrimCheck: TrustRegionLBFGS" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureAtomistic, PureMLIP
            const SY = PureAtomistic.Systems
            const PP = PureAtomistic.Potentials
            const IN = PureAtomistic.Integrators
            const L64 = typeof(IN.TrustRegionLBFGS{Float64}(4))
            const L32 = typeof(IN.TrustRegionLBFGS{Float32}(4))
            new_l64(n) = IN.TrustRegionLBFGS{Float64}(n)
            new_l32(n) = IN.TrustRegionLBFGS{Float32}(n)
            relax64(s, p, o) = IN.relax!(s, p, o; fmax = 1.0e-3, maxsteps = 5)
            relax32(s, p, o) = IN.relax!(s, p, o; fmax = 1.0e-3, maxsteps = 5)
            const S64 = SY.System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}
            const S32 = SY.System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}
        end,
        new_l64(Int),
        new_l32(Int),
        IN._trust_step!(L64),
        IN._trust_step!(L32),
        relax64(S64, PP.LennardJones{Float64}, L64),
        relax32(S32, PP.LennardJones{Float32}, L32),
        relax64(S64, PP.MACEPotential{Float64}, L64),
        relax32(S32, PP.MACEPotential{Float32}, L32),
    )
end

@testitem "TrustRegionLBFGS return types are concrete" tags = [:checks] setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: TrustRegionLBFGS, _trust_step!
    F = RelaxFixtures
    for T in (Float64, Float32)
        sys, _, p = F.lj_crystal(T)
        @test Base.return_types(TrustRegionLBFGS{T}, (Int,))[1] |> isconcretetype
        opt = TrustRegionLBFGS(sys)
        @test Base.return_types(relax!, (typeof(sys), typeof(p), typeof(opt)))[1] === RelaxResult{T}
        @test Base.return_types(_trust_step!, (typeof(opt),))[1] === T
    end
end
