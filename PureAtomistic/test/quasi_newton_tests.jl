@testmodule QNFixtures begin
    using LinearAlgebra, Random
    using PureAtomistic
    using PureAtomistic.Systems

    # One atom on a spring of stiffness k (eV/Å²) in a 20 Å cubic cell, displaced by d from its site.
    function spring(::Type{T}, d, k) where {T}
        cell = Matrix{T}(20I, 3, 3)
        site = fill(T(10), 3, 1)
        x = site .+ T.(reshape(collect(d), 3, 1))
        sys = System{T}(x, zeros(T, 3, 1), T[1], [1], cell)
        return sys, site, HarmonicPotential(site, k)
    end

    # A symmetric positive definite matrix with eigenvalues spread over [lo, hi], from a seeded
    # random rotation.
    function spd(::Type{T}, n, lo, hi; seed = 1) where {T}
        Q = Matrix(qr(randn(Random.Xoshiro(seed), n, n)).Q)
        return T.(Symmetric(Q * Diagonal(range(lo, hi; length = n)) * Q'))
    end
end

@testitem "TrustRegionBFGS on one spring: clamped steps, then the exact minimum" setup = [QNFixtures] begin
    using LinearAlgebra
    using PureAtomistic.Integrators: TrustRegionBFGS
    F = QNFixtures
    T = Float64
    # Hand-derived with H₀ = I/α, α = 10, k = 5, start displacement d₀ = (0.5, -0.01, 0.002), radius 0.1:
    # step 1: -g/α = -(k/α) d₀ = (-0.25, 0.005, -0.001), clamped by the box to (-0.1, 0.005, -0.001).
    #   predicted decrease 0.20013, actual 0.225195, ratio 1.125 > 3/4 on the boundary: radius 0.2.
    sys, site, p = F.spring(T, (0.5, -0.01, 0.002), 5.0)
    opt = TrustRegionBFGS(sys; alpha = 10.0, radius = 0.1, max_radius = 1.0)
    one = relax!(sys, p, opt; fmax = 1.0e-6, maxsteps = 1)
    @test !one.converged && one.steps == 1
    # Rounding bound: a handful of operations on values of order one, c = 8, N = 1.
    @test sys.positions ≈ site .+ reshape([0.4, -0.005, 0.001], 3, 1) rtol = 8 * eps(T)
    @test opt.radius == 0.2
    # Step 2: y = k s, so the rescaled H₀ = (sᵀy/yᵀy) I = I/k and the BFGS update keep H = I/k, the
    # exact inverse Hessian. The Newton step -d₁ is clamped to -0.2 in x; the ratio is 1 on the
    # boundary, radius 0.4. Step 3 is the unclamped Newton step to the site: three steps in all.
    sys, site, p = F.spring(T, (0.5, -0.01, 0.002), 5.0)
    opt = TrustRegionBFGS(sys; alpha = 10.0, radius = 0.1, max_radius = 1.0)
    info = relax!(sys, p, opt; fmax = 1.0e-6)
    @test info.converged && info.steps == 3
    @test opt.radius == 0.4
    # The minimum is reached up to rounding of positions of size 10 Å: c = 16, N = 1.
    @test maximum(abs, sys.positions - site) < 16 * eps(T) * 10
    @test opt.H ≈ Matrix(I / 5, 3, 3) rtol = 16 * eps(T)
end

@testitem "TrustRegionBFGS step solves the box-constrained model (KKT oracle)" setup = [QNFixtures] begin
    using LinearAlgebra
    using PureDAQP: ActiveSet
    using PureAtomistic.Integrators: TrustRegionBFGS, _trust_step!
    F = QNFixtures
    # Both working sets of the solver: :gram factors the Gram matrix, :rows (the default) the rows.
    for T in (Float64, Float32), solver in (ActiveSet(; working_set = :gram), ActiveSet(; working_set = :rows))
        n_at = 8
        n = 3n_at
        sys = PureAtomistic.Systems.fcc_cell(T(5.26), (2, 1, 1), :Ag)
        opt = TrustRegionBFGS(sys; radius = 0.05, fixed = [3], solver)
        # Inverse Hessian with condition number 10 and a gradient large enough that some, not all,
        # coordinates reach the box.
        H = F.spd(T, n, 0.01, 0.1)
        opt.H .= H
        opt.radius = T(0.05)
        g = T.(reshape(sin.(1.0:n) .* 2, 3, n_at))
        opt.grad .= g
        opt.grad[:, 3] .= 0
        pred = _trust_step!(opt)
        p = vec(copy(opt.step))
        gv = vec(copy(opt.grad))
        B = inv(Float64.(H))
        r = B * Float64.(p) + Float64.(gv)
        # Rounding bound for the residual: c = 10, N = n κ(B) with κ = 10, on gradients of size 2.
        # Measured worst residual: 6.7e-16 (Float64), 1.4e-7 (Float32).
        tol = 10 * eps(T) * n * 10 * 2
        Δ = T(0.05)
        fixedrows = 7:9
        at_upper = at_lower = interior = 0
        for i in 1:n
            i in fixedrows && continue
            if p[i] > Δ * (1 - 100eps(T))
                at_upper += 1
                @test r[i] <= tol
            elseif p[i] < -Δ * (1 - 100eps(T))
                at_lower += 1
                @test r[i] >= -tol
            else
                interior += 1
                @test abs(r[i]) <= tol
            end
            @test abs(p[i]) <= Δ * (1 + 100eps(T))
        end
        # The draw exercises both bounds and the interior.
        @test at_upper > 0 && at_lower > 0 && interior > 0
        @test all(iszero, p[fixedrows])
        # The predicted decrease is the model's: -(gᵀp + ½ pᵀBp). Bound as for the residual.
        @test pred ≈ -(dot(gv, p) + dot(p, B * p) / 2) rtol = tol
    end
end

@testitem "TrustRegionBFGS step with frozen directions (KKT oracle)" setup = [QNFixtures] begin
    using LinearAlgebra
    using PureAtomistic.Integrators: TrustRegionBFGS, _trust_step!
    F = QNFixtures
    T = Float64
    sys = PureAtomistic.Systems.fcc_cell(5.26, (2, 1, 1), :Ag)
    n = 24
    d1 = [1.0, 2.0, 0.0] / sqrt(5)
    d2 = [0.0, 0.0, 1.0]
    # Atom 2 moves only along (2, -1, 0)/√5; atom 5 only in the plane normal to z. Radius 10 Å
    # keeps the box inactive, so the step solves the equality-constrained model.
    opt = TrustRegionBFGS(sys; radius = 10.0, max_radius = 10.0, fixed = [1], frozen = [2 => d1, 2 => d2, 5 => (0, 0, 2)])
    H = F.spd(T, n, 0.01, 0.1; seed = 2)
    opt.H .= H
    opt.grad .= reshape(cos.(1.0:n), 3, 8)
    opt.grad[:, 1] .= 0
    _trust_step!(opt)
    p = vec(copy(opt.step))
    # Oracle: the KKT system of min gᵀp + ½ pᵀBp subject to C p = 0, solved by LU.
    C = zeros(6, n)
    C[1, 1] = C[2, 2] = C[3, 3] = 1
    C[4, 4:6] = d1
    C[5, 4:6] = d2
    C[6, 15] = 1
    B = inv(H)
    K = [B C'; C zeros(6, 6)]
    ref = (K \ [-vec(opt.grad); zeros(6)])[1:n]
    # Rounding bound: c = 10, N = n κ(K); κ(K) measured 3.2e3 for this draw (bounded by 4e3 here),
    # gradients of size 1. Measured relative error 5.5e-16.
    @test cond(K) < 4.0e3
    @test p ≈ ref rtol = 10 * eps(T) * n * 4.0e3
    @test all(iszero, p[1:3])
    @test abs(dot(p[4:6], d1)) < 4 * eps(T) && abs(dot(p[4:6], d2)) < 4 * eps(T)
    @test abs(p[15]) < 4 * eps(T)
end

@testitem "TrustRegionBFGS relaxes a Lennard-Jones crystal" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: TrustRegionBFGS
    F = RelaxFixtures
    # Oracle as for FIRE: the undisplaced lattice is an equilibrium of the fixed cell, and a fresh
    # potential gives the energy and forces at the returned positions. Measured at fmax = 1e-3:
    # 13 steps against FIRE's 70 in both precisions; E - E_lattice = 5.6e-5 eV (Float64) and
    # -8.8e-5 eV (Float32, below the lattice by the rounding of the Float32 energy). The bound
    # 1e-3 on either side is 11 to 18 times those.
    for T in (Float64, Float32)
        sys, perfect, p = F.lj_crystal(T)
        E0 = F.energy_and_forces(F.lj_crystal(T)[3], perfect, sys.cell)[1]
        info = relax!(sys, p, TrustRegionBFGS(sys); fmax = 1.0e-3)
        @test info.converged
        @test info isa RelaxResult{T}
        Eref, Fref = F.energy_and_forces(F.lj_crystal(T)[3], sys.positions, sys.cell)
        @test F.max_force(Fref) < 1.0e-3
        @test info.max_force ≈ F.max_force(Fref) rtol = 8 * eps(T)
        @test info.energy == p.result.energy
        @test info.energy ≈ Eref rtol = 4 * eps(T)
        @test -1.0e-3 < info.energy - E0 < 1.0e-3
        fire_sys, _, fire_p = F.lj_crystal(T)
        @test info.steps < relax!(fire_sys, fire_p; fmax = 1.0e-3).steps
    end
end

@testitem "TrustRegionBFGS keeps fixed atoms and frozen directions" setup = [RelaxFixtures] begin
    using LinearAlgebra
    using PureAtomistic.Integrators: TrustRegionBFGS
    F = RelaxFixtures
    sys, _, p = F.lj_crystal(Float64)
    start = copy(sys.positions)
    d = [1.0, 2.0, 2.0] / 3
    opt = TrustRegionBFGS(sys; fixed = [1, 7], frozen = [4 => d])
    info = relax!(sys, p, opt; fmax = 1.0e-3)
    @test info.converged
    @test sys.positions[:, [1, 7]] == start[:, [1, 7]]
    # Each step's component along d is removed after the solve, so what remains is the rounding of
    # adding the steps to coordinates of size |x| ≤ 3 Å (atom 4): c = 4, N = steps · |x|. Measured
    # 1.8e-16 after 27 steps.
    @test abs(dot(sys.positions[:, 4] - start[:, 4], d)) < 4 * eps() * info.steps * 3
    # Convergence is decided on the forces the constraints leave free: the fixed atoms and the
    # frozen component keep forces above fmax, everything else is below it.
    Fref = F.energy_and_forces(F.lj_crystal(Float64)[3], sys.positions, sys.cell)[2]
    free = Fref[:, setdiff(1:size(Fref, 2), [1, 7])]
    free[:, 4 - 1] .-= dot(Fref[:, 4], d) .* d
    @test F.max_force(free) < 1.0e-3
    @test info.max_force ≈ F.max_force(free) rtol = 1.0e-10
    @test F.max_force(Fref[:, [1, 7]]) > 1.0e-3
end

@testitem "TrustRegionBFGS iteration count and limits" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: TrustRegionBFGS
    F = RelaxFixtures
    sys, perfect, p = F.lj_crystal(Float64)
    x0 = copy(sys.positions)
    full = relax!(sys, p, TrustRegionBFGS(sys); fmax = 1.0e-3)
    xfull = copy(sys.positions)
    # A run stopped after 3 steps leaves the potential's result at the returned positions.
    sys2, _, p2 = F.lj_crystal(Float64)
    part = relax!(sys2, p2, TrustRegionBFGS(sys2); fmax = 1.0e-3, maxsteps = 3)
    @test !part.converged && part.steps == 3
    Eref = F.energy_and_forces(F.lj_crystal(Float64)[3], sys2.positions, sys2.cell)[1]
    @test part.energy == p2.result.energy
    @test part.energy ≈ Eref rtol = 4 * eps()
    # The same steps again, and the state is reset by every call.
    opt = TrustRegionBFGS(sys2)
    for _ in 1:2
        again, _, p3 = F.lj_crystal(Float64)
        r = relax!(again, p3, opt; fmax = 1.0e-3)
        @test r.steps == full.steps && again.positions == xfull
    end
    # Converging on the last permitted step still reports convergence.
    edge, _, p4 = F.lj_crystal(Float64)
    last = relax!(edge, p4, TrustRegionBFGS(edge); fmax = 1.0e-3, maxsteps = full.steps)
    @test last.converged && last.steps == full.steps
    # Zero steps only evaluates; a relaxed configuration is returned untouched.
    sys5, _, p5 = F.lj_crystal(Float64)
    r0 = relax!(sys5, p5, TrustRegionBFGS(sys5); maxsteps = 0)
    @test r0.steps == 0 && !r0.converged && sys5.positions == x0
    sys6, perfect6, p6 = F.lj_crystal(Float64)
    sys6.positions .= perfect6
    done = relax!(sys6, p6, TrustRegionBFGS(sys6); fmax = 1.0e-3)
    @test done.converged && done.steps == 0 && sys6.positions == perfect6
end

@testitem "TrustRegionBFGS rejects a step that raises the energy" setup = [QNFixtures] begin
    using PureAtomistic.Integrators: TrustRegionBFGS
    F = QNFixtures
    # α = 0.1 makes the first model step -g/α = -50 d₀, clamped to the radius 2: x moves from 0.5 to
    # -1.5, which raises the spring energy (ratio < 0). The step is undone and the radius quartered.
    sys, site, p = F.spring(Float64, (0.5, 0.0, 0.0), 1.0)
    start = copy(sys.positions)
    opt = TrustRegionBFGS(sys; alpha = 0.1, radius = 2.0, max_radius = 2.0)
    r = relax!(sys, p, opt; fmax = 1.0e-8, maxsteps = 1)
    @test r.steps == 1 && !r.converged
    @test sys.positions == start
    @test opt.radius == 0.5
    # The potential's result is that of the returned positions, not of the rejected trial.
    @test p.result.energy == r.energy == 0.125
    # The run still converges.
    sys, site, p = F.spring(Float64, (0.5, 0.0, 0.0), 1.0)
    r = relax!(sys, p, TrustRegionBFGS(sys; alpha = 0.1, radius = 2.0, max_radius = 2.0); fmax = 1.0e-8)
    @test r.converged
end

@testitem "TrustRegionBFGS with a MACE potential" setup = [RelaxFixtures] begin
    using PureMLIP
    using PureAtomistic.Systems
    using PureAtomistic.Integrators: TrustRegionBFGS
    F = RelaxFixtures
    for T in (Float64, Float32)
        model = F.tiny_model(T)
        sys = PureAtomistic.Systems.fcc_cell(T(3.6), 2, 29)
        sys.positions .+= T(0.05) .* T.(reshape(sin.(1.0:(3 * natoms(sys))), 3, natoms(sys)))
        p = MACEPotential(model, sys.positions, sys.cell, sys.species)
        E_start = F.energy_and_forces(MACEPotential(model, sys.positions, sys.cell, sys.species), sys.positions, sys.cell)[1]
        info = relax!(sys, p, TrustRegionBFGS(sys); fmax = 1.0e-4, maxsteps = 2000)
        Eref, Fref = F.energy_and_forces(MACEPotential(model, sys.positions, sys.cell, sys.species), sys.positions, sys.cell)
        @test info.converged
        @test F.max_force(Fref) < 1.0e-4
        @test info.energy < E_start
        @test info.energy ≈ Eref rtol = 100 * eps(T)
    end
end

@testitem "TrustRegionBFGS rejects invalid input" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: TrustRegionBFGS
    using PureAtomistic.Systems
    F = RelaxFixtures
    sys, _, p = F.lj_crystal(Float64)
    small = fcc_cell(5.26, 1, :Ag)
    @test_throws "pureatomistic: the TrustRegionBFGS state has 4 atoms, the system 108" relax!(sys, p, TrustRegionBFGS(small))
    @test_throws "pureatomistic: fmax must be positive and finite, got 0.0" relax!(sys, p, TrustRegionBFGS(sys); fmax = 0.0)
    @test_throws "pureatomistic: TrustRegionBFGS needs at least one atom, got 0" TrustRegionBFGS{Float64}(0)
    @test_throws "pureatomistic: TrustRegionBFGS radius must be positive and finite, got 0.0" TrustRegionBFGS(small; radius = 0.0)
    @test_throws "pureatomistic: TrustRegionBFGS max_radius must be finite and at least radius = 0.1, got 0.05" TrustRegionBFGS(small; max_radius = 0.05)
    @test_throws "pureatomistic: TrustRegionBFGS alpha must be positive and finite, got -1.0" TrustRegionBFGS(small; alpha = -1.0)
    @test_throws "pureatomistic: TrustRegionBFGS eta must lie in [0, 1/4), got 0.5" TrustRegionBFGS(small; eta = 0.5)
    @test_throws "pureatomistic: fixed atom 5 is outside 1:4" TrustRegionBFGS(small; fixed = [5])
    @test_throws "pureatomistic: frozen atom 0 is outside 1:4" TrustRegionBFGS(small; frozen = [0 => (1, 0, 0)])
    @test_throws "pureatomistic: atom 2 is both fixed and frozen" TrustRegionBFGS(small; fixed = [2], frozen = [2 => (1, 0, 0)])
    @test_throws "pureatomistic: the frozen direction of atom 1 must be a finite nonzero 3-vector" TrustRegionBFGS(small; frozen = [1 => (0, 0, 0)])
    @test_throws "pureatomistic: the frozen direction of atom 1 must be a finite nonzero 3-vector" TrustRegionBFGS(small; frozen = [1 => (1, 0)])
    @test_throws "pureatomistic: the frozen directions of atom 1 must be orthogonal" TrustRegionBFGS(small; frozen = [1 => (1, 0, 0), 1 => (1, 1, 0)])
    nanp = F.NaNPotential(PotentialResult{Float64}(natoms(sys)))
    @test_throws "pureatomistic: relax! met a non-finite energy or force after 0 steps" relax!(sys, nanp, TrustRegionBFGS(sys))
    # A step problem the solver cannot solve is reported, not taken: an inverse Hessian that is
    # not positive definite has no Cholesky factor.
    opt = TrustRegionBFGS(small)
    opt.H .= 0
    opt.grad .= 1
    @test_throws "pureatomistic: the BFGS inverse Hessian is not positive definite" PureAtomistic.Integrators._trust_step!(opt)
end

@testitem "TrustRegionBFGS on views and OffsetArrays" setup = [RelaxFixtures] begin
    using OffsetArrays
    using PureAtomistic.Integrators: TrustRegionBFGS
    using PureAtomistic.Systems
    F = RelaxFixtures
    sys, _, p = F.lj_crystal(Float64)
    n = natoms(sys)
    base = copy(sys.positions)
    ref = relax!(sys, p, TrustRegionBFGS(sys); fmax = 1.0e-3)
    parent = zeros(3, n + 5)
    parent[:, 1:n] .= base
    vsys = System{Float64}(view(parent, :, 1:n), zeros(3, n), copy(sys.masses), copy(sys.species), copy(sys.cell))
    vp = LennardJones(vsys.positions, vsys.cell; epsilon = 0.0104, sigma = 3.4, r_cut = 8.5)
    # A view of a larger parent: the same arithmetic, so equality is exact.
    @test relax!(vsys, vp, TrustRegionBFGS(vsys); fmax = 1.0e-3) == ref
    @test vsys.positions == sys.positions
    @test parent[:, (n + 1):end] == zeros(3, 5)
    osys = System{Float64}(
        OffsetArray(copy(base), 1:3, 0:(n - 1)), OffsetArray(zeros(3, n), 1:3, 0:(n - 1)),
        OffsetArray(copy(sys.masses), 0:(n - 1)), OffsetArray(copy(sys.species), 0:(n - 1)), copy(sys.cell)
    )
    @test_throws "offset arrays are not supported" relax!(osys, p, TrustRegionBFGS(sys))
end

@testitem "TrustRegionBFGS solver holds the constraint matrix it is given" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: TrustRegionBFGS
    # The step rewrites the rows of `opt.M` in place and the PureDAQP workspace reads them from
    # there: that needs the workspace to hold this very matrix through an implicit reduction
    # (an identity `Diagonal` objective) whose row scales are all one.
    small = PureAtomistic.Systems.fcc_cell(5.26, 1, :Ag)
    for T in (Float64, Float32)
        sys = PureAtomistic.Systems.fcc_cell(T(5.26), 1, :Ag)
        opt = TrustRegionBFGS(sys; frozen = [2 => (1, 1, 1)])
        @test opt.qp.prob.A === opt.M
        @test opt.qp.red.ws.M.A === opt.M
        @test all(isone, opt.qp.red.scale)
        @test size(opt.M) == (13, 12)
    end
end

@testitem "TrustRegionBFGS does not allocate" tags = [:checks] setup = [RelaxFixtures] begin
    using PureMLIP
    using PureAtomistic.Systems
    using StrictModeTest, StrictMode
    using PureAtomistic.Integrators: TrustRegionBFGS, _trust_step!, _step_problem!, _step_from_solution!,
        _bfgs_update!, _gradient!
    F = RelaxFixtures
    register_alloc_barrier!(PureAtomistic.Potentials._regrow!)
    @noinline step!(opt) = _trust_step!(opt)
    @noinline problem!(opt) = _step_problem!(opt)
    @noinline recover!(opt, w) = _step_from_solution!(opt, w)
    @noinline update!(opt, s, y) = _bfgs_update!(opt, s, y)
    @noinline gradient!(G, F, opt) = _gradient!(G, F, opt)
    @noinline run!(sys, p, opt) = relax!(sys, p, opt; fmax = 1.0e-3, maxsteps = 5)
    for T in (Float64, Float32)
        sys, _, lj = F.lj_crystal(T)
        msys = PureAtomistic.Systems.fcc_cell(T(3.6), 2, 29)
        msys.positions .+= T(0.05) .* T.(reshape(sin.(1.0:(3 * natoms(msys))), 3, natoms(msys)))
        mace = MACEPotential(F.tiny_model(T), msys.positions, msys.cell, msys.species)
        for (s, p) in ((sys, lj), (msys, mace))
            opt = TrustRegionBFGS(s; fixed = [1], frozen = [2 => (0, 0, 1)])
            opt.grad .= T(0.1)
            # The code of this package is proven allocation-free on every path. The PureDAQP calls
            # in between read the clock (`time_ns`, a `jl_hrtime` foreign call returning an integer),
            # which AllocCheck cannot see through and reports as an allocation; the whole step and
            # the whole relaxation are therefore measured instead.
            step!(opt)
            @test_noalloc problem!(opt)
            @test_noalloc recover!(opt, opt.qp.x)
            @test_noalloc update!(opt, opt.stepv, opt.y)
            @test_noalloc gradient!(opt.trial, p.result.forces, opt)
            @test_steady_state step!(opt)
            run!(s, p, opt)
            @test_steady_state run!(s, p, opt)
            @test p.n_regrows == 0
        end
    end
end

@testitem "TrimCheck: TrustRegionBFGS" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureAtomistic, PureMLIP
            const SY = PureAtomistic.Systems
            const PP = PureAtomistic.Potentials
            const IN = PureAtomistic.Integrators
            const Q64 = typeof(IN.TrustRegionBFGS{Float64}(4))
            const Q32 = typeof(IN.TrustRegionBFGS{Float32}(4))
            new_qn64(n) = IN.TrustRegionBFGS{Float64}(n)
            new_qn32(n) = IN.TrustRegionBFGS{Float32}(n)
            relax64(s, p, o) = IN.relax!(s, p, o; fmax = 1.0e-3, maxsteps = 5)
            relax32(s, p, o) = IN.relax!(s, p, o; fmax = 1.0e-3, maxsteps = 5)
            const S64 = SY.System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}
            const S32 = SY.System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}
        end,
        new_qn64(Int),
        new_qn32(Int),
        IN._trust_step!(Q64),
        IN._trust_step!(Q32),
        relax64(S64, PP.LennardJones{Float64}, Q64),
        relax32(S32, PP.LennardJones{Float32}, Q32),
        relax64(S64, PP.MACEPotential{Float64}, Q64),
        relax32(S32, PP.MACEPotential{Float32}, Q32),
    )
end

@testitem "TrustRegionBFGS return types are concrete" tags = [:checks] setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: TrustRegionBFGS, _trust_step!
    F = RelaxFixtures
    for T in (Float64, Float32)
        sys, _, p = F.lj_crystal(T)
        @test Base.return_types(TrustRegionBFGS{T}, (Int,))[1] |> isconcretetype
        opt = TrustRegionBFGS(sys)
        @test Base.return_types(relax!, (typeof(sys), typeof(p), typeof(opt)))[1] === RelaxResult{T}
        @test Base.return_types(_trust_step!, (typeof(opt),))[1] === T
    end
end
