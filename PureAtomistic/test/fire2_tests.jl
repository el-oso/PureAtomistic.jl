@testitem "fire2_step! applies the FIRE 2.0 rules" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: FIRE2, fire2_step!
    T = Float64
    # Hand-derived single steps of FIRE 2.0 with unit masses and the semi-implicit Euler integrator,
    # mixing after the velocity update (no displacement cap: maxstep = 10).
    # P > 0: F = (3, 4, 0), v = (2, 0, 0), α = 0.25, dt = 0.1. v += dt F → (2.3, 0.4, 0), |v| = √5.45;
    #   v ← 0.75 v + 0.25 √5.45 F/5;  x += dt v.
    opt = FIRE2{T}(1; dt = 0.1, dtmax = 1.0, maxstep = 10.0)
    opt.velocity .= [2.0, 0.0, 0.0]
    opt.nneg = 3
    x = zeros(T, 3, 1)
    @test fire2_step!(opt, x, reshape([3.0, 4.0, 0.0], 3, 1))
    v = 0.75 .* [2.3, 0.4, 0.0] .+ 0.25 * sqrt(5.45) .* [0.6, 0.8, 0.0]
    # Rounding bound: about ten operations per component, c = 10, N = 1 (values of order 1).
    @test opt.velocity ≈ reshape(v, 3, 1) rtol = 10 * eps(T)
    @test x ≈ reshape(0.1 .* v, 3, 1) rtol = 10 * eps(T)
    @test opt.npos == 1 && opt.nneg == 0 && opt.iter == 1
    @test opt.dt == 0.1 && opt.alpha == 0.25
    # P ≤ 0 after the initial delay: dt ← dt/2 = 0.05, α ← α_start, then the uphill correction
    # x ← x - ½ dt v = (-0.05, 0, 0) with the reduced dt, v ← 0, v += dt F = (-0.15, 0, 0) (mixing
    # leaves a velocity parallel to F alone), x += dt v → (-0.0575, 0, 0).
    opt = FIRE2{T}(1; dt = 0.1, dtmax = 1.0, maxstep = 10.0)
    opt.velocity .= [2.0, 0.0, 0.0]
    opt.iter = 20
    opt.npos = 7
    opt.alpha = 0.05
    x = zeros(T, 3, 1)
    @test fire2_step!(opt, x, reshape([-3.0, 0.0, 0.0], 3, 1))
    @test opt.dt == 0.05 && opt.alpha == 0.25 && opt.npos == 0 && opt.nneg == 1
    @test opt.velocity ≈ reshape([-0.15, 0.0, 0.0], 3, 1) rtol = 4 * eps(T)
    @test x ≈ reshape([-0.0575, 0.0, 0.0], 3, 1) rtol = 4 * eps(T)
    # Within the initial delay (iteration 3 < ndelay = 20) dt and α stay, the correction still runs.
    opt = FIRE2{T}(1; dt = 0.1, dtmax = 1.0, maxstep = 10.0)
    opt.velocity .= [2.0, 0.0, 0.0]
    opt.iter = 2
    opt.alpha = 0.05
    x = zeros(T, 3, 1)
    fire2_step!(opt, x, reshape([-3.0, 0.0, 0.0], 3, 1))
    @test opt.dt == 0.1 && opt.alpha == 0.05
    # x - ½ · 0.1 · 2 = -0.1, then v = 0.1 F = (-0.3, 0, 0), x += 0.1 v → -0.13.
    @test x ≈ reshape([-0.13, 0.0, 0.0], 3, 1) rtol = 4 * eps(T)
    # Without the initial delay the same step reduces dt at once.
    opt = FIRE2{T}(1; dt = 0.1, dtmax = 1.0, maxstep = 10.0, initialdelay = false)
    opt.velocity .= [2.0, 0.0, 0.0]
    fire2_step!(opt, zeros(T, 3, 1), reshape([-3.0, 0.0, 0.0], 3, 1))
    @test opt.dt == 0.05
    # dt is not reduced below dtmin: 0.1 · 0.5 < 0.06.
    opt = FIRE2{T}(1; dt = 0.1, dtmax = 1.0, maxstep = 10.0, dtmin = 0.06, initialdelay = false)
    opt.velocity .= [2.0, 0.0, 0.0]
    fire2_step!(opt, zeros(T, 3, 1), reshape([-3.0, 0.0, 0.0], 3, 1))
    @test opt.dt == 0.1 && opt.alpha == 0.25
    # More than nneg_max consecutive steps with P ≤ 0 stop the iteration without moving.
    opt = FIRE2{T}(1; dt = 0.1, dtmax = 1.0, maxstep = 10.0, nneg_max = 1)
    x = zeros(T, 3, 1)
    f = reshape([1.0, 0.0, 0.0], 3, 1)
    @test fire2_step!(opt, x, f)                  # v = 0: P = 0, first non-positive step
    opt.velocity .= [-1.0, 0.0, 0.0]
    x .= 0
    @test !fire2_step!(opt, x, f)                 # second in a row: stop
    @test x == zeros(T, 3, 1) && opt.velocity == reshape([-1.0, 0.0, 0.0], 3, 1)
end

@testitem "FIRE2 time step and mixing schedule" setup = [RelaxFixtures] begin
    using LinearAlgebra
    using PureAtomistic.Integrators: FIRE2, fire2_step!
    T = Float64
    # Force and velocity along +x keep the power positive. The first ndelay = 20 steps leave dt and α
    # alone; every later one multiplies dt by finc (until dtmax) and α by falpha.
    opt = FIRE2{T}(1; dt = 0.1, dtmax = 1.0, maxstep = 1.0e3)
    opt.velocity .= [1.0, 0.0, 0.0]
    x = zeros(T, 3, 1)
    F = reshape([1.0, 0.0, 0.0], 3, 1)
    for k in 1:20
        fire2_step!(opt, x, F)
        @test opt.dt == 0.1 && opt.alpha == 0.25 && opt.npos == k
    end
    for k in 1:4
        fire2_step!(opt, x, F)
        # Powers of 1.1 and 0.99 carry rounding of about 2k operations: c = 2, N = k.
        @test opt.dt ≈ 0.1 * 1.1^k rtol = 2 * k * eps(T)
        @test opt.alpha ≈ 0.25 * 0.99^k rtol = 2 * k * eps(T)
    end
    for _ in 1:100
        fire2_step!(opt, x, F)
    end
    @test opt.dt == 1.0   # dtmax, reached through min(): exact
    # The displacement cap: one atom moving 10 Å in a step is scaled to maxstep exactly, and so is
    # the uphill correction.
    opt = FIRE2{T}(1; dt = 0.1, dtmax = 1.0, maxstep = 0.1)
    opt.velocity .= [100.0, 0.0, 0.0]
    x = zeros(T, 3, 1)
    fire2_step!(opt, x, F)
    @test norm(x) ≈ 0.1 rtol = 4 * eps(T)
    opt = FIRE2{T}(1; dt = 0.1, dtmax = 1.0, maxstep = 0.1, initialdelay = false)
    opt.velocity .= [-100.0, 0.0, 0.0]
    x = zeros(T, 3, 1)
    fire2_step!(opt, x, F)
    # back by 0.1 (capped from ½ · 0.05 · 100 = 2.5), then forward by 0.05 · 0.05 · 1 = 0.0025
    @test x ≈ reshape([0.1 + 0.0025, 0.0, 0.0], 3, 1) rtol = 4 * eps(T)
end

@testitem "FIRE2 relaxes a displaced Lennard-Jones crystal" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: FIRE2
    F = RelaxFixtures
    # Oracles as for FIRE: a fresh potential at the returned positions, and the undisplaced lattice,
    # an equilibrium of the fixed cell. Measured at fmax = 1e-3: E - E_lattice = 1.2e-4 eV (Float64,
    # 36 steps against FIRE's 41) and -3.4e-5 eV (Float32); the bound 1e-3 on either side is 8
    # times the larger.
    for T in (Float64, Float32)
        sys, perfect, p = F.lj_crystal(T)
        E0 = F.energy_and_forces(F.lj_crystal(T)[3], perfect, sys.cell)[1]
        info = relax!(sys, p, FIRE2(sys); fmax = 1.0e-3)
        @test info.converged
        @test info isa RelaxResult{T}
        @test 0 < info.steps < 500
        Eref, Fref = F.energy_and_forces(F.lj_crystal(T)[3], sys.positions, sys.cell)
        @test F.max_force(Fref) < 1.0e-3
        @test info.max_force ≈ F.max_force(Fref) rtol = 8 * eps(T)
        @test info.energy == p.result.energy
        @test info.energy ≈ Eref rtol = 4 * eps(T)
        @test -1.0e-3 < info.energy - E0 < 1.0e-3
    end
end

@testitem "FIRE2 iteration count, limits and the uphill stop" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: FIRE2
    F = RelaxFixtures
    sys, perfect, p = F.lj_crystal(Float64)
    x0 = copy(sys.positions)
    opt = FIRE2(sys)
    full = relax!(sys, p, opt; fmax = 1.0e-3)
    xfull = copy(sys.positions)
    # The state is reset by every call: the same run again gives the same positions.
    again, _, p2 = F.lj_crystal(Float64)
    @test relax!(again, p2, opt; fmax = 1.0e-3) == full && again.positions == xfull
    part, _, p3 = F.lj_crystal(Float64)
    r = relax!(part, p3, FIRE2(part); fmax = 1.0e-3, maxsteps = 10)
    @test !r.converged && r.steps == 10
    edge, _, p4 = F.lj_crystal(Float64)
    last = relax!(edge, p4, FIRE2(edge); fmax = 1.0e-3, maxsteps = full.steps)
    @test last.converged && last.steps == full.steps
    z0, _, p5 = F.lj_crystal(Float64)
    @test relax!(z0, p5, FIRE2(z0); maxsteps = 0).steps == 0 && z0.positions == x0
    # The first iteration starts at rest, so its power is zero: with nneg_max = 0 the run stops
    # there, unmoved and not converged.
    stuck, _, p6 = F.lj_crystal(Float64)
    s = relax!(stuck, p6, FIRE2(stuck; nneg_max = 0); fmax = 1.0e-3)
    @test !s.converged && s.steps == 0 && stuck.positions == x0
    @test s.energy == p6.result.energy
end

@testitem "FIRE2 with a MACE potential" setup = [RelaxFixtures] begin
    using PureMLIP
    using PureAtomistic.Systems
    using PureAtomistic.Integrators: FIRE2
    F = RelaxFixtures
    for T in (Float64, Float32)
        model = F.tiny_model(T)
        sys = PureAtomistic.Systems.fcc_cell(T(3.6), 2, 29)
        sys.positions .+= T(0.05) .* T.(reshape(sin.(1.0:(3 * natoms(sys))), 3, natoms(sys)))
        p = MACEPotential(model, sys.positions, sys.cell, sys.species)
        E_start = F.energy_and_forces(MACEPotential(model, sys.positions, sys.cell, sys.species), sys.positions, sys.cell)[1]
        info = relax!(sys, p, FIRE2(sys); fmax = 1.0e-4, maxsteps = 2000)
        Eref, Fref = F.energy_and_forces(MACEPotential(model, sys.positions, sys.cell, sys.species), sys.positions, sys.cell)
        @test info.converged
        @test F.max_force(Fref) < 1.0e-4
        @test info.energy < E_start
        @test info.energy ≈ Eref rtol = 100 * eps(T)
    end
end

@testitem "FIRE2 rejects invalid input" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: FIRE2
    using PureAtomistic.Systems
    F = RelaxFixtures
    sys, _, p = F.lj_crystal(Float64)
    small = fcc_cell(5.26, 1, :Ag)
    @test_throws "pureatomistic: the FIRE2 state has 4 atoms, the system 108" relax!(sys, p, FIRE2(small))
    @test_throws "pureatomistic: fmax must be positive and finite, got 0.0" relax!(sys, p, FIRE2(sys); fmax = 0.0)
    @test_throws "pureatomistic: the number of atoms must not be negative, got -1" FIRE2{Float64}(-1)
    @test_throws "pureatomistic: FIRE2 dt must be positive and finite, got 0.0" FIRE2(sys; dt = 0.0)
    @test_throws "pureatomistic: FIRE2 dtmax must be finite and at least dt = 0.2, got 0.05" FIRE2(sys; dtmax = 0.05)
    @test_throws "pureatomistic: FIRE2 dtmin must lie in [0, dt = 0.2], got 0.3" FIRE2(sys; dtmin = 0.3)
    @test_throws "pureatomistic: FIRE2 maxstep must be positive and finite, got -1.0" FIRE2(sys; maxstep = -1.0)
    @test_throws "pureatomistic: FIRE2 ndelay must not be negative, got -1" FIRE2(sys; ndelay = -1)
    @test_throws "pureatomistic: FIRE2 nneg_max must not be negative, got -1" FIRE2(sys; nneg_max = -1)
    @test_throws "pureatomistic: FIRE2 finc must exceed 1, got 1.0" FIRE2(sys; finc = 1.0)
    @test_throws "pureatomistic: FIRE2 fdec must lie in (0, 1), got 1.0" FIRE2(sys; fdec = 1.0)
    @test_throws "pureatomistic: FIRE2 alpha_start must lie in (0, 1], got 0.0" FIRE2(sys; alpha_start = 0.0)
    @test_throws "pureatomistic: FIRE2 falpha must lie in (0, 1], got 1.5" FIRE2(sys; falpha = 1.5)
    @test_throws "pureatomistic: positions" PureAtomistic.Integrators.fire2_step!(FIRE2{Float64}(2), zeros(3, 3), zeros(3, 3))
    nanp = F.NaNPotential(PotentialResult{Float64}(natoms(sys)))
    @test_throws "pureatomistic: relax! met a non-finite energy or force after 0 steps" relax!(sys, nanp, FIRE2(sys))
end

@testitem "FIRE2 on views and OffsetArrays" setup = [RelaxFixtures] begin
    using OffsetArrays
    using PureAtomistic.Integrators: FIRE2
    using PureAtomistic.Systems
    F = RelaxFixtures
    sys, _, p = F.lj_crystal(Float64)
    n = natoms(sys)
    base = copy(sys.positions)
    ref = relax!(sys, p, FIRE2(sys); fmax = 1.0e-3)
    parent = zeros(3, n + 5)
    parent[:, 1:n] .= base
    vsys = System{Float64}(view(parent, :, 1:n), zeros(3, n), copy(sys.masses), copy(sys.species), copy(sys.cell))
    vp = LennardJones(vsys.positions, vsys.cell; epsilon = 0.0104, sigma = 3.4, r_cut = 8.5)
    # A view of a larger parent: the same arithmetic, so equality is exact.
    @test relax!(vsys, vp, FIRE2(vsys); fmax = 1.0e-3) == ref
    @test vsys.positions == sys.positions
    @test parent[:, (n + 1):end] == zeros(3, 5)
    osys = System{Float64}(
        OffsetArray(copy(base), 1:3, 0:(n - 1)), OffsetArray(zeros(3, n), 1:3, 0:(n - 1)),
        OffsetArray(copy(sys.masses), 0:(n - 1)), OffsetArray(copy(sys.species), 0:(n - 1)), copy(sys.cell)
    )
    @test_throws "offset arrays are not supported" relax!(osys, p, FIRE2(sys))
end

@testitem "FIRE2 does not allocate" tags = [:checks] setup = [RelaxFixtures] begin
    using PureMLIP
    using PureAtomistic.Systems
    using StrictModeTest, StrictMode
    using PureAtomistic.Integrators: FIRE2, fire2_step!
    F = RelaxFixtures
    register_alloc_barrier!(PureAtomistic.Potentials._regrow!)
    @noinline step!(opt, x, f) = fire2_step!(opt, x, f)
    @noinline run!(sys, p, opt) = relax!(sys, p, opt; fmax = 1.0e-3, maxsteps = 5)
    for T in (Float64, Float32)
        sys, _, lj = F.lj_crystal(T)
        msys = PureAtomistic.Systems.fcc_cell(T(3.6), 2, 29)
        msys.positions .+= T(0.05) .* T.(reshape(sin.(1.0:(3 * natoms(msys))), 3, natoms(msys)))
        mace = MACEPotential(F.tiny_model(T), msys.positions, msys.cell, msys.species)
        for (s, p) in ((sys, lj), (msys, mace))
            opt = FIRE2(s)
            f = fill(T(0.1), size(s.positions))
            opt.velocity .= T(0.01)
            step!(opt, s.positions, f)
            @test_noalloc step!(opt, s.positions, f)
            run!(s, p, opt)
            @test_noalloc run!(s, p, opt)
            @test_steady_state run!(s, p, opt)
            @test p.n_regrows == 0
        end
    end
end

@testitem "TrimCheck: FIRE2" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureAtomistic, PureMLIP
            const SY = PureAtomistic.Systems
            const PP = PureAtomistic.Potentials
            const IN = PureAtomistic.Integrators
            new_fire2_64(n) = IN.FIRE2{Float64}(n)
            new_fire2_32(n) = IN.FIRE2{Float32}(n)
            relax64(s, p, o) = IN.relax!(s, p, o; fmax = 1.0e-3, maxsteps = 5)
            relax32(s, p, o) = IN.relax!(s, p, o; fmax = 1.0e-3, maxsteps = 5)
            const S64 = SY.System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}
            const S32 = SY.System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}
        end,
        new_fire2_64(Int),
        new_fire2_32(Int),
        IN.fire2_step!(IN.FIRE2{Float64}, Matrix{Float64}, Matrix{Float64}),
        IN.fire2_step!(IN.FIRE2{Float32}, Matrix{Float32}, Matrix{Float32}),
        relax64(S64, PP.LennardJones{Float64}, IN.FIRE2{Float64}),
        relax32(S32, PP.LennardJones{Float32}, IN.FIRE2{Float32}),
        relax64(S64, PP.MACEPotential{Float64}, IN.FIRE2{Float64}),
        relax32(S32, PP.MACEPotential{Float32}, IN.FIRE2{Float32}),
    )
end

@testitem "FIRE2 return types are concrete" tags = [:checks] setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: FIRE2, fire2_step!
    F = RelaxFixtures
    for T in (Float64, Float32)
        sys, _, p = F.lj_crystal(T)
        opt = FIRE2(sys)
        @test Base.return_types(FIRE2{T}, (Int,))[1] === FIRE2{T}
        @test Base.return_types(relax!, (typeof(sys), typeof(p), typeof(opt)))[1] === RelaxResult{T}
        @test Base.return_types(fire2_step!, (typeof(opt), typeof(sys.positions), typeof(sys.positions)))[1] === Bool
    end
end
