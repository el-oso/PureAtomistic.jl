@testmodule RelaxFixtures begin
    using PureMLIP
    using PureAtomistic
    using PureAtomistic.Systems
    using PureAtomistic.Potentials
    using LinearAlgebra

    using SHA, Downloads

    # Path of the MACE-MPA-0 checkpoint in PUREMLIP_CHECKPOINT_DIR, downloaded on first use and
    # verified by SHA-256; a file with the wrong digest is deleted.
    function mpa0_path()
        file = "mace-mpa-0-medium.model"
        url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/" * file
        sha = "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"
        dir = get(ENV, "PUREMLIP_CHECKPOINT_DIR", joinpath(first(DEPOT_PATH), "scratchspaces", "PureAtomistic-checkpoints"))
        mkpath(dir)
        path = joinpath(dir, file)
        isfile(path) || (Downloads.download(url, path * ".part"); mv(path * ".part", path; force = true))
        digest = bytes2hex(open(sha256, path))
        if digest != sha
            rm(path)
            error("SHA-256 of $path is $digest, expected $sha; the file was deleted")
        end
        return path
    end

    # Argon-magnitude Lennard-Jones parameters (eV, Å) with a cutoff of 2.5σ, on a 3 × 3 × 3
    # block of fcc cells (108 atoms) whose atoms are displaced by up to 0.15 Å per component.
    # Every atom of the undisplaced lattice feels zero force by symmetry, so the lattice is an
    # equilibrium of the fixed cell and its energy is the oracle for the relaxed energy.
    function lj_crystal(::Type{T}) where {T}
        sys = fcc_cell(T(5.26), 3, :Ag)
        n = natoms(sys)
        perfect = copy(sys.positions)
        sys.positions .+= T(0.15) .* T.(reshape(sin.(1.0:(3n)), 3, n))
        p = LennardJones(sys.positions, sys.cell; epsilon = T(0.0104), sigma = T(3.4), r_cut = 8.5)
        return sys, perfect, p
    end

    function energy_and_forces(p, x, cell)
        prepare!(p, x, cell)
        energy_forces!(p, x, cell)
        return p.result.energy, copy(p.result.forces)
    end

    max_force(F) = maximum(norm, eachcol(F))

    tiny_model(::Type{T}) where {T} = MACEModel{T}(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint()))

    # One atom at the center of a 20 Å cell with a constant force of 1e-9 eV/Å along x and an
    # energy of 1 eV plus the distance from its start: every move raises the energy.
    mutable struct JumpPotential <: AbstractPotential
        result::PotentialResult{Float64}
        start::Matrix{Float64}
    end
    Potentials.n_atoms(::JumpPotential) = 1
    Potentials.cutoff(::JumpPotential) = 0.0
    Potentials.prepare!(::JumpPotential, ::AbstractMatrix, ::AbstractMatrix) = nothing
    function Potentials.energy_forces!(p::JumpPotential, x::AbstractMatrix, ::AbstractMatrix)
        p.result.energy = 1.0 + norm(x - p.start)
        p.result.forces .= 0
        p.result.forces[1, 1] = 1.0e-9
        return nothing
    end
    function jump_system()
        x = fill(10.0, 3, 1)
        sys = System{Float64}(copy(x), zeros(3, 1), [1.0], [1], Matrix(20.0I, 3, 3))
        return sys, JumpPotential(PotentialResult{Float64}(1), x)
    end

    # A potential whose energy is NaN, to exercise the non-finite check.
    mutable struct NaNPotential <: AbstractPotential
        result::PotentialResult{Float64}
    end
    Potentials.n_atoms(p::NaNPotential) = size(p.result.forces, 2)
    Potentials.cutoff(::NaNPotential) = 0.0
    Potentials.prepare!(::NaNPotential, ::AbstractMatrix, ::AbstractMatrix) = nothing
    function Potentials.energy_forces!(p::NaNPotential, ::AbstractMatrix, ::AbstractMatrix)
        p.result.energy = NaN
        return nothing
    end
end

@testitem "fire_step! applies the FIRE rules" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: FIRE, fire_step!
    T = Float64
    # Hand-derived single steps of the paper's rules with unit masses (no step cap: maxstep = 10).
    # P > 0: F = (3, 4, 0), v = (2, 0, 0), α = 0.1, dt = 0.1, so |v| = 2, |F| = 5 and
    #   v ← 0.9 v + 0.1 · 2 · F/5 = (1.92, 0.16, 0);  v += dt F → (2.22, 0.56, 0);  x += dt v.
    opt = FIRE{T}(1; dt = 0.1, nmin = 5, maxstep = 10.0)
    opt.velocity .= [2.0, 0.0, 0.0]
    x = zeros(T, 3, 1)
    fire_step!(opt, x, reshape([3.0, 4.0, 0.0], 3, 1))
    # Rounding bound: about eight operations per component, c = 8, N = 1 (values of order 1).
    @test x ≈ reshape([0.222, 0.056, 0.0], 3, 1) rtol = 8 * eps(T)
    @test opt.velocity ≈ reshape([2.22, 0.56, 0.0], 3, 1) rtol = 8 * eps(T)
    @test opt.npos == 1
    @test opt.dt == 0.1 && opt.alpha == 0.1
    # P ≤ 0: v = (2, 0, 0), F = (-3, 0, 0): freeze, dt ← dt/2, α ← α_start, then v = dt F, x += dt v.
    opt = FIRE{T}(1; dt = 0.1, nmin = 5, maxstep = 10.0)
    opt.velocity .= [2.0, 0.0, 0.0]
    opt.npos = 4
    opt.alpha = 0.05
    x = zeros(T, 3, 1)
    fire_step!(opt, x, reshape([-3.0, 0.0, 0.0], 3, 1))
    @test opt.dt == 0.05 && opt.alpha == 0.1 && opt.npos == 0
    @test opt.velocity ≈ reshape([-0.15, 0.0, 0.0], 3, 1) rtol = 4 * eps(T)
    @test x ≈ reshape([-0.0075, 0.0, 0.0], 3, 1) rtol = 4 * eps(T)
    # Zero power (v ⟂ F) freezes as well: the rule is P ≤ 0.
    opt = FIRE{T}(1; dt = 0.1, nmin = 5, maxstep = 10.0)
    opt.velocity .= [0.0, 1.0, 0.0]
    fire_step!(opt, zeros(T, 3, 1), reshape([1.0, 0.0, 0.0], 3, 1))
    @test opt.dt == 0.05 && opt.velocity == reshape([0.05, 0.0, 0.0], 3, 1)
end

@testitem "FIRE time step and mixing schedule" setup = [RelaxFixtures] begin
    using LinearAlgebra
    using PureAtomistic.Integrators: FIRE, fire_step!
    T = Float64
    # With the force along +x and the velocity along +x the power stays positive. The first
    # nmin = 5 steps leave dt and α alone; every later step multiplies dt by finc (until dtmax)
    # and α by falpha.
    opt = FIRE{T}(1; dt = 0.1, nmin = 5, maxstep = 1.0e3)
    opt.velocity .= [1.0, 0.0, 0.0]
    x = zeros(T, 3, 1)
    F = reshape([1.0, 0.0, 0.0], 3, 1)
    for k in 1:5
        fire_step!(opt, x, F)
        @test opt.dt == 0.1 && opt.alpha == 0.1 && opt.npos == k
    end
    for k in 1:4
        fire_step!(opt, x, F)
        # Powers of 1.1 and 0.99 carry rounding of about 2k operations: c = 2, N = k.
        @test opt.dt ≈ 0.1 * 1.1^k rtol = 2 * k * eps(T)
        @test opt.alpha ≈ 0.1 * 0.99^k rtol = 2 * k * eps(T)
    end
    for _ in 1:100
        fire_step!(opt, x, F)
    end
    @test opt.dt == 1.0   # dtmax, reached through min(): exact
    # The displacement cap: one atom moving 10 Å in a step is scaled to maxstep exactly.
    opt = FIRE{T}(1; dt = 0.1, nmin = 5, maxstep = 0.1)
    opt.velocity .= [100.0, 0.0, 0.0]
    x = zeros(T, 3, 1)
    fire_step!(opt, x, F)
    @test norm(x) ≈ 0.1 rtol = 4 * eps(T)
end

@testitem "FIRE relaxes a displaced Lennard-Jones crystal" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: FIRE
    F = RelaxFixtures
    # Oracles: a fresh potential evaluated at the returned positions gives the forces and the
    # energy, and the undisplaced lattice is an equilibrium whose energy the relaxed energy
    # approaches from above, in Float32 to within the rounding of its energy. Measured at
    # fmax = 1e-3, 41 steps: E - E_lattice = 6.8e-5 eV (Float64) and -1.1e-4 eV (Float32). The bound
    # above is 1e-3 (15 times the Float64 value); below, 1e-4 in Float64 and 1e-3 in Float32
    # (9 times the Float32 value).
    for (T, fmax, ebound, below) in ((Float64, 1.0e-3, 1.0e-3, 1.0e-4), (Float32, 1.0e-3, 1.0e-3, 1.0e-3))
        sys, perfect, p = F.lj_crystal(T)
        E0 = F.energy_and_forces(F.lj_crystal(T)[3], perfect, sys.cell)[1]
        start = copy(sys.positions)
        info = relax!(sys, p, FIRE(sys); fmax)
        @test info.converged
        @test info isa RelaxResult{T}
        @test 0 < info.steps < 500
        Eref, Fref = F.energy_and_forces(F.lj_crystal(T)[3], sys.positions, sys.cell)
        @test F.max_force(Fref) < fmax
        @test info.max_force < fmax
        @test info.max_force ≈ F.max_force(Fref) rtol = 8 * eps(T)
        # The reported values are those of the final configuration, evaluated by the same code.
        @test info.energy == p.result.energy
        @test info.energy ≈ Eref rtol = 4 * eps(T)
        @test -below < info.energy - E0 < ebound
        @test sys.positions != start
        # Positions are not wrapped: the sum of the displacements stays zero (net force is zero).
        @test abs(sum(sys.positions .- start)) < 100 * sqrt(eps(T))
    end
end

@testitem "FIRE reports the iteration count and resumes consistently" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: FIRE
    F = RelaxFixtures
    sys, perfect, p = F.lj_crystal(Float64)
    x0 = copy(sys.positions)
    full = relax!(sys, p, FIRE(sys); fmax = 1.0e-3)
    xfull = copy(sys.positions)
    # Stopping after k iterations and the full run take the same k first steps.
    sys2, _, p2 = F.lj_crystal(Float64)
    part = relax!(sys2, p2, FIRE(sys2); fmax = 1.0e-3, maxsteps = 10)
    @test !part.converged && part.steps == 10
    sys3, _, p3 = F.lj_crystal(Float64)
    relax!(sys3, p3, FIRE(sys3); fmax = 1.0e-3, maxsteps = 10)
    @test sys3.positions == sys2.positions
    again, _, p4 = F.lj_crystal(Float64)
    @test relax!(again, p4, FIRE(again); fmax = 1.0e-3).steps == full.steps
    @test again.positions == xfull
    # Converging on the last permitted iteration still reports convergence.
    edge, _, p7 = F.lj_crystal(Float64)
    last = relax!(edge, p7, FIRE(edge); fmax = 1.0e-3, maxsteps = full.steps)
    @test last.converged && last.steps == full.steps && edge.positions == xfull
    # Zero iterations only evaluates.
    sys5, _, p5 = F.lj_crystal(Float64)
    zero_steps = relax!(sys5, p5, FIRE(sys5); maxsteps = 0)
    @test zero_steps.steps == 0 && !zero_steps.converged && sys5.positions == x0
    # A configuration that already satisfies fmax is returned untouched.
    sys6, perfect6, p6 = F.lj_crystal(Float64)
    sys6.positions .= perfect6
    done = relax!(sys6, p6, FIRE(sys6); fmax = 1.0e-3)
    @test done.converged && done.steps == 0 && sys6.positions == perfect6
end

@testitem "relax! with a MACE potential" setup = [RelaxFixtures] begin
    using PureMLIP
    using PureAtomistic.Systems
    F = RelaxFixtures
    for T in (Float64, Float32)
        model = F.tiny_model(T)
        sys = PureAtomistic.Systems.fcc_cell(T(3.6), 2, 29)
        sys.positions .+= T(0.05) .* T.(reshape(sin.(1.0:(3 * natoms(sys))), 3, natoms(sys)))
        p = MACEPotential(model, sys.positions, sys.cell, sys.species)
        E_start = F.energy_and_forces(MACEPotential(model, sys.positions, sys.cell, sys.species), sys.positions, sys.cell)[1]
        info = relax!(sys, p; fmax = 1.0e-4, maxsteps = 2000)
        Eref, Fref = F.energy_and_forces(MACEPotential(model, sys.positions, sys.cell, sys.species), sys.positions, sys.cell)
        @test info.converged
        @test F.max_force(Fref) < 1.0e-4
        @test info.steps > 0
        @test info.energy < E_start
        @test info.energy ≈ Eref rtol = 100 * eps(T)
    end
end

@testitem "relaxed vacancy formation energy of Cu with MPA-0" setup = [RelaxFixtures] begin
    using PureMLIP, LinearAlgebra
    using PureAtomistic.Systems
    F = RelaxFixtures
    model = MACEModel(load_mace_checkpoint(F.mpa0_path()))
    mace(sys) = MACEPotential(model, sys.positions, sys.cell, sys.species)
    energy(sys) = F.energy_and_forces(mace(sys), sys.positions, sys.cell)[1]
    # The lattice constant from a parabola through nine energies of the 4-atom cell, as in the
    # PureMLIP vacancy example.
    as = range(3.6, 3.68; length = 9)
    c = [ones(length(as)) as as .^ 2] \ [energy(fcc_cell(a, 1, :Cu)) / 4 for a in as]
    a0 = -c[2] / (2c[3])
    perfect = fcc_cell(a0, 4, :Cu)
    N = natoms(perfect)
    E_N = energy(perfect)
    v = argmin(j -> norm(perfect.positions[:, j] - fill(2a0, 3)), 1:N)
    defect = remove_atom(perfect, v)
    p = mace(defect)
    unrelaxed = F.energy_and_forces(p, defect.positions, defect.cell)
    info = relax!(defect, p; fmax = 1.0e-3)
    E_vac = info.energy - (N - 1) / N * E_N
    Eref, Fref = F.energy_and_forces(mace(defect), defect.positions, defect.cell)
    @test info.converged
    @test F.max_force(Fref) < 1.0e-3
    @test info.energy < unrelaxed[1]
    # Oracle: the example's 1.068 eV (its FIRE stops at 0.01 eV/Å, 3 digits shown). Measured here
    # with the default relaxer: 1.066942 eV at fmax = 1e-3, so the distance to 1.068 is 1.1e-3;
    # the bound 1e-2 is 9 times that and far below the 0.1 eV that a wrong convention
    # (for example the unrelaxed value) would move it by.
    @test abs(E_vac - 1.068) < 1.0e-2
    # Relaxation lowers the formation energy; measured 0.031 eV, the bound 0.01 eV is a third of it.
    unrelaxed_vac = unrelaxed[1] - (N - 1) / N * E_N
    @test unrelaxed_vac - E_vac > 0.01
    @test 1.0 < E_vac < 1.1
end

@testitem "TrustRegionBFGS agrees with FIRE on the MPA-0 vacancy" setup = [RelaxFixtures] begin
    using PureMLIP, LinearAlgebra
    using PureAtomistic.Systems
    using PureAtomistic.Integrators: TrustRegionBFGS
    F = RelaxFixtures
    # Measured at fmax = 1e-3: E_vac = 1.066952 eV (FIRE, 24 steps) and 1.066942 eV (TrustRegionBFGS,
    # 6 steps); the relaxed energies differ by 1.0e-5 eV, bound 1e-4 (10 times).
    model = MACEModel(load_mace_checkpoint(F.mpa0_path()))
    mace(sys) = MACEPotential(model, sys.positions, sys.cell, sys.species)
    a0 = 3.640391   # the model's lattice constant to six digits (parabola fit, 4-atom cell)
    perfect = fcc_cell(a0, 4, :Cu)
    N = natoms(perfect)
    v = argmin(j -> norm(perfect.positions[:, j] - fill(2a0, 3)), 1:N)
    a = remove_atom(perfect, v)
    b = remove_atom(perfect, v)
    ra = relax!(a, mace(a), FIRE(a); fmax = 1.0e-3)
    rb = relax!(b, mace(b), TrustRegionBFGS(b); fmax = 1.0e-3)
    @test ra.converged && rb.converged
    @test rb.steps < ra.steps
    @test abs(ra.energy - rb.energy) < 1.0e-4
    @test F.max_force(F.energy_and_forces(mace(b), b.positions, b.cell)[2]) < 1.0e-3
end

@testitem "relax! and FIRE reject invalid input" setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: FIRE
    using PureAtomistic.Systems
    F = RelaxFixtures
    sys, _, p = F.lj_crystal(Float64)
    @test_throws "pureatomistic: fmax must be positive and finite, got 0.0" relax!(sys, p; fmax = 0.0)
    @test_throws "pureatomistic: fmax must be positive and finite, got NaN" relax!(sys, p; fmax = NaN)
    @test_throws "pureatomistic: maxsteps must not be negative, got -1" relax!(sys, p; maxsteps = -1)
    small = fcc_cell(5.26, 1, :Ag)
    @test_throws "pureatomistic: the potential has 108 atoms, the system 4" relax!(small, p)
    @test_throws "pureatomistic: the FIRE state has 4 atoms, the system 108" relax!(sys, p, FIRE(small))
    nanp = F.NaNPotential(PotentialResult{Float64}(natoms(sys)))
    @test_throws "pureatomistic: relax! met a non-finite energy or force after 0 steps" relax!(sys, nanp)
    @test_throws "pureatomistic: FIRE dt must be positive and finite, got 0.0" FIRE(sys; dt = 0.0)
    @test_throws "pureatomistic: FIRE dtmax must be finite and at least dt = 1.0, got 0.05" FIRE(sys; dtmax = 0.05)
    @test_throws "pureatomistic: FIRE maxstep must be positive and finite, got -1.0" FIRE(sys; maxstep = -1.0)
    @test_throws "pureatomistic: FIRE nmin must not be negative, got -1" FIRE(sys; nmin = -1)
    @test_throws "pureatomistic: FIRE finc must exceed 1, got 1.0" FIRE(sys; finc = 1.0)
    @test_throws "pureatomistic: FIRE fdec must lie in (0, 1), got 1.0" FIRE(sys; fdec = 1.0)
    @test_throws "pureatomistic: FIRE alpha_start must lie in (0, 1], got 0.0" FIRE(sys; alpha_start = 0.0)
    @test_throws "pureatomistic: FIRE falpha must lie in (0, 1], got 1.5" FIRE(sys; falpha = 1.5)
    @test_throws "pureatomistic: the number of atoms must not be negative, got -1" FIRE{Float64}(-1)
    @test_throws "pureatomistic: positions" PureAtomistic.Integrators.fire_step!(FIRE{Float64}(2), zeros(3, 3), zeros(3, 3))
end

@testitem "relax! on views and OffsetArrays" setup = [RelaxFixtures] begin
    using OffsetArrays
    using PureAtomistic.Systems
    F = RelaxFixtures
    sys, _, p = F.lj_crystal(Float64)
    n = natoms(sys)
    base = copy(sys.positions)
    ref = relax!(sys, p; fmax = 1.0e-3)
    # A view of a larger parent: the same arithmetic, so equality is exact.
    parent = zeros(3, n + 5)
    parent[:, 1:n] .= base
    vsys = System{Float64}(view(parent, :, 1:n), zeros(3, n), copy(sys.masses), copy(sys.species), copy(sys.cell))
    vp = LennardJones(vsys.positions, vsys.cell; epsilon = 0.0104, sigma = 3.4, r_cut = 8.5)
    vinfo = relax!(vsys, vp; fmax = 1.0e-3)
    @test vinfo == ref
    @test vsys.positions == sys.positions
    @test parent[:, (n + 1):end] == zeros(3, 5)
    # An offset atom axis is rejected up front.
    osys = System{Float64}(
        OffsetArray(copy(base), 1:3, 0:(n - 1)), OffsetArray(zeros(3, n), 1:3, 0:(n - 1)),
        OffsetArray(copy(sys.masses), 0:(n - 1)), OffsetArray(copy(sys.species), 0:(n - 1)), copy(sys.cell)
    )
    @test_throws "offset arrays are not supported" relax!(osys, p)
end

@testitem "relax! does not allocate" tags = [:checks] setup = [RelaxFixtures] begin
    using PureMLIP
    using PureAtomistic.Systems
    using StrictModeTest, StrictMode
    using PureAtomistic.Integrators: FIRE, fire_step!
    F = RelaxFixtures
    register_alloc_barrier!(PureAtomistic.Potentials._regrow!)
    @noinline step!(opt, x, f) = fire_step!(opt, x, f)
    @noinline run!(sys, p, opt) = relax!(sys, p, opt; fmax = 1.0e-3, maxsteps = 5)
    for T in (Float64, Float32)
        sys, _, lj = F.lj_crystal(T)
        msys = PureAtomistic.Systems.fcc_cell(T(3.6), 2, 29)
        msys.positions .+= T(0.05) .* T.(reshape(sin.(1.0:(3 * natoms(msys))), 3, natoms(msys)))
        mace = MACEPotential(F.tiny_model(T), msys.positions, msys.cell, msys.species)
        for (s, p) in ((sys, lj), (msys, mace))
            opt = FIRE(s)
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

@testitem "TrimCheck: relax!" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureAtomistic, PureMLIP
            const SY = PureAtomistic.Systems
            const PP = PureAtomistic.Potentials
            const IN = PureAtomistic.Integrators
            new_fire64(n) = IN.FIRE{Float64}(n)
            new_fire32(n) = IN.FIRE{Float32}(n)
            relax64(s, p, o) = IN.relax!(s, p, o; fmax = 1.0e-3, maxsteps = 5)
            relax32(s, p, o) = IN.relax!(s, p, o; fmax = 1.0e-3, maxsteps = 5)
            relax_default64(s, p) = IN.relax!(s, p; fmax = 1.0e-3, maxsteps = 5)
            relax_default32(s, p) = IN.relax!(s, p; fmax = 1.0e-3, maxsteps = 5)
            const S64 = SY.System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}
            const S32 = SY.System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}
        end,
        new_fire64(Int),
        new_fire32(Int),
        IN.fire_step!(IN.FIRE{Float64}, Matrix{Float64}, Matrix{Float64}),
        IN.fire_step!(IN.FIRE{Float32}, Matrix{Float32}, Matrix{Float32}),
        relax64(S64, PP.LennardJones{Float64}, IN.FIRE{Float64}),
        relax32(S32, PP.LennardJones{Float32}, IN.FIRE{Float32}),
        relax64(S64, PP.MACEPotential{Float64}, IN.FIRE{Float64}),
        relax32(S32, PP.MACEPotential{Float32}, IN.FIRE{Float32}),
        relax_default64(S64, PP.LennardJones{Float64}),
        relax_default32(S32, PP.LennardJones{Float32}),
    )
end

@testitem "relax! return types are concrete" tags = [:checks] setup = [RelaxFixtures] begin
    using PureAtomistic.Integrators: FIRE, fire_step!
    F = RelaxFixtures
    for T in (Float64, Float32)
        sys, _, p = F.lj_crystal(T)
        opt = FIRE(sys)
        @test Base.return_types(relax!, (typeof(sys), typeof(p), typeof(opt)))[1] === RelaxResult{T}
        @test Base.return_types(fire_step!, (typeof(opt), typeof(sys.positions), typeof(sys.positions)))[1] === Nothing
    end
end

@testitem "relax! of a system without atoms converges at once" setup = [RelaxFixtures] begin
    using LinearAlgebra
    using PureAtomistic.Systems
    using PureAtomistic.Integrators: TrustRegionLBFGS
    using PureAtomistic.Potentials: LennardJones
    for T in (Float64, Float32)
        cell = Matrix{T}(20I, 3, 3)
        sys = System{T}(zeros(T, 3, 0), zeros(T, 3, 0), T[], Int[], cell)
        p = LennardJones(sys.positions, sys.cell; epsilon = T(0.0104), sigma = T(3.4), r_cut = 8.5)
        res = relax!(sys, p)
        @test res.converged && res.steps == 0
        @test_throws "pureatomistic: TrustRegionLBFGS needs at least one atom, got 0" TrustRegionLBFGS(sys)
    end
end
