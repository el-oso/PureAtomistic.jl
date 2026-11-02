@testmodule VerletFixtures begin
    using PureAtomistic
    using PureAtomistic.Systems
    using PureMLIP
    using Random, Statistics

    # The atomic mass unit in eV fs² / Å², written out so that the unit factor is not checked
    # against the constant it tests.
    const AMU_LITERAL = 103.64269652680504

    # A face-centered cubic crystal of 32 atoms of mass 39.948 amu, lattice constant 5.26 Å
    # (nearest-neighbor distance 1.09σ for σ = 3.4 Å), displaced by 0.03 Å per component, with
    # velocities at 10 K. The cutoff 6.85 Å lies in the gap between the third (6.44 Å) and fourth
    # (7.44 Å) neighbor shells, so no pair crosses it and the edge count stays 1344.
    function lj_simulation(::Type{T}, dt; temp = 10.0, seed = 1, kick = nothing) where {T}
        sys = fcc_cell(T(5.26), 2, :Cu; mass = 39.948)
        sys.positions .+= T(0.03) .* randn(Xoshiro(3), T, 3, size(sys.positions, 2))
        maxwell_boltzmann!(sys, temp, Xoshiro(seed))
        if !isnothing(kick)
            sys.positions[1, kick[1]] = T(0.05)
            sys.velocities[1, kick[1]] = T(kick[2])
        end
        p = LennardJones(copy(sys.positions), copy(sys.cell); epsilon = 0.0104, sigma = 3.4, r_cut = 6.85)
        return Simulation(sys, p, VelocityVerlet(dt); rng = Xoshiro(0))
    end

    # Copper atoms at 300 K tied to the sites of a 3.6 Å fcc lattice by springs of 0.5 eV/Å²
    # (angular frequency 8.7e-3 / fs, so ω dt <= 0.07 for dt <= 8 fs).
    function harmonic_simulation(::Type{T}, dt; temp = 300.0) where {T}
        sys = fcc_cell(T(3.6), 2, :Cu)
        sites = copy(sys.positions)
        sys.positions .+= T(0.05) .* randn(Xoshiro(3), T, 3, size(sys.positions, 2))
        maxwell_boltzmann!(sys, temp, Xoshiro(1))
        return Simulation(sys, HarmonicPotential(sites, 0.5), VelocityVerlet(dt); rng = Xoshiro(0))
    end

    # The tiny checkpoint's model on three atoms (two Cu, one H) in a triclinic cell, at 300 K.
    function mace_simulation(::Type{T}, dt) where {T}
        m = MACEModel{T}(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint()))
        cell = T[6.5 0.0 0.0; 0.4 6.2 0.0; 0.3 0.2 6.8]
        x = T[0.0 1.4 0.2; 0.0 0.1 1.3; 0.0 0.3 0.1]
        z = [29, 1, 29]
        sys = System{T}(copy(x), zeros(T, 3, 3), T[63.546, 1.008, 63.546], z, cell)
        maxwell_boltzmann!(sys, 300.0, Xoshiro(1))
        return Simulation(sys, MACEPotential(m, copy(x), cell, z), VelocityVerlet(dt); rng = Xoshiro(0))
    end

    # Standard deviation of the total energy over a run of `total` fs, sampled every step.
    function energy_std(make, dt, total)
        sim = make(Float64, dt)
        E = Float64[]
        run!(sim, round(Int, total / dt); observer = s -> push!(E, total_energy(s)))
        return std(E)
    end

    # Least-squares slope of log y against log x.
    function loglog_slope(x, y)
        lx, ly = log.(x), log.(y)
        return sum((lx .- mean(lx)) .* (ly .- mean(ly))) / sum((lx .- mean(lx)) .^ 2)
    end
end

@testitem "VelocityVerlet and Simulation reject invalid arguments" setup = [VerletFixtures] begin
    using Random
    F = VerletFixtures
    @test_throws "pureatomistic: the time step must be positive and finite, got 0.0" VelocityVerlet(0.0)
    @test_throws "pureatomistic: the time step must be positive and finite, got -1.0" VelocityVerlet(-1.0)
    @test_throws "pureatomistic: the time step must be positive and finite, got Inf" VelocityVerlet(Inf)
    @test VelocityVerlet(2).dt === 2.0

    sim = F.harmonic_simulation(Float64, 1.0)
    sys, pot, vv = sim.system, sim.potential, sim.integrator
    @test_throws "pureatomistic: the driver supports only thermostat = nothing and barostat = nothing" Simulation(
        sys, pot, vv; thermostat = 1, rng = Xoshiro(0)
    )
    @test_throws "pureatomistic: the driver supports only thermostat = nothing and barostat = nothing" Simulation(
        sys, pot, vv; barostat = 1, rng = Xoshiro(0)
    )
    small = HarmonicPotential(sys.positions[:, 1:5], 0.5)
    @test_throws "pureatomistic: the potential has 5 atoms, the system has 32" Simulation(sys, small, vv; rng = Xoshiro(0))
    @test_throws UndefKeywordError Simulation(sys, pot, vv)
    @test_throws "pureatomistic: nsteps must not be negative, got -1" run!(sim, -1)
    @test_throws "pureatomistic: every must be at least 1, got 0" run!(sim, 1; every = 0)
    # A force matrix that does not match the velocities is refused.
    @test_throws "pureatomistic: forces axes" PureAtomistic.Integrators._kick!(
        sys.velocities, zeros(3, 4), sys.masses, 1.0
    )
end

@testitem "one velocity Verlet step matches the explicit formula" setup = [VerletFixtures] begin
    using PureAtomistic.Systems
    using Random
    F = VerletFixtures
    # One atom of mass m on a spring k tied to the origin of a large cell. Oracle: the kick-drift-
    # kick update written out, with the unit factor as a literal. Measured largest deviation:
    # 0.25 eps(T) in x (values near 0.3 Å) and 0.002 eps(T) in v (values near 0.003 Å/fs) in both
    # types; the bounds are 16 and 20 times those.
    for T in (Float64, Float32)
        m, k, dt = T(63.546), T(0.5), T(2)
        x0, v0 = T[0.3, -0.2, 0.1], T[0.002, 0.001, -0.003]
        sys = System{T}(reshape(copy(x0), 3, 1), reshape(copy(v0), 3, 1), [m], [29], T[50 0 0; 0 50 0; 0 0 50])
        sim = Simulation(sys, HarmonicPotential(zeros(T, 3, 1), k), VelocityVerlet(dt); rng = Xoshiro(0))
        c = T(F.AMU_LITERAL)
        a0 = -k .* x0 ./ (m * c)
        x1 = x0 .+ dt .* v0 .+ dt^2 / 2 .* a0
        a1 = -k .* x1 ./ (m * c)
        v1 = v0 .+ dt / 2 .* (a0 .+ a1)
        step!(sim)
        @test sim.step_count == 1
        @test maximum(abs, vec(sys.positions) .- x1) <= 4 * eps(T)
        @test maximum(abs, vec(sys.velocities) .- v1) <= eps(T) / 25
    end
end

@testitem "velocity Verlet follows the exact discrete oscillator" setup = [VerletFixtures] begin
    using PureAtomistic.Systems
    using Random
    F = VerletFixtures
    # For v(0) = 0 on a spring, velocity Verlet gives x_n = x_0 cos(nθ) with cos θ = 1 - (ω dt)²/2
    # exactly (the recurrence x_{n+1} = (2 - (ω dt)²) x_n - x_{n-1}, x_1 = x_0 (1 - (ω dt)²/2)),
    # that is θ = 2 asin(ω dt / 2). Measured largest error over 2000 steps: 1.7e-15 Å at
    # x_0 = 0.3 Å; the bound is 21 times that.
    m, k, dt, x0 = 63.546, 0.5, 2.0, 0.3
    ω = sqrt(k / (m * F.AMU_LITERAL))
    θ = 2 * asin(ω * dt / 2)
    sys = System{Float64}(fill(x0, 3, 1), zeros(3, 1), [m], [29], [50.0 0 0; 0 50 0; 0 0 50])
    sim = Simulation(sys, HarmonicPotential(zeros(3, 1), k), VelocityVerlet(dt); rng = Xoshiro(0))
    worst = let worst = 0.0
        for n in 1:2000
            step!(sim)
            worst = max(worst, abs(sys.positions[1, 1] - x0 * cos(n * θ)))
        end
        worst
    end
    @test worst <= 3.5e-14
end

@testitem "total-energy fluctuation scales as dt squared" setup = [VerletFixtures] begin
    using Random
    F = VerletFixtures
    dts = [1.0, 2.0, 4.0, 8.0]
    # Velocity Verlet conserves a shadow Hamiltonian that differs from H by O(dt²), so the
    # standard deviation of E over a fixed time span scales as dt². Measured log-log slopes over
    # dt = 1, 2, 4, 8 fs over 4000 fs: 2.0002 (Lennard-Jones) and 2.0008 (harmonic);
    # the brief's bound is 2 ± 0.2.
    for make in (F.lj_simulation, F.harmonic_simulation)
        sd = [F.energy_std(make, dt, 4000.0) for dt in dts]
        @test all(>(0), sd)
        @test abs(F.loglog_slope(dts, sd) - 2) <= 0.2
    end
end

@testitem "velocity Verlet is time reversible" setup = [VerletFixtures] begin
    using Random
    F = VerletFixtures
    # n steps, flip v, n steps return to (x0, -v0). Rounding bounds from the measured largest
    # deviation after n = 1000 steps of 2 fs (Lennard-Jones 5.3e-15 Å and 5.3e-17 Å/fs, harmonic
    # 4.0e-15 Å and 4.5e-17 Å/fs): positions 1e-13 Å (19x), velocities 1e-15 Å/fs (19x).
    for make in (F.lj_simulation, F.harmonic_simulation)
        sim = make(Float64, 2.0)
        x0, v0 = copy(sim.system.positions), copy(sim.system.velocities)
        run!(sim, 1000)
        @test maximum(abs, sim.system.positions .- x0) > 1.0e-4  # the system moved
        sim.system.velocities .*= -1
        run!(sim, 1000)
        @test maximum(abs, sim.system.positions .- x0) <= 1.0e-13
        @test maximum(abs, sim.system.velocities .+ v0) <= 1.0e-15
    end
end

@testitem "velocity Verlet conserves total momentum" setup = [VerletFixtures] begin
    using PureAtomistic.Systems
    using Random
    F = VerletFixtures
    # Pair forces sum to zero, so momentum is conserved up to rounding. Measured drift over 2000
    # steps: 8.0e-16 amu Å/fs (the momentum of one atom at the largest speed is 0.047); bound 20x.
    sim = F.lj_simulation(Float64, 2.0)
    p0 = collect(total_momentum(sim.system))
    run!(sim, 2000)
    @test maximum(abs, collect(total_momentum(sim.system)) .- p0) <= 1.6e-14
end

@testitem "Float32 runs do not drift beyond rounding" setup = [VerletFixtures] begin
    using Random
    F = VerletFixtures
    # 20000 steps of 2 fs (40 ps) with the kinetic energy accumulated in Float32. Measured
    # (Lennard-Jones, E = -2.08 eV): largest |E - E0| 7.9e-6 eV in Float32 and 1.6e-6 eV in
    # Float64, final |E - E0| 4.5e-6 eV in Float32. Harmonic: 8.9e-5 eV in Float32 against 8.7e-5 in
    # Float64 (the dt² fluctuation dominates). Bounds are 20 times the Float32 values.
    for (make, maxdev) in ((F.lj_simulation, 1.6e-4), (F.harmonic_simulation, 1.8e-3))
        sim = make(Float32, 2.0)
        E0 = Float64(total_energy(sim))
        worst = 0.0
        run!(sim, 20000; every = 100, observer = s -> (worst = max(worst, abs(Float64(total_energy(s)) - E0))))
        @test total_energy(sim) isa Float32
        @test worst <= maxdev
    end
end

@testitem "NVE on the tiny MACE checkpoint conserves energy" setup = [VerletFixtures] begin
    using Random
    F = VerletFixtures
    # 50 steps of 0.5 fs at 300 K (kinetic energy 0.307 eV). Measured largest |E - E0|: 1.47e-5 eV
    # in Float64 and 1.55e-5 eV in Float32; bound 3e-4 eV (20x).
    for T in (Float64, Float32)
        sim = F.mace_simulation(T, 0.5)
        E0 = Float64(total_energy(sim))
        worst = 0.0
        run!(sim, 50; observer = s -> (worst = max(worst, abs(Float64(total_energy(s)) - E0))))
        @test worst <= 3.0e-4
        @test worst > 0
        @test sim.potential.n_regrows == 0
        @test sim.step_count == 50
    end
end

@testitem "run! calls the observer every k steps" setup = [VerletFixtures] begin
    using Random
    F = VerletFixtures
    sim = F.harmonic_simulation(Float64, 1.0)
    seen = Int[]
    run!(sim, 10; every = 3, observer = s -> push!(seen, s.step_count))
    @test seen == [3, 6, 9]
    run!(sim, 5; every = 5, observer = s -> push!(seen, s.step_count))
    @test seen == [3, 6, 9, 15]
    run!(sim, 0; observer = s -> push!(seen, -1))
    @test seen == [3, 6, 9, 15]
    # Without an observer the run is the same as repeated steps, bit for bit.
    a = F.lj_simulation(Float64, 2.0)
    b = F.lj_simulation(Float64, 2.0)
    run!(a, 20)
    for _ in 1:20
        step!(b)
    end
    @test a.system.positions == b.system.positions
    @test a.system.velocities == b.system.velocities
    @test a.step_count == b.step_count == 20
end

@testitem "Simulation accepts views and rejects offset arrays" setup = [VerletFixtures] begin
    using PureAtomistic.Systems
    using OffsetArrays
    using Random
    F = VerletFixtures
    ref = F.lj_simulation(Float64, 2.0)
    n = natoms(ref.system)
    bigx = zeros(3, n + 4)
    bigv = zeros(3, n + 4)
    bigx[:, 1:n] .= ref.system.positions
    bigv[:, 1:n] .= ref.system.velocities
    sys = System{Float64}(
        view(bigx, :, 1:n), view(bigv, :, 1:n), ref.system.masses, ref.system.species, ref.system.cell
    )
    p = LennardJones(copy(ref.system.positions), copy(ref.system.cell); epsilon = 0.0104, sigma = 3.4, r_cut = 6.85)
    sim = Simulation(sys, p, VelocityVerlet(2.0); rng = Xoshiro(0))
    run!(sim, 10)
    run!(ref, 10)
    # The same arithmetic on the same values: equality is exact.
    @test sys.positions == ref.system.positions
    @test sys.velocities == ref.system.velocities
    @test total_energy(sim) == total_energy(ref)

    osys = System{Float64}(
        OffsetArray(copy(ref.system.positions), 0:2, 0:(n - 1)), OffsetArray(zeros(3, n), 0:2, 0:(n - 1)),
        OffsetArray(copy(ref.system.masses), 0:(n - 1)), OffsetArray(copy(ref.system.species), 0:(n - 1)),
        ref.system.cell
    )
    @test_throws "offset arrays are not supported" Simulation(osys, p, VelocityVerlet(2.0); rng = Xoshiro(0))
end

@testitem "step! does not allocate" tags = [:checks] setup = [VerletFixtures] begin
    using StrictModeTest, StrictMode
    using Random
    F = VerletFixtures
    # The neighbor-list potentials rebuild their list and workspace when a configuration
    # outgrows them (`_regrow!`, registered as an allocation barrier); the harmonic step is
    # proved by AllocCheck directly. The measurement after warm-up shows that the steady step
    # allocates nothing.
    register_alloc_barrier!(PureAtomistic.Potentials._regrow!)
    @noinline stepper!(s) = step!(s)
    @noinline runner!(s) = run!(s, 3)
    @noinline energy(s) = total_energy(s)
    for T in (Float64, Float32)
        for make in (F.harmonic_simulation, F.lj_simulation, F.mace_simulation)
            sim = make === F.mace_simulation ? make(T, 0.5) : make(T, 2.0)
            stepper!(sim)
            @test_noalloc stepper!(sim)
            @test_steady_state stepper!(sim)
            @test_steady_state runner!(sim)
            @test_noalloc energy(sim)
            make === F.harmonic_simulation || @test sim.potential.n_regrows == 0
        end
    end
end

@testitem "TrimCheck: integrator and driver" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureAtomistic, PureAtomistic.Systems, Random
            const SYS64 = System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}
            const SYS32 = System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}
            const SIM64 = Simulation{Float64, SYS64, HarmonicPotential{Float64}, VelocityVerlet, Xoshiro}
            const SIM32 = Simulation{Float32, SYS32, HarmonicPotential{Float32}, VelocityVerlet, Xoshiro}
            const SIML64 = Simulation{Float64, SYS64, LennardJones{Float64}, VelocityVerlet, Xoshiro}
            const SIML32 = Simulation{Float32, SYS32, LennardJones{Float32}, VelocityVerlet, Xoshiro}
            new_vv(dt) = VelocityVerlet(dt)
            new_sim(s, p, dt, rng) = Simulation(s, p, VelocityVerlet(dt); rng)
            step_it(s) = step!(s)
            run_it(s, n) = run!(s, n; every = 2, observer = x -> nothing)
            run_plain(s, n) = run!(s, n)
        end,
        new_vv(Float64),
        new_sim(SYS64, HarmonicPotential{Float64}, Float64, Xoshiro),
        new_sim(SYS32, HarmonicPotential{Float32}, Float64, Xoshiro),
        new_sim(SYS64, LennardJones{Float64}, Float64, Xoshiro),
        new_sim(SYS32, LennardJones{Float32}, Float64, Xoshiro),
        step_it(SIM64),
        step_it(SIM32),
        step_it(SIML64),
        step_it(SIML32),
        run_it(SIM64, Int),
        run_it(SIM32, Int),
        run_plain(SIML64, Int),
        total_energy(SIM64),
        total_energy(SIM32),
        total_energy(SIML64),
        total_energy(SIML32),
    )
end

@testitem "driver return types are concrete and the integrator contract holds" tags = [:checks] begin
    using PureAtomistic.Systems, PureAtomistic.Integrators
    using TypeContracts, Random
    for T in (Float64, Float32)
        sys = fcc_cell(T(3.6), 1, :Cu)
        sim = Simulation(sys, HarmonicPotential(copy(sys.positions), 0.5), VelocityVerlet(1.0); rng = Xoshiro(0))
        @test Base.return_types(step!, (typeof(sim),)) == [Nothing]
        @test Base.return_types(total_energy, (typeof(sim),)) == [T]
        @test Base.return_types(advance!, (VelocityVerlet, typeof(sys), typeof(sim.potential))) == [Nothing]
    end
    @test VelocityVerlet <: AbstractIntegrator
    @verify VelocityVerlet trim_compat = true
    # An integrator without the contract's method fails the check.
    struct NoAdvance <: AbstractIntegrator end
    @test_throws Exception @verify NoAdvance
end

@testitem "step moves atoms across the cell boundary" setup = [VerletFixtures] begin
    using Random
    using PureAtomistic.Systems
    F = VerletFixtures
    # Atom 1 starts 0.05 Å inside the cell and moves at -0.012 Å/fs: its unwrapped x coordinate
    # becomes negative, and pairs with the neighbors it passes cross the cutoff, so the edge
    # count changes (1344 down to 1330 in the measured run). After every step the energy and
    # forces equal those of a potential built from scratch at the current positions. The two
    # evaluations are bitwise equal in the measured run; the bounds are the rounding of a
    # reordered sum, 4 eps N: E over N = 1344 edges of terms up to 0.01 eV gives 1.2e-14 eV, a
    # force over about 42 neighbors of terms up to 0.43 eV/Å gives 1.6e-14 eV/Å.
    sim = F.lj_simulation(Float64, 2.0; kick = (1, -0.012))
    sys, pot = sim.system, sim.potential
    edge_counts = Set{Int}()
    worstE, worstF = let worstE = 0.0, worstF = 0.0
        for _ in 1:40
            step!(sim)
            push!(edge_counts, pot.nl.n_edges)
            fresh = LennardJones(copy(sys.positions), copy(sys.cell); epsilon = 0.0104, sigma = 3.4, r_cut = 6.85)
            prepare!(fresh, sys.positions, sys.cell)
            energy_forces!(fresh, sys.positions, sys.cell)
            worstE = max(worstE, abs(pot.result.energy - fresh.result.energy))
            worstF = max(worstF, maximum(abs, pot.result.forces .- fresh.result.forces))
        end
        (worstE, worstF)
    end
    @test sys.positions[1, 1] < -0.5
    @test length(edge_counts) > 1
    @test worstE <= 1.2e-14
    @test worstF <= 1.6e-14
end
