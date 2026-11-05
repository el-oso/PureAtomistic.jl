@testmodule ThermostatFixtures begin
    using PureAtomistic
    using PureAtomistic.Systems
    using PureAtomistic.Thermostats
    using Random, Statistics
    import PureAtomistic: n_atoms, cutoff, prepare!, energy_forces!

    # k_B in eV/K and the atomic mass unit in eV fs² / Å², written out so that the unit factors
    # are not checked against the constants they test.
    const KB_LITERAL = 8.617333262145179e-5
    const AMU_LITERAL = 103.64269652680504

    # Free particles: zero energy and forces. The potential counts its evaluations.
    struct FreeParticles{T} <: AbstractPotential
        result::PotentialResult{T}
        evaluations::Base.RefValue{Int}
    end
    FreeParticles{T}(n::Integer) where {T} = FreeParticles{T}(PotentialResult{T}(n), Ref(0))
    n_atoms(p::FreeParticles) = size(p.result.forces, 2)
    cutoff(::FreeParticles) = 0.0
    prepare!(::FreeParticles, positions, cell) = nothing
    function energy_forces!(p::FreeParticles, positions, cell)
        p.evaluations[] += 1
        return nothing
    end

    # An ideal gas of `n` argon-mass atoms in a 20 Å cube, with velocities at `temp` K and zero
    # center-of-mass momentum, so `3n - 3` degrees of freedom.
    function ideal_gas(::Type{T}, n, thermostat; temp = 300.0, dt = 2.0, seed = 1, rng = Xoshiro(7)) where {T}
        x = 20 .* rand(Xoshiro(seed + 100), T, 3, n)
        sys = System{T}(x, zeros(T, 3, n), fill(T(39.948), n), fill(18, n), T[20 0 0; 0 20 0; 0 0 20])
        maxwell_boltzmann!(sys, temp, Xoshiro(seed))
        return Simulation(sys, FreeParticles{T}(n), VelocityVerlet(dt); thermostat, rng)
    end

    # Mean of `x` and its standard error from the means of `nb` consecutive blocks.
    function block_mean(x, nb)
        L = length(x) ÷ nb
        b = [mean(@view x[((i - 1) * L + 1):(i * L)]) for i in 1:nb]
        return mean(b), std(b) / sqrt(nb)
    end

    # Mean of the variances of `nb` consecutive blocks of `x` and its standard error.
    function block_var(x, nb)
        L = length(x) ÷ nb
        b = [var(@view x[((i - 1) * L + 1):(i * L)]) for i in 1:nb]
        return mean(b), std(b) / sqrt(nb)
    end

    # The kinetic energy of an 11-atom ideal gas (N_f = 30) under the Bussi thermostat at 300 K,
    # τ = 1 fs, dt = 2 fs, sampled every 5 steps over 100000 steps.
    function bussi_ideal_gas_samples()
        sim = ideal_gas(Float64, 11, BussiThermostat(300.0, 1.0))
        K = Float64[]
        run!(sim, 100_000; every = 5, observer = s -> push!(K, kinetic_energy(s.system)))
        return K
    end

    # The z scores of the four quantiles `ps` of `K` against an independent sample of
    # Gamma(N_f/2, kT) drawn as kT/2 times a sum of N_f squared standard normals; the standard
    # error of each difference comes from `B` bootstrap resamples of both samples.
    function quantile_z(K, nf, kT; ps = (0.1, 0.25, 0.5, 0.9), M = 100_000, B = 100)
        rng = Xoshiro(99)
        ref = [kT / 2 * sum(abs2, randn(rng, nf)) for _ in 1:M]
        d = quantile(K, collect(ps)) .- quantile(ref, collect(ps))
        brng = Xoshiro(5)
        boot = [
            quantile(rand(brng, K, length(K)), collect(ps)) .- quantile(rand(brng, ref, M), collect(ps))
                for _ in 1:B
        ]
        se = [std(getindex.(boot, j)) for j in eachindex(d)]
        return d ./ se
    end
end

@testitem "thermostats reject invalid arguments" setup = [VerletFixtures, ThermostatFixtures] begin
    using PureAtomistic.Thermostats
    using Random
    using PureAtomistic.Systems
    using OffsetArrays
    @test_throws "pureatomistic: the temperature must be non-negative and finite, got -1.0" LangevinBAOAB(-1.0, 0.01)
    @test_throws "pureatomistic: the temperature must be non-negative and finite, got NaN" LangevinBAOAB(NaN, 0.01)
    @test_throws "pureatomistic: the friction must be non-negative and finite, got -0.1" LangevinBAOAB(300, -0.1)
    @test_throws "pureatomistic: the friction must be non-negative and finite, got Inf" LangevinBAOAB(300, Inf)
    @test LangevinBAOAB(300, 0) isa LangevinBAOAB
    @test_throws "pureatomistic: the temperature must be non-negative and finite, got Inf" BussiThermostat(Inf, 10.0)
    @test_throws "pureatomistic: the coupling time must be positive, got 0.0" BussiThermostat(300, 0.0)
    @test_throws "pureatomistic: the coupling time must be positive, got NaN" BussiThermostat(300, NaN)
    @test_throws "pureatomistic: removed_dof must not be negative, got -1" BussiThermostat(300, 10.0; removed_dof = -1)
    @test BussiThermostat(300, Inf).tau === Inf
    @test_throws "pureatomistic: the gamma shape must be positive and finite, got 0.0" randgamma(Xoshiro(1), 0.0)
    @test_throws "pureatomistic: the gamma shape must be positive and finite, got -2" randgamma(Xoshiro(1), -2)
    @test_throws "pureatomistic: the gamma shape must be positive and finite, got Inf" randgamma(Xoshiro(1), Inf)

    sim = VerletFixtures.harmonic_simulation(Float64, 1.0)
    struct NoStepIntegrator <: AbstractIntegrator end
    @test_throws "pureatomistic: a thermostat needs the VelocityVerlet integrator, got" Simulation(
        sim.system, sim.potential, NoStepIntegrator(); thermostat = LangevinBAOAB(300, 0.01), rng = Xoshiro(0)
    )
    @test_throws "pureatomistic: forces axes" PureAtomistic.Thermostats._kick!(
        sim.system.velocities, zeros(3, 4), sim.system.masses, 1.0
    )
    # The BAOAB step indexes the masses by the velocity columns, so it declares 1-based arrays.
    s = sim.system
    n = natoms(s)
    osys = System{Float64}(
        OffsetArray(copy(s.positions), 0:2, 0:(n - 1)), OffsetArray(copy(s.velocities), 0:2, 0:(n - 1)),
        OffsetArray(copy(s.masses), 0:(n - 1)), OffsetArray(copy(s.species), 0:(n - 1)), s.cell
    )
    @test_throws "offset arrays are not supported" thermostat_step!(
        LangevinBAOAB(300, 0.01), osys, sim.potential, 1.0, Xoshiro(0), nothing
    )
end

@testitem "Marsaglia-Tsang gamma sampler has the gamma moments" setup = [ThermostatFixtures] begin
    using PureAtomistic.Thermostats
    using Random, Statistics
    # Gamma(a, 1) has mean a and variance a; the sample variance of N draws has standard error
    # √((μ₄ - a²)/N) with the central fourth moment μ₄ = 3a² + 6a. The bound is 5 standard
    # errors (a correct sampler exceeds it with probability 6e-7). Measured with N = 200000 and
    # Xoshiro(11): mean z 0.08, 1.85, 1.76 and variance z 0.76, 2.02, 2.18 for a = 0.3, 1, 15, in
    # both precisions; over seeds 1-20 the mean z averages -0.06, 0.28, 0.45.
    N = 200_000
    for T in (Float64, Float32), a in (0.3, 1.0, 15.0)
        rng = Xoshiro(11)
        @test randgamma(rng, T(a)) isa T
        x = [Float64(randgamma(rng, T(a))) for _ in 1:N]
        @test all(>(0), x)
        @test abs(mean(x) - a) <= 5 * sqrt(a / N)
        @test abs(var(x) - a) <= 5 * sqrt((2a^2 + 6a) / N)
    end
    # Shape 1 is the exponential distribution, CDF 1 - e^(-x): the CDF at the sample quantile q_p
    # is p within the binomial standard error √(p(1-p)/N), bound 5 of them. Measured deviations
    # at p = 0.1, 0.25, 0.5, 0.9 are 0.46, 0.56, 0.42, 0.13 standard errors.
    rng = Xoshiro(12)
    x = [randgamma(rng, 1.0) for _ in 1:N]
    for p in (0.1, 0.25, 0.5, 0.9)
        @test abs(1 - exp(-quantile(x, p)) - p) <= 5 * sqrt(p * (1 - p) / N)
    end
    # An integer shape is accepted and promoted.
    @test randgamma(Xoshiro(1), 2) isa Float64
end

@testitem "Bussi ideal-gas kinetic energy follows Gamma(N_f/2, kT)" setup = [ThermostatFixtures] begin
    using Statistics
    F = ThermostatFixtures
    # Without forces only the thermostat changes K, and its stationary distribution is Gamma with
    # shape N_f/2 = 15 and scale kT: mean N_f kT/2, variance N_f (kT)²/2. With τ = 1 fs and
    # samples every 5 steps the lag-1 correlation of the samples is -0.005; the standard errors
    # are from 20 blocks of 1000 samples, the quantile errors from a bootstrap. Bounds are 5
    # standard errors. Measured z: mean -0.90, variance -2.06, quantiles 0.1, 0.25, 0.5, 0.9:
    # 0.11, -0.04, -1.98, -2.23 (over seeds 1-60 the mean and variance z average 0.20 ± 0.14
    # and -0.09 ± 0.15). The cross term of α² has zero mean and sets the variance: without it
    # the stationary variance is (1 - c)/(1 + c) = 0.46 of the canonical one (c = e^-1 per
    # half-step), and the variance z measured -83 to -147 over 20 seeds (relative standard
    # error 1.0%), so the 5-standard-error bound rejects that update in every seed.
    kT = F.KB_LITERAL * 300
    nf = 30
    K = F.bussi_ideal_gas_samples()
    m, se = F.block_mean(K, 20)
    @test abs(m - nf * kT / 2) <= 5 * se
    v, sev = F.block_var(K, 20)
    @test abs(v - nf * kT^2 / 2) <= 5 * sev
    @test all(z -> abs(z) <= 5, F.quantile_z(K, nf, kT))
end

@testitem "Bussi kinetic energy relaxes with the coupling time" setup = [ThermostatFixtures] begin
    using PureAtomistic.Systems
    using Random, Statistics
    F = ThermostatFixtures
    # The mean of α² K over the noise is c K + (1 - c) K̄ with c = exp(-t/τ), so after n steps
    # (2n half-steps of dt/2) the mean is exactly K̄ + (K₀ - K̄) exp(-n dt/τ). 400 replicas start
    # at 30 K (K₀ = 15 k_B · 30 K exactly) toward 300 K with τ = 20 fs. Bound: 5 standard errors
    # of the replica mean. Measured z at n = 5, 10, 20: 1.74, -0.21, -0.34. The initial kinetic
    # energy equals K₀ to 1.5 eps (maxwell_boltzmann! rescales to the target); bound 25 eps.
    kT, nf, τ, dt = F.KB_LITERAL * 300, 30, 20.0, 2.0
    K0, Kbar = nf / 2 * F.KB_LITERAL * 30, nf * kT / 2
    steps = (5, 10, 20)
    K = zeros(400, length(steps))
    for r in axes(K, 1)
        sim = F.ideal_gas(Float64, 11, BussiThermostat(300.0, τ); temp = 30.0, rng = Xoshiro(r))
        @test kinetic_energy(sim.system) ≈ K0 rtol = 25 * eps()
        for (j, n) in pairs(steps)
            run!(sim, n - sim.step_count)
            K[r, j] = kinetic_energy(sim.system)
        end
    end
    for (j, n) in pairs(steps)
        expected = Kbar + (K0 - Kbar) * exp(-n * dt / τ)
        @test abs(mean(K[:, j]) - expected) <= 5 * std(K[:, j]) / sqrt(size(K, 1))
    end
end

@testitem "Bussi with an infinite coupling time is NVE bit for bit" setup = [VerletFixtures] begin
    using Random
    F = VerletFixtures
    for T in (Float64, Float32)
        a = F.lj_simulation(T, 2.0)
        b = F.lj_simulation(T, 2.0; thermostat = BussiThermostat(10.0, Inf), rng = Xoshiro(3))
        run!(a, 500)
        run!(b, 500)
        @test a.system.positions == b.system.positions
        @test a.system.velocities == b.system.velocities
        @test total_energy(a) == total_energy(b)
    end
end

@testitem "BAOAB harmonic oscillators sample kT/k and kT/m" setup = [VerletFixtures, ThermostatFixtures] begin
    using Statistics
    F = ThermostatFixtures
    # 32 copper atoms on springs k = 0.5 eV/Å² (96 independent oscillators, ω dt = 0.017) at
    # 300 K with γ = 0.01/fs and dt = 2 fs over 200000 steps, sampled every 10 steps after the
    # first 1000 samples. Oracle: ⟨x²⟩ = kT/k and ⟨v²⟩ = kT/m per component; the O(dt²)
    # discretization error of the velocities, (ω dt)²/4 = 8e-5 relative, is below the standard
    # error. Standard errors from 19 blocks of 10 ps (1/γ = 100 fs); bound 5 of them. Measured:
    # relative standard errors 4.1e-3 (x²) and 2.3e-3 (v²), z 0.644 and 1.107 in both
    # precisions: `randn(rng, Float32)` rounds the Float64 draw, so both runs see the same noise.
    kT = F.KB_LITERAL * 300
    for T in (Float64, Float32)
        sim = VerletFixtures.harmonic_simulation(T, 2.0; thermostat = LangevinBAOAB(300.0, 0.01))
        sites = copy(sim.potential.sites)
        X2, V2 = Float64[], Float64[]
        run!(
            sim, 200_000; every = 10,
            observer = s -> (push!(X2, mean(abs2, s.system.positions .- sites)); push!(V2, mean(abs2, s.system.velocities)))
        )
        x2, sx = F.block_mean(X2[1001:end], 19)
        v2, sv = F.block_mean(V2[1001:end], 19)
        @test abs(x2 - kT / 0.5) <= 5 * sx
        @test abs(v2 - kT / (63.546 * F.AMU_LITERAL)) <= 5 * sv
    end
end

@testitem "BAOAB friction and force evaluations" setup = [ThermostatFixtures] begin
    using Random
    F = ThermostatFixtures
    # At zero temperature on free particles the O step is v ← c₁ v with c₁ = exp(-γ dt) and the
    # other sub-steps leave v unchanged, so after n steps v = exp(-n γ dt) v₀. Rounding bound:
    # c₁ is rounded once and each of the n products once, 2n eps(T) relative (measured 3.9 eps
    # in Float64 and 1.1 eps in Float32 for n = 100). Each step
    # evaluates the forces once, so n steps and the constructor make n + 1 evaluations.
    n = 100
    for T in (Float64, Float32)
        sim = F.ideal_gas(T, 4, LangevinBAOAB(0.0, 0.01); dt = 2.0)
        v0 = Float64.(sim.system.velocities)
        run!(sim, n)
        @test maximum(abs, Float64.(sim.system.velocities) .- exp(-n * 0.01 * 2.0) .* v0) <=
            2n * eps(T) * maximum(abs, v0)
        @test sim.potential.evaluations[] == n + 1
        b = F.ideal_gas(T, 4, BussiThermostat(300.0, 10.0); dt = 2.0)
        run!(b, n)
        @test b.potential.evaluations[] == n + 1
    end
end

@testitem "seeded thermostatted runs reproduce bit for bit" setup = [VerletFixtures] begin
    using Random
    F = VerletFixtures
    for T in (Float64, Float32), th in (LangevinBAOAB(300.0, 0.01), BussiThermostat(300.0, 50.0; removed_dof = 0))
        a = F.harmonic_simulation(T, 2.0; thermostat = th, rng = Xoshiro(42))
        b = F.harmonic_simulation(T, 2.0; thermostat = th, rng = Xoshiro(42))
        c = F.harmonic_simulation(T, 2.0; thermostat = th, rng = Xoshiro(43))
        run!(a, 200)
        run!(b, 200)
        run!(c, 200)
        @test a.system.positions == b.system.positions
        @test a.system.velocities == b.system.velocities
        @test a.system.velocities != c.system.velocities
    end
end

@testitem "thermostats on one atom and on a cell without neighbors" setup = [ThermostatFixtures] begin
    using PureAtomistic.Systems
    using Random
    F = ThermostatFixtures
    one_atom(v) = System{Float64}(zeros(3, 1), reshape(v, 3, 1), [63.546], [29], [10.0 0 0; 0 10 0; 0 0 10])
    sim = Simulation(
        one_atom([0.01, 0.0, 0.0]), F.FreeParticles{Float64}(1), VelocityVerlet(2.0);
        thermostat = BussiThermostat(300.0, 10.0), rng = Xoshiro(1)
    )
    @test_throws "pureatomistic: the Bussi thermostat needs at least one degree of freedom, got 0" step!(sim)
    sim = Simulation(
        one_atom([0.01, 0.0, 0.0]), F.FreeParticles{Float64}(1), VelocityVerlet(2.0);
        thermostat = BussiThermostat(300.0, 10.0; removed_dof = 0), rng = Xoshiro(1)
    )
    run!(sim, 10)
    @test all(isfinite, sim.system.velocities)
    sim = Simulation(
        one_atom([0.0, 0.0, 0.0]), F.FreeParticles{Float64}(1), VelocityVerlet(2.0);
        thermostat = BussiThermostat(300.0, 10.0; removed_dof = 0), rng = Xoshiro(1)
    )
    @test_throws "pureatomistic: the Bussi thermostat cannot rescale velocities with zero kinetic energy" step!(sim)
    sim = Simulation(
        one_atom([0.0, 0.0, 0.0]), F.FreeParticles{Float64}(1), VelocityVerlet(2.0);
        thermostat = LangevinBAOAB(300.0, 0.01), rng = Xoshiro(1)
    )
    run!(sim, 10)
    @test all(isfinite, sim.system.velocities)
    @test any(!iszero, sim.system.velocities)

    # Two argon atoms 10 Å apart in a 30 Å cube with a 6 Å cutoff: no pair, zero force.
    x = [0.0 10.0; 0.0 0.0; 0.0 0.0]
    cell = [30.0 0 0; 0 30 0; 0 0 30]
    for th in (LangevinBAOAB(100.0, 0.01), BussiThermostat(100.0, 10.0))
        sys = System{Float64}(copy(x), [0.001 -0.001; 0.0 0.0; 0.0 0.0], [39.948, 39.948], [18, 18], cell)
        lj = LennardJones(copy(x), cell; epsilon = 0.0104, sigma = 3.4, r_cut = 6.0)
        gas = Simulation(sys, lj, VelocityVerlet(2.0); thermostat = th, rng = Xoshiro(1))
        run!(gas, 20)
        @test iszero(gas.potential.result.forces)
        @test all(isfinite, gas.system.velocities)
    end
end

@testitem "thermostatted runs on views equal runs on arrays" setup = [VerletFixtures] begin
    using PureAtomistic.Systems
    using Random
    F = VerletFixtures
    for th in (LangevinBAOAB(300.0, 0.01), BussiThermostat(300.0, 50.0; removed_dof = 0))
        ref = F.harmonic_simulation(Float64, 2.0; thermostat = th, rng = Xoshiro(5))
        n = natoms(ref.system)
        bigx, bigv = zeros(3, n + 4), zeros(3, n + 4)
        bigx[:, 1:n] .= ref.system.positions
        bigv[:, 1:n] .= ref.system.velocities
        sys = System{Float64}(
            view(bigx, :, 1:n), view(bigv, :, 1:n), ref.system.masses, ref.system.species, ref.system.cell
        )
        sim = Simulation(sys, HarmonicPotential(copy(ref.potential.sites), 0.5), VelocityVerlet(2.0); thermostat = th, rng = Xoshiro(5))
        run!(ref, 50)
        run!(sim, 50)
        @test sys.positions == ref.system.positions
        @test sys.velocities == ref.system.velocities
        @test iszero(bigv[:, (n + 1):end])
    end
end

@testitem "thermostat steps and the gamma sampler do not allocate" tags = [:checks] setup = [VerletFixtures] begin
    using StrictModeTest, StrictMode
    using PureAtomistic.Thermostats
    using Random
    F = VerletFixtures
    register_alloc_barrier!(PureAtomistic.Potentials._regrow!)
    @noinline stepper!(s) = step!(s)
    @noinline runner!(s) = run!(s, 3)
    @noinline gamma(rng, a) = randgamma(rng, a)
    for T in (Float64, Float32)
        rng = Xoshiro(1)
        for a in (T(0.3), T(15))
            gamma(rng, a)
            @test_noalloc gamma(rng, a)
        end
        for th in (LangevinBAOAB(300.0, 0.01), BussiThermostat(300.0, 50.0))
            for make in (F.harmonic_simulation, F.lj_simulation, F.mace_simulation)
                sim = make === F.mace_simulation ? make(T, 0.5; thermostat = th) : make(T, 2.0; thermostat = th)
                stepper!(sim)
                @test_noalloc stepper!(sim)
                @test_steady_state stepper!(sim)
                @test_steady_state runner!(sim)
                make === F.harmonic_simulation || @test sim.potential.n_regrows == 0
            end
        end
    end
end

@testitem "TrimCheck: thermostats" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureAtomistic, PureAtomistic.Systems, PureAtomistic.Thermostats, Random
            const SYS64 = System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}
            const SYS32 = System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}
            const SIMB64 = Simulation{Float64, SYS64, HarmonicPotential{Float64}, VelocityVerlet, LangevinBAOAB, Xoshiro}
            const SIMB32 = Simulation{Float32, SYS32, HarmonicPotential{Float32}, VelocityVerlet, LangevinBAOAB, Xoshiro}
            const SIMS64 = Simulation{Float64, SYS64, LennardJones{Float64}, VelocityVerlet, BussiThermostat, Xoshiro}
            const SIMS32 = Simulation{Float32, SYS32, LennardJones{Float32}, VelocityVerlet, BussiThermostat, Xoshiro}
            new_baoab(t, g) = LangevinBAOAB(t, g)
            new_bussi(t, τ) = BussiThermostat(t, τ)
            new_sim(s, p, dt, th, rng) = Simulation(s, p, VelocityVerlet(dt); thermostat = th, rng)
            step_it(s) = step!(s)
            run_it(s, n) = run!(s, n)
        end,
        new_baoab(Float64, Float64),
        new_bussi(Float64, Float64),
        new_sim(SYS64, HarmonicPotential{Float64}, Float64, LangevinBAOAB, Xoshiro),
        new_sim(SYS32, LennardJones{Float32}, Float64, BussiThermostat, Xoshiro),
        step_it(SIMB64),
        step_it(SIMB32),
        step_it(SIMS64),
        step_it(SIMS32),
        run_it(SIMB64, Int),
        run_it(SIMS32, Int),
        randgamma(Xoshiro, Float64),
        randgamma(Xoshiro, Float32),
    )
end

@testitem "thermostat return types are concrete and the contract holds" tags = [:checks] begin
    using PureAtomistic.Systems, PureAtomistic.Thermostats
    using TypeContracts, Random
    for T in (Float64, Float32), th in (LangevinBAOAB(300.0, 0.01), BussiThermostat(300.0, 50.0))
        sys = fcc_cell(T(3.6), 1, :Cu)
        sim = Simulation(sys, HarmonicPotential(copy(sys.positions), 0.5), VelocityVerlet(1.0); thermostat = th, rng = Xoshiro(0))
        @test Base.return_types(step!, (typeof(sim),)) == [Nothing]
        @test Base.return_types(randgamma, (Xoshiro, T)) == [T]
    end
    @test LangevinBAOAB <: AbstractThermostat
    @test BussiThermostat <: AbstractThermostat
    @verify LangevinBAOAB trim_compat = true
    @verify BussiThermostat trim_compat = true
    # A thermostat without the contract's method fails the check.
    struct NoStep <: AbstractThermostat end
    @test_throws Exception @verify NoStep
end
