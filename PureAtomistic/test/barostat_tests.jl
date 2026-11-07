@testmodule BarostatFixtures begin
    using PureAtomistic
    using PureAtomistic.Systems
    using LinearAlgebra, Random

    # One eV/Å³ in GPa, k_B in eV/K and the atomic mass unit in eV fs² / Å², written out so that
    # the unit factors are not checked against the constants they test.
    const GPA_LITERAL = 160.2176634
    const KB_LITERAL = 8.617333262145179e-5
    const AMU_LITERAL = 103.64269652680504

    # A 32-atom fcc crystal of argon-mass atoms with argon-magnitude Lennard-Jones parameters,
    # zero velocities and lattice constant `a`. The cutoff 6.85 Å lies between the third
    # (√(3/2) a) and fourth (√2 a) neighbor shells for every `a` from 5.6 Å down to 4.85 Å, so
    # the edge set does not change while the barostat rescales the cell.
    function lj_crystal(::Type{T}, a) where {T}
        sys = fcc_cell(T(a), 2, :Cu; mass = 39.948)
        p = LennardJones(copy(sys.positions), copy(sys.cell); epsilon = T(0.0104), sigma = T(3.4), r_cut = 6.85)
        return sys, p
    end

    # The static pressure `-tr(stress)/3` of the perfect crystal at lattice constant `a`, in GPa.
    function lj_static_pressure(a)
        sys, p = lj_crystal(Float64, a)
        prepare!(p, sys.positions, sys.cell)
        energy_forces!(p, sys.positions, sys.cell)
        return -tr(stress(p)) / 3 * GPA_LITERAL
    end

    # The lattice constant where `f` changes sign from positive to negative in `[lo, hi]`, by
    # bisection to the resolution of Float64.
    function bisect(f, lo, hi)
        f(lo) > 0 > f(hi) || error("no sign change in [$lo, $hi]")
        for _ in 1:60
            m = (lo + hi) / 2
            f(m) > 0 ? (lo = m) : (hi = m)
        end
        return (lo + hi) / 2
    end

    # A Lennard-Jones potential that counts its `prepare!` and `energy_forces!` calls; `result`
    # is the wrapped potential's result object.
    struct CountingLJ{T} <: AbstractStressPotential
        lj::LennardJones{T}
        result::PotentialResult{T}
        prepares::Base.RefValue{Int}
        evaluations::Base.RefValue{Int}
    end
    CountingLJ(lj::LennardJones) = CountingLJ(lj, lj.result, Ref(0), Ref(0))
    PureAtomistic.n_atoms(p::CountingLJ) = n_atoms(p.lj)
    PureAtomistic.cutoff(p::CountingLJ) = cutoff(p.lj)
    PureAtomistic.stress(p::CountingLJ) = stress(p.lj)
    function PureAtomistic.prepare!(p::CountingLJ, x, c)
        p.prepares[] += 1
        return prepare!(p.lj, x, c)
    end
    function PureAtomistic.energy_forces!(p::CountingLJ, x, c)
        p.evaluations[] += 1
        return energy_forces!(p.lj, x, c)
    end

    # The pressure in GPa from the stress and the velocities, with the kinetic term
    # 2K/(3V) written out here so that it is not checked against `Systems.pressure`.
    function pressure_gpa(sys, S)
        V = abs(det(sys.cell))
        K = AMU_LITERAL / 2 * sum(sys.masses[a] * sum(abs2, sys.velocities[:, a]) for a in axes(sys.velocities, 2))
        return (-tr(S) / 3 + 2K / (3V)) * GPA_LITERAL
    end

    # Mean of `x` and its standard error from the means of `nb` consecutive blocks.
    function block_mean(x, nb)
        L = length(x) ÷ nb
        b = [sum(@view x[((i - 1) * L + 1):(i * L)]) / L for i in 1:nb]
        m = sum(b) / nb
        return m, sqrt(sum(abs2, b .- m) / (nb - 1) / nb)
    end

    # Least-squares line through `(t, log|y|)`: the decay rate (minus the slope) and the largest
    # residual.
    function log_linear_fit(t, y)
        X = [ones(length(t)) t]
        c = X \ log.(abs.(y))
        return -c[2], maximum(abs, log.(abs.(y)) .- X * c)
    end
end

@testitem "stress is the strain derivative of the energy over the volume" setup = [BarostatFixtures, RelaxFixtures] begin
    using PureMLIP, LinearAlgebra, Random
    using PureAtomistic.Systems
    B = BarostatFixtures
    # Oracle: central finite differences of the energy under a homogeneous strain x → (I + ε) x
    # of positions and cell. A symmetric strain with ε_ab = ε_ba = δ has dE/dδ = V (S_ab + S_ba)
    # off the diagonal and V S_aa on it, for stress S. Positive S is tension.
    sys, p = B.lj_crystal(Float64, 5.26)
    sys.positions .+= 0.03 .* randn(Xoshiro(3), 3, natoms(sys))
    sys.cell .+= [0.0 0.1 -0.05; 0.1 0.0 0.08; -0.05 0.08 0.0]
    function energy_at(ε)
        D = I + ε
        x, c = D * sys.positions, D * sys.cell
        q = LennardJones(x, c; epsilon = 0.0104, sigma = 3.4, r_cut = 6.85)
        prepare!(q, x, c)
        energy_forces!(q, x, c)
        return q.result.energy
    end
    prepare!(p, sys.positions, sys.cell)
    energy_forces!(p, sys.positions, sys.cell)
    S, V = copy(stress(p)), volume(sys)
    h = 1.0e-5
    function strain_error(a, b)
        E = zeros(3, 3)
        E[a, b] = E[b, a] = h
        fd = (energy_at(E) - energy_at(-E)) / (2h)
        exact = a == b ? V * S[a, a] : V * (S[a, b] + S[b, a])
        return abs(fd - exact)
    end
    worst = maximum(strain_error(a, b) for a in 1:3 for b in a:3)
    # Measured: largest |fd - exact| 4.8e-9 eV against strain derivatives of size 0.01-1 eV.
    # Bound 5e-8 (10 times).
    @test worst < 5.0e-8
    @test S == S'

    # The pressure -tr(S)/3 equals -dE/dV under isotropic scaling, for Lennard-Jones and MPA-0.
    # Measured: relative difference 8.1e-8 (LJ) and absolute difference 6.7e-9 eV/Å³ (MPA-0);
    # bounds 1e-6 relative and 1e-7 eV/Å³ (12 and 15 times).
    function static(make, a)
        s, q = make(a)
        prepare!(q, s.positions, s.cell)
        energy_forces!(q, s.positions, s.cell)
        return q.result.energy, -tr(stress(q)) / 3, volume(s)
    end
    ljmake(a) = B.lj_crystal(Float64, a)
    model = MACEModel(load_mace_checkpoint(RelaxFixtures.mpa0_path()))
    function macemake(a)
        s = fcc_cell(a, 1, :Cu)
        return s, MACEPotential(model, s.positions, s.cell, s.species)
    end
    for (make, as, rel, abs_bound) in ((ljmake, (5.2, 5.4), 1.0e-6, 0.0), (macemake, (3.55, 3.7), 0.0, 1.0e-7))
        for a in as
            Ep, _, Vp = static(make, a + 1.0e-4)
            Em, _, Vm = static(make, a - 1.0e-4)
            _, P, _ = static(make, a)
            Pfd = -(Ep - Em) / (Vp - Vm)
            @test abs(P - Pfd) <= rel * abs(Pfd) + abs_bound
        end
    end
end

@testitem "Berendsen pressure relaxes exponentially on a Lennard-Jones solid" setup = [BarostatFixtures] begin
    using PureAtomistic.Systems
    using Random
    B = BarostatFixtures
    # Oracle: with β_T the coupling compressibility and B the bulk modulus, the pressure obeys
    # dP/dt = -(β_T B / τ_P)(P - P0) near P0. The crystal is static (zero velocities, every atom
    # at a center of symmetry), so the pressure is the static one and only the barostat moves it.
    a0 = B.bisect(B.lj_static_pressure, 5.2, 5.4)
    # B = -V dP/dV by a central difference of the stress pressure at the zero-pressure point.
    h = 1.0e-4
    bulk = -a0^3 * (B.lj_static_pressure(a0 + h) - B.lj_static_pressure(a0 - h)) / ((a0 + h)^3 - (a0 - h)^3)
    rates = Float64[]
    for (β, τ) in ((0.5, 500.0), (0.25, 125.0))
        sys, p = B.lj_crystal(Float64, a0 * (1 - 0.002))
        dt = 1.0
        sim = Simulation(sys, p, VelocityVerlet(dt); barostat = BerendsenBarostat(0.0, τ, β), rng = Xoshiro(0))
        t, P = Float64[], Float64[]
        run!(sim, round(Int, 14 * τ / (β * bulk)); observer = s -> (push!(t, s.step_count * dt); push!(P, pressure(s.system, stress(s.potential)) * B.GPA_LITERAL)))
        @test P[1] > 0
        keep = findall(x -> abs(x) > 1.0e-6 * P[1], P)
        k, worst = B.log_linear_fit(t[keep], P[keep])
        push!(rates, k)
        # Measured: k / (β_T B / τ_P) - 1 = 2.7e-3 (β_T = 0.5) and 4.1e-3 (β_T = 0.25); the
        # discrete map accounts for 1.4e-3 and 2.8e-3 of it, the change of B over the 0.6 %
        # volume change for the rest. Bound 0.04 (10 times the larger). Wrong unit factors or a
        # missing cube root change k by a factor of 160 or 3.
        @test abs(k / (β * bulk / τ) - 1) < 0.04
        # Measured largest residual of the log-linear fit 0.036 (the first steps, where B differs
        # most); bound 0.36 (10 times).
        @test worst < 0.36
        @test all(>(0), P[keep])
        # Converged to P0 = 0: the final cell is the zero-pressure lattice. Measured 1.6e-9
        # relative; bound 2e-8 (12 times).
        @test abs(sys.cell[1, 1] / (2a0) - 1) < 2.0e-8
    end
    # The rate scales as β_T / τ_P: the second pair has twice the first one's ratio.
    # Measured ratio - 2 = 2.8e-3; bound 0.04 (14 times).
    @test abs(rates[2] / rates[1] - 2) < 0.04
end

@testitem "Berendsen ideal gas relaxes to N k_B T / P0" setup = [BarostatFixtures] begin
    using PureAtomistic.Systems
    using Random
    B = BarostatFixtures
    # One argon-mass atom in a cube with no neighbor within the cutoff: the stress is zero, so
    # the pressure is the kinetic term N k_B T / V alone and the barostat must settle at
    # V = N k_B T / P0, with T = m v² / (3 k_B) for the N = 1 atom (3 degrees of freedom).
    for T in (Float64, Float32)
        v = T[0.01, -0.004, 0.002]
        m = 39.948
        sys = System{T}(zeros(T, 3, 1), reshape(v, 3, 1), T[m], [18], T[20 0 0; 0 20 0; 0 0 20])
        lj = LennardJones(copy(sys.positions), copy(sys.cell); epsilon = T(0.0104), sigma = T(3.4), r_cut = 5.0)
        P0 = 0.006
        sim = Simulation(sys, lj, VelocityVerlet(2.0); barostat = BerendsenBarostat(P0, 100.0, 1 / P0), rng = Xoshiro(0))
        run!(sim, 1500)
        temp = m * B.AMU_LITERAL * sum(abs2, Float64.(v)) / (3 * B.KB_LITERAL)
        Vstar = B.KB_LITERAL * temp / (P0 / B.GPA_LITERAL)
        @test iszero(stress(lj))
        # Measured relative error 5.3e-14 (Float64) and 4.3e-6 (Float32) after 1500 steps of a
        # decay at rate 0.01/fs (30 coupling times); bounds 1e-12 and 5e-5 (19 and 12 times).
        @test volume(sys) ≈ Vstar rtol = (T == Float64 ? 1.0e-12 : 5.0e-5)
    end
end

@testitem "Berendsen MPA-0 Cu relaxes to the zero-stress lattice constant of E(V)" setup = [BarostatFixtures, RelaxFixtures] begin
    using PureMLIP, Random
    using PureAtomistic.Systems
    model = MACEModel(load_mace_checkpoint(RelaxFixtures.mpa0_path()))
    function energy(a)
        s = fcc_cell(a, 1, :Cu)
        q = MACEPotential(model, s.positions, s.cell, s.species)
        prepare!(q, s.positions, s.cell)
        energy_forces!(q, s.positions, s.cell)
        return q.result.energy
    end
    # Oracle: the zero of the central finite difference dE/dV of the 4-atom cell's energy, step
    # 1e-4 Å in the lattice constant, found by bisection. Its truncation error falls as h²
    # (measured roots at h = 1e-3, 3e-4, 1e-4, 3e-5: 3.63986929, 3.63986889, 3.639868855,
    # 3.639868851 Å).
    h = 1.0e-4
    dEdV(a) = (energy(a + h) - energy(a - h)) / ((a + h)^3 - (a - h)^3)
    a_E = BarostatFixtures.bisect(a -> -dEdV(a), 3.6, 3.7)
    # The PureMLIP gallery's estimate: a parabola through the nine lowest of 33 energies per
    # atom from 3.45 to 3.85 Å. The asymmetry of E(a) over its ±0.05 Å window biases it by a
    # measured 8.09e-4 Å above the barostat's value.
    as = range(3.45, 3.85; length = 33)
    Es = [energy(a) / 4 for a in as]
    i = argmin(Es)
    w = (i - 4):(i + 4)
    c = [ones(9) as[w] as[w] .^ 2] \ Es[w]
    a_gallery = -c[2] / (2c[3])
    # The static crystal (zero velocities, zero forces by symmetry) under the barostat at
    # P0 = 0, started on either side of the minimum.
    for a_start in (3.6, 3.7)
        sys = fcc_cell(a_start, 1, :Cu)
        p = MACEPotential(model, copy(sys.positions), copy(sys.cell), sys.species)
        sim = Simulation(sys, p, VelocityVerlet(1.0); barostat = BerendsenBarostat(0.0, 10.0, 0.01), rng = Xoshiro(0))
        run!(sim, 250)
        a = sys.cell[1, 1]
        # Measured |a - a_E| = 4.4e-9 Å (the h² truncation of the oracle; with h = 3e-5 it is
        # 5e-10 Å). Bound 1e-7 Å (23 times).
        @test abs(a - a_E) < 1.0e-7
        # Measured a_gallery - a = 8.09e-4 Å; bound 1e-3 Å (the bias plus 24 %), far below
        # the 0.03 Å by which a wrong stress convention or unit factor moves a.
        @test abs(a_gallery - a) < 1.0e-3
        @test sys.cell == a * [1 0 0; 0 1 0; 0 0 1]
        # Measured final pressure 2.8e-13 GPa; bound 5e-12 GPa (18 times).
        @test abs(pressure(sys, stress(p)) * BarostatFixtures.GPA_LITERAL) < 5.0e-12
    end
end

@testitem "thermal NPT: the mean pressure equals P0" setup = [BarostatFixtures] begin
    using PureAtomistic.Systems
    using Random
    B = BarostatFixtures
    # 32-atom Lennard-Jones crystal at 30 K under the Bussi thermostat (τ_T = 100 fs) and the
    # Berendsen barostat (P0 = 0.05 GPa, τ_P = 200 fs, β_T = 0.4/GPa), dt = 2 fs. After 10 ps
    # of equilibration the pressure, computed here from the stress and the velocities, is
    # sampled every 10 steps for 60 ps (3000 samples) and split into 20 blocks of 3 ps, 15
    # times τ_P. In the steady state the volume does not drift, so ⟨μ³ - 1⟩ = 0 and ⟨P⟩ = P0
    # up to the second-order term (β_T dt/τ_P) σ_P²/2 = 4e-7 GPa (σ_P = 9e-3 GPa measured).
    # Measured: ⟨P⟩ - P0 = -8e-5 GPa, standard error 1.7e-4 GPa (z = -0.5). The bound is 5
    # standard errors. The kinetic term is 0.011 GPa here (measured), so a pressure without it
    # misses P0 by about 65 standard errors.
    sys, p = B.lj_crystal(Float64, 5.3)
    maxwell_boltzmann!(sys, 30.0, Xoshiro(1))
    sim = Simulation(
        sys, p, VelocityVerlet(2.0); thermostat = BussiThermostat(30.0, 100.0),
        barostat = BerendsenBarostat(0.05, 200.0, 0.4), rng = Xoshiro(1)
    )
    run!(sim, 5000)
    P = Float64[]
    run!(sim, 30_000; every = 10, observer = s -> push!(P, B.pressure_gpa(s.system, stress(s.potential))))
    m, se = B.block_mean(P, 20)
    @test abs(m - 0.05) < 5 * se
    # The standard error is small enough to resolve the kinetic term: measured 1.7e-4 GPa,
    # bound 1e-3 GPa.
    @test se < 1.0e-3
end

@testitem "a barostatted step evaluates the potential once, at the rescaled configuration" setup = [BarostatFixtures] begin
    using PureAtomistic.Systems
    using PureAtomistic.Barostats
    using Random
    B = BarostatFixtures
    for th in (nothing, BussiThermostat(30.0, 50.0), LangevinBAOAB(30.0, 0.01))
        sys, lj = B.lj_crystal(Float64, 5.26)
        maxwell_boltzmann!(sys, 30.0, Xoshiro(2))
        p = B.CountingLJ(lj)
        sim = Simulation(sys, p, VelocityVerlet(2.0); thermostat = th, barostat = BerendsenBarostat(1.0, 100.0, 0.3), rng = Xoshiro(0))
        @test p.evaluations[] == 1
        V0 = volume(sys)
        run!(sim, 5)
        @test volume(sys) != V0
        # Exactly one force evaluation and one neighbor-list refresh per step.
        @test p.evaluations[] == 6
        @test p.prepares[] == 6
        fresh = LennardJones(copy(sys.positions), copy(sys.cell); epsilon = 0.0104, sigma = 3.4, r_cut = 6.85)
        prepare!(fresh, sys.positions, sys.cell)
        energy_forces!(fresh, sys.positions, sys.cell)
        # The same edges in the same order give the same sums.
        @test p.result.energy == fresh.result.energy
        @test p.result.forces == fresh.result.forces
        @test stress(p) == stress(fresh)
    end

    # scale_factor changes nothing; rescale! scales positions and cell by one factor, so the
    # fractional coordinates are unchanged, and leaves the velocities alone. Rounding bound:
    # one multiplication each, so 2 eps relative.
    sys, p = B.lj_crystal(Float64, 5.26)
    maxwell_boltzmann!(sys, 30.0, Xoshiro(2))
    prepare!(p, sys.positions, sys.cell)
    energy_forces!(p, sys.positions, sys.cell)
    v, x, c = copy(sys.velocities), copy(sys.positions), copy(sys.cell)
    μ = scale_factor(BerendsenBarostat(1.0, 100.0, 0.3), sys, p, 2.0)
    @test μ isa Float64 && μ != 1
    @test sys.positions == x && sys.cell == c && sys.velocities == v
    # Oracle for μ: the formula with the pressure computed here; rounding bound 8 eps.
    P = B.pressure_gpa(sys, stress(p))
    @test μ ≈ cbrt(1 - 0.3 * 2.0 / 100.0 * (1.0 - P)) rtol = 8 * eps()
    # The factor from the pressure alone is the same computation.
    @test scale_factor(BerendsenBarostat(1.0, 100.0, 0.3), pressure(sys, stress(p)) * EV_PER_A3_IN_GPA, 2.0) === μ
    rescale!(sys, μ)
    @test sys.velocities == v
    @test sys.cell ≈ μ .* c rtol = 2 * eps()
    @test sys.positions ≈ μ .* x rtol = 2 * eps()
end

# Oracle: the product x·μ of each entry, rounded once. `rescale!` multiplies each entry of the
# cell and the positions by μ once and leaves the rest alone, so the result equals the elementwise
# product bit for bit, on arrays, views and offset arrays.
@testitem "rescale! gives the bits of the elementwise product" setup = [BarostatFixtures] begin
    using PureAtomistic.Systems
    using PureAtomistic.Barostats
    using OffsetArrays, Random
    for T in (Float64, Float32)
        rng = Xoshiro(3)
        X, V, C = 10 .* rand(rng, T, 3, 5), rand(rng, T, 3, 5), T[5 0.1 0.2; 0 5 0.3; 0 0 5] .+ rand(rng, T, 3, 3)
        μ = T(1.0003)
        sys = System{T}(copy(X), copy(V), ones(T, 5), fill(29, 5), copy(C))
        rescale!(sys, μ)
        @test sys.positions == [x * μ for x in X]
        @test sys.cell == [c * μ for c in C]
        @test sys.velocities == V
        bigx = zeros(T, 3, 7)
        bigx[:, 2:6] .= X
        vsys = System{T}(view(bigx, :, 2:6), copy(V), ones(T, 5), fill(29, 5), OffsetArray(copy(C), 0:2, 0:2))
        rescale!(vsys, μ)
        @test vsys.positions == [x * μ for x in X]
        @test parent(vsys.cell) == [c * μ for c in C]
        @test iszero(bigx[:, 1]) && iszero(bigx[:, 7])
    end
end

@testitem "barostat rejects invalid arguments" setup = [BarostatFixtures, VerletFixtures] begin
    using PureAtomistic.Systems
    using PureAtomistic.Barostats
    using Random
    B = BarostatFixtures
    @test_throws "pureatomistic: the barostat coupling time must be positive and finite, got 0.0" BerendsenBarostat(0.0, 0.0, 1.0)
    @test_throws "pureatomistic: the barostat coupling time must be positive and finite, got -1.0" BerendsenBarostat(0.0, -1.0, 1.0)
    @test_throws "pureatomistic: the barostat coupling time must be positive and finite, got Inf" BerendsenBarostat(0.0, Inf, 1.0)
    @test_throws "pureatomistic: the barostat compressibility must be positive and finite, got 0.0" BerendsenBarostat(0.0, 1.0, 0.0)
    @test_throws "pureatomistic: the barostat compressibility must be positive and finite, got -0.5" BerendsenBarostat(0.0, 1.0, -0.5)
    @test_throws "pureatomistic: the barostat compressibility must be positive and finite, got NaN" BerendsenBarostat(0.0, 1.0, NaN)
    @test_throws "pureatomistic: the target pressure must be finite, got Inf" BerendsenBarostat(Inf, 1.0, 1.0)
    b = BerendsenBarostat(1, 2, 3)
    @test (b.P0, b.τ_P, b.β_T) === (1.0, 2.0, 3.0)

    # A potential without stress is refused when the simulation is built.
    h = VerletFixtures.harmonic_simulation(Float64, 1.0)
    @test_throws "pureatomistic: a barostat needs an AbstractStressPotential, got HarmonicPotential{Float64}" Simulation(
        h.system, h.potential, h.integrator; barostat = b, rng = Xoshiro(0)
    )
    @test_throws "pureatomistic: barostat must be an AbstractBarostat or nothing, got Int64" Simulation(
        h.system, h.potential, h.integrator; barostat = 1, rng = Xoshiro(0)
    )
    sys, p = B.lj_crystal(Float64, 5.26)
    struct NoStepIntegrator <: AbstractIntegrator end
    @test_throws "pureatomistic: a barostat needs the VelocityVerlet integrator, got" Simulation(
        sys, p, NoStepIntegrator(); barostat = b, rng = Xoshiro(0)
    )

    # A step whose volume factor would not be positive throws before anything moves.
    maxwell_boltzmann!(sys, 30.0, Xoshiro(2))
    for th in (nothing, BussiThermostat(30.0, 50.0), LangevinBAOAB(30.0, 0.01))
        sim = Simulation(sys, p, VelocityVerlet(1.0); thermostat = th, barostat = BerendsenBarostat(1.0e4, 1.0, 1.0), rng = Xoshiro(0))
        x, v, c = copy(sys.positions), copy(sys.velocities), copy(sys.cell)
        @test_throws "pureatomistic: the Berendsen step would make the cell degenerate (volume factor" step!(sim)
        @test sys.positions == x && sys.velocities == v && sys.cell == c
    end
    # A non-finite stress gives a non-finite volume factor, which is refused too.
    p.stress[1, 1] = NaN
    @test_throws "pureatomistic: the Berendsen step would make the cell degenerate (volume factor NaN" scale_factor(
        BerendsenBarostat(0.0, 1.0, 1.0), sys, p, 1.0
    )
    # A singular cell has no pressure.
    flat = System{Float64}(copy(sys.positions), copy(sys.velocities), sys.masses, sys.species, zeros(3, 3))
    @test_throws "pureatomistic: the cell is singular" scale_factor(b, flat, p, 1.0)
end

@testitem "barostatted runs on views equal runs on arrays; offset arrays" setup = [BarostatFixtures] begin
    using PureAtomistic.Systems
    using PureAtomistic.Barostats
    using OffsetArrays, Random
    B = BarostatFixtures
    run_with(sys, th) = begin
        p = LennardJones(Matrix(sys.positions), Matrix(sys.cell); epsilon = 0.0104, sigma = 3.4, r_cut = 6.85)
        sim = Simulation(sys, p, VelocityVerlet(2.0); thermostat = th, barostat = BerendsenBarostat(0.5, 50.0, 0.3), rng = Xoshiro(0))
        run!(sim, 20)
        return sim
    end
    for th in (nothing, LangevinBAOAB(30.0, 0.01))
        ref, _ = B.lj_crystal(Float64, 5.26)
        maxwell_boltzmann!(ref, 30.0, Xoshiro(4))
        n = natoms(ref)
        bigx, bigv, bigc = zeros(3, n + 2), zeros(3, n + 2), zeros(4, 4)
        bigx[:, 1:n] .= ref.positions
        bigv[:, 1:n] .= ref.velocities
        bigc[1:3, 1:3] .= ref.cell
        vsys = System{Float64}(view(bigx, :, 1:n), view(bigv, :, 1:n), ref.masses, ref.species, view(bigc, 1:3, 1:3))
        run_with(ref, th)
        run_with(vsys, th)
        @test vsys.positions == ref.positions
        @test vsys.velocities == ref.velocities
        @test vsys.cell == ref.cell
        @test iszero(bigx[:, (n + 1):end]) && iszero(bigc[4, :]) && iszero(bigc[:, 4])
    end

    # rescale! is generic over axes: offset arrays scale like plain ones. scale_factor refuses
    # an offset cell, which the force evaluation after the rescaling could not use.
    sys, p = B.lj_crystal(Float64, 5.26)
    n = natoms(sys)
    prepare!(p, sys.positions, sys.cell)
    energy_forces!(p, sys.positions, sys.cell)
    osys = System{Float64}(
        OffsetArray(copy(sys.positions), 0:2, 0:(n - 1)), OffsetArray(copy(sys.velocities), 0:2, 0:(n - 1)),
        OffsetArray(copy(sys.masses), 0:(n - 1)), OffsetArray(copy(sys.species), 0:(n - 1)), OffsetArray(copy(sys.cell), 0:2, 0:2)
    )
    @test_throws "offset arrays are not supported" scale_factor(BerendsenBarostat(0.5, 50.0, 0.3), osys, p, 1.0)
    rescale!(osys, 1.01)
    rescale!(sys, 1.01)
    @test parent(osys.positions) == sys.positions
    @test parent(osys.cell) == sys.cell
end

@testitem "barostat steps do not allocate" tags = [:checks] setup = [VerletFixtures, BarostatFixtures] begin
    using StrictModeTest, StrictMode
    using PureAtomistic.Systems
    using PureAtomistic.Barostats
    using Random
    register_alloc_barrier!(PureAtomistic.Potentials._regrow!)
    @noinline stepper!(s) = step!(s)
    @noinline runner!(s) = run!(s, 3)
    @noinline factor(b, s, p, dt) = scale_factor(b, s, p, dt)
    @noinline scale!(s, μ) = rescale!(s, μ)
    b = BerendsenBarostat(0.5, 50.0, 0.3)
    for T in (Float64, Float32)
        sys, p = BarostatFixtures.lj_crystal(T, 5.26)
        maxwell_boltzmann!(sys, 30.0, Xoshiro(2))
        m = VerletFixtures.mace_simulation(T, 0.5)
        for th in (nothing, BussiThermostat(30.0, 50.0), LangevinBAOAB(30.0, 0.01))
            for sim in (
                    Simulation(sys, p, VelocityVerlet(2.0); thermostat = th, barostat = b, rng = Xoshiro(0)),
                    Simulation(m.system, m.potential, VelocityVerlet(0.5); thermostat = th, barostat = b, rng = Xoshiro(0)),
                )
                stepper!(sim)
                @test_noalloc stepper!(sim)
                @test_steady_state stepper!(sim)
                @test_steady_state runner!(sim)
                factor(b, sim.system, sim.potential, 1.0)
                @test_noalloc factor(b, sim.system, sim.potential, 1.0)
                scale!(sim.system, one(T))
                @test_noalloc scale!(sim.system, one(T))
                @test sim.potential.n_regrows == 0
            end
        end
    end
end

@testitem "TrimCheck: barostat" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureAtomistic, PureAtomistic.Systems, PureAtomistic.Barostats, Random
            const SYS64 = System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}
            const SYS32 = System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}
            const SIM64 = Simulation{Float64, SYS64, LennardJones{Float64}, VelocityVerlet, Nothing, Xoshiro, BerendsenBarostat}
            const SIM32 = Simulation{Float32, SYS32, LennardJones{Float32}, VelocityVerlet, Nothing, Xoshiro, BerendsenBarostat}
            const SIMM32 = Simulation{Float32, SYS32, MACEPotential{Float32}, VelocityVerlet, BussiThermostat, Xoshiro, BerendsenBarostat}
            const SIMB64 = Simulation{Float64, SYS64, LennardJones{Float64}, VelocityVerlet, LangevinBAOAB, Xoshiro, BerendsenBarostat}
            const SIMBM32 = Simulation{Float32, SYS32, MACEPotential{Float32}, VelocityVerlet, LangevinBAOAB, Xoshiro, BerendsenBarostat}
            new_berendsen(P0, τ, β) = BerendsenBarostat(P0, τ, β)
            new_sim(s, p, dt, b, rng) = Simulation(s, p, VelocityVerlet(dt); barostat = b, rng)
            step_it(s) = step!(s)
            run_it(s, n) = run!(s, n)
        end,
        new_berendsen(Float64, Float64, Float64),
        new_sim(SYS64, LennardJones{Float64}, Float64, BerendsenBarostat, Xoshiro),
        new_sim(SYS32, MACEPotential{Float32}, Float64, BerendsenBarostat, Xoshiro),
        step_it(SIM64),
        step_it(SIM32),
        step_it(SIMM32),
        step_it(SIMB64),
        step_it(SIMBM32),
        run_it(SIM64, Int),
        run_it(SIMB64, Int),
        scale_factor(BerendsenBarostat, SYS64, LennardJones{Float64}, Float64),
        scale_factor(BerendsenBarostat, SYS32, LennardJones{Float32}, Float64),
        scale_factor(BerendsenBarostat, SYS64, MACEPotential{Float64}, Float64),
        scale_factor(BerendsenBarostat, SYS32, MACEPotential{Float32}, Float64),
        rescale!(SYS64, Float64),
        rescale!(SYS32, Float32),
    )
end

@testitem "barostat return types are concrete and the contract holds" tags = [:checks] setup = [BarostatFixtures] begin
    using PureAtomistic.Systems, PureAtomistic.Barostats
    using TypeContracts, Random
    b = BerendsenBarostat(0.0, 100.0, 0.5)
    for T in (Float64, Float32), th in (nothing, BussiThermostat(30.0, 50.0), LangevinBAOAB(30.0, 0.01))
        sys, p = BarostatFixtures.lj_crystal(T, 5.26)
        sim = Simulation(sys, p, VelocityVerlet(1.0); thermostat = th, barostat = b, rng = Xoshiro(0))
        @test Base.return_types(step!, (typeof(sim),)) == [Nothing]
        @test Base.return_types(scale_factor, (BerendsenBarostat, typeof(sys), typeof(p), Float64)) == [T]
        @test Base.return_types(rescale!, (typeof(sys), T)) == [Nothing]
        @test isconcretetype(typeof(sim))
    end
    @test BerendsenBarostat <: AbstractBarostat
    @verify BerendsenBarostat trim_compat = true
    # A barostat without the contract's method fails the check.
    struct NoScaleFactor <: AbstractBarostat end
    @test_throws Exception @verify NoScaleFactor
end
