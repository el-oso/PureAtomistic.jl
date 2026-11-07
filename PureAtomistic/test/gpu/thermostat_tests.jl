# The thermostats and the barostat on the device: the Philox generator and its samplers, the Bussi
# and BAOAB steps, and the Berendsen factor, against known answers, distributions and the CPU
# arithmetic. Device trajectories match the CPU's statistically, not in their random numbers.
@testmodule DeviceThermo begin
    using PureAtomistic, PureAtomistic.Systems, PureMLIP, Random, KernelAbstractions

    const ext = Base.get_extension(PureAtomistic, :PureAtomisticGPUExt)

    # k_B in eV/K and the atomic mass unit in eV fs² / Å², written out so that the unit factors
    # are not checked against the constants they test.
    const KB_LITERAL = 8.617333262145179e-5
    const AMU_LITERAL = 103.64269652680504

    tiny(::Type{T}) where {T} = MACEModel{T}(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint()))

    # Two Cu and one H in a triclinic cell at 300 K, the system of the device MD step items.
    function host_system(::Type{T}) where {T}
        cell = T[6.5 0.0 0.0; 0.4 6.2 0.0; 0.3 0.2 6.8]
        x = T[0.0 1.4 0.2; 0.0 0.1 1.3; 0.0 0.3 0.1]
        sys = System{T}(x, zeros(T, 3, 3), T[63.546, 1.008, 63.546], [29, 1, 29], cell)
        maxwell_boltzmann!(sys, 300.0, Xoshiro(1))
        return sys
    end

    # `host` on the backend with a device potential of the tiny checkpoint.
    function simulation(host::System{T}, backend; dt = 0.5, thermostat = nothing, barostat = nothing, rng = Xoshiro(0), kw...) where {T}
        sys = PureAtomistic.to_device(host, backend)
        p = PureAtomistic.device_potential(tiny(T), sys; skin = 0, kw...)
        return Simulation(sys, p, VelocityVerlet(dt); thermostat, barostat, rng)
    end

    # An ideal gas for the device potential: `n` Cu atoms on the sites of a 10 Å grid in a 30 Å cube,
    # beyond the tiny checkpoint's 6 Å cutoff from each other, so the list holds no edge and the
    # forces are exactly zero. The mass of 10⁸ amu keeps the atoms within 0.4 Å of their sites
    # over 200 ps at 300 K (√(k_B T/m) = 1.6e-6 Å/fs), so no pair comes within the cutoff. The
    # velocities have zero center-of-mass momentum: 3n - 3 degrees of freedom.
    function ideal_gas(::Type{T}, temp; n = 11, seed = 1) where {T}
        sites = vec([(i, j, k) for i in 0:2, j in 0:2, k in 0:2])[1:n]
        x = T[10 * s[c] + 1 for c in 1:3, s in sites]
        sys = System{T}(x, zeros(T, 3, n), fill(T(1.0e8), n), fill(29, n), T[30 0 0; 0 30 0; 0 0 30])
        iszero(temp) || maxwell_boltzmann!(sys, temp, Xoshiro(seed))
        return sys
    end

    # A host copy of a device system.
    host_copy(s::System{T}) where {T} = System{T}(Array(s.positions), Array(s.velocities), Array(s.masses), Array(s.species), Array(s.cell))

    # A stress potential that only holds a stress, the CPU oracle's input to `scale_factor`.
    struct FixedStress{T} <: AbstractStressPotential
        S::Matrix{T}
    end
    PureAtomistic.Potentials.stress(p::FixedStress) = p.S

    # A source of numbers whose normals are always -100 and uniforms 1/2: every Marsaglia–Tsang
    # proposal has 1 + c x <= 0 and is rejected. `calls` counts the normals drawn on the host.
    struct StuckSource
        calls::Base.RefValue{Int}
    end
    struct StuckDeviceSource end
    ext._normal(s::StuckSource, ::Type{T}, i, j, stream) where {T} = (s.calls[] += 1; T(-100))
    ext._uniform(::StuckSource, ::Type{T}, i, j, stream) where {T} = T(0.5)
    ext._normal(::StuckDeviceSource, ::Type{T}, i, j, stream) where {T} = T(-100)
    ext._uniform(::StuckDeviceSource, ::Type{T}, i, j, stream) where {T} = T(0.5)

    mean(x) = sum(x) / length(x)
    var(x) = (m = mean(x); sum(abs2, x .- m) / (length(x) - 1))
    std(x) = sqrt(var(x))

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

    # The sample quantile at `p` with linear interpolation between order statistics (the default
    # definition of Statistics.quantile).
    function quantile(x, p)
        s = sort(x)
        h = (length(s) - 1) * p + 1
        lo = floor(Int, h)
        return lo >= length(s) ? s[end] : s[lo] + (h - lo) * (s[lo + 1] - s[lo])
    end

    # The z scores of the four quantiles `ps` of `K` against an independent sample of
    # Gamma(N_f/2, kT) drawn as kT/2 times a sum of N_f squared standard normals; the standard
    # error of each difference comes from `B` bootstrap resamples of both samples.
    function quantile_z(K, nf, kT; ps = (0.1, 0.25, 0.5, 0.9), M = 100_000, B = 100)
        rng = Xoshiro(99)
        ref = [kT / 2 * sum(abs2, randn(rng, nf)) for _ in 1:M]
        d = [quantile(K, p) - quantile(ref, p) for p in ps]
        brng = Xoshiro(5)
        boot = [
            (bK = rand(brng, K, length(K)); br = rand(brng, ref, M); [quantile(bK, p) - quantile(br, p) for p in ps])
                for _ in 1:B
        ]
        se = [std(getindex.(boot, j)) for j in eachindex(d)]
        return d ./ se
    end

    # Prints a measurement when PUREMLIP_GPU_VERBOSE=1.
    report(args...) = get(ENV, "PUREMLIP_GPU_VERBOSE", "") == "1" && (println(args...); flush(stdout))

    @kernel function philox_kernel!(out, @Const(ctr), @Const(key))
        i = @index(Global)
        w = ext._philox((ctr[1, i], ctr[2, i], ctr[3, i], ctr[4, i]), (key[1, i], key[2, i]))
        for k in 1:4
            out[k, i] = w[k]
        end
    end

    # Both normals of the Box–Muller pair of the draw at counter (i, 0, 7, 0).
    @kernel function normals_kernel!(z, src)
        i = @index(Global)
        w = ext._draw(src, i, 0, 0x00000007)
        T = eltype(z)
        z1, z2 = ext._normal2(ext._uniform(T, w[1], w[2]), ext._uniform(T, w[3], w[4]))
        z[2i - 1] = z1
        z[2i] = z2
    end

    @kernel function gamma_kernel!(g, flags, src, a)
        i = @index(Global)
        x, accepted = ext._device_gamma(src, a, i)
        g[i] = x
        flags[i] = accepted ? Int32(0) : Int32(1)
    end

    # F = -k x for springs tied to the origin.
    @kernel function spring_kernel!(F, @Const(X), k)
        c, a = @index(Global, NTuple)
        F[c, a] = -k * X[c, a]
    end

    # acc[b, 1] += Σ x²/3 and acc[b, 2] += Σ v²/3 over the components of atom 1.
    @kernel function accumulate_kernel!(acc, @Const(X), @Const(V), b)
        acc[b, 1] += (X[1, 1]^2 + X[2, 1]^2 + X[3, 1]^2) / 3
        acc[b, 2] += (V[1, 1]^2 + V[2, 1]^2 + V[3, 1]^2) / 3
    end

    # The force evaluation of the oscillator: springs of stiffness `k` into `F`.
    struct Springs{A, T, B}
        F::A
        k::T
        backend::B
    end
    (s::Springs)(sys) = spring_kernel!(s.backend)(s.F, sys.positions, s.k; ndrange = size(s.F))

    export ext, KB_LITERAL, AMU_LITERAL, tiny, host_system, simulation, ideal_gas, host_copy, FixedStress,
        StuckSource, StuckDeviceSource, mean, var, std, block_mean, block_var, quantile, quantile_z, report,
        philox_kernel!, normals_kernel!, gamma_kernel!, accumulate_kernel!, Springs
end

# Oracle: the known-answer vectors of the authors' reference implementation, Random123
# (https://github.com/DEShawResearch/random123, file tests/kat_vectors), the three lines
#   philox4x32 10 00000000 00000000 00000000 00000000 00000000 00000000   6627e8d5 e169c58d bc57ac4c 9b00dbd8
#   philox4x32 10 ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff   408f276d 41c83b0e a20bc7c6 6d5451fd
#   philox4x32 10 243f6a88 85a308d3 13198a2e 03707344 a4093822 299f31d0   d16cfe09 94fdcceb 5001e420 24126ea1
# (rounds, four counter words, two key words, four output words). Exact integer arithmetic: `==`.
@testitem "thermostats: Philox4x32-10 gives Random123's known-answer vectors" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    h(s) = Tuple(parse.(UInt32, split(s); base = 16))
    kat = [
        (h("00000000 00000000 00000000 00000000"), h("00000000 00000000"), h("6627e8d5 e169c58d bc57ac4c 9b00dbd8")),
        (h("ffffffff ffffffff ffffffff ffffffff"), h("ffffffff ffffffff"), h("408f276d 41c83b0e a20bc7c6 6d5451fd")),
        (h("243f6a88 85a308d3 13198a2e 03707344"), h("a4093822 299f31d0"), h("d16cfe09 94fdcceb 5001e420 24126ea1")),
    ]
    for (ctr, key, out) in kat
        @test ext._philox(ctr, key) == out
    end
    # The same in a kernel on the backend.
    C = todevice(UInt32[c[k] for k in 1:4, (c, _, _) in kat])
    K = todevice(UInt32[c[k] for k in 1:2, (_, c, _) in kat])
    out = GPUBackend.KernelAbstractions.zeros(backend, UInt32, 4, 3)
    philox_kernel!(backend)(out, C, K; ndrange = 3)
    @test Array(out) == UInt32[o[k] for k in 1:4, (_, _, o) in kat]
    # A step's key splits into its low and high words.
    @test ext._PhiloxStream(0x0123456789abcdef).key == (0x89abcdef, 0x01234567)
    # Uniforms on (0, 1]: the smallest word gives 2^-b, the largest 1 (exact by construction).
    @test ext._uniform(Float32, 0x00000000, 0xffffffff) === Float32(2)^-24
    @test ext._uniform(Float32, 0xffffffff, 0x00000000) === 1.0f0
    @test ext._uniform(Float64, 0x00000000, 0x00000000) === 2.0^-53
    @test ext._uniform(Float64, 0xffffffff, 0xffffffff) === 1.0
    @test ext._uniform(Float64, 0x00000000, 0x00000800) === 2 * 2.0^-53
end

# Oracle: the standard normal CDF Φ(0) = 0.5, Φ(1) = 0.8413447460685429, Φ(2) = 0.9772498680518208
# and Φ(-t) = 1 - Φ(t). 10^7 normals, 5 × 10^6 from each output of the Box–Muller pair, which are
# checked apart, so a wrong angle shows in the sine half. The fraction of n = 5 × 10^6 normals below
# t is binomial with standard error √(p(1-p)/n) (2.2e-4 at p = 0.5, 0.67e-4 at p = 0.023); the bound
# is 4 standard errors (a correct sampler exceeds one given bound with probability 6e-5). Power: a
# bias of the fraction by 8.9e-4 at p = 0.5 or 2.7e-4 at p = 0.023 fails; a radius √(-log u)
# instead of √(-2 log u) moves the fraction below 1 from 0.841 to Φ(√2) = 0.921, about 490
# standard errors. Measured |z| at most 1.55 on the CPU backend in both precisions and on Metal.
@testitem "thermostats: Box–Muller normals have the normal fractions" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    Φ = Dict(0 => 0.5, 1 => 0.8413447460685429, 2 => 0.9772498680518208, -1 => 1 - 0.8413447460685429, -2 => 1 - 0.9772498680518208)
    N = 10^7
    for T in precisions
        src = ext._PhiloxStream(0x9e3779b97f4a7c15)
        z = GPUBackend.KernelAbstractions.zeros(backend, T, N)
        normals_kernel!(backend)(z, src; ndrange = N ÷ 2)
        x = Array(z)
        @test all(isfinite, x)
        for half in (1, 2)
            y = x[half:2:end]
            n = length(y)
            for (t, p) in Φ
                f = count(<(t), y) / n
                z_score = (f - p) / sqrt(p * (1 - p) / n)
                report("normal fraction below $t, $(GPUBackend.NAME), $T, output $half: z = $z_score")
                @test abs(z_score) <= 4
            end
        end
        # `_normal` returns the first normal of the pair. The host compiles Base's `log` and
        # `sincospi` and the kernel the backend's, which may round differently: bound (b), c = 8
        # (the logarithm, the square root, the sine and cosine, the products) relative to
        # max(1, |z|). Measured: 1 ulp in Float32 on the CPU backend.
        @test abs(ext._normal(src, T, 5, 0, 0x00000007) - x[9]) <= 8 * eps(T) * max(1, abs(x[9]))
    end
end

# Oracle: Gamma(a, 1) has mean a and variance a; the sample variance of N draws has standard error
# √((μ₄ - a²)/N) with the central fourth moment μ₄ = 3a² + 6a. N = 10^6 draws per shape, each with
# its own counter slot; the bound is 4 standard errors. Power: the standard errors of the mean are
# 5.5e-4, 1.0e-3, 3.9e-3 and 2.7e-2 for a = 0.3, 1, 15, 748.5 (relative 1.8e-3, 1.0e-3, 2.6e-4,
# 3.7e-5), so a relative bias of the mean of 4 times that fails, and those of the variance are
# 1.4e-3, 2.8e-3, 2.3e-2 and 1.1 (relative 4.7e-3, 2.8e-3, 1.5e-3, 1.4e-3). Measured |z| at most
# 1.45 (mean at a = 748.5) on the CPU backend in both precisions and on Metal.
@testitem "thermostats: the device gamma sampler has the gamma moments and stops at its cap" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    using PureAtomistic.Potentials
    N = 10^6
    for T in precisions, a in (0.3, 1.0, 15.0, 748.5)
        g = GPUBackend.KernelAbstractions.zeros(backend, T, N)
        flags = GPUBackend.KernelAbstractions.zeros(backend, Int32, N)
        gamma_kernel!(backend)(g, flags, ext._PhiloxStream(0x0000000000000b0b), T(a); ndrange = N)
        x = Float64.(Array(g))
        @test iszero(Array(flags))
        @test all(>(0), x)
        zm = (mean(x) - a) / sqrt(a / N)
        zv = (var(x) - a) / sqrt((2a^2 + 6a) / N)
        report("gamma a = $a, $(GPUBackend.NAME), $T: mean z = $zm, variance z = $zv")
        @test abs(zm) <= 4
        @test abs(zv) <= 4
    end
    # A source that never yields an acceptable proposal stops after exactly 10,000 passes.
    s = StuckSource(Ref(0))
    @test ext._device_gamma(s, 2.0, 1) === (0.0, false)
    @test s.calls[] == 10_000
    @test ext._device_gamma(StuckSource(Ref(0)), 0.5f0, 1) === (0.0f0, false)
    # In the draw kernel of a Bussi step, the cap sets the thermostat's flag in the list's status
    # word, and the next status read throws.
    T = first(precisions)
    sim = simulation(host_system(T), backend; thermostat = BussiThermostat(300.0, 10.0))
    p = sim.potential
    ext._bussi_draw_kernel!(backend)(p.draws, p.nl.status, StuckDeviceSource(), T(3), true; ndrange = 1)
    @test Array(p.nl.status)[3] == 1
    @test_throws "pureatomistic: the gamma sampler rejected 10000 proposals in a row" energy_forces!(p, sim.system.positions, sim.system.cell)
    @test_throws "pureatomistic: the gamma sampler rejected 10000 proposals in a row" step!(sim)
end

# Oracle: without forces only the thermostat changes K, and its stationary distribution is Gamma
# with shape N_f/2 = 15 and scale kT: mean N_f kT/2, variance N_f (kT)²/2. The CPU suite's test
# ("Bussi ideal-gas kinetic energy follows Gamma(N_f/2, kT)") with its parameters and power: 11
# atoms (N_f = 30) at 300 K, τ = 1 fs, dt = 2 fs, samples every 5 steps over 100,000 steps (lag-1
# correlation -0.005 on the CPU), standard errors from 20 blocks of 1000 samples, quantile errors
# from a bootstrap, bounds 5 standard errors. A Bussi update without the cross term of α² gives
# 0.46 of the canonical variance, which the CPU suite measured at z -83 to -147. Measured z (CPU
# backend in both precisions, and Metal): mean -0.22, variance 0.97, quantiles 0.1, 0.25, 0.5, 0.9:
# -0.98, -1.11, -1.01, 1.16; the forces stay exactly zero and the list empty.
@testitem "thermostats: Bussi ideal-gas kinetic energy follows Gamma(N_f/2, kT) on the device" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    kT = KB_LITERAL * 300
    nf = 30
    for T in precisions
        sim = simulation(ideal_gas(T, 300.0), backend; dt = 2.0, thermostat = BussiThermostat(300.0, 1.0), rng = Xoshiro(1))
        K = Float64[]
        t = @elapsed run!(sim, 100_000; every = 5, observer = s -> push!(K, Float64(kinetic_energy(s))))
        @test iszero(Array(sim.potential.result.forces))
        @test Array(sim.potential.nl.n_edges)[1] == 0
        m, se = block_mean(K, 20)
        v, sev = block_var(K, 20)
        qz = quantile_z(K, nf, kT)
        report("Bussi ideal gas, $(GPUBackend.NAME), $T: mean z $((m - nf * kT / 2) / se), variance z $((v - nf * kT^2 / 2) / sev), quantile z $qz, $t s")
        @test abs(m - nf * kT / 2) <= 5 * se
        @test abs(v - nf * kT^2 / 2) <= 5 * sev
        @test all(z -> abs(z) <= 5, qz)
    end
end

# τ = Inf makes c = 1 and q = 0, so α = √((1 + 0)² + 0) = 1 exactly and the velocities are
# multiplied by 1: the run equals the device NVE run bit for bit, on the ideal gas and on the
# interacting tiny system.
@testitem "thermostats: Bussi with τ = Inf equals the device NVE run bit for bit" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    for T in precisions, (host, dt) in ((ideal_gas(T, 300.0), 2.0), (host_system(T), 0.5))
        a = simulation(host, backend; dt)
        b = simulation(host, backend; dt, thermostat = BussiThermostat(10.0, Inf), rng = Xoshiro(3))
        run!(a, 100)
        run!(b, 100)
        @test Array(a.system.positions) == Array(b.system.positions)
        @test Array(a.system.velocities) == Array(b.system.velocities)
        @test total_energy(a) == total_energy(b)
    end
end

# Oracle: the BAOAB step of the docstring evaluated on the host with the CPU's `kick!` and
# `drift!`, the O step written out here (c₁ = exp(-γ h), √((1 - c₁²) k_B T/m) with the literal
# constants), the device's normals of the step's key (Philox at counter (atom, component) of the
# Langevin stream, checked on their own above) and the forces that the device evaluation left at
# the new positions. This pins the order of the sub-steps and the constants c₁ and c₃. Rounding
# bound (b): c₃ differs by up to 2 roundings from the device's (other constants), and the O update
# rounds 4 times (c₃/√(m·amu) and the products and the sum), the kicks 8 times and the drifts 4
# times: c = 16 relative to the largest velocity or noise term, c = 4 relative to the largest
# position or displacement. Measured: positions 0 and 0.0007 eps, velocities 0 and 1.1 eps
# (Float32 and Float64, CPU backend); Metal 0 and 0.56 eps.
@testitem "thermostats: a device BAOAB step equals the splitting for the same noise and forces" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    using PureAtomistic.Systems
    for T in precisions
        γ, temp = 0.05, 300.0
        sim = simulation(host_system(T), backend; thermostat = LangevinBAOAB(temp, γ), rng = Xoshiro(11))
        run!(sim, 2)
        sys, p = sim.system, sim.potential
        h = T(sim.integrator.dt)
        for _ in 1:3
            X0, V0, F0, m = Array(sys.positions), Array(sys.velocities), Array(p.result.forces), Array(sys.masses)
            src = ext._PhiloxStream(rand(copy(sim.rng), UInt64))
            step!(sim)
            F1 = Array(p.result.forces)
            X, V = copy(X0), copy(V0)
            PureAtomistic.Systems.kick!(V, F0, m, h / 2)
            PureAtomistic.Systems.drift!(X, V, h / 2)
            c1 = exp(-γ * Float64(h))
            noise = 0.0
            for a in axes(V, 2), k in axes(V, 1)
                s = T(sqrt((1 - c1^2) * KB_LITERAL * temp / (m[a] * AMU_LITERAL)))
                term = s * ext._normal(src, T, a, k, ext._STREAM_LANGEVIN)
                noise = max(noise, abs(term))
                V[k, a] = T(c1) * V[k, a] + term
            end
            PureAtomistic.Systems.drift!(X, V, h / 2)
            Vh = copy(V)
            PureAtomistic.Systems.kick!(V, F1, m, h / 2)
            vscale = max(maximum(abs, V0), maximum(abs, V), noise, maximum(abs, V - Vh))
            xscale = max(maximum(abs, X), maximum(abs, X - X0))
            dx, dv = maximum(abs, Array(sys.positions) - X), maximum(abs, Array(sys.velocities) - V)
            report("BAOAB step vs splitting, $(GPUBackend.NAME), $T: positions $(dx / (eps(T) * xscale)) eps, velocities $(dv / (eps(T) * vscale)) eps")
            @test dx <= 4 * eps(T) * xscale
            @test dv <= 16 * eps(T) * vscale
        end
    end
end

# Oracle: the CPU suite's "BAOAB splitting order on one oscillator at ω h = 1", run through the
# device's BAOAB sequence (`_baoab!` with the kick, drift and O kernels and Philox noise) with a
# spring force kernel in place of the potential. One atom in reduced units, m·amu = k = kT = 1 and
# γ = h = 1: BAOAB samples ⟨x²⟩ = 1 exactly with ⟨v²⟩ = 1 - (ω h/2)² = 0.75; ABOBA gives 1 and
# 4/3, OBABO 4/3 and 1. 200,000 steps after 100 of burn-in, the 3 components averaged per step,
# standard errors from 20 blocks, bound 5 of them, as on the CPU, where a reordered step failed by
# at least 19 times the bound. The sums are kept on the device in 2,000 blocks of 100 steps.
# Measured: ⟨x²⟩ 1.0012 ± 0.0022 (z 0.55), ⟨v²⟩ 0.7502 ± 0.0013 (z 0.13) on the CPU backend in
# both precisions and on Metal.
@testitem "thermostats: the device BAOAB sequence samples the oscillator at ω h = 1" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    KA = GPUBackend.KernelAbstractions
    for T in precisions
        host = System{T}(zeros(T, 3, 1), zeros(T, 3, 1), T[1 / AMU_LITERAL], [1], T[100 0 0; 0 100 0; 0 0 100])
        sys = PureAtomistic.to_device(host, backend)
        F = KA.zeros(backend, T, 3, 1)
        springs = Springs(F, one(T), backend)
        c1 = T(exp(-1.0))
        c3 = T(sqrt(-expm1(-2.0) * PureAtomistic.Systems.KB_EV_PER_K * (1 / KB_LITERAL)))
        rng = Xoshiro(1)
        step() = ext._baoab!(backend, sys, F, one(T), c1, c3, ext._PhiloxStream(rand(rng, UInt64)), PureAtomistic.Systems.no_hook, springs)
        for _ in 1:100
            step()
        end
        nb, L = 2_000, 100
        acc = KA.zeros(backend, T, nb, 2)
        t = @elapsed for b in 1:nb, _ in 1:L
            step()
            accumulate_kernel!(backend)(acc, sys.positions, sys.velocities, b; ndrange = 1)
        end
        A = Float64.(Array(acc)) ./ L
        x2, sx = block_mean(A[:, 1], 20)
        v2, sv = block_mean(A[:, 2], 20)
        report("BAOAB oscillator, $(GPUBackend.NAME), $T: ⟨x²⟩ $x2 ± $sx, ⟨v²⟩ $v2 ± $sv, $t s")
        @test abs(x2 - 1) <= 5 * sx
        @test abs(v2 - 0.75) <= 5 * sv
    end
end

# Oracle: without forces the O step is the exact Ornstein–Uhlenbeck update, so each velocity
# component has ⟨v²⟩ = kT/m. 11 atoms at 300 K with γ = 0.01/fs and dt = 2 fs over 40,000 steps,
# sampled every 10 steps after the first 1000 samples (1/γ = 50 steps); standard error from 20
# blocks of 150 samples (3 ps, 30/γ); bound 5 of them. Measured: ⟨v²⟩ m/kT = 1.0071 ± 0.0111
# (z 0.64) on the CPU backend in both precisions and on Metal. Power: a relative bias of 5.5 %
# fails; an O step with (1 - c₁) in place of (1 - c₁²) gives ⟨v²⟩ m/kT = 1/(1 + c₁) = 0.51.
@testitem "thermostats: Langevin on the device ideal gas gives ⟨v²⟩ = kT/m" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    for T in precisions
        sim = simulation(ideal_gas(T, 300.0), backend; dt = 2.0, thermostat = LangevinBAOAB(300.0, 0.01), rng = Xoshiro(2))
        V2 = Float64[]
        t = @elapsed run!(sim, 40_000; every = 10, observer = s -> push!(V2, mean(abs2.(Float64.(Array(s.system.velocities))))))
        @test iszero(Array(sim.potential.result.forces))
        v2, sv = block_mean(V2[1001:end], 20)
        ref = KB_LITERAL * 300 / (1.0e8 * AMU_LITERAL)
        report("Langevin ideal gas, $(GPUBackend.NAME), $T: ⟨v²⟩ / (kT/m) = $(v2 / ref) ± $(sv / ref), z $((v2 - ref) / sv), $t s")
        @test abs(v2 - ref) <= 5 * sv
    end
end

@testitem "thermostats: seeded device runs reproduce bit for bit" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    for T in precisions, th in (LangevinBAOAB(300.0, 0.01), BussiThermostat(300.0, 50.0; removed_dof = 0))
        a, b, c = (simulation(host_system(T), backend; thermostat = th, rng = Xoshiro(s)) for s in (42, 42, 43))
        run!(a, 20)
        run!(b, 20)
        run!(c, 20)
        @test Array(a.system.positions) == Array(b.system.positions)
        @test Array(a.system.velocities) == Array(b.system.velocities)
        @test Array(a.system.velocities) != Array(c.system.velocities)
    end
end

# A thermostatted step reads back only the status word of its force evaluation; the barostat adds
# the pressure. Each step also draws exactly one key from the simulation's generator.
@testitem "thermostats: readbacks per step are 1 with a thermostat and 2 with the barostat" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    T = first(precisions)
    b = BerendsenBarostat(0.0, 100.0, 0.01)
    for th in (nothing, BussiThermostat(300.0, 50.0), LangevinBAOAB(300.0, 0.01)), baro in (nothing, b)
        sim = simulation(host_system(T), backend; thermostat = th, barostat = baro, rng = Xoshiro(5))
        p = sim.potential
        per_step = isnothing(baro) ? 1 : 2
        for k in 1:5
            r = p.readbacks
            step!(sim)
            @test p.readbacks == r + per_step
        end
        reference = Xoshiro(5)
        isnothing(th) || foreach(_ -> rand(reference, UInt64), 1:5)
        @test sim.rng == reference
        @test p.n_regrows == 0
    end
end

# Host bytes allocated by one thermostatted or barostatted `step!`, through a compiled function so
# that the dynamic dispatch of the call site is not counted: the fewest of five steps after two
# warm-up steps, on the tiny checkpoint's three atoms. Measured:
#   CPU backend, Julia 1.12 and 1.13, Float32 and Float64: 2,464 bytes for every combination of
#   Bussi, Langevin or no thermostat with or without the barostat, the bytes of the NVE step;
#   Metal, Julia 1.13, Float32, the largest fewest-of-five over 4 groups in each of 3 processes:
#   Bussi 462,384, Bussi and barostat 465,936, Langevin 458,656, Langevin and barostat 462,272,
#   barostat alone 461,104.
# The gate is the measured figure on the CPU backend on macOS and twice it on other platforms; on
# Metal 470,000 (0.9 % above the largest).
@testitem "thermostats: host bytes per thermostatted and barostatted step" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    @noinline step_bytes(sim) = @allocated step!(sim)
    cpu = Sys.isapple() ? 1 : 2
    bounds = Dict("CPU" => cpu * 2_464, "Metal" => 470_000)
    for T in precisions, th in (BussiThermostat(300.0, 50.0), LangevinBAOAB(300.0, 0.01), nothing), baro in (nothing, BerendsenBarostat(0.0, 100.0, 0.01))
        isnothing(th) && isnothing(baro) && continue
        sim = simulation(host_system(T), backend; thermostat = th, barostat = baro)
        step_bytes(sim)
        step_bytes(sim)
        b = minimum(step_bytes(sim) for _ in 1:5)
        report("host bytes per step!, $(GPUBackend.NAME), $T, $(typeof(th)), $(typeof(baro)), Julia $VERSION: $b")
        if haskey(bounds, GPUBackend.NAME)
            @test b <= bounds[GPUBackend.NAME]
        else
            # CUDA and AMDGPU are measured on their machines.
            @test_skip b <= 0
        end
    end
end

# A regrowth of the list does not touch the generator: a step draws one key whatever its
# evaluation does. Two runs from one seed, one with no spare edge room (the compression regrows
# it) and one with room for 4 times its edges more (it never regrows), end with the same generator
# state, that of the seed advanced by one key per step, and the same trajectory bit for bit (the
# evaluation does not depend on the capacity).
@testitem "thermostats: a regrowth does not shift the random stream" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    for T in precisions, th in (LangevinBAOAB(300.0, 0.05), BussiThermostat(300.0, 20.0))
        compress!(s) = (s.cell .*= T(0.85); s.positions .*= T(0.85); nothing)
        sims = [simulation(host_system(T), backend; thermostat = th, rng = Xoshiro(7), extra_capacity_fraction = f) for f in (0.0, 4.0)]
        for sim in sims
            run!(sim, 3)
            PureAtomistic._step!(sim.thermostat, sim, compress!)
            run!(sim, 5)
        end
        a, b = sims
        @test a.potential.n_regrows >= 1 && b.potential.n_regrows == 0
        reference = Xoshiro(7)
        foreach(_ -> rand(reference, UInt64), 1:9)
        @test a.rng == reference && b.rng == reference
        @test Array(a.system.positions) == Array(b.system.positions)
        @test Array(a.system.velocities) == Array(b.system.velocities)
    end
end

# Oracle: the CPU `pressure` and `scale_factor` on a host copy of the state, with the device's
# stress. The device sums the kinetic energy in a tree: its error is within (n + 4) eps K (the
# md step items), and the pressure adds a rounding of each of its 5 further operations: c = n + 9
# relative to |tr S|/3 + 2K/(3V). μ³ = 1 + a (P - P0) with a = β_T dt/τ_P moves by a ΔP, and
# μ by that over 3 μ², plus 2 roundings of μ. Measured: the pressure and μ equal the CPU's bit for
# bit on the CPU backend in both precisions and on Metal.
@testitem "barostat: the device scale factor equals the CPU's on the same state" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    using PureAtomistic.Systems, PureAtomistic.Barostats
    for T in precisions
        b = BerendsenBarostat(0.5, 100.0, 0.02)
        sim = simulation(host_system(T), backend; thermostat = LangevinBAOAB(300.0, 0.01))
        run!(sim, 3)
        sys, p = sim.system, sim.potential
        h = host_copy(sys)
        S = Array(p.result.stress)
        K = kinetic_energy(h)
        V = volume(h)
        scale = abs(S[1, 1] + S[2, 2] + S[3, 3]) / 3 + 2K / (3V)
        r = p.readbacks
        P = ext._device_pressure!(sys, p)
        @test p.readbacks == r + 1
        @test P isa T
        dP = abs(P - pressure(h, S))
        report("device pressure, $(GPUBackend.NAME), $T: $(dP / (eps(T) * scale)) eps")
        @test dP <= (natoms(h) + 9) * eps(T) * scale
        μ = scale_factor(b, sys, p, 0.5)
        μ_cpu = scale_factor(b, h, FixedStress(S), 0.5)
        @test μ isa T
        a = 0.02 * 0.5 / 100.0
        bound = a * (natoms(h) + 9) * eps(T) * scale * PureAtomistic.Systems.EV_PER_A3_IN_GPA / (3 * μ_cpu^2) + 2 * eps(T) * μ_cpu
        report("device scale factor, $(GPUBackend.NAME), $T: μ - 1 = $(μ_cpu - 1), |Δμ| = $(abs(μ - μ_cpu)), bound $bound")
        @test abs(μ - μ_cpu) <= bound
        # The kinetic part matters at this state: it is above the bound by a large factor.
        @test 2K / (3V) > 1000 * (natoms(h) + 9) * eps(T) * scale
        # rescale! scales the device arrays by one product per entry.
        X, C = Array(sys.positions), Array(sys.cell)
        rescale!(sys, μ)
        @test Array(sys.positions) == X .* μ && Array(sys.cell) == C .* μ
        # A factor that would collapse the cell throws before anything moves.
        @test_throws "pureatomistic: the Berendsen step would make the cell degenerate (volume factor" scale_factor(BerendsenBarostat(1.0e6, 1.0, 1.0), sys, p, 1.0)
    end
end

@testitem "thermostats: device thermostat steps fail fast" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    T = first(precisions)
    # One atom with the center-of-mass momentum removed has no degree of freedom.
    one_atom = System{T}(zeros(T, 3, 1), T[0.01; 0; 0;;], T[63.546], [29], T[10 0 0; 0 10 0; 0 0 10])
    sim = simulation(one_atom, backend; thermostat = BussiThermostat(300.0, 10.0))
    @test_throws "pureatomistic: the Bussi thermostat needs at least one degree of freedom, got 0" step!(sim)
    # With N_f = 1 there is no gamma variate; the step runs.
    sim = simulation(one_atom, backend; thermostat = BussiThermostat(300.0, 10.0; removed_dof = 2))
    run!(sim, 5)
    @test all(isfinite, Array(sim.system.velocities))
    # Zero kinetic energy: the device sets the thermostat's flag, and the step's status read throws.
    sim = simulation(ideal_gas(T, 0.0), backend; dt = 2.0, thermostat = BussiThermostat(300.0, 10.0))
    @test_throws "pureatomistic: the Bussi thermostat cannot rescale velocities with zero kinetic energy" step!(sim)
    # An unknown code in the thermostat's entry.
    @test_throws "pureatomistic: the thermostat flag holds the unknown code 7" ext._check_thermostat_flag(Int32[0, 0, 7, 0, 0])
    # Arrays of the system on another backend than the potential's.
    sim = simulation(host_system(T), backend; thermostat = LangevinBAOAB(300.0, 0.01))
    s = sim.system
    small = PureAtomistic.to_device(System{T}(zeros(T, 3, 2), zeros(T, 3, 2), T[1, 1], [29, 29], Array(s.cell)), backend)
    for th in (LangevinBAOAB(300.0, 0.01), BussiThermostat(300.0, 10.0))
        @test_throws "pureatomistic: the system has 2 atoms, the potential 3" thermostat_step!(th, small, sim.potential, 0.5, Xoshiro(0), nothing)
    end
    @test_throws "pureatomistic: the system has 2 atoms, the potential 3" scale_factor(BerendsenBarostat(0.0, 100.0, 0.01), small, sim.potential, 0.5)
end

# The kernels' arithmetic as host code in Float32 and the extension's compiled functions after
# Float32 thermostatted and barostatted steps.
@testitem "thermostats: Float32 thermostat and barostat steps do not widen to Float64" tags = [:gpu] setup = [GPUBackend, DeviceThermo] begin
    using Random, PureAtomistic.Systems, PureAtomistic.Thermostats, PureAtomistic.Barostats
    using StrictMode
    T = Float32
    for th in (BussiThermostat(300.0, 10.0), LangevinBAOAB(300.0, 0.01))
        sim = simulation(host_system(T), backend; thermostat = th, barostat = BerendsenBarostat(0.0, 100.0, 0.01))
        run!(sim, 2)
    end
    src = ext._PhiloxStream(0x0000000000000001)
    @test ext._normal2(0.5f0, 0.25f0) isa NTuple{2, T}
    @test ext._normal(src, T, 1, 1, ext._STREAM_LANGEVIN) isa T
    @test ext._device_gamma(src, 0.5f0, 1)[1] isa T
    @test ext._device_gamma(src, 15.0f0, 1)[1] isa T
    @test ext._o_update(1.0f0, 2.0f0, 0.5f0, 0.1f0, 103.0f0, 0.3f0) isa T
    @test ext._bussi_alpha(zeros(Int32, 5), 1.0f0, 0.5f0, 2.0f0, 0.9f0, 0.1f0, 30.0f0) isa T
    @test ext._pressure(1.0f0, rand(T, 3, 3), rand(T, 3, 3) + T[4 0 0; 0 4 0; 0 0 4]) isa T
    @test isempty(StrictMode.conversion_report(ext; from = Float32, to = Float64, exempt = (:cutoff,)))
end
