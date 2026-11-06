@testmodule SystemFixtures begin
    using PureAtomistic.Systems
    using OffsetArrays, Random

    # A small triclinic cell (columns are the lattice vectors) and a few atoms inside it.
    const CELL = [6.5 0.4 0.3; 0.0 6.2 0.2; 0.0 0.0 6.8]

    # A cell with all nine entries nonzero.
    const GENERAL_CELL = [6.5 0.4 -0.3; 0.5 6.2 0.2; -0.3 0.6 6.8]

    function small_system(::Type{T}, cell = CELL) where {T}
        P = T[0.0 1.4 0.2 3.0; 0.0 0.1 1.3 2.5; 0.0 0.3 0.1 4.0]
        V = T[0.01 -0.02 0.005 0.0; 0.0 0.015 -0.01 0.02; -0.012 0.0 0.02 0.01]
        return System{T}(P, V, T[63.546, 1.008, 63.546, 12.011], [29, 1, 29, 6], cell)
    end

    # A Cu fcc crystal with every atom displaced by a random amount.
    function perturbed_cu(::Type{T}, rng) where {T}
        sys = fcc_cell(T(3.6), 2, :Cu)
        sys.positions .+= T(0.05) .* randn(rng, T, size(sys.positions))
        return sys
    end
end

@testitem "unit constants equal their published values" begin
    using PureAtomistic.Systems
    # Oracle: CODATA 2018 / SI 2019 values as published, each to the digits quoted, so the
    # relative tolerance is one unit in the last quoted digit (10 digits: 1e-9).
    @test KB_EV_PER_K ≈ 8.617333262e-5 rtol = 1.0e-9
    @test AMU_IN_EV_FS2_PER_A2 ≈ 103.6426965 rtol = 1.0e-9
    # 160.2176634 is exact (e / 1 Å³ in GPa).
    @test EV_PER_A3_IN_GPA ≈ 160.2176634 rtol = 1.0e-10
end

@testitem "System validates its inputs" setup = [SystemFixtures] begin
    using PureAtomistic.Systems
    P = zeros(3, 2); V = zeros(3, 2); m = [1.0, 2.0]; Z = [1, 1]; C = SystemFixtures.CELL
    sys = System(P, V, m, Z, C)
    @test sys isa System{Float64}
    @test natoms(sys) == 2
    sys32 = System{Float32}(P, V, m, Z, C)
    @test sys32 isa System{Float32}
    @test eltype(sys32.cell) == Float32
    @test_throws "pureatomistic: positions need 3 rows, got 2" System(zeros(2, 2), zeros(2, 2), m, Z, C)
    @test_throws "pureatomistic: velocities axes" System(P, zeros(3, 3), m, Z, C)
    @test_throws "pureatomistic: masses axes" System(P, V, [1.0], Z, C)
    @test_throws "pureatomistic: species axes" System(P, V, m, [1], C)
    @test_throws "pureatomistic: the cell must be 3 × 3" System(P, V, m, Z, zeros(3, 2))
    @test_throws "pureatomistic: masses must be positive" System(P, V, [1.0, 0.0], Z, C)
end

@testitem "volume and wrap! agree with a LinearAlgebra oracle on triclinic cells" setup = [SystemFixtures] begin
    using PureAtomistic.Systems, LinearAlgebra, Random
    rng = Xoshiro(1)
    # Oracle: det and inv from LinearAlgebra on the same matrix.
    for C in (SystemFixtures.CELL, SystemFixtures.GENERAL_CELL), T in (Float64, Float32)
        sys = SystemFixtures.small_system(T, C)
        # Rounding bound: det is 14 flops on entries of size <= 7; c = 8, N = 7^3 relative to |det|.
        @test volume(sys) ≈ abs(det(C)) rtol = 8 * eps(T) * 343 / abs(det(C))
        @test cell_volume(T.(C)) == volume(sys)
        orig = copy(sys.positions)
        # Shift every atom by a random integer combination of the lattice vectors.
        shifts = rand(rng, -3:3, 3, 4)
        sys.positions .+= T.(C) * shifts
        wrap!(sys)
        frac = inv(C) * Float64.(sys.positions)
        # Every fractional coordinate lies in [0, 1) up to the rounding of the shifted position:
        # c = 8 (shift, three products), N = 20 (the magnitude of a shifted coordinate).
        slack = 8 * eps(T) * 20
        @test all(frac .>= -slack) && all(frac .< 1 + slack)
        # Wrapping moves each atom by a lattice vector: the fractional displacement from the
        # original is integral. (An atom at the origin may land on the opposite face, since its
        # fractional coordinate is an integer to within rounding.) Same rounding bound as above.
        d = inv(C) * (Float64.(sys.positions) - Float64.(orig))
        @test maximum(abs.(d - round.(d))) < slack
    end
end

@testitem "wrap! leaves atoms inside the cell untouched and rejects bad input" setup = [SystemFixtures] begin
    using PureAtomistic.Systems
    sys = SystemFixtures.small_system(Float64)
    orig = copy(sys.positions)
    wrap!(sys)
    @test sys.positions == orig
    sys.positions[2, 3] = NaN
    @test_throws "pureatomistic: atom 3 has a non-finite position" wrap!(sys)
    flat = System(zeros(3, 1), zeros(3, 1), [1.0], [1], [1.0 0 0; 0 1.0 0; 0 0 0.0])
    @test_throws "pureatomistic: the cell is singular" wrap!(flat)
    @test_throws "pureatomistic: the cell is singular" pressure(flat, zeros(3, 3))
end

@testitem "kinetic energy, temperature and pressure match closed forms" setup = [SystemFixtures] begin
    using PureAtomistic.Systems
    for T in (Float64, Float32)
        sys = SystemFixtures.small_system(T)
        V = Float64.(sys.velocities); m = Float64.(sys.masses)
        # Closed form with the constants of the Systems module; four additions and a product per
        # atom: c = 4, N = 4 atoms x 3 components.
        ke = 0.5 * AMU_IN_EV_FS2_PER_A2 * sum(m[a] * sum(abs2, V[:, a]) for a in 1:4)
        @test kinetic_energy(sys) ≈ ke rtol = 4 * eps(T) * 12
        @test temperature(sys) ≈ 2ke / (9 * KB_EV_PER_K) rtol = 8 * eps(T) * 12
        @test temperature(sys; dof = 12) ≈ 2ke / (12 * KB_EV_PER_K) rtol = 8 * eps(T) * 12
        # Ideal gas, no interactions: P V = N k_B T with the 3N-dof temperature.
        vol = abs(1.0 * 6.5 * 6.2 * 6.8)
        @test volume(sys) ≈ vol rtol = 8 * eps(T) * 3
        @test pressure(sys, zeros(3, 3)) * volume(sys) ≈ 4 * KB_EV_PER_K * temperature(sys; dof = 12) rtol = 16 * eps(T) * 12
        # A hydrostatic tension raises -tr/3 into P with the opposite sign.
        s = T[0.01 0 0; 0 0.01 0; 0 0 0.01]
        @test pressure(sys, s) ≈ pressure(sys, zeros(3, 3)) - 0.01 rtol = 16 * eps(T) * 50
        p = total_momentum(sys)
        @test all(p[d] ≈ sum(m[a] * V[d, a] for a in 1:4) for d in 1:3)
    end
    sys = SystemFixtures.small_system(Float64)
    @test_throws "pureatomistic: the stress must be 3 × 3, got (2, 2)" pressure(sys, zeros(2, 2))
    @test_throws "pureatomistic: the cell must be 3 × 3, got (2, 2)" cell_volume(zeros(2, 2))
end

@testitem "temperature with no degree of freedom throws" begin
    using PureAtomistic.Systems, Random
    one = System(zeros(3, 1), zeros(3, 1), [1.0], [1], [5.0 0 0; 0 5.0 0; 0 0 5.0])
    none = System(zeros(3, 0), zeros(3, 0), Float64[], Int[], [5.0 0 0; 0 5.0 0; 0 0 5.0])
    @test_throws "pureatomistic: temperature needs at least one degree of freedom" temperature(one)
    @test_throws "pureatomistic: temperature needs at least one degree of freedom" temperature(none)
    @test_throws "pureatomistic: temperature needs at least one degree of freedom" temperature(one; dof = 0)
    # With the center-of-mass motion kept, one atom has three degrees of freedom.
    one.velocities .= 0.01
    @test temperature(one; dof = 3) > 0
    @test_throws "pureatomistic: temperature needs at least one degree of freedom" maxwell_boltzmann!(one, 300, Xoshiro(1))
    @test_throws "pureatomistic: temperature needs at least one degree of freedom" maxwell_boltzmann!(none, 300, Xoshiro(1); remove_com = false)
    maxwell_boltzmann!(one, 300, Xoshiro(1); remove_com = false)
    @test temperature(one; dof = 3) ≈ 300 rtol = 1.0e-12
end

@testitem "maxwell_boltzmann! hits the target temperature exactly and removes the momentum" setup = [SystemFixtures] begin
    using PureAtomistic.Systems, Random
    for T in (Float64, Float32)
        sys = SystemFixtures.perturbed_cu(T, Xoshiro(2))
        maxwell_boltzmann!(sys, 900, Xoshiro(3))
        # The rescale divides by a temperature computed from the same sum; the result is exact up
        # to the rounding of that sum: c = 4, N = 3n = 96 terms.
        @test temperature(sys) ≈ 900 rtol = 4 * eps(T) * 96
        p = total_momentum(sys)
        # Momentum after subtracting P/M: each component is a sum of 32 products of size
        # m v ~ 63.5 * sqrt(kT/(m AMU)) ~ 0.4; c = 4, N = 32 terms of that size.
        @test all(abs(p[d]) < 4 * eps(T) * 32 * 0.5 for d in 1:3)
        sys2 = SystemFixtures.perturbed_cu(T, Xoshiro(2))
        maxwell_boltzmann!(sys2, 900, Xoshiro(3); remove_com = false)
        @test temperature(sys2; dof = 96) ≈ 900 rtol = 4 * eps(T) * 96
        @test any(abs(total_momentum(sys2)[d]) > 1.0e-3 for d in 1:3)
        # A seed reproduces the velocities bit for bit; a different seed does not.
        sys3 = SystemFixtures.perturbed_cu(T, Xoshiro(2))
        maxwell_boltzmann!(sys3, 900, Xoshiro(3))
        @test sys3.velocities == sys.velocities
        maxwell_boltzmann!(sys3, 900, Xoshiro(4))
        @test sys3.velocities != sys.velocities
    end
    sys = SystemFixtures.perturbed_cu(Float64, Xoshiro(2))
    maxwell_boltzmann!(sys, 0, Xoshiro(3))
    @test all(iszero, sys.velocities)
    @test_throws "pureatomistic: the target temperature must be nonnegative, got -1" maxwell_boltzmann!(sys, -1, Xoshiro(3))
end

@testitem "maxwell_boltzmann! samples a Gaussian whose variance scales as 1/m" begin
    using PureAtomistic.Systems, Random, Statistics
    n = 20000
    m = [i <= n ÷ 2 ? 4.0 : 16.0 for i in 1:n]
    sys = System(zeros(3, n), zeros(3, n), m, ones(Int, n), [50.0 0 0; 0 50.0 0; 0 0 50.0])
    maxwell_boltzmann!(sys, 300, Xoshiro(11))
    V = sys.velocities
    light = vec(V[:, 1:(n ÷ 2)]); heavy = vec(V[:, (n ÷ 2 + 1):end])
    # Equipartition per species: m <v^2> is the same, so the variance ratio equals the inverse mass
    # ratio (4). Statistical error of a variance ratio from 30000 samples each: sqrt(2/3e4)·sqrt(2)
    # = 0.012 relative; tolerance 0.05 (about 4 sigma).
    @test var(light) / var(heavy) ≈ 4 rtol = 0.05
    # Kurtosis 3 for a Gaussian; its standard error is sqrt(24/N) = 0.0283 for N = 30000; the
    # tolerance 0.15 is 5 sigma.
    for x in (light, heavy)
        @test mean(x .^ 4) / var(x)^2 ≈ 3 atol = 0.15
    end
end

@testitem "maxwell_boltzmann! rejects velocities with no kinetic energy" begin
    using PureAtomistic.Systems, Random
    # A generator that draws only zero bits makes every normal deviate zero.
    struct ZeroBitsRNG <: Random.AbstractRNG end
    Random.rng_native_52(::ZeroBitsRNG) = UInt64
    Random.rand(::ZeroBitsRNG, ::Random.SamplerType{UInt64}) = 0x0000000000000000
    sys = fcc_cell(3.6, 1, :Cu)
    @test_throws "pureatomistic: the sampled velocities carry no kinetic energy" maxwell_boltzmann!(sys, 300, ZeroBitsRNG())
end

@testitem "fcc_cell builds the crystal and remove_atom deletes an atom" begin
    using PureAtomistic.Systems, LinearAlgebra
    a = 3.6
    sys = fcc_cell(a, (2, 3, 1), :Cu)
    @test natoms(sys) == 24
    @test sys.cell == Diagonal([2a, 3a, a])
    @test all(==(29), sys.species) && all(==(63.546), sys.masses)
    @test all(iszero, sys.velocities)
    @test volume(sys) ≈ 6 * a^3 rtol = 4 * eps()
    # Brute-force oracle: each atom has 12 nearest neighbors at a / sqrt(2) under the minimum image
    # convention (cell edges are at least 2 a, so one image shell suffices).
    big = fcc_cell(a, 2, :Cu)
    L = big.cell[1, 1]
    for i in 1:size(big.positions, 2)
        d = [
            norm(
                let x = big.positions[:, i] - big.positions[:, j]
                    x .- L .* round.(x ./ L)
                end
            ) for j in 1:32 if j != i
        ]
        @test count(<(a / sqrt(2) + 1.0e-9), d) == 12
        @test minimum(d) ≈ a / sqrt(2) rtol = 8 * eps()
    end
    @test fcc_cell(Float32(a), 1, 29) isa System{Float32}
    @test fcc_cell(a, 1, :Cu; mass = 10.0).masses == fill(10.0, 4)
    @test atomic_number(:Au) == 79 && atomic_mass(29) == 63.546
    @test_throws "pureatomistic: unknown element Xx" fcc_cell(a, 1, :Xx)
    @test_throws "pureatomistic: unknown element 3" atomic_mass(3)
    @test_throws "pureatomistic: nrep must be at least 1 in each direction, got (1, 0, 1)" fcc_cell(a, (1, 0, 1), :Cu)
    @test_throws "pureatomistic: the lattice constant must be positive, got -1.0" fcc_cell(-1.0, 1, :Cu)
    v = remove_atom(sys, 5)
    @test natoms(v) == 23
    keep = [1:4; 6:24]
    @test v.positions == sys.positions[:, keep] && v.cell === sys.cell
    @test v.species == sys.species[keep]
    @test_throws "pureatomistic: atom 25 is outside the atom axis" remove_atom(sys, 25)
    @test_throws "pureatomistic: atom 0 is outside the atom axis" remove_atom(sys, 0)
end

@testitem "wrapping leaves a MACE evaluation unchanged (atoms leaving a triclinic cell)" setup = [SystemFixtures] begin
    using PureAtomistic.Systems, Random
    PM = PureAtomistic.PureMLIP
    model = PM.MACEModel(PM.load_mace_checkpoint(PM.tiny_mace_checkpoint()))
    rng = Xoshiro(5)
    # Cu in a sheared cell: lattice vectors with off-diagonal components.
    cell = [7.2 0.5 0.4; 0.0 7.1 0.3; 0.0 0.0 7.3]
    sys = SystemFixtures.perturbed_cu(Float64, rng)
    sys = System(sys.positions, sys.velocities, sys.masses, sys.species, cell)
    ref = PM.mace_energy_forces(model, sys.positions, cell, sys.species)
    shifts = rand(rng, -3:3, 3, natoms(sys))
    moved = copy(sys.positions) .+ cell * shifts
    @test moved != sys.positions
    unwrapped = PM.mace_energy_forces(model, moved, cell, sys.species)
    shifted = System(moved, sys.velocities, sys.masses, sys.species, cell)
    wrap!(shifted)
    wrapped = PM.mace_energy_forces(model, shifted.positions, cell, sys.species)
    # Oracle: the evaluation of the original positions. Measured margin over 20 random shifts of
    # 32 atoms, unwrapped and wrapped: the largest differences are 3.6e-15 eV (energy, |E| = 17.6),
    # 8.1e-16 eV/Å (forces) and 5.8e-18 eV/Å³ (stress); the tolerances are 14, 18 and 17 times
    # those.
    for r in (unwrapped, wrapped)
        @test r.energy ≈ ref.energy atol = 5.0e-14
        @test r.forces ≈ ref.forces atol = 1.5e-14
        @test r.stress ≈ ref.stress atol = 1.0e-16
    end
end

@testitem "observables accept OffsetArray and view inputs" setup = [SystemFixtures] begin
    using PureAtomistic.Systems, OffsetArrays, Random
    for T in (Float64, Float32)
        plain = SystemFixtures.small_system(T)
        big = zeros(T, 3, 10); big[:, 3:6] .= plain.positions
        bigv = zeros(T, 3, 10); bigv[:, 3:6] .= plain.velocities
        viewed = System(view(big, :, 3:6), view(bigv, :, 3:6), view(T[0, 0, 63.546, 1.008, 63.546, 12.011, 0, 0], 3:6), view([0, 0, 29, 1, 29, 6, 0, 0], 3:6), plain.cell)
        off = System(
            OffsetArray(copy(plain.positions), 0:2, -1:2), OffsetArray(copy(plain.velocities), 0:2, -1:2),
            OffsetArray(copy(plain.masses), -1:2), OffsetArray([29, 1, 29, 6], -1:2), OffsetArray(copy(plain.cell), 0:2, 5:7)
        )
        for s in (viewed, off)
            @test natoms(s) == 4
            @test kinetic_energy(s) == kinetic_energy(plain)
            @test temperature(s) == temperature(plain)
            @test total_momentum(s) == total_momentum(plain)
            @test volume(s) == volume(plain)
            @test pressure(s, OffsetArray(zeros(T, 3, 3), 0:2, 0:2)) == pressure(plain, zeros(T, 3, 3))
        end
        # wrap! moves the atoms as in the plain array, and the output axes follow the input axes.
        shift = T.(plain.cell) * [1 0 -1 2; 0 2 1 0; 1 1 0 -1]
        viewed.positions .+= shift
        parent(off.positions) .+= shift
        plain.positions .+= shift
        wrap!(plain); wrap!(viewed); wrap!(off)
        @test collect(viewed.positions) == plain.positions
        @test collect(off.positions) == plain.positions
        @test axes(off.positions) == (0:2, -1:2)
        for s in (viewed, off)
            maxwell_boltzmann!(s, 300, Xoshiro(7))
        end
        maxwell_boltzmann!(plain, 300, Xoshiro(7))
        @test collect(viewed.velocities) == plain.velocities
        @test collect(off.velocities) == plain.velocities
        @test temperature(off) ≈ 300 rtol = 4 * eps(T) * 12
        r = remove_atom(off, 0)
        @test natoms(r) == 3 && r.positions == plain.positions[:, [1, 3, 4]]
        r = remove_atom(viewed, 3)
        @test natoms(r) == 3 && r.positions == plain.positions[:, [1, 2, 4]]
        @test_throws "pureatomistic: atom 3 is outside the atom axis" remove_atom(off, 3)
    end
end

@testitem "StrictMode: observables and in-place updates are allocation-free" tags = [:checks] setup = [SystemFixtures] begin
    using StrictModeTest, StrictMode, PureAtomistic.Systems, Random
    @noinline ke(s) = kinetic_energy(s)
    @noinline temp(s) = temperature(s)
    @noinline temp3(s) = temperature(s; dof = 3)
    @noinline pres(s, st) = pressure(s, st)
    @noinline mom(s) = total_momentum(s)
    @noinline vol(s) = volume(s)
    @noinline wrp!(s) = wrap!(s)
    @noinline mb!(s, rng) = maxwell_boltzmann!(s, 300, rng)
    @noinline mb0!(s, rng) = maxwell_boltzmann!(s, 300, rng; remove_com = false)
    for T in (Float64, Float32)
        sys = SystemFixtures.perturbed_cu(T, Xoshiro(1)); st = zeros(T, 3, 3); rng = Xoshiro(2)
        ke(sys); temp(sys); temp3(sys); pres(sys, st); mom(sys); vol(sys); wrp!(sys); mb!(sys, rng); mb0!(sys, rng)
        @test_noalloc ke(sys)
        @test_noalloc temp(sys)
        @test_noalloc temp3(sys)
        @test_noalloc pres(sys, st)
        @test_noalloc mom(sys)
        @test_noalloc vol(sys)
        sys.positions .+= T(20)
        @test_noalloc wrp!(sys)
        @test_noalloc mb!(sys, rng)
        @test_noalloc mb0!(sys, rng)
    end
end

@testitem "TrimCheck: system functions" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureAtomistic.Systems, Random
            new_system64(p, v, m, z, c) = System{Float64}(p, v, m, z, c)
            new_system32(p, v, m, z, c) = System{Float32}(p, v, m, z, c)
            fcc64(a, n, e) = fcc_cell(a, n, e)
            fcc32(a, n, e) = fcc_cell(a, n, e)
        end,
        new_system64(Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}),
        new_system32(Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}),
        natoms(System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}),
        volume(System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}),
        wrap!(System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}),
        wrap!(System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}),
        kinetic_energy(System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}),
        kinetic_energy(System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}),
        total_momentum(System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}),
        total_momentum(System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}),
        temperature(System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}),
        temperature(System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}),
        pressure(System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}, Matrix{Float64}),
        pressure(System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}, Matrix{Float32}),
        maxwell_boltzmann!(System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}, Float64, Xoshiro),
        maxwell_boltzmann!(System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}, Float64, Xoshiro),
        fcc64(Float64, Int, Symbol),
        fcc32(Float32, Int, Symbol),
        remove_atom(System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}, Int),
        atomic_mass(Symbol),
        atomic_number(Int),
    )
end
