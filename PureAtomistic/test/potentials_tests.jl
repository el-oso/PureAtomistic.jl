@testmodule PotentialFixtures begin
    using PureMLIP
    using PureAtomistic.Potentials

    # Three atoms of the tiny checkpoint's elements in a triclinic cell, and a configuration
    # with the atoms spread apart, which brings more periodic images inside the 6 Å cutoff.
    const CELL = [6.5 0.0 0.0; 0.4 6.2 0.0; 0.3 0.2 6.8]
    const P = [0.0 1.4 0.2; 0.0 0.1 1.3; 0.0 0.3 0.1]
    const P_SPREAD = [0.0 3.1 0.2; 0.0 3.0 3.3; 0.0 0.3 3.0]
    const Z = [29, 1, 29]

    tiny_model(::Type{T}) where {T} = MACEModel{T}(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint()))

    # Eight atoms near the sites of a simple cubic lattice of spacing 4 Å in an 8 Å cube.
    const LJ_CELL = [8.0 0.0 0.0; 0.0 8.0 0.0; 0.0 0.0 8.0]
    const LJ_P = [
        0.1 4.0 0.2 4.1 0.0 3.9 0.3 4.2;
        0.0 0.2 3.9 4.1 -0.2 0.1 4.0 3.8;
        0.2 -0.1 0.1 0.0 4.1 3.9 4.2 4.0
    ]
    # Test parameters of argon magnitude (eV, Å) and a cutoff of 2.5σ.
    lennard_jones(P, cell; kw...) = LennardJones(P, cell; epsilon = 0.0104, sigma = 3.4, r_cut = 8.5, kw...)

    energy(p, x, cell) = (prepare!(p, x, cell); energy_forces!(p, x, cell); p.result.energy)

    # What `f()` writes to the standard error stream, including writes through `Core.stderr`:
    # redirecting to a file replaces the process's file descriptor.
    function stderr_of(f)
        return mktemp() do path, io
            redirect_stderr(f, io)
            close(io)
            read(path, String)
        end
    end

    # Largest difference between the forces of `p` at `x` and minus the central difference
    # (step h) of its energy along every coordinate.
    function fd_force_error(p, x, cell, h)
        energy(p, x, cell)
        F = copy(p.result.forces)
        worst = zero(eltype(x))
        for i in axes(x, 2), a in 1:3
            xp = copy(x); xp[a, i] += h
            xm = copy(x); xm[a, i] -= h
            fd = -(energy(p, xp, cell) - energy(p, xm, cell)) / 2h
            worst = max(worst, abs(fd - F[a, i]))
        end
        return worst
    end

    # A rotation about the axis (1, 2, 3) by 0.7 rad.
    function rotation()
        u = [1.0, 2.0, 3.0] / sqrt(14.0)
        θ = 0.7
        K = [0 -u[3] u[2]; u[3] 0 -u[1]; -u[2] u[1] 0]
        return [1.0 0 0; 0 1 0; 0 0 1] + sin(θ) * K + (1 - cos(θ)) * K * K
    end
end

@testitem "potentials satisfy the potential contract" setup = [PotentialFixtures] begin
    using TypeContracts
    using PureMLIP
    F = PotentialFixtures
    for T in (Float64, Float32)
        types = (
            typeof(HarmonicPotential(T.(F.P), 2.0)),
            typeof(F.lennard_jones(T.(F.LJ_P), T.(F.LJ_CELL))),
            typeof(MACEPotential(F.tiny_model(T), T.(F.P), T.(F.CELL), F.Z)),
        )
        for P in types
            @test isconcretetype(P)
            @test check_contract(P).passed
            @test check_trim_compat(P).passed
            @test implements(P, AbstractPotential)
        end
        @test implements(types[3], AbstractStressPotential)
    end
end

@testitem "MACEPotential equals mace_energy_forces exactly" setup = [PotentialFixtures] begin
    using PureMLIP
    F = PotentialFixtures
    for T in (Float64, Float32)
        m = F.tiny_model(T)
        x, cell = T.(F.P), T.(F.CELL)
        p = MACEPotential(m, x, cell, F.Z)
        @test n_atoms(p) == 3 && cutoff(p) == 6.0
        # The default room is 10 % more edges than the configuration has, rounded up.
        @test length(p.nl.senders) == p.nl.n_edges + ceil(Int, 0.1 * p.nl.n_edges)
        @test prepare!(p, x, cell) === nothing
        @test energy_forces!(p, x, cell) === nothing
        ref = mace_energy_forces(m, x, cell, F.Z)
        # The same neighbor list order and the same arithmetic: equality is exact.
        @test p.result.energy == ref.energy
        @test p.result.forces == ref.forces
        @test stress(p) == ref.stress
        @test node_energies(p) == ref.node_energies
        @test p.n_regrows == 0
    end
end

@testitem "potential forces equal minus the finite-difference gradient" setup = [PotentialFixtures] begin
    using PureMLIP
    F = PotentialFixtures
    h = 1.0e-5
    # Central differences of step 1e-5 Å on every coordinate, in Float64. Largest differences
    # observed: harmonic 6.2e-12 eV/Å, Lennard-Jones 1.3e-11, MACE 1.4e-9; each tolerance
    # is about 15 times its measurement.
    hp = HarmonicPotential(F.P .+ 0.3, 2.0)
    @test F.fd_force_error(hp, F.P, F.CELL, h) < 1.0e-10
    lj = F.lennard_jones(F.LJ_P, F.LJ_CELL)
    @test F.fd_force_error(lj, F.LJ_P, F.LJ_CELL, h) < 2.0e-10
    mp = MACEPotential(F.tiny_model(Float64), F.P, F.CELL, F.Z)
    @test F.fd_force_error(mp, F.P, F.CELL, h) < 2.0e-8
end

@testitem "harmonic springs take the nearest periodic image of the site" setup = [PotentialFixtures] begin
    F = PotentialFixtures
    sites = copy(F.P)
    p = HarmonicPotential(sites, 2.0)
    x = copy(F.P)
    x[:, 2] .+= [0.1, -0.2, 0.05]
    E = F.energy(p, x, F.CELL)
    # 2/2 · |(0.1, -0.2, 0.05)|² = 0.0525 eV; the forces are exact up to one rounding.
    @test E ≈ 0.0525 rtol = 4eps()
    @test p.result.forces[:, 2] ≈ -2 .* [0.1, -0.2, 0.05] rtol = 4eps()
    @test iszero(p.result.forces[:, 1]) && iszero(p.result.forces[:, 3])
    # Moving the atom by a lattice vector changes nothing: it is the nearest image that counts.
    y = copy(x)
    y[:, 2] .+= F.CELL[:, 1] .- F.CELL[:, 3]
    # The lattice translation adds and removes about 7 Å from each coordinate: a few eps of that.
    @test F.energy(p, y, F.CELL) ≈ E atol = 64eps()
    @test cutoff(p) == 0.0 && n_atoms(p) == 3
end

@testitem "Lennard-Jones energy equals the pair sum of the cut and shifted potential" setup = [PotentialFixtures] begin
    F = PotentialFixtures
    # Two atoms 3.8 Å apart along x in a 20 Å cube: one pair, no periodic image within 8.5 Å.
    cell = [20.0 0.0 0.0; 0.0 20.0 0.0; 0.0 0.0 20.0]
    x = [0.0 3.8; 0.0 0.0; 0.0 0.0]
    p = F.lennard_jones(x, cell)
    V(r) = 4 * 0.0104 * ((3.4 / r)^12 - (3.4 / r)^6)
    E = F.energy(p, x, cell)
    # The same formula evaluated directly: a few roundings of a value of order 0.01 eV.
    @test E ≈ V(3.8) - V(8.5) rtol = 16eps()
    dV = -24 * 0.0104 * (2 * (3.4 / 3.8)^12 - (3.4 / 3.8)^6) / 3.8
    @test p.result.forces[1, 2] ≈ -dV rtol = 16eps()
    @test p.result.forces[1, 1] ≈ dV rtol = 16eps()
    @test cutoff(p) == 8.5 && n_atoms(p) == 2
end

@testitem "potential energies are rotation and translation invariant" setup = [PotentialFixtures] begin
    using PureMLIP
    F = PotentialFixtures
    R = F.rotation()
    t = [0.37, -1.21, 2.05]
    # Relative changes of the energy and of the largest force component under the rotation and the
    # translation, Float64: the largest observed is 2.3e-15 (Lennard-Jones forces under rotation),
    # and the tolerance is 13 times that.
    tol = 3.0e-14
    cases = (
        (cell -> (x -> F.lennard_jones(x, cell)), F.LJ_P, F.LJ_CELL),
        (cell -> (x -> MACEPotential(F.tiny_model(Float64), x, cell, F.Z)), F.P, F.CELL),
    )
    for (make, x, cell) in cases
        p = make(cell)(x)
        E = F.energy(p, x, cell)
        Fx = copy(p.result.forces)
        xr, cr = R * x, R * cell
        pr = make(cr)(xr)
        Er = F.energy(pr, xr, cr)
        @test abs(Er - E) <= tol * abs(E)
        @test maximum(abs, pr.result.forces - R * Fx) <= tol * maximum(abs, Fx)
        xt = x .+ t
        pt = make(cell)(xt)
        Et = F.energy(pt, xt, cell)
        @test abs(Et - E) <= tol * abs(E)
        @test maximum(abs, pt.result.forces - Fx) <= tol * maximum(abs, Fx)
    end
end

@testitem "potentials regrow the neighbor list when a configuration outgrows it" setup = [PotentialFixtures] begin
    using PureMLIP
    F = PotentialFixtures
    m = F.tiny_model(Float64)
    p = MACEPotential(m, F.P, F.CELL, F.Z; extra_capacity_fraction = 0.0)
    capacity = length(p.nl.senders)
    @test occursin(r"^pureatomistic: MACEPotential rebuilt its neighbor list and buffers with capacity \d+ edges\n$", F.stderr_of(() -> prepare!(p, F.P_SPREAD, F.CELL)))
    @test p.n_regrows == 1
    @test p.nl.n_edges > capacity
    @test length(p.nl.senders) == p.nl.n_edges
    energy_forces!(p, F.P_SPREAD, F.CELL)
    ref = mace_energy_forces(m, F.P_SPREAD, F.CELL, F.Z)
    @test p.result.energy == ref.energy
    @test p.result.forces == ref.forces
    @test p.result.stress == ref.stress
    # The rebuilt list has room for the new edge count; a second step at the same
    # configuration and one back at the first fit without another rebuild.
    @test isempty(F.stderr_of(() -> prepare!(p, F.P_SPREAD, F.CELL)))
    @test isempty(F.stderr_of(() -> prepare!(p, F.P, F.CELL)))
    @test p.n_regrows == 1

    lj = F.lennard_jones(F.LJ_P, F.LJ_CELL; extra_capacity_fraction = 0.0)
    capacity = length(lj.nl.senders)
    # Shrinking the cell by 10 % brings more pairs within the cutoff.
    x2, c2 = 0.9 .* F.LJ_P, 0.9 .* F.LJ_CELL
    @test startswith(F.stderr_of(() -> prepare!(lj, x2, c2)), "pureatomistic: LennardJones rebuilt its neighbor list")
    @test lj.n_regrows == 1 && lj.nl.n_edges > capacity
    energy_forces!(lj, x2, c2)
    fresh = F.lennard_jones(x2, c2)
    @test lj.result.energy == F.energy(fresh, x2, c2)
    @test lj.result.forces == fresh.result.forces
end

@testitem "potential errors" setup = [PotentialFixtures] begin
    using PureMLIP
    using OffsetArrays
    F = PotentialFixtures
    m = F.tiny_model(Float64)
    p = MACEPotential(m, F.P, F.CELL, F.Z)
    @test_throws "pureatomistic: positions and cell must have element type Float64, got Float32 and Float64" prepare!(p, Float32.(F.P), F.CELL)
    @test_throws "pureatomistic: positions and cell must have element type Float64, got Float64 and Float32" energy_forces!(p, F.P, Float32.(F.CELL))
    @test_throws "pureatomistic: positions have size (3, 2), expected (3, 3)" prepare!(p, F.P[:, 1:2], F.CELL)
    @test_throws "pureatomistic: the cell has size (3, 2), expected (3, 3)" prepare!(p, F.P, F.CELL[:, 1:2])
    @test_throws "pureatomistic: 2 atomic numbers for 3 atoms" MACEPotential(m, F.P, F.CELL, [29, 1])
    @test_throws "pureatomistic: extra_capacity_fraction must be finite and non-negative, got -0.1" MACEPotential(m, F.P, F.CELL, F.Z; extra_capacity_fraction = -0.1)
    @test_throws "pureatomistic: extra_capacity_fraction must be finite and non-negative, got Inf" F.lennard_jones(F.LJ_P, F.LJ_CELL; extra_capacity_fraction = Inf)
    @test_throws "pureatomistic: epsilon and sigma must be positive and finite, got -1.0 and 3.4" LennardJones(F.LJ_P, F.LJ_CELL; epsilon = -1.0, sigma = 3.4, r_cut = 8.5)
    @test_throws "pureatomistic: r_cut must be positive and finite, got 0.0" LennardJones(F.LJ_P, F.LJ_CELL; epsilon = 1.0, sigma = 3.4, r_cut = 0.0)
    @test_throws "pureatomistic: the spring constant must be positive and finite, got 0.0" HarmonicPotential(F.P, 0.0)
    @test_throws "pureatomistic: sites have size (2, 3), expected 3 rows" HarmonicPotential(F.P[1:2, :], 1.0)
    @test_throws "pureatomistic: the number of atoms must not be negative, got -1" PotentialResult{Float64}(-1)
    hp = HarmonicPotential(F.P, 1.0)
    @test_throws "pureatomistic: the cell is singular" energy_forces!(hp, F.P, zeros(3, 3))
    # An invalid configuration is reported as such, not taken for a full neighbor list.
    bad = copy(F.P)
    bad[:, 2] .= bad[:, 1]
    @test_throws "neighbor list: atoms 1 and 2 coincide" prepare!(p, bad, F.CELL)
    @test p.n_regrows == 0
    # The potentials are written for 1-based arrays and say so.
    @test_throws ArgumentError prepare!(p, OffsetArray(F.P, 0:2, 1:3), F.CELL)
    @test_throws ArgumentError energy_forces!(hp, F.P, OffsetArray(F.CELL, 0:2, 0:2))
end

@testitem "potentials accept views and give the same result" setup = [PotentialFixtures] begin
    using PureMLIP
    F = PotentialFixtures
    m = F.tiny_model(Float64)
    big = zeros(4, 5)
    big[1:3, 2:4] .= F.P
    xv = view(big, 1:3, 2:4)
    cbig = zeros(3, 4)
    cbig[:, 2:4] .= F.CELL
    cv = view(cbig, :, 2:4)
    for make in (
            (x, c) -> MACEPotential(m, x, c, F.Z),
            (x, c) -> HarmonicPotential(F.P .+ 0.2, 1.5),
        )
        a = make(F.P, F.CELL)
        b = make(xv, cv)
        # The same values through a view: the same arithmetic, so equality is exact.
        @test F.energy(a, F.P, F.CELL) == F.energy(b, xv, cv)
        @test a.result.forces == b.result.forces
    end
end

@testitem "prepare! and energy_forces! do not allocate" tags = [:checks] setup = [PotentialFixtures] begin
    using StrictModeTest, StrictMode
    using PureMLIP
    F = PotentialFixtures
    # `_regrow!` allocates a larger neighbor list and workspace when a configuration outgrows them,
    # and the capacity then covers that configuration. An all-paths proof counts that path on every
    # call, so it is registered as an allocation barrier; `energy_forces!` alone is still proved by
    # AllocCheck, and the measurement after warm-up, with the regrow counter unchanged, shows that
    # the warmed step allocates nothing.
    register_alloc_barrier!(PureAtomistic.Potentials._regrow!)
    @noinline step!(p, x, c) = (prepare!(p, x, c); energy_forces!(p, x, c))
    @noinline evaluate!(p, x, c) = energy_forces!(p, x, c)
    for T in (Float64, Float32)
        x, c = T.(F.P), T.(F.CELL)
        xl, cl = T.(F.LJ_P), T.(F.LJ_CELL)
        for (p, y, d) in (
                (MACEPotential(F.tiny_model(T), x, c, F.Z), x, c),
                (HarmonicPotential(x .+ T(0.2), 1.5), x, c),
                (F.lennard_jones(xl, cl), xl, cl),
            )
            step!(p, y, d)
            @test_noalloc evaluate!(p, y, d)
            @test_noalloc step!(p, y, d)
            @test_steady_state step!(p, y, d)
            @test p.result.energy isa T
            p isa HarmonicPotential || @test p.n_regrows == 0
        end
    end
end

@testitem "TrimCheck: potentials" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureAtomistic, PureMLIP
            const PP = PureAtomistic.Potentials
            new_mace64(m, x, c, z) = PP.MACEPotential{Float64}(m, x, c, z)
            new_mace32(m, x, c, z) = PP.MACEPotential{Float32}(m, x, c, z)
            new_lj64(x, c) = PP.LennardJones{Float64}(x, c; epsilon = 1.0, sigma = 1.0, r_cut = 2.5)
            new_lj32(x, c) = PP.LennardJones{Float32}(x, c; epsilon = 1.0, sigma = 1.0, r_cut = 2.5)
            new_harmonic64(s, k) = PP.HarmonicPotential{Float64}(s, k)
            new_harmonic32(s, k) = PP.HarmonicPotential{Float32}(s, k)
        end,
        new_mace64(PureMLIP.MACEModel{Float64}, Matrix{Float64}, Matrix{Float64}, Vector{Int}),
        new_mace32(PureMLIP.MACEModel{Float32}, Matrix{Float32}, Matrix{Float32}, Vector{Int}),
        new_lj64(Matrix{Float64}, Matrix{Float64}),
        new_lj32(Matrix{Float32}, Matrix{Float32}),
        new_harmonic64(Matrix{Float64}, Float64),
        new_harmonic32(Matrix{Float32}, Float64),
        PP.prepare!(PP.MACEPotential{Float64}, Matrix{Float64}, Matrix{Float64}),
        PP.energy_forces!(PP.MACEPotential{Float64}, Matrix{Float64}, Matrix{Float64}),
        PP.prepare!(PP.MACEPotential{Float32}, Matrix{Float32}, Matrix{Float32}),
        PP.energy_forces!(PP.MACEPotential{Float32}, Matrix{Float32}, Matrix{Float32}),
        PP.prepare!(PP.LennardJones{Float64}, Matrix{Float64}, Matrix{Float64}),
        PP.energy_forces!(PP.LennardJones{Float64}, Matrix{Float64}, Matrix{Float64}),
        PP.prepare!(PP.LennardJones{Float32}, Matrix{Float32}, Matrix{Float32}),
        PP.energy_forces!(PP.LennardJones{Float32}, Matrix{Float32}, Matrix{Float32}),
        PP.prepare!(PP.HarmonicPotential{Float64}, Matrix{Float64}, Matrix{Float64}),
        PP.energy_forces!(PP.HarmonicPotential{Float64}, Matrix{Float64}, Matrix{Float64}),
        PP.prepare!(PP.HarmonicPotential{Float32}, Matrix{Float32}, Matrix{Float32}),
        PP.energy_forces!(PP.HarmonicPotential{Float32}, Matrix{Float32}, Matrix{Float32}),
        PP.n_atoms(PP.MACEPotential{Float64}),
        PP.cutoff(PP.MACEPotential{Float64}),
        PP.stress(PP.MACEPotential{Float64}),
        PP.node_energies(PP.MACEPotential{Float64}),
        PP.stress(PP.MACEPotential{Float32}),
        PP.node_energies(PP.MACEPotential{Float32}),
        PP.n_atoms(PP.LennardJones{Float32}),
        PP.cutoff(PP.LennardJones{Float32}),
        PP.n_atoms(PP.HarmonicPotential{Float32}),
        PP.cutoff(PP.HarmonicPotential{Float32}),
    )
end
