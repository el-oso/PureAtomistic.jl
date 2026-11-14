# The calls whose compiled specializations the StrictMode policy item judges. It runs in a child
# process that loads only the package, so no test fixture adds a specialization of its own.
using PureAtomistic, PureMLIP, Random, LinearAlgebra
using PureAtomistic.Systems, PureAtomistic.Potentials, PureAtomistic.Integrators
using PureAtomistic.Thermostats, PureAtomistic.Barostats, PureAtomistic.Analysis, PureAtomistic.IO

function strict_workload(::Type{T}, dir) where {T}
    model = MACEModel{T}(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint()))

    # Systems
    sys = fcc_cell(T(3.61), 2, :Cu)
    n = natoms(sys)
    sys.positions .+= T(0.05) .* T.(reshape(sin.(1.0:(3n)), 3, n))
    maxwell_boltzmann!(sys, 300.0, Xoshiro(1))
    maxwell_boltzmann!(sys, 300.0, Xoshiro(1); remove_com = false)
    kinetic_energy(sys); temperature(sys); total_momentum(sys); volume(sys)
    wrap!(sys)
    other = fcc_cell(T(3.61), (2, 2, 2), 29)
    copy_state!(other, sys)
    vac = remove_atom(sys, 1)
    atomic_mass(:Cu); atomic_number(29)

    # Potentials
    lj = LennardJones(sys.positions, sys.cell; epsilon = T(0.0104), sigma = T(2.3), r_cut = 5.0)
    hp = HarmonicPotential(copy(sys.positions), 2.0)
    mace = MACEPotential(model, sys.positions, sys.cell, sys.species)
    for p in (lj, hp, mace)
        prepare!(p, sys.positions, sys.cell)
        energy_forces!(p, sys.positions, sys.cell)
        n_atoms(p); cutoff(p)
    end
    for p in (lj, mace)
        pressure(sys, stress(p))
    end
    node_energies(mace)

    # Integrators, thermostats, barostats and the driver
    for p in (lj, mace), th in (nothing, LangevinBAOAB(300.0, 0.01), BussiThermostat(300.0, 50.0)),
            b in (nothing, BerendsenBarostat(0.5, 50.0, 0.3))
        s = fcc_cell(T(3.61), 2, :Cu)
        copy_state!(s, sys)
        sim = Simulation(s, p, VelocityVerlet(1.0); thermostat = th, barostat = b, rng = Xoshiro(0))
        step!(sim)
        run!(sim, 2)
        total_energy(sim); kinetic_energy(sim); temperature(sim)
    end
    for make in (FIRE, FIRE2, TrustRegionBFGS, TrustRegionLBFGS), p in (lj, mace)
        s = fcc_cell(T(3.61), 2, :Cu)
        copy_state!(s, sys)
        relax!(s, p, make(s); fmax = 1.0e-3, maxsteps = 3)
    end
    s = fcc_cell(T(3.61), 2, :Cu)
    copy_state!(s, sys)
    relax!(s, lj; maxsteps = 3)
    band = [copy(sys.positions) for _ in 1:4]
    for k in eachindex(band)
        band[k][1, 2] += T(0.1) * k
    end
    neb(band, sys.cell, lj; maxsteps = 3)
    randgamma(Xoshiro(2), T(2.5))

    # Analysis
    a, nrep = T(3.61), 2
    sites = reference_sites(sys.cell, a, nrep)
    occ = zeros(Int, size(sites, 2))
    wigner_seitz_occupancy!(occ, vac.positions, sites, vac.cell)
    tr = VacancyTracker(sites; cell = vac.cell, window = 1, dwell = 1, capacity = 8)
    vac2 = remove_atom(sys, 2)
    for (t, s) in enumerate((vac, vac, vac2, vac2, vac, vac, vac2, vac2))
        update!(tr, s.positions, T(t))
    end
    hop_rate(tr)
    msd_diffusion(tr; nblocks = 2)
    frames = [vac for _ in 1:8]
    rdf(frames, 2.5, 10)
    h = zeros(Int, 10)
    rdf_histogram!(h, vac.positions, vac.cell, vac.species, T(2.5))
    coordination_numbers(vac, 3.0)
    coordination_numbers!(zeros(Int, natoms(vac)), vac.positions, vac.cell, vac.species, T(3))
    u, prev = copy(vac.positions), copy(vac.positions)
    unwrap!(u, prev, vac.positions, vac.cell)
    lag, m = msd(frames; max_lag = 2, nblocks = 2, dt = 1.0)
    diffusion_coefficient(lag, m; fit_window = (0.5, 2.5))
    rmsd(frames, vac.positions)
    rmsd(frames, vac.positions; align = true)
    rmsd!(zeros(T, 8), frames, vac.positions; align = true)
    reference_sites!(similar(sites), sys.cell, a, nrep)
    rdf!(zeros(T, 10), zeros(T, 10), zeros(T, 10), zeros(Int, 10), frames, 2.5)
    msd!(zeros(T, 3), zeros(T, 3, 2), similar(vac.positions), similar(vac.positions), zeros(T, 3, natoms(vac), 3), frames; dt = 1.0)
    arrhenius_fit([900.0, 1000.0], [1.0, 2.0], [0.1, 0.1])

    # IO
    path = joinpath(dir, "w$(T).traj")
    w = TrajectoryWriter(path, natoms(vac), fill("Cu", natoms(vac)))
    write_frame(w, 0.0, vac.positions; velocities = vac.velocities, cell = vac.cell, energy = 1.0)
    close(w)
    r = TrajectoryReader(path)
    read_frame(r, 1, T)
    close(r)
    f = Frame(fill("Cu", natoms(vac)), vac.positions; cell = vac.cell, energy = 1.0)
    xyz = joinpath(dir, "f$(T).extxyz")
    open(io -> write_extxyz(io, f), xyz, "w")
    open(read_extxyz, xyz)
    return nothing
end

# `run_md` from a config file, with each thermostat, with and without the barostat.
function strict_run_md(dir)
    model = joinpath(dir, "tiny.model")
    write(model, PureMLIP.tiny_mace_checkpoint())
    couplings = ("none" => "", "langevin" => "gamma = 0.01", "bussi" => "tau = 50.0")
    barostat = "[barostat]\ntype = \"berendsen\"\npressure = 0.0\ntau = 1000.0\ncompressibility = 0.01\n"
    for (k, (th, param)) in enumerate(couplings), baro in (false, true)
        cfg = joinpath(dir, "md$(k)$(baro).toml")
        write(
            cfg, """
            steps = 2
            dt = 1.0
            seed = 1
            temperature = 300.0
            vacancy = true
            output_every = 1
            [lattice]
            type = "fcc"
            element = "Cu"
            a = 3.61
            size = [2, 2, 2]
            [thermostat]
            type = "$th"
            $param
            $(baro ? barostat : "")"""
        )
        PureAtomistic.run_md(model, cfg, joinpath(dir, "md$(k)$(baro).traj"))
    end
    return nothing
end

mktempdir() do dir
    for T in (Float64, Float32)
        strict_workload(T, dir)
    end
    strict_run_md(dir)
end
