# The MD step on the device: `DeviceMACEPotential`, the velocity Verlet kernels and the energy
# queries of a `Simulation`, against the CPU driver in the same precision.
@testmodule DeviceMD begin
    using PureAtomistic, PureAtomistic.Systems, PureMLIP, Random, KernelAbstractions

    tiny(::Type{T}) where {T} = MACEModel{T}(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint()))

    # The system of the CPU suite's "NVE on the tiny MACE checkpoint conserves energy": two Cu and
    # one H in a triclinic cell, at 300 K.
    function host_system(::Type{T}) where {T}
        cell = T[6.5 0.0 0.0; 0.4 6.2 0.0; 0.3 0.2 6.8]
        x = T[0.0 1.4 0.2; 0.0 0.1 1.3; 0.0 0.3 0.1]
        sys = System{T}(x, zeros(T, 3, 3), T[63.546, 1.008, 63.546], [29, 1, 29], cell)
        maxwell_boltzmann!(sys, 300.0, Xoshiro(1))
        return sys
    end

    # The CPU simulation of `host_system` with MACEPotential.
    function cpu_simulation(::Type{T}, dt) where {T}
        sys = host_system(T)
        p = MACEPotential(tiny(T), copy(sys.positions), copy(sys.cell), sys.species)
        return Simulation(sys, p, VelocityVerlet(dt); rng = Xoshiro(0))
    end

    # The same simulation with its system and potential on the backend.
    function device_simulation(::Type{T}, dt, backend; extra_capacity_fraction = 0.1) where {T}
        sys = PureAtomistic.to_device(host_system(T), backend)
        p = PureAtomistic.device_potential(tiny(T), sys; skin = 0, extra_capacity_fraction)
        return Simulation(sys, p, VelocityVerlet(dt); rng = Xoshiro(0))
    end

    # What `f()` writes to the standard error stream, including writes through `Core.stderr`:
    # redirecting to a file replaces the process's file descriptor.
    function stderr_of(f)
        return mktemp() do path, io
            redirect_stderr(f, io)
            close(io)
            read(path, String)
        end
    end

    # A host matrix that reports a backend of its own.
    struct OtherBackend <: KernelAbstractions.Backend end
    struct OtherMatrix{T} <: AbstractMatrix{T}
        data::Matrix{T}
    end
    Base.size(A::OtherMatrix) = size(A.data)
    Base.getindex(A::OtherMatrix, i::Int, j::Int) = A.data[i, j]
    KernelAbstractions.get_backend(::OtherMatrix) = OtherBackend()

    # Prints a measurement when PUREMLIP_GPU_VERBOSE=1.
    report(args...) = get(ENV, "PUREMLIP_GPU_VERBOSE", "") == "1" && (println(args...); flush(stdout))

    export tiny, host_system, cpu_simulation, device_simulation, stderr_of, report, OtherMatrix
end

@testitem "md step: the device potential satisfies the potential contract" tags = [:gpu] setup = [GPUBackend, DeviceMD] begin
    using PureAtomistic.Potentials
    TC = PureAtomistic.TypeContracts
    ext = Base.get_extension(PureAtomistic, :PureAtomisticGPUExt)
    for T in precisions
        sim = device_simulation(T, 0.5, backend)
        p = sim.potential
        P = typeof(p)
        @test P <: ext.DeviceMACEPotential{T}
        @test isconcretetype(P) && all(isconcretetype, fieldtypes(P))
        @test TC.check_contract(P).passed
        @test (TC.@verify P).passed
        @test TC.implements(P, AbstractStressPotential)
        @test n_atoms(p) == 3
        @test cutoff(p) == Float64(tiny(T).r_max)
        @test stress(p) === p.result.stress && node_energies(p) === p.result.node_energies
        @test PureAtomistic.Systems.natoms(sim.system) == 3
        # The system's arrays are on the backend, with the host system's values.
        h = host_system(T)
        @test all(A -> GPUBackend.KernelAbstractions.get_backend(A) == backend, (sim.system.velocities, sim.system.masses, sim.system.species, sim.system.cell))
        @test Array(sim.system.masses) == h.masses && Array(sim.system.species) == h.species && Array(sim.system.cell) == h.cell
        @test Array(p.numbers) == Int32.(h.species)
        @test p.n_regrows == 0 && p.readbacks == 1
    end
end

@testitem "md step: device entry points fail fast" tags = [:gpu] setup = [GPUBackend, DeviceMD] begin
    using PureAtomistic.Potentials, PureAtomistic.Systems
    using OffsetArrays
    T = first(precisions)
    m = tiny(T)
    sim = device_simulation(T, 0.5, backend)
    sys, p = sim.system, sim.potential
    X, C = sys.positions, sys.cell
    @test_throws "pureatomistic: the skin must be finite and non-negative, got -1.0" PureAtomistic.device_potential(m, sys; skin = -1.0)
    @test_throws "pureatomistic: extra_capacity_fraction must be finite and non-negative, got NaN" PureAtomistic.device_potential(m, sys; skin = 0, extra_capacity_fraction = NaN)
    # A model of the other precision lands on the core's check.
    other = T === Float32 ? Float64 : Float32
    @test_throws "pureatomistic: the model has element type $other and the system $T; they must match" PureAtomistic.device_potential(tiny(other), sys; skin = 0)
    # An offset array among the arrays of the system.
    mixed = System{T}(X, sys.velocities, sys.masses, sys.species, OffsetArray(Array(C), 0:2, 0:2))
    @test_throws "offset arrays are not supported" PureAtomistic.device_potential(m, mixed; skin = 0)
    # Arrays of another backend: a host copy that reports a foreign backend, on every backend.
    foreign(A) = OtherMatrix(Array(A))
    @test_throws "pureatomistic: the arrays of the system must all live on one backend" PureAtomistic.device_potential(
        m, System{T}(X, sys.velocities, sys.masses, sys.species, foreign(C)); skin = 0
    )
    @test_throws "pureatomistic: positions and cell must live on the backend of the potential" prepare!(p, X, foreign(C))
    @test_throws "pureatomistic: velocities and masses must live on the backend of the potential" PureAtomistic.Integrators.advance!(
        sim.integrator, System{T}(X, foreign(sys.velocities), sys.masses, sys.species, C), p
    )
    @test_throws "pureatomistic: positions and cell must have element type $T, got $other and $other" prepare!(p, other.(Array(X)), other.(Array(C)))
    @test_throws "pureatomistic: positions have size (3, 2), expected (3, 3)" energy_forces!(p, X[:, 1:2], C)
    @test_throws "pureatomistic: the cell has size (2, 2), expected (3, 3)" energy_forces!(p, X, C[1:2, 1:2])
    @test_throws "offset arrays are not supported" prepare!(p, OffsetArray(X, 0:2, 0:2), C)
    small = PureAtomistic.to_device(System{T}(zeros(T, 3, 2), zeros(T, 3, 2), T[1, 1], [29, 29], Array(C)), backend)
    @test_throws "pureatomistic: the system has 2 atoms, the potential 3" PureAtomistic.Integrators.advance!(sim.integrator, small, p)
    @test_throws "pureatomistic: temperature needs at least one degree of freedom" temperature(sim; dof = 0)
    # Two atoms at one place: the evaluation's status read throws the device list's message.
    Xc = Array(X)
    Xc[:, 3] = Xc[:, 1]
    Xd = GPUBackend.todevice(Xc)
    prepare!(p, Xd, C)
    @test_throws "has length zero; the two atoms are at one place" energy_forces!(p, Xd, C)
end

# Oracle: the CPU's `kick!` and `drift!` applied on the host to copies of the state and of the
# forces that the device step used. Rounding bound (b): a kick v + τ/(m c) F rounds four times
# (m c, the quotient, the product, the sum), two kicks eight times, and the drift x + h v twice;
# c = 8 relative to the largest velocity or velocity increment, c = 2 relative to the largest
# position or displacement. Measured: 0 (the same bits) on the CPU backend and on Metal, Float32 and
# Float64.
@testitem "md step: kick and drift equal the CPU velocity Verlet for the same forces" tags = [:gpu] setup = [GPUBackend, DeviceMD] begin
    using PureAtomistic.Systems, PureAtomistic.Integrators
    for T in precisions
        sim = device_simulation(T, 0.5, backend)
        sys, p = sim.system, sim.potential
        for _ in 1:3
            X0, V0, F0, m = Array(sys.positions), Array(sys.velocities), Array(p.result.forces), Array(sys.masses)
            advance!(sim.integrator, sys, p)
            F1 = Array(p.result.forces)
            h = T(sim.integrator.dt)
            X, V = copy(X0), copy(V0)
            PureAtomistic.Systems.kick!(V, F0, m, h / 2)
            dv1 = maximum(abs, V - V0)
            PureAtomistic.Systems.drift!(X, V, h)
            Vh = copy(V)
            PureAtomistic.Systems.kick!(V, F1, m, h / 2)
            vscale = max(maximum(abs, V0), maximum(abs, V), dv1, maximum(abs, V - Vh))
            xscale = max(maximum(abs, X), maximum(abs, X - X0))
            dx, dv = maximum(abs, Array(sys.positions) - X), maximum(abs, Array(sys.velocities) - V)
            report("kick/drift vs CPU, $(GPUBackend.NAME), $T: positions $(dx / (eps(T) * xscale)) eps, velocities $(dv / (eps(T) * vscale)) eps")
            @test dx <= 2 * eps(T) * xscale
            @test dv <= 8 * eps(T) * vscale
        end
    end
end

# Oracle: the CPU simulation with MACEPotential in the same precision. Measured margin: the
# largest position and velocity differences over 10 steps of 0.5 fs, in units of eps(T) times the
# largest magnitude, are 0.00080 and 1.05 in Float32 (CPU backend on Julia 1.12 and 1.13, and
# Metal), and in Float64 on the CPU backend 0.00047 and 1.06 on Julia 1.13 and 0.17 and 1.57 on
# Julia 1.12; the bounds are 20 times the largest. The position differences lie below one unit in
# the last place of the largest coordinate: they are in the coordinates near zero.
@testitem "md step: 10 device steps track the CPU trajectory" tags = [:gpu] setup = [GPUBackend, DeviceMD] begin
    for T in precisions
        cpu = cpu_simulation(T, 0.5)
        dev = device_simulation(T, 0.5, backend)
        worst_x = worst_v = 0.0
        for _ in 1:10
            step!(cpu)
            step!(dev)
            X, V = cpu.system.positions, cpu.system.velocities
            worst_x = max(worst_x, maximum(abs, Array(dev.system.positions) - X) / (eps(T) * maximum(abs, X)))
            worst_v = max(worst_v, maximum(abs, Array(dev.system.velocities) - V) / (eps(T) * maximum(abs, V)))
        end
        report("trajectory vs CPU, $(GPUBackend.NAME), $T: positions $worst_x eps, velocities $worst_v eps")
        @test worst_x <= 3.4
        @test worst_v <= 31.4
        @test dev.step_count == 10
    end
end

# Oracle: the CPU suite's measured margin for the same run ("NVE on the tiny MACE checkpoint
# conserves energy": largest |E - E0| over 50 steps of 0.5 fs at 300 K 1.47e-5 eV in Float64 and
# 1.55e-5 eV in Float32, bound 3e-4 eV). The device run is held to the CPU's bound.
@testitem "md step: NVE energy over 50 device steps stays within the CPU's margin" tags = [:gpu] setup = [GPUBackend, DeviceMD] begin
    for T in precisions
        sim = device_simulation(T, 0.5, backend)
        E0 = Float64(total_energy(sim))
        worst = 0.0
        run!(sim, 50; observer = s -> (worst = max(worst, abs(Float64(total_energy(s)) - E0))))
        report("NVE 50 steps, $(GPUBackend.NAME), $T: largest |E - E0| $worst eV")
        @test total_energy(sim) isa T
        @test 0 < worst <= 3.0e-4
        @test sim.potential.n_regrows == 0
    end
end

# Oracle: the CPU queries on a host copy of the state. The device sums the n per-atom terms in a
# tree and the CPU in order: c = n + 4 (the sum, the three squares and their sum, the factor)
# relative to the kinetic energy, plus |E| for the total energy, plus three roundings for the
# temperature.
@testitem "md step: kinetic energy, total energy and temperature equal the CPU's" tags = [:gpu] setup = [GPUBackend, DeviceMD] begin
    using PureAtomistic.Systems
    for T in precisions
        sim = device_simulation(T, 0.5, backend)
        run!(sim, 3)
        h = System{T}(Array(sim.system.positions), Array(sim.system.velocities), Array(sim.system.masses), Array(sim.system.species), Array(sim.system.cell))
        K = kinetic_energy(h)
        E = Array(sim.potential.result.energy)[1]
        c = natoms(h) + 4
        @test kinetic_energy(sim) isa T && total_energy(sim) isa T && temperature(sim) isa T
        @test abs(kinetic_energy(sim) - K) <= c * eps(T) * K
        @test abs(total_energy(sim) - (K + E)) <= (c + 1) * eps(T) * (K + abs(E))
        @test abs(temperature(sim) - temperature(h)) <= (c + 3) * eps(T) * temperature(h)
        @test abs(temperature(sim; dof = 4) - temperature(h; dof = 4)) <= (c + 3) * eps(T) * temperature(h; dof = 4)
    end
end

@testitem "md step: each step reads back the status word once and nothing else" tags = [:gpu] setup = [GPUBackend, DeviceMD] begin
    using PureAtomistic.Systems
    for T in precisions
        sim = device_simulation(T, 0.5, backend)
        p = sim.potential
        # Construction evaluates once.
        @test p.readbacks == 1
        for k in 1:5
            step!(sim)
            @test p.readbacks == 1 + k
        end
        run!(sim, 7)
        @test p.readbacks == 13
        # Each energy query reads one scalar.
        kinetic_energy(sim)
        total_energy(sim)
        temperature(sim)
        @test p.readbacks == 16
    end
    # `readbacks` covers every copy from device to host because the extension has one such copy,
    # in `_read_status!`. The remaining copies of its source go from host to device (the atomic
    # numbers and `to_device`), and the only host copy of a device array it takes is the atomic
    # numbers read once by `device_potential`.
    dir = joinpath(pkgdir(PureAtomistic), "ext", "PureAtomisticGPUExt")
    lines = [strip(l) for f in readdir(dir; join = true) for l in eachline(f)]
    @test filter(l -> occursin("copyto!", l), lines) == ["copyto!(numbers, Int32.(species))", "copyto!(H, A)", "copyto!(D, H)", "copyto!(host, device)"]
    @test filter(l -> occursin(r"\bArray\(", l), lines) == ["species = Array(sys.species)"]
end

# Host bytes allocated by one `step!`, through a compiled function so that the dynamic dispatch
# of the call site is not counted: the fewest of five steps after two warm-up steps, on the tiny
# checkpoint's three atoms. The bytes are the backend's objects for each kernel launch and matrix
# product of the evaluation and the list rebuild, and the read of the status word. Measured:
#   CPU backend, Julia 1.12 and 1.13, Float32 and Float64: 2,464 bytes on every step;
#   Metal, Julia 1.13, Float32: 456,736 to 456,928 as the fewest of five steps over 24 groups of
#   five in three processes; single steps reached 457,344.
# The gate is the measured figure on the CPU backend on macOS, where it was measured, and twice it
# on other platforms; on Metal the largest fewest-of-five rounded up to 460,000 (0.7 % above it).
@testitem "md step: host bytes per step" tags = [:gpu] setup = [GPUBackend, DeviceMD] begin
    @noinline step_bytes(sim) = @allocated step!(sim)
    cpu = Sys.isapple() ? 1 : 2
    bounds = Dict("CPU" => cpu * 2_464, "Metal" => 460_000)
    for T in precisions
        sim = device_simulation(T, 0.5, backend)
        step_bytes(sim)
        step_bytes(sim)
        b = minimum(step_bytes(sim) for _ in 1:5)
        report("host bytes per step!, $(GPUBackend.NAME), $T, Julia $VERSION: $b")
        if haskey(bounds, GPUBackend.NAME)
            @test b <= bounds[GPUBackend.NAME]
        else
            # CUDA and AMDGPU are measured on their machines.
            @test_skip b <= 0
        end
    end
end

# The device memory in use does not grow from one step to the next. The vendor's query is written
# for Metal; on the CPU backend there is no device memory apart from the host's, and the CUDA and
# AMDGPU queries are written on their machines. Over 500 fs the three atoms' edge count drifts from
# 14 to as many as 26, which outgrows 10 % of spare room; the list gets room for twice its edges
# more, so that no step regrows it.
@testitem "md step: device memory does not grow over 1,000 steps" tags = [:gpu] setup = [GPUBackend, DeviceMD] begin
    if GPUBackend.NAME != "Metal"
        @test_skip GPUBackend.NAME == "Metal"
    else
        device_bytes() = Int(GPUBackend.vendor.device().currentAllocatedSize)
        sim = device_simulation(Float32, 0.5, backend; extra_capacity_fraction = 2.0)
        run!(sim, 10)
        GPUBackend.KernelAbstractions.synchronize(backend)
        # The arrays that earlier items left for the garbage collector are freed first, so that
        # the first reading counts only what is live.
        GC.gc(true)
        after10 = device_bytes()
        run!(sim, 990)
        GPUBackend.KernelAbstractions.synchronize(backend)
        @test device_bytes() == after10
        @test sim.potential.n_regrows == 0 && sim.step_count == 1000
    end
end

# Review Focus 2: a step whose configuration holds more edges than the list has room for. The
# after-drift hook compresses the cell and the positions by 0.85 (the edge count rises from 14 to
# 50 with room for 16); the evaluation's status read finds the overflow, the potential rebuilds
# its list and workspace once and evaluates again. Oracle: a fresh device list, workspace and
# result at the same configuration, evaluated by `mace_energy_forces!`; the per-edge stages sum per
# receiver over its own edges, so the results do not depend on the capacity and equal bit for bit.
@testitem "md step: an edge overflow mid-run regrows once and gives a fresh evaluation's forces" tags = [:gpu] setup = [GPUBackend, DeviceMD] begin
    using PureMLIP, PureAtomistic.Integrators
    for T in precisions
        sim = device_simulation(T, 0.5, backend)
        sys, p = sim.system, sim.potential
        run!(sim, 2)
        capacity = p.nl.capacity
        compress!(s) = (s.cell .*= T(0.85); s.positions .*= T(0.85); nothing)
        r = p.readbacks
        log = stderr_of(() -> advance!(sim.integrator, sys, p, compress!))
        @test occursin(r"^pureatomistic: DeviceMACEPotential rebuilt its neighbor list and buffers with capacity \d+ edges\n$", log)
        @test p.n_regrows == 1
        # The overflowed evaluation and the repeated one each read the status word.
        @test p.readbacks == r + 2
        m = Array(p.nl.n_edges)[1]
        @test m > capacity
        @test p.nl.capacity == m + ceil(Int, 0.1 * m)
        report("regrow, $(GPUBackend.NAME), $T: capacity $capacity -> $(p.nl.capacity), $m edges")
        dnl = PureMLIP.device_neighbor_list(sys.positions, sys.cell, tiny(T).r_max)
        ws = PureMLIP.device_workspace(p.model, 3, dnl.capacity)
        res = PureMLIP.device_result(p.model, 3)
        mace_energy_forces!(res, ws, p.model, sys.positions, sys.cell, p.numbers, dnl)
        @test Array(res.forces) == Array(p.result.forces)
        @test Array(res.energy) == Array(p.result.energy)
        @test Array(res.stress) == Array(p.result.stress)
        # Later steps keep the grown list.
        r = p.readbacks
        run!(sim, 5)
        @test p.n_regrows == 1 && p.readbacks == r + 5
    end
end

# The kernels' arithmetic as host code in Float32 (the functions the kernels call on each entry),
# and the extension's compiled functions after Float32 steps and energy queries. `cutoff` returns
# the Float32 cutoff as the Float64 that the potential contract declares.
@testitem "md step: Float32 steps do not widen to Float64" tags = [:gpu] setup = [GPUBackend, DeviceMD] begin
    using PureAtomistic.Systems
    using StrictMode
    ext = Base.get_extension(PureAtomistic, :PureAtomisticGPUExt)
    T = Float32
    sim = device_simulation(T, 0.5, backend)
    run!(sim, 2)
    total_energy(sim)
    temperature(sim)
    V, m = rand(T, 3, 2), rand(T, 2) .+ 1
    @test ext._kicked(T(1), T(2), T(3), T(0.25), T(103)) isa T
    @test ext._drifted(T(1), T(2), T(0.5)) isa T
    @test ext._kinetic_term(V, m, 1) isa T
    @test ext._energy_value(T(1), T(2), T(51), true) isa T
    @test isempty(StrictMode.conversion_report(ext; from = Float32, to = Float64, exempt = (:cutoff,)))
end
