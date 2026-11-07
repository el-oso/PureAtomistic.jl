# The device backward pass, `mace_energy_forces!` with device types: against CPU
# `mace_energy_forces!` on the five LAMMPS reference structures with MACE-MPA-0, against central
# differences of the device energy, on systems without edges, and its host allocations.

# Bound (b) of the tolerance policy: 10·eps(T)·N relative to the largest magnitude, separately for
# the energy, the node energies, ∂E/∂r of the edges, the forces, the virial and the stress, with
# N = 1280 the largest inner dimension of any matrix product in the model. The device path sums in
# other orders than the CPU path (its matrix products, the matrix form of both product blocks and
# their pullbacks, the message pullback reduced over channels per edge, the forces gathered per
# atom, the virial as a tree sum of per-atom parts), and a backend's exp, tanh, sincos and sqrt may
# round differently from Base's. The list has a skin and spare capacity; edge chunks of 100 slots
# split the 32-atom cells so that a receiver's edges span chunks, and contraction blocks of 8 atoms
# split them into several blocks.
# The force on an atom is a sum of the ∂E/∂r of its edges, and in a nearly force-free cell they
# cancel: in triclinic_ni3al the largest force is 0.002 eV/Å while the sums of the terms' magnitudes
# reach 3.1 eV/Å. Each edge's ∂E/∂r carries its rounding (checked against the largest |∂E/∂r|
# below), and the force carries the sum of it, so each force component of each atom is compared
# relative to the sum of term magnitudes, Σ_e |∂E/∂r_e| over the edges that atom sends and receives
# (from the CPU's ∂E/∂r). Against the Float64 CPU forces, the Float32 forces of that cell are off by
# 4.0e-6 eV/Å on the CPU and 4.4e-6 on the device, 11 and 12 eps(Float32) of that sum; relative to
# the largest force, the device and CPU Float32 results differ by 34,800 eps(Float32).
# Measured, largest over the five structures and over Julia 1.12.7 and 1.13.3, in eps(T):
#   CPU backend, Float32: energy 0.85, node energies 4.9, ∂E/∂r 321, forces 144, virial 84, stress 85;
#   CPU backend, Float64: energy 1.9, node energies 5.8, ∂E/∂r 326, forces 388, virial 150, stress 150;
#   Metal, Float32: energy 0.95, node energies 5.8, ∂E/∂r 321, forces 114, virial 152, stress 152.
@testitem "backward: forces, virial and stress agree with CPU mace_energy_forces! on the reference structures" tags = [:gpu] setup = [GPUBackend, DeviceStructures, EdgeStages] begin
    ck = mpa0()
    for T in precisions
        m = MACEModel{T}(ck)
        tol = 10 * 1280   # in units of eps(T)
        worst = Dict{String, Float64}()
        note!(name, err) = (worst[name] = max(get(worst, name, 0.0), err); err)
        for s in structures(T)
            P, C, Z = s.positions, s.cell, s.atomic_numbers
            n = size(P, 2)
            nl = neighbor_list(P, C, Float64(m.r_max) + 0.5; extra_capacity = 200 * n)
            E = nl.n_edges
            res = PureMLIP.MACEResult{T}(n)
            cws = MACEWorkspace{T}(m, n, E; elements = unique(Z))
            mace_energy_forces!(res, cws, m, P, C, Z, nl)
            dm = PureMLIP.device_model(m, backend; elements = unique(Z))
            dnl = PureMLIP.device_neighbor_list(nl, backend)
            ws = PureMLIP.device_workspace(dm, n, dnl.capacity; edge_chunk = 100, contraction_block = 8)
            dres = PureMLIP.device_result(dm, n)
            dZ = s.name == "multi_element" ? todevice(Z) : todevice(Int32.(Z))
            @test mace_energy_forces!(dres, ws, dm, todevice(P), todevice(C), dZ, dnl) === dres
            g = cws.dEdr[:, 1:E]
            scale = zeros(Float64, 3, n)
            for e in 1:E, a in 1:3
                scale[a, nl.senders[e]] += abs(g[a, e])
                scale[a, nl.receivers[e]] += abs(g[a, e])
            end
            F = Array(dres.forces)
            Δ = abs.(F .- res.forces)
            ferr = maximum(ifelse.(iszero.(scale), ifelse.(iszero.(Δ), 0.0, Inf), Δ ./ scale)) / eps(T)
            @test note!("energy", Float64(abs(Array(dres.energy)[1] - res.energy) / abs(res.energy) / eps(T))) <= tol
            @test note!("node energies", relerr(Array(dres.node_energies), res.node_energies)) <= tol
            @test note!("dE/dr", relerr(Array(ws.dEdr)[:, 1:E], g)) <= tol
            @test note!("forces", ferr) <= tol
            @test note!("virial", relerr(Array(dres.virial), res.virial)) <= tol
            @test note!("stress", relerr(Array(dres.stress), res.stress)) <= tol
            # triclinic_ni3al is the one structure whose forces are (nearly) zero by symmetry; there the literal
            # bound relative to the largest force is not a meaningful test (recorded below).
            lit = note!("forces relative to the largest force", relerr(F, res.forces))
            s.name == "triclinic_ni3al" || @test lit <= tol
            @test Array(dnl.status) == zeros(Int32, 5)
        end
        report("backward, $(GPUBackend.NAME), $T: largest error in eps($T): ", sort(collect(worst)))
    end
end

# Oracle: central differences of the device energy, -(E(x + h e) - E(x - h e)) / (2h') for each
# coordinate of cu_bench(2) (32 Cu atoms, displaced from fcc), with h' the displacement as stored
# in T. The list holds the edges within r_max + 0.5 Å, so it stays valid for every displaced
# configuration. The error is rounding of the energy over 2h and the h² term of the difference:
# h = 1e-2 Å in Float32, 1e-4 Å in Float64. Measured margins (largest |F - F_fd|, eV/Å, over
# 1.4 eV/Å the largest force): Float32 1.34e-3 (CPU backend, Julia 1.12 and 1.13) and 1.38e-3
# (Metal); Float64 7.9e-9 (CPU backend). The tolerances are 18 and 15 times the largest.
@testitem "backward: forces agree with central differences of the device energy" tags = [:gpu] setup = [GPUBackend, DeviceStructures, EdgeStages] begin
    ck = mpa0()
    cu = DeviceStructures.cu_bench(2)
    for T in precisions
        m = MACEModel{T}(ck)
        P, C = T.(permutedims(cu.positions)), T.(permutedims(cu.cell))
        n = size(P, 2)
        Z = Int32.(cu.atomic_numbers)
        h, tol = T === Float32 ? (1.0f-2, 2.5e-2) : (1.0e-4, 1.2e-7)
        nl = neighbor_list(P, C, Float64(m.r_max) + 0.5)
        dm = PureMLIP.device_model(m, backend; elements = [29])
        dnl = PureMLIP.device_neighbor_list(nl, backend)
        ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
        dres = PureMLIP.device_result(dm, n)
        dC, dZ = todevice(C), todevice(Z)
        mace_energy_forces!(dres, ws, dm, todevice(P), dC, dZ, dnl)
        F = Array(dres.forces)
        @test maximum(abs, F) > 0.1
        energy(Q) = (mace_energy!(dres, ws, dm, todevice(Q), dC, dZ, dnl); Float64(Array(dres.energy)[1]))
        Ffd = zeros(n * 3)
        for q in eachindex(P)
            Qp, Qm = copy(P), copy(P)
            Qp[q] += h
            Qm[q] -= h
            Ffd[q] = -(energy(Qp) - energy(Qm)) / (Float64(Qp[q]) - Float64(Qm[q]))
        end
        err = maximum(abs, vec(F) - Ffd)
        report("backward, $(GPUBackend.NAME), $T: largest |F - F_fd| = ", err)
        @test err <= tol
    end
end

@testitem "backward: evaluations repeat bit for bit, and views of device positions give the same result" tags = [:gpu] setup = [GPUBackend, DeviceStructures, EdgeStages] begin
    ck = mpa0()
    for T in precisions
        m = MACEModel{T}(ck)
        s = structures(T)[2]
        P, C, Z = s.positions, s.cell, s.atomic_numbers
        n = size(P, 2)
        dm = PureMLIP.device_model(m, backend; elements = unique(Z))
        dnl = PureMLIP.device_neighbor_list(neighbor_list(P, C, Float64(m.r_max)), backend)
        ws = PureMLIP.device_workspace(dm, n, dnl.capacity; edge_chunk = 300)
        res = PureMLIP.device_result(dm, n)
        dpos, dcell, dZ = todevice(P), todevice(C), todevice(Int32.(Z))
        mace_energy_forces!(res, ws, dm, dpos, dcell, dZ, dnl)
        out(r) = map(Array, (r.energy, r.node_energies, r.forces, r.virial, r.stress))
        first_result = out(res)
        for _ in 1:3
            foreach(A -> fill!(A, zero(T)), (res.energy, res.node_energies, res.forces, res.virial, res.stress))
            mace_energy_forces!(res, ws, dm, dpos, dcell, dZ, dnl)
            @test out(res) == first_result
        end
        wide = todevice(hcat(P, ones(T, 3, 4)))
        mace_energy_forces!(res, ws, dm, view(wide, :, 1:n), dcell, view(todevice(Int32[Z; 1; 1]), 1:n), dnl)
        @test out(res) == first_result
    end
end

# Review Focus 4: without edges each atom's energy is its E0 (the readouts of zero features are
# zero, and MACE-MPA-0's shift is zero), and the forces, virial and stress are exactly zero. One
# atom, and two atoms of different elements farther apart than r_max, in a cell whose images are
# out of range too; the list has no spare capacity, so every edge array is empty.
@testitem "backward: one atom, and two atoms out of range, give the E0 sum and zero forces" tags = [:gpu] setup = [GPUBackend, DeviceStructures, EdgeStages] begin
    ck = mpa0()
    for T in precisions
        m = MACEModel{T}(ck)
        @test iszero(m.shift)
        C = T[20 0 0; 0 20 0; 0 0 20]
        for (P, Z) in ((reshape(T[10, 10, 10], 3, 1), [29]), (T[5 13; 5 5; 5 5], [29, 8]))
            n = size(P, 2)
            nl = neighbor_list(P, C, Float64(m.r_max))
            @test nl.n_edges == 0 && length(nl.senders) == 0
            dm = PureMLIP.device_model(m, backend; elements = Z)
            dnl = PureMLIP.device_neighbor_list(nl, backend)
            ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
            res = PureMLIP.device_result(dm, n)
            fill!(res.forces, one(T))
            mace_energy_forces!(res, ws, dm, todevice(P), todevice(C), todevice(Z), dnl)
            e0 = T[m.e0[m.species[z + 1]] for z in Z]
            @test Array(res.energy) == [sum(e0)]
            @test Array(res.node_energies) == e0
            @test all(iszero, Array(res.forces))
            @test all(iszero, Array(res.virial)) && all(iszero, Array(res.stress))
        end
    end
end

@testitem "backward: inconsistent results are rejected before any launch" tags = [:gpu] setup = [GPUBackend, DeviceStructures] begin
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    T = first(precisions)
    m = tiny(T)
    s = structures(T)[1]
    P, C, Z = s.positions, s.cell, s.atomic_numbers
    n = size(P, 2)
    dm = PureMLIP.device_model(m, backend; elements = [29])
    # A model with 257 channels is refused when the device model is built, before any launch. The
    # copy of `m` differs from it in the field `C` only, which `device_model` reads before any array.
    fields = Any[name === :C ? 257 : getfield(m, name) for name in fieldnames(typeof(m))]
    wide = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), typeof(m), fields, length(fields))
    @test_throws "gpu: the message pullback holds at most 256 channels, the model has 257" PureMLIP.device_model(wide, backend; elements = [29])
    dnl = PureMLIP.device_neighbor_list(neighbor_list(P, C, Float64(m.r_max)), backend)
    ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
    good = PureMLIP.device_result(dm, n)
    z(dims...) = todevice(zeros(T, dims...))
    dpos, dcell, dZ = todevice(P), todevice(C), todevice(Z)
    bad = ext.DeviceMACEResult(backend, good.energy, good.node_energies, z(3, n + 1), good.virial, good.stress)
    @test_throws "gpu: the result forces have size (3, $(n + 1)), expected (3, $n)" mace_energy_forces!(bad, ws, dm, dpos, dcell, dZ, dnl)
    bad = ext.DeviceMACEResult(backend, good.energy, good.node_energies, good.forces, good.virial, z(2, 2))
    @test_throws "gpu: the result virial and stress must be 3 × 3, got (3, 3) and (2, 2)" mace_energy_forces!(bad, ws, dm, dpos, dcell, dZ, dnl)
    @test_throws "gpu: $(n - 1) atomic numbers for $n atoms" mace_energy_forces!(good, ws, dm, dpos, dcell, dZ[1:(n - 1)], dnl)

    # The node pullbacks check their sizes.
    @test_throws "gpu: linear pullback from $(size(ws.x[2])) to $(size(ws.h0))" ext._linear_pullback!(ws.h0, dm.up[2], ws.x[2])
    @test_throws "gpu: species linear pullback from $(size(ws.X1)) to $(size(ws.h0))" ext._species_linear_pullback!(ws.h0, dm.skip[1], ws.X1, ws.species, [1], ws.dM[1], backend)
    @test_throws "gpu: scratch of 1 entries for an array of size (1, 2)" ext._front(todevice(zeros(T, 1)), (1, 2))
    Zf = ext._contraction_buffer(ws.Zf, dm.contraction[1], 1, ws.contraction_block, m.C)
    Yc = ext._contraction_buffer(ws.Yc, dm.contraction[1], 2, ws.contraction_block, m.C)
    @test_throws "gpu: symmetric contraction pullback from $(size(ws.B[2])) to $(size(ws.dX1)) for $n nodes" ext._symmetric_contraction_pullback!(ws.dX1, dm.contraction[1], ws.B[2], ws.X1, ws.species, n, Zf, Yc, backend)
end

# The device entry points return concrete types, the extension's own code dispatches nothing at
# run time, and every compiled signature of the extension resolves. These are the two layers of
# `test_signatures(:typestable)` (the inferred return type, and JET's optimization analysis),
# with the second confined to the extension: a kernel launch dispatches at run time inside
# KernelAbstractions, GPUCompiler and Base (over a hundred JET reports for `mace_energy_forces!` on
# the CPU backend), so the whole call tree that `test_signatures` analyzes cannot
# pass. `device_model` is not among them: its return type carries the coupling tables of the tensor
# products as a type parameter, which depends on the checkpoint's values.
@testitem "backward: device entry points are type-stable, and every compiled signature resolves" tags = [:gpu] setup = [GPUBackend, DeviceStructures] begin
    using StrictModeTest
    JET = StrictModeTest.JET
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    s = structures(Float32)[1]
    workspace(dm, n, e) = PureMLIP.device_workspace(dm, n, e; edge_chunk = 100)
    for T in precisions
        m = tiny(T)
        P, C, Z = T.(s.positions), T.(s.cell), s.atomic_numbers
        n = size(P, 2)
        nl = neighbor_list(P, C, Float64(m.r_max))
        dm = PureMLIP.device_model(m, backend; elements = [29])
        dnl = PureMLIP.device_neighbor_list(nl, backend)
        ws = workspace(dm, n, dnl.capacity)
        res = PureMLIP.device_result(dm, n)
        args = (res, ws, dm, todevice(P), todevice(C), todevice(Z), dnl)
        mace_energy!(args...)
        mace_energy_forces!(args...)
        types = map(typeof, args)
        sigs = (
            (mace_energy!, types, typeof(res)), (mace_energy_forces!, types, typeof(res)),
            (PureMLIP.device_result, (typeof(dm), Int), typeof(res)), (workspace, (typeof(dm), Int, Int), typeof(ws)),
            (PureMLIP.device_neighbor_list, (typeof(nl), typeof(backend)), typeof(dnl)),
            (PureMLIP.device_energy_forces!, types, typeof(res)), (PureMLIP.read_device_status!, (typeof(ws), typeof(dnl)), Vector{Int32}),
            (PureMLIP.check_device_status, (typeof(ws), typeof(dm), types[6], typeof(dnl)), Int),
            (PureMLIP.device_rebuild!, (typeof(dnl), types[4], types[5]), typeof(dnl)),
            (PureMLIP.device_sum!, (typeof(res.energy), typeof(res.node_energies), Int), typeof(res.energy)),
        )
        for (f, tt, rt) in sigs
            @test Base.return_types(f, tt) == [rt]
            @test isempty(JET.get_reports(JET.report_opt(f, tt; target_modules = (ext,))))
        end
    end
    @test length(test_calls(ext)) > 0
end

# Host bytes allocated by one call, through a compiled function so that the dynamic dispatch of
# the call site is not counted: the fewest of five calls after two warm-up calls, on cu_bench(2)
# with MACE-MPA-0, one element, one edge chunk and one contraction block (the defaults). The bytes
# are the backend's objects for each launch and matrix product and the read of the status word, so
# they grow with the edge chunks, the contraction blocks and the elements. On Metal one `mul!`
# allocates 3,008 bytes on the host, and the symmetric contraction runs one per channel (128) per
# product block, which is most of the figure. Measured:
#   CPU backend, Julia 1.12.7 and 1.13.3, Float32 and Float64: mace_energy! 1,168 bytes and
#     mace_energy_forces! 2,400, the same on every call;
#   Metal, Julia 1.13.3: mace_energy! 991,408 to 991,504 bytes and mace_energy_forces! 2,007,888 to
#     2,009,344, as the fewest of five calls over 24 groups of five in three processes; single
#     calls reached 996,688 and 2,170,192.
# The gate is the measured figure on the CPU backend on macOS, where it was measured, and twice it
# on other platforms; on Metal the largest fewest-of-five rounded up to 1,000,000 and 2,020,000
# bytes (0.9 % and 0.5 % above it).
# The figures hold when the items run through test/gpu/runtests.jl, which puts Revise (loaded by
# JET) in manual mode: in automatic mode its REPL poll allocates a 112-byte timer every 50 ms for a
# second after JET loads, and in fresh processes where this item ran within that second every call
# read 1,280 bytes.
# `neighbor_list!` on device types is gated in "neighbor list on the device: host bytes per call of
# neighbor_list!".
@testitem "backward: host bytes per call of the device evaluations" tags = [:gpu] setup = [GPUBackend, DeviceStructures, EdgeStages] begin
    @noinline call_bytes(f::F, a, b, c, d, e, g, h) where {F} = @allocated f(a, b, c, d, e, g, h)
    function fewest_bytes(f, args)
        f(args...)
        f(args...)
        return minimum(call_bytes(f, args...) for _ in 1:5)
    end
    cpu = Sys.isapple() ? 1 : 2
    bounds = Dict("CPU" => (energy = cpu * 1_168, forces = cpu * 2_400), "Metal" => (energy = 1_000_000, forces = 2_020_000))
    ck = mpa0()
    cu = DeviceStructures.cu_bench(2)
    for T in precisions
        m = MACEModel{T}(ck)
        P, C = T.(permutedims(cu.positions)), T.(permutedims(cu.cell))
        n = size(P, 2)
        dm = PureMLIP.device_model(m, backend; elements = [29])
        dnl = PureMLIP.device_neighbor_list(neighbor_list(P, C, Float64(m.r_max)), backend)
        ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
        args = (PureMLIP.device_result(dm, n), ws, dm, todevice(P), todevice(C), todevice(Int32.(cu.atomic_numbers)), dnl)
        be = fewest_bytes(mace_energy!, args)
        bf = fewest_bytes(mace_energy_forces!, args)
        report("host bytes, $(GPUBackend.NAME), $T, Julia $VERSION: mace_energy! $be, mace_energy_forces! $bf")
        if haskey(bounds, GPUBackend.NAME)
            @test be <= bounds[GPUBackend.NAME].energy
            @test bf <= bounds[GPUBackend.NAME].forces
        else
            # CUDA and AMDGPU are measured on their machines.
            @test_skip be <= 0
        end
    end
end

# The device memory in use does not grow from one evaluation to the next. The vendor's query is
# written for Metal; on the CPU backend there is no device memory apart from the host's, and the
# CUDA and AMDGPU queries are written on their machines.
@testitem "backward: device memory does not grow over repeated evaluations" tags = [:gpu] setup = [GPUBackend, DeviceStructures, EdgeStages] begin
    if GPUBackend.NAME != "Metal"
        @test_skip GPUBackend.NAME == "Metal"
    else
        device_bytes() = Int(GPUBackend.vendor.device().currentAllocatedSize)
        ck = mpa0()
        cu = DeviceStructures.cu_bench(2)
        T = Float32
        m = MACEModel{T}(ck)
        P, C = T.(permutedims(cu.positions)), T.(permutedims(cu.cell))
        n = size(P, 2)
        dm = PureMLIP.device_model(m, backend; elements = [29])
        dnl = PureMLIP.device_neighbor_list(neighbor_list(P, C, Float64(m.r_max)), backend)
        ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
        args = (PureMLIP.device_result(dm, n), ws, dm, todevice(P), todevice(C), todevice(Int32.(cu.atomic_numbers)), dnl)
        for _ in 1:10
            mace_energy_forces!(args...)
        end
        GPUBackend.KernelAbstractions.synchronize(backend)
        after10 = device_bytes()
        for _ in 1:200
            mace_energy_forces!(args...)
        end
        GPUBackend.KernelAbstractions.synchronize(backend)
        @test device_bytes() == after10
    end
end

# The kernels' arithmetic of the backward pass as host code in Float32 (see "edge stages: Float32
# stages do not widen to Float64"), and the extension's compiled functions after a Float32
# evaluation of the forces.
@testitem "backward: Float32 stages do not widen to Float64" tags = [:gpu] setup = [GPUBackend, DeviceStructures] begin
    using StrictMode, Random
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    T = Float32
    rng = Xoshiro(3)
    m = tiny(T)
    s = structures(T)[1]
    P, C, Z = s.positions, s.cell, s.atomic_numbers
    n = size(P, 2)
    dm = PureMLIP.device_model(m, backend; elements = [29])
    nl = neighbor_list(P, C, Float64(m.r_max))
    dnl = PureMLIP.device_neighbor_list(nl, backend)
    ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
    res = PureMLIP.device_result(dm, n)
    mace_energy_forces!(res, ws, dm, todevice(P), todevice(C), todevice(Z), dnl)

    E, Cc, tp = nl.n_edges, m.C, m.conv[2]
    table = ext._tp_table(tp)
    row_ptr, rev = Array(dnl.row_ptr), Array(dnl.reverse)
    Y = ext.StoredHarmonics(randn(rng, T, 16, E))
    w, x, dM = randn(rng, T, Cc, length(tp.paths), E), randn(rng, T, Cc, n, tp.K_in1), randn(rng, T, Cc, n, tp.K_out)
    dw = zeros(T, size(w))
    part = ext._tp_edge_pullback!(dw, 1, 1, 1, Y, w, x, dM, nl.senders[1], nl.receivers[1], table)
    dx = zeros(T, size(x))
    ext._tp_sender_receiver!(dx, 1, 1, Y, w, dM, nl.senders, ext._receiver_edges(row_ptr, 1, 0, E), 0, table)
    buf = randn(rng, T, 16, Cc)
    dYe = zeros(T, 16, 1)
    ext._channel_sum!(dYe, buf, 1, 1, Cc)
    nb = length(m.radial.bessel_weights)
    dEdr, r, d = zeros(T, 3, E), randn(rng, T, 3, E), rand(rng, T, E) .+ 1
    ext._edge_gradient!(dEdr, randn(rng, T, 16, 1), randn(rng, T, 16, 3, 1), randn(rng, T, nb, 1), randn(rng, T, nb, 1), randn(rng, T, nb, 1), randn(rng, T, n), m.density[1], r, d, nl.receivers, 1, 0)
    ext._zbl_gradient!(dEdr, r, d, nl.senders, nl.receivers, Int32.(Z), m.zbl.covalent_radii, ext._zbl_params(m.zbl, m.r_max), m.scale, 1)
    F, parts = zeros(T, 3, n), zeros(T, n, 9)
    ext._forces!(F, parts, dEdr, r, row_ptr, rev, 1)
    virial, stress = zeros(T, 3, 3), zeros(T, 3, 3)
    ext._virial_stress!(virial, stress, randn(rng, T, 9), C)
    L = m.product[1].contractions[1].n_ell
    X, Yc, dB, dX = randn(rng, T, Cc, n, L), randn(rng, T, 4L, 1, Cc), randn(rng, T, Cc, n, 4), zeros(T, Cc, n, L)
    ext._contract_pullback!(dX, Yc, dB, ones(Int32, n), 0, L, 4, 1, 1, 1, 1)
    ddens = zeros(T, n)
    ext._normalization_pullback!(ddens, X, copy(X), X, rand(rng, T, n), 1)
    g = randn(rng, T, 4)
    ext._readout_slope!(g, randn(rng, T, 4), m.readout_cst, 1)
    @test isfinite(part[1]) && any(!iszero, dw) && any(!iszero, dx) && any(!iszero, F)
    @test isempty(StrictMode.conversion_report(ext; from = Float32, to = Float64))
end

# A list whose status word holds -3 (an edge without a reverse, which the build marks with a
# reverse of 0) still runs the whole backward pass before the status read. The force gather skips
# that edge's reverse term instead of indexing outside the edge gradients (the items run with
# --check-bounds=yes), and the read throws the list's message.
@testitem "backward: a list with an edge without a reverse throws without reading outside the edges" tags = [:gpu] setup = [GPUBackend, DeviceStructures] begin
    T = first(precisions)
    m = tiny(T)
    s = structures(T)[1]
    P, C, Z = s.positions, s.cell, s.atomic_numbers
    n = size(P, 2)
    dm = PureMLIP.device_model(m, backend; elements = [29])
    dnl = PureMLIP.device_neighbor_list(neighbor_list(P, C, Float64(m.r_max)), backend)
    ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
    res = PureMLIP.device_result(dm, n)
    rev = Array(dnl.reverse)
    rev[1] = 0
    copyto!(dnl.reverse, rev)
    status = Array(dnl.status)
    status[2] = -3
    copyto!(dnl.status, status)
    @test_throws "gpu: an edge of the neighbor list has no reverse edge" mace_energy_forces!(res, ws, dm, todevice(P), todevice(C), todevice(Z), dnl)
    @test all(isfinite, Array(res.forces))
end

# The entry points of a step that reads the status word once: `device_rebuild!` launches the
# build and reads nothing, `device_energy_forces!` evaluates and reads nothing, and
# `read_device_status!` with `check_device_status` act on both: an overflow is returned as the
# number of edges found, not thrown, and after a valid build the device evaluation equals
# `mace_energy_forces!` bit for bit (the same kernels). `device_sum!` equals the device energy's
# own tree sum of the node energies.
@testitem "backward: the unread entry points and the status check" tags = [:gpu] setup = [GPUBackend, DeviceStructures] begin
    T = first(precisions)
    m = tiny(T)
    s = structures(T)[1]
    P, C, Z = s.positions, s.cell, s.atomic_numbers
    n = size(P, 2)
    r_max = Float64(m.r_max)
    dP, dC, dZ = todevice(P), todevice(C), todevice(Z)
    count = neighbor_list(P, C, r_max).n_edges
    dnl = PureMLIP.device_neighbor_list(dP, dC, r_max; extra_capacity = 2, extra_capacity_fraction = 0.05)
    @test dnl.capacity == count + 2 + ceil(Int, 0.05 * count)
    @test_throws "gpu: extra_capacity_fraction must be finite and non-negative, got -0.1" PureMLIP.device_neighbor_list(dP, dC, r_max; extra_capacity_fraction = -0.1)
    dm = PureMLIP.device_model(m, backend; elements = [29])
    ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
    res, ref = PureMLIP.device_result(dm, n), PureMLIP.device_result(dm, n)
    mace_energy_forces!(ref, ws, dm, dP, dC, dZ, dnl)
    @test PureMLIP.device_rebuild!(dnl, dP, dC) === dnl
    @test PureMLIP.device_energy_forces!(res, ws, dm, dP, dC, dZ, dnl) === res
    @test PureMLIP.read_device_status!(ws, dnl) === ws.status
    @test PureMLIP.check_device_status(ws, dm, dZ, dnl) == 0
    @test Array(res.forces) == Array(ref.forces) && Array(res.energy) == Array(ref.energy)
    out = todevice(zeros(T, 1))
    @test PureMLIP.device_sum!(out, res.node_energies, n) === out
    @test Array(out) == Array(res.energy)
    @test_throws "gpu: positions and cell must have the list's element type" PureMLIP.device_rebuild!(dnl, todevice(Float16.(P)), todevice(Float16.(C)))
    # A compressed copy overflows: the build reads nothing back, the check returns the count.
    f = T(0.8)
    more = neighbor_list(f * P, f * C, r_max).n_edges
    @test more > dnl.capacity
    PureMLIP.device_rebuild!(dnl, todevice(f * P), todevice(f * C))
    PureMLIP.device_energy_forces!(res, ws, dm, todevice(f * P), todevice(f * C), dZ, dnl)
    PureMLIP.read_device_status!(ws, dnl)
    @test PureMLIP.check_device_status(ws, dm, dZ, dnl) == more
    @test_throws "gpu: $more edges exceed the capacity $(dnl.capacity)" mace_energy_forces!(res, ws, dm, todevice(f * P), todevice(f * C), dZ, dnl)
end
