# The device forward pass, `mace_energy!` with device types, against CPU `mace_energy!` on the five
# LAMMPS reference structures with MACE-MPA-0.

# Bound (b) of the tolerance policy: 10·eps(T)·N relative to the largest magnitude, separately for
# the energy and the node energies, with N = 1280 the largest inner dimension of any matrix product
# in the model. The device path takes the stages of the CPU path in another order of summation (its
# matrix products, the matrix form of both product blocks, a tree sum for the energy), so the two
# differ by rounding; a backend's exp, tanh, sincos and sqrt may round differently from Base's.
# Measured on the CPU backend and on Metal: at most 5.9·eps(T) (the node energies, Float32 on Metal).
# The list has a skin and spare capacity, so the chunks past the edges hold slots without an edge;
# edge chunks of 100 slots split the 32-atom cells so that a receiver's edges span chunks;
# contraction blocks of 8 atoms split them into several blocks. The device model holds the elements
# of each structure, one species for the copper cells and up to eight for the others.
@testitem "forward: energy and node energies agree with CPU mace_energy! on the reference structures" tags = [:gpu] setup = [GPUBackend, DeviceStructures, EdgeStages] begin
    ck = mpa0()
    for T in precisions
        m = MACEModel{T}(ck)
        tol = 10 * 1280   # in units of eps(T)
        worst = [0.0, 0.0]
        nspan = 0
        for s in structures(T)
            P, C, Z = s.positions, s.cell, s.atomic_numbers
            n = size(P, 2)
            nl = neighbor_list(P, C, Float64(m.r_max) + 0.5; extra_capacity = 200 * n)
            res = PureMLIP.MACEResult{T}(n)
            mace_energy!(res, MACEWorkspace{T}(m, n, nl.n_edges; elements = unique(Z)), m, P, C, Z, nl)
            dm = PureMLIP.device_model(m, backend; elements = unique(Z))
            dnl = PureMLIP.device_neighbor_list(nl, backend)
            @test dnl.capacity > 2 * nl.n_edges
            nspan += nl.n_edges > 100
            ws = PureMLIP.device_workspace(dm, n, dnl.capacity; edge_chunk = 100, contraction_block = 8)
            dres = PureMLIP.device_result(dm, n)
            # Integer atomic numbers of either width.
            dZ = s.name == "multi_element" ? todevice(Z) : todevice(Int32.(Z))
            @test mace_energy!(dres, ws, dm, todevice(P), todevice(C), dZ, dnl) === dres
            eE = Float64(abs(Array(dres.energy)[1] - res.energy) / abs(res.energy) / eps(T))
            eN = relerr(Array(dres.node_energies), res.node_energies)
            @test eE <= tol
            @test eN <= tol
            worst .= max.(worst, (eE, eN))
            @test Array(dnl.status) == zeros(Int32, 5)
        end
        # The three crystals have more edges than one chunk holds.
        @test nspan == 3
        report("forward, $(GPUBackend.NAME), $T: energy, node energies in eps($T): ", worst)
    end
end

# The per-receiver kernels of a chunk launch work items for the receivers of a chunk at the list's
# mean degree per slot; with ten times more slots than edges a chunk of 2000 slots holds more
# receivers than that, and each work item takes several. Same bound as the item above.
@testitem "forward: per-receiver kernels cover chunks with more receivers than work items" tags = [:gpu] setup = [GPUBackend, DeviceStructures, EdgeStages] begin
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    ck = mpa0()
    for T in precisions
        m = MACEModel{T}(ck)
        s = structures(T)[1]
        P, C, Z = s.positions, s.cell, s.atomic_numbers
        n = size(P, 2)
        nl = neighbor_list(P, C, Float64(m.r_max); extra_capacity = 10 * 100 * n)
        E = nl.n_edges
        res = PureMLIP.MACEResult{T}(n)
        mace_energy!(res, MACEWorkspace{T}(m, n, E; elements = [29]), m, P, C, Z, nl)
        dm = PureMLIP.device_model(m, backend; elements = [29])
        dnl = PureMLIP.device_neighbor_list(nl, backend)
        @test dnl.capacity > 10 * E
        R = ext._chunk_receivers(dnl, 2000)
        @test length(unique(nl.receivers[1:2000])) > R
        ws = PureMLIP.device_workspace(dm, n, dnl.capacity; edge_chunk = 2000)
        dres = PureMLIP.device_result(dm, n)
        mace_energy!(dres, ws, dm, todevice(P), todevice(C), todevice(Z), dnl)
        @test abs(Array(dres.energy)[1] - res.energy) / abs(res.energy) / eps(T) <= 10 * 1280
        @test relerr(Array(dres.node_energies), res.node_energies) <= 10 * 1280
    end
end

@testitem "forward: evaluations repeat bit for bit, and views of device positions give the same result" tags = [:gpu] setup = [GPUBackend, DeviceStructures, EdgeStages] begin
    ck = mpa0()
    for T in precisions
        m = MACEModel{T}(ck)
        s = structures(T)[4]
        P, C, Z = s.positions, s.cell, s.atomic_numbers
        n = size(P, 2)
        dm = PureMLIP.device_model(m, backend; elements = Z)
        dnl = PureMLIP.device_neighbor_list(neighbor_list(P, C, Float64(m.r_max)), backend)
        ws = PureMLIP.device_workspace(dm, n, dnl.capacity; edge_chunk = 50)
        res = PureMLIP.device_result(dm, n)
        dpos, dcell, dZ = todevice(P), todevice(C), todevice(Int32.(Z))
        mace_energy!(res, ws, dm, dpos, dcell, dZ, dnl)
        E, En = Array(res.energy), Array(res.node_energies)
        for _ in 1:3
            fill!(res.energy, zero(T))
            fill!(res.node_energies, zero(T))
            mace_energy!(res, ws, dm, dpos, dcell, dZ, dnl)
            @test Array(res.energy) == E && Array(res.node_energies) == En
        end
        wide = todevice(hcat(P, ones(T, 3, 4)))
        mace_energy!(res, ws, dm, view(wide, :, 1:n), dcell, view(todevice(Int32[Z; 1; 1]), 1:n), dnl)
        @test Array(res.energy) == E && Array(res.node_energies) == En
    end
end

# Review Focus 5: an atomic number that the device model does not hold is found on the device and
# fails the evaluation, whether the model's element table has it (silver) or not (200); the next
# evaluation of valid input succeeds.
@testitem "forward: an atomic number outside the device model's elements throws" tags = [:gpu] setup = [GPUBackend, DeviceStructures, EdgeStages] begin
    ck = mpa0()
    T = first(precisions)
    m = MACEModel{T}(ck)
    s = structures(T)[1]
    P, C, Z = s.positions, s.cell, s.atomic_numbers
    n = size(P, 2)
    dm = PureMLIP.device_model(m, backend; elements = [29])
    dnl = PureMLIP.device_neighbor_list(neighbor_list(P, C, Float64(m.r_max)), backend)
    ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
    res = PureMLIP.device_result(dm, n)
    dpos, dcell = todevice(P), todevice(C)
    for z in (47, 200)
        Zx = copy(Z)
        Zx[5] = z
        @test_throws "gpu: atomic number $z is not among the device model's elements [29]" mace_energy!(res, ws, dm, dpos, dcell, todevice(Zx), dnl)
    end
    mace_energy!(res, ws, dm, dpos, dcell, todevice(Z), dnl)
    @test isfinite(Array(res.energy)[1])
end

# Two atoms at one place (a list built before they met) give an edge of length zero, whose
# harmonics are undefined; the device flags it in the status word and the evaluation throws.
@testitem "forward: coincident atoms throw" tags = [:gpu] setup = [GPUBackend, DeviceStructures] begin
    T = first(precisions)
    m = tiny(T)
    s = structures(T)[1]
    P, C, Z = s.positions, s.cell, s.atomic_numbers
    n = size(P, 2)
    dm = PureMLIP.device_model(m, backend; elements = [29])
    nl = neighbor_list(P, C, Float64(m.r_max))
    dnl = PureMLIP.device_neighbor_list(nl, backend)
    ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
    res = PureMLIP.device_result(dm, n)
    Q = copy(P)
    Q[:, 2] = Q[:, 1]
    e = findfirst(e -> nl.senders[e] == 2 && nl.receivers[e] == 1 && iszero(nl.shifts[:, e]), 1:nl.n_edges)
    @test_throws "gpu: edge $e from atom 2 to atom 1 has length zero" mace_energy!(res, ws, dm, todevice(Q), todevice(C), todevice(Z), dnl)
    mace_energy!(res, ws, dm, todevice(P), todevice(C), todevice(Z), dnl)
    @test isfinite(Array(res.energy)[1])
end

@testitem "forward: inconsistent arguments are rejected before any launch" tags = [:gpu] setup = [GPUBackend, DeviceStructures, FakeDevice] begin
    using OffsetArrays
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    T = first(precisions)
    m = tiny(T)
    s = structures(T)[1]
    P, C, Z = s.positions, s.cell, s.atomic_numbers
    n = size(P, 2)
    dm = PureMLIP.device_model(m, backend; elements = [29])
    dnl = PureMLIP.device_neighbor_list(neighbor_list(P, C, Float64(m.r_max)), backend)
    ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
    res = PureMLIP.device_result(dm, n)
    dpos, dcell, dZ = todevice(P), todevice(C), todevice(Z)
    @test_throws "gpu: $(n - 1) atomic numbers for $n atoms" mace_energy!(res, ws, dm, dpos, dcell, dZ[1:(n - 1)], dnl)
    @test_throws "gpu: the result holds $(n + 1) atoms, the neighbor list has $n" mace_energy!(PureMLIP.device_result(dm, n + 1), ws, dm, dpos, dcell, dZ, dnl)
    @test_throws "gpu: the workspace holds $(n - 1) atoms, the neighbor list has $n" mace_energy!(res, PureMLIP.device_workspace(dm, n - 1, dnl.capacity), dm, dpos, dcell, dZ, dnl)
    short = PureMLIP.device_neighbor_list(neighbor_list(P, C, 4.0), backend)
    @test_throws "gpu: the neighbor list cutoff 4.0 is below r_max" mace_energy!(res, PureMLIP.device_workspace(dm, n, short.capacity), dm, dpos, dcell, dZ, short)
    @test_throws "gpu: all arrays must be on the backend of the positions" mace_energy!(res, ws, dm, OtherMatrix(P), dcell, dZ, dnl)
    @test_throws "gpu: arrays must be 1-based" mace_energy!(res, ws, dm, dpos, dcell, OffsetArray(dZ, 0:(n - 1)), dnl)
    @test_throws "gpu: the contraction block must be positive, got 0" PureMLIP.device_workspace(dm, n, dnl.capacity; contraction_block = 0)
    @test PureMLIP.device_workspace(dm, 7, 40).contraction_block == 7

    # The node stages check their sizes.
    x = similar(ws.x[2])
    @test_throws "gpu: linear map from $(size(x)) to $(size(ws.h0))" ext._linear!(ws.h0, dm.up[2], x)
    @test_throws "gpu: species linear map from $(size(ws.X1)) to $(size(ws.h0))" ext._species_linear!(ws.h0, dm.skip[1], ws.X1, ws.species, [1], ws.dX1, backend)
    Zf = ext._contraction_buffer(ws.Zf, dm.contraction[1], 1, ws.contraction_block, m.C)
    Yc = ext._contraction_buffer(ws.Yc, dm.contraction[1], 2, ws.contraction_block, m.C)
    @test_throws "gpu: symmetric contraction from $(size(ws.X1)) to $(size(ws.B[2])) for $n nodes" ext._symmetric_contraction!(ws.B[2], dm.contraction[1], ws.X1, ws.species, n, Zf, Yc, backend)
    @test_throws "gpu: contraction buffers of sizes $(size(Yc)) and $(size(Yc))" ext._symmetric_contraction!(ws.B[1], dm.contraction[1], ws.X1, ws.species, n, Yc, Yc, backend)
    @test_throws "gpu: the sum of 9 entries of a vector of length 8" ext._sum!(res.energy, todevice(zeros(T, 8)), 9, backend)
end

# The energy is a fixed-shape tree sum; with integers below 2^24 every partial sum is exact, so the
# sum of 1, …, 1000 (more entries than the 256 work items) must be exact.
@testitem "forward: the tree sum adds every entry" tags = [:gpu] setup = [GPUBackend] begin
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    for T in precisions, n in (0, 1, 255, 256, 257, 1000)
        out = todevice(T[-1])
        ext._sum!(out, todevice(T.(1:1000)), n, backend)
        @test Array(out) == [T(n * (n + 1) ÷ 2)]
    end
end

# The kernels' arithmetic as host code in Float32 (see "edge stages: Float32 stages do not widen to
# Float64"), and the extension's compiled functions after a Float32 evaluation.
@testitem "forward: Float32 stages do not widen to Float64" tags = [:gpu] setup = [GPUBackend, DeviceStructures] begin
    using StrictMode
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    T = Float32
    m = tiny(T)
    s = structures(T)[1]
    P, C, Z = s.positions, s.cell, s.atomic_numbers
    n = size(P, 2)
    dm = PureMLIP.device_model(m, backend; elements = [29])
    nl = neighbor_list(P, C, Float64(m.r_max))
    dnl = PureMLIP.device_neighbor_list(nl, backend)
    ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
    res = PureMLIP.device_result(dm, n)
    mace_energy!(res, ws, dm, todevice(P), todevice(C), todevice(Z), dnl)

    E, Cc, tp = nl.n_edges, m.C, m.conv[2]
    senders, row_ptr = nl.senders, Array(dnl.row_ptr)
    Y, w, x, M = randn(T, 16, E), randn(T, Cc, length(tp.paths), E), randn(T, Cc, n, tp.K_in1), zeros(T, Cc, n, tp.K_out)
    edges = ext._receiver_edges(row_ptr, 1, 0, E)
    ext._tp_receiver!(M, 1, 1, ext.StoredHarmonics(Y), w, x, senders, edges, 0, ext._tp_table(tp))
    density, feats = zeros(T, n), randn(T, length(m.radial.bessel_weights), E)
    ext._density_receiver!(density, 1, feats, m.density[1], edges, 0)
    L = m.product[1].contractions[1].n_ell
    cm = PureMLIP.Tensors.ContractionMatrices{T}(m.product[1])
    pairs = Int32[p[a] for a in 1:2, p in cm.pairs]
    X, Zf = randn(T, Cc, n, L), zeros(T, size(pairs, 2) + L + 1, 1, Cc)
    ext._features!(Zf, X, pairs, L, 1, 1, 1)
    Yc, B, sp = randn(T, 4L, 1, Cc), zeros(T, Cc, n, 4), ones(Int32, n)
    ext._contract!(B, Yc, X, sp, 0, L, 1, 1, 1, 1)
    ext._mask!(similar(X), X, sp, 1, 1, 1, 1)
    h0 = zeros(T, Cc, n, 1)
    ext._embed!(h0, m.embedding, sp, 1, 1)
    ext._normalize!(X, density, 1, 1, 1)
    ext._add!(X, X, 1)
    g = zeros(T, 4)
    ext._readout_silu!(g, randn(T, 4), m.readout_cst, 1)
    e1, e2, energies = randn(T, 1, n, 1), randn(T, 1, n, 1), zeros(T, n)
    ext._node_energy!(energies, m.e0, sp, zeros(T, n), e1, e2, m.scale, m.shift, 1)
    @test isfinite(energies[1]) && any(!iszero, M)
    @test isempty(StrictMode.conversion_report(ext; from = Float32, to = Float64))
end
