# Arrays that report another backend, and a backend without Float64, so that the mixed-argument
# checks run on every backend, the KernelAbstractions CPU backend included.
@testmodule FakeDevice begin
    using KernelAbstractions

    struct OtherBackend <: KernelAbstractions.Backend end
    struct NoFloat64Backend <: KernelAbstractions.Backend end
    KernelAbstractions.supports_float64(::NoFloat64Backend) = false
    # A backend that allocates host vectors, as a GPU backend allocates its arrays (not `Memory`).
    struct HostAllocBackend <: KernelAbstractions.Backend end
    KernelAbstractions.allocate(::HostAllocBackend, ::Type{T}, dims::Tuple) where {T} = Array{T}(undef, dims)

    # A host matrix that reports `OtherBackend` as its backend.
    struct OtherMatrix{T} <: AbstractMatrix{T}
        data::Matrix{T}
    end
    Base.size(A::OtherMatrix) = size(A.data)
    Base.getindex(A::OtherMatrix, i::Int, j::Int) = A.data[i, j]
    KernelAbstractions.get_backend(::OtherMatrix) = OtherBackend()

    export OtherBackend, NoFloat64Backend, HostAllocBackend, OtherMatrix
end

@testmodule DeviceStructures begin
    using PureMLIP, LinearAlgebra
    include(pkgdir(PureMLIP, "test", "reference", "structures.jl"))
    # The five LAMMPS reference structures in `T`, positions and lattice vectors as columns.
    structures(::Type{T}) where {T} =
        [(; s.name, positions = T.(permutedims(s.positions)), cell = T.(permutedims(s.cell)), s.atomic_numbers) for s in reference_structures()]
    tiny(::Type{T}) where {T} = MACEModel{T}(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint()))
    export structures, tiny
end

@testitem "device model: every parameter array is uploaded exactly" tags = [:gpu] setup = [GPUBackend, DeviceStructures] begin
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    for T in precisions
        m = tiny(T)
        dm = PureMLIP.device_model(m, backend; elements = [29, 1, 29])
        @test dm isa ext.DeviceMACEModel{T}
        @test all(isconcretetype, fieldtypes(typeof(dm)))
        @test dm.host === m
        @test dm.elements == [29, 1]
        @test Array(dm.species) == m.species
        @test Array(dm.e0) == m.e0
        @test Array(dm.embedding) == m.embedding
        @test Array(dm.bessel_weights) == m.radial.bessel_weights
        @test Array(dm.radial_radii) == m.radial.covalent_radii
        @test Array(dm.zbl_radii) == m.zbl.covalent_radii
        for k in 1:2
            @test all(Array(dm.mlp[k][j]) == m.mlp[k].W[j] for j in 1:4)
            @test Array(dm.density[k]) == m.density[k]
            tp = m.conv[k]
            @test Array(dm.conv[k].index) == [e[a] for a in 1:3, e in tp.entries]
            @test Array(dm.conv[k].value) == [e[4] for e in tp.entries]
            sl = m.skip[k]
            @test all(Array(dm.skip[k].W)[:, :, :, p] == sl.W[p] for p in eachindex(sl.W))
            @test Array(dm.skip[k].c) == sl.c
            for (dct, ct) in zip(dm.product[k].contractions, m.product[k].contractions)
                @test (dct.l, dct.n_ell) == (ct.l, ct.n_ell)
                for (I, V, U) in ((dct.U1_index, dct.U1_value, ct.U1), (dct.U2_index, dct.U2_value, ct.U2), (dct.U3_index, dct.U3_value, ct.U3))
                    @test Array(I) == [u[1][a] for a in eachindex(first(U)[1]), u in U]
                    @test Array(V) == [u[2] for u in U]
                end
                @test Array(dct.Wmax) == ct.Wmax && Array(dct.W0) == ct.W0 && Array(dct.W1) == ct.W1
            end
            @test length(dm.product[k].contractions) == length(m.product[k].contractions)
            cm = PureMLIP.Tensors.ContractionMatrices{T}(m.product[k])
            dcm = dm.contraction[k]
            @test Array(dcm.pairs) == [p[a] for a in 1:2, p in cm.pairs]
            for s in (m.species[30], m.species[2])
                PureMLIP.Tensors.contraction_matrices!(cm, m.product[k], s)
                @test Array(dcm.F)[:, :, :, dcm.slot[s]] == cm.F[s]
                @test Array(dcm.G)[:, :, :, dcm.slot[s]] == cm.G[s]
            end
        end
        for (dl, l) in (
                (dm.up[1], m.up[1]), (dm.up[2], m.up[2]), (dm.linear[1], m.linear[1]), (dm.linear[2], m.linear[2]),
                (dm.product_linear[1], m.product_linear[1]), (dm.product_linear[2], m.product_linear[2]),
                (dm.readout0, m.readout0), (dm.readout1[1], m.readout1[1]), (dm.readout1[2], m.readout1[2]),
            )
            @test size(dl.W, 3) == length(l.W) && all(Array(dl.W)[:, :, p] == l.W[p] for p in eachindex(l.W))
            @test Array(dl.pw) == l.pw
        end
        # The model built for one element holds the contraction matrices of that species alone.
        dcu = PureMLIP.device_model(m, backend; elements = [29])
        @test Array(dcu.species) == [z == 29 ? m.species[30] : 0 for z in 0:(length(m.species) - 1)]
        @test size(dcu.contraction[1].F, 4) == 1 && dcu.contraction[1].slot[m.species[2]] == 0
    end
end

@testitem "device neighbor list: the CSR and the reverse map of the reference structures" tags = [:gpu] setup = [GPUBackend, DeviceStructures] begin
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    for T in precisions, s in structures(T)
        nl = neighbor_list(s.positions, s.cell, 6.0; extra_capacity = 5)
        dnl = PureMLIP.device_neighbor_list(nl, backend)
        @test dnl isa ext.DeviceNeighborList{T}
        @test all(isconcretetype, fieldtypes(typeof(dnl)))
        E, n = nl.n_edges, nl.n_atoms
        @test (dnl.r_cut, dnl.skin, dnl.n_atoms, dnl.capacity) == (T(6), zero(T), n, E + 5)
        @test Array(dnl.n_edges) == [E]
        @test Array(dnl.status) == zeros(Int32, 4)
        row_ptr = Array(dnl.row_ptr)
        rcv, snd, sh, rev = Array(dnl.receivers), Array(dnl.senders), Array(dnl.shifts), Array(dnl.reverse)
        @test row_ptr[end] - 1 == E
        @test length(row_ptr) == n + 1 && row_ptr[1] == 1
        @test all(all(rcv[e] == i for e in row_ptr[i]:(row_ptr[i + 1] - 1)) for i in 1:n)
        @test rcv[1:E] == nl.receivers[1:E] && snd[1:E] == nl.senders[1:E] && sh[:, 1:E] == nl.shifts[:, 1:E]
        @test all(iszero, rcv[(E + 1):end]) && all(iszero, rev[(E + 1):end])
        @test all(e -> rev[rev[e]] == e, 1:E)
        @test all(e -> snd[rev[e]] == rcv[e] && rcv[rev[e]] == snd[e], 1:E)
        @test all(e -> sh[:, rev[e]] == -sh[:, e], 1:E)
        s.name == "isolated_cu" && @test E == 0
    end
end

@testitem "device neighbor list from device positions and cell" tags = [:gpu] setup = [GPUBackend, DeviceStructures] begin
    for T in precisions, s in structures(T)
        n = size(s.positions, 2)
        host = PureMLIP.device_neighbor_list(neighbor_list(s.positions, s.cell, 6.5; extra_capacity = 3), backend)
        dpos, dcell = todevice(s.positions), todevice(s.cell)
        dnl = PureMLIP.device_neighbor_list(dpos, dcell, 6.0; skin = 0.5, extra_capacity = 3)
        @test (dnl.r_cut, dnl.skin, dnl.capacity) == (T(6), T(0.5), host.capacity)
        for f in (:row_ptr, :receivers, :senders, :shifts, :reverse, :n_edges)
            @test Array(getfield(dnl, f)) == Array(getfield(host, f))
        end
        # A view of the first n columns of a wider device array gives the same list.
        wide = todevice(hcat(s.positions, zeros(T, 3, 2)))
        dview = PureMLIP.device_neighbor_list(view(wide, :, 1:n), dcell, 6.0; skin = 0.5, extra_capacity = 3)
        @test Array(dview.senders) == Array(dnl.senders) && Array(dview.shifts) == Array(dnl.shifts)
    end
end

@testitem "device workspace and result: sizes, zeros and shared pools" tags = [:gpu] setup = [GPUBackend, DeviceStructures, FakeDevice] begin
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    for T in precisions
        m = tiny(T)
        dm = PureMLIP.device_model(m, backend; elements = [1, 29])
        ws = PureMLIP.device_workspace(dm, 7, 40; edge_chunk = 16)
        @test ws isa ext.DeviceMACEWorkspace{T}
        @test all(isconcretetype, fieldtypes(typeof(ws)))
        @test (ws.n_atoms, ws.n_edges, ws.edge_chunk) == (7, 40, 16)
        @test size(ws.r) == (3, 40) && size(ws.d) == (40,) && size(ws.dEdr) == (3, 40)
        W, H, nb = size(m.mlp[2].W[4], 2), size(m.mlp[1].W[1], 2), length(m.radial.bessel_weights)
        @test size(ws.Y) == (16, 16) && size(ws.dY) == (16, 3, 16) && size(ws.feats) == (nb, 16)
        @test size(ws.tpw) == (W, 16) && size(ws.tpw3) == (m.C, W ÷ m.C, 16) && size(ws.pre) == (H, 16, 3)
        # Per-node buffers are (C, n, K).
        @test size(ws.M[2]) == (m.C, 7, m.conv[2].K_out) && size(ws.X2) == (m.C, 7, m.linear[2].K_out)
        @test size(ws.h0) == (m.C, 7, 1) && size(ws.X1) == (m.C, 7, m.skip[1].K) && size(ws.B[1]) == (m.C, 7, 4)
        @test size(ws.x[2]) == (m.C, 7, m.up[2].K_out) && size(ws.dx) == (m.C, 7, m.up[2].K_out) && size(ws.e2) == (1, 7, 1)
        @test all(f -> all(iszero, Array(getfield(ws, f))), (:r, :feats, :tpw, :h0, :X1, :pair, :dx, :ddensity))
        # Members of one pool share memory; tpw3 is tpw.
        fill!(ws.h0, one(T))
        @test Array(ws.dh2)[1] == one(T)
        fill!(ws.M[1], T(2))
        @test Array(ws.dM[2])[1] == T(2)
        fill!(ws.L0, T(3))
        @test Array(ws.dL0)[1] == T(3)
        fill!(ws.tpw, T(4))
        @test all(==(T(4)), Array(ws.tpw3))
        # On a backend that allocates vectors, as GPU backends do, pool members are reshaped
        # contiguous views of one block, side by side within a group and overlapping across groups.
        ((a, b), (c,)) = ext._device_pool(HostAllocBackend(), T, (((2, 1, 1), (1, 2, 2)), ((3, 1, 1),)))
        @test size(a) == (2, 1, 1) && size(b) == (1, 2, 2) && size(c) == (3, 1, 1)
        @test all(iszero, a) && all(iszero, b) && all(iszero, c)
        b[1, 1, 1] = one(T)
        @test c[3, 1, 1] == one(T) && parent(parent(b)) === parent(parent(c))
        # The edge chunk holds at most every edge.
        @test PureMLIP.device_workspace(dm, 7, 40; edge_chunk = 1000).edge_chunk == 40
        @test PureMLIP.device_workspace(dm, 0, 0).edge_chunk == 1
        @test PureMLIP.device_workspace(dm, 7, 40).edge_chunk == 40
        @test_throws "gpu: workspace sizes must not be negative, got -1 atoms and 40 edges" PureMLIP.device_workspace(dm, -1, 40)
        @test_throws "gpu: the edge chunk must be positive, got 0" PureMLIP.device_workspace(dm, 7, 40; edge_chunk = 0)

        res = PureMLIP.device_result(dm, 7)
        @test res isa ext.DeviceMACEResult{T}
        @test all(isconcretetype, fieldtypes(typeof(res)))
        @test size(res.energy) == (1,) && size(res.node_energies) == (7,) && size(res.forces) == (3, 7)
        @test size(res.virial) == (3, 3) && size(res.stress) == (3, 3)
        @test all(iszero, Array(res.forces)) && all(iszero, Array(res.energy))
        @test_throws "gpu: the number of atoms must not be negative, got -2" PureMLIP.device_result(dm, -2)
    end
end

@testitem "device entry points reject mixed arguments before any launch" tags = [:gpu] setup = [GPUBackend, DeviceStructures, FakeDevice] begin
    using OffsetArrays
    m32 = tiny(Float32)
    ck = load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint())
    # A backend without Float64 refuses a Float64 model, naming the backend.
    @test_throws "gpu: NoFloat64 has no Float64; build the model as MACEModel{Float32}" PureMLIP.device_model(MACEModel{Float64}(ck), NoFloat64Backend(); elements = [29])
    if isgpu && !GPUBackend.KernelAbstractions.supports_float64(backend)
        @test_throws "gpu: Metal has no Float64; build the model as MACEModel{Float32}" PureMLIP.device_model(MACEModel{Float64}(ck), backend; elements = [29])
    end
    @test_throws "gpu: element 47 is not in the model's element table" PureMLIP.device_model(m32, backend; elements = [29, 47])
    @test_throws "gpu: elements must name at least one atomic number" PureMLIP.device_model(m32, backend; elements = Int[])
    @test_throws "gpu: element -1 is not in the model's element table" PureMLIP.device_model(m32, backend; elements = [-1])

    s = structures(Float32)[3]
    dpos, dcell = todevice(s.positions), todevice(s.cell)
    @test_throws "gpu: all arrays must be on the backend of the positions" PureMLIP.device_neighbor_list(dpos, OtherMatrix(s.cell), 6.0)
    # Host positions with a device cell: on the CPU backend host arrays are device arrays.
    if isgpu
        @test_throws "gpu: all arrays must be on the backend of the positions" PureMLIP.device_neighbor_list(s.positions, dcell, 6.0)
    end
    @test_throws "gpu: arrays must be 1-based" PureMLIP.device_neighbor_list(OffsetArray(dpos, 0:2, 1:4), dcell, 6.0)
    @test_throws "gpu: positions and cell must share one element type, got Float32 and Float64" PureMLIP.device_neighbor_list(s.positions, Float64.(s.cell), 6.0)
    @test_throws "gpu: positions must hold floating-point numbers, got Int64" PureMLIP.device_neighbor_list(ones(Int, 3, 4), ones(Int, 3, 3), 6.0)
    @test_throws "gpu: the skin must not be negative, got -0.5" PureMLIP.device_neighbor_list(dpos, dcell, 6.0; skin = -0.5)

    # A host list that is not sorted by receiver, or that lacks a reverse edge.
    nl = neighbor_list(s.positions, s.cell, 6.0)
    bad = deepcopy(nl); bad.receivers[1], bad.receivers[end] = bad.receivers[end], bad.receivers[1]
    @test_throws "gpu: neighbor list edges must be sorted by receiver" PureMLIP.device_neighbor_list(bad, backend)
    bad = deepcopy(nl); bad.receivers[bad.n_edges] = 5
    @test_throws "gpu: neighbor list receivers must lie in 1:4" PureMLIP.device_neighbor_list(bad, backend)
    bad = deepcopy(nl); bad.shifts[1, 1] += 7
    @test_throws "gpu: neighbor list edge 1 has no reverse edge" PureMLIP.device_neighbor_list(bad, backend)
end
