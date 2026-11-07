# The per-edge stages on the device against their CPU functions, on the edges of the five LAMMPS
# reference structures with MACE-MPA-0. Each stage gets the CPU's inputs, so that a disagreement
# belongs to the stage itself.
@testmodule EdgeStages begin
    using PureMLIP, Random
    include(pkgdir(PureMLIP, "test", "reference", "checkpoints.jl"))

    mpa0() = load_mace_checkpoint(checkpoint_path(:mpa0))

    # A device copy of the host array `A` placed in the first columns of a device buffer `D`.
    function upload_columns!(D, A::AbstractArray)
        H = zeros(eltype(D), size(D))
        H[axes(A)...] = A
        copyto!(D, H)
        return D
    end

    # The largest difference relative to the largest magnitude of the CPU value, in units of eps(T).
    function relerr(gpu, cpu)
        isempty(cpu) && return 0.0
        m = maximum(abs, cpu)
        δ = maximum(abs, gpu - cpu)
        return iszero(m) ? (iszero(δ) ? 0.0 : Inf) : Float64(δ / m / eps(eltype(cpu)))
    end

    # The ZBL energy of an edge is h·v·f and its length derivative h·(v'·f + v·f'), with the screened
    # Coulomb term v, h = 1/2 and the envelope f(t) = 1 - c1 t^p + c2 t^(p+1) - c3 t^(p+2) at
    # t = d / R. Near t = 1 the envelope's terms cancel (f, f' and f'' vanish there), so a rounding
    # of one term by eps changes the result by eps times the term, not times the result. These are
    # the magnitudes that the rounding scales with: h·|v|·Σ|terms of f| and
    # h·(|v'|·Σ|terms of f| + |v|·Σ|terms of f'|), in Float64; zero for d ≥ R.
    function zbl_scales(d::Real, Zu::Int, Zv::Int, zbl)
        R = Float64(zbl.covalent_radii[Zu + 1] + zbl.covalent_radii[Zv + 1])
        d = Float64(d)
        d < R || return 0.0, 0.0
        ks = (3.2, 0.9423, 0.4028, 0.2016)
        a = Float64(zbl.a_prefactor) * 0.529 / (Zu^Float64(zbl.a_exp) + Zv^Float64(zbl.a_exp))
        φ = sum(Float64(zbl.c[i]) * exp(-ks[i] * d / a) for i in 1:4)
        dφ = -sum(ks[i] * Float64(zbl.c[i]) * exp(-ks[i] * d / a) / a for i in 1:4)
        K = 14.3996 * Zu * Zv
        v, dv = K / d * φ, K * (dφ / d - φ / d^2)
        p, t = Float64(zbl.p), d / R
        c1, c2, c3 = (p + 1) * (p + 2) / 2, p * (p + 2), p * (p + 1) / 2
        F = 1 + c1 * t^p + c2 * t^(p + 1) + c3 * t^(p + 2)
        dF = (c1 * p * t^(p - 1) + c2 * (p + 1) * t^p + c3 * (p + 2) * t^(p + 1)) / R
        return 0.5 * abs(v) * F, 0.5 * (abs(dv) * F + abs(v) * dF)
    end

    # The largest difference relative to the largest scale, in units of eps(T).
    function scalederr(gpu, cpu, scale)
        isempty(cpu) && return 0.0
        m = maximum(scale)
        δ = maximum(abs, gpu - cpu)
        return iszero(m) ? (iszero(δ) ? 0.0 : Inf) : Float64(δ / m / eps(eltype(cpu)))
    end

    export mpa0, upload_columns!, relerr, zbl_scales, scalederr
end

# The rounding bound of the tolerance policy, 10·eps(T)·N relative to the largest magnitude of each
# quantity, with N = 1280 the largest inner dimension of any matrix product in the model. Each
# stage is shorter than that: the edge vector takes 7 operations; the harmonics recurrence and its
# gradient at l ≤ 3 about 40 per entry; the radial features about 30, with 10 steps of the Bessel
# recurrence; the radial-MLP products have inner dimensions 10 and 64 forward and up to 1280 in the
# pullback; ZBL about 40 operations and a receiver sum over at most 86 edges. The polynomial
# cutoffs cancel near their end (the sum of the absolute values of their terms is about 72 for
# p = 5), and a backend's sqrt, exp, sincos and pow may round differently from Base's; the Agnesi
# power x^q has condition q·|log x| ≤ 1.7 on these edges (x = d/r0 ≤ 6.1, q = 0.9183).
# Measured on the CPU backend and on Metal: at most 103·eps(T) (the radial-feature derivatives).
# The ZBL energies, their derivatives and the receiver sums are compared relative to the largest
# magnitude of their uncancelled terms (`zbl_scales`) rather than of the result: every edge within
# the ZBL range of these structures lies at t = d/R between 0.88 and 1.0, where the envelope is small
# against its terms. Measured: at most 0.65·eps(T) of that scale.
@testitem "edge stages: each stage agrees with its CPU function on the reference structures" tags = [:gpu] setup = [GPUBackend, DeviceStructures, EdgeStages] begin
    using Random
    M = PureMLIP.Model
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    ck = mpa0()
    worst = Dict{String, Float64}()
    note!(name, err) = (worst[name] = max(get(worst, name, 0.0), err); err)
    for T in precisions
        m = MACEModel{T}(ck)
        dm = PureMLIP.device_model(m, backend; elements = [29])
        tol = 10 * 1280   # in units of eps(T)
        skin = 0.5
        nbasis = length(m.radial.bessel_weights)
        nshell = 0
        for s in structures(T)
            P, C, Z = s.positions, s.cell, s.atomic_numbers
            n = size(P, 2)
            nl = neighbor_list(P, C, Float64(m.r_max) + skin; extra_capacity = 3)
            E = nl.n_edges
            dnl = PureMLIP.device_neighbor_list(nl, backend)
            # An edge chunk of 512 splits the two 32-atom cells into several chunks.
            ws = PureMLIP.device_workspace(dm, n, dnl.capacity; edge_chunk = 512)
            dZ = todevice(Int32.(Z))
            snd, rcv = nl.senders[1:E], nl.receivers[1:E]

            # Edges.
            cws = MACEEnergyWorkspace{T}(m, n, E)
            M._edges!(cws, P, C, nl)
            r, d = cws.r[:, 1:E], cws.d[1:E]
            ext._edges!(ws, todevice(P), todevice(C), dnl)
            @test note!("r", relerr(Array(ws.r)[:, 1:E], r)) <= tol
            @test note!("d", relerr(Array(ws.d)[1:E], d)) <= tol
            upload_columns!(ws.r, r)
            upload_columns!(ws.d, d)

            # CPU stages over all edges.
            Y, dY = zeros(T, 16, E), zeros(T, 16, 3, E)
            feats, dfeats = zeros(T, nbasis, E), zeros(T, nbasis, E)
            zbl, dzbl, pair = zeros(T, E), zeros(T, E), zeros(T, n)
            zscale, dzscale, pscale = zeros(E), zeros(E), zeros(n)
            for e in 1:E
                PureMLIP.e3nn_spherical_harmonics_and_gradient!(view(Y, :, e), view(dY, :, :, e), r[:, e], 3)
                M.radial_embedding!(view(feats, :, e), view(dfeats, :, e), m.radial, d[e], Z[snd[e]], Z[rcv[e]])
                zbl[e], dzbl[e] = M.zbl_edge_energy(d[e], Z[snd[e]], Z[rcv[e]], m.zbl)
                pair[rcv[e]] += zbl[e]
                zscale[e], dzscale[e] = zbl_scales(d[e], Z[snd[e]], Z[rcv[e]], m.zbl)
                pscale[rcv[e]] += zscale[e]
            end
            mlp = map(1:2) do k
                H, W = size(m.mlp[k].W[1], 2), size(m.mlp[k].W[4], 2)
                out, pre, act = zeros(T, W, E), zeros(T, H, E, 3), zeros(T, H, E, 3)
                M.radial_mlp!(out, pre, act, m.mlp[k], feats)
                dout = randn(Xoshiro(k), T, W, E)
                df = zeros(T, nbasis, E)
                M.radial_mlp_pullback!(df, zeros(T, H, E, 2), m.mlp[k], dout, pre, act)
                (; H, W, out, pre, act, dout, df)
            end

            # Device stages, chunk by chunk, gathered by edge.
            gY, gdY, gfeats, gdfeats = similar(Y), similar(dY), similar(feats), similar(dfeats)
            gzbl, gdzbl = similar(zbl), similar(dzbl)
            gmlp = [(; out = similar(c.out), pre = similar(c.pre), act = similar(c.act), df = similar(c.df)) for c in mlp]
            chunks = ext._edge_chunks(dnl, dZ, Array(dnl.row_ptr), ws.edge_chunk)
            @test reduce(vcat, [collect(c.atoms) for c in chunks]) == 1:n
            @test reduce(vcat, [collect(c.edges) for c in chunks]; init = Int[]) == 1:E
            E > 512 && @test length(chunks) > 1
            for c in chunks
                nb = length(c.edges)
                ext._edge_features!(ws, dm, 1, c)
                gY[:, c.edges] = Array(ws.Y)[:, 1:nb]
                gdY[:, :, c.edges] = Array(ws.dY)[:, :, 1:nb]
                gfeats[:, c.edges] = Array(ws.feats)[:, 1:nb]
                gdfeats[:, c.edges] = Array(ws.dfeats)[:, 1:nb]
                ext._zbl_edges!(ws, dm, c)
                gzbl[c.edges] = Array(ws.zbl)[1:nb]
                gdzbl[c.edges] = Array(ws.dzbl)[1:nb]
                for k in 1:2
                    (; H, W) = mlp[k]
                    upload_columns!(ws.feats, feats[:, c.edges])
                    ext._radial_mlp!(ws, dm, k, c)
                    gmlp[k].out[:, c.edges] = Array(ws.tpw)[1:W, 1:nb]
                    gmlp[k].pre[:, c.edges, :] = Array(ws.pre)[1:H, 1:nb, :]
                    gmlp[k].act[:, c.edges, :] = Array(ws.act)[1:H, 1:nb, :]
                    upload_columns!(ws.pre, mlp[k].pre[:, c.edges, :])
                    upload_columns!(ws.act, mlp[k].act[:, c.edges, :])
                    upload_columns!(ws.dtpw, mlp[k].dout[:, c.edges])
                    ext._radial_mlp_pullback!(ws, dm, k, c)
                    gmlp[k].df[:, c.edges] = Array(ws.dfeats_chunk)[:, 1:nb]
                end
            end
            gpair = Array(ws.pair)[1:n]

            @test note!("Y", relerr(gY, Y)) <= tol
            @test note!("dY", relerr(gdY, dY)) <= tol
            @test note!("feats", relerr(gfeats, feats)) <= tol
            @test note!("dfeats", relerr(gdfeats, dfeats)) <= tol
            @test note!("zbl", scalederr(gzbl, zbl, zscale)) <= tol
            @test note!("dzbl", scalederr(gdzbl, dzbl, dzscale)) <= tol
            @test note!("pair", scalederr(gpair, pair, pscale)) <= tol
            for k in 1:2
                @test note!("mlp out $k", relerr(gmlp[k].out, mlp[k].out)) <= tol
                @test note!("mlp sigma $k", relerr(gmlp[k].pre, mlp[k].pre)) <= tol
                @test note!("mlp act $k", relerr(gmlp[k].act, mlp[k].act)) <= tol
                @test note!("mlp pullback $k", relerr(gmlp[k].df, mlp[k].df)) <= tol
            end

            # Edges in the skin shell, r_max ≤ d < r_max + skin, contribute exactly zero.
            shell = findall(>=(m.r_max), d)
            nshell += length(shell)
            @test all(iszero, gfeats[:, shell]) && all(iszero, gdfeats[:, shell])
            @test all(iszero, gzbl[shell]) && all(iszero, gdzbl[shell])
            @test all(k -> all(iszero, gmlp[k].out[:, shell]), 1:2)
        end
        # The three crystals have skin-shell edges; the sparse multi_element cell and the isolated atom have none.
        @test nshell > 0
        # A pair range R beyond r_max (no real element pair has one) is cut at r_max too.
        P = (; ext._zbl_params(m.zbl, m.r_max)..., r_max = T(2))
        @test ext._zbl_edge(T(2), 29, 29, T(2.64), P) === (zero(T), zero(T))
        @test ext._zbl_edge(T(1.9), 29, 29, T(2.64), P) !== (zero(T), zero(T))
        println("edge stages, $(GPUBackend.NAME), $T: largest error in eps($T): ", sort(collect(worst)))
        empty!(worst)
    end
end

# The tiny checkpoint's Bessel weights are not exact multiples of the first, so its features take a
# sincos per weight. Bound as in the item above: 10·eps(T)·1280 relative to the largest magnitude.
@testitem "edge stages: radial features with Bessel weights that are not multiples" tags = [:gpu] setup = [GPUBackend, DeviceStructures, EdgeStages] begin
    M = PureMLIP.Model
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    for T in precisions
        m = tiny(T)
        @test !m.radial.multiples
        dm = PureMLIP.device_model(m, backend; elements = [29])
        s = structures(T)[1]
        n = size(s.positions, 2)
        nl = neighbor_list(s.positions, s.cell, 6.5)
        E, nbasis = nl.n_edges, length(m.radial.bessel_weights)
        dnl = PureMLIP.device_neighbor_list(nl, backend)
        ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
        dZ = todevice(Int32.(s.atomic_numbers))
        ext._edges!(ws, todevice(s.positions), todevice(s.cell), dnl)
        d = Array(ws.d)[1:E]
        feats, dfeats = zeros(T, nbasis, E), zeros(T, nbasis, E)
        for e in 1:E
            M.radial_embedding!(view(feats, :, e), view(dfeats, :, e), m.radial, d[e], 29, 29)
        end
        (c,) = ext._edge_chunks(dnl, dZ, Array(dnl.row_ptr), ws.edge_chunk)
        ext._edge_features!(ws, dm, 2, c)
        @test relerr(Array(ws.feats)[:, 1:E], feats) <= 10 * 1280
        @test relerr(Array(ws.dfeats)[:, 1:E], dfeats) <= 10 * 1280
        println("radial features, tiny model, $(GPUBackend.NAME), $T: largest error in eps($T): ", relerr(Array(ws.feats)[:, 1:E], feats), ", ", relerr(Array(ws.dfeats)[:, 1:E], dfeats))
    end
end

@testitem "edge stages: views of device positions give the same edges" tags = [:gpu] setup = [GPUBackend, DeviceStructures] begin
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    for T in precisions
        m = tiny(T)
        dm = PureMLIP.device_model(m, backend; elements = [29])
        s = structures(T)[1]
        n = size(s.positions, 2)
        dnl = PureMLIP.device_neighbor_list(neighbor_list(s.positions, s.cell, 6.0), backend)
        ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
        ext._edges!(ws, todevice(s.positions), todevice(s.cell), dnl)
        r, d = Array(ws.r), Array(ws.d)
        wide = todevice(hcat(zeros(T, 3, 2), s.positions))
        ext._edges!(ws, view(wide, :, 3:(n + 2)), todevice(s.cell), dnl)
        @test Array(ws.r) == r && Array(ws.d) == d
    end
end

@testitem "edge stages reject inconsistent arguments before any launch" tags = [:gpu] setup = [GPUBackend, DeviceStructures, FakeDevice] begin
    using OffsetArrays
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    T = Float32
    m = tiny(T)
    dm = PureMLIP.device_model(m, backend; elements = [29])
    s = structures(T)[1]
    n = size(s.positions, 2)
    dnl = PureMLIP.device_neighbor_list(neighbor_list(s.positions, s.cell, 6.0; extra_capacity = 2), backend)
    dpos, dcell, dZ = todevice(s.positions), todevice(s.cell), todevice(Int32.(s.atomic_numbers))
    ws = PureMLIP.device_workspace(dm, n, dnl.capacity; edge_chunk = 256)
    @test_throws "gpu: the workspace holds $(dnl.capacity - 1) edges, the neighbor list up to $(dnl.capacity)" ext._edges!(
        PureMLIP.device_workspace(dm, n, dnl.capacity - 1), dpos, dcell, dnl
    )
    @test_throws "gpu: positions and cell must have the workspace's element type Float32, got Float16 and Float32" ext._edges!(ws, todevice(Float16.(s.positions)), dcell, dnl)
    @test_throws "gpu: positions have size (3, $(n - 1))" ext._edges!(ws, todevice(s.positions[:, 2:end]), dcell, dnl)
    @test_throws "gpu: the cell has size (3, 2)" ext._edges!(ws, dpos, todevice(s.cell[:, 1:2]), dnl)
    @test_throws "gpu: all arrays must be on the backend of the positions" ext._edges!(ws, OtherMatrix(s.positions), dcell, dnl)
    @test_throws "gpu: arrays must be 1-based" ext._edges!(ws, OffsetArray(dpos, 0:2, 1:n), dcell, dnl)

    row_ptr = Array(dnl.row_ptr)
    degree = row_ptr[2] - row_ptr[1]
    @test_throws "gpu: receiver 1 has $degree edges, more than the edge chunk $(degree - 1)" ext._edge_chunks(dnl, dZ, row_ptr, degree - 1)
    @test_throws "gpu: row_ptr has length $n, expected $(n + 1)" ext._edge_chunks(dnl, dZ, row_ptr[1:n], 256)
    @test_throws "gpu: $(n - 1) atomic numbers for $n atoms" ext._edge_chunks(dnl, dZ[1:(n - 1)], row_ptr, 256)
    chunks = ext._edge_chunks(dnl, dZ, row_ptr, ws.edge_chunk)
    small = PureMLIP.device_workspace(dm, n, dnl.capacity; edge_chunk = degree)
    @test_throws "gpu: the chunk has $(degree + 1) edges, the workspace holds $degree" ext._edge_features!(small, dm, 1, ext.EdgeChunk(dnl, dZ, 1:2, 1:(degree + 1)))
    beyond = ext.EdgeChunk(dnl, dZ, 1:n, (dnl.capacity):(dnl.capacity + 1))
    @test_throws "gpu: the chunk reaches edge $(dnl.capacity + 1), the workspace holds $(dnl.capacity)" ext._zbl_edges!(ws, dm, beyond)
    @test_throws "gpu: the chunk reaches atom $(n + 1), the workspace holds $n" ext._radial_mlp!(ws, dm, 1, ext.EdgeChunk(dnl, dZ, 1:(n + 1), 1:1))
    for f in (ext._edge_features!, ext._radial_mlp!, ext._radial_mlp_pullback!)
        @test_throws "gpu: the interaction must be 1 or 2, got 3" f(ws, dm, 3, chunks[1])
    end
    zbl = (; c = m.zbl.c, p = 5.5, a_exp = m.zbl.a_exp, a_prefactor = m.zbl.a_prefactor)
    @test_throws "gpu: the ZBL envelope exponent must be an integer, got 5.5" ext._zbl_params(zbl, T(6))
    @test_throws "gpu: the ZBL envelope exponent must be at least 1, got 0" ext._zbl_params((; zbl..., p = 0.0), T(6))
    @test_throws "gpu: the cutoff exponent must be at least 1, got 0" ext._radial_params((; m.radial.r_max, cutoff_p = 0))
end

# The device code computes in the model's element type: no compiled function of the extension
# converts Float32 data to Float64. The kernels compile through the backend's own compiler (SPIR-V
# on the KernelAbstractions CPU backend), which the conversion sweep does not read, so the item
# also calls the `@inline` functions that hold each kernel's arithmetic as host code in Float32.
# On Metal a kernel that used Float64 would not compile at all.
@testitem "edge stages: Float32 stages do not widen to Float64" tags = [:gpu] setup = [GPUBackend, DeviceStructures] begin
    using StrictMode
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    T = Float32
    m = tiny(T)
    dm = PureMLIP.device_model(m, backend; elements = [29])
    s = structures(T)[1]
    n = size(s.positions, 2)
    nl = neighbor_list(s.positions, s.cell, 6.5)
    dnl = PureMLIP.device_neighbor_list(nl, backend)
    ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
    dZ = todevice(Int32.(s.atomic_numbers))
    ext._edges!(ws, todevice(s.positions), todevice(s.cell), dnl)
    for c in ext._edge_chunks(dnl, dZ, Array(dnl.row_ptr), ws.edge_chunk)
        ext._edge_features!(ws, dm, 1, c)
        ext._zbl_edges!(ws, dm, c)
        for k in 1:2
            ext._radial_mlp!(ws, dm, k, c)
            ext._radial_mlp_pullback!(ws, dm, k, c)
        end
    end

    # The kernels' arithmetic as host code, on the first edge and its receiver.
    E, nb, H = nl.n_edges, length(m.radial.bessel_weights), size(m.mlp[1].W[1], 2)
    r, d = zeros(T, 3, E), zeros(T, E)
    for e in 1:E
        ext._edge!(r, d, e, s.positions, s.cell, nl.senders[e], nl.receivers[e], nl.shifts)
    end
    ext._e3nn_harmonics_column!(zeros(T, 16, 1), zeros(T, 16, 3, 1), 1, 3, r[1, 1], r[2, 1], r[3, 1])
    Z = Int32.(s.atomic_numbers)
    for radial in (ext._radial_params(m.radial), (; ext._radial_params(m.radial)..., multiples = true))
        ext._radial_column!(zeros(T, nb, 1), zeros(T, nb, 1), 1, d[1], m.radial.covalent_radii[30], m.radial.covalent_radii[30], m.radial.bessel_weights, radial)
    end
    pre, act, g = randn(T, H, 1, 3), zeros(T, H, 1, 3), randn(T, H, 1, 2)
    ext._silu!(act, pre, 1, 1, 1, m.mlp[1].cst)
    ext._slope!(g, 1, pre, act, 1, 1, 1, m.mlp[1].cst)
    row_ptr = Array(dnl.row_ptr)
    zbl, dzbl, pair = zeros(T, E), zeros(T, E), zeros(T, n)
    ext._zbl_receiver!(zbl, dzbl, pair, d, nl.senders, row_ptr, Z, m.zbl.covalent_radii, ext._zbl_params(m.zbl, m.r_max), 1, 0)
    @test any(!iszero, zbl)
    @test isempty(StrictMode.conversion_report(ext; from = Float32, to = Float64))
end
