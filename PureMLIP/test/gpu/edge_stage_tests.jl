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

    # The largest difference of an entry relative to its own scale, in units of eps(T); an entry
    # whose scale is zero must agree exactly.
    function scalederr(gpu, cpu, scale)
        err = 0.0
        for i in eachindex(gpu, cpu, scale)
            δ = Float64(abs(gpu[i] - cpu[i]))
            err = max(err, iszero(scale[i]) ? (iszero(δ) ? 0.0 : Inf) : δ / scale[i] / eps(eltype(cpu)))
        end
        return err
    end

    # Prints the measured errors when PUREMLIP_GPU_VERBOSE=1.
    report(args...) = get(ENV, "PUREMLIP_GPU_VERBOSE", "") == "1" && println(args...)

    export mpa0, upload_columns!, relerr, zbl_scales, scalederr, report
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
# The ZBL energies and their derivatives are compared edge by edge with the magnitude of that
# edge's uncancelled terms (`zbl_scales`), and each receiver's sum with the sum of its edges'
# scales, rather than with the result: every edge within the ZBL range of these structures lies at
# t = d/R between 0.88 and 1.0, where the envelope is small against its terms (the item "ZBL at
# short range" checks the result itself). The bound is 10·eps(T)·N with N from the operations of
# `zbl_edge_energy`: the energy takes about 50, and the screening sum has condition at most 12.2
# with respect to ρ = d/a on these edges, multiplying the rounding of its four exponents (1 each)
# and of a (about 6: four operations and a power of condition a_exp·log Z ≤ 1.3), so
# N = 50 + 12·(1 + 6) ≈ 140; the derivative takes about 30 more operations, N = 170; a receiver's
# sum adds up to 86 terms, N = 140 + 86. Measured: at most 0.79·eps(T) of the edge's scale.
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
        zbl_tol, dzbl_tol, pair_tol = 10 * 140, 10 * 170, 10 * (140 + 86)
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
            # The chunks cover every slot of the list; slots past the E edges hold no edge and get
            # zero harmonics and features. A receiver's edges may lie in two chunks.
            chunks = ext._edge_chunks(dnl, dZ, ws.edge_chunk)
            @test reduce(vcat, [collect(c.edges) for c in chunks]; init = Int[]) == 1:(dnl.capacity)
            E > 512 && @test length(chunks) > 1
            fill!(ws.pair, zero(T))
            for c in chunks
                nb = length(c.edges)
                v = first(c.edges):min(last(c.edges), E)
                j = 1:length(v)
                ext._edge_features!(ws, dm, 1, c)
                gY[:, v] = Array(ws.Y)[:, j]
                gdY[:, :, v] = Array(ws.dY)[:, :, j]
                gfeats[:, v] = Array(ws.feats)[:, j]
                gdfeats[:, v] = Array(ws.dfeats)[:, j]
                past = (length(v) + 1):nb
                @test all(iszero, Array(ws.Y)[:, past]) && all(iszero, Array(ws.dY)[:, :, past])
                @test all(iszero, Array(ws.feats)[:, past]) && all(iszero, Array(ws.dfeats)[:, past])
                ext._zbl_edges!(ws, dm, c)
                gzbl[v] = Array(ws.zbl)[j]
                gdzbl[v] = Array(ws.dzbl)[j]
                for k in 1:2
                    (; H, W) = mlp[k]
                    upload_columns!(ws.feats, feats[:, v])
                    ext._radial_mlp!(ws, dm, k, c)
                    gmlp[k].out[:, v] = Array(ws.tpw)[1:W, j]
                    gmlp[k].pre[:, v, :] = Array(ws.pre)[1:H, j, :]
                    gmlp[k].act[:, v, :] = Array(ws.act)[1:H, j, :]
                    upload_columns!(ws.pre, mlp[k].pre[:, v, :])
                    upload_columns!(ws.act, mlp[k].act[:, v, :])
                    upload_columns!(ws.dtpw, mlp[k].dout[:, v])
                    ext._radial_mlp_pullback!(ws, dm, k, c)
                    gmlp[k].df[:, v] = Array(ws.dfeats_chunk)[:, j]
                end
            end
            gpair = Array(ws.pair)[1:n]

            @test note!("Y", relerr(gY, Y)) <= tol
            @test note!("dY", relerr(gdY, dY)) <= tol
            @test note!("feats", relerr(gfeats, feats)) <= tol
            @test note!("dfeats", relerr(gdfeats, dfeats)) <= tol
            @test note!("zbl", scalederr(gzbl, zbl, zscale)) <= zbl_tol
            @test note!("dzbl", scalederr(gdzbl, dzbl, dzscale)) <= dzbl_tol
            @test note!("pair", scalederr(gpair, pair, pscale)) <= pair_tol
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
        report("edge stages, $(GPUBackend.NAME), $T: largest error in eps($T): ", sort(collect(worst)))
        empty!(worst)
    end
end

# ZBL at short range, where the envelope is not small (t = d/R ≤ 0.75, f ≥ 0.2 for p = 5) and the
# result is of the size of its terms: a cluster of Cu with an H, a Cu and an O neighbor, an H–H
# dimer and a W–Ni dimer, in a cell large enough to keep the groups apart. Each edge energy, its
# derivative and each receiver's sum is compared with the CPU relative to its own magnitude, with
# the rounding bound of the item above, 10·eps(T)·1280.
@testitem "edge stages: ZBL at short range agrees with its CPU function" tags = [:gpu] setup = [GPUBackend, EdgeStages] begin
    M = PureMLIP.Model
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    ck = mpa0()
    for T in precisions
        m = MACEModel{T}(ck)
        dm = PureMLIP.device_model(m, backend; elements = [29])
        R(a, b) = Float64(m.zbl.covalent_radii[a + 1] + m.zbl.covalent_radii[b + 1])
        Z = [29, 1, 29, 8, 1, 1, 74, 28]
        P = [
            5 5 + 0.5 * R(1, 29) 5 5 15 15 + 0.4 * R(1, 1) 5 5 + 0.25 * R(74, 28)
            5 5 5 + 0.3 * R(29, 29) 5 5 5 15 15
            5 5 5 5 + 0.6 * R(8, 29) 5 5 5 5
        ]
        P, C = T.(P), T.(30 * [1 0 0; 0 1 0; 0 0 1])
        n = length(Z)
        nl = neighbor_list(P, C, Float64(m.r_max))
        E = nl.n_edges
        snd, rcv = nl.senders[1:E], nl.receivers[1:E]
        cws = MACEEnergyWorkspace{T}(m, n, E)
        M._edges!(cws, P, C, nl)
        d = cws.d[1:E]
        zbl, dzbl, pair = zeros(T, E), zeros(T, E), zeros(T, n)
        for e in 1:E
            zbl[e], dzbl[e] = M.zbl_edge_energy(d[e], Z[snd[e]], Z[rcv[e]], m.zbl)
            pair[rcv[e]] += zbl[e]
        end
        inrange = [d[e] < R(Z[snd[e]], Z[rcv[e]]) for e in 1:E]
        @test count(inrange) >= 10
        @test maximum(e -> d[e] / R(Z[snd[e]], Z[rcv[e]]), findall(inrange)) <= 0.75
        @test Set((Z[snd[e]], Z[rcv[e]]) for e in findall(inrange)) ⊇ Set([(29, 29), (1, 29), (29, 1), (8, 29), (1, 1), (74, 28)])

        dnl = PureMLIP.device_neighbor_list(nl, backend)
        ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
        upload_columns!(ws.d, d)
        dZ = todevice(Int32.(Z))
        gzbl, gdzbl = similar(zbl), similar(dzbl)
        fill!(ws.pair, zero(T))
        for c in ext._edge_chunks(dnl, dZ, ws.edge_chunk)
            ext._zbl_edges!(ws, dm, c)
            gzbl[c.edges] = Array(ws.zbl)[1:length(c.edges)]
            gdzbl[c.edges] = Array(ws.dzbl)[1:length(c.edges)]
        end
        gpair = Array(ws.pair)[1:n]
        tol = 10 * 1280
        own(x) = abs.(x)
        errs = (scalederr(gzbl, zbl, own(zbl)), scalederr(gdzbl, dzbl, own(dzbl)), scalederr(gpair, pair, own(pair)))
        @test all(<=(tol), errs)
        report("ZBL at short range, $(GPUBackend.NAME), $T: energy, derivative, sum in eps($T): ", errs)
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
        (c,) = ext._edge_chunks(dnl, dZ, ws.edge_chunk)
        ext._edge_features!(ws, dm, 2, c)
        @test relerr(Array(ws.feats)[:, 1:E], feats) <= 10 * 1280
        @test relerr(Array(ws.dfeats)[:, 1:E], dfeats) <= 10 * 1280
        report("radial features, tiny model, $(GPUBackend.NAME), $T: largest error in eps($T): ", relerr(Array(ws.feats)[:, 1:E], feats), ", ", relerr(Array(ws.dfeats)[:, 1:E], dfeats))
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

    @test_throws "gpu: the edge chunk must be positive, got 0" ext._edge_chunks(dnl, dZ, 0)
    @test_throws "gpu: $(n - 1) atomic numbers for $n atoms" ext._edge_chunks(dnl, dZ[1:(n - 1)], 256)
    @test_throws "gpu: arrays must be 1-based" ext._edge_chunks(dnl, OffsetArray(dZ, 0:(n - 1)), 256)
    chunks = ext._edge_chunks(dnl, dZ, ws.edge_chunk)
    small = PureMLIP.device_workspace(dm, n, dnl.capacity; edge_chunk = 9)
    @test_throws "gpu: the chunk has 10 edges, the workspace holds 9" ext._edge_features!(small, dm, 1, ext.EdgeChunk(dnl, dZ, 1:10))
    beyond = ext.EdgeChunk(dnl, dZ, (dnl.capacity):(dnl.capacity + 1))
    @test_throws "gpu: the chunk reaches edge $(dnl.capacity + 1), the workspace holds $(dnl.capacity) and the list $(dnl.capacity)" ext._zbl_edges!(ws, dm, beyond)
    fewer = PureMLIP.device_workspace(dm, n - 1, dnl.capacity)
    @test_throws "gpu: the list has $n atoms, the workspace holds $(n - 1)" ext._radial_mlp!(fewer, dm, 1, chunks[1])
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
    for c in ext._edge_chunks(dnl, dZ, ws.edge_chunk)
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
    ext._zbl_receiver!(zbl, dzbl, pair, d, nl.senders, row_ptr, Z, m.zbl.covalent_radii, ext._zbl_params(m.zbl, m.r_max), 1, 0, E)
    @test any(!iszero, zbl)
    @test isempty(StrictMode.conversion_report(ext; from = Float32, to = Float64))

    # The sweep reads each compiled function alone, and Base's power of a Float32 by an integer
    # widens inside a function it does not inline; so the integer powers of the polynomial
    # envelopes are checked over their whole call tree: no function reached from them converts to
    # Float64.
    val(x) = x isa GlobalRef ? getglobal(x.mod, x.name) : x
    function widens(tt::Type, seen = Set{Any}())
        tt in seen && return false
        push!(seen, tt)
        for (ci, _) in Base.code_typed_by_type(tt; optimize = true), st in ci.code
            st isa Expr || continue
            if st.head === :invoke
                target = st.args[1]
                mi = target isa Core.CodeInstance ? target.def : target
                widens(mi.specTypes, seen) && return true
            elseif st.head === :call && val(st.args[1]) in (Core.Intrinsics.fpext, Core.Intrinsics.sitofp, Core.Intrinsics.uitofp)
                val(st.args[2]) === Float64 && return true
            end
        end
        return false
    end
    @test !widens(Tuple{typeof(ext._poly_envelope_device), Float32, Int, Float32})
    # The check sees a widening that Base does not inline.
    @test widens(Tuple{typeof(^), Float32, Int})
end
