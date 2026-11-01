@testitem "energies match LAMMPS-MACE" setup = [LammpsReference, Checkpoints] begin
    for (ref, src) in ((:mpa0_f64, :mpa0), (:mp0b3_f64, :mp0b3))
        model = MACEModel{Float64}(load_mace_checkpoint(Checkpoints.fetch(src)))
        for s in LammpsReference.structures(LammpsReference.load(ref))
            nl = neighbor_list(s.positions, s.cell, model.r_max)
            ws = MACEWorkspace{Float64}(model, size(s.positions, 2), nl.n_edges)
            res = mace_energy!(MACEResult{Float64}(size(s.positions, 2)), ws, model, s.positions, s.cell, s.atomic_numbers, nl)
            # Target from the design (section 9): 1e-6 eV per structure against LAMMPS-MACE.
            @test abs(res.energy - s.energy) <= 1.0e-6
            @test maximum(abs, res.node_energies - s.node_energies) <= 1.0e-6
        end
    end
end

@testitem "MPA-0 model contents and Float32" setup = [LammpsReference, Checkpoints] begin
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); m = MACEModel(ck)
    @test m.scale == 0.7735790334431056 && m.shift == 0.0 && m.r_max == 6.0 && length(m.z_table) == 89
    @test MACEModel(load_mace_checkpoint(Checkpoints.fetch(:mp0b3))).scale == 0.8140724313304148
    s = LammpsReference.structure(LammpsReference.load(:mpa0_f64), "isolated_cu")
    nl = neighbor_list(s.positions, s.cell, 6.0)
    r = mace_energy!(MACEResult{Float64}(1), MACEWorkspace{Float64}(m, 1, 0), m, s.positions, s.cell, s.atomic_numbers, nl)
    @test r.energy == m.e0[findfirst(==(29), m.z_table)]   # no edges: E0 exactly
    m32 = MACEModel{Float32}(ck)
    # Measured margin: the largest |E - E_LAMMPS| over the five structures is 1.55e-5 eV
    # (crconi_alloy, -235 eV, where one Float32 ulp is 1.5e-5 eV); tol32 is 16 times that.
    tol32 = 2.5e-4
    for s in LammpsReference.structures(LammpsReference.load(:mpa0_f32))
        P, C = Float32.(s.positions), Float32.(s.cell)
        local nl = neighbor_list(P, C, 6.0)
        local r = mace_energy!(MACEResult{Float32}(size(P, 2)), MACEWorkspace{Float32}(m32, size(P, 2), nl.n_edges), m32, P, C, s.atomic_numbers, nl)
        @test abs(r.energy - s.energy) <= tol32
    end
end

@testitem "model errors" setup = [Checkpoints] begin
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); m = MACEModel(ck); h = ck.hyperparameters
    bad = MACECheckpoint(
        MACEHyperparameters(
            h.model_class, h.r_max, h.num_interactions, h.atomic_numbers, h.heads,
            h.interactions, [2, 3], h.readouts
        ), ck.state, ck.constants
    )
    @test_throws "model: unsupported checkpoint: correlations [2, 3], expected [3, 3]" MACEModel(bad)
    P = [0.0 2.5; 0.0 0.0; 0.0 0.0]; cell = [10.0 0 0; 0 10.0 0; 0 0 10.0]
    nl = neighbor_list(P, cell, 6.0); ws = MACEWorkspace{Float64}(m, 2, nl.n_edges)
    @test_throws "model: element 84 is not in the model's element table" mace_energy!(MACEResult{Float64}(2), ws, m, P, cell, [29, 84], nl)
    @test_throws "model: the workspace holds 1 edges" mace_energy!(MACEResult{Float64}(2), MACEWorkspace{Float64}(m, 2, 1), m, P, cell, [29, 29], nl)

    # Further architecture checks, each on a copy of the checkpoint with one entry changed.
    k = ck.constants
    with_h(; kw...) = MACECheckpoint(
        MACEHyperparameters(
            h.model_class, h.r_max, h.num_interactions, h.atomic_numbers, get(kw, :heads, h.heads),
            get(kw, :interactions, h.interactions), h.correlations, get(kw, :readouts, h.readouts)
        ), ck.state, ck.constants
    )
    with_k(; kw...) = MACECheckpoint(
        h, ck.state, MACEConstants(
            get(kw, :normalize2mom, k.normalize2mom), get(kw, :linear, k.linear), k.tensor_product,
            get(kw, :pair_repulsion, k.pair_repulsion), get(kw, :distance_transform, k.distance_transform),
            get(kw, :product_use_sc, k.product_use_sc)
        )
    )
    bad_class = MACECheckpoint(
        MACEHyperparameters(
            "mace.modules.models.MACE", h.r_max, h.num_interactions, h.atomic_numbers, h.heads,
            h.interactions, h.correlations, h.readouts
        ), ck.state, ck.constants
    )
    @test_throws "model: unsupported checkpoint: model class \"mace.modules.models.MACE\", expected \"mace.modules.models.ScaleShiftMACE\"" MACEModel(bad_class)
    @test_throws "model: unsupported checkpoint: heads [\"a\", \"b\"], expected one head" MACEModel(with_h(heads = ["a", "b"]))
    @test_throws "model: unsupported checkpoint: interaction classes" MACEModel(with_h(interactions = h.interactions[1:1]))
    @test_throws "model: unsupported checkpoint: readout classes" MACEModel(with_h(readouts = reverse(h.readouts)))
    I = PureMLIP.InteractionSpec
    i1 = h.interactions[1]
    hid = I(i1.class, i1.avg_num_neighbors, i1.radial_MLP, i1.node_feats_irreps, i1.edge_attrs_irreps, i1.edge_feats_irreps, i1.target_irreps, i1.hidden_irreps[1:1])
    @test_throws "model: unsupported checkpoint: hidden irreps" MACEModel(with_h(interactions = [hid, h.interactions[2]]))
    att = I(i1.class, i1.avg_num_neighbors, i1.radial_MLP, i1.node_feats_irreps, i1.edge_attrs_irreps[1:3], i1.edge_feats_irreps, i1.target_irreps, i1.hidden_irreps)
    @test_throws "model: unsupported checkpoint: edge attributes" MACEModel(with_h(interactions = [att, h.interactions[2]]))
    @test_throws "model: unsupported checkpoint: pair_repulsion false, expected true" MACEModel(with_k(pair_repulsion = false))
    @test_throws "model: unsupported checkpoint: distance transform \"\", expected \"mace.modules.radial.AgnesiTransform\"" MACEModel(with_k(distance_transform = ""))
    @test_throws "model: unsupported checkpoint: product use_sc Bool[1, 1], expected Bool[0, 1]" MACEModel(with_k(product_use_sc = [true, true]))
    n2m = copy(k.normalize2mom)
    n2m["readouts.1.non_linearity.acts.0"] = PureMLIP.Normalize2Mom(1.0, "torch.nn.functional.silu")
    @test_throws "model: unsupported checkpoint: normalize2mom constants" MACEModel(with_k(normalize2mom = n2m))
    n2m["readouts.1.non_linearity.acts.0"] = PureMLIP.Normalize2Mom(k.normalize2mom["readouts.1.non_linearity.acts.0"].cst, "torch.tanh")
    @test_throws "model: unsupported checkpoint: normalize2mom function \"torch.tanh\", expected \"torch.nn.functional.silu\"" MACEModel(with_k(normalize2mom = n2m))
    lin = copy(k.linear)
    s1 = lin["interactions.1.linear"]
    lin["interactions.1.linear"] = PureMLIP.LinearSpec(s1.irreps_in, s1.irreps_out, s1.instructions[1:3])
    @test_throws "model: unsupported checkpoint: interactions.1.linear input slots" MACEModel(with_k(linear = lin))
    lin["interactions.1.linear"] = PureMLIP.LinearSpec(
        s1.irreps_in, s1.irreps_out,
        [PureMLIP.LinearInstruction(0, 0, [128, 128], 0.0625); s1.instructions[2:end]]
    )
    @test_throws "model: unsupported checkpoint: interactions.1.linear instruction path shape [128, 128], expected [256, 128]" MACEModel(with_k(linear = lin))
    delete!(lin, "interactions.1.linear")
    @test_throws "model: unsupported checkpoint: module interactions.1.linear missing" MACEModel(with_k(linear = lin))

    # Run-time checks.
    nl5 = neighbor_list(P, cell, 5.0)
    @test_throws "model: the neighbor list cutoff 5.0 is below r_max 6.0" mace_energy!(MACEResult{Float64}(2), ws, m, P, cell, [29, 29], nl5)
    @test_throws "model: the workspace holds 1 atoms, the neighbor list has 2" mace_energy!(MACEResult{Float64}(2), MACEWorkspace{Float64}(m, 1, 2), m, P, cell, [29, 29], nl)
    @test_throws "model: 1 atomic numbers for 2 atoms" mace_energy!(MACEResult{Float64}(2), ws, m, P, cell, [29], nl)
    @test_throws "model: the result holds 3 atoms, the neighbor list has 2" mace_energy!(MACEResult{Float64}(3), ws, m, P, cell, [29, 29], nl)
    @test_throws "model: the cell has size (2, 3), expected (3, 3)" mace_energy!(MACEResult{Float64}(2), ws, m, P, cell[1:2, :], [29, 29], nl)
    @test_throws "model: positions have size (3, 3), the neighbor list has 2 atoms" mace_energy!(MACEResult{Float64}(2), ws, m, [P [1.0; 1.0; 1.0]], cell, [29, 29, 29], nl)
    @test_throws "model: element -1 is not in the model's element table" mace_energy!(MACEResult{Float64}(2), ws, m, P, cell, [29, -1], nl)
    @test_throws "model: element 1000 is not in the model's element table" mace_energy!(MACEResult{Float64}(2), ws, m, P, cell, [29, 1000], nl)
    @test_throws "model: the edge chunk must be positive, got 0" MACEWorkspace{Float64}(m, 2, 2; edge_chunk = 0)
    @test_throws "model: workspace sizes must not be negative" MACEWorkspace{Float64}(m, -1, 2)
    @test_throws "model: the number of atoms must not be negative" MACEResult{Float64}(-1)
end

@testitem "mace_energy! accepts views and rejects OffsetArrays" setup = [LammpsReference, Checkpoints] begin
    using OffsetArrays
    m = MACEModel(load_mace_checkpoint(Checkpoints.fetch(:mp0b3)))
    s = LammpsReference.structure(LammpsReference.load(:mp0b3_f64), "triclinic_ni3al")
    n = size(s.positions, 2)
    nl = neighbor_list(s.positions, s.cell, 6.0)
    ws = MACEWorkspace{Float64}(m, n, nl.n_edges; edge_chunk = 50)
    r = mace_energy!(MACEResult{Float64}(n), ws, m, s.positions, s.cell, s.atomic_numbers, nl)
    Pb = zeros(5, n + 2); Pb[2:4, 2:(n + 1)] = s.positions
    Cb = zeros(4, 4); Cb[2:4, 1:3] = s.cell
    Zb = [0; s.atomic_numbers; 0]
    rv = mace_energy!(
        MACEResult{Float64}(n), ws, m, view(Pb, 2:4, 2:(n + 1)), view(Cb, 2:4, 1:3),
        view(Zb, 2:(n + 1)), nl
    )
    # Same inputs, same arithmetic: exact.
    @test rv.energy == r.energy && rv.node_energies == r.node_energies
    # A workspace larger than the system and a different edge chunk give the same energies up
    # to the summation order of the radial-MLP products. Rounding bound: 64·eps·Σ|E_i|.
    big = MACEWorkspace{Float64}(m, n + 3, nl.n_edges + 7)
    rb = mace_energy!(MACEResult{Float64}(n), big, m, s.positions, s.cell, s.atomic_numbers, nl)
    @test abs(rb.energy - r.energy) <= 64 * eps() * sum(abs, r.node_energies)
    # The message comes from Base.require_one_based_indexing, so only the type is checked.
    @test_throws ArgumentError mace_energy!(
        MACEResult{Float64}(n), ws, m, OffsetArray(s.positions, 0:2, 0:(n - 1)), s.cell, s.atomic_numbers, nl
    )
end

@testitem "model builds MPA-0's interactions.1.linear as ten slots" setup = [Checkpoints] begin
    using Random
    Tn = PureMLIP.Tensors
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); m = MACEModel(ck)
    lin = m.linear[2]
    a, b = 0.0625, 0.05103103630798288
    pws = [a, a, b, b, b, b, b, b, a, a]
    out_slot = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
    @test lin.pw == pws && lin.slots == [(p, out_slot[p]) for p in 1:10]
    @test MACEModel(load_mace_checkpoint(Checkpoints.fetch(:mp0b3))).linear[2].pw == pws
    # mace blocks.py, interactions.1.linear: out_l = pw_l · Σ over the slots of degree l of Σ_u M_slot[u, m] W_slot[u, w], each
    # slot its own consecutive [128, 128] block of interactions.1.linear.weight.
    W = ck.state.f64["interactions.1.linear.weight"].data
    in_ls = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    off = cumsum([0; [2l + 1 for l in in_ls]])
    deg(q) = isqrt(q - 1)
    x = randn(Xoshiro(7), 128, 40, 2)
    y = zeros(128, 16, 2); Tn.linear!(y, lin, x)
    naive = [
        sum(
            pws[p] * sum(x[u, off[p] + q - deg(q)^2, i] * W[((p - 1) * 128 + u - 1) * 128 + w] for u in 1:128)
                for p in 1:10 if out_slot[p] == deg(q) + 1
        ) for w in 1:128, q in 1:16, i in 1:2
    ]
    # Rounding bound: 4·eps·N·max|terms|, N = 384 terms per output (three slots of 128).
    @test y ≈ naive atol = 4 * eps() * 384 * maximum(abs, x) * maximum(abs, W)
end

@testitem "checked_eachindex matches eachindex and reports mismatches" begin
    E = PureMLIP.Tensors.checked_eachindex
    a, b, c = zeros(4), ones(4), zeros(4)
    @test E(a, b, c) === eachindex(a, b, c)
    M = zeros(2, 3)
    V = view(zeros(4, 6), 1:2, 1:3)
    @test E(M, V, M) === eachindex(M, V, M)
    @test E(M, V) === eachindex(M, V)
    @test E(zeros(2, 3), zeros(6)) === eachindex(zeros(2, 3), zeros(6))
    @test E(a) === eachindex(a)
    msg = "all inputs to eachindex must have the same indices, got Base.OneTo(5) and Base.OneTo(4)"
    @test_throws msg E(zeros(5), zeros(4))
    @test_throws msg E(zeros(5), zeros(5), zeros(4))
    @test_throws msg E(zeros(5), zeros(4), zeros(5))
end

# The energy workspace runs the forward pass of `mace_energy_forces!` with the per-atom layers of
# both interactions fused per block of receivers. The oracle is `mace_energy_forces!` on a
# MACEWorkspace; both compute the same energies, the energy workspace with its matrix products
# split into blocks of receivers.
@testitem "energy workspace: energies equal mace_energy_forces!" setup = [LammpsReference, Checkpoints] begin
    for (src, refs) in ((:mpa0, (:mpa0_f64,)), (:mp0b3, (:mp0b3_f64,)))
        ck = load_mace_checkpoint(Checkpoints.fetch(src))
        for T in (Float64, Float32)
            m = MACEModel{T}(ck)
            structures = [s for ref in refs for s in LammpsReference.structures(LammpsReference.load(ref))]
            lists = [neighbor_list(T.(s.positions), T.(s.cell), m.r_max) for s in structures]
            # Rounding bound 10·eps(T)·N relative to the largest magnitude, with N = 1280 the
            # largest inner dimension of any matrix product in the model.
            tol = 10 * eps(T) * 1280
            nmax, emax = maximum(nl -> nl.n_atoms, lists), maximum(nl -> nl.n_edges, lists)
            # One workspace for every structure, with blocks of 2 receivers and chunks of 37 edges.
            shared = MACEEnergyWorkspace{T}(m, nmax, emax; edge_chunk = 37, block = 2)
            for (s, nl) in zip(structures, lists)
                P, C, Z = T.(s.positions), T.(s.cell), s.atomic_numbers
                n = nl.n_atoms
                ref = mace_energy_forces!(MACEResult{T}(n), MACEWorkspace{T}(m, n, nl.n_edges), m, P, C, Z, nl)
                for ws in (MACEEnergyWorkspace{T}(m, n, nl.n_edges), shared)
                    # Every buffer is filled with NaN first, so a read before a write within the
                    # call shows as a non-finite energy.
                    for f in fieldnames(typeof(ws))
                        v = getfield(ws, f)
                        for a in (v isa Tuple ? v : (v,))
                            a isa AbstractArray{T} && fill!(a, T(NaN))
                        end
                    end
                    res = MACEResult{T}(n)
                    fill!(res.forces, T(7)); fill!(res.virial, T(7)); fill!(res.stress, T(7))
                    @test mace_energy!(res, ws, m, P, C, Z, nl) === res
                    @test abs(res.energy - ref.energy) <= tol * abs(ref.energy)
                    @test maximum(abs, res.node_energies - ref.node_energies) <= tol * maximum(abs, ref.node_energies)
                    # The forces, virial and stress of the result are left as they were.
                    @test all(==(T(7)), res.forces) && all(==(T(7)), res.virial) && all(==(T(7)), res.stress)
                end
            end
        end
    end
end

@testitem "energy workspace: one atom and no edges give E0" setup = [LammpsReference, Checkpoints] begin
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0))
    for T in (Float64, Float32)
        m = MACEModel{T}(ck)
        s = LammpsReference.structure(LammpsReference.load(:mpa0_f64), "isolated_cu")
        P, C = T.(s.positions), T.(s.cell)
        nl = neighbor_list(P, C, m.r_max)
        @test nl.n_atoms == 1 && nl.n_edges == 0
        for block in (1, 256)
            r = mace_energy!(MACEResult{T}(1), MACEEnergyWorkspace{T}(m, 1, 0; block), m, P, C, s.atomic_numbers, nl)
            @test r.energy == m.e0[findfirst(==(29), m.z_table)]   # no edges: E0 exactly
            @test r.node_energies == [r.energy]
        end
    end
end

@testitem "energy workspace memory per atom" setup = [Checkpoints, WorkspaceMemory] begin
    m64 = MACEModel(load_mace_checkpoint(Checkpoints.fetch(:mpa0)))
    # cu_bench(14) (test/reference/structures.jl) under mpa-0: 10,976 atoms, 856,128 directed
    # edges within r_max.
    b64 = WorkspaceMemory.workspace_bytes_per_atom(m64, Float64, 10_976, 856_128; workspace = MACEEnergyWorkspace, edge_chunk = 4096)
    @test b64 <= 30_000
    GC.gc()
end

@testitem "energy workspace: views, OffsetArrays and errors" setup = [LammpsReference, Checkpoints] begin
    using OffsetArrays
    m = MACEModel(load_mace_checkpoint(Checkpoints.fetch(:mp0b3)))
    s = LammpsReference.structure(LammpsReference.load(:mp0b3_f64), "triclinic_ni3al")
    n = size(s.positions, 2)
    nl = neighbor_list(s.positions, s.cell, 6.0)
    ws = MACEEnergyWorkspace{Float64}(m, n, nl.n_edges; edge_chunk = 50)
    r = mace_energy!(MACEResult{Float64}(n), ws, m, s.positions, s.cell, s.atomic_numbers, nl)
    Pb = zeros(5, n + 2); Pb[2:4, 2:(n + 1)] = s.positions
    Cb = zeros(4, 4); Cb[2:4, 1:3] = s.cell
    Zb = [0; s.atomic_numbers; 0]
    rv = mace_energy!(
        MACEResult{Float64}(n), ws, m, view(Pb, 2:4, 2:(n + 1)), view(Cb, 2:4, 1:3), view(Zb, 2:(n + 1)), nl
    )
    # Same inputs, same arithmetic: exact.
    @test rv.energy == r.energy && rv.node_energies == r.node_energies
    # The message comes from Base.require_one_based_indexing, so only the type is checked.
    @test_throws ArgumentError mace_energy!(
        MACEResult{Float64}(n), ws, m, OffsetArray(s.positions, 0:2, 0:(n - 1)), s.cell, s.atomic_numbers, nl
    )
    @test_throws "model: the workspace holds 1 atoms, the neighbor list has $n" mace_energy!(
        MACEResult{Float64}(n), MACEEnergyWorkspace{Float64}(m, 1, nl.n_edges), m, s.positions, s.cell, s.atomic_numbers, nl
    )
    @test_throws "model: the workspace holds 1 edges" mace_energy!(
        MACEResult{Float64}(n), MACEEnergyWorkspace{Float64}(m, n, 1), m, s.positions, s.cell, s.atomic_numbers, nl
    )
    @test_throws "model: element 84 is not in the model's element table" mace_energy!(
        MACEResult{Float64}(n), ws, m, s.positions, s.cell, fill(84, n), nl
    )
    bad = deepcopy(nl)
    E = nl.n_edges
    for v in (bad.senders, bad.receivers)
        v[1], v[E] = v[E], v[1]
    end
    bad.shifts[:, [1, E]] = bad.shifts[:, [E, 1]]
    @test_throws "model: neighbor list edges must be sorted by receiver" mace_energy!(
        MACEResult{Float64}(n), ws, m, s.positions, s.cell, s.atomic_numbers, bad
    )
    @test_throws "model: workspace sizes must not be negative, got -1 atoms and 2 edges" MACEEnergyWorkspace{Float64}(m, -1, 2)
    @test_throws "model: the edge chunk must be positive, got 0" MACEEnergyWorkspace{Float64}(m, 2, 2; edge_chunk = 0)
    @test_throws "model: the receiver block must be positive, got 0" MACEEnergyWorkspace{Float64}(m, 2, 2; block = 0)
end

# The energy workspace uses the radial MLP without its pullback buffers and the radial features
# without their derivatives; the arithmetic of each value is that of the derivative form, so the
# values are equal exactly.
@testitem "radial MLP and features without derivatives equal the derivative forms" setup = [Checkpoints] begin
    using Random
    M = PureMLIP.Model
    m = MACEModel(load_mace_checkpoint(Checkpoints.fetch(:mpa0)))
    mlp = m.mlp[2]
    nf, H, W = size(mlp.W[1], 1), size(mlp.W[1], 2), size(mlp.W[4], 2)
    feats = randn(Xoshiro(3), nf, 9)
    out5 = M.radial_mlp!(zeros(W, 9), zeros(H, 9, 3), zeros(H, 9, 3), mlp, feats)
    out4 = zeros(W, 9)
    @test M.radial_mlp!(out4, zeros(H, 9, 2), mlp, feats) === out4
    @test out4 == out5
    @test_throws "model: radial MLP activations have size ($H, 9, 3), expected ($H, 9, 2)" M.radial_mlp!(out4, zeros(H, 9, 3), mlp, feats)
    @test_throws "model: radial MLP features have $(nf + 1) rows, expected $nf" M.radial_mlp!(out4, zeros(H, 9, 2), mlp, zeros(nf + 1, 9))
    @test_throws "model: radial MLP output has size ($W, 8), expected ($W, 9)" M.radial_mlp!(zeros(W, 8), zeros(H, 9, 2), mlp, feats)

    emb = m.radial
    @test emb.multiples
    for d in (0.9, 2.5, 5.9, 6.0, 6.5)       # r_max = 6.0: the last two give zero features
        f, df = zeros(nf), zeros(nf)
        M.radial_embedding!(f, df, emb, d, 29, 8)
        @test M.radial_embedding!(zeros(nf), nothing, emb, d, 29, 8) == f
    end
    w = [1.0, 2.5, 3.25]                       # not multiples of the first: the per-term form
    B, dB = zeros(3), zeros(3)
    M.bessel_basis!(B, dB, 0.7, w, 1.3)
    @test M.bessel_basis!(zeros(3), nothing, 0.7, w, 1.3) == B
    @test_throws "model: B and weights must share axes: (Base.OneTo(2),), (Base.OneTo(3),)" M.bessel_basis!(zeros(2), nothing, 0.7, w, 1.3)
end
