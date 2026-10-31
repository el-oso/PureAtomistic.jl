@testitem "StrictMode: public API is type-stable" tags = [:checks] begin
    using StrictModeTest, StrictMode
    if !StrictMode.checks_enabled()
        @info "StrictMode checks disabled; enable them in test/Project.toml to run"
        @test_skip StrictMode.checks_enabled()
    else
        P = PureMLIP
        test_signatures(
            [
                (P.load_mace_checkpoint, (Vector{UInt8},)),
                (P.load_mace_checkpoint, (String,)),
                (P.open_torch_archive, (Vector{UInt8},)),
                (P.unpickle, (Vector{UInt8},)),
                (P.torch_array, (P.CheckpointTensor{Float64}, Val{3})),
                (P.wigner_D, (Int, Matrix{Float64})),
                (P.real_clebsch_gordan, (Int, Int, Int)),
                (P.clebsch_gordan_complex, (Int, Int, Int, Int, Int, Int)),
                (P.spherical_harmonics!, (Vector{Float64}, Vector{Float64}, Int)),
                (P.spherical_harmonics_and_gradient!, (Vector{Float64}, Matrix{Float64}, Vector{Float64}, Int)),
                (P.e3nn_spherical_harmonics_and_gradient!, (Vector{Float64}, Matrix{Float64}, Vector{Float64}, Int)),
                (P.e3nn_wigner_3j, (Int, Int, Int)),
                (P.e3nn_spherical_harmonics!, (Vector{Float64}, Vector{Float64}, Int)),
                (P.e3nn_basis_change, (Int,)),
                (P.sh_length, (Int,)),
                (P.sh_index, (Int, Int)),
                (P.embedded_archives, (P.PickleTree,)),
                (P.entry_data, (P.TorchArchive, String)),
                (P.neighbor_list, (Matrix{Float64}, Matrix{Float64}, Float64)),
                (P.neighbor_list!, (P.NeighborList{Float64}, Matrix{Float64}, Matrix{Float64})),
                (P.MACEModel{Float64}, (P.MACECheckpoint,)),
                (P.MACEModel{Float32}, (P.MACECheckpoint,)),
                (P.MACEResult{Float64}, (Int,)),
                (P.MACEWorkspace{Float64}, (P.MACEModel{Float64}, Int, Int)),
                (
                    P.mace_energy!, (
                        P.MACEResult{Float64}, P.MACEWorkspace{Float64}, P.MACEModel{Float64},
                        Matrix{Float64}, Matrix{Float64}, Vector{Int}, P.NeighborList{Float64},
                    ),
                ),
                (
                    P.mace_energy!, (
                        P.MACEResult{Float32}, P.MACEWorkspace{Float32}, P.MACEModel{Float32},
                        Matrix{Float32}, Matrix{Float32}, Vector{Int}, P.NeighborList{Float32},
                    ),
                ),
                (P.Model.zbl_edge_energy, (Float64, Int, Int, P.Model.ZBLParams{Float64})),
                (P.Model.zbl_edge_energy, (Float32, Int, Int, P.Model.ZBLParams{Float32})),
            ]; guarantees = (:typestable,)
        )
    end
    # `:typestable` accepts a small union of isbits returns; the radial and ZBL kernels must
    # return exactly their working type, so that a Float32 model computes in Float32.
    M = PureMLIP.Model
    for S in (Float32, Float64)
        @test Base.return_types(M.zbl_edge_energy, (S, Int, Int, M.ZBLParams{S})) == [Tuple{S, S}]
        @test Base.return_types(M.radial_embedding!, (Vector{S}, Vector{S}, M.RadialEmbedding{S}, S, Int, Int)) == [Vector{S}]
    end
end

@testitem "StrictMode: spherical harmonics are allocation-free" tags = [:checks] begin
    using StrictModeTest, StrictMode
    Y = zeros(25); dY = zeros(25, 3); v = [0.1, 0.2, 0.9]
    @noinline f!(Y, v) = PureMLIP.spherical_harmonics!(Y, v, 4)
    @noinline g!(Y, dY, v) = PureMLIP.e3nn_spherical_harmonics_and_gradient!(Y, dY, v, 4)
    f!(Y, v); g!(Y, dY, v)
    @test_noalloc f!(Y, v)
    @test_noalloc g!(Y, dY, v)
    @noinline h!(Y, dY, v) = PureMLIP.spherical_harmonics_and_gradient!(Y, dY, v, 4)
    h!(Y, dY, v)
    @test_noalloc h!(Y, dY, v)
end

@testitem "StrictMode: neighbor_list! is allocation-free" tags = [:checks] begin
    using StrictModeTest, StrictMode
    P = [0.0 1.5 0.2; 0.0 0.0 2.0; 0.0 0.0 1.0]; C = [6.0 0.0 0.0; 0.0 6.0 0.0; 0.0 0.0 6.0]
    nl = PureMLIP.neighbor_list(P, C, 4.0)
    @noinline f!(nl, P, C) = PureMLIP.neighbor_list!(nl, P, C)
    f!(nl, P, C)
    @test_noalloc f!(nl, P, C)
end

@testitem "StrictMode: mace_energy! is allocation-free" tags = [:checks] setup = [Checkpoints] begin
    using StrictModeTest, StrictMode
    m = MACEModel(load_mace_checkpoint(Checkpoints.fetch(:mp0b3)))
    P = [0.0 2.5 1.2; 0.0 0.0 1.9; 0.0 0.0 0.3]; C = [7.0 0.0 0.0; 0.0 7.0 0.0; 0.0 0.0 7.0]; Z = [29, 29, 8]
    nl = neighbor_list(P, C, 6.0)
    # A chunk smaller than the edge count exercises the chunked radial MLP.
    ws = MACEWorkspace{Float64}(m, 3, nl.n_edges; edge_chunk = 7)
    res = MACEResult{Float64}(3)
    @noinline f!(res, ws, m, P, C, Z, nl) = mace_energy!(res, ws, m, P, C, Z, nl)
    f!(res, ws, m, P, C, Z, nl)
    @test nl.n_edges > 7
    @test_noalloc f!(res, ws, m, P, C, Z, nl)
    m32 = MACEModel{Float32}(load_mace_checkpoint(Checkpoints.fetch(:mp0b3)))
    P32, C32 = Float32.(P), Float32.(C)
    nl32 = neighbor_list(P32, C32, 6.0)
    ws32 = MACEWorkspace{Float32}(m32, 3, nl32.n_edges; edge_chunk = 7)
    res32 = MACEResult{Float32}(3)
    f!(res32, ws32, m32, P32, C32, Z, nl32)
    @test_noalloc f!(res32, ws32, m32, P32, C32, Z, nl32)
end
