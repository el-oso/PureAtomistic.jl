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
            ]; guarantees = (:typestable,)
        )
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
end
