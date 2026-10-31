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
            ]; guarantees = (:typestable,)
        )
    end
end
