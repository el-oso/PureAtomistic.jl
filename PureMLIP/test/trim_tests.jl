@testitem "TrimCheck: checkpoint loading" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureMLIP
        end,
        PureMLIP.load_mace_checkpoint(Vector{UInt8}),
        PureMLIP.load_mace_checkpoint(String),
        PureMLIP.torch_array(PureMLIP.CheckpointTensor{Float64}, Val{3}),
        PureMLIP.torch_array(PureMLIP.CheckpointTensor{Int64}, Val{1}),
        PureMLIP.wigner_D(Int, Matrix{Float64}),
        PureMLIP.real_clebsch_gordan(Int, Int, Int),
        PureMLIP.clebsch_gordan_complex(Int, Int, Int, Int, Int, Int),
        PureMLIP.spherical_harmonics!(Vector{Float64}, Vector{Float64}, Int),
        PureMLIP.spherical_harmonics_and_gradient!(Vector{Float64}, Matrix{Float64}, Vector{Float64}, Int),
        PureMLIP.e3nn_spherical_harmonics_and_gradient!(Vector{Float64}, Matrix{Float64}, Vector{Float64}, Int),
        PureMLIP.e3nn_wigner_3j(Int, Int, Int),
        PureMLIP.entry_data(PureMLIP.TorchArchive, String)
    )
end
