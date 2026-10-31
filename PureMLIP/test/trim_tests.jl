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
        PureMLIP.clebsch_gordan_complex(Int, Int, Int, Int, Int, Int)
    )
end
