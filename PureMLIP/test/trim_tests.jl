@testitem "TrimCheck: checkpoint loading" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureMLIP
        end,
        PureMLIP.load_mace_checkpoint(Vector{UInt8}),
        PureMLIP.load_mace_checkpoint(String)
    )
end
