@testitem "TrimCheck: checkpoint loading" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureMLIP
            # TrimCheck takes the signature of a call from `typeof(f)`, which is `DataType` for a
            # constructor; these wrappers carry the constructor calls.
            new_model64(ck) = PureMLIP.MACEModel{Float64}(ck)
            new_model32(ck) = PureMLIP.MACEModel{Float32}(ck)
            new_result64(n) = PureMLIP.MACEResult{Float64}(n)
            new_workspace64(m, n, e) = PureMLIP.MACEWorkspace{Float64}(m, n, e)
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
        PureMLIP.e3nn_spherical_harmonics!(Vector{Float64}, Vector{Float64}, Int),
        PureMLIP.e3nn_basis_change(Int),
        PureMLIP.sh_length(Int),
        PureMLIP.sh_index(Int, Int),
        PureMLIP.embedded_archives(PureMLIP.PickleTree),
        PureMLIP.entry_data(PureMLIP.TorchArchive, String),
        PureMLIP.neighbor_list(Matrix{Float64}, Matrix{Float64}, Float64),
        PureMLIP.neighbor_list!(PureMLIP.NeighborList{Float64}, Matrix{Float64}, Matrix{Float64}),
        new_model64(PureMLIP.MACECheckpoint),
        new_model32(PureMLIP.MACECheckpoint),
        new_result64(Int),
        new_workspace64(PureMLIP.MACEModel{Float64}, Int, Int),
        PureMLIP.mace_energy!(
            PureMLIP.MACEResult{Float64}, PureMLIP.MACEWorkspace{Float64}, PureMLIP.MACEModel{Float64},
            Matrix{Float64}, Matrix{Float64}, Vector{Int}, PureMLIP.NeighborList{Float64}
        ),
        PureMLIP.mace_energy_forces!(
            PureMLIP.MACEResult{Float64}, PureMLIP.MACEWorkspace{Float64}, PureMLIP.MACEModel{Float64},
            Matrix{Float64}, Matrix{Float64}, Vector{Int}, PureMLIP.NeighborList{Float64}
        ),
        PureMLIP.mace_energy_forces(PureMLIP.MACEModel{Float64}, Matrix{Float64}, Matrix{Float64}, Vector{Int})
    )
end
