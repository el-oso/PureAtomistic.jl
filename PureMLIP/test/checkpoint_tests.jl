@testitem "MP-0b3 matches the llmlip.rs inspection" setup = [Checkpoints] begin
    ck = load_mace_checkpoint(Checkpoints.fetch(:mp0b3))
    ref = Checkpoints.inspection()
    h = ck.hyperparameters
    # The file holds 89 state tensors. The reference lists 93; the four extra names are
    # `_compiled_main_left_right._w3j_*` buffers (llmlip classification
    # `compiled_tensor_product_wigner_buffer`), which are absent from this state dict.
    # data.pkl embeds TorchScript archive bytes.
    @test length(ck.state.names) == 89
    @test Checkpoints.absent(ck, ref) == sort(
        [
            "interactions.1.conv_tp._compiled_main_left_right._w3j_1_1_2",
            "interactions.1.conv_tp._compiled_main_left_right._w3j_1_2_1",
            "interactions.1.conv_tp._compiled_main_left_right._w3j_1_2_3",
            "interactions.1.conv_tp._compiled_main_left_right._w3j_1_3_2",
        ]
    )
    @test sum(PureMLIP.numel(ck.state.f64[n]) for n in ck.state.names if haskey(ck.state.f64, n) && ck.state.f64[n].requires_grad) == 9063204
    @test h.r_max == 6.0 && h.num_interactions == 2 && h.heads == ["default"]
    @test length(h.atomic_numbers) == 89 && h.atomic_numbers[(end - 2):end] == [92, 93, 94]
    @test [i.class for i in h.interactions] == [
        "mace.modules.blocks.RealAgnosticDensityInteractionBlock",
        "mace.modules.blocks.RealAgnosticDensityResidualInteractionBlock",
    ]
    @test all(i -> i.avg_num_neighbors == 61.964672446250916 && i.radial_MLP == [64, 64, 64], h.interactions)
    @test h.correlations == [3, 3]
    @test h.readouts[2].hidden_irreps == [PureMLIP.IrrepMul(16, 0, 1)]
    U = ck.state.f64["products.0.symmetric_contractions.contractions.0.U_matrix_3"]
    @test U.shape == [16, 16, 16, 23] && count(!iszero, U.data) == 353
    @test ck.state.f64["products.0.symmetric_contractions.contractions.0.weights_max"].shape == [89, 23, 128]
    # Every tensor the reference lists (covered names, missing-tensor entries with shape,
    # dtype and nonzero count, and all product-block parameters and buffers) is present
    # with the same shape, dtype and nonzero count; the state holds nothing else.
    @test Checkpoints.mismatches(ck, ref) == String[]
    @test Checkpoints.interaction_mismatches(ck, ref) == String[]
end

@testitem "MPA-0 loads" setup = [Checkpoints] begin
    h = load_mace_checkpoint(Checkpoints.fetch(:mpa0)).hyperparameters
    @test h.r_max == 6.0 && h.correlations == [3, 3] && length(h.atomic_numbers) == 89
    @test h.interactions[1].hidden_irreps == [PureMLIP.IrrepMul(128, 0, 1), PureMLIP.IrrepMul(128, 1, -1)]
    @test all(i -> isfinite(i.avg_num_neighbors) && i.avg_num_neighbors > 0, h.interactions)
end

@testitem "non-MACE pickle is rejected" setup = [ZipFixtures] begin
    @test_throws "not a MACE checkpoint: the pickle root is not a module" load_mace_checkpoint(ZipFixtures.dict_root_zip())
end

@testitem "avg_num_neighbors buffer form and irreps strings" setup = [ZipFixtures] begin
    ck = load_mace_checkpoint(ZipFixtures.tiny_mace_zip(avg_buffer = true))
    h = ck.hyperparameters
    @test h.model_class == "mace.modules.models.ScaleShiftMACE"
    @test h.interactions[1].avg_num_neighbors == 61.964672446250916
    @test "interactions.0.avg_num_neighbors" in ck.state.names
    @test h.interactions[1].hidden_irreps == [PureMLIP.IrrepMul(128, 0, 1), PureMLIP.IrrepMul(128, 1, -1)]
    @test h.interactions[1].edge_attrs_irreps == [PureMLIP.IrrepMul(1, 0, 1), PureMLIP.IrrepMul(1, 1, -1)]
    @test h.r_max == 5.0 && h.num_interactions == 1 && h.atomic_numbers == [1, 8] && h.heads == ["default"]
    @test h.correlations == [3]
    @test [r.class for r in h.readouts] == ["mace.modules.blocks.LinearReadoutBlock"] && isempty(h.readouts[1].hidden_irreps)
    h2 = load_mace_checkpoint(ZipFixtures.tiny_mace_zip(avg_buffer = false)).hyperparameters
    @test h2.interactions[1].avg_num_neighbors == 61.964672446250916
end

@testitem "unsupported model class" setup = [ZipFixtures] begin
    @test_throws "unsupported model class mace.modules.blocks.LinearReadoutBlock" load_mace_checkpoint(
        ZipFixtures.tiny_mace_zip(root_class = ("mace.modules.blocks", "LinearReadoutBlock"))
    )
end

@testitem "load_mace_checkpoint reads a path" setup = [ZipFixtures] begin
    mktempdir() do dir
        p = joinpath(dir, "tiny.model")
        write(p, ZipFixtures.tiny_mace_zip())
        @test load_mace_checkpoint(p).hyperparameters.correlations == [3]
    end
end

@testitem "module without its dicts" setup = [ZipFixtures] begin
    F = ZipFixtures
    cls = ("mace.modules.models", "ScaleShiftMACE")
    # A root module whose state has `_parameters` and `_modules` but no `_buffers`.
    body = F.pk_dict("_parameters" => F.pk_dict(), "_modules" => F.pk_dict())
    z = F.zip_with_pickle(F.pk_finish(F.pk_module(body; cls)), Pair{String, Vector{UInt8}}[])
    @test_throws "checkpoint: root module has no _buffers" load_mace_checkpoint(z)
end

@testitem "parse_irreps: implicit multiplicity and y parity" begin
    parse_irreps = PureMLIP.Checkpoint.parse_irreps
    I = PureMLIP.IrrepMul
    # e3nn: a term without a multiplicity has multiplicity 1; parity 'y' is (-1)^l.
    @test parse_irreps("0e + 1o + 2y + 3y") == [I(1, 0, 1), I(1, 1, -1), I(1, 2, 1), I(1, 3, -1)]
    @test parse_irreps("16x2y") == [I(16, 2, 1)]
    @test_throws "unknown parity z" parse_irreps("1x0z")
    @test_throws "malformed irreps string" parse_irreps("1x2x0e")
end
