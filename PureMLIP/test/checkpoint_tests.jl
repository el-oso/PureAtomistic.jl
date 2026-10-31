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

@testitem "MPA-0 and MP-0b3 pickled constants" setup = [Checkpoints] begin
    s3, s5, s7 = 1.7320508075688772, 2.23606797749979, 2.6457513110645907
    a, b = 0.0625, 0.05103103630798288
    for (name, cst, lin1) in ((:mpa0, 1.679176792398942, [a, b, b, a]), (:mp0b3, 1.6791767923989418, [a, a, b, b, b, b, b, b, a, a]))
        k = load_mace_checkpoint(Checkpoints.fetch(name)).constants
        acts = [["interactions.$i.conv_tp_weights.layer$j.act" for i in 0:1 for j in 0:2]; "readouts.1.non_linearity.acts.0"]
        @test sort(collect(keys(k.normalize2mom))) == sort(acts)
        # Pickled floats pass through unchanged: exact comparison.
        @test all(k.normalize2mom[p].cst == cst && k.normalize2mom[p].f == "torch.nn.functional.silu" for p in acts)
        @test cst != 1.6765324703310911                     # the exact second-moment integral
        @test k.pair_repulsion && k.distance_transform == "mace.modules.radial.AgnesiTransform"
        @test k.product_use_sc == [false, true]
        lw(p) = [i.path_weight for i in k.linear[p].instructions]
        tw(p) = [i.path_weight for i in k.tensor_product[p].instructions]
        @test lw("node_embedding.linear") == [0.105999788000636]
        @test lw("interactions.0.linear") == fill(0.08838834764831845, 4)
        @test lw("interactions.1.linear_up") == fill(0.08838834764831845, 2)
        @test lw("interactions.1.linear") == lin1
        @test lw("readouts.1.linear_2") == [0.25]
        @test tw("interactions.0.conv_tp") == [1.0, s3, s5, s7]
        @test tw("interactions.1.conv_tp") == [1.0, 1.0, s3, s3, s3, s5, s5, s5, s7, s7]
        @test tw("interactions.0.skip_tp") == [0.009369146112448268, 0.01622783709029683, 0.020950047598562217, 0.024788430610565716]
        @test [(i.i_in1, i.i_in2, i.i_out) for i in k.tensor_product["interactions.1.conv_tp"].instructions] ==
            [(0, 0, 0), (1, 1, 1), (0, 1, 2), (1, 0, 3), (1, 2, 4), (0, 2, 5), (1, 1, 6), (1, 3, 7), (0, 3, 8), (1, 2, 9)]
        @test all(i -> i.connection_mode == "uvu", k.tensor_product["interactions.1.conv_tp"].instructions)
    end
    P = PureMLIP.Checkpoint.parse_irreps
    @test load_mace_checkpoint(Checkpoints.fetch(:mpa0)).constants.linear["interactions.1.linear"].irreps_in == P("256x0e+384x1o+384x2e+256x3o")
end

@testitem "constants of a skeleton checkpoint" setup = [ZipFixtures] begin
    k = load_mace_checkpoint(ZipFixtures.tiny_mace_zip()).constants
    @test isempty(k.normalize2mom) && isempty(k.linear) && !k.pair_repulsion && k.distance_transform == ""
    @test k.product_use_sc == [false]
end

@testitem "malformed pickled constants are rejected" setup = [ZipFixtures] begin
    Z = ZipFixtures
    n2m(; cst = Z.pk_float(1.5)) = Z.pk_module(
        Z.pk_module_state(attrs = ["f" => Z.pk_global("torch.nn.functional", "silu"), "cst" => cst]);
        cls = ("e3nn.math._normalize_activation", "normalize2mom")
    )
    lin(instr) = Z.pk_module(
        Z.pk_module_state(attrs = ["irreps_in" => Z.pk_irreps((1, 0, 1)), "irreps_out" => Z.pk_irreps((1, 0, 1)), "instructions" => instr]);
        cls = ("e3nn.o3._linear", "Linear")
    )
    linsn(args...) = Z.pk_call(Z.pk_global("e3nn.o3._linear", "Instruction"), Z.pk_tuple(args...))
    load(m) = load_mace_checkpoint(Z.tiny_mace_zip(extra_modules = ["act" => m]))
    # A well-formed module is read back.
    k = load(n2m()).constants
    @test k.normalize2mom["act"] == PureMLIP.Normalize2Mom(1.5, "torch.nn.functional.silu")
    ok = linsn(Z.pk_int(0), Z.pk_int(0), Z.pk_tuple(Z.pk_int(1), Z.pk_int(1)), Z.pk_float(0.5))
    @test load(lin(Z.pk_list(ok))).constants.linear["act"].instructions[1].path_weight == 0.5
    @test_throws "checkpoint: act: malformed cst" load(n2m(cst = Z.pk_str("x")))
    @test_throws "checkpoint: act: malformed instruction" load(lin(Z.pk_list(linsn(Z.pk_int(0)))))
    @test_throws "checkpoint: act: malformed instruction" load(lin(Z.pk_list(linsn(Z.pk_int(0), Z.pk_int(0), Z.pk_int(1), Z.pk_float(0.5)))))
    @test_throws "checkpoint: act: malformed instructions" load(lin(Z.pk_tuple()))
end
