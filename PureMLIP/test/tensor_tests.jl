@testitem "torch_array index order" begin
    x = PureMLIP.CheckpointTensor{Float64}([2, 3], collect(1.0:6.0), false)
    A = PureMLIP.torch_array(x, Val(2))
    @test size(A) == (2, 3) && A[1, 2] == 2.0 && A[2, 1] == 4.0
    @test PureMLIP.numel(x) == 6
    s = PureMLIP.CheckpointTensor{Int64}(Int[], [7], false)
    @test PureMLIP.numel(s) == 1 && PureMLIP.torch_array(s, Val(0))[] == 7
end

@testitem "strided view of a shared storage" setup = [ZipFixtures] begin
    F = ZipFixtures
    st() = F.pk_storage("DoubleStorage", "0", 6)
    state = F.pk_module_state(params = ["x" => F.pk_tensor(st(), 1, [2], [2]),
                                        "y" => F.pk_param(F.pk_tensor(st(), 0, [2, 3], [3, 1]), true)])
    z = F.zip_with_pickle(F.pk_finish(F.pk_module(state)), ["0" => F.f64bytes(1:6)])
    a = PureMLIP.open_torch_archive(z)
    sd = PureMLIP.read_state_dict(a, PureMLIP.unpickle(PureMLIP.entry_data(a, "data.pkl")))
    @test sd.names == ["x", "y"]
    @test sd.f64["x"].data == [2.0, 4.0] && sd.f64["x"].shape == [2] && !sd.f64["x"].requires_grad
    @test sd.f64["y"].data == 1.0:6.0 && sd.f64["y"].shape == [2, 3] && sd.f64["y"].requires_grad
    # A transposed view reads in torch (row-major) order of the view.
    state2 = F.pk_module_state(params = ["t" => F.pk_tensor(st(), 0, [3, 2], [1, 3])])
    a2 = PureMLIP.open_torch_archive(F.zip_with_pickle(F.pk_finish(F.pk_module(state2)), ["0" => F.f64bytes(1:6)]))
    sd2 = PureMLIP.read_state_dict(a2, PureMLIP.unpickle(PureMLIP.entry_data(a2, "data.pkl")))
    @test sd2.f64["t"].data == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
end

@testitem "state dict order, nesting, skips, int64, 0-dim" setup = [ZipFixtures] begin
    F = ZipFixtures
    child = F.pk_module(F.pk_module_state(params = ["w" => F.pk_tensor(F.pk_storage("DoubleStorage", "0", 2), 0, [2], [1]),
                                                    "skip" => F.pk_none()]))
    root = F.pk_module(F.pk_module_state(
        params = ["p" => F.pk_tensor(F.pk_storage("DoubleStorage", "0", 2), 0, Int[], Int[])],
        buffers = ["b" => F.pk_tensor(F.pk_storage("LongStorage", "1", 3), 0, [3], [1]),
                   "np" => F.pk_tensor(F.pk_storage("LongStorage", "1", 3), 0, Int[], Int[])],
        modules = ["m" => child, "none" => F.pk_none()], nonpersistent = ["np"]))
    a = PureMLIP.open_torch_archive(F.zip_with_pickle(F.pk_finish(root), ["0" => F.f64bytes([5, 6]), "1" => F.i64bytes([7, 8, 9])]))
    sd = PureMLIP.read_state_dict(a, PureMLIP.unpickle(PureMLIP.entry_data(a, "data.pkl")))
    @test sd.names == ["p", "b", "m.w"]
    @test sd.f64["p"].shape == Int[] && sd.f64["p"].data == [5.0]
    @test sd.i64["b"].data == [7, 8, 9]
    @test sd.f64["m.w"].data == [5.0, 6.0]
    @test !haskey(sd.i64, "np")
end

@testitem "fx graph module body" setup = [ZipFixtures] begin
    F = ZipFixtures
    body = F.pk_module_state(params = ["a" => F.pk_tensor(F.pk_storage("DoubleStorage", "0", 1), 0, [1], [1])])
    root = F.pk_call(F.pk_global("torch.fx.graph_module", "reduce_graph_module"), F.pk_tuple(body, F.pk_str("import torch")))
    a = PureMLIP.open_torch_archive(F.zip_with_pickle(F.pk_finish(root), ["0" => F.f64bytes([3])]))
    sd = PureMLIP.read_state_dict(a, PureMLIP.unpickle(PureMLIP.entry_data(a, "data.pkl")))
    @test sd.names == ["a"] && sd.f64["a"].data == [3.0]
end

@testitem "state dict errors" setup = [ZipFixtures] begin
    F = ZipFixtures
    read(z) = (a = PureMLIP.open_torch_archive(z); PureMLIP.read_state_dict(a, PureMLIP.unpickle(PureMLIP.entry_data(a, "data.pkl"))))
    mod(params; kw...) = F.pk_finish(F.pk_module(F.pk_module_state(; params, kw...)))
    t(cls, off, size, stride) = F.pk_tensor(F.pk_storage(cls, "0", 2), off, size, stride)
    @test_throws "big-endian checkpoints are not supported" read(F.zip_with_pickle(mod(["x" => t("DoubleStorage", 0, [2], [1])]), ["0" => F.f64bytes(1:2)]; byteorder = "big"))
    @test_throws "torch.nn.functional.silu" read(F.zip_with_pickle(mod(["x" => t("nn.functional.silu", 0, [2], [1])]), ["0" => F.f64bytes(1:2)]))
    @test_throws "tensor x reads outside its storage" read(F.zip_with_pickle(mod(["x" => t("DoubleStorage", 1, [2], [1])]), ["0" => F.f64bytes(1:2)]))
    dup = mod(["x" => t("DoubleStorage", 0, [2], [1])]; buffers = ["x" => t("DoubleStorage", 0, [2], [1])])
    @test_throws "duplicate state dict name x" read(F.zip_with_pickle(dup, ["0" => F.f64bytes(1:2)]))
end
