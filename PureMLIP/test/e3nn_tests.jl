@testitem "e3nn SH and gradient match e3nn-core-rs" setup = [E3nnFixtures] begin
    for L in (2, 3, 4)
        f = E3nnFixtures.sh(L)
        n = PureMLIP.sh_length(L)
        for i in axes(f.vectors, 1)
            Y = zeros(n); dY = zeros(n, 3)
            PureMLIP.e3nn_spherical_harmonics_and_gradient!(Y, dY, f.vectors[i, :], L)
            @test Y ≈ f.y[i, :] atol = 1.0e-10
            @test dY' * f.grad_y[i, :] ≈ f.grad_vectors[i, :] atol = 1.0e-10
            Y2 = zeros(n)
            @test PureMLIP.e3nn_spherical_harmonics!(Y2, f.vectors[i, :], L) === Y2
            @test Y2 == Y
        end
    end
end

@testitem "e3nn SH on offset and view inputs" begin
    using OffsetArrays
    v = [0.3, -0.5, 0.8]; L = 3; n = PureMLIP.sh_length(L)
    Y = zeros(n); dY = zeros(n, 3)
    PureMLIP.e3nn_spherical_harmonics_and_gradient!(Y, dY, v, L)
    Yo = OffsetVector(zeros(n), -4); dYo = OffsetArray(zeros(n, 3), 2, -1)
    PureMLIP.e3nn_spherical_harmonics_and_gradient!(Yo, dYo, OffsetVector(copy(v), 7), L)
    @test parent(Yo) == Y && parent(dYo) == dY
    buf = zeros(n + 2); vbuf = [9.0, v..., 9.0]
    PureMLIP.e3nn_spherical_harmonics!(view(buf, 2:(n + 1)), view(vbuf, 2:4), L)
    @test buf[2:(n + 1)] == Y && buf[1] == 0 && buf[end] == 0
end

@testitem "e3nn_wigner_3j matches the tensor-product fixtures" setup = [E3nnFixtures] begin
    @test [l[1:3] for l in E3nnFixtures.cg_instructions("fully_connected_tensor_product_lmax4_smoke_f64")] ==
        [(3, 1, 2), (3, 1, 4), (3, 2, 3), (4, 1, 3), (4, 2, 2), (4, 2, 4)]
    for name in ("fully_connected_tensor_product_smoke_f64", "fully_connected_tensor_product_lmax4_smoke_f64")
        for (l1, l2, l3, entries) in E3nnFixtures.cg_instructions(name)
            C = PureMLIP.e3nn_wigner_3j(l1, l2, l3)
            @test all(isapprox(C[i + 1, j + 1, k + 1], v; atol = 1.0e-12) for (i, j, k, v) in entries)
            @test count(!iszero, C) == length(entries)
        end
    end
end

@testitem "e3nn_wigner_3j matches the tables stored in MP-0b3" setup = [Checkpoints] begin
    a = PureMLIP.open_torch_archive(read(Checkpoints.fetch(:mp0b3)))
    arch = only(filter(x -> haskey(x.entries, x.prefix * "data/0"), PureMLIP.embedded_archives(PureMLIP.unpickle(PureMLIP.entry_data(a, "data.pkl")))))
    # (l1, l2, l3), entry, shape, strides in elements: read by hand from that archive's own
    # data.pkl, which uses torch.fx globals outside the unpickler's allowlist.
    for (l, entry, shape, strides) in (
            ((1, 2, 1), "data/0", (3, 5, 3), (5, 1, 15)), ((1, 1, 2), "data/1", (3, 3, 5), (15, 5, 1)),
            ((1, 3, 2), "data/2", (3, 7, 5), (35, 1, 7)), ((1, 2, 3), "data/3", (3, 5, 7), (35, 7, 1)),
        )
        raw = reinterpret(Float64, collect(PureMLIP.entry_data(arch, entry)))
        T = [raw[1 + (i - 1) * strides[1] + (j - 1) * strides[2] + (k - 1) * strides[3]] for i in 1:shape[1], j in 1:shape[2], k in 1:shape[3]]
        @test PureMLIP.e3nn_wigner_3j(l...) ≈ T atol = 1.0e-14
    end
end

@testitem "e3nn_basis_change for l = 1, 2" begin
    @test PureMLIP.e3nn_basis_change(0) ≈ ones(1, 1) atol = 1.0e-14
    @test PureMLIP.e3nn_basis_change(1) ≈ [0 0 1; 1 0 0; 0 1 0] atol = 1.0e-14
    @test PureMLIP.e3nn_basis_change(2) ≈ [0 0 0 1 0; 1 0 0 0 0; 0 0 -1 / 2 0 -sqrt(3) / 2; 0 1 0 0 0; 0 0 sqrt(3) / 2 0 -1 / 2] atol = 1.0e-14
end

@testitem "e3nn_wigner_3j rejects odd sums and triangle violations" begin
    @test_throws "real Clebsch-Gordan needs |l1-l2| ≤ l3 ≤ l1+l2 and even l1+l2+l3, got (1, 1, 1)" PureMLIP.e3nn_wigner_3j(1, 1, 1)
    @test_throws "got (1, 1, 4)" PureMLIP.e3nn_wigner_3j(1, 1, 4)
end

@testitem "e3nn_wigner_3j zero threshold separates residue from coefficients" begin
    for l1 in 0:8, l2 in 0:8, l3 in abs(l1 - l2):min(l1 + l2, 8)
        iseven(l1 + l2 + l3) || continue
        raw = PureMLIP.Equivariant._e3nn_wigner_3j_unzeroed(l1, l2, l3)
        W = PureMLIP.e3nn_wigner_3j(l1, l2, l3)
        for i in eachindex(raw)
            if iszero(W[i])
                @test abs(raw[i]) <= 1.0e-14
            else
                @test abs(raw[i]) >= 1.0e-6
                @test W[i] == raw[i]
            end
        end
    end
end
