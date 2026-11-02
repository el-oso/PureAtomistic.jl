@testitem "symmetric contraction matches the dense formula" setup = [Checkpoints] begin
    M = PureMLIP.Model
    Tn = PureMLIP.Tensors
    using Random
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); st = ck.state.f64
    sc = M.symmetric_contraction_from_state(st, "products.0", Float64)
    @test [[length(c.U1), length(c.U2), length(c.U3)] for c in sc.contractions] == [[1, 16, 353], [3, 70, 2846]]
    X = 0.3 .* randn(Xoshiro(5), 128, 16, 3); species = [29, 1, 63]                    # z-table indices
    B = zeros(128, 4, 3); Tn.symmetric_contraction!(B, sc, X, species, zeros(Tn.symmetric_contraction_scratch_size(sc)))
    p = "products.0.symmetric_contractions.contractions.0."
    U1, U2, U3 = torch_array(st[p * "U_matrix_1"], Val(2)), torch_array(st[p * "U_matrix_2"], Val(3)), torch_array(st[p * "U_matrix_3"], Val(4))
    Wm, W0, W1 = (torch_array(st[p * k], Val(3)) for k in ("weights_max", "weights.0", "weights.1"))
    for a in 1:3, c in (1, 64, 128)
        s = species[a]; x = X[c, :, a]
        C2 = [
            sum(U2[w, y, k] * W0[s, k, c] for k in axes(U2, 3)) +
                sum(U3[w, y, i, k] * Wm[s, k, c] * x[i] for i in 1:16, k in axes(U3, 4)) for w in 1:16, y in 1:16
        ]
        C1 = [sum(U1[w, k] * W1[s, k, c] for k in axes(U1, 2)) + sum(C2[w, y] * x[y] for y in 1:16) for w in 1:16]
        # Rounding bound: 4·eps·N relative, N = 16·16·23 terms per output.
        @test B[c, 1, a] ≈ sum(C1[w] * x[w] for w in 1:16) rtol = 4 * eps() * 16 * 16 * 23
    end
    # The same loops with the leading m index for contraction 1 check B[c, 2:4, a].
    p = "products.0.symmetric_contractions.contractions.1."
    U1, U2, U3 = torch_array(st[p * "U_matrix_1"], Val(3)), torch_array(st[p * "U_matrix_2"], Val(4)), torch_array(st[p * "U_matrix_3"], Val(5))
    Wm, W0, W1 = (torch_array(st[p * k], Val(3)) for k in ("weights_max", "weights.0", "weights.1"))
    for a in 1:3, c in (1, 64, 128), m in 1:3
        s = species[a]; x = X[c, :, a]
        C2 = [
            sum(U2[m, w, y, k] * W0[s, k, c] for k in axes(U2, 4)) +
                sum(U3[m, w, y, i, k] * Wm[s, k, c] * x[i] for i in 1:16, k in axes(U3, 5)) for w in 1:16, y in 1:16
        ]
        C1 = [sum(U1[m, w, k] * W1[s, k, c] for k in axes(U1, 3)) + sum(C2[w, y] * x[y] for y in 1:16) for w in 1:16]
        # Rounding bound: 4·eps·N relative, N = 16·16·51 terms per output.
        @test B[c, 1 + m, a] ≈ sum(C1[w] * x[w] for w in 1:16) rtol = 4 * eps() * 16 * 16 * 51
    end
    # products.1 has the single 0e contraction.
    sc1 = M.symmetric_contraction_from_state(st, "products.1", Float64)
    @test [[length(c.U1), length(c.U2), length(c.U3)] for c in sc1.contractions] == [[1, 16, 353]]
end

@testitem "symmetric contraction is equivariant" setup = [Checkpoints] begin
    M = PureMLIP.Model
    Tn = PureMLIP.Tensors
    using Random
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); st = ck.state.f64
    sc = M.symmetric_contraction_from_state(st, "products.0", Float64)
    X = 0.3 .* randn(Xoshiro(11), 128, 16, 2); species = [29, 8]
    Rz = [cos(0.7) -sin(0.7) 0; sin(0.7) cos(0.7) 0; 0 0 1]
    Rx = [1 0 0; 0 cos(1.1) -sin(1.1); 0 sin(1.1) cos(1.1)]
    R = Rx * Rz
    D = [e3nn_basis_change(l) * wigner_D(l, R) * e3nn_basis_change(l)' for l in 0:3]
    RX = similar(X)
    for a in axes(X, 3), c in axes(X, 1), l in 0:3
        r = (l^2 + 1):((l + 1)^2)
        RX[c, r, a] = D[l + 1] * X[c, r, a]
    end
    scratch = zeros(Tn.symmetric_contraction_scratch_size(sc))
    B = zeros(128, 4, 2); BR = zeros(128, 4, 2)
    Tn.symmetric_contraction!(B, sc, X, species, scratch)
    Tn.symmetric_contraction!(BR, sc, RX, species, scratch)
    # Measured margin: largest error 2.3e-15 (outputs up to 1.7); tolerance 4e-14.
    @test maximum(abs, BR[:, 1, :] - B[:, 1, :]) < 4.0e-14
    @test maximum(abs((BR[c, 2:4, a] - D[2] * B[c, 2:4, a])[k]) for a in 1:2, c in 1:128, k in 1:3) < 4.0e-14
end

@testitem "symmetric contraction pullback matches ForwardDiff and finite differences" setup = [Checkpoints] begin
    M = PureMLIP.Model
    Tn = PureMLIP.Tensors
    using Random, ForwardDiff
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); st = ck.state.f64
    sc = M.symmetric_contraction_from_state(st, "products.0", Float64)
    rng = Xoshiro(17)
    X = 0.3 .* randn(rng, 128, 16, 2); species = [29, 1]
    dB = randn(rng, 128, 4, 2)
    n = Tn.symmetric_contraction_scratch_size(sc)
    function f(Y)
        S = eltype(Y)
        B = zeros(S, 128, 4, 2)
        Tn.symmetric_contraction!(B, sc, Y, species, zeros(S, n))
        return sum(dB .* B)
    end
    dX = zeros(128, 16, 2)
    Tn.symmetric_contraction_pullback!(dX, sc, dB, X, species, zeros(n))
    g = ForwardDiff.gradient(f, X)
    # Rounding bound: 4·eps·N relative to the largest gradient entry, N = 16·16·51 terms per entry.
    @test maximum(abs, dX - g) <= 4 * eps() * 16 * 16 * 51 * maximum(abs, g)
    h = 1.0e-6
    Xp = copy(X); Xp[5, 7, 1] += h
    Xm = copy(X); Xm[5, 7, 1] -= h
    fd = (f(Xp) - f(Xm)) / 2h
    # Measured margin: central-difference error 8.2e-11; tolerance 1e-9.
    @test abs(fd - dX[5, 7, 1]) < 1.0e-9
end

@testitem "symmetric contraction: views, offset arrays, errors" setup = [Checkpoints] begin
    M = PureMLIP.Model
    Tn = PureMLIP.Tensors
    using Random, OffsetArrays
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); st = ck.state.f64
    sc = M.symmetric_contraction_from_state(st, "products.0", Float64)
    X = 0.3 .* randn(Xoshiro(3), 128, 16, 2); species = [29, 1]
    n = Tn.symmetric_contraction_scratch_size(sc)
    B = zeros(128, 4, 2)
    Tn.symmetric_contraction!(B, sc, X, species, zeros(n))
    Xbig = zeros(130, 16, 3); Xbig[2:129, :, 2:3] = X
    Bbig = zeros(128, 5, 2)
    Tn.symmetric_contraction!(view(Bbig, :, 2:5, :), sc, view(Xbig, 2:129, :, 2:3), view([0, 29, 1], 2:3), view(zeros(n + 1), 2:(n + 1)))
    @test view(Bbig, :, 2:5, :) == B
    dB = ones(128, 4, 2)
    dX = zeros(128, 16, 2)
    Tn.symmetric_contraction_pullback!(dX, sc, dB, X, species, zeros(n))
    dXv = zeros(128, 17, 2)
    Tn.symmetric_contraction_pullback!(view(dXv, :, 2:17, :), sc, view(ones(128, 5, 2), :, 2:5, :), view(Xbig, 2:129, :, 2:3), species, zeros(n))
    @test view(dXv, :, 2:17, :) == dX
    @test_throws ArgumentError Tn.symmetric_contraction!(OffsetArray(B, 0, 0, -1), sc, OffsetArray(X, 0, 0, -1), species, zeros(n))
    @test_throws "tensors: symmetric contraction expects X of size (128, 16, n), got (128, 15, 2)" Tn.symmetric_contraction!(B, sc, X[:, 1:15, :], species, zeros(n))
    @test_throws "tensors: symmetric contraction expects B of size (128, 4, 2), got (128, 3, 2)" Tn.symmetric_contraction!(zeros(128, 3, 2), sc, X, species, zeros(n))
    @test_throws "tensors: symmetric contraction expects dB of size (128, 4, 2), got (128, 3, 2)" Tn.symmetric_contraction_pullback!(dX, sc, zeros(128, 3, 2), X, species, zeros(n))
    @test_throws "tensors: symmetric contraction expects 2 species, got 1" Tn.symmetric_contraction!(B, sc, X, [29], zeros(n))
    @test_throws "tensors: symmetric contraction scratch has $(n - 1) elements, needs $n" Tn.symmetric_contraction!(B, sc, X, species, zeros(n - 1))
    @test_throws "tensors: species index 90 is outside 1:89" Tn.symmetric_contraction!(B, sc, X, [29, 90], zeros(n))
    @test_throws "tensors: species index 0 is outside 1:89" Tn.symmetric_contraction_pullback!(dX, sc, dB, X, [0, 1], zeros(n))
    @test_throws "model: missing products.7.symmetric_contractions.contractions.0.U_matrix_1" M.symmetric_contraction_from_state(st, "products.7", Float64)
    # Direct construction validates the shapes of U and W against each other.
    U1 = zeros(16, 1); U2 = zeros(16, 16, 4); U3 = zeros(16, 16, 16, 23)
    W = zeros(89, 23, 128); W0 = zeros(89, 4, 128); W1 = zeros(89, 1, 128)
    @test Tn.Contraction{Float64}(0, U1, U2, U3, W, W0, W1) isa Tn.Contraction{Float64}
    @test_throws "tensors: symmetric contraction: U_matrix_3 has size (16, 16, 16, 23), weights_max has size (89, 22, 128)" Tn.Contraction{Float64}(0, U1, U2, U3, zeros(89, 22, 128), W0, W1)
    @test_throws "tensors: symmetric contraction: U_matrix_2 has size (16, 15, 4), expected (16, 16, K)" Tn.Contraction{Float64}(0, U1, zeros(16, 15, 4), U3, W, W0, W1)
    @test_throws "tensors: symmetric contraction: U_matrix_1 has size (16, 1), weights.1 has size (89, 2, 128)" Tn.Contraction{Float64}(0, U1, U2, U3, W, W0, zeros(89, 2, 128))
    @test_throws "tensors: symmetric contraction: weights.0 has size (88, 4, 128), expected (89, 4, 128)" Tn.Contraction{Float64}(0, U1, U2, U3, W, zeros(88, 4, 128), W1)
    @test_throws "tensors: symmetric contraction: l = 1 needs U arrays with a leading index of length 3" Tn.Contraction{Float64}(1, U1, U2, U3, W, W0, W1)
    @test_throws "tensors: symmetric contraction: l must be non-negative, got -1" Tn.Contraction{Float64}(-1, U1, U2, U3, W, W0, W1)
    @test_throws "tensors: symmetric contraction: l = 0 needs U arrays of 2, 3 and 4 dimensions" Tn.Contraction{Float64}(0, zeros(1, 16, 1), U2, U3, W, W0, W1)
    @test_throws "tensors: symmetric contraction: U_matrix_3 has size (16, 16, 15, 23), expected (16, 16, 16, K)" Tn.Contraction{Float64}(0, U1, U2, zeros(16, 16, 15, 23), W, W0, W1)
    @test_throws "tensors: symmetric contraction: U_matrix_2 has size (3, 16, 15, 6), expected (3, 16, 16, K)" Tn.Contraction{Float64}(1, zeros(3, 16, 1), zeros(3, 16, 15, 6), zeros(3, 16, 16, 16, 51), zeros(89, 51, 128), zeros(89, 6, 128), W1)
    @test_throws "tensors: symmetric contraction needs at least one contraction" Tn.SymmetricContraction{Float64}(Tn.Contraction{Float64}[])
    @test_throws "tensors: symmetric contraction expects dX of size (128, 16, 2), got (128, 16, 3)" Tn.symmetric_contraction_pullback!(zeros(128, 16, 3), sc, dB, X, species, zeros(n))
    @test_throws "tensors: symmetric contraction: contractions see different channel, ell or species counts" Tn.SymmetricContraction{Float64}([Tn.Contraction{Float64}(0, U1, U2, U3, W, W0, W1), Tn.Contraction{Float64}(0, U1, U2, U3, W[1:88, :, :], W0[1:88, :, :], W1[1:88, :, :])])
end

# Oracle: the per-atom kernels, an independent evaluation of the same sums. 150 atoms of three
# species, 70 of one of them, so that a species fills a group of 64 atoms and leaves a partial one,
# and the species interleave.
@testitem "symmetric contraction as matrix products matches the per-atom form" setup = [Checkpoints] begin
    Tn = PureMLIP.Tensors
    using Random
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0))
    for T in (Float64, Float32)
        m = MACEModel{T}(ck)
        rng = Xoshiro(23)
        species = shuffle(rng, [fill(29, 70); fill(1, 50); fill(63, 30)])
        n = length(species)
        for (k, sc) in enumerate(m.product)
            cm = Tn.ContractionMatrices{T}(sc)
            foreach(s -> Tn.contraction_matrices!(cm, sc, s), (29, 1, 63))
            buf = Tn.ContractionBuffers{T}(cm)
            K = Tn.n_outputs(sc)
            X = T.(0.3 .* randn(rng, 128, 16, n))
            dB = T.(randn(rng, 128, K, n))
            scratch = zeros(T, Tn.symmetric_contraction_scratch_size(sc))
            B = Tn.symmetric_contraction!(zeros(T, 128, K, n), sc, X, species, scratch)
            dX = Tn.symmetric_contraction_pullback!(zeros(T, size(X)), sc, dB, X, species, scratch)
            Bm = Tn.symmetric_contraction!(fill(T(NaN), 128, K, n), cm, X, species, buf)
            dXm = Tn.symmetric_contraction_pullback!(fill(T(NaN), size(X)), cm, dB, X, species, buf)
            # Rounding bound 10·eps(T)·N relative to the largest magnitude, with N = 1280 the
            # largest inner dimension of any matrix product in the model. Measured: at most
            # 3.3e-16 (forward) and 1.4e-15 (pullback) in Float64, 2.4e-7 and 9.0e-7 in Float32.
            tol = 10 * eps(T) * 1280
            @test maximum(abs, Bm - B) <= tol * maximum(abs, B)
            @test maximum(abs, dXm - dX) <= tol * maximum(abs, dX)
            # The pullback matrices alone give the same pullback.
            cmb = Tn.ContractionMatrices{T}(sc; forward = false)
            foreach(s -> Tn.contraction_matrices!(cmb, sc, s), (29, 1, 63))
            @test Tn.symmetric_contraction_pullback!(similar(X), cmb, dB, X, species, buf) == dXm
            # A second call for a built species keeps its matrices.
            G = cm.G[29]
            @test Tn.contraction_matrices!(cm, sc, 29) === cm && cm.G[29] === G
        end
    end
end

@testitem "symmetric contraction as matrix products: pullback matches ForwardDiff and finite differences" setup = [Checkpoints] begin
    Tn = PureMLIP.Tensors
    using Random, ForwardDiff
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); st = ck.state.f64
    sc = PureMLIP.Model.symmetric_contraction_from_state(st, "products.0", Float64)
    rng = Xoshiro(29)
    X = 0.3 .* randn(rng, 128, 16, 3); species = [29, 1, 29]
    dB = randn(rng, 128, 4, 3)
    cm = Tn.ContractionMatrices{Float64}(sc)
    foreach(s -> Tn.contraction_matrices!(cm, sc, s), (29, 1))
    buf = Tn.ContractionBuffers{Float64}(cm; atoms = 2)
    dX = Tn.symmetric_contraction_pullback!(zeros(128, 16, 3), cm, dB, X, species, buf)
    # Oracle 1: ForwardDiff through the per-atom forward kernel.
    n = Tn.symmetric_contraction_scratch_size(sc)
    function f(Y)
        S = eltype(Y)
        return sum(dB .* Tn.symmetric_contraction!(zeros(S, 128, 4, 3), sc, Y, species, zeros(S, n)))
    end
    g = ForwardDiff.gradient(f, X)
    # Rounding bound: 4·eps·N relative to the largest gradient entry, N = 16·16·51 terms per entry.
    @test maximum(abs, dX - g) <= 4 * eps() * 16 * 16 * 51 * maximum(abs, g)
    # Oracle 2: central differences of the matrix-product forward kernel.
    fm(Y) = sum(dB .* Tn.symmetric_contraction!(zeros(128, 4, 3), cm, Y, species, buf))
    h = 1.0e-6
    for (c, w, a) in ((5, 7, 1), (100, 1, 2), (64, 16, 3))
        Xp = copy(X); Xp[c, w, a] += h
        Xm = copy(X); Xm[c, w, a] -= h
        # Measured margin: largest central-difference error 4.4e-10; tolerance 5e-9.
        @test abs((fm(Xp) - fm(Xm)) / 2h - dX[c, w, a]) < 5.0e-9
    end
end

@testitem "symmetric contraction as matrix products: views, offset arrays, errors" setup = [Checkpoints] begin
    Tn = PureMLIP.Tensors
    using Random, OffsetArrays
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); st = ck.state.f64
    sc = PureMLIP.Model.symmetric_contraction_from_state(st, "products.0", Float64)
    sc1 = PureMLIP.Model.symmetric_contraction_from_state(st, "products.1", Float64)
    X = 0.3 .* randn(Xoshiro(31), 128, 16, 2); species = [29, 1]
    cm = Tn.ContractionMatrices{Float64}(sc)
    foreach(s -> Tn.contraction_matrices!(cm, sc, s), species)
    buf = Tn.ContractionBuffers{Float64}(cm)
    B = Tn.symmetric_contraction!(zeros(128, 4, 2), cm, X, species, buf)
    dB = ones(128, 4, 2)
    dX = Tn.symmetric_contraction_pullback!(zeros(128, 16, 2), cm, dB, X, species, buf)
    # Views give the same arithmetic as the arrays they view.
    Xbig = zeros(130, 16, 3); Xbig[2:129, :, 2:3] = X
    Bbig = zeros(128, 5, 2)
    Tn.symmetric_contraction!(view(Bbig, :, 2:5, :), cm, view(Xbig, 2:129, :, 2:3), view([0, 29, 1], 2:3), buf)
    @test view(Bbig, :, 2:5, :) == B
    dXv = zeros(128, 17, 2)
    Tn.symmetric_contraction_pullback!(view(dXv, :, 2:17, :), cm, view(ones(128, 5, 2), :, 2:5, :), view(Xbig, 2:129, :, 2:3), species, buf)
    @test view(dXv, :, 2:17, :) == dX
    # The message comes from Base.require_one_based_indexing, so only the type is checked.
    @test_throws ArgumentError Tn.symmetric_contraction!(OffsetArray(B, 0, 0, -1), cm, OffsetArray(X, 0, 0, -1), species, buf)
    @test_throws ArgumentError Tn.symmetric_contraction_pullback!(OffsetArray(dX, 0, 0, -1), cm, dB, X, species, buf)
    @test_throws "tensors: symmetric contraction expects X of size (128, 16, n), got (128, 15, 2)" Tn.symmetric_contraction!(B, cm, X[:, 1:15, :], species, buf)
    @test_throws "tensors: symmetric contraction expects B of size (128, 4, 2), got (128, 3, 2)" Tn.symmetric_contraction!(zeros(128, 3, 2), cm, X, species, buf)
    @test_throws "tensors: symmetric contraction expects dB of size (128, 4, 2), got (128, 3, 2)" Tn.symmetric_contraction_pullback!(dX, cm, zeros(128, 3, 2), X, species, buf)
    @test_throws "tensors: symmetric contraction expects dX of size (128, 16, 2), got (128, 16, 3)" Tn.symmetric_contraction_pullback!(zeros(128, 16, 3), cm, dB, X, species, buf)
    @test_throws "tensors: symmetric contraction expects 2 species, got 1" Tn.symmetric_contraction!(B, cm, X, [29], buf)
    @test_throws "tensors: species index 90 is outside 1:89" Tn.symmetric_contraction!(B, cm, X, [29, 90], buf)
    @test_throws "tensors: species 63 has no contraction matrices" Tn.symmetric_contraction_pullback!(dX, cm, dB, X, [29, 63], buf)
    @test_throws "tensors: species index 0 is outside 1:89" Tn.contraction_matrices!(cm, sc, 0)
    cm1 = Tn.ContractionMatrices{Float64}(sc1; forward = false)
    Tn.contraction_matrices!(cm1, sc1, 29)
    @test isempty(cm1.F[29]) && size(cm1.G[29]) == (153, 16, 128)
    @test_throws "tensors: these contraction matrices hold no forward matrices" Tn.symmetric_contraction!(zeros(128, 1, 2), cm1, X, [29, 29], buf)
    @test_throws "tensors: contraction matrices and symmetric contraction differ in channels, ells, outputs or species" Tn.contraction_matrices!(cm1, sc, 1)
    @test_throws "tensors: contraction buffers are smaller than the contraction matrices" Tn.symmetric_contraction!(B, cm, X, species, Tn.ContractionBuffers{Float64}(cm1))
    @test_throws "tensors: contraction buffers need a positive atom count, got 0" Tn.ContractionBuffers{Float64}(cm; atoms = 0)
end
