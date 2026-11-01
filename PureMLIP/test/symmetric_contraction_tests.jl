@testitem "symmetric contraction matches the dense formula" setup = [Checkpoints] begin
    using Random
    M = PureMLIP.Model
    Tn = PureMLIP.Tensors
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); st = ck.state.f64
    sc = M.symmetric_contraction_from_state(st, "products.0", Float64)
    @test [[length(c.U1), length(c.U2), length(c.U3)] for c in sc.contractions] == [[1, 16, 353], [3, 70, 2846]]
    X = 0.3 .* randn(Xoshiro(5), 128, 16, 3); species = [29, 1, 63]                    # z-table indices
    B = zeros(128, 4, 3); M.symmetric_contraction!(B, sc, X, species, zeros(Tn.symmetric_contraction_scratch_size(sc)))
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
    using Random
    M = PureMLIP.Model
    Tn = PureMLIP.Tensors
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
    M.symmetric_contraction!(B, sc, X, species, scratch)
    M.symmetric_contraction!(BR, sc, RX, species, scratch)
    # Measured margin: largest error 2.3e-15 (outputs up to 1.7); tolerance 4e-14.
    @test maximum(abs, BR[:, 1, :] - B[:, 1, :]) < 4.0e-14
    @test maximum(abs((BR[c, 2:4, a] - D[2] * B[c, 2:4, a])[k]) for a in 1:2, c in 1:128, k in 1:3) < 4.0e-14
end

@testitem "symmetric contraction pullback matches ForwardDiff and finite differences" setup = [Checkpoints] begin
    using Random, ForwardDiff
    M = PureMLIP.Model
    Tn = PureMLIP.Tensors
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); st = ck.state.f64
    sc = M.symmetric_contraction_from_state(st, "products.0", Float64)
    rng = Xoshiro(17)
    X = 0.3 .* randn(rng, 128, 16, 2); species = [29, 1]
    dB = randn(rng, 128, 4, 2)
    n = Tn.symmetric_contraction_scratch_size(sc)
    function f(Y)
        S = eltype(Y)
        B = zeros(S, 128, 4, 2)
        M.symmetric_contraction!(B, sc, Y, species, zeros(S, n))
        return sum(dB .* B)
    end
    dX = zeros(128, 16, 2)
    M.symmetric_contraction_pullback!(dX, sc, dB, X, species, zeros(n))
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
    using Random, OffsetArrays
    M = PureMLIP.Model
    Tn = PureMLIP.Tensors
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); st = ck.state.f64
    sc = M.symmetric_contraction_from_state(st, "products.0", Float64)
    X = 0.3 .* randn(Xoshiro(3), 128, 16, 2); species = [29, 1]
    n = Tn.symmetric_contraction_scratch_size(sc)
    B = zeros(128, 4, 2)
    M.symmetric_contraction!(B, sc, X, species, zeros(n))
    Xbig = zeros(130, 16, 3); Xbig[2:129, :, 2:3] = X
    Bbig = zeros(128, 5, 2)
    M.symmetric_contraction!(view(Bbig, :, 2:5, :), sc, view(Xbig, 2:129, :, 2:3), view([0, 29, 1], 2:3), view(zeros(n + 1), 2:(n + 1)))
    @test view(Bbig, :, 2:5, :) == B
    dB = ones(128, 4, 2)
    dX = zeros(128, 16, 2)
    M.symmetric_contraction_pullback!(dX, sc, dB, X, species, zeros(n))
    dXv = zeros(128, 17, 2)
    M.symmetric_contraction_pullback!(view(dXv, :, 2:17, :), sc, view(ones(128, 5, 2), :, 2:5, :), view(Xbig, 2:129, :, 2:3), species, zeros(n))
    @test view(dXv, :, 2:17, :) == dX
    @test_throws ArgumentError M.symmetric_contraction!(OffsetArray(B, 0, 0, -1), sc, OffsetArray(X, 0, 0, -1), species, zeros(n))
    @test_throws "model: symmetric contraction expects X of size (128, 16, n), got (128, 15, 2)" M.symmetric_contraction!(B, sc, X[:, 1:15, :], species, zeros(n))
    @test_throws "model: symmetric contraction expects B of size (128, 4, 2), got (128, 3, 2)" M.symmetric_contraction!(zeros(128, 3, 2), sc, X, species, zeros(n))
    @test_throws "model: symmetric contraction expects dB of size (128, 4, 2), got (128, 3, 2)" M.symmetric_contraction_pullback!(dX, sc, zeros(128, 3, 2), X, species, zeros(n))
    @test_throws "model: symmetric contraction expects 2 species, got 1" M.symmetric_contraction!(B, sc, X, [29], zeros(n))
    @test_throws "model: symmetric contraction scratch has $(n - 1) elements, needs $n" M.symmetric_contraction!(B, sc, X, species, zeros(n - 1))
    @test_throws "model: species index 90 is outside 1:89" M.symmetric_contraction!(B, sc, X, [29, 90], zeros(n))
    @test_throws "model: species index 0 is outside 1:89" M.symmetric_contraction_pullback!(dX, sc, dB, X, [0, 1], zeros(n))
    @test_throws "model: missing products.7.symmetric_contractions.contractions.0.U_matrix_1" M.symmetric_contraction_from_state(st, "products.7", Float64)
    # Direct construction validates the shapes of U and W against each other.
    U1 = zeros(16, 1); U2 = zeros(16, 16, 4); U3 = zeros(16, 16, 16, 23)
    W = zeros(89, 23, 128); W0 = zeros(89, 4, 128); W1 = zeros(89, 1, 128)
    @test Tn.Contraction{Float64}(0, U1, U2, U3, W, W0, W1) isa Tn.Contraction{Float64}
    @test_throws "model: symmetric contraction: U_matrix_3 has size (16, 16, 16, 23), weights_max has size (89, 22, 128)" Tn.Contraction{Float64}(0, U1, U2, U3, zeros(89, 22, 128), W0, W1)
    @test_throws "model: symmetric contraction: U_matrix_2 has size (16, 15, 4), expected (16, 16, K)" Tn.Contraction{Float64}(0, U1, zeros(16, 15, 4), U3, W, W0, W1)
    @test_throws "model: symmetric contraction: U_matrix_1 has size (16, 1), weights.1 has size (89, 2, 128)" Tn.Contraction{Float64}(0, U1, U2, U3, W, W0, zeros(89, 2, 128))
    @test_throws "model: symmetric contraction: weights.0 has size (88, 4, 128), expected (89, 4, 128)" Tn.Contraction{Float64}(0, U1, U2, U3, W, zeros(88, 4, 128), W1)
    @test_throws "model: symmetric contraction: l = 1 needs U arrays with a leading index of length 3" Tn.Contraction{Float64}(1, U1, U2, U3, W, W0, W1)
    @test_throws "model: symmetric contraction: l must be non-negative, got -1" Tn.Contraction{Float64}(-1, U1, U2, U3, W, W0, W1)
    @test_throws "model: symmetric contraction: l = 0 needs U arrays of 2, 3 and 4 dimensions" Tn.Contraction{Float64}(0, zeros(1, 16, 1), U2, U3, W, W0, W1)
    @test_throws "model: symmetric contraction: U_matrix_3 has size (16, 16, 15, 23), expected (16, 16, 16, K)" Tn.Contraction{Float64}(0, U1, U2, zeros(16, 16, 15, 23), W, W0, W1)
    @test_throws "model: symmetric contraction: U_matrix_2 has size (3, 16, 15, 6), expected (3, 16, 16, K)" Tn.Contraction{Float64}(1, zeros(3, 16, 1), zeros(3, 16, 15, 6), zeros(3, 16, 16, 16, 51), zeros(89, 51, 128), zeros(89, 6, 128), W1)
    @test_throws "model: symmetric contraction needs at least one contraction" Tn.SymmetricContraction{Float64}(Tn.Contraction{Float64}[])
    @test_throws "model: symmetric contraction expects dX of size (128, 16, 2), got (128, 16, 3)" M.symmetric_contraction_pullback!(zeros(128, 16, 3), sc, dB, X, species, zeros(n))
    @test_throws "model: symmetric contraction: contractions see different channel, ell or species counts" Tn.SymmetricContraction{Float64}([Tn.Contraction{Float64}(0, U1, U2, U3, W, W0, W1), Tn.Contraction{Float64}(0, U1, U2, U3, W[1:88, :, :], W0[1:88, :, :], W1[1:88, :, :])])
end
