@testitem "channelwise TP matches the e3nn-core-rs tensor-product fixtures" setup = [E3nnFixtures] begin
    M = PureMLIP.Model
    for name in ("fully_connected_tensor_product_smoke_f64", "fully_connected_tensor_product_lmax4_smoke_f64")
        d = E3nnFixtures.load(name)
        l1, l2, l3 = (E3nnFixtures.segment_degrees(d[k]) for k in ("irreps_in1", "irreps_in2", "irreps_out"))
        ins = [
            (
                d["instruction_in1_segment"][p] + 1, d["instruction_in2_segment"][p] + 1, d["instruction_out_segment"][p] + 1,
                Float64(d["instruction_path_weight"][p]),
            ) for p in eachindex(d["instruction_path_weight"])
        ]
        tp = M.ChannelwiseTP{Float64}(l1, l2, l3, ins)
        for s in eachindex(d["x"])
            x = reshape(Float64.(d["x"][s]), 1, :); y = Float64.(d["y"][s]); w = reshape(Float64.(d["w"][s]), 1, :)
            out = zeros(1, length(d["out"][s]))
            M.tp_edge_add!(out, tp, x, y, w)
            # Oracle tolerance: e3nn-core-rs fixtures are stated to 1e-10.
            @test vec(out) ≈ d["out"][s] atol = 1.0e-10
            dx = zero(x); dy = zero(y); dw = zero(w)
            M.tp_edge_pullback_add!(dx, dy, dw, tp, reshape(Float64.(d["grad_out"][s]), 1, :), x, y, w)
            @test vec(dx) ≈ d["grad_x"][s] atol = 1.0e-10
            @test dy ≈ d["grad_y"][s] atol = 1.0e-10
            @test vec(dw) ≈ d["grad_w"][s] atol = 1.0e-10
        end
    end
end

@testitem "MACE layers match the forward-pass formulas" setup = [Checkpoints] begin
    M = PureMLIP.Model
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); st = ck.state.f64; k = ck.constants
    # Folded skip_tp constants, bit-exact against the embedded TorchScript literals.
    sk = M.SpeciesLinear{Float64}(
        st["interactions.0.skip_tp.weight"].data, [0, 1, 2, 3],
        [(p, p, k.tensor_product["interactions.0.skip_tp"].instructions[p].path_weight) for p in 1:4], 128, 89, 128
    )
    @test sk.c == [0.0093691461124482678, 0.0093691461124482695, 0.0093691461124482678, 0.009369146112448266]
    using Random
    x = randn(Xoshiro(3), 128, 16, 3)
    up = M.E3nnLinear{Float64}(st["interactions.0.linear_up.weight"].data, [0], [0], [(1, 1, 0.08838834764831845)], 128, 128)
    y = zeros(128, 1, 3); M.linear!(y, up, view(x, :, 1:1, :))
    W = st["interactions.0.linear_up.weight"].data
    naive = [0.08838834764831845 * sum(x[u, 1, i] * W[(u - 1) * 128 + w] for u in 1:128) for w in 1:128, i in 1:3]
    # Rounding bound: 4·eps·N·max|terms|, N = 128 terms per output.
    @test y[:, 1, :] ≈ naive atol = 4 * eps() * 128 * maximum(abs, x) * maximum(abs, W)

    # Degree `l` of e3nn component `q` (1-based) in the 16-component 0..3 layout: q - 1 = l² + m.
    deg(q) = isqrt(q - 1)

    # interactions.0.linear: one [128, 128] block per l, torch index (l·128 + u-1)·128 + w.
    W = st["interactions.0.linear.weight"].data
    lin = M.E3nnLinear{Float64}(W, [0, 1, 2, 3], [0, 1, 2, 3], [(p, p, 0.08838834764831845) for p in 1:4], 128, 128)
    y = zeros(128, 16, 3); M.linear!(y, lin, x)
    naive = [
        0.08838834764831845 * sum(x[u, q, i] * W[(deg(q) * 128 + u - 1) * 128 + w] for u in 1:128)
            for w in 1:128, q in 1:16, i in 1:3
    ]
    # Rounding bound: 4·eps·N·max|terms|, N = 128 terms per output.
    @test y ≈ naive atol = 4 * eps() * 128 * maximum(abs, x) * maximum(abs, W)

    # interactions.1.linear: ten [128, 128] blocks, slots 0e,0e,1o,1o,1o,2e,2e,2e,3o,3o into
    # 0e+1o+2e+3o; slots of one degree sum into the same output.
    W = st["interactions.1.linear.weight"].data
    a, b = 0.0625, 0.05103103630798288
    in_ls = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    out_slot = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
    pws = [a, a, b, b, b, b, b, b, a, a]
    lin1 = M.E3nnLinear{Float64}(W, in_ls, [0, 1, 2, 3], [(p, out_slot[p], pws[p]) for p in 1:10], 128, 128)
    x10 = randn(Xoshiro(4), 128, 40, 3)
    off = cumsum([0; [2l + 1 for l in in_ls]])
    y = zeros(128, 16, 3); M.linear!(y, lin1, x10)
    naive = [
        sum(
            pws[p] * sum(x10[u, off[p] + q - deg(q)^2, i] * W[((p - 1) * 128 + u - 1) * 128 + w] for u in 1:128)
                for p in 1:10 if out_slot[p] == deg(q) + 1
        ) for w in 1:128, q in 1:16, i in 1:3
    ]
    # Rounding bound: 4·eps·N·max|terms|, N = 384 terms per output (three slots of 128).
    @test y ≈ naive atol = 4 * eps() * 384 * maximum(abs, x10) * maximum(abs, W)

    # skip_tp: c_l Σ_u W[u, s, w]·x[u, m], torch index ((u-1)·89 + s-1)·128 + w per l block.
    species = [5, 89, 1]
    cl = [0.0093691461124482678, 0.0093691461124482695, 0.0093691461124482678, 0.009369146112448266]
    W = st["interactions.0.skip_tp.weight"].data
    y = zeros(128, 16, 3); M.species_linear!(y, sk, x, species)
    blk = 128 * 89 * 128
    naive = [
        cl[deg(q) + 1] * sum(x[u, q, i] * W[deg(q) * blk + ((u - 1) * 89 + species[i] - 1) * 128 + w] for u in 1:128)
            for w in 1:128, q in 1:16, i in 1:3
    ]
    # Rounding bound: 4·eps·N·max|terms|, N = 128 terms per output.
    @test y ≈ naive atol = 4 * eps() * 128 * maximum(abs, x) * maximum(abs, W)
    W = st["interactions.1.skip_tp.weight"].data
    sk1 = M.SpeciesLinear{Float64}(W, [0], [(1, 1, k.tensor_product["interactions.1.skip_tp"].instructions[1].path_weight)], 128, 89, 128)
    y = zeros(128, 1, 3); M.species_linear!(y, sk1, view(x, :, 1:1, :), species)
    naive = [cl[1] * sum(x[u, 1, i] * W[((u - 1) * 89 + species[i] - 1) * 128 + w] for u in 1:128) for w in 1:128, i in 1:3]
    # Rounding bound: 4·eps·N·max|terms|, N = 128 terms per output.
    @test y[:, 1, :] ≈ naive atol = 4 * eps() * 128 * maximum(abs, x) * maximum(abs, W)

    # conv_tp_weights: a1 = cst·silu(feats·L0/√10), a2, a3 likewise with /8, out = a3·L3/8.
    cst = k.normalize2mom["interactions.0.conv_tp_weights.layer0.act"].cst
    @test cst == 1.679176792398942
    Ls = ntuple(j -> torch_array(st["interactions.0.conv_tp_weights.layer$(j - 1).weight"], Val(2)), 4)
    nb = 7
    feats = randn(Xoshiro(5), 10, nb)
    silu(z) = z / (1 + exp(-z))
    for T in (Float64, Float32)
        local mlp = M.RadialMLP{T}(Ls, cst)
        local fT = T.(feats)
        local out = zeros(T, 512, nb); local pre = zeros(T, 64, nb, 3); local act = zeros(T, 64, nb, 3)
        M.radial_mlp!(out, pre, act, mlp, fT)
        local ins = (fT, act[:, :, 1], act[:, :, 2], act[:, :, 3])
        for j in 1:4
            local L = Ls[j]
            local h = size(L, 1)
            local X = Float64.(ins[j])
            # The oracle z = naive is computed in Float64 from the same inputs.
            local naive = [sum(X[c, e] * (L[c, o] / sqrt(h)) for c in 1:h) for o in 1:size(L, 2), e in 1:nb]
            # S[o, e]: the sum of the absolute terms of naive[o, e].
            local S = [sum(abs(X[c, e] * L[c, o]) for c in 1:h) / sqrt(h) for o in 1:size(L, 2), e in 1:nb]
            if j == 4
                # Rounding bound: 4·eps·N·max|terms|, N = h (10 or 64) terms per output.
                @test out ≈ naive atol = 4 * eps(T) * h * maximum(abs, X) * maximum(abs, L) / sqrt(h)
            else
                # `pre` holds the logistic σ(z) of the pre-activations z. Rounding bound: each z is
                # within γ_h·S of the exact value (the Float32 kernel rounds each weight once more,
                # and its Float64 oracle is exact to this order), so the computed z and the oracle
                # differ by at most h·eps·S; σ' = σ(1 - σ), and the factor 1.1 covers its change
                # over that gap; 4·eps covers the rounding of σ itself (exp, one add, one inverse).
                local s = 1 ./ (1 .+ exp.(-naive))
                @test all(abs.(pre[:, :, j] .- s) .<= 4 * eps(T) .+ 1.1 .* s .* (1 .- s) .* h .* eps(T) .* S)
                # Rounding bound: 8·eps relative per element (exp, one add, one division, one
                # product), plus the slope |∂act/∂z| ≤ 1.1·cst times the h·eps·S gap in z.
                @test all(abs.(act[:, :, j] .- cst .* silu.(naive)) .<= 8 * eps(T) .* abs.(act[:, :, j]) .+ 1.1 * cst * h * eps(T) .* S)
            end
        end
    end

    # density_fn: ρ = tanh((feats · L/√10)²).
    Ld = st["interactions.0.density_fn.layer0.weight"].data
    for e in 1:nb
        ρ, _ = M.density_edge(view(feats, :, e), Ld)
        z = sum(feats[c, e] * (Ld[c] / sqrt(10)) for c in 1:10)
        zb = sum(abs(feats[c, e] * Ld[c]) for c in 1:10) / sqrt(10)
        # Rounding bound: 4·eps·N·Σ|terms| on z (N = 10), times |∂ρ/∂z| ≤ 2|z|, plus 4·eps for tanh.
        @test ρ ≈ tanh(z^2) atol = 4 * eps() * 10 * zb * 2 * abs(z) + 4 * eps()
    end

    # readouts.1.linear_1: [128, 16], torch index (u-1)·16 + w.
    W = st["readouts.1.linear_1.weight"].data
    r1 = M.E3nnLinear{Float64}(W, [0], [0], [(1, 1, 0.08838834764831845)], 128, 16)
    y = zeros(16, 1, 3); M.linear!(y, r1, view(x, :, 1:1, :))
    naive = [0.08838834764831845 * sum(x[u, 1, i] * W[(u - 1) * 16 + w] for u in 1:128) for w in 1:16, i in 1:3]
    # Rounding bound: 4·eps·N·max|terms|, N = 128 terms per output.
    @test y[:, 1, :] ≈ naive atol = 4 * eps() * 128 * maximum(abs, x) * maximum(abs, W)
end

@testitem "layer pullbacks match ForwardDiff" begin
    using ForwardDiff, Random, LinearAlgebra
    M = PureMLIP.Model
    rng = Xoshiro(11)
    # Inputs, weights and cotangents are uniform in [-1, 1], so every product of them is at most 1
    # in magnitude and the rounding bounds below need only the term count and the constants.
    u(dims...) = 2 .* rand(rng, dims...) .- 1

    # Central difference of L(x) = sum(dy .* f(x)) along v, against dot(pullback, v).
    function fd(f, dy, x, v, h)
        L(t) = sum(dy .* f(x .+ t .* v))
        return (L(h) - L(-h)) / (2h)
    end

    # E3nnLinear: slots 0e, 0e, 1o, 2e into 0e, 1o, 2e; both 0e slots feed output 0e, and the
    # 2e input slot has no instruction.
    C_in, C_out, n = 5, 4, 3
    ins = [(1, 1, 0.3), (2, 1, 0.7), (3, 2, 0.5)]
    lin = M.E3nnLinear{Float64}(u(length(ins) * C_in * C_out), [0, 0, 1, 2], [0, 1, 2], ins, C_in, C_out)
    flin(x) = M.linear!(zeros(eltype(x), C_out, 9, n), lin, x)
    x = u(C_in, 10, n); dy = u(C_out, 9, n)
    dx = M.linear_pullback!(zeros(C_in, 10, n), lin, dy)
    g = ForwardDiff.gradient(x -> sum(dy .* flin(x)), x)
    # Rounding bound: 4·eps·N per element, N = C_out terms (|pw·W·dy| ≤ 1).
    @test dx ≈ g atol = 4 * eps() * C_out
    @test all(iszero, dx[:, 10:end, :])
    v = u(size(x)...)
    # Measured margin: largest error 7.5e-12 at h = 1e-5 (rounding of the difference quotient).
    @test fd(flin, dy, x, v, 1.0e-5) ≈ dot(dx, v) atol = 1.5e-10

    # SpeciesLinear: slots 0e, 1o, one weight block per slot and species.
    nS = 3
    sl = M.SpeciesLinear{Float64}(u(2 * C_in * nS * C_out), [0, 1], [(1, 1, 0.9), (2, 2, 0.6)], C_in, nS, C_out)
    species = [2, 3, 2]
    fsl(x) = M.species_linear!(zeros(eltype(x), C_out, 4, n), sl, x, species)
    x = u(C_in, 4, n); dy = u(C_out, 4, n)
    dx = M.species_linear_pullback!(zeros(C_in, 4, n), sl, dy, species)
    g = ForwardDiff.gradient(x -> sum(dy .* fsl(x)), x)
    # Rounding bound: 4·eps·N per element, N = C_out terms (|c·W·dy| ≤ 1).
    @test dx ≈ g atol = 4 * eps() * C_out
    v = u(size(x)...)
    # Measured margin: largest error 9.4e-12 at h = 1e-5 (rounding of the difference quotient).
    @test fd(fsl, dy, x, v, 1.0e-5) ≈ dot(dx, v) atol = 1.9e-10

    # ChannelwiseTP with the interaction-1 conv_tp structure: in1 0e+1o, in2 0e+1o+2e+3o, ten outputs.
    C = 4
    tins = [
        (1, 1, 1, 1.0), (2, 2, 2, 1.0), (1, 2, 3, 1.7320508075688772), (2, 1, 4, 1.7320508075688772),
        (2, 3, 5, 1.7320508075688772), (1, 3, 6, 2.23606797749979), (2, 2, 7, 2.23606797749979),
        (2, 4, 8, 2.23606797749979), (1, 4, 9, 2.6457513110645907), (2, 3, 10, 2.6457513110645907),
    ]
    tp = M.ChannelwiseTP{Float64}([0, 1], [0, 1, 2, 3], [0, 0, 1, 1, 1, 2, 2, 2, 3, 3], tins)
    nx, nY, nw = C * 4, 16, C * 10
    unpack(z) = (reshape(z[1:nx], C, 4), z[(nx + 1):(nx + nY)], reshape(z[(nx + nY + 1):end], C, 10))
    function ftp(z)
        x, Y, w = unpack(z)
        return M.tp_edge_add!(zeros(eltype(z), C, 40), tp, x, Y, w)
    end
    z = u(nx + nY + nw); dM = u(C, 40)
    x, Y, w = unpack(z)
    dx, dY, dw = zero(x), zero(Y), zero(w)
    @test M.tp_edge_pullback_add!(dx, dY, dw, tp, dM, x, Y, w) === nothing
    g = ForwardDiff.gradient(z -> sum(dM .* ftp(z)), z)
    nnz = length(tp.entries)
    vmax = maximum(e -> abs(e[4]), tp.entries)
    # Rounding bound: 4·eps·N·max|terms|, N = nnz·C terms at most per gradient entry, each at
    # most max|pw·C| in magnitude.
    @test [vec(dx); dY; vec(dw)] ≈ g atol = 4 * eps() * nnz * C * vmax
    # The pullback accumulates: from a start of ones the result is 1 + gradient, within the
    # rounding bound above plus eps for the extra addition.
    dx2, dY2, dw2 = one.(dx), one.(dY), one.(dw)
    M.tp_edge_pullback_add!(dx2, dY2, dw2, tp, dM, x, Y, w)
    @test [vec(dx2); dY2; vec(dw2)] .- 1 ≈ [vec(dx); dY; vec(dw)] atol = 4 * eps() * nnz * C * vmax + 2 * eps()
    v = u(length(z))
    # Measured margin: largest error 1.2e-10 at h = 1e-5.
    @test fd(ftp, dM, z, v, 1.0e-5) ≈ dot([vec(dx); dY; vec(dw)], v) atol = 2.4e-9

    # RadialMLP: 10 → 6 → 6 → 6 → 12.
    cst = 1.679176792398942
    Ls = (u(10, 6), u(6, 6), u(6, 6), u(6, 12))
    mlp = M.RadialMLP{Float64}(Ls, cst)
    nb = 5
    function fmlp(f)
        S = eltype(f)
        return M.radial_mlp!(zeros(S, 12, nb), zeros(S, 6, nb, 3), zeros(S, 6, nb, 3), mlp, f)
    end
    f = u(10, nb); dout = u(12, nb)
    pre = zeros(6, nb, 3); act = zeros(6, nb, 3)
    M.radial_mlp!(zeros(12, nb), pre, act, mlp, f)
    df = M.radial_mlp_pullback!(zeros(10, nb), zeros(6, nb, 2), mlp, dout, pre, act)
    g = ForwardDiff.gradient(f -> sum(dout .* fmlp(f)), f)
    # Rounding bound for a chain of products: 4·eps·N·(|W1|·|D1|·|W2|·|D2|·|W3|·|D3|·|W4|·|dout|)
    # elementwise, N = 10 + 3·6 + 12 terms along the chain, |Dk| ≤ 1.1·cst the activation slope.
    D = 1.1 * cst
    B = abs.(mlp.W[1]) * (D .* (abs.(mlp.W[2]) * (D .* (abs.(mlp.W[3]) * (D .* (abs.(mlp.W[4]) * abs.(dout)))))))
    @test all(abs.(df .- g) .<= 4 * eps() * 40 .* B)
    v = u(size(f)...)
    # Measured margin: largest error 7.6e-13 at h = 1e-5.
    @test fd(fmlp, dout, f, v, 1.0e-5) ≈ dot(df, v) atol = 1.5e-11

    # density_edge: ∂ρ/∂feats = c .* w.
    w = u(10)
    for _ in 1:5
        local f = 3 .* u(10)
        ρ, c = M.density_edge(f, w)
        local g = ForwardDiff.gradient(f -> M.density_edge(f, w)[1], f)
        # Rounding bound: 8·eps relative on the final products (2z(1-ρ²)/√n, then times w).
        @test c .* w ≈ g atol = 8 * eps() * maximum(abs, g) + eps()
        local v = u(10)
        # Measured margin: largest error 4.4e-12 over the five draws at h = 1e-5.
        @test (M.density_edge(f .+ 1.0e-5 .* v, w)[1] - M.density_edge(f .- 1.0e-5 .* v, w)[1]) / 2.0e-5 ≈ c * dot(w, v) atol = 9.0e-11
    end

    # normalized_silu: da = ∂a/∂z.
    for z in (-40.0, -5.0, -0.3, 0.0, 0.7, 6.0, 40.0)
        a, da = M.normalized_silu(z, cst)
        @test a ≈ cst * z / (1 + exp(-z)) atol = 8 * eps() * abs(a)
        # Rounding bound: 8·eps·cst, a few operations of magnitude ≤ 1.1·cst.
        @test da ≈ ForwardDiff.derivative(z -> M.normalized_silu(z, cst)[1], z) atol = 8 * eps() * cst
        # Measured margin: largest error 2.4e-10 (at z = 40) at h = 1e-5.
        @test (M.normalized_silu(z + 1.0e-5, cst)[1] - M.normalized_silu(z - 1.0e-5, cst)[1]) / 2.0e-5 ≈ da atol = 4.8e-9
    end
end

@testitem "TP pullback without dx matches the full pullback" begin
    using Random
    M = PureMLIP.Model
    rng = Xoshiro(17)
    # The interaction-1 conv_tp structure; C = 37 is not a multiple of any SIMD width, so both
    # the vector body and the remainder of the channel loop run.
    C = 37
    tins = [
        (1, 1, 1, 1.0), (2, 2, 2, 1.0), (1, 2, 3, 1.7320508075688772), (2, 1, 4, 1.7320508075688772),
        (2, 3, 5, 1.7320508075688772), (1, 3, 6, 2.23606797749979), (2, 2, 7, 2.23606797749979),
        (2, 4, 8, 2.23606797749979), (1, 4, 9, 2.6457513110645907), (2, 3, 10, 2.6457513110645907),
    ]
    for T in (Float64, Float32)
        tp = M.ChannelwiseTP{T}([0, 1], [0, 1, 2, 3], [0, 0, 1, 1, 1, 2, 2, 2, 3, 3], tins)
        x, Y, w, dM = randn(rng, T, C, 4), randn(rng, T, 16), randn(rng, T, C, 10), randn(rng, T, C, 40)
        dY0, dw0 = randn(rng, T, 16), randn(rng, T, C, 10)
        dx, dY, dw = zeros(T, C, 4), copy(dY0), copy(dw0)
        M.tp_edge_pullback_add!(dx, dY, dw, tp, dM, x, Y, w)
        dxn, dYn, dwn = fill(T(NaN), C, 4), copy(dY0), copy(dw0)
        @test M.tp_edge_pullback_add!(dxn, dYn, dwn, tp, dM, x, Y, w; need_dx = false) === nothing
        # dw: each element gets the same operations in the same order in both loops, so it is equal.
        @test dwn == dw
        # dY[j] sums `v * acc` over the n_j entries with that j, and each `acc` is a `@simd`
        # reduction over C channels whose order LLVM chooses per loop, so the two loops may round
        # differently. Rounding bound, to first order: each result is within (C + 2 + n_j)·eps/2·S_j
        # of the exact value (C three-factor products and C - 1 additions, the scaling by v, n_j
        # additions into dY[j]), where S_j = |dY0[j]| + Σ|v|·Σ_u|dM·w·x| is the sum of the absolute
        # terms; their difference is within c·eps·S_j with c = C + 2 + n_j.
        S, nj = abs.(dY0), zeros(Int, length(dY0))
        for p in eachindex(tp.paths), q in tp.paths[p]
            i, j, k, v = tp.entries[q]
            S[j] += abs(v) * sum(abs.(dM[:, k] .* w[:, p] .* x[:, i]))
            nj[j] += 1
        end
        @test all(abs.(dYn .- dY) .<= (C .+ 2 .+ nj) .* eps(T) .* S)
        @test all(isnan, dxn)
    end
end

@testitem "layers accept views" begin
    using Random, OffsetArrays
    M = PureMLIP.Model
    rng = Xoshiro(21)
    C_in, C_out, n = 6, 4, 3
    lin = M.E3nnLinear{Float64}(randn(rng, 2 * C_in * C_out), [0, 1], [0, 1], [(1, 1, 0.4), (2, 2, 0.4)], C_in, C_out)
    sl = M.SpeciesLinear{Float64}(randn(rng, 2 * C_in * 2 * C_out), [0, 1], [(1, 1, 0.9), (2, 2, 0.6)], C_in, 2, C_out)
    species = [2, 1, 2]
    X = randn(rng, C_in + 2, 7, n + 2)
    x = view(X, 2:(C_in + 1), 2:5, 2:(n + 1))
    xc = copy(x)
    # Two evaluations of the same sums in possibly different orders: twice the rounding bound
    # 4·eps·N·max|terms|, N = C_in.
    tol(W) = 2 * 4 * eps() * C_in * maximum(abs, x) * maximum(abs, W)
    for (f!, W, y0) in (
            ((y, x) -> M.linear!(y, lin, x), lin.W[1], zeros(C_out, 4, n)),
            ((y, x) -> M.species_linear!(y, sl, x, species), sl.W[1], zeros(C_out, 4, n)),
        )
        f!(y0, xc)
        Y = fill(7.0, C_out + 3, 6, n + 1)
        f!(view(Y, 3:(C_out + 2), 2:5, 1:n), x)
        @test Y[3:(C_out + 2), 2:5, 1:n] ≈ y0 atol = tol(W)
        Y[3:(C_out + 2), 2:5, 1:n] .= 7.0
        @test all(==(7.0), Y)   # entries outside the view are untouched
        @test_throws "offset arrays are not supported" f!(OffsetArray(zeros(C_out, 4, n), 0, 0, -1), xc)
    end
    dx0 = M.linear_pullback!(zeros(C_in, 4, n), lin, ones(C_out, 4, n))
    D = zeros(C_in + 1, 4, n)
    M.linear_pullback!(view(D, 2:(C_in + 1), :, :), lin, view(ones(C_out, 5, n), :, 2:5, :))
    @test D[2:end, :, :] ≈ dx0 atol = tol(lin.W[1])
    @test all(iszero, D[1, :, :])
end

@testitem "layers reject inconsistent shapes" begin
    M = PureMLIP.Model
    @test_throws "model: linear weight has 3 entries, expected 4" M.E3nnLinear{Float64}(zeros(3), [0], [0], [(1, 1, 1.0)], 2, 2)
    @test_throws "model: linear instruction (2, 1) is outside the 1 input and 1 output slots" M.E3nnLinear{Float64}(zeros(4), [0], [0], [(2, 1, 1.0)], 2, 2)
    @test_throws "model: linear instruction (1, 1) couples l = 0 to l = 1" M.E3nnLinear{Float64}(zeros(4), [0], [1], [(1, 1, 1.0)], 2, 2)
    lin = M.E3nnLinear{Float64}(zeros(4), [0], [0], [(1, 1, 1.0)], 2, 2)
    @test_throws "model: linear input has size (3, 1, 2), expected (2, 1, n)" M.linear!(zeros(2, 1, 2), lin, zeros(3, 1, 2))
    @test_throws "model: linear output has size (2, 1, 3), expected (2, 1, 2)" M.linear!(zeros(2, 1, 3), lin, zeros(2, 1, 2))
    @test_throws "model: linear output has size (2, 2, 2), expected (2, 1, 2)" M.linear_pullback!(zeros(2, 1, 2), lin, zeros(2, 2, 2))

    @test_throws "model: species linear weight has 3 entries, expected 8" M.SpeciesLinear{Float64}(zeros(3), [0], [(1, 1, 1.0)], 2, 2, 2)
    @test_throws "model: species linear instruction (1, 2) is outside the 1 slots" M.SpeciesLinear{Float64}(zeros(8), [0], [(1, 2, 1.0)], 2, 2, 2)
    @test_throws "model: species linear instruction (1, 2) couples l = 0 to l = 1" M.SpeciesLinear{Float64}(zeros(16), [0, 1], [(1, 2, 1.0), (2, 2, 1.0)], 2, 2, 2)
    sl = M.SpeciesLinear{Float64}(zeros(8), [0], [(1, 1, 1.0)], 2, 2, 2)
    @test_throws "model: species linear input has size (3, 1, 2), expected (2, 1, n)" M.species_linear!(zeros(2, 1, 2), sl, zeros(3, 1, 2), [1, 1])
    @test_throws "model: species linear output has size (2, 1, 1), expected (2, 1, 2)" M.species_linear_pullback!(zeros(2, 1, 2), sl, zeros(2, 1, 1), [1, 1])
    @test_throws "model: 3 species for 2 nodes" M.species_linear!(zeros(2, 1, 2), sl, zeros(2, 1, 2), [1, 1, 1])
    @test_throws "model: species index 3 is outside 1:2" M.species_linear!(zeros(2, 1, 2), sl, zeros(2, 1, 2), [1, 3])

    @test_throws "model: tensor-product instruction (1, 2, 1) is outside the 1, 1 and 1 slots" M.ChannelwiseTP{Float64}([0], [0], [0], [(1, 2, 1, 1.0)])
    @test_throws ArgumentError M.ChannelwiseTP{Float64}([1], [1], [1], [(1, 1, 1, 1.0)])   # odd l1 + l2 + l3
    tp = M.ChannelwiseTP{Float64}([0], [1], [1], [(1, 1, 1, 1.0)])
    @test_throws "model: tensor product with 2 channels: x has size (2, 2), expected (2, 1)" M.tp_edge_add!(zeros(2, 3), tp, zeros(2, 2), zeros(3), zeros(2, 1))
    @test_throws "model: tensor product with 2 channels: Y has length 2, expected 3" M.tp_edge_add!(zeros(2, 3), tp, zeros(2, 1), zeros(2), zeros(2, 1))
    @test_throws "model: tensor product with 2 channels: w has size (2, 2), expected (2, 1)" M.tp_edge_add!(zeros(2, 3), tp, zeros(2, 1), zeros(3), zeros(2, 2))
    @test_throws "model: tensor product with 2 channels: the output has size (2, 2), expected (2, 3)" M.tp_edge_add!(zeros(2, 2), tp, zeros(2, 1), zeros(3), zeros(2, 1))
    @test_throws "model: tensor product with 2 channels: dx has size (2, 2), expected (2, 1)" M.tp_edge_pullback_add!(zeros(2, 2), zeros(3), zeros(2, 1), tp, zeros(2, 3), zeros(2, 1), zeros(3), zeros(2, 1))
    @test_throws "model: tensor product with 2 channels: dY has length 2, expected 3" M.tp_edge_pullback_add!(zeros(2, 1), zeros(2), zeros(2, 1), tp, zeros(2, 3), zeros(2, 1), zeros(3), zeros(2, 1))
    @test_throws "model: tensor product with 2 channels: dw has size (2, 2), expected (2, 1)" M.tp_edge_pullback_add!(zeros(2, 1), zeros(3), zeros(2, 2), tp, zeros(2, 3), zeros(2, 1), zeros(3), zeros(2, 1))

    @test_throws "model: radial MLP layer 2 has 5 inputs, expected 6" M.RadialMLP{Float64}((zeros(3, 6), zeros(5, 6), zeros(6, 6), zeros(6, 2)), 1.0)
    @test_throws "model: radial MLP hidden layers must share one width, got 6 and 5" M.RadialMLP{Float64}((zeros(3, 6), zeros(6, 5), zeros(5, 6), zeros(6, 2)), 1.0)
    mlp = M.RadialMLP{Float64}((zeros(3, 6), zeros(6, 6), zeros(6, 6), zeros(6, 2)), 1.0)
    @test_throws "model: radial MLP features have 4 rows, expected 3" M.radial_mlp!(zeros(2, 5), zeros(6, 5, 3), zeros(6, 5, 3), mlp, zeros(4, 5))
    @test_throws "model: radial MLP output has size (2, 4), expected (2, 5)" M.radial_mlp!(zeros(2, 4), zeros(6, 5, 3), zeros(6, 5, 3), mlp, zeros(3, 5))
    @test_throws "model: radial MLP activations have size (6, 4, 3), expected (6, 5, 3)" M.radial_mlp!(zeros(2, 5), zeros(6, 5, 3), zeros(6, 4, 3), mlp, zeros(3, 5))
    @test_throws "model: radial MLP logistic buffer has size (6, 5, 2), expected (6, 5, 3)" M.radial_mlp!(zeros(2, 5), zeros(6, 5, 2), zeros(6, 5, 3), mlp, zeros(3, 5))
    @test_throws "model: radial MLP scratch has size (6, 5, 3), expected (6, 5, 2)" M.radial_mlp_pullback!(zeros(3, 5), zeros(6, 5, 3), mlp, zeros(2, 5), zeros(6, 5, 3), zeros(6, 5, 3))
    @test_throws "model: radial MLP feature gradient has size (3, 4), expected (3, 5)" M.radial_mlp_pullback!(zeros(3, 4), zeros(6, 5, 2), mlp, zeros(2, 5), zeros(6, 5, 3), zeros(6, 5, 3))
    @test_throws "model: radial MLP output cotangent has 3 rows, expected 2" M.radial_mlp_pullback!(zeros(3, 5), zeros(6, 5, 2), mlp, zeros(3, 5), zeros(6, 5, 3), zeros(6, 5, 3))
    # The message comes from Base's eachindex over two vectors, so only the type is checked.
    @test_throws DimensionMismatch M.density_edge(zeros(3), zeros(4))
end
