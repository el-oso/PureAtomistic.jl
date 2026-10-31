@testitem "radial functions match the MACE radial fixture" setup = [E3nnFixtures] begin
    M = PureMLIP.Model
    f = E3nnFixtures.radial()                       # parsed mace_radial_reference_f64.json
    Z = [1, 6, 8]; R = Dict(1 => 0.31, 6 => 0.76, 8 => 0.66)   # MACE covalent radii (Cordero 2008)
    snd = f["graph"]["edge_index"][1] .+ 1; rcv = f["graph"]["edge_index"][2] .+ 1
    d = [x[1] for x in f["graph"]["distances"]]
    bp = f["bessel"]["parameters"]; nb = bp["num_basis"]
    for e in eachindex(d)
        B = zeros(nb); dB = zeros(nb)
        M.bessel_basis!(B, dB, d[e], Float64.(bp["bessel_weights"]), bp["prefactor"])
        fc, dfc = M.polynomial_cutoff(d[e], bp["r_max"], bp["cutoff_p"])
        # Oracle tolerance: e3nn-core-rs fixtures are stated to 1e-10.
        @test B ≈ Float64.(f["bessel"]["basis"][e]) atol = 1.0e-10
        @test fc ≈ f["bessel"]["cutoff"][e][1] atol = 1.0e-10
        @test B .* fc ≈ Float64.(f["bessel"]["embedding"][e]) atol = 1.0e-10
        @test sum(Float64.(f["bessel"]["cotangent_embedding"][e]) .* (dB .* fc .+ B .* dfc)) ≈ f["bessel"]["vjp_distances"][e][1] atol = 1.0e-10
        ap = f["agnesi"]["parameters"]
        t, dt = M.agnesi_transform(d[e], 0.5 * (R[Z[snd[e]]] + R[Z[rcv[e]]]), ap["a"], ap["p"], ap["q"])
        @test t ≈ f["agnesi"]["output"][e][1] atol = 1.0e-10
        @test f["agnesi"]["cotangent"][e][1] * dt ≈ f["agnesi"]["vjp_distances"][e][1] atol = 1.0e-10
    end
    zp = f["zbl"]["parameters"]
    radii = zeros(119); for (z, r) in R; radii[z + 1] = r; end
    zbl = M.ZBLParams{Float64}(Tuple(Float64.(zp["c"])), Float64(zp["p"]), zp["a_exp"], zp["a_prefactor"], radii)
    V = zeros(3); vjp = zeros(length(d))
    for e in eachindex(d)
        v, dv = M.zbl_edge_energy(d[e], Z[snd[e]], Z[rcv[e]], zbl)
        V[rcv[e]] += v; vjp[e] = f["zbl"]["cotangent_node_energy"][rcv[e]][1] * dv
    end
    @test V ≈ Float64.(f["zbl"]["output_node_energy"]) atol = 1.0e-10
    @test vjp ≈ [x[1] for x in f["zbl"]["vjp_distances"]] atol = 1.0e-10
end

@testitem "radial derivatives match ForwardDiff and finite differences" begin
    using ForwardDiff
    M = PureMLIP.Model
    radii = fill(0.2, 119); radii[30] = 1.32; radii[14] = 1.21       # Cu, Al
    emb = M.RadialEmbedding{Float64}(pi / 6.0 .* (1:10), sqrt(2 / 6), 6.0, 5, 1.0805, 4.5791, 0.9183, radii)
    for d in 0.6:0.37:5.95
        F = zeros(10); dF = zeros(10)
        M.radial_embedding!(F, dF, emb, d, 29, 13)
        J = ForwardDiff.derivative(x -> M.radial_embedding!(zeros(typeof(x), 10), zeros(typeof(x), 10), emb, x, 29, 13), d)
        # Measured margin: the largest elementwise |dF - J| over this scan is 68·eps. Both the
        # analytic derivative and the dual-number derivative sit within 54·eps of a BigFloat
        # evaluation (cancellation between the two terms of d/dx sin(wx)/x), so the pair
        # cannot agree much better; tolerance 128·eps·max(1, max|J|), under 2 times the margin.
        @test maximum(abs, dF .- J) <= 128 * eps() * max(1.0, maximum(abs, J))
    end
    F = zeros(10); dF = zeros(10)
    @test M.radial_embedding!(F, dF, emb, 6.0, 29, 29) == zeros(10) && dF == zeros(10)      # at r_max: exactly zero
end

@testitem "ZBL edge derivative matches a central finite difference" begin
    M = PureMLIP.Model
    radii = fill(0.2, 119); radii[30] = 1.32; radii[2] = 0.31; radii[7] = 0.76    # Cu, H, C
    zbl = M.ZBLParams{Float64}((0.1818, 0.5099, 0.2802, 0.02817), 6.0, 0.3, 0.4543, radii)
    h = 1.0e-6
    for (Zu, Zv, d, tol) in ((29, 29, 2.5, 2.0e-12), (1, 6, 1.0, 5.0e-9))
        e, de = M.zbl_edge_energy(d, Zu, Zv, zbl)
        e⁺, _ = M.zbl_edge_energy(d + h, Zu, Zv, zbl)
        e⁻, _ = M.zbl_edge_energy(d - h, Zu, Zv, zbl)
        @test e > 0
        # Measured margin: |de - fd| is 1.4e-13 (Cu-Cu) and 2.3e-10 (H-C); the tolerance is
        # 14 and 22 times these. The error is the O(h²) truncation of the central difference.
        @test de ≈ (e⁺ - e⁻) / 2h atol = tol
    end
    @test M.zbl_edge_energy(2.64, 29, 29, zbl) == (0.0, 0.0)      # at R_u + R_v: exactly zero
end

@testitem "radial embedding accepts views and rejects mismatched axes" begin
    using OffsetArrays
    M = PureMLIP.Model
    radii = fill(0.2, 119)
    emb = M.RadialEmbedding{Float64}(pi / 6.0 .* (1:4), 0.5, 6.0, 5, 1.0805, 4.5791, 0.9183, radii)
    F = zeros(4); dF = zeros(4)
    M.radial_embedding!(F, dF, emb, 1.3, 6, 8)
    buf = zeros(8); dbuf = zeros(8)
    M.radial_embedding!(view(buf, 3:6), view(dbuf, 3:6), emb, 1.3, 6, 8)
    @test view(buf, 3:6) == F && view(dbuf, 3:6) == dF            # same arithmetic: exact
    @test_throws "B, dB and weights must share axes" M.bessel_basis!(OffsetArray(zeros(4), 0:3), zeros(4), 1.0, emb.bessel_weights, 0.5)
end
