@testsnippet SHQuadrature begin
    using LinearAlgebra

    # Product rule on the unit sphere: Gauss-Legendre in cos θ (`nθ` nodes, Golub-Welsch)
    # times uniform φ (`nφ` nodes). Returns (weight, unit vector) pairs whose weights sum to 1
    # times the sphere area 4π, so ∫ f dΩ ≈ Σ w f(v).
    function PureMLIP_test_quadrature(nθ::Int, nφ::Int)
        J = SymTridiagonal(zeros(nθ), [k / sqrt(4k^2 - 1) for k in 1:(nθ - 1)])
        E = eigen(J)
        t = E.values
        wt = 2 .* E.vectors[1, :] .^ 2
        G = Tuple{Float64, Vector{Float64}}[]
        for i in eachindex(t, wt), j in 0:(nφ - 1)
            φ = 2pi * j / nφ
            s = sqrt(1 - t[i]^2)
            push!(G, (wt[i] * 2pi / nφ, [s * cos(φ), s * sin(φ), t[i]]))
        end
        return G
    end
end

@testitem "closed forms for l ≤ 2" begin
    v = [0.3, -0.5, 0.8]; u = v / sqrt(sum(abs2, v)); x, y, z = u
    Y = zeros(9); PureMLIP.spherical_harmonics!(Y, v, 2)
    c0 = 1 / (2sqrt(pi)); c1 = sqrt(3 / (4pi)); c2 = sqrt(15 / (4pi)); c20 = sqrt(5 / (16pi))
    # Rounding: each entry is a product of at most 4 factors of size ≤ 1.6, so the error is at
    # most 4 · eps(Float64) · 1.6 = 1.4e-15; atol = 1e-14 is 7 times that. Measured: 0.
    @test Y ≈ [c0, c1 * y, c1 * z, c1 * x, c2 * x * y, c2 * y * z, c20 * (3z^2 - 1), c2 * x * z, c2 / 2 * (x^2 - y^2)] atol = 1.0e-14
end

@testitem "orthonormal up to lmax 8" setup = [SHQuadrature] begin
    using LinearAlgebra
    # Gauss-Legendre in cos θ (n = 20 nodes) × uniform φ (n = 40): exact for degree ≤ 16 polynomials
    G = PureMLIP_test_quadrature(20, 40)
    L = 8; n = PureMLIP.sh_length(L)
    S = zeros(n, n); Y = zeros(n)
    for (w, v) in G
        PureMLIP.spherical_harmonics!(Y, v, L); S .+= w .* (Y * Y')
    end
    # Measured margin: largest deviation 8.6e-15; the bound is 12 times that.
    @test maximum(abs, S - I) < 1.0e-13
end

@testitem "gradient matches finite differences and ForwardDiff" begin
    using ForwardDiff
    v = [1.0, 2.0, 3.0]; L = 4; n = PureMLIP.sh_length(L)
    Y = zeros(n); dY = zeros(n, 3); PureMLIP.spherical_harmonics_and_gradient!(Y, dY, v, L)
    J = ForwardDiff.jacobian(w -> PureMLIP.spherical_harmonics!(zeros(eltype(w), n), w, L), v)
    # Measured margin: ‖dY - J‖ ≤ 2.9e-16; atol is 34 times that.
    @test dY ≈ J atol = 1.0e-14
    h = 1.0e-6
    for i in 1:3
        e = zeros(3); e[i] = h
        Yp = PureMLIP.spherical_harmonics!(zeros(n), v + e, L)
        Ym = PureMLIP.spherical_harmonics!(zeros(n), v - e, L)
        # Rounding: the central difference loses about eps(Float64) · max|Y| / h = 2e-10; measured
        # ‖dY - fd‖ ≤ 4.3e-10, and atol is 7 times that.
        @test dY[:, i] ≈ (Yp - Ym) / 2h atol = 3.0e-9
    end
    @test Y == PureMLIP.spherical_harmonics!(zeros(n), v, L)
end

@testitem "zero vector and size errors" begin
    @test_throws "spherical harmonics of a zero vector are undefined" PureMLIP.spherical_harmonics!(zeros(4), zeros(3), 1)
    @test_throws DimensionMismatch PureMLIP.spherical_harmonics!(zeros(3), [1.0, 0, 0], 1)
    @test_throws "Y has length 3, expected sh_length(1) = 4" PureMLIP.spherical_harmonics!(zeros(3), [1.0, 0, 0], 1)
    @test_throws "v has length 2, expected 3" PureMLIP.spherical_harmonics!(zeros(4), [1.0, 0], 1)
    @test_throws "dY has size (4, 2), expected (4, 3)" PureMLIP.spherical_harmonics_and_gradient!(zeros(4), zeros(4, 2), [1.0, 0, 0], 1)
    @test_throws "spherical harmonics of a zero vector are undefined" PureMLIP.spherical_harmonics_and_gradient!(zeros(4), zeros(4, 3), zeros(3), 1)
end

@testitem "OffsetArray and view inputs" begin
    using OffsetArrays
    v = [0.2, 0.4, -0.9]; Y = zeros(16); PureMLIP.spherical_harmonics!(Y, v, 3)
    Yo = OffsetVector(zeros(16), -3); PureMLIP.spherical_harmonics!(Yo, OffsetVector(v, 5), 3)
    @test parent(Yo) == Y
    buf = zeros(20); PureMLIP.spherical_harmonics!(view(buf, 3:18), view([9.0, v...], 2:4), 3)
    @test buf[3:18] == Y
    dY = zeros(16, 3); PureMLIP.spherical_harmonics_and_gradient!(zeros(16), dY, v, 3)
    dYo = OffsetArray(zeros(16, 3), -3, 4); Yo2 = OffsetVector(zeros(16), 7)
    PureMLIP.spherical_harmonics_and_gradient!(Yo2, dYo, OffsetVector(v, 5), 3)
    @test parent(dYo) == dY
    @test parent(Yo2) == Y
end

@testitem "Float32 stays Float32" begin
    v = Float32[0.3, -0.5, 0.8]; Y = zeros(Float32, 25); dY = zeros(Float32, 25, 3)
    @test PureMLIP.spherical_harmonics_and_gradient!(Y, dY, v, 4) isa Tuple{Vector{Float32}, Matrix{Float32}}
    Y64 = PureMLIP.spherical_harmonics!(zeros(25), Float64.(v), 4)
    # Rounding: 16 · eps(Float32) = 1.9e-6. Measured ‖Y - Y64‖ = 1.5e-7, 12 times smaller.
    @test Y ≈ Y64 atol = 16 * eps(Float32)
end

@testitem "poles: finite values and gradients matching ForwardDiff" begin
    using ForwardDiff
    L = 6; n = PureMLIP.sh_length(L)
    for v in ([0.0, 0.0, 1.0], [1.0e-9, 0.0, 1.0], [0.0, 0.0, -2.0])
        Y = zeros(n); dY = zeros(n, 3)
        PureMLIP.spherical_harmonics_and_gradient!(Y, dY, v, L)
        @test all(isfinite, Y) && all(isfinite, dY)
        J = ForwardDiff.jacobian(w -> PureMLIP.spherical_harmonics!(zeros(eltype(w), n), w, L), v)
        # Rounding: 4 · eps(Float64) · max|J|. Measured ‖dY - J‖ ≤ 5.3e-24.
        @test dY ≈ J atol = 4 * eps() * maximum(abs, J)
    end
end

@testitem "recurrence overflow throws" begin
    v = Float32[0.3, -0.5, 0.8]
    @test_throws "lmax = 40 overflows Float32 in the spherical-harmonic recurrence" PureMLIP.spherical_harmonics!(zeros(Float32, 41^2), v, 40)
    @test_throws ArgumentError PureMLIP.e3nn_spherical_harmonics!(zeros(Float32, 41^2), v, 40)
end
