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
    @test maximum(abs, S - I) < 1.0e-12
end

@testitem "gradient matches finite differences and ForwardDiff" begin
    using ForwardDiff
    v = [1.0, 2.0, 3.0]; L = 4; n = PureMLIP.sh_length(L)
    Y = zeros(n); dY = zeros(n, 3); PureMLIP.spherical_harmonics_and_gradient!(Y, dY, v, L)
    J = ForwardDiff.jacobian(w -> PureMLIP.spherical_harmonics!(zeros(eltype(w), n), w, L), v)
    @test dY ≈ J atol = 1.0e-12
    h = 1.0e-6
    for i in 1:3
        e = zeros(3); e[i] = h
        Yp = PureMLIP.spherical_harmonics!(zeros(n), v + e, L)
        Ym = PureMLIP.spherical_harmonics!(zeros(n), v - e, L)
        @test dY[:, i] ≈ (Yp - Ym) / 2h atol = 1.0e-8
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
    @test Y ≈ Y64 atol = 1.0e-5
end
