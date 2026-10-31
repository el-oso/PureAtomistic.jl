@testitem "complex CG matches WignerSymbols" begin
    using WignerSymbols
    for l1 in 0:6, l2 in 0:6, l3 in abs(l1 - l2):min(l1 + l2, 6), m1 in -l1:l1, m2 in -l2:l2
        m3 = m1 + m2; abs(m3) <= l3 || continue
        @test PureMLIP.clebsch_gordan_complex(l1, m1, l2, m2, l3, m3) ≈ WignerSymbols.clebschgordan(Float64, l1, m1, l2, m2, l3, m3) atol = 1.0e-14
    end
    @test PureMLIP.clebsch_gordan_complex(1, 1, 1, 0, 2, 0) == 0.0
    @test PureMLIP.clebsch_gordan_complex(1, 2, 1, 0, 2, 2) == 0.0
    @test PureMLIP.clebsch_gordan_complex(1, 0, 1, 0, 3, 0) == 0.0
end

@testitem "real CG: norm, equivariance, errors" begin
    using LinearAlgebra
    R = [cos(0.7) -sin(0.7) 0; sin(0.7) cos(0.7) 0; 0 0 1] * [1 0 0; 0 cos(1.1) -sin(1.1); 0 sin(1.1) cos(1.1)]
    for (l1, l2, l3) in ((1, 1, 0), (1, 1, 2), (1, 2, 1), (1, 2, 3), (2, 2, 2), (3, 3, 4), (4, 4, 8))
        C = PureMLIP.real_clebsch_gordan(l1, l2, l3)
        @test C isa Array{Float64, 3}
        @test size(C) == (2l1 + 1, 2l2 + 1, 2l3 + 1)
        @test norm(C) ≈ 1 atol = 1.0e-14
        D1, D2, D3 = (PureMLIP.wigner_D(l, R) for l in (l1, l2, l3))
        C2 = [sum(C[a, b, c] * D1[a, i] * D2[b, j] * D3[c, k] for a in axes(C, 1), b in axes(C, 2), c in axes(C, 3)) for i in axes(C, 1), j in axes(C, 2), k in axes(C, 3)]
        @test C2 ≈ C atol = 1.0e-13
    end
    @test_throws "even l1+l2+l3" PureMLIP.real_clebsch_gordan(1, 1, 1)
    @test_throws "|l1-l2| ≤ l3" PureMLIP.real_clebsch_gordan(1, 1, 3)
end

@testitem "real CG at high l" begin
    using LinearAlgebra
    for (l1, l2, l3) in ((8, 8, 8), (8, 8, 16), (7, 8, 7))
        @test norm(PureMLIP.real_clebsch_gordan(l1, l2, l3)) ≈ 1 atol = 1.0e-13
    end
end

@testitem "wigner_D is a representation" begin
    using LinearAlgebra
    Ra = [cos(0.3) 0 sin(0.3); 0 1 0; -sin(0.3) 0 cos(0.3)]; Rb = [1 0 0; 0 cos(1.2) -sin(1.2); 0 sin(1.2) cos(1.2)]
    for l in 0:8
        Da, Db = PureMLIP.wigner_D(l, Ra), PureMLIP.wigner_D(l, Rb)
        @test PureMLIP.wigner_D(l, Ra * Rb) ≈ Da * Db atol = 1.0e-12
        @test Da' * Da ≈ I atol = 1.0e-12
        v = [0.3, -0.7, 0.4]; Y = zeros((l + 1)^2); Yr = zeros((l + 1)^2)
        PureMLIP.spherical_harmonics!(Y, v, l); PureMLIP.spherical_harmonics!(Yr, Ra * v, l)
        r = PureMLIP.sh_index(l, -l):PureMLIP.sh_index(l, l)
        @test Yr[r] ≈ Da * Y[r] atol = 1.0e-13
    end
    @test_throws "rotation" PureMLIP.wigner_D(1, [2.0 0 0; 0 1 0; 0 0 1])
    @test_throws "rotation" PureMLIP.wigner_D(1, [1.0 0 0; 0 1 0; 0 0 -1])
end
