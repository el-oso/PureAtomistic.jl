@testitem "multi-index order" begin
    using PureTaylor: multi_indices
    @test multi_indices(Val(2), Val(2)) == ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    @test multi_indices(Val(3), Val(2))[5:10] == ((2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2))
    @test length(multi_indices(Val(5), Val(3))) == 56
end

@testitem "seeds and read-out" begin
    a = variable(Val(2), Val(2), Val(2), 3.0, 0.5)     # 3 + 0.5 u₂
    @test a isa Taylor{2,2,2,Float64,3}
    @test value(a) == 3.0 && PureTaylor.coefficient(a, (0, 1)) == 0.5 && PureTaylor.coefficient(a, (1, 0)) == 0.0
    # ∂/∂x₂ of x₂ is 1 after unscaling by 1/s
    @test derivatives(a, (1.0, 2.0)) == (3.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    @test constant(Val(2), Val(2), 4.0f0) isa Taylor{2,2,0,Float32,1}
    @test value(2.5) == 2.5
end

@testitem "order-0 seed has only the value" begin
    a = variable(Val(2), Val(0), Val(1), 3.0)
    @test a isa Taylor{2,0,1,Float64,1}
    @test a.c == (3.0,)
end

@testitem "addition over different masks" begin
    x = variable(Val(2), Val(2), Val(1), 1.0); y = variable(Val(2), Val(2), Val(2), 2.0)
    z = x - y + 0.5
    @test z isa Taylor{2,2,3,Float64,6}
    @test z.c == (-0.5, 1.0, -1.0, 0.0, 0.0, 0.0)            # exact: integer-valued arithmetic
    @test (x + constant(Val(2), Val(2), 1.0)) isa Taylor{2,2,1,Float64,3}
end

@testitem "scalar arithmetic, unary minus, zero and one" begin
    x = variable(Val(2), Val(2), Val(1), 1.0, 2.0)
    @test (x - 3).c == (-2.0, 2.0, 0.0)
    @test (3 - x).c == (2.0, -2.0, -0.0)
    @test (3 + x).c == (4.0, 2.0, 0.0)
    @test (-x).c == (-1.0, -2.0, -0.0)
    @test zero(x).c == (0.0, 0.0, 0.0) && zero(typeof(x)) isa typeof(x)
    @test one(x).c == (1.0, 0.0, 0.0) && one(typeof(x)) isa typeof(x)
    @test isfinite(x) && !isnan(x)
    @test !isfinite(x + Inf) && isnan(x + NaN)
    @test isnan(variable(Val(1), Val(2), Val(1), 1.0, NaN)) && !isfinite(variable(Val(1), Val(2), Val(1), 1.0, Inf))
end

@testitem "Float32 stays Float32; mixed T is an error" begin
    x = variable(Val(1), Val(2), Val(1), 1.0f0)
    @test (x + 0.1) isa Taylor{1,2,1,Float32,3}
    @test_throws MethodError x + variable(Val(1), Val(2), Val(1), 1.0)
end

@testitem "invalid parameters" begin
    @test_throws "PureTaylor: " Taylor{2,2,4,Float64,1}((1.0,))
    @test_throws "PureTaylor: " Taylor{2,2,1,Float64,2}((1.0, 2.0))
    @test Taylor{2,2,1,Real,3}((1, 2.0, 3)) isa Taylor
end
