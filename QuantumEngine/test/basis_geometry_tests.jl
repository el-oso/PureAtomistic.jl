# The shell geometry of `Basis`: the Cartesian component order, the primitive normalization, the
# contraction renormalization, the shell and basis-set types, the AO ranges and the shell pairs.

@testitem "component order matches Rust" begin
    using QuantumEngine.Basis: ncart, cartesian_component
    # The oracle's table (`shell.rs:84-117`), typed from the specification's list of 56 triples.
    RUST_TABLE = [
        (0, 0, 0),
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1),
        (3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 2, 0), (2, 1, 0), (2, 0, 1), (1, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1),
        (4, 0, 0), (0, 4, 0), (0, 0, 4), (3, 1, 0), (3, 0, 1), (1, 3, 0), (0, 3, 1), (1, 0, 3), (0, 1, 3),
        (2, 2, 0), (2, 0, 2), (0, 2, 2), (2, 1, 1), (1, 2, 1), (1, 1, 2),
        (5, 0, 0), (0, 5, 0), (0, 0, 5), (4, 1, 0), (4, 0, 1), (1, 4, 0), (0, 4, 1), (1, 0, 4), (0, 1, 4),
        (3, 2, 0), (3, 0, 2), (2, 3, 0), (0, 3, 2), (2, 0, 3), (0, 2, 3), (3, 1, 1), (1, 3, 1), (1, 1, 3),
        (2, 2, 1), (2, 1, 2), (1, 2, 2),
    ]
    @test length(RUST_TABLE) == 56
    @test [cartesian_component(l, i) for l in 0:5 for i in 1:ncart(l)] == RUST_TABLE
    for l in 0:5
        comps = [cartesian_component(l, i) for i in 1:ncart(l)]
        @test all(c -> sum(c) == l, comps)
        @test allunique(comps)
    end
    @test ncart.(0:6) == [1, 3, 6, 10, 15, 21, 28]
    @test cartesian_component(Int32(2), Int32(4)) == (1, 1, 0)
    @test_throws "basis: l must be ≥ 0, got -1" ncart(-1)
    @test_throws "basis: l = 6 is above the supported l ≤ 5" cartesian_component(6, 1)
    @test_throws ArgumentError cartesian_component(6, 1)
    @test_throws "basis: component 7 is outside 1:6 for l = 2" cartesian_component(2, 7)
    @test_throws "basis: component 0 is outside 1:1 for l = 0" cartesian_component(0, 0)
    @test_throws "basis: l must be ≥ 0, got -1" cartesian_component(-1, 1)
end

@testitem "basis: primitive norm" begin
    using QuantumEngine.Basis: ncart, cartesian_component, primitive_norm
    dfact(n) = n <= 0 ? big(1) : prod(big(k) for k in n:-2:1)
    for T in (Float64, Float32), α0 in (0.8, 3.425250914, 0.01285, 5.0e4)
        α = T(α0)
        a = big(α)
        u = eps(T) / 2
        for l in 0:5, i in 1:ncart(l)
            lx, ly, lz = cartesian_component(l, i)
            N = primitive_norm(α, lx, ly, lz)
            @test N isa T
            # Oracle: the self-overlap of the normalized primitive, evaluated in BigFloat from the
            # returned N: N² (π/(2α))^{3/2} Π_k (2l_k−1)!!/(4α)^{l_k}, which is 1 for the exact N.
            S = big(N)^2 * (big(π) / (2a))^(3 // 2) *
                prod(dfact(2lk - 1) / (4a)^lk for lk in (lx, ly, lz))
            # Tolerance (b), u = eps(T)/2: T(π) and the division give x within 2u;
            # sqrt(x) * sqrt(sqrt(x)) gives (3/4)·2u + 3.5u = 5u; (4α)^l by repeated multiplication
            # (l − 1)u (exact for l ≤ 1); the division by the exact integer product of double
            # factorials u; the outer sqrt half of that plus u; the final product u. At most
            # (7 + l/2)u in N, doubled by N²: (14 + l)u. Measured with this form: at most 0.32 of
            # the bound.
            @test abs(S - 1) <= (14 + l) * u
        end
        # xx and xy share the prefactor and (4α)^2, and differ by the division by 3, two sqrts and
        # the product with sqrt(3): at most 5u, within 4eps(T) (rule b).
        @test isapprox(primitive_norm(α, 2, 0, 0) * sqrt(T(3)), primitive_norm(α, 1, 1, 0); rtol = 4eps(T))
    end
    @test primitive_norm(0.8f0, 1, 1, 0) isa Float32
    @test primitive_norm(0.8, 1, 1, 0) isa Float64
    @test_throws "basis: Cartesian exponents must be ≥ 0, got -1, 0, 0" primitive_norm(0.8, -1, 0, 0)
    for α in (-1.0, 0.0, Inf, NaN)
        @test_throws lazy"basis: the exponent is $(α); exponents must be finite and positive" primitive_norm(α, 0, 0, 0)
        @test_throws ArgumentError primitive_norm(α, 0, 0, 0)
    end
    # 33!! = 6332659870762850625 fits in Int; 35!! does not, nor does 33!!·3!!, whose factors do.
    N17 = primitive_norm(0.8, 17, 0, 0)
    @test N17 isa Float64
    # The self-overlap oracle and (14 + l)u bound of the loop above, at l = 17.
    a = big(0.8)
    @test abs(big(N17)^2 * (big(π) / (2a))^(3 // 2) * dfact(33) / (4a)^17 - 1) <= 31 * eps(Float64) / 2
    @test_throws "basis: (2lx−1)!!(2ly−1)!!(2lz−1)!! overflows Int for the Cartesian exponents 18, 0, 0" primitive_norm(0.8, 18, 0, 0)
    @test_throws "basis: (2lx−1)!!(2ly−1)!!(2lz−1)!! overflows Int for the Cartesian exponents 17, 2, 0" primitive_norm(0.8, 17, 2, 0)
end

@testitem "basis: contraction renormalization" begin
    using QuantumEngine.Basis: contraction_renorm
    # One primitive: sqrt(0.8 * 0.8) == 0.8 in Float64, so b = 1, the sum is 0.25 and the factor 2.
    @test contraction_renorm([0.8], [0.5], 2) == 2.0
    # STO-3G H (`sto-3g.gbs:15-17`), at its own l = 0 and, to pin the power b^{l+3/2}, with the same
    # data taken as l = 1 and l = 2.
    e = [3.425250914, 0.6239137298, 0.168855404]
    c = [0.1543289673, 0.5353281423, 0.4446345422]
    for l in 0:2
        f = contraction_renorm(e, c, l)
        @test f isa Float64
        # Oracle: the same sum written out in BigFloat from the same Float64 data.
        terms = [
            begin
                b = 2 * sqrt(big(ai) * big(aj)) / (big(ai) + big(aj))
                big(ci) * big(cj) * b^(l + 1) * sqrt(b)
            end for (ai, ci) in zip(e, c) for (aj, cj) in zip(e, c)
        ]
        D_big = sum(terms)
        κ = sum(abs, terms) / abs(D_big)
        f_big = 1 / sqrt(D_big)
        # Tolerance (b), u = eps/2: the sum has relative error at most γ_{k_D}·κ with
        # k_D = n² + 5l + 9 terms of rounding (n = 3); the factor takes half of that through the
        # sqrt, plus u for the sqrt and u for the division.
        u = eps(Float64) / 2
        n = length(e)
        k_D = n^2 + 5l + 9
        γ = k_D * u / (1 - k_D * u)
        @test abs(big(f) / f_big - 1) <= γ * κ / 2 + 2u
    end
    @test contraction_renorm(Float32[0.8], Float32[0.5], 2) isa Float32
    @test_throws "basis: exponents and coefficients have different axes" contraction_renorm([1.0, 2.0], [1.0], 0)
    @test_throws DimensionMismatch contraction_renorm([1.0, 2.0], [1.0], 0)
    @test_throws "basis: a contraction needs at least one primitive" contraction_renorm(Float64[], Float64[], 0)
    @test_throws "basis: the contraction has zero or negative norm" contraction_renorm([0.8], [0.0], 0)
    @test_throws "basis: the contraction has zero or negative norm" contraction_renorm([0.8], [NaN], 0)
    @test_throws "basis: l must be ≥ 0, got -1" contraction_renorm([0.8], [0.5], -1)
    # An exponent whose square is below floatmin: the base `b` would underflow to 0, even on the
    # diagonal, where it is exactly 1.
    @test_throws "basis: exponent 1 of a contraction is 1.0e-200, whose square is below floatmin(Float64)" contraction_renorm(
        [1.0e-200, 1.0], [1.0, 1.0], 0
    )
    @test_throws "basis: exponent 1 of a contraction is 1.0000000031710769e-30, whose square is below floatmin(Float32)" contraction_renorm(
        Float32[1.0f-30, 1], Float32[1, 1], 0
    )
    @test_throws ArgumentError contraction_renorm([1.0e-200, 1.0], [1.0, 1.0], 0)
    for α in (-1.0, 0.0, Inf, NaN)
        @test_throws lazy"basis: exponent 2 of a contraction is $(α); exponents must be finite and positive" contraction_renorm(
            [1.0, α], [1.0, 1.0], 0
        )
        @test_throws ArgumentError contraction_renorm([1.0, α], [1.0, 1.0], 0)
    end
end

@testitem "basis: generic indexing" begin
    using QuantumEngine.Basis: contraction_renorm, Shell, BasisSet
    using OffsetArrays
    using StaticArrays: SVector
    e = [3.425250914, 0.6239137298, 0.168855404]
    c = [0.1543289673, 0.5353281423, 0.4446345422]
    f = contraction_renorm(e, c, 1)
    @test contraction_renorm(view(e, 1:3), view(c, 1:3), 1) == f
    @test contraction_renorm(OffsetArray(e, 0:2), OffsetArray(c, 0:2), 1) == f
    @test_throws DimensionMismatch contraction_renorm(OffsetArray(e, 0:2), c, 1)
    # Shell and BasisSet store a Vector as given and copy any other vector in iteration order.
    r = SVector(0.0, 0.0, 1.0)
    s = Shell(r, 1, 1, e, c, 0)
    for (ev, cv, rv) in (
            (view(e, 1:3), view(c, 1:3), view([0.0, 0.0, 1.0], 1:3)),
            (OffsetArray(e, 0:2), OffsetArray(c, 0:2), OffsetArray([0.0, 0.0, 1.0], 0:2)),
        )
        so = Shell(rv, 1, 1, ev, cv, 0)
        @test so isa Shell{Float64}
        @test (so.center, so.exponents, so.coefficients) == (s.center, s.exponents, s.coefficients)
        @test so.exponents isa Vector{Float64}
    end
    @test_throws "basis: exponent 2 of a shell is -0.5; exponents must be finite and positive" Shell(
        r, 1, 0, OffsetArray([1.0, -0.5], 0:1), OffsetArray([0.6, 0.4], 0:1), 0
    )
    @test_throws "basis: a shell center needs 3 coordinates, got 2" Shell([0.0, 1.0], 1, 0, e, c, 0)
    @test_throws ArgumentError Shell([0.0, 1.0], 1, 0, e, c, 0)
    shells = [s, Shell(r, 1, 0, e, c, 3)]
    bs = BasisSet(shells, 4, "t")
    for sv in (view(shells, 1:2), OffsetArray(shells, 0:1))
        bo = BasisSet(sv, 4, "t")
        @test bo.shells == bs.shells
        @test bo.shells isa Vector{Shell{Float64}}
    end
    @test_throws "basis: shell 2 starts at AO offset 3, expected 6" BasisSet(
        OffsetArray([Shell(r, 1, 2, e, c, 0), Shell(r, 1, 0, e, c, 3)], 0:1), 7, "t"
    )
end

@testitem "basis: shells and pairs" begin
    using QuantumEngine.Basis: Shell, BasisSet, ao_range, shell_pairs, ShellPairs
    using StaticArrays: SVector
    r = SVector(0.0, 0.0, 0.0)
    e = [1.0, 0.5]
    c = [0.6, 0.4]
    shells = [Shell(r, 1, 0, e, c, 0), Shell(r, 1, 1, e, c, 1), Shell(r, 2, 0, e, c, 4), Shell(r, 2, 2, e, c, 5)]
    bs = BasisSet(shells, 11, "test")
    @test bs isa BasisSet{Float64}
    @test ao_range.(bs.shells) == [1:1, 2:4, 5:5, 6:11]
    @test bs.nao == 11
    @test bs.shells[2].exponents === e
    @test collect(shell_pairs(bs)) == [(i, j) for i in 1:4 for j in 1:i]
    @test length(shell_pairs(bs)) == 10
    @test eltype(shell_pairs(bs)) == Tuple{Int, Int}
    @test isempty(collect(shell_pairs(BasisSet(Shell{Float64}[], 0, "empty"))))
    @test length(ShellPairs(0)) == 0
    # Float32 data give a Float32 shell, and a Vector center is stored as an SVector.
    s32 = Shell(Float32[0, 0, 1], 1, 0, Float32[1], Float32[1], 0)
    @test s32 isa Shell{Float32}
    @test s32.center == SVector(0.0f0, 0.0f0, 1.0f0)
    @test_throws "basis: l must be ≥ 0, got -1" Shell(r, 1, -1, e, c, 0)
    @test_throws "basis: a shell has 2 exponents and 1 coefficients" Shell(r, 1, 0, e, [1.0], 0)
    @test_throws "basis: a contraction needs at least one primitive" Shell(r, 1, 0, Float64[], Float64[], 0)
    @test_throws "basis: exponent 2 of a shell is -0.5; exponents must be finite and positive" Shell(r, 1, 0, [1.0, -0.5], c, 0)
    @test_throws "basis: exponent 1 of a shell is 0.0; exponents must be finite and positive" Shell(r, 1, 0, [0.0, 1.0], c, 0)
    @test_throws "basis: exponent 1 of a shell is Inf; exponents must be finite and positive" Shell(r, 1, 0, [Inf, 1.0], c, 0)
    @test_throws "basis: exponent 1 of a shell is NaN; exponents must be finite and positive" Shell(r, 1, 0, [NaN, 1.0], c, 0)
    @test_throws "basis: ao_offset must be ≥ 0, got -1" Shell(r, 1, 0, e, c, -1)
    @test_throws ArgumentError Shell(r, 1, 0, e, c, -1)
    gap = [shells[1], shells[2], Shell(r, 2, 0, e, c, 5)]
    @test_throws "basis: shell 3 starts at AO offset 5, expected 4" BasisSet(gap, 6, "gap")
    @test_throws "basis: nao is 12, but the shells hold 11 AOs" BasisSet(shells, 12, "test")
    @test_throws ArgumentError BasisSet(shells, 12, "test")
end

@testitem "basis: data types" begin
    using QuantumEngine.Basis: Primitive, ShellSpec, ECPChannel, ECPPotential, ElementBasis
    p = Primitive(0.8, 0.5)
    @test p isa Primitive{Float64}
    @test (p.exponent, p.coefficient) == (0.8, 0.5)
    @test Primitive{Float32}(1, 0) == Primitive(1.0f0, 0.0f0)
    sp = ShellSpec(2, [p], 17)
    @test sp isa ShellSpec{Float64}
    @test (sp.l, sp.primitives, sp.line) == (2, [p], 17)
    ch = ECPChannel(2, [1, 2], [175.55, 35.05], [-10.0, -47.5])
    @test ch isa ECPChannel{Float64}
    @test (ch.l, ch.powers, ch.exponents, ch.coefficients) == (2, [1, 2], [175.55, 35.05], [-10.0, -47.5])
    ecp = ECPPotential("Na", 10, 2, [ch])
    @test ecp isa ECPPotential{Float64}
    @test (ecp.symbol, ecp.ncore, ecp.lmax, ecp.channels) == ("Na", 10, 2, [ch])
    eb = ElementBasis("Na", [sp], ecp)
    @test eb isa ElementBasis{Float64}
    @test eb.ecp === ecp
    @test isnothing(ElementBasis("H", [sp], nothing).ecp)
    @test ElementBasis{Float32}("H", ShellSpec{Float32}[], nothing) isa ElementBasis{Float32}
end

@testitem "basis: allocation-free helpers" begin
    using QuantumEngine.Basis: ncart, cartesian_component, primitive_norm, contraction_renorm, Shell, BasisSet,
        shell_pairs
    using StaticArrays: SVector
    using StrictModeTest
    r = SVector(0.0, 0.0, 0.0)
    bs = BasisSet([Shell(r, 1, 0, [1.0], [1.0], 0), Shell(r, 1, 1, [1.0], [1.0], 1), Shell(r, 1, 0, [1.0], [1.0], 4)], 5, "t")
    function pair_sum(bs)
        s = 0
        for (i, j) in shell_pairs(bs)
            s += i + j
        end
        return s
    end
    e64, c64 = [3.425250914, 0.6239137298], [0.1543289673, 0.5353281423]
    e32, c32 = Float32.(e64), Float32.(c64)
    # @test_noalloc throws a StrictViolation unless AllocCheck proves the call allocation-free.
    @test (@test_noalloc ncart(3)) == 10
    @test (@test_noalloc cartesian_component(3, 5)) == (2, 1, 0)
    @test (@test_noalloc primitive_norm(0.8, 1, 1, 0)) == primitive_norm(0.8, 1, 1, 0)
    @test (@test_noalloc primitive_norm(0.8f0, 1, 1, 0)) == primitive_norm(0.8f0, 1, 1, 0)
    @test (@test_noalloc contraction_renorm(e64, c64, 1)) == contraction_renorm(e64, c64, 1)
    @test (@test_noalloc contraction_renorm(e32, c32, 1)) == contraction_renorm(e32, c32, 1)
    @test (@test_noalloc pair_sum(bs)) == sum(i + j for i in 1:3 for j in 1:i)
end

@testitem "basis: helper signatures" begin
    using QuantumEngine.Basis: ncart, cartesian_component, primitive_norm, contraction_renorm, ao_range, Shell
    using StrictMode, StrictModeTest
    StrictMode.assert_enabled()
    sigs = [
        (ncart, (Int,)),
        (cartesian_component, (Int, Int)),
        (primitive_norm, (Float64, Int, Int, Int)),
        (primitive_norm, (Float32, Int, Int, Int)),
        (contraction_renorm, (Vector{Float64}, Vector{Float64}, Int)),
        (contraction_renorm, (Vector{Float32}, Vector{Float32}, Int)),
        (ao_range, (Shell{Float64},)),
    ]
    @test (test_signatures(sigs; guarantees = (:typestable, :noalloc)); true)
end

# The StrictMode sweep judges no type constructor, so this item proves the constructors of the
# seven basis types type-stable and allocation-free in Float64 and Float32, inner and outer, except
# `ElementBasis` given an `ECPPotential`, which the item shows allocates: that is the allocation the
# exemption of `ElementBasis` names.
@testitem "basis: constructor signatures" begin
    using QuantumEngine.Basis: Primitive, ShellSpec, ECPChannel, ECPPotential, ElementBasis, Shell, BasisSet
    using StaticArrays: SVector
    using StrictMode, StrictModeTest
    StrictMode.assert_enabled()
    # A proof of an exempt function itself passes, so the ElementBasis construction is proved
    # through this caller, which is not exempt.
    build(::Type{T}, symbol, shells, ecp) where {T} = ElementBasis{T}(symbol, shells, ecp)
    # The bytes of one construction kept in a `Ref`, measured inside a compiled function after a
    # warm-up call. The result must be kept: a construction whose value is dropped is removed.
    function bytes_kept(r::Ref{ElementBasis{T}}, symbol, shells, ecp) where {T}
        r[] = ElementBasis{T}(symbol, shells, ecp)
        return @allocated r[] = ElementBasis{T}(symbol, shells, ecp)
    end
    for T in (Float64, Float32)
        V = Vector{T}
        sigs = [
            (Primitive{T}, (T, T)), (Primitive, (T, T)),
            (ShellSpec{T}, (Int, Vector{Primitive{T}}, Int)), (ShellSpec, (Int, Vector{Primitive{T}}, Int)),
            (ECPChannel{T}, (Int, Vector{Int}, V, V)), (ECPChannel, (Int, Vector{Int}, V, V)),
            (ECPPotential{T}, (String, Int, Int, Vector{ECPChannel{T}})),
            (ECPPotential, (String, Int, Int, Vector{ECPChannel{T}})),
            (ElementBasis{T}, (String, Vector{ShellSpec{T}}, Nothing)), (ElementBasis, (String, Vector{ShellSpec{T}}, Nothing)),
            (build, (Type{T}, String, Vector{ShellSpec{T}}, Nothing)),
            (Shell{T}, (SVector{3, T}, Int, Int, V, V, Int)), (Shell, (SVector{3, T}, Int, Int, V, V, Int)),
            (BasisSet{T}, (Vector{Shell{T}}, Int, String)), (BasisSet, (Vector{Shell{T}}, Int, String)),
        ]
        @test (test_signatures(sigs; guarantees = (:typestable, :noalloc)); true)
        withecp = (String, Vector{ShellSpec{T}}, ECPPotential{T})
        @test (test_signatures([(ElementBasis{T}, withecp), (ElementBasis, withecp)]; guarantees = (:typestable,)); true)
        @test_throws StrictMode.StrictViolation test_signatures([(build, (Type{T}, withecp...))]; guarantees = (:noalloc,))
        ecp = ECPPotential("Na", 10, 0, ECPChannel{T}[])
        shells = ShellSpec{T}[]
        r = Ref{ElementBasis{T}}()
        @test bytes_kept(r, "Na", shells, ecp) > 0
        @test bytes_kept(r, "Na", shells, nothing) == 0
    end
end
