@testmodule ArrayFixtures begin
    using PureXC
    using Random

    # Every shipped functional: the point functionals and the composite pbe.
    functionals(T) = (lda_x(T), lda_c_pw(T), lda_c_pw_mod(T), pbe_x(T), pbe_c(T), pbe(T))
    isgga(f) = PureXC.family(f) == :gga

    # n ≥ 4 points: restricted vectors, polarized matrices with σαβ inside its bound. Point 1 has
    # zero density, point 2 a nonzero density below every density threshold (1e-15 and 1e-12),
    # point 3 one between them (below that of pbe_c only); polarized, point 4 has an empty β
    # channel (ρβ = σαβ = σββ = 0).
    function inputs(T, n; polarized)
        rng = MersenneTwister(5)
        if polarized
            ρ = T.(0.01 .+ 2 .* rand(rng, 2, n))
            g = T.(0.05 .+ rand(rng, 3, n))
            g[2, :] .= T(0.5) .* sqrt.(g[1, :] .* g[3, :])
            ρ[:, 1] .= 0
            g[:, 1] .= 0
            ρ[:, 2] .= T(1.0e-16)
            ρ[:, 3] .= T(1.0e-13)
            ρ[2, 4] = 0
            g[2:3, 4] .= 0
            return ρ, g
        end
        ρ = T.(0.01 .+ 2 .* rand(rng, n))
        σ = T.(0.05 .+ rand(rng, n))
        ρ[1] = 0
        σ[1] = 0
        ρ[2] = T(1.0e-16)
        ρ[3] = T(1.0e-13)
        return ρ, σ
    end

    pt(ρ::AbstractVector, i) = ρ[i]
    pt(ρ::AbstractMatrix, i) = Tuple(ρ[:, i])

    # The expected field values of point i, as a Tuple or a number.
    function reference(f, k, ρ, σ, i)
        isgga(f) ? xc_derivatives(f, k, pt(ρ, i), pt(σ, i)) : xc_derivatives(f, k, pt(ρ, i))
    end

    # Compare a filled output with the point calls, exactly.
    function agrees(out, f, k, ρ, σ, pts)
        ok = true
        for i in pts
            r = reference(f, k, ρ, σ, i)
            for n in keys(r)
                a = out[n]
                got = r[n] isa Real ? a[1, i] : Tuple(a[:, i])
                ok &= got == r[n]
            end
        end
        return ok
    end

    call(f, k, ρ, σ) = isgga(f) ? (f, k, ρ, σ) : (f, k, ρ)
end

@testitem "array layer equals the point calls" setup = [ArrayFixtures] begin
    using .ArrayFixtures
    for T in (Float64, Float32), polarized in (false, true), f in ArrayFixtures.functionals(T), K in 0:3
        ρ, σ = ArrayFixtures.inputs(T, 40; polarized)
        args = ArrayFixtures.call(f, Val(K), ρ, σ)
        out = xc_outputs(args...)
        @test xc_evaluate!(out, args...) === out
        @test ArrayFixtures.agrees(out, f, Val(K), ρ, σ, axes(ρ, ndims(ρ)))
    end
end

@testitem "array layer on OffsetArray and view inputs" setup = [ArrayFixtures] begin
    using .ArrayFixtures
    using OffsetArrays
    for polarized in (false, true), f in (lda_c_pw(), pbe_x(), pbe_c())
        ρ, σ = ArrayFixtures.inputs(Float64, 30; polarized)
        args = ArrayFixtures.call(f, Val(2), ρ, σ)
        base = xc_evaluate!(xc_outputs(args...), args...)
        # Offset points; the spin dimension is offset as well.
        ro = polarized ? OffsetArray(ρ, -1, 6) : OffsetArray(ρ, 6)
        so = polarized ? OffsetArray(σ, 4, 6) : OffsetArray(σ, 6)
        argso = ArrayFixtures.call(f, Val(2), ro, so)
        out = xc_outputs(argso...)
        xc_evaluate!(out, argso...)
        for n in keys(base)
            @test axes(out[n], 2) == axes(ro, ndims(ro))
            @test parent(out[n]) == base[n]
        end
        # Views of a larger buffer, with a stride.
        big = polarized ? hcat(ρ, ρ) : vcat(ρ, ρ)
        gig = polarized ? hcat(σ, σ) : vcat(σ, σ)
        rv = polarized ? view(big, :, 1:2:60) : view(big, 1:2:60)
        sv = polarized ? view(gig, :, 1:2:60) : view(gig, 1:2:60)
        argsv = ArrayFixtures.call(f, Val(2), rv, sv)
        outv = xc_evaluate!(xc_outputs(argsv...), argsv...)
        argsc = ArrayFixtures.call(f, Val(2), collect(rv), collect(sv))
        copy_out = xc_evaluate!(xc_outputs(argsc...), argsc...)
        for n in keys(base)
            @test outv[n] == copy_out[n]
        end
    end
end

@testitem "array layer validates before writing" setup = [ArrayFixtures] begin
    using .ArrayFixtures
    n = 1000
    for f in (lda_x(), pbe_c()), polarized in (false, true)
        ρ, σ = ArrayFixtures.inputs(Float64, n; polarized)
        polarized ? (ρ[2, 500] = -1.0) : (ρ[500] = -1.0)
        args = ArrayFixtures.call(f, Val(1), ρ, σ)
        out = map(a -> fill!(similar(a), 42.0), xc_outputs(args...))
        @test_throws "xc: ρ" xc_evaluate!(out, args...)
        @test all(a -> all(==(42.0), a), out)
    end
end

@testitem "array layer rejects mismatched axes" setup = [ArrayFixtures] begin
    using .ArrayFixtures
    f = pbe_x()
    ρ, σ = ArrayFixtures.inputs(Float64, 10; polarized = false)
    out = xc_outputs(f, Val(1), ρ, σ)
    @test_throws DimensionMismatch xc_evaluate!(out, f, Val(1), ρ, σ[1:9])
    @test_throws DimensionMismatch xc_evaluate!(map(a -> a[:, 1:9], out), f, Val(1), ρ, σ)
    @test_throws "must both be vectors" xc_evaluate!(out, f, Val(1), ρ, reshape(repeat(σ, 3), 3, :))
    ρp, σp = ArrayFixtures.inputs(Float64, 10; polarized = true)
    outp = xc_outputs(f, Val(1), ρp, σp)
    @test_throws DimensionMismatch xc_evaluate!(outp, f, Val(1), ρp, σp[:, 1:9])
    @test_throws "two rows" xc_evaluate!(outp, f, Val(1), ρp[1:1, :], σp)
    @test_throws "three rows" xc_evaluate!(outp, f, Val(1), ρp, σp[1:2, :])
    @test_throws "rows, the field has" xc_evaluate!(merge(outp, (; vrho = outp.vrho[1:1, :])), f, Val(1), ρp, σp)
    @test_throws "which differ" xc_evaluate!((; e = out.e), f, Val(1), ρ, σ)
end

@testitem "pbe equals the sum of its terms" setup = [ArrayFixtures] begin
    using .ArrayFixtures
    for T in (Float64, Float32), polarized in (false, true), K in 0:3
        f = pbe(T)
        x, c = pbe_x(T), pbe_c(T)
        ρ, σ = ArrayFixtures.inputs(T, 20; polarized)
        out = xc_evaluate!(xc_outputs(f, Val(K), ρ, σ), f, Val(K), ρ, σ)
        ox = xc_evaluate!(xc_outputs(x, Val(K), ρ, σ), x, Val(K), ρ, σ)
        oc = xc_evaluate!(xc_outputs(c, Val(K), ρ, σ), c, Val(K), ρ, σ)
        for n in keys(out)
            @test out[n] == ox[n] + oc[n]
        end
        i = 7
        pρ = polarized ? Tuple(ρ[:, i]) : ρ[i]
        pσ = polarized ? Tuple(σ[:, i]) : σ[i]
        @test energy_density(f, pρ, pσ) == energy_density(x, pρ, pσ) + energy_density(c, pρ, pσ)
    end
    # Weights scale the terms.
    f = XC(((2, pbe_x()), (0.5, pbe_c())))
    @test xc_derivatives(f, Val(1), 0.3, 0.02).vrho == 2 * xc_derivatives(pbe_x(), Val(1), 0.3, 0.02).vrho + 0.5 * xc_derivatives(pbe_c(), Val(1), 0.3, 0.02).vrho
    @test PureXC.family(f) == :gga && PureXC.kind(f) == :exchange_correlation
    @test_throws "share one family" XC(((1, lda_x()), (1, pbe_c())))
    @test_throws "at least one term" XC(())
    @test_throws "xc: ρ" xc_derivatives(pbe(), Val(1), -1.0, 0.1)
end

@testitem "zk divides by the density" begin
    @test zk(-0.5, 2.0) == -0.25
    @test zk(-0.5, 0.0) == 0.0
    @test zk(-0.5, (1.0, 1.0)) == -0.25
    @test zk(1.0f0, 0.0f0) isa Float32
end

@testitem "array layer allocates nothing" setup = [ArrayFixtures] begin
    using .ArrayFixtures
    using StrictModeTest
    for polarized in (false, true), f in (lda_x(), lda_c_pw(), pbe_x(), pbe_c(), pbe())
        ρ, σ = ArrayFixtures.inputs(Float64, 50; polarized)
        k = Val(1)
        args = ArrayFixtures.call(f, k, ρ, σ)
        out = xc_outputs(args...)
        xc_evaluate!(out, args...)
        # @test_noalloc throws a StrictViolation unless AllocCheck proves the call allocation-free.
        if ArrayFixtures.isgga(f)
            @test (@test_noalloc xc_evaluate!(out, f, k, ρ, σ)) === out
        else
            @test (@test_noalloc xc_evaluate!(out, f, k, ρ)) === out
        end
    end
end

@testitem "array loop vectorizes" begin
    using InteractiveUtils
    f = pbe_x()
    ρ, σ = rand(64) .+ 0.1, rand(64)
    out = xc_outputs(f, Val(1), ρ, σ)
    ir = sprint(io -> code_llvm(io, xc_evaluate!, typeof.((out, f, Val(1), ρ, σ)); optimize = true, debuginfo = :none))
    # `vector.body` is the block the loop vectorizer emits; `<2 x double>` alone also appears when
    # only the SLP vectorizer has combined scalar operations.
    @test occursin("vector.body", ir)
    @test occursin("<2 x double>", ir)
end
