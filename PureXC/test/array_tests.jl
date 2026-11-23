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

# The terms are summed in their evaluation type and each output is rounded once. pbe(Float32) at
# order 4 is evaluated in Float64 (above `_float32_max_order`); at (1f-10, 8.247848f-32) the terms
# of v4rho3sigma round to +Inf32 (pbe_x) and -Inf32 (pbe_c), while their sum is -6.03e36. Oracle:
# the same functional converted exactly to BigFloat (256 bits). Bound (c): eps(Float32) relative,
# measured 0.033 eps(Float32). Every other output is finite, or the infinity of the sign of its
# exact value where that exceeds floatmax(Float32).
@testitem "pbe(Float32) sums its terms before rounding" begin
    ρ, σ = 1.0f-10, 8.247848f-32
    r = xc_derivatives(pbe(Float32), Val(4), ρ, σ)
    @test xc_derivatives(pbe_x(Float32), Val(4), ρ, σ).v4rho3sigma == Inf32
    @test xc_derivatives(pbe_c(Float32), Val(4), ρ, σ).v4rho3sigma == -Inf32
    b = setprecision(BigFloat, 256) do
        PureXC._xc_derivatives_unchecked(PureXC._with_number_type(pbe(Float32), BigFloat), Val(4), big(ρ), big(σ))
    end
    @test isfinite(r.v4rho3sigma)
    @test abs(big(r.v4rho3sigma) - b.v4rho3sigma) <= eps(Float32) * abs(b.v4rho3sigma)
    for n in keys(r)
        g, w = getfield(r, n), getfield(b, n)
        @test abs(w) <= floatmax(Float32) ? isfinite(g) : (isinf(g) && sign(g) == sign(w))
    end
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
    # The IR comes from a child process started with `--check-bounds=auto`, the setting of a normal
    # session: the suite runs under `--check-bounds=yes`, which keeps every bounds check and so
    # prevents the vectorization this item checks.
    script = """
    using PureXC, InteractiveUtils
    Base.JLOptions().check_bounds == 0 || error("the child process does not run with --check-bounds=auto")
    f = pbe_x()
    ρ, σ = rand(64) .+ 0.1, rand(64)
    ρp, σp = rand(2, 64) .+ 0.1, rand(3, 64)
    # The point loops of xc_evaluate!: the dense loop of `Array`s, restricted and polarized, and the
    # loop with checked loads that other arrays (here a view) take.
    for (loop, x) in (
            (PureXC._evaluate_points!, (ρ, σ)), (PureXC._evaluate_points!, (ρp, σp)),
            (PureXC._evaluate_points_checked!, (view(ρ, :), σ)),
        )
        out = xc_outputs(f, Val(1), x...)
        code_llvm(stdout, loop, typeof.((out, f, Val(1), x...)); optimize = true, debuginfo = :none)
        println("END OF LOOP")
    end
    """
    out = read(`$(Base.julia_cmd()) --check-bounds=auto --project=$(Base.active_project()) -e $script`, String)
    irs = split(out, "END OF LOOP")[1:3]
    # `vector.body` is the block the loop vectorizer emits; `<2 x double>` alone also appears when
    # only the SLP vectorizer has combined scalar operations.
    for ir in irs
        @test occursin("vector.body", ir)
        @test occursin("<2 x double>", ir)
    end
end

@testitem "the input predicate agrees with the input checks" setup = [ArrayFixtures] begin
    using .ArrayFixtures
    using Random
    # Whether `_check_inputs` accepts the inputs, from whether it throws.
    function accepted(f, k, x...)
        try
            PureXC._check_inputs(f, k, x...)
        catch err
            err isa ArgumentError || rethrow()
            return false
        end
        return true
    end
    # Edge values of every condition: signs (−0.0, −1e-300), finiteness, the density thresholds
    # (1e-15, 1e-12) and limits (2^45, 2^95, 2^381, 2^731, 2^767 and the powers of two above them),
    # σ floors (1e-40, 1e-32), large reduced gradients and Float32 subnormals.
    ρs = (
        0.0, -0.0, -1.0e-300, 1.0e-300, 1.0e-16, 2.0e-15, 1.0e-12, 1.5e-12, 0.3, 1.0e3, 2.0^45, 2.0^46,
        2.0^95, 2.0^96, 2.0^381, 2.0^382, 2.0^731, 2.0^732, 2.0^767, 2.0^768, floatmax(Float64), Inf, -Inf, NaN,
    )
    σs = (
        0.0, -0.0, -1.0e-300, 1.0e-45, 1.0e-40, 1.0e-32, 1.0e-20, 0.01, 1.0, 1.0e10, 1.0e40, 1.0e300,
        floatmax(Float64), Inf, -Inf, NaN,
    )
    rng = Xoshiro(20261123)
    for T in (Float64, Float32), f in ArrayFixtures.functionals(T), K in (0, 1, 2, 3, 7, 10)
        k = Val(K)
        conv(x) = convert(T, x)
        if ArrayFixtures.isgga(f)
            for ρ in ρs, σ in σs
                @test PureXC._inputs_ok(f, k, conv(ρ), conv(σ)) == accepted(f, k, conv(ρ), conv(σ))
            end
            for _ in 1:400
                ρ = conv.((rand(rng, ρs), rand(rng, ρs)))
                aa, bb = conv(rand(rng, σs)), conv(rand(rng, σs))
                # σαβ from the edge set, or at and just beyond ±√(σαα σββ).
                edge = sqrt(abs(aa)) * sqrt(abs(bb))
                ab = conv(rand(rng, (rand(rng, σs), -rand(rng, σs), edge, -edge, edge * (1 + 16 * eps(T)), -edge * (1 + 16 * eps(T)))))
                σ = (aa, ab, bb)
                @test PureXC._inputs_ok(f, k, ρ, σ) == accepted(f, k, ρ, σ)
            end
        else
            for ρ in ρs
                @test PureXC._inputs_ok(f, k, conv(ρ)) == accepted(f, k, conv(ρ))
            end
            for ρa in ρs, ρb in ρs
                ρ = (conv(ρa), conv(ρb))
                @test PureXC._inputs_ok(f, k, ρ) == accepted(f, k, ρ)
            end
        end
    end
end

@testitem "array layer reports a predicate that disagrees with the checks" begin
    using PureXC
    # A functional whose predicate rejects every point that its checks accept.
    struct _Disagreeing <: PureXC.AbstractLDA{Float64}
        thresholds::PureXC.Thresholds{Float64}
    end
    PureXC._energy_density(::_Disagreeing, ρ) = ρ
    PureXC._cutoff(::_Disagreeing, ρ) = false
    PureXC._inputs_ok(::_Disagreeing, ::Val, ρ::Real) = false
    f = _Disagreeing(PureXC.Thresholds{Float64}(1.0e-15, 0.0, eps(Float64)))
    ρ = [0.1, 0.2]
    @test_throws "xc: _inputs_ok rejects a point that _check_inputs accepts" xc_evaluate!(xc_outputs(f, Val(0), ρ), f, Val(0), ρ)
end

@testitem "array layer rejects outputs that share memory" setup = [ArrayFixtures] begin
    using .ArrayFixtures
    msg = "xc: an output shares memory with an input or another output"
    n = 64
    # Two Arrays over one buffer at different offsets: ρ is its first n numbers and vrho the n numbers
    # from n ÷ 2 on. Every number is 42.0, a valid density and the sentinel of the outputs.
    for f in (lda_x(), pbe_x())
        buf = fill(42.0, 2n)
        σ = fill(0.5, n)
        e = fill(42.0, 1, n)
        GC.@preserve buf begin
            ρ = unsafe_wrap(Array, pointer(buf), n)
            vrho = unsafe_wrap(Array, pointer(buf, n ÷ 2 + 1), (1, n))
            args = ArrayFixtures.call(f, Val(1), ρ, σ)
            out = merge(xc_outputs(args...), (; e, vrho))
            map(a -> fill!(a, 42.0), out)
            @test_throws msg xc_evaluate!(out, args...)
            @test all(a -> all(==(42.0), a), out)
            @test all(==(42.0), buf)
            # Two outputs over the buffer at different offsets, the input apart.
            ρs = fill(0.3, n)
            args = ArrayFixtures.call(f, Val(1), ρs, σ)
            e2 = unsafe_wrap(Array, pointer(buf, 3), (1, n))
            out = merge(xc_outputs(args...), (; e = e2, vrho))
            map(a -> fill!(a, 42.0), out)
            @test_throws msg xc_evaluate!(out, args...)
            @test all(a -> all(==(42.0), a), out)
        end
    end
    for f in (lda_x(), pbe_c(), pbe()), polarized in (false, true)
        ρ, σ = ArrayFixtures.inputs(Float64, 12; polarized)
        args = ArrayFixtures.call(f, Val(1), ρ, σ)
        out = map(a -> fill!(a, 42.0), xc_outputs(args...))
        # An output that is an input (polarized) or shares its memory through `reshape` (restricted).
        same = merge(out, (; vrho = polarized ? ρ : reshape(ρ, 1, :)))
        ρ0 = copy(ρ)
        @test_throws msg xc_evaluate!(same, args...)
        @test ρ == ρ0
        @test all(a -> all(==(42.0), a), out)
        # Two outputs that are one array, and two overlapping views of one matrix.
        twice = merge(out, (; e = polarized ? view(out.vrho, 1:1, :) : out.vrho))
        @test_throws msg xc_evaluate!(twice, args...)
        if polarized
            M = fill(42.0, 3, 12)
            views = merge(out, (; e = view(M, 2:2, :), vrho = view(M, 1:2, :)))
            @test_throws msg xc_evaluate!(views, args...)
            @test all(==(42.0), M)
        end
        @test all(a -> all(==(42.0), a), out)
    end
end

@testitem "array layer rejects outputs that share memory through wrappers" begin
    using OffsetArrays
    msg = "xc: an output shares memory with an input or another output"
    n = 64
    # Every number is 42.0, a valid input and the sentinel of the outputs: a rejected call leaves
    # every input and output all 42.0.
    function rejects(out, f, x...)
        map(a -> fill!(a, 42.0), out)
        @test_throws msg xc_evaluate!(out, f, Val(1), x...)
        @test all(a -> all(==(42.0), a), out)
        @test all(a -> all(==(42.0), a), x)
        return nothing
    end
    f = lda_x()
    buf = fill(42.0, 2n)
    GC.@preserve buf begin
        vrho = unsafe_wrap(Array, pointer(buf, n ÷ 2 + 1), (1, n))
        # A reshape of a view of the buffer as ρ.
        ρ = reshape(view(reshape(buf, n, 2), :, 1:1), n)
        rejects(merge(xc_outputs(f, Val(1), ρ), (; vrho)), f, ρ)
        # OffsetArrays of Arrays over the buffer as ρ and vrho.
        ρo = OffsetArray(unsafe_wrap(Array, pointer(buf), n), 0)
        rejects(merge(xc_outputs(f, Val(1), ρo), (; vrho = OffsetArray(vrho, 0, 0))), f, ρo)
        # A Memory over the buffer as ρ, and a reshape of one as vrho.
        mem = unsafe_wrap(Memory{Float64}, pointer(buf), n)
        rejects(merge(xc_outputs(f, Val(1), fill(1.0, n)), (; vrho)), f, mem)
        vm = reshape(unsafe_wrap(Memory{Float64}, pointer(buf, n ÷ 2 + 1), n), 1, n)
        rejects(merge(xc_outputs(f, Val(1), fill(1.0, n)), (; vrho = vm)), f, unsafe_wrap(Array, pointer(buf), n))
        @test all(==(42.0), buf)
    end
    # A reinterpretation of complex numbers as ρ.
    cbuf = fill(complex(42.0, 42.0), n)
    GC.@preserve cbuf begin
        ρc = reinterpret(Float64, view(cbuf, 1:(n ÷ 2)))
        vc = unsafe_wrap(Array, Ptr{Float64}(pointer(cbuf)) + 8 * (n ÷ 2), (1, n))
        rejects(merge(xc_outputs(f, Val(1), fill(1.0, n)), (; vrho = vc)), f, ρc)
        @test all(==(complex(42.0, 42.0)), cbuf)
    end
    # Rows of a PermutedDimsArray over a view with repeated indices: row 2 is A[1:n] (or A[2:n+1]),
    # as row 1 is, so index sets of P prove nothing.
    for shift in (0, 1)
        A = fill(42.0, n + 1)
        P = PermutedDimsArray(reshape(view(A, vcat(1:n, (1 + shift):(n + shift))), n, 2), (2, 1))
        rejects(merge(xc_outputs(f, Val(1), fill(1.0, n)), (; vrho = view(P, 2:2, :))), f, view(P, 1, :))
        rejects(merge(xc_outputs(f, Val(1), fill(1.0, n)), (; e = view(P, 1:1, :), vrho = view(P, 2:2, :))), f, fill(42.0, n))
        @test all(==(42.0), A)
    end
    # Rows of a reshape of a view with repeated indices.
    x = fill(42.0, 2)
    Pa = reshape(view(x, [1, 1, 2, 2]), 2, 2)
    rejects((; e = view(Pa, 1:1, :), vrho = view(Pa, 2:2, :)), f, fill(42.0, 2))
    @test all(==(42.0), x)
    # Polarized: an OffsetArray vsigma over σ at an offset, and a transposed vrho over ρ at an offset.
    fc = pbe_c()
    sbuf = fill(42.0, 6n)
    ρbuf = fill(42.0, 4n)
    GC.@preserve sbuf ρbuf begin
        σw = unsafe_wrap(Array, pointer(sbuf), (3, n))
        ρw = unsafe_wrap(Array, pointer(ρbuf), (2, n))
        vs = OffsetArray(unsafe_wrap(Array, pointer(sbuf, 3 * (n ÷ 2) + 1), (3, n)), 0, 0)
        rejects(merge(xc_outputs(fc, Val(1), ρw, σw), (; vsigma = vs)), fc, ρw, σw)
        vr = transpose(unsafe_wrap(Array, pointer(ρbuf, 3), (n, 2)))
        rejects(merge(xc_outputs(fc, Val(1), ρw, σw), (; vrho = vr)), fc, ρw, σw)
        @test all(==(42.0), sbuf) && all(==(42.0), ρbuf)
    end
    # Wrappers over disjoint parts of one buffer pass when their roots do not intersect.
    buf = fill(42.0, 2n)
    GC.@preserve buf begin
        ρo = OffsetArray(unsafe_wrap(Array, pointer(buf), n), 0)
        out = merge(xc_outputs(f, Val(1), ρo), (; vrho = transpose(unsafe_wrap(Array, pointer(buf, n + 1), n))))
        xc_evaluate!(out, f, Val(1), ρo)
        ref = xc_evaluate!(xc_outputs(f, Val(1), fill(42.0, n)), f, Val(1), fill(42.0, n))
        @test collect(out.vrho) == ref.vrho && collect(out.e) == ref.e
        @test all(==(42.0), view(buf, 1:n))
    end
end


@testitem "array layer accepts disjoint views of one array as outputs" setup = [ArrayFixtures] begin
    using .ArrayFixtures
    using OffsetArrays
    n = 20
    for f in (lda_x(), pbe_x(), pbe_c(), pbe()), polarized in (false, true)
        ρ, σ = ArrayFixtures.inputs(Float64, n; polarized)
        args = ArrayFixtures.call(f, Val(1), ρ, σ)
        ref = xc_evaluate!(xc_outputs(args...), args...)
        rows = map(a -> size(a, 1), values(ref))
        starts = cumsum((1, rows...))
        # Each field a different block of rows of one matrix.
        R = fill(NaN, sum(rows), n)
        outr = NamedTuple{keys(ref)}(ntuple(i -> view(R, starts[i]:(starts[i + 1] - 1), :), length(rows)))
        xc_evaluate!(outr, args...)
        @test all(m -> outr[m] == ref[m], keys(ref))
        # Each field a different block of columns of one matrix.
        C = fill(NaN, maximum(rows), length(rows) * n)
        outc = NamedTuple{keys(ref)}(ntuple(i -> view(C, 1:rows[i], ((i - 1) * n + 1):(i * n)), length(rows)))
        xc_evaluate!(outc, args...)
        @test all(m -> outc[m] == ref[m], keys(ref))
    end
    # A restricted input that is row 1 of a matrix and an output that is row 2.
    f = lda_x()
    ρ, _ = ArrayFixtures.inputs(Float64, n; polarized = false)
    B = vcat(ρ', fill(NaN, 1, n))
    ref = xc_evaluate!(xc_outputs(f, Val(0), ρ), f, Val(0), ρ)
    out = (; e = view(B, 2:2, :))
    xc_evaluate!(out, f, Val(0), view(B, 1, :))
    @test out.e == ref.e
    # An OffsetArray input with its OffsetArray outputs, and a view input of a buffer.
    ro = OffsetArray(ρ, 6)
    outo = xc_evaluate!(xc_outputs(f, Val(1), ro), f, Val(1), ro)
    @test parent(outo.vrho) == xc_evaluate!(xc_outputs(f, Val(1), ρ), f, Val(1), ρ).vrho
    big = vcat(ρ, ρ)
    rv = view(big, 1:2:(2n))
    @test xc_evaluate!(xc_outputs(f, Val(1), rv), f, Val(1), rv).vrho == xc_evaluate!(xc_outputs(f, Val(1), collect(rv)), f, Val(1), collect(rv)).vrho
end

@testitem "overlap test of views and Arrays" begin
    using PureXC: _overlaps
    using OffsetArrays
    n = 16
    buf = zeros(2n)
    A = rand(4, n)
    # Views of one parent: exact for integers and ranges, in every dimension.
    @test !_overlaps(view(A, 1:2, :), view(A, 3:4, :))
    @test _overlaps(view(A, 1:2, :), view(A, 2:3, :))
    @test !_overlaps(view(A, 1:2:4, :), view(A, 2:2:4, :))
    @test !_overlaps(view(A, 1, :), view(A, 2:4, :))
    @test !_overlaps(view(A, :, 1:4), view(A, :, n:-1:5))
    @test _overlaps(view(A, :, 1:4), view(A, :, n:-1:4))
    # Another index type counts as intersecting in its dimension; different numbers of indices count
    # as overlapping.
    @test _overlaps(view(A, [1, 2], :), view(A, [3, 4], :))
    @test !_overlaps(view(A, [1, 2], 1:2), view(A, [3, 4], 3:4))
    @test _overlaps(view(A, 1:2:8), view(A, 2:2:4, 1:2))
    # An empty array shares memory with nothing.
    @test !_overlaps(A, view(A, 1:0, :))
    GC.@preserve buf begin
        P = unsafe_wrap(Array, pointer(buf), n)
        Q = unsafe_wrap(Array, pointer(buf, n ÷ 2 + 1), n)
        @test _overlaps(P, Q) && _overlaps(Q, P)
        @test !_overlaps(P, unsafe_wrap(Array, pointer(buf, n + 1), n))
        # Views of different parents: disjoint address spans, else the parents.
        @test !_overlaps(view(P, 1:4), view(Q, (n - 3):n))
        @test !_overlaps(view(P, 1:2:7), Q)
        @test _overlaps(view(P, n:-1:1), Q)
        @test _overlaps(view(P, n:-1:(n - 1)), view(Q, (n ÷ 2 - 1):(n ÷ 2)))
        @test _overlaps(view(P, (n ÷ 2 + 1):2:n), view(Q, 2:2:n))
    end
    @test !_overlaps(view(A, 1:2, :), rand(2, n))
    @test _overlaps(view(A, 1:2, :), A)
    # Index sets of a parent that is not an Array prove nothing: rows 1 and 2 of Pa are both x.
    x = rand(2)
    Pa = reshape(view(x, [1, 1, 2, 2]), 2, 2)
    @test _overlaps(view(Pa, 1:1, :), view(Pa, 2:2, :))
    # Wrappers of disjoint parts of one Array count as overlapping.
    @test _overlaps(view(A, 1:1, :), reshape(view(A, 2, :), 1, n))
    @test _overlaps(OffsetArray(view(A, 1:1, :), 0, 0), view(A, 2:2, :))
    # Roots without a known address range are tested by their Base.dataids.
    @test !_overlaps(1:n, A)
    # Index sets out of bounds of the parent prove nothing: row 3 of a 2-row matrix is row 1 shifted
    # by one column.
    B = rand(2, 2n)
    @test _overlaps(view(B, 1, 1:n), Base.unsafe_view(B, 3:3, 1:n))
    @test !_overlaps(view(B, 1, 1:n), view(B, 2:2, 1:n))
    # A range of step zero selects its first value.
    z = StepRangeLen(1, 0, n)
    @test !_overlaps(view(B, 1:1, z), view(B, 2:2, 1:n))
    @test _overlaps(view(B, 1:1, z), view(B, 1:1, 1:1))
    @test !_overlaps(view(B, 1:1, z), view(B, 1:1, 2:n))
end

@testitem "array layer rejects overlap through array types without an Array root" begin
    using SparseArrays
    using PureXC: _overlaps
    msg = "xc: an output shares memory with an input or another output"
    n = 16
    f = lda_x()
    ref = xc_evaluate!(xc_outputs(f, Val(1), fill(42.0, n)), f, Val(1), fill(42.0, n))
    # Every number is 42.0, a valid input and the sentinel of the outputs.
    function rejects(out, x...)
        map(a -> fill!(a, 42.0), out)
        @test_throws msg xc_evaluate!(out, f, Val(1), x...)
        @test all(a -> all(==(42.0), a), out)
        @test all(a -> all(==(42.0), a), x)
        return nothing
    end
    # isbits arrays over raw pointers: one with `pointer` and `strides` (no `Base.elsize`), a matrix
    # type with all three, one with `Base.dataids` alone, and one with none of them.
    struct PtrVec <: AbstractVector{Float64}
        p::Ptr{Float64}
        n::Int
    end
    Base.size(a::PtrVec) = (a.n,)
    Base.getindex(a::PtrVec, i::Int) = unsafe_load(a.p, i)
    Base.setindex!(a::PtrVec, v, i::Int) = unsafe_store!(a.p, v, i)
    Base.pointer(a::PtrVec) = a.p
    Base.strides(::PtrVec) = (1,)
    struct PtrMat <: DenseMatrix{Float64}
        p::Ptr{Float64}
        sz::NTuple{2, Int}
    end
    Base.size(a::PtrMat) = a.sz
    Base.getindex(a::PtrMat, i::Int, j::Int) = unsafe_load(a.p, i + (j - 1) * a.sz[1])
    Base.setindex!(a::PtrMat, v, i::Int, j::Int) = unsafe_store!(a.p, v, i + (j - 1) * a.sz[1])
    Base.unsafe_convert(::Type{Ptr{Float64}}, a::PtrMat) = a.p
    Base.elsize(::Type{PtrMat}) = sizeof(Float64)
    struct IdVec <: AbstractVector{Float64}
        p::Ptr{Float64}
        n::Int
    end
    Base.size(a::IdVec) = (a.n,)
    Base.getindex(a::IdVec, i::Int) = unsafe_load(a.p, i)
    Base.setindex!(a::IdVec, v, i::Int) = unsafe_store!(a.p, v, i)
    Base.dataids(a::IdVec) = (UInt(a.p),)
    struct BareVec <: AbstractVector{Float64}
        p::Ptr{Float64}
        n::Int
    end
    Base.size(a::BareVec) = (a.n,)
    Base.getindex(a::BareVec, i::Int) = unsafe_load(a.p, i)
    Base.setindex!(a::BareVec, v, i::Int) = unsafe_store!(a.p, v, i)
    buf = fill(42.0, 3n)
    GC.@preserve buf begin
        vrho = unsafe_wrap(Array, pointer(buf, n ÷ 2 + 1), (1, n))
        out1 = merge(xc_outputs(f, Val(1), fill(1.0, n)), (; vrho))
        # Address ranges from `pointer` and `strides`, at an offset.
        rejects(out1, PtrVec(pointer(buf), n))
        rejects(merge(xc_outputs(f, Val(1), fill(1.0, n)), (; vrho = PtrMat(pointer(buf, 3), (1, n)))), unsafe_wrap(Array, pointer(buf), n))
        rejects(merge(xc_outputs(f, Val(1), fill(1.0, n)), (; e = PtrMat(pointer(buf), (1, n)), vrho = PtrMat(pointer(buf, 2), (1, n)))), fill(42.0, n))
        # Base.dataids of an isbits root, compared directly: IdVec at the start of the buffer names
        # the buffer's memory, as the Array does.
        rejects(merge(xc_outputs(f, Val(1), fill(1.0, n)), (; vrho = view(reshape(buf, 1, :), :, 1:n))), IdVec(pointer(buf), n))
        # Pointer types apart from the outputs are accepted.
        far = fill(42.0, n)
        GC.@preserve far begin
            out = xc_outputs(f, Val(1), far)
            xc_evaluate!(out, f, Val(1), PtrVec(pointer(far), n))
            @test out == ref
            out = merge(xc_outputs(f, Val(1), far), (; vrho = PtrMat(pointer(buf, 2n + 1), (1, n))))
            xc_evaluate!(out, f, Val(1), far)
            @test collect(out.vrho) == ref.vrho
        end
        # Limits: dataids name an identity, not an address range, and a type that exposes neither
        # its memory nor dataids is its own identity.
        @test !_overlaps(IdVec(pointer(buf), n), vrho)
        @test !_overlaps(BareVec(pointer(buf), n), vrho)
        @test all(==(42.0), view(buf, 1:2n))
    end
    # Wrappers that define Base.dataids and no parent.
    struct Wrap <: AbstractVector{Float64}
        data::Vector{Float64}
    end
    Base.size(a::Wrap) = size(a.data)
    Base.getindex(a::Wrap, i::Int) = a.data[i]
    Base.setindex!(a::Wrap, v, i::Int) = (a.data[i] = v)
    Base.dataids(a::Wrap) = Base.dataids(a.data)
    struct WrapM <: AbstractMatrix{Float64}
        data::Matrix{Float64}
    end
    Base.size(a::WrapM) = size(a.data)
    Base.getindex(a::WrapM, i::Int, j::Int) = a.data[i, j]
    Base.setindex!(a::WrapM, v, i::Int, j::Int) = (a.data[i, j] = v)
    Base.dataids(a::WrapM) = Base.dataids(a.data)
    x = fill(42.0, n)
    rejects(merge(xc_outputs(f, Val(1), x), (; vrho = reshape(x, 1, n))), Wrap(x))
    rejects(merge(xc_outputs(f, Val(1), x), (; e = WrapM(reshape(x, 1, n)))), x)
    o = fill(42.0, 1, n)
    rejects((; e = WrapM(o), vrho = o), fill(42.0, n))
    # SparseArrays defines Base.dataids of a SparseVector: its index and value arrays.
    s = sparsevec(collect(1:n), fill(42.0, n))
    rejects(merge(xc_outputs(f, Val(1), x), (; e = reshape(nonzeros(s), 1, n))), s)
end

@testitem "array layer rejects views that leave their parent or overlap themselves" begin
    msg = "xc: an output shares memory with an input or another output"
    n = 16
    f = lda_x()
    ref = xc_evaluate!(xc_outputs(f, Val(1), fill(42.0, n)), f, Val(1), fill(42.0, n))
    function rejects(out, x...)
        map(a -> fill!(a, 42.0), out)
        @test_throws msg xc_evaluate!(out, f, Val(1), x...)
        @test all(a -> all(==(42.0), a), out)
        @test all(a -> all(==(42.0), a), x)
        return nothing
    end
    # An unchecked view out of bounds: row 3 of a 2-row matrix lies on row 1, one column on.
    A = fill(42.0, 2, 2n)
    # With bounds checks forced, any access to it outside the call throws a BoundsError.
    @test_throws "xc: an array view lies outside its parent" xc_evaluate!((; e = Base.unsafe_view(A, 3:3, 1:n), vrho = fill(42.0, 1, n)), f, Val(1), view(A, 1, 1:n))
    @test all(==(42.0), A)
    # Outputs whose points share an address: a column index of step zero (on the input's row and
    # on another), a repeated column index, and a view of such a view.
    z = StepRangeLen(1, 0, n)
    rejects(merge(xc_outputs(f, Val(1), fill(1.0, n)), (; vrho = view(A, 1:1, z))), view(A, 1, 1:n))
    rejects(merge(xc_outputs(f, Val(1), fill(1.0, n)), (; vrho = view(A, 2:2, z))), view(A, 1, 1:n))
    rejects(merge(xc_outputs(f, Val(1), fill(1.0, n)), (; e = view(A, 2:2, [1:(n - 1); 1]))), fill(42.0, n))
    rejects(merge(xc_outputs(f, Val(1), fill(1.0, n)), (; e = view(view(A, 2:2, [1:(n - 1); 1]), :, 1:n))), fill(42.0, n))
    @test all(==(42.0), A)
    # An input with a column index of step zero reads one point n times; an output on another row
    # is apart from it.
    x = fill(42.0, n)
    B = vcat(x', fill(NaN, 1, n))
    out = merge(xc_outputs(f, Val(1), x), (; vrho = view(B, 2:2, 1:n)))
    xc_evaluate!(out, f, Val(1), view(B, 1, z))
    @test out == ref
    # Distinct vector indices are accepted.
    C = fill(NaN, 2, n)
    out = merge(xc_outputs(f, Val(1), x), (; vrho = view(C, 2:2, reverse(1:n) |> collect)))
    xc_evaluate!(out, f, Val(1), x)
    @test out == ref
end
@testitem "array layer rejects views out of bounds of their parent" begin
    using OffsetArrays
    msg = "xc: an array view lies outside its parent"
    n = 16
    f = lda_x()
    # One Memory holds two Arrays: A on its first n elements and B on the next n. Indices n+1:2n of
    # A, unchecked, are the elements of B. All numbers are 42.0, a valid input and the sentinel.
    M = fill!(Memory{Float64}(undef, 2n), 42.0)
    A = Base.wrap(Array, memoryref(M, 1), (n,))
    B = Base.wrap(Array, memoryref(M, n + 1), (1, n))
    function rejects(out, x...)
        @test_throws msg xc_evaluate!(out, f, Val(1), x...)
        @test all(==(42.0), M)
        return nothing
    end
    # Not read by the test outside the call: with bounds checks forced, such a read throws.
    unsafe_in = Base.unsafe_view(A, (n + 1):(2n))
    out = merge(xc_outputs(f, Val(1), A), (; vrho = B))
    # An input view out of bounds, directly, under a wrapper, and under an in-bounds view.
    rejects(out, unsafe_in)
    rejects(out, OffsetArray(unsafe_in, 0))
    rejects(out, view(unsafe_in, 1:n))
    rejects(merge(xc_outputs(f, Val(1), A), (; vrho = reshape(view(B, 1, :), 1, n))), reshape(unsafe_in, n))
    # An output view out of bounds, directly and under a wrapper; the input is in bounds.
    unsafe_out = Base.unsafe_view(reshape(A, 1, n), 1:1, (n + 1):(2n))
    rejects(merge(xc_outputs(f, Val(1), A), (; vrho = unsafe_out)), view(B, 1, :))
    rejects(merge(xc_outputs(f, Val(1), A), (; e = OffsetArray(unsafe_out, 0, 0))), fill(42.0, n))
    # In-bounds views of the same arrays are accepted.
    out = xc_outputs(f, Val(1), A)
    ref = xc_evaluate!(xc_outputs(f, Val(1), A), f, Val(1), fill(42.0, n))
    xc_evaluate!(merge(out, (; vrho = B)), f, Val(1), view(A, 1:n))
    @test B == ref.vrho
end

@testitem "array layer rejects outputs whose root repeats an address by a zero stride" begin
    using PureXC: _self_overlaps
    msg = "xc: an output shares memory with an input or another output"
    n = 16
    f = lda_x()
    ref = xc_evaluate!(xc_outputs(f, Val(1), fill(42.0, n)), f, Val(1), fill(42.0, n))
    # A matrix over a raw pointer with the strides it is given.
    struct StridedPtr <: AbstractMatrix{Float64}
        p::Ptr{Float64}
        sz::NTuple{2, Int}
        st::NTuple{2, Int}
    end
    Base.size(a::StridedPtr) = a.sz
    Base.strides(a::StridedPtr) = a.st
    Base.pointer(a::StridedPtr) = a.p
    Base.elsize(::Type{StridedPtr}) = sizeof(Float64)
    Base.getindex(a::StridedPtr, i::Int, j::Int) = unsafe_load(a.p, 1 + (i - 1) * a.st[1] + (j - 1) * a.st[2])
    Base.setindex!(a::StridedPtr, v, i::Int, j::Int) = unsafe_store!(a.p, v, 1 + (i - 1) * a.st[1] + (j - 1) * a.st[2])
    buf = fill(42.0, n)
    x = fill(42.0, n)
    GC.@preserve buf begin
        # Every point of e at one address.
        out = merge(xc_outputs(f, Val(1), x), (; e = StridedPtr(pointer(buf), (1, n), (1, 0))))
        @test_throws msg xc_evaluate!(out, f, Val(1), x)
        @test all(==(42.0), buf) && all(==(42.0), x)
        # A zero stride in a dimension of length 1 repeats no address.
        @test !_self_overlaps(StridedPtr(pointer(buf), (1, n), (0, 1)))
        @test _self_overlaps(StridedPtr(pointer(buf), (2, n), (0, 1)))
        out = merge(xc_outputs(f, Val(1), x), (; vrho = StridedPtr(pointer(buf), (1, n), (0, 1))))
        xc_evaluate!(out, f, Val(1), x)
        @test buf == vec(ref.vrho)
    end
end

@testitem "plain disjoint Arrays take the dense point loop" setup = [ArrayFixtures] begin
    using .ArrayFixtures
    # The dense method of `_evaluate_points!` has no other branch, so plain Arrays that pass the
    # overlap check run the dense loop; a view runs the checked loop.
    for f in (lda_x(), pbe_x(), pbe_c()), polarized in (false, true)
        ρ, σ = ArrayFixtures.inputs(Float64, 12; polarized)
        args = ArrayFixtures.call(f, Val(1), ρ, σ)
        out = xc_outputs(args...)
        @test isnothing(PureXC._check_no_overlap(values(out), args[3:end]))
        dense = which(PureXC._evaluate_points!, typeof.((out, args...)))
        vargs = (args[1], args[2], view(args[3], axes(args[3])...), args[4:end]...)
        checked = which(PureXC._evaluate_points!, typeof.((out, vargs...)))
        generic = which(PureXC._evaluate_points!, Tuple{NamedTuple, Any, Val, AbstractArray})
        @test checked === generic && dense !== generic
        xc_evaluate!(out, args...)
        @test ArrayFixtures.agrees(out, f, Val(1), ρ, σ, axes(ρ, ndims(ρ)))
    end
end
