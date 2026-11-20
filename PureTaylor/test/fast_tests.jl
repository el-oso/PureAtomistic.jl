@testmodule FastKernels begin
    using PureTaylor

    # Each kernel but fast_sqrt (checked for equality with Base's sqrt) with its Base reference and its
    # documented domain: (name, kernel, reference, Float32 intervals, Float64 intervals). The intervals
    # are closed and do not cross zero; together they cover the domain.
    const KERNELS = (
        (:cbrt, fast_cbrt, cbrt, ((floatmin(Float32), floatmax(Float32)),), ((floatmin(Float64), floatmax(Float64)),)),
        (:invcbrt, fast_invcbrt, x -> inv(cbrt(x)), ((floatmin(Float32), floatmax(Float32)),), ((floatmin(Float64), floatmax(Float64)),)),
        (:log, fast_log, log, ((floatmin(Float32), floatmax(Float32)),), ((floatmin(Float64), floatmax(Float64)),)),
        (:log1p, fast_log1p, log1p, ((nextfloat(-1.0f0), -0.0f0), (0.0f0, floatmax(Float32))), ((nextfloat(-1.0), -0.0), (0.0, floatmax(Float64)))),
        (:exp, fast_exp, exp, ((-85.0f0, -0.0f0), (0.0f0, 85.0f0)), ((-700.0, -0.0), (0.0, 700.0))),
        (:expm1, fast_expm1, expm1, ((-85.0f0, -0.0f0), (0.0f0, 85.0f0)), ((-700.0, -0.0), (0.0, 700.0))),
    )

    # Error of `y` in units of the spacing `eps(r)` at the reference `r`; zero when they are equal,
    # which covers infinite and zero references. A NaN error counts as Inf, so that it is the largest.
    _nan_as_inf(e::Float64) = isnan(e) ? Inf : e
    ulps(y::Float32, r::Float32) = y == r ? 0.0 : _nan_as_inf(abs(Float64(y) - Float64(r)) / Float64(eps(r)))
    ulps(y::Float64, r::BigFloat) = y == r ? 0.0 : _nan_as_inf(Float64(abs(big(y) - r) / eps(Float64(r))))

    # The largest error of `f` over the points `decode.(xs)` and the point where it occurs, against `g`
    # evaluated by `ref`, computed in `Threads.nthreads()` chunks.
    function max_error(f, g, ref, xs; decode = identity)
        chunks = collect(Iterators.partition(xs, cld(length(xs), Threads.nthreads())))
        tasks = map(chunks) do chunk
            Threads.@spawn begin
                worst, xworst = 0.0, decode(first(chunk))
                for u in chunk
                    x = decode(u)
                    e = ulps(f(x), ref(g, x))
                    if e > worst
                        worst, xworst = e, x
                    end
                end
                (worst, xworst)
            end
        end
        return reduce((a, b) -> a[1] >= b[1] ? a : b, fetch.(tasks))
    end

    # The number of points `decode.(xs)` where `f` and `g` differ, computed in `Threads.nthreads()` chunks.
    function mismatches(f, g, xs; decode = identity)
        chunks = collect(Iterators.partition(xs, cld(length(xs), Threads.nthreads())))
        tasks = map(chunk -> Threads.@spawn(count(u -> f(decode(u)) != g(decode(u)), chunk)), chunks)
        return sum(fetch.(tasks))
    end

    # The bit patterns of every Float32 in [lo, hi] (same sign).
    function float32_bits(lo::Float32, hi::Float32)
        a, b = minmax(reinterpret(UInt32, lo), reinterpret(UInt32, hi))
        return a:b
    end
    decode32(u::UInt32) = reinterpret(Float32, u)

    # Float32 oracle: the Base function evaluated in Float64, rounded to Float32.
    ref32(g, x::Float32) = Float32(g(Float64(x)))
    # Float64 oracle: the Base function in 128-bit BigFloat.
    ref64(g, x::Float64) = setprecision(() -> g(big(x)), BigFloat, 128)

    # Float64 test points in [lo, hi] (same sign): `n` log-uniform samples, the ends, every power of
    # two in the interval and its two neighbors.
    function points64(lo::Float64, hi::Float64, n::Int, rng)
        s = signbit(lo) ? -1.0 : 1.0
        a, b = minmax(abs(lo), abs(hi))
        a = max(a, nextfloat(0.0))
        b = min(b, floatmax(Float64))
        la, lb = log(a), log(b)
        xs = [s * clamp(exp(la + (lb - la) * rand(rng)), a, b) for _ in 1:n]
        for e in exponent(a):exponent(b), x in (prevfloat(2.0^e), 2.0^e, nextfloat(2.0^e))
            a <= x <= b && push!(xs, s * x)
        end
        append!(xs, (lo, hi))
        return xs
    end
end

@testitem "fast kernels keep the precision of their argument" begin
    for f in (fast_sqrt, fast_cbrt, fast_invcbrt, fast_log, fast_log1p, fast_exp, fast_expm1)
        @test f(1.5f0) isa Float32
        @test f(1.5) isa Float64
    end
end

@testitem "fast kernels fall back to Base for other Reals" begin
    using ForwardDiff
    @test fast_cbrt(big"2.0") == cbrt(big"2.0")                    # exact: the same Base call
    for (f, g) in ((fast_sqrt, sqrt), (fast_invcbrt, x -> inv(cbrt(x))), (fast_log, log), (fast_log1p, log1p), (fast_exp, exp), (fast_expm1, expm1))
        @test f(big"0.7") == g(big"0.7")                            # exact: the same Base call
    end
    @test ForwardDiff.derivative(fast_log, 2.0) == ForwardDiff.derivative(log, 2.0)   # exact: the same Base call
end

# The property the kernels exist for: a loop calling them can be vectorized.
@testitem "fast kernels compile to branch-free code without calls" begin
    using InteractiveUtils: code_llvm
    for f in (fast_sqrt, fast_cbrt, fast_invcbrt, fast_log, fast_log1p, fast_exp, fast_expm1), T in (Float32, Float64)
        io = IOBuffer()
        code_llvm(io, f, (T,); debuginfo = :none, optimize = true)
        ir = String(take!(io))
        @test occursin("ret ", ir)                                  # the IR was captured
        @test !occursin(r"\bbr i1\b", ir)
        calls = [m.match for m in eachmatch(r"\bcall\b[^\n]*", ir)]
        @test all(c -> occursin(r"@llvm\.", c), calls)
    end
end

# Exhaustive over every Float32 of each documented domain. Oracle: the Base function evaluated in
# Float64 and rounded to Float32. Bound (c), measured margin: 4 ulps for every kernel but fast_sqrt.
# fast_sqrt is the IEEE square root, correctly rounded, so it equals Base's sqrt exactly.
@testitem "fast kernels: Float32 accuracy, exhaustive" tags = [:exhaustive] setup = [FastKernels] begin
    using .FastKernels: KERNELS, max_error, mismatches, float32_bits, decode32, ref32
    @testset "sqrt Float32" begin
        @test mismatches(fast_sqrt, sqrt, float32_bits(0.0f0, Inf32); decode = decode32) == 0   # exact
    end
    # Largest error measured over the whole domain (Apple M-series, Julia 1.13), in ulps of the rounded
    # reference: cbrt 1, invcbrt 1, log 1, log1p 1, exp 1, expm1 2. The tolerance, 4 ulps, is at most
    # 4 times (≤ 100 times) each of them.
    tol = 4.0
    for (name, f, g, iv32, _) in KERNELS
        @testset "$name Float32" begin
            worst = maximum(first(max_error(f, g, ref32, float32_bits(lo, hi); decode = decode32)) for (lo, hi) in iv32)
            @test worst <= tol
        end
    end
end

# 10^7 log-uniform samples over each documented domain, its ends, the powers of two and their
# neighbors, and for log, log1p and expm1 the region near the point where the function is zero.
# Oracle: the Base function in 128-bit BigFloat. Bound (c), measured margin: 4 ulps for every kernel
# but fast_sqrt, which is correctly rounded and equals Base's sqrt exactly.
@testitem "fast kernels: Float64 accuracy, sampled" setup = [FastKernels] begin
    using .FastKernels: KERNELS, max_error, mismatches, points64, ref64
    using Random: Xoshiro
    n = 10^7
    @testset "sqrt Float64" begin
        @test mismatches(fast_sqrt, sqrt, points64(0.0, Inf, n, Xoshiro(20261120))) == 0           # exact
    end
    # Largest error measured with this seed (Apple M-series, Julia 1.13), in ulps: cbrt 1.3249,
    # invcbrt 1.5603, log 0.8489, log1p 1.4369, exp 1.0404, expm1 2.1782. The tolerance, 4 ulps, is at
    # most 5 times (≤ 100 times) each of them.
    tol = 4.0
    for (name, f, g, _, iv64) in KERNELS
        rng = Xoshiro(20261120)
        xs = reduce(vcat, (points64(lo, hi, n ÷ length(iv64), rng) for (lo, hi) in iv64))
        if name === :log
            # log is zero at 1: points 1 ± d, d log-uniform in [eps/2, 1/4].
            d = points64(eps(1.0) / 2, 0.25, n ÷ 10, rng)
            append!(xs, 1 .+ d, 1 .- d)
        end
        @testset "$name Float64" begin
            @test first(max_error(f, g, ref64, xs)) <= tol
        end
    end
end

# The accuracy checkers report a NaN result inside the domain as an infinite error, wherever it lies
# among the points: a kernel that is wrong at one point by returning NaN is the worst case.
@testitem "fast kernels: accuracy checkers report a NaN result" setup = [FastKernels] begin
    using .FastKernels: max_error, mismatches, float32_bits, decode32, ref32, points64, ref64
    using Random: Xoshiro
    nan_at_2(x) = x == 2 ? oftype(x, NaN) : fast_log(x)
    @test max_error(nan_at_2, log, ref32, float32_bits(1.5f0, 3.0f0); decode = decode32) == (Inf, 2.0f0)
    xs = points64(1.5, 3.0, 1000, Xoshiro(1))
    @test 2.0 in xs && last(xs) != 2.0
    @test max_error(nan_at_2, log, ref64, xs) == (Inf, 2.0)
    nan_sqrt(x) = x == 2 ? oftype(x, NaN) : fast_sqrt(x)
    @test mismatches(nan_sqrt, sqrt, float32_bits(1.5f0, 3.0f0); decode = decode32) == 1
    @test mismatches(nan_sqrt, sqrt, xs) == 1
end
