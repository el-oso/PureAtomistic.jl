# The measurement behind the accuracy bounds above order 3 of `point_bound` in test/pbe_tests.jl
# and test/lda_tests.jl: for every point that each bound covers, the error the tests compare with
# it, less the 8κ_ζ term the bound adds at that point. Not run by the test suite.
#
#     julia --project=PureXC/test PureXC/test/scans/point_bounds.jl run W NW [bound] > out_W.txt
#     julia --project=PureXC/test PureXC/test/scans/point_bounds.jl summary out_*.txt
#
# `run W NW` evaluates the points whose index is W modulo NW (W = 0 … NW − 1), so NW processes
# cover every point once, and `bound`, one of the names below, restricts the run to the points of
# that bound; `summary` merges their outputs. The bounds and the points they cover:
#
# - "pbe Float32": the Float32 bound of pbe_x and pbe_c above `_float32_max_order` (orders 4 to 10);
# - "pbe 4-7" and "pbe 8-10": the Float64 bounds of pbe_x and pbe_c at orders 4 to 7 and 8 to 10;
# - "lda Float32": the Float32 bound of PW92 above `_float32_max_order` (orders 3 to 10);
# - "lda 4-10": the bound at orders 4 to 10 of lda_x (Float64 and Float32) and PW92 (Float64).
#
# PBE points (restricted and polarized): those of "PBE functionals at orders 0 to 10" (every
# point of `sweep_points` at s² or t² of 0, 1 and the order's limit, including the polarized pbe_c
# points of the item "polarized pbe_c at orders 7 to 10, every point"), the extra t² of 2^14, 2^22
# and 2^30 of that item's Float64 subset, and every row of the t² scan of pbe_c
# (`pbe_c_t2_max.jl`) at t² = 2^q for q from 0 to the order's limit. LDA points: those of "LDA
# functionals at orders 0 to 10" and "LDA functionals at uneven splits".
#
# The error is `point_error` of PBEReference (in eps of the order scale) and `error_eps` of
# LDAReference (relative to the largest entry of each output). For pbe_c the tests reduce each
# entry's error by SENSITIVITY_C times its sensitivity before comparing it with the bound, when the
# plain error exceeds the bound, so a point passes when the reduced error is within the bound; both
# the plain and the reduced maxima are reported. The sign of an infinite entry is judged against
# the current bound, as in the tests. A point whose error is infinite (an output that is NaN, or an
# infinity where the exact value fits) is listed.

using PureXC, PureTaylor, Printf

for (file, name) in (("pbe_tests.jl", :PBEReference), ("lda_tests.jl", :LDAReference))
    ex = Meta.parseall(read(joinpath(dirname(@__DIR__), file), String))
    modex = only(filter(e -> Meta.isexpr(e, :macrocall) && e.args[1] == Symbol("@testmodule") && e.args[3] == name, ex.args))
    Core.eval(Main, Expr(:module, true, modex.args[3], modex.args[4]))
end
using .PBEReference: PBEReference
using .LDAReference: LDAReference

const P = PBEReference
const L = LDAReference

# The bound a point falls under, or `nothing` where the bound is of kind (b) (orders 0 to 3).
function bucket(f, T, K)
    k32 = PureXC._float32_max_order(typeof(f))
    if family(f) === :gga
        T === Float32 && K > k32 && return "pbe Float32"
        T === Float64 && 4 <= K <= 7 && return "pbe 4-7"
        T === Float64 && K >= 8 && return "pbe 8-10"
        return nothing
    end
    T === Float32 && K > k32 && return "lda Float32"
    K >= 4 && return "lda 4-10"
    return nothing
end

# The constant of each bound, as `point_bound` returns it where it adds no 8κ term.
const CONSTANT = Dict(
    "pbe Float32" => () -> P.point_bound(pbe_x(Float32), 1.0f0, Float32, 10),
    "pbe 4-7" => () -> P.point_bound(pbe_x(), 1.0, Float64, 4),
    "pbe 8-10" => () -> P.point_bound(pbe_x(), 1.0, Float64, 8),
    "lda Float32" => () -> L.point_bound(lda_c_pw(Float32), 1.0f0, Float32, 10),
    "lda 4-10" => () -> L.point_bound(lda_x(), 1.0, Float64, 4),
)

# ---------------------------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------------------------

# The t² scan rows of pbe_c at order K in type T, as in `pbe_c_t2_max.jl`: density rows, the
# polarized splits of each, t² = 2^q for q up to the order's limit.
function t2_rows(T, K)
    f = pbe_c(T)
    Pn = exponent(PureXC._gga_density_max(T, Val(K)))
    rows = [T(f.thresholds.density); [ldexp(one(T), p) for p in -39:(T === Float64 ? 30 : 6):Pn]]
    near = T === Float32 ? (5.0e-8, 1.0e-7, 2.0e-7) : (1.1e-16, 1.15e-16, 2.25e-16)
    rs = T === Float32 ? exp10.(range(-9, -1; length = 5)) : exp10.(range(-17, -1; length = 5))
    function splits(n)
        part(r) = (T(n / (1 + r)), T(r * n / (1 + r)))
        out = Any[(n, zero(T)), (zero(T), n), (n / 2, n / 2), (n / 4 * 3, n / 4)]
        for x in (T(2.0e-15), T(1.0e-6), one(T))
            push!(out, (n, x), (x, n))
        end
        for r in near
            a, b = part(r)
            push!(out, (a, b), (b, a))
        end
        append!(out, part(r) for r in rs)
        return out
    end
    straddles(ρ::Real) = false
    straddles(ρ::Tuple) = (big(ρ[1]) + big(ρ[2]) < big(f.thresholds.density)) != (ρ[1] + ρ[2] < f.thresholds.density)
    qmax = exponent(PureXC._pbe_c_t2_max(T, Val(K)))
    out = Any[]
    for n in rows, ρ in Any[n; splits(n)], q in 0:qmax
        straddles(ρ) && continue
        push!(out, (ρ, big(2.0)^q))
    end
    return out
end

# Every point: (functional builder, T, K, ρ, g), with g the reduced gradient (s² or t²) for a GGA
# and `nothing` for an LDA. Repeated points are kept once.
function all_points()
    pts = Any[]
    for T in (Float32, Float64), build in (pbe_x, pbe_c), K in 0:10
        f = build(T)
        isnothing(bucket(f, T, K)) && continue
        lim = build === pbe_x ? PureXC._pbe_x_s2_max(T, Val(K)) : PureXC._pbe_c_t2_max(T, Val(K))
        gs = Any[big(0), big(1), big(lim) * (1 - big(2.0)^-10)]
        T === Float64 && build === pbe_c && K >= 7 && append!(gs, (big(2.0)^14, big(2.0)^22, big(2.0)^30))
        for ρ in P.sweep_points(build, T, K), g in gs
            push!(pts, (build, T, K, ρ, g))
        end
        if build === pbe_c
            for (ρ, g) in t2_rows(T, K)
                push!(pts, (build, T, K, ρ, g))
            end
        end
    end
    for T in (Float32, Float64), build in (lda_x, lda_c_pw, lda_c_pw_mod), K in 0:10
        g = build(T)
        isnothing(bucket(g, T, K)) && continue
        Pn = exponent(PureXC._lda_density_max(T, Val(K)))
        for p in unique([0:16:Pn; 5; Pn])
            n = ldexp(one(T), p)
            splits = Any[n, (n, zero(T)), (zero(T), n), (n / 2, n / 2), (n / 4 * 3, n / 4)]
            for x in (T(2.0e-15), T(1.0e-6), one(T))
                push!(splits, (n, x), (x, n))
            end
            for r in (T === Float32 ? (5.0e-8, 6.3e-8, 1.0e-7, 2.0e-7) : (1.1e-16, 1.122e-16, 1.15e-16, 2.25e-16))
                push!(splits, (n / 2, T(r * n / 2)), (T(r * n / 2), n / 2))
            end
            for x in splits
                push!(pts, (build, T, K, x, nothing))
            end
        end
        # The points of "LDA functionals at uneven splits" (orders 1 to 3).
        1 <= K <= 3 || continue
        for p in unique([0:8:Pn; 5; Pn]), ρβ in (2.0e-15, 1.0e-6, 1.0), flip in (false, true)
            x = (ldexp(one(T), p), T(ρβ))
            push!(pts, (build, T, K, flip ? reverse(x) : x, nothing))
        end
    end
    return unique(pts)
end

# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------

# The plain error of a PBE point and a function computing its reduced error (pbe_c; the plain
# error for pbe_x), as `point_error` computes them, with the sign of an infinite entry judged
# against `allow`.
function pbe_errors(f, K, ρ, σ, allow)
    T = PureXC._result_type(f, ρ, σ)
    got = xc_derivatives(f, Val(K), ρ, σ)
    want = P.U(P.exact_big(f), Val(K), map(big, ρ), map(big, σ))
    S = PureXC._evaluation_type(f, T, Val(K))
    lay = P.layout(ρ isa Tuple ? 2 : 1, ρ isa Tuple ? 3 : 1, K)
    nrm = P.normalizers(f, K, ρ, σ, S, lay)
    sc = P.scales(f, K, ρ, σ, want, T, nrm, lay)
    plain = P.error_eps(got, want, nrm, lay, sc, T; bound = allow)
    reduced() = if f isa PureXC.PBECorrelation
        Δ = P.sensitivity(f, K, ρ, σ, want, S)
        P.error_eps(got, want, nrm, lay, sc, T; less = [P.SENSITIVITY_C .* Δ[k] .* nrm[k] for k in eachindex(nrm)], bound = allow)
    else
        plain
    end
    return plain, reduced
end

mutable struct Max
    plain::Float64
    plain_at::Any
    reduced::Float64
    reduced_at::Any
    n::Int
    skipped::Int
    infinite::Vector{Any}
end
Max() = Max(-Inf, nothing, -Inf, nothing, 0, 0, Any[])

function scan_points(w, nw, only = nothing)
    pts = all_points()
    isnothing(only) || filter!(p -> bucket(p[1](p[2]), p[2], p[3]) == only, pts)
    res = Dict{Tuple{String, String, String, Symbol, Int}, Max}()
    t0 = time()
    for i in eachindex(pts)
        mod(i, nw) == w || continue
        build, T, K, ρ, g = pts[i]
        f = build(T)
        b = bucket(f, T, K)
        spin = ρ isa Tuple ? :polarized : :restricted
        m = get!(Max, res, (b, string(build), string(T), spin, K))
        if family(f) === :gga
            σ = P.gradient(f, T, ρ, g)
            all(isfinite, σ) || (m.skipped += 1; continue)
            allow = P.point_bound(f, ρ, T, K)
            extra = allow - CONSTANT[b]()
            plain, reduced = try
                pbe_errors(f, K, ρ, σ, allow)
            catch err
                err isa ArgumentError || rethrow()
                m.skipped += 1
                continue
            end
            m.n += 1
            isfinite(plain) || push!(m.infinite, (ρ, σ, plain))
            if plain - extra > m.plain
                m.plain, m.plain_at = plain - extra, (ρ, σ)
            end
            # The reduced error is at most the plain one, so it can raise the maximum only where
            # the plain error does.
            if plain - extra > m.reduced
                r = reduced() - extra
                r > m.reduced && ((m.reduced, m.reduced_at) = (r, (ρ, σ)))
            end
        else
            allow = L.point_bound(f, ρ, T, K)
            extra = allow - CONSTANT[b]()
            e = L.error_eps(xc_derivatives(f, Val(K), ρ), xc_derivatives(L.exact_big(f), Val(K), map(big, ρ)), T)
            m.n += 1
            isfinite(e) || push!(m.infinite, (ρ, e))
            e - extra > m.plain && ((m.plain, m.plain_at) = (e - extra, ρ))
            e - extra > m.reduced && ((m.reduced, m.reduced_at) = (e - extra, ρ))
        end
    end
    for (k, m) in sort(collect(res); by = first)
        println("RESULT\t", join(k, "\t"), "\t", m.n, "\t", m.skipped, "\t", m.plain, "\t", repr(m.plain_at), "\t", m.reduced, "\t", repr(m.reduced_at), "\t", repr(m.infinite))
    end
    @printf("DONE worker %d of %d, %d points, %.0f s\n", w, nw, count(i -> mod(i, nw) == w, eachindex(pts)), time() - t0)
    return
end

function scan_summary(files)
    res = Dict{Tuple, Any}()
    for file in files, line in eachline(file)
        startswith(line, "RESULT\t") || continue
        c = split(line, '\t')
        key = (c[2], c[3], c[4], c[5], parse(Int, c[6]))
        n, skipped, plain, reduced = parse(Int, c[7]), parse(Int, c[8]), parse(Float64, c[9]), parse(Float64, c[11])
        inf = c[13]
        old = get(res, key, (0, 0, -Inf, "", -Inf, "", String[]))
        res[key] = (
            old[1] + n, old[2] + skipped,
            plain > old[3] ? plain : old[3], plain > old[3] ? c[10] : old[4],
            reduced > old[5] ? reduced : old[5], reduced > old[5] ? c[12] : old[6],
            inf == "Any[]" ? old[7] : vcat(old[7], inf),
        )
    end
    println("bound\tfunctional\tT\tspin\tK\tpoints\tskipped\tmax plain\tat\tmax reduced\tat\tinfinite")
    for (k, v) in sort(collect(res); by = first)
        println(join(k, "\t"), "\t", v[1], "\t", v[2], "\t", @sprintf("%.4g", v[3]), "\t", v[4], "\t", @sprintf("%.4g", v[5]), "\t", v[6], "\t", isempty(v[7]) ? "none" : join(v[7], " "))
    end
    println("\nPer bound: the largest reduced error over every case it covers")
    for b in sort(unique(first.(collect(keys(res)))))
        ks = filter(k -> k[1] == b, collect(keys(res)))
        best = argmax(k -> res[k][5], ks)
        @printf("%-12s constant %-8g max reduced %.4g (%s %s %s K=%d at %s), max plain %.4g\n", b, CONSTANT[b](), res[best][5], best[2], best[3], best[4], best[5], res[best][6], maximum(k -> res[k][3], ks))
    end
    return
end

if ARGS[1] == "run"
    scan_points(parse(Int, ARGS[2]), parse(Int, ARGS[3]), get(ARGS, 4, nothing))
elseif ARGS[1] == "summary"
    scan_summary(ARGS[2:end])
elseif ARGS[1] == "count"
    pts = all_points()
    for b in unique(bucket(p[1](p[2]), p[2], p[3]) for p in pts)
        println(b, " ", count(p -> bucket(p[1](p[2]), p[2], p[3]) == b, pts))
    end
else
    error("usage: point_bounds.jl run W NW | summary files... | count")
end
