# The measurement of `_float32_max_order(::Type{<:XC})` in `src/composite.jl`: for each order K
# from 0 to 3 (the largest `_float32_max_order` of the terms of `pbe`), whether every output of
# pbe(Float32), evaluated in pure Float32 (each term in Float32 and their weighted sum formed in
# Float32, the path `xc_derivatives` takes at an order at most `_float32_max_order`), stays within
# its bound over the accepted domain. Not run by the test suite.
#
#     julia --project=PureXC/test PureXC/test/scans/xc_float32_order.jl run W NW > out_W.txt
#     julia --project=PureXC/test PureXC/test/scans/xc_float32_order.jl summary out_*.txt
#
# `run W NW` evaluates the points whose index is W modulo NW (W = 0 … NW − 1), so NW processes
# cover every point once; `summary` merges their outputs and prints, for each order, whether it
# passes, and the measured value of `_float32_max_order`: the largest K such that every order from
# 0 to K passes.
#
# Points, in Float32, each kept at order K only where the input check of pbe(Float32) at order K
# accepts it:
# - densities n: the pbe_c density threshold n_thr = 1e-12, 2^p for p = −39 … 45 (2^45 is
#   `_gga_density_max(Float32, Val(K))` for K = 0 to 3), 21 log-spaced values over [1e-12, 1e-11],
#   and the largest accepted density;
# - restricted σ: 0, 10^m for m = −40 … −28, and 2^m for m = −149, −147, …, 127;
# - polarized: the splits of each n of `splits` below; with g each restricted σ value except the
#   odd powers of 2 (2^m for m = −149, −145, …, 127), σ_cc = g·(ρ_c/n)^(8/3) in each channel c
#   (0 in an empty channel) and σαβ = +√σαα·√σββ, 0 and −√σαα·√σββ;
# - the points ((0, 1.1215681e-12), (0, −0, 0)) and (1e-12, 0).
#
# Oracle: pbe(Float32) converted exactly to 256-bit BigFloat, evaluated on the inputs converted
# exactly. An order passes when, at every point and every output entry:
# - the non-finite rule holds: an entry is finite where its exact value fits in Float32, and the
#   infinity of the sign of its exact value where that exceeds floatmax(Float32) (a NaN never
#   passes); at every accepted point;
# - the Float32 bound of `test/libxc/runtests.jl` holds, max(|g − w| − floatmin(Float32), 0) ≤
#   c_32·eps(Float32)·max(1, κ̂)·|w| with c_32 of `pbe` (4.7e4 restricted, 5500 polarized), κ̂ the
#   error of pbe() (Float64) in ulps of Float64 against its exact function at the same point, and
#   |w| replaced by the largest |w| of its field where w is 0; at the points above every threshold
#   of both terms (`is_above` of `test/libxc/grid.jl`), as in the comparison with Libxc. Polarized
#   points within one rounding of 1 ± ζ = ε_ζ (`zeta_straddles`) and those whose total density,
#   summed exactly and in Float32, lies on different sides of a term's density threshold are
#   skipped: the threshold rule then applies to one evaluation and not to the other.

using PureXC, Printf

include(joinpath(dirname(@__DIR__), "libxc", "grid.jl"))

setprecision(BigFloat, 256)

const F32 = pbe(Float32)
const F64 = pbe(Float64)
const FB32 = PureXC._with_number_type(F32, BigFloat)
const FB64 = PureXC._with_number_type(F64, BigFloat)
const ORDERS = 0:3
const C_32 = Dict(:restricted => 4.7e4, :polarized => 5500.0)
const THRESHOLDS = thresholds_of(F32)

# The outputs of pbe(Float32) at order K with every term and the sum in Float32.
pure32(::Val{K}, x...) where {K} = PureXC._outputs_in(Float32, PureXC._term_sum(F32.terms, Val(K), x...))

components(x) = x isa Tuple ? collect(x) : [x]
fields(r) = [components(getfield(r, n)) for n in keys(r)]
big_point(p) = map(x -> x isa Tuple ? map(big, x) : big(x), p)
to_float64(p) = map(x -> x isa Tuple ? map(Float64, x) : Float64(x), p)
exact(fb, K, p) = [BigFloat.(c) for c in fields(PureXC._xc_derivatives_unchecked(fb, Val(K), big_point(p)...))]

function magnitudes(want)
    m = maximum(abs, want)
    return [iszero(w) ? m : abs(w) for w in want]
end

const ROWS = let
    rows = Float32[THRESHOLDS[2].density]
    append!(rows, [ldexp(1.0f0, p) for p in -39:45])
    append!(rows, Float32.(exp10.(range(-12, -11; length = 21))))
    push!(rows, maximum(PureXC._gga_density_max(Float32, Val(K)) for K in ORDERS))
    sort!(unique!(rows))
end
const SIGMAS = Float32[0; Float32.(exp10.(-40:-28)); [ldexp(1.0f0, m) for m in -149:2:127]]
const SIGMAS_POLARIZED = Float32[0; Float32.(exp10.(-40:-28)); [ldexp(1.0f0, m) for m in -149:4:127]]

# The polarized splits of the density n, as in `pbe_c_t2_max.jl`: one channel empty, equal and 3:1
# channels, a second channel of 2e-15, 1e-6 or 1 beside n in either order, the ratios near ε_ζ in
# either order and five log-spaced ratios.
function splits(n)
    part(r) = (Float32(n / (1 + r)), Float32(r * n / (1 + r)))
    out = Tuple{Float32, Float32}[(n, 0.0f0), (0.0f0, n), (n / 2, n / 2), (n / 4 * 3, n / 4)]
    for x in (2.0f-15, 1.0f-6, 1.0f0)
        push!(out, (n, x), (x, n))
    end
    for r in (5.0e-8, 1.0e-7, 2.0e-7)
        a, b = part(r)
        push!(out, (a, b), (b, a))
    end
    append!(out, part(r) for r in exp10.(range(-9, -1; length = 5)))
    return out
end

function all_points()
    pts = Any[(1.0f-12, 0.0f0), ((0.0f0, 1.1215681f-12), (0.0f0, -0.0f0, 0.0f0))]
    for n in ROWS, σ in SIGMAS
        push!(pts, (n, σ))
    end
    for n in ROWS, ρ in splits(n), g in SIGMAS_POLARIZED
        a = iszero(ρ[1]) ? 0.0f0 : Float32(big(g) * (big(ρ[1]) / big(n))^(8 // 3))
        b = iszero(ρ[2]) ? 0.0f0 : Float32(big(g) * (big(ρ[2]) / big(n))^(8 // 3))
        c = Float32(sqrt(big(a)) * sqrt(big(b)))
        for ab in unique((c, 0.0f0, -c))
            push!(pts, (ρ, (a, ab, b)))
        end
    end
    return pts
end

# Whether the threshold rule of a term applies to the Float32 evaluation and not to the exact one,
# or the reverse.
function straddles(p)
    p[1] isa Tuple || return false
    zeta_straddles(p, THRESHOLDS) && return true
    a, b = p[1]
    return any(t -> (big(a) + big(b) < big(t.density)) != (a + b < t.density), THRESHOLDS)
end

consistent(g, w) = isfinite(g) ? abs(w) <= floatmax(Float32) : (isinf(g) && abs(w) > floatmax(Float32) && sign(g) == sign(w))

mutable struct Tally
    n::Int
    above::Int
    skipped::Int
    nonfinite::Int
    nonfinite_at::Vector{Any}
    worst::Float64
    worst_at::Any
end
Tally() = Tally(0, 0, 0, 0, Any[], 0.0, nothing)

function scan_points(w, nw)
    pts = all_points()
    res = Dict{Tuple{Symbol, Int}, Tally}()
    t0 = time()
    for i in eachindex(pts)
        mod(i, nw) == w || continue
        p = pts[i]
        spin = p[1] isa Tuple ? :polarized : :restricted
        for K in ORDERS
            m = get!(Tally, res, (spin, K))
            PureXC._inputs_ok(F32, Val(K), p...) || continue
            straddles(p) && (m.skipped += 1; continue)
            m.n += 1
            r = pure32(Val(K), p...)
            if K <= PureXC._float32_max_order(typeof(F32))
                isequal(r, xc_derivatives(F32, Val(K), p...)) || error("pure32 differs from xc_derivatives at K = $K, $p")
            end
            got = fields(r)
            want = exact(FB32, K, p)
            names = keys(r)
            for k in eachindex(got), j in eachindex(got[k])
                consistent(got[k][j], want[k][j]) && continue
                m.nonfinite += 1
                length(m.nonfinite_at) < 5 && push!(m.nonfinite_at, (p, names[k], j, got[k][j], Float64(want[k][j])))
            end
            is_above(p, THRESHOLDS) || continue
            m.above += 1
            ref = nothing
            for k in eachindex(got)
                mag = magnitudes(want[k])
                for j in eachindex(got[k])
                    g = got[k][j]
                    isfinite(g) || continue
                    iszero(mag[j]) && continue
                    d = max(abs(big(g) - want[k][j]) - big(floatmin(Float32)), 0)
                    ratio = Float64(d / (big(eps(Float32)) * mag[j]))
                    if ratio > C_32[spin]
                        # κ̂ is formed only where the ratio at κ̂ = 1 exceeds the bound.
                        if isnothing(ref)
                            p64 = to_float64(p)
                            ref = (fields(PureXC._xc_derivatives_unchecked(F64, Val(K), p64...)), exact(FB64, K, p64))
                        end
                        mag64 = magnitudes(ref[2][k])
                        κ = iszero(mag64[j]) ? 0.0 : Float64(abs(big(ref[1][k][j]) - ref[2][k][j]) / (big(eps(Float64)) * mag64[j]))
                        ratio /= max(1, κ)
                    end
                    ratio > m.worst && ((m.worst, m.worst_at) = (ratio, (p, names[k], j)))
                end
            end
        end
    end
    for (k, m) in sort(collect(res); by = first)
        println("RESULT\t", k[1], "\t", k[2], "\t", m.n, "\t", m.above, "\t", m.skipped, "\t", m.nonfinite, "\t", repr(m.nonfinite_at), "\t", m.worst, "\t", repr(m.worst_at))
    end
    @printf("DONE worker %d of %d, %d points, %.0f s\n", w, nw, count(i -> mod(i, nw) == w, eachindex(pts)), time() - t0)
    return
end

function scan_summary(files)
    res = Dict{Tuple{String, Int}, Any}()
    for file in files, line in eachline(file)
        startswith(line, "RESULT\t") || continue
        c = split(line, '\t')
        key = (c[2], parse(Int, c[3]))
        n, above, skipped, nonfinite, worst = parse(Int, c[4]), parse(Int, c[5]), parse(Int, c[6]), parse(Int, c[7]), parse(Float64, c[9])
        old = get(res, key, (0, 0, 0, 0, String[], -Inf, ""))
        res[key] = (
            old[1] + n, old[2] + above, old[3] + skipped, old[4] + nonfinite,
            c[8] == "Any[]" ? old[5] : vcat(old[5], c[8]),
            worst > old[6] ? worst : old[6], worst > old[6] ? c[10] : old[7],
        )
    end
    println("spin\tK\tpoints\tabove\tskipped\tnon-finite entries\tmax ratio\tat\tbound\tverdict\tnon-finite examples")
    pass = Dict{Int, Bool}()
    for (k, v) in sort(collect(res); by = first)
        bound = C_32[Symbol(k[1])]
        ok = iszero(v[4]) && v[6] <= bound
        pass[k[2]] = get(pass, k[2], true) & ok
        println(join(k, "\t"), "\t", v[1], "\t", v[2], "\t", v[3], "\t", v[4], "\t", @sprintf("%.4g", v[6]), "\t", v[7], "\t", bound, "\t", ok ? "pass" : "FAIL", "\t", isempty(v[5]) ? "none" : join(v[5], " "))
    end
    k32 = -1
    for K in ORDERS
        get(pass, K, false) || break
        k32 = K
    end
    println("\nmeasured _float32_max_order(XC) for pbe(Float32): ", k32)
    return
end

if ARGS[1] == "run"
    scan_points(parse(Int, ARGS[2]), parse(Int, ARGS[3]))
elseif ARGS[1] == "summary"
    scan_summary(ARGS[2:end])
elseif ARGS[1] == "count"
    println(length(all_points()), " points")
else
    error("usage: xc_float32_order.jl run W NW | summary files... | count")
end
