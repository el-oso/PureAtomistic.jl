# Step-1 scan of the reduced gradient t² = 2^q of pbe_c, which measures the table
# `_PBE_C_T2_EXPONENT_FLOAT32` / `_PBE_C_T2_EXPONENT_FLOAT64` of `src/gga/pbe_c.jl` (`_pbe_c_t2_max`)
# and the pbe_c part of the `_gga_density_max` measurement. Not run by the test suite.
#
#     julia --project=PureXC/test PureXC/test/scans/pbe_c_t2_max.jl T K mode [rows]
#
# T is Float32 or Float64, K the order, mode `restricted` or `polarized`, and rows a comma-separated
# list of row numbers (default all; the rows are independent, so they can run in separate
# processes). For each density row n (n_thr and 2^p for p = −39, −9, …, 381 in Float64 and
# p = −39, −33, …, 45 in Float32, up to n_max = `_gga_density_max`), q runs from 0 upward and each
# point is the split of n (restricted: n itself) at t² = 2^q with σ_αα = σ_αβ = σ_ββ, the polarized
# splits being those of `splits` below; points whose σ is not finite are skipped. A point passes when
# `point_error` of the test module PBEReference (test/pbe_tests.jl) is at most `point_bound`. The row
# prints its first failing q, with the split and the error, or `none` up to q = 140 (Float64) or 127
# (Float32), and the largest error (less its 8κ_ζ term) at each q. The table entry for (T, K) is the
# smallest first failing q over all rows and both modes, minus 1; above the end of the scan, the end.
# The domain check of t² is lifted, so that the scan can pass the current table. A polarized point
# whose total density, summed exactly and summed in T, lies on different sides of n_thr is skipped:
# the cut-off of rule 1 then applies to one of the two evaluations and not to the other.

using PureXC, PureTaylor, Printf

const TESTS = joinpath(dirname(@__DIR__), "pbe_tests.jl")
let ex = Meta.parseall(read(TESTS, String))
    modex = only(filter(e -> Meta.isexpr(e, :macrocall) && e.args[1] == Symbol("@testmodule"), ex.args))
    Core.eval(Main, Expr(:module, true, modex.args[3], modex.args[4]))
end
using .PBEReference: point_error, point_bound, gradient, zeta_conditioning

@eval PureXC _pbe_c_t2_max(::Type{R}, ::Val{K}) where {R, K} = ldexp(one(R), 1000)
@eval PureXC _pbe_c_t2_max(::Type{Float32}, ::Val{K}) where {K} = floatmax(Float32)

const T = ARGS[1] == "Float64" ? Float64 : ARGS[1] == "Float32" ? Float32 : error("T must be Float32 or Float64")
const K = parse(Int, ARGS[2])
const MODE = Symbol(ARGS[3])
MODE in (:restricted, :polarized) || error("mode must be restricted or polarized")
const F = pbe_c(T)
const P = exponent(PureXC._gga_density_max(T, Val(K)))
const ROWS = [T(F.thresholds.density); [ldexp(one(T), p) for p in -39:(T === Float64 ? 30 : 6):P]]
const QMAX = T === Float64 ? 140 : 127
rows = length(ARGS) >= 4 ? parse.(Int, split(ARGS[4], ",")) : collect(eachindex(ROWS))

# The polarized splits of the density n: one channel empty, equal and 3:1 channels, a second
# channel of 2e-15, 1e-6 or 1 beside n in either order, and splits of n with the smaller channel r
# times the larger for ratios r near ε_ζ (in either order) and five log-spaced r. A split by a ratio
# sums to n, so that it stays within n_max at the last row.
function splits(n)
    near = T === Float32 ? (5.0e-8, 1.0e-7, 2.0e-7) : (1.1e-16, 1.15e-16, 2.25e-16)
    rs = T === Float32 ? exp10.(range(-9, -1; length = 5)) : exp10.(range(-17, -1; length = 5))
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

# Whether the cut-off of rule 1 applies to the total of ρ in T and not to its exact total, or the
# reverse.
straddles(ρ::Real) = false
straddles(ρ::Tuple) = (big(ρ[1]) + big(ρ[2]) < big(F.thresholds.density)) != (ρ[1] + ρ[2] < F.thresholds.density)

for r in rows
    n = ROWS[r]
    points = MODE === :restricted ? Any[n] : splits(n)
    fail = nothing
    worst = Float64[]
    for q in 0:QMAX
        w = 0.0
        for ρ in points
            straddles(ρ) && continue
            σ = gradient(F, T, ρ, big(2.0)^q)
            all(isfinite, σ) || continue
            b = point_bound(F, ρ, T, K)
            e = point_error(F, K, ρ, σ; allow = b)
            if !(e <= b)
                fail = (q, ρ, e, b)
                break
            end
            w = max(w, e - 8 * zeta_conditioning(F, ρ))
        end
        isnothing(fail) || break
        push!(worst, w)
    end
    if isnothing(fail)
        @printf("ROW %d n=%s none to q=%d\n", r, string(n), QMAX)
    else
        q, ρ, e, b = fail
        @printf("ROW %d n=%s first failing q=%d at ρ=%s: error %.4g eps > bound %.4g\n", r, string(n), q, string(ρ), e, b)
    end
    println("WORST $r ", join((@sprintf("%.3g", w) for w in worst), " "))
    flush(stdout)
end
