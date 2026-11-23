# Comparison of PureXC with Libxc 7.0.0 and with the same Julia code in 256-bit BigFloat, on the
# oracle grid of `grid.jl`.
#
#   julia --check-bounds=yes --project=PureXC/test/libxc PureXC/test/libxc/runtests.jl
#
# For each functional (lda_x, lda_c_pw, lda_c_pw_mod, pbe_x, pbe_c and pbe() = pbe_x + pbe_c) and
# spin case, the grid points are labeled above or below the thresholds (`is_above`). Every gated
# item takes the above points; the below points are printed, not gated (the rule items of the main
# suite gate them). Polarized points within one rounding of 1 ± ζ = ε_ζ (`zeta_straddles`) are known
# differences: the factor (1 ± ζ)^p is kept by one evaluation and replaced by another, so their
# derivatives differ by the series part of that factor. They are excluded from the gated items and
# listed with their count and largest differences.
#
# Oracles: Libxc's outputs (orders 0 to 2, Float64; Libxc 7.0.0 is built without third
# derivatives), and the same functional converted exactly to BigFloat at 256 bits (`exact_big`),
# evaluated on the same inputs converted exactly. Every component of every output field is compared
# in Libxc's layout (zk·n against e; the rows of vrho, vsigma, v2rho2, v2rhosigma and v2sigma2),
# so a layout difference fails. A Float32 call is compared with the exact function of the Float32
# functional (its parameters rounded to Float32), which measures its evaluation error; the
# difference between that function and the Float64 functional's is printed, not gated.
#
# The zero rule: where the BigFloat value of a component is exactly 0, a relative bound r·|x| is
# replaced by r·m, with m the largest |BigFloat value| among the components of that field at that
# point. The Richardson check also holds an entry below 1e-6·m to 1e-6·m (`richardson_relerr`).

using Libxc: Libxc
using PureXC
using Printf
using Test

include("grid.jl")

setprecision(BigFloat, 256)

const FUNCTIONALS = (
    ("lda_x", lda_x, (:lda_x,)),
    ("lda_c_pw", lda_c_pw, (:lda_c_pw,)),
    ("lda_c_pw_mod", lda_c_pw_mod, (:lda_c_pw_mod,)),
    ("pbe_x", pbe_x, (:gga_x_pbe,)),
    ("pbe_c", pbe_c, (:gga_c_pbe,)),
    ("pbe", pbe, (:gga_x_pbe, :gga_c_pbe)),
)
const SPINS = (:restricted, :polarized)

# Bound (a), the user's floor: relative difference from Libxc (orders 0 to 2) and from BigFloat
# (orders 3 and 4), Float64.
const FLOOR = 1.0e-6
# Bound (a): Richardson central differences of Libxc's order 2 are accurate to about 1e-6 to 1e-4
# (design, "Validation against Libxc").
const RICHARDSON_TOL = 1.0e-3
# Bound (c) of item (B), |jl − big| ≤ c·max(|libxc − big|, 4·eps·|big|), Float64, orders 0 to 2, per
# functional and spin case: (c, the measured maximum over the above points). Each c is about 10
# times its measurement.
const C_B = Dict(
    ("lda_x", :restricted) => (13, 1.251), ("lda_x", :polarized) => (14, 1.322),
    ("lda_c_pw", :restricted) => (21, 2.021), ("lda_c_pw", :polarized) => (50, 4.972),
    ("lda_c_pw_mod", :restricted) => (22, 2.145), ("lda_c_pw_mod", :polarized) => (58, 5.758),
    ("pbe_x", :restricted) => (490, 48.89), ("pbe_x", :polarized) => (690, 68.24),
    ("pbe_c", :restricted) => (940, 93.55), ("pbe_c", :polarized) => (3400, 333.6),
    ("pbe", :restricted) => (1.2e4, 1121), ("pbe", :polarized) => (1400, 132.5),
)
# Bound (c) of the Float32 items, max(|jl32 − big32| − floatmin(Float32), 0) ≤
# c·eps(Float32)·max(1, κ̂)·|big32| (`float32_ratio`), orders 0 to 4, per functional and spin case:
# (c, the measured maximum over the above points and the orders 0 to 4, except order 3 of polarized
# correlation). Each c is about 10 times its measurement. The order-3 maxima of polarized
# correlation are within c: lda_c_pw 0.45, lda_c_pw_mod 0.47, pbe_c 870.5. pbe() at order 3, which
# evaluates its terms in Float64 and rounds once: 0.50 (restricted and polarized).
const C_32 = Dict(
    ("lda_x", :restricted) => (33, 3.266), ("lda_x", :polarized) => (39, 3.874),
    ("lda_c_pw", :restricted) => (79, 7.818), ("lda_c_pw", :polarized) => (320, 31.45),
    ("lda_c_pw_mod", :restricted) => (73, 7.236), ("lda_c_pw_mod", :polarized) => (220, 21.97),
    ("pbe_x", :restricted) => (1600, 160.0), ("pbe_x", :polarized) => (1600, 159.0),
    ("pbe_c", :restricted) => (1.8e4, 1736), ("pbe_c", :polarized) => (6.2e4, 6156),
    ("pbe", :restricted) => (4.7e4, 4653), ("pbe", :polarized) => (5500, 549.1),
)
# The number of known differences per functional and spin case: the polarized rows ζ = ±(1 − eps)
# of the grid, 2·131 points (LDA) and 2·131·42 (GGA).
const KNOWN_COUNT = Dict(
    ("lda_x", :restricted) => 0, ("lda_x", :polarized) => 262,
    ("lda_c_pw", :restricted) => 0, ("lda_c_pw", :polarized) => 262,
    ("lda_c_pw_mod", :restricted) => 0, ("lda_c_pw_mod", :polarized) => 262,
    ("pbe_x", :restricted) => 0, ("pbe_x", :polarized) => 11004,
    ("pbe_c", :restricted) => 0, ("pbe_c", :polarized) => 11004,
    ("pbe", :restricted) => 0, ("pbe", :polarized) => 11004,
)

# ---------------------------------------------------------------------------------------------
# Layouts and components
# ---------------------------------------------------------------------------------------------

# The functional converted exactly to BigFloat: its parameters and thresholds are Float64 (or
# Float32) values, which BigFloat holds exactly, so it is the same function.
exact_big(f) = PureXC._with_number_type(f, BigFloat)

# The order of each output field, from its name.
field_order(name::Symbol) = name === :e ? 0 : (s = String(name); isdigit(s[2]) ? parse(Int, s[2]) : 1)

# The components of an output field as a vector.
components(x::Real) = [x]
components(x::Tuple) = collect(x)

# The multi-index α (counts over the nρ density and nσ σ inputs) of each component of each output
# field to order K, in the output layout of `xc_derivatives`.
function layout(nρ, nσ, K)
    ms(n, m) = m == 0 ? [()] : [(i, r...) for i in 1:n for r in ms(n, m - 1) if isempty(r) || r[1] >= i]
    cnt(s, n) = ntuple(i -> count(==(i), s), n)
    out = Vector{Vector{NTuple{nρ + nσ, Int}}}([[ntuple(_ -> 0, nρ + nσ)]])
    for j in 1:K, a in j:-1:0
        b = j - a
        b > 0 && iszero(nσ) && continue
        push!(out, [(cnt(r, nρ)..., cnt(s, nσ)...) for r in ms(nρ, a) for s in ms(nσ, b)])
    end
    return out
end

nvars(family, spin) = (spin === :restricted ? 1 : 2, family === :lda ? 0 : (spin === :restricted ? 1 : 3))

# ---------------------------------------------------------------------------------------------
# Libxc
# ---------------------------------------------------------------------------------------------

# Libxc's outputs at orders 0 to 2 at the points `pts`, as the components of each field of
# `xc_derivatives` at order 2, in its order: e = n·zk, then the rows of each derivative. A sum of
# functionals adds the outputs of its terms.
function libxc_fields(ids, family, spin, pts)
    pol = spin === :polarized
    rho = pol ? reduce(vcat, [collect(p[1]) for p in pts]) : [p[1] for p in pts]
    kw = if family === :lda
        (; rho)
    else
        (; rho, sigma = pol ? reduce(vcat, [collect(p[2]) for p in pts]) : [p[2] for p in pts])
    end
    names = family === :lda ? (:vrho, :v2rho2) : (:vrho, :vsigma, :v2rho2, :v2rhosigma, :v2sigma2)
    total = Dict{Symbol, Matrix{Float64}}()
    zk = zeros(length(pts))
    for id in ids
        r = Libxc.evaluate(Libxc.Functional(id; n_spin = pol ? 2 : 1); kw..., derivatives = 0:2)
        zk .+= vec(r.zk)
        for n in names
            m = getfield(r, n)
            m = pol ? m : reshape(m, 1, :)
            total[n] = haskey(total, n) ? total[n] .+ m : copy(m)
        end
    end
    n = [p[1] isa Tuple ? big(p[1][1]) + big(p[1][2]) : big(p[1]) for p in pts]
    out = Vector{Vector{Vector{BigFloat}}}(undef, length(pts))
    for i in eachindex(pts)
        out[i] = pushfirst!([big.(total[nm][:, i]) for nm in names], [n[i] * big(zk[i])])
    end
    return out
end

# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------

# The magnitude of each component of a field for a relative bound: |big|, or by the zero rule m.
function magnitudes(want::Vector{BigFloat})
    m = maximum(abs, want)
    return [iszero(w) ? m : abs(w) for w in want]
end

# The largest relative error of `got` against `want` over a field, by the zero rule; a NaN, or an
# infinity, is Inf. A field whose BigFloat values are all 0 must be exactly 0.
function field_relerr(got, want::Vector{BigFloat})
    mag = magnitudes(want)
    worst = 0.0
    for i in eachindex(want)
        g = big(got[i])
        isfinite(g) || return Inf
        d = abs(g - want[i])
        e = iszero(mag[i]) ? (iszero(d) ? 0.0 : Inf) : Float64(d / mag[i])
        worst = max(worst, e)
    end
    return worst
end

# The largest relative error of Richardson differences `got` against `want` over a field: each
# entry relative to the larger of |want| and 1e-6·m, with m the largest |want| of the field, and an
# entry whose BigFloat value is exactly 0 relative to m (the zero rule). Libxc's
# order-2 components carry rounding noise relative to the largest component of their field, which
# their differences carry into the small entries: at ((388, 9.95), (3.07e10, 2.32e8, 1.76e6)),
# gga_x_pbe's v2rhosigma_αββ, exactly 0, is −7.4e-24 beside −1.8e-7 in the same field, and its
# difference in σββ adds 1.2e-27 (1.8% relative) to the v3rhosigma2 entry 6.9e-26 of pbe(), whose
# field's largest entry is 1.8e-13. A field whose BigFloat values are all 0 must be exactly 0.
function richardson_relerr(got, want::Vector{BigFloat})
    m = maximum(abs, want)
    worst = 0.0
    for i in eachindex(want)
        g = big(got[i])
        isfinite(g) || return Inf
        d = abs(g - want[i])
        ref = iszero(want[i]) ? m : max(abs(want[i]), m / 10^6)
        e = iszero(ref) ? (iszero(d) ? 0.0 : Inf) : Float64(d / ref)
        worst = max(worst, e)
    end
    return worst
end

# Item (A), against Libxc: |jl − libxc| ≤ FLOOR·|libxc| per component, and |jl| ≤ FLOOR·m where
# the BigFloat value is exactly 0. Returns the largest ratio of the difference to FLOOR·magnitude.
function libxc_ratio(got, lx::Vector{BigFloat}, want::Vector{BigFloat})
    m = maximum(abs, want)
    worst = 0.0
    for i in eachindex(want)
        g = big(got[i])
        isfinite(g) || return Inf
        num, den = iszero(want[i]) ? (abs(g), m) : (abs(g - lx[i]), abs(lx[i]))
        r = iszero(den) ? (iszero(num) ? 0.0 : Inf) : Float64(num / den) / FLOOR
        worst = max(worst, r)
    end
    return worst
end

# Item (B): the largest |jl − big| / max(|libxc − big|, 4·eps·|big|) over a field, with |big|
# replaced by m by the zero rule.
function conditioning_ratio(got, lx::Vector{BigFloat}, want::Vector{BigFloat})
    mag = magnitudes(want)
    worst = 0.0
    for i in eachindex(want)
        g = big(got[i])
        isfinite(g) || return Inf
        den = max(abs(lx[i] - want[i]), 4 * big(eps(Float64)) * mag[i])
        num = abs(g - want[i])
        r = iszero(den) ? (iszero(num) ? 0.0 : Inf) : Float64(num / den)
        worst = max(worst, r)
    end
    return worst
end

# The Float32 ratio max(|jl32 − big32| − floatmin(Float32), 0) / (eps(Float32)·max(1, κ̂)·|big32|),
# with big32 the exact value of the Float32 functional, κ̂ = |jl64 − big64|/(eps(Float64)·|big64|)
# the Float64 error in ulps at the same point (big64 the exact value of the Float64 functional), and
# |big| replaced by m by the zero rule. floatmin(Float32) is the absolute accuracy to which a Float32
# output can hold a value, as in the error metric of the main suite: an exact value below it
# underflows. An infinity is accepted, with ratio 0, only where |big32| > floatmax(Float32) with the
# same sign; a NaN or another infinity is Inf.
function float32_ratio(got32, got64, want32::Vector{BigFloat}, want64::Vector{BigFloat})
    mag32, mag64 = magnitudes(want32), magnitudes(want64)
    worst = 0.0
    for i in eachindex(want32)
        g = got32[i]
        if !isfinite(g)
            (isinf(g) && abs(want32[i]) > floatmax(Float32) && sign(g) == sign(want32[i])) || return Inf
            continue
        end
        d = max(abs(big(g) - want32[i]) - big(floatmin(Float32)), 0)
        iszero(mag32[i]) && (iszero(d) ? continue : return Inf)
        κ = iszero(mag64[i]) ? 0.0 : Float64(abs(big(got64[i]) - want64[i]) / (big(eps(Float64)) * mag64[i]))
        worst = max(worst, Float64(d / (big(eps(Float32)) * max(1, κ) * mag32[i])))
    end
    return worst
end

# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------

# The output fields of `f` at order K at the point `p`, as component vectors, or `nothing` where
# the inputs are a caller error.
function fields_at(f, K, p)
    r = try
        xc_derivatives(f, Val(K), p...)
    catch err
        err isa ArgumentError || rethrow()
        return nothing
    end
    return [components(getfield(r, n)) for n in keys(r)]
end

big_point(p) = map(x -> x isa Tuple ? map(big, x) : big(x), p)

# The BigFloat outputs at order K at the point, without the input check: a σαβ formed as
# √σαα·√σββ in Float64 can exceed the exact bound by a rounding, which the check of a BigFloat
# functional, at 256 bits, rejects. The point has passed the check of the Float64 functional.
function bigfields(fb, K, p)
    r = PureXC._xc_derivatives_unchecked(fb, Val(K), big_point(p)...)
    return [BigFloat.(components(getfield(r, n))) for n in keys(r)]
end

# Richardson differences of Libxc's order 2 for the order-3 components at the points `pts`: each
# order-3 component ∂_i∂_j∂_k e is the derivative in input i of the order-2 component ∂_j∂_k e.
#
# A density input takes central differences D(h) = (f(x + h) − f(x − h))/(2h); a σ input,
# one-sided second-order differences D(h) = s·(−3f(x) + 4f(x + s·h) − f(x + 2s·h))/(2h), forward
# for σαα, σββ and restricted σ and backward for σαβ (the grid's parallel gradients put σαβ at its
# bound √σαα·√σββ, which an increase of σαβ or a decrease of σαα or σββ would cross). The value is
# the Richardson extrapolation (4D(h/2) − D(h))/3.
#
# The step h is 1e-3 of the scale on which the input varies: x for a density; for σ (restricted,
# σαα and σββ) max(σ, σ₁), with σ₁ the smaller of the σ at which the reduced gradient s of its
# channel (restricted: of n) is 1 and the σ at which t is 1 (φ = 1; polarized, a quarter of it, at
# the total density). At σ ≪ σ₁ a step of 1e-3·σ changes the outputs by less than their rounding,
# and a step of 1e-3 of the larger of the two scales has a truncation error near 1e-3 where the
# other is far smaller. For σαβ, h = 1e-3·max(|σαβ|, √(σ₁α·σ₁β)), at most
# (σαβ + √σαα·√σββ)/4, so that σαβ − 2h stays within the bound.
#
# Of the inputs i of the component, the one whose order-2 component g has the smallest |g|/h_i is
# taken, with g the BigFloat value at that point (`wants`, the BigFloat fields of each point to
# order 2 or more): the differences of g over the steps are of size h_i·|∂_i g|, so |g|/h_i
# measures the digits they lose to cancellation, for each i against the same ∂_i g. A large
# order-2 component loses most of them: for example a v3sigma3 entry that only correlation
# contributes to, taken from the exchange part of v2sigma2.
function richardson_order3(ids, family, spin, pts, wants)
    nρ, nσ = nvars(family, spin)
    N = nρ + nσ
    lay2 = layout(nρ, nσ, 2)
    lay3 = layout(nρ, nσ, 3)
    # position of each order-2 multi-index: (field index, component index) in the fields to order 2
    pos2 = Dict{NTuple{N, Int}, Tuple{Int, Int}}()
    for (fi, αs) in enumerate(lay2), (ci, α) in enumerate(αs)
        sum(α) == 2 && (pos2[α] = (fi, ci))
    end
    flat(p) = (spin === :restricted ? (p[1],) : p[1])..., (nσ == 0 ? () : spin === :restricted ? (p[2],) : p[2])...
    unflat(v) = if spin === :restricted
        nσ == 0 ? (v[1],) : (v[1], v[2])
    else
        nσ == 0 ? ((v[1], v[2]),) : ((v[1], v[2]), (v[3], v[4], v[5]))
    end
    xs = [collect(flat(p)) for p in pts]
    # σ at t = 1 (φ = 1) at the total density n, 16·k_F·n²/π.
    sigma_t(n) = 16 * cbrt(3π^2 * n) * n^2 / π
    function step(x, i)
        i <= nρ && return 1.0e-3 * x[i]
        spin === :restricted && return 1.0e-3 * max(x[i], min(_sigma_restricted(x[1], 1), sigma_t(x[1])))
        st = sigma_t(x[1] + x[2]) / 4
        s1a, s1b = min(_sigma_channel(x[1], 1), st), min(_sigma_channel(x[2], 1), st)
        i == nρ + 1 && return 1.0e-3 * max(x[i], s1a)
        i == nρ + 3 && return 1.0e-3 * max(x[i], s1b)
        return min(1.0e-3 * max(abs(x[i]), sqrt(s1a * s1b)), (x[i] + sqrt(x[3]) * sqrt(x[5])) / 4)
    end
    hs = [[step(x, i) for i in 1:N] for x in xs]
    direction(i) = i <= nρ ? 0 : (spin === :polarized && i == nρ + 2 ? -1 : 1)
    # Libxc order 2 at the points shifted by t·h_i in input i
    shifted(i, t) = libxc_fields(ids, family, spin, [unflat(setindex!(copy(x), x[i] + t * hs[pt][i], i)) for (pt, x) in pairs(xs)])
    deriv = Vector{Any}(undef, N)
    base = libxc_fields(ids, family, spin, pts)
    for i in 1:N
        s = direction(i)
        if s == 0
            fp1, fm1, fph, fmh = shifted(i, 1), shifted(i, -1), shifted(i, 0.5), shifted(i, -0.5)
            deriv[i] = (pt, f2, c) -> begin
                h = big(hs[pt][i])
                D1 = (fp1[pt][f2][c] - fm1[pt][f2][c]) / (2h)
                Dh = (fph[pt][f2][c] - fmh[pt][f2][c]) / h
                (4Dh - D1) / 3
            end
        else
            f1, f2s, fh = shifted(i, s), shifted(i, 2s), shifted(i, s / 2)
            deriv[i] = (pt, f2, c) -> begin
                h = big(hs[pt][i])
                D1 = s * (-3 * base[pt][f2][c] + 4 * f1[pt][f2][c] - f2s[pt][f2][c]) / (2h)
                Dh = s * (-3 * base[pt][f2][c] + 4 * fh[pt][f2][c] - f1[pt][f2][c]) / h
                (4Dh - D1) / 3
            end
        end
    end
    order3 = filter(αs -> sum(first(αs)) == 3, lay3)
    return map(eachindex(pts)) do pt
        map(order3) do αs
            map(αs) do α
                below(i) = pos2[Base.setindex(α, α[i] - 1, i)]
                cost(i) = iszero(hs[pt][i]) ? big(Inf) : abs(wants[pt][below(i)[1]][below(i)[2]]) / hs[pt][i]
                i = argmin(cost, [i for i in 1:N if α[i] > 0])
                fi, ci = below(i)
                deriv[i](pt, fi, ci)
            end
        end
    end
end

# ---------------------------------------------------------------------------------------------
# The comparison of one functional and spin case
# ---------------------------------------------------------------------------------------------

# The largest value of each measured quantity, with the point and field where it occurs.
mutable struct Worst
    value::Float64
    point::Any
    field::Symbol
end
Worst() = Worst(0.0, nothing, :none)
function record!(w::Worst, v, p, field)
    (isnan(v) || v > w.value) && ((w.value, w.point, w.field) = (isnan(v) ? Inf : v, p, field))
    return w
end

const RESULTS = Dict{Tuple{String, Symbol, Symbol}, Worst}()
result(name, spin, item) = get!(Worst, RESULTS, (name, spin, item))

# Ten points spread evenly over a population, in its order.
subgrid(pop) = isempty(pop) ? pop : pop[unique(round.(Int, range(1, length(pop); length = 10)))]

function compare_float64(name, ids, f, fb, spin, above, names)
    fam = family(f)
    lx = libxc_fields(ids, fam, spin, above)
    wants = [bigfields(fb, 3, p) for p in above]
    rich = richardson_order3(ids, fam, spin, above, wants)
    for (i, p) in pairs(above)
        jl2 = fields_at(f, 2, p)
        jl3 = fields_at(f, 3, p)
        want = wants[i]
        if isnothing(jl2) || isnothing(jl3)
            record!(result(name, spin, :rejected), 1.0 + result(name, spin, :rejected).value, p, :none)
            continue
        end
        length(jl2) == length(lx[i]) || error("$name $spin: $(length(jl2)) fields at order 2, Libxc gives $(length(lx[i]))")
        for k in eachindex(jl2)
            length(jl2[k]) == length(lx[i][k]) || error("$name $spin $(names[k]): layout lengths differ from Libxc")
            record!(result(name, spin, :libxc), libxc_ratio(jl2[k], lx[i][k], want[k]), p, names[k])
            record!(result(name, spin, :conditioning), conditioning_ratio(jl2[k], lx[i][k], want[k]), p, names[k])
            record!(result(name, spin, :libxc_vs_big), field_relerr(lx[i][k], want[k]), p, names[k])
            record!(result(name, spin, :jl_vs_big), field_relerr(jl2[k], want[k]), p, names[k])
        end
        n2 = length(jl2)
        for k in (n2 + 1):length(jl3)
            record!(result(name, spin, :order3), field_relerr(jl3[k], want[k]), p, names[k])
            record!(result(name, spin, :richardson), richardson_relerr(rich[i][k - n2], want[k]), p, names[k])
        end
    end
    for p in subgrid(above)
        got = fields_at(f, 4, p)
        want = bigfields(fb, 4, p)
        for k in eachindex(want)
            field_order(names[k]) == 4 && record!(result(name, spin, :order4), field_relerr(got[k], want[k]), p, names[k])
        end
    end
    return
end

# Below points and known differences: the largest relative difference of the outputs at orders 0
# to 2 from Libxc and from BigFloat, per the zero rule. Printed, not gated.
function compare_ungated(name, ids, f, fb, spin, pop, item, names)
    isempty(pop) && return
    lx = libxc_fields(ids, family(f), spin, pop)
    for (i, p) in pairs(pop)
        got = fields_at(f, 2, p)
        isnothing(got) && continue
        want = bigfields(fb, 2, p)
        for k in eachindex(got)
            record!(result(name, spin, Symbol(item, :_libxc)), field_relerr(got[k], lx[i][k]), p, names[k])
            record!(result(name, spin, Symbol(item, :_big)), field_relerr(got[k], want[k]), p, names[k])
        end
    end
    return
end

# Float32: the accuracy items on the points that, rounded to Float32, are above both the Float64
# and the Float32 thresholds (the Float32 ε_ζ is eps(Float32)) and straddle neither, and the
# finiteness item on every grid point. The oracle is the exact function of the Float32 functional
# (`exact_big(f32)`, its parameters rounded to Float32), so the items measure the evaluation error;
# κ̂ comes from the Float64 functional and its exact function at the same point. The difference of
# the two exact functions (the rounding of the parameters) is recorded per order, not gated.
function compare_float32(name, f64, f32, fb, spin, pts, names)
    t64, t32 = thresholds_of(f64), thresholds_of(f32)
    fb32 = exact_big(f32)
    isabove(q) = is_above(q, t64) && is_above(q, t32) && !zeta_straddles(q, t64) && !zeta_straddles(q, t32)
    above32 = 0
    for p in pts
        q = to_float32(p)
        ab = isabove(q)
        for (K, orders) in ((2, 0:2), (3, 3:3), (4, 4:4))
            got = fields_at(f32, K, q)
            if isnothing(got)
                ab && K == 2 && record!(result(name, spin, :rejected32), 1.0 + result(name, spin, :rejected32).value, q, :none)
                continue
            end
            # The exact values are needed only where an output is not finite, or at an above point.
            want32 = all(c -> all(isfinite, c), got) ? nothing : bigfields(fb32, K, q)
            for k in eachindex(got), i in eachindex(got[k])
                j = field_order(names[k])
                (j in orders && !isfinite(got[k][i])) || continue
                g, w = got[k][i], want32[k][i]
                item = Symbol(:nonfinite32_order, j)
                isinf(g) && abs(w) > floatmax(Float32) && sign(g) == sign(w) && continue
                record!(result(name, spin, item), 1.0 + result(name, spin, item).value, q, names[k])
            end
            ab && K <= 3 || continue
            want32 = something(want32, bigfields(fb32, K, q))
            K == 2 && (above32 += 1)
            want64 = bigfields(fb, K, q)
            ref = fields_at(f64, K, q)
            for k in eachindex(got)
                j = field_order(names[k])
                j in orders || continue
                record!(result(name, spin, Symbol(:float32_order, j)), float32_ratio(got[k], ref[k], want32[k], want64[k]), q, names[k])
                record!(result(name, spin, Symbol(:param32_order, j)), field_relerr(want32[k], want64[k]), q, names[k])
            end
        end
    end
    for p in subgrid(filter(p -> isabove(to_float32(p)), pts))
        q = to_float32(p)
        got = fields_at(f32, 4, q)
        isnothing(got) && continue
        want32, want64, ref = bigfields(fb32, 4, q), bigfields(fb, 4, q), fields_at(f64, 4, q)
        for k in eachindex(got)
            field_order(names[k]) == 4 || continue
            record!(result(name, spin, :float32_order4), float32_ratio(got[k], ref[k], want32[k], want64[k]), q, names[k])
            record!(result(name, spin, :param32_order4), field_relerr(want32[k], want64[k]), q, names[k])
        end
    end
    return above32
end

# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------

const COUNTS = Dict{Tuple{String, Symbol}, NamedTuple}()
const KNOWN = Dict{Tuple{String, Symbol}, Vector{Any}}()

for (name, build, ids) in FUNCTIONALS, spin in SPINS
    f64, f32 = build(Float64), build(Float32)
    fb = exact_big(f64)
    t64 = thresholds_of(f64)
    pts = grid_points(family(f64), spin)
    known = filter(p -> zeta_straddles(p, t64), pts)
    above = filter(p -> is_above(p, t64) && !zeta_straddles(p, t64), pts)
    below = filter(p -> !is_above(p, t64) && !zeta_straddles(p, t64), pts)
    names = collect(keys(xc_derivatives(f64, Val(4), first(above)...)))
    t = @elapsed begin
        compare_float64(name, ids, f64, fb, spin, above, names)
        compare_ungated(name, ids, f64, fb, spin, below, :below, names)
        compare_ungated(name, ids, f64, fb, spin, known, :known, names)
        above32 = compare_float32(name, f64, f32, fb, spin, pts, names)
    end
    KNOWN[(name, spin)] = known
    COUNTS[(name, spin)] = (; grid = length(pts), above = length(above), below = length(below), known = length(known), above32)
    @printf("%-13s %-10s grid %6d above %6d below %6d known %4d above32 %6d  %.1f s\n", name, spin, length(pts), length(above), length(below), length(known), above32, t)
    flush(stdout)
end

println("\nMeasured maxima (item, functional, spin: value at field, point)")
for key in sort(collect(keys(RESULTS)); by = k -> (string(k[3]), k[1], string(k[2])))
    w = RESULTS[key]
    @printf("%-22s %-13s %-10s %.4g  %s  %s\n", key[3], key[1], key[2], w.value, w.field, string(w.point))
end
println("\nKnown differences (polarized points within one rounding of 1 ± ζ = ε_ζ)")
for ((name, spin), pts) in sort(collect(KNOWN); by = first)
    isempty(pts) && continue
    @printf("%-13s %-10s %d points: %s\n", name, spin, length(pts), join(string.(pts), ", "))
end
flush(stdout)

value(name, spin, item) = haskey(RESULTS, (name, spin, item)) ? RESULTS[(name, spin, item)].value : 0.0

@testset "PureXC against Libxc 7.0.0 and BigFloat" begin
    for (name, _, _) in FUNCTIONALS, spin in SPINS
        @testset "$name $spin" begin
            v(item) = value(name, spin, item)
            # No above point is a caller error.
            @test v(:rejected) == 0
            @test v(:rejected32) == 0
            # The known differences are exactly the rows ζ = ±(1 − eps) of the grid.
            known = KNOWN[(name, spin)]
            @test length(known) == KNOWN_COUNT[(name, spin)]
            @test all(p -> abs(1 - abs((big(p[1][1]) - big(p[1][2])) / (big(p[1][1]) + big(p[1][2])))) < 4 * big(eps(1.0)), known)
            # Float64, orders 0 to 2: (A) against Libxc, (B) against BigFloat.
            @test v(:libxc) <= 1
            @test v(:conditioning) <= C_B[(name, spin)][1]
            # Float64, orders 3 and 4, against BigFloat; Richardson differences of Libxc's order 2.
            @test v(:order3) <= FLOOR
            @test v(:order4) <= FLOOR
            @test v(:richardson) <= RICHARDSON_TOL
            # Float32: accuracy at orders 0 to 4 and finiteness at orders 0 to 4.
            for j in 0:4
                @test v(Symbol(:float32_order, j)) <= C_32[(name, spin)][1]
                @test v(Symbol(:nonfinite32_order, j)) == 0
            end
        end
    end
end
