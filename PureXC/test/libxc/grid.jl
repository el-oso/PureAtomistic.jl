# The oracle grid of the comparison with Libxc, shared by the oracle items and the speed bench.
#
# - ρ: the total density, log-spaced over [1e-10, 1e3] (131 points, 10 per decade).
# - s: the reduced gradient, 0 and 41 points log-spaced over [1e-3, 10] (10 per decade).
# - ζ: the spin polarization, from the set `GRID_ZETA`.
#
# A restricted LDA point is ρ; a restricted GGA point is (ρ, σ) with σ = 4(3π²)^(2/3)·ρ^(8/3)·s².
# A polarized point has the channels ρα = ρ(1 + ζ)/2 and ρβ = ρ(1 − ζ)/2; a polarized GGA point has
# σ_σσ = 4(6π²)^(2/3)·ρσ^(8/3)·s² in each channel (the same s in both) and parallel gradients,
# σαβ = √σαα·√σββ. Each σ is the one at which its channel's reduced gradient equals s.
#
# A point is "above" for a set of thresholds (density n_thr, σ floor σ_thr², spin threshold ε_ζ)
# when n ≥ n_thr, every channel n_σ > n_thr (restricted: n/2), every σ_σσ (restricted: σ) is
# above σ_thr², and 1 ± ζ ≥ ε_ζ for the ζ = (ρα − ρβ)/(ρα + ρβ) rounded from the channels;
# otherwise it is "below". For a sum of functionals a point is above when it is above for every
# term.

using PureXC: PureXC, Thresholds

const GRID_RHO = exp10.(range(-10, 3; length = 131))
const GRID_S = [0.0; exp10.(range(-3, 1; length = 41))]
const GRID_ZETA = (-1.0, -(1 - eps(1.0)), -0.6, 0.0, 0.25, 0.95, 1 - eps(1.0), 1.0)

_sigma_restricted(n, s) = 4 * cbrt(3π^2)^2 * n^(8 / 3) * s^2
_sigma_channel(nσ, s) = 4 * cbrt(6π^2)^2 * nσ^(8 / 3) * s^2

"""
    grid_points(family, spin)

The grid points of a functional family (`:lda` or `:gga`) and spin case (`:restricted` or
`:polarized`) in Float64. Restricted LDA points are densities; restricted GGA points are `(ρ, σ)`;
polarized LDA points are `((ρα, ρβ),)`; polarized GGA points are `((ρα, ρβ), (σαα, σαβ, σββ))`.
"""
function grid_points(family::Symbol, spin::Symbol)
    family in (:lda, :gga) || throw(ArgumentError("grid: family must be :lda or :gga, got $family"))
    spin in (:restricted, :polarized) || throw(ArgumentError("grid: spin must be :restricted or :polarized, got $spin"))
    if spin === :restricted
        family === :lda && return [(n,) for n in GRID_RHO]
        return [(n, _sigma_restricted(n, s)) for n in GRID_RHO for s in GRID_S]
    end
    channels(n, ζ) = (n * (1 + ζ) / 2, n * (1 - ζ) / 2)
    family === :lda && return [(channels(n, ζ),) for n in GRID_RHO for ζ in GRID_ZETA]
    pts = Tuple{NTuple{2, Float64}, NTuple{3, Float64}}[]
    for n in GRID_RHO, ζ in GRID_ZETA, s in GRID_S
        a, b = channels(n, ζ)
        σa, σb = _sigma_channel(a, s), _sigma_channel(b, s)
        push!(pts, ((a, b), (σa, sqrt(σa) * sqrt(σb), σb)))
    end
    return pts
end

"""
    is_above(point, thresholds)

Whether a point of `grid_points` is above the `PureXC.Thresholds` `thresholds`, or above every
element of a tuple of them.
"""
is_above(p::Tuple, t::Tuple) = all(x -> is_above(p, x), t)
is_above(p::Tuple{Real}, t::Thresholds) = p[1] >= t.density && p[1] / 2 > t.density
is_above(p::Tuple{Real, Real}, t::Thresholds) = is_above((p[1],), t) && p[2] > t.sigma_floor
function is_above(p::Tuple{NTuple{2, Real}}, t::Thresholds)
    a, b = p[1]
    ζ = (a - b) / (a + b)
    return a + b >= t.density && a > t.density && b > t.density && 1 + ζ >= t.zeta && 1 - ζ >= t.zeta
end
is_above(p::Tuple{NTuple{2, Real}, NTuple{3, Real}}, t::Thresholds) =
    is_above((p[1],), t) && p[2][1] > t.sigma_floor && p[2][3] > t.sigma_floor

"""
    thresholds_of(f)

The `PureXC.Thresholds` of a functional, or a tuple of those of the terms of an `XC`.
"""
thresholds_of(f::PureXC.XC) = map(t -> t[2].thresholds, f.terms)
thresholds_of(f::PureXC.AbstractFunctional) = f.thresholds
"""
    zeta_straddles(point, thresholds)

Whether a polarized point lies within one rounding of 1 ± ζ = ε_ζ: the smaller of 1 + ζ and
1 − ζ, for ζ = (ρα − ρβ)/(ρα + ρβ) either rounded from the channels in their type or exact, is
within eps/2 (one rounding of ζ near ±1) of ε_ζ. The rounded and the exact ζ may then fall on
different sides of ε_ζ, and an evaluation that compares with ≤ and one that compares with < may
decide differently at the threshold itself, so one keeps the factor (1 ± ζ)^p and another
replaces it by the constant ε_ζ^p; their derivatives differ by the series part of that factor.
For a tuple of thresholds, whether it holds for any of them. Restricted points never straddle.
"""
zeta_straddles(p::Tuple, t::Tuple) = any(x -> zeta_straddles(p, x), t)
zeta_straddles(p::Tuple{Vararg{Real}}, t::Thresholds) = false
function zeta_straddles(p::Tuple{NTuple{2, Real}, Vararg}, t::Thresholds)
    a, b = p[1]
    (iszero(a) || iszero(b)) && return false
    ζ = (a - b) / (a + b)
    ζe = (big(a) - big(b)) / (big(a) + big(b))
    ε = big(t.zeta)
    u = big(eps(typeof(ζ))) / 2
    return abs(1 - abs(big(ζ)) - ε) <= u || abs(1 - abs(ζe) - ε) <= u
end

"""
    to_float32(point)

The grid point with every input rounded to Float32.
"""
to_float32(p::Tuple) = map(x -> x isa Tuple ? Float32.(x) : Float32(x), p)
