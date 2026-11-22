# Perdew–Burke–Ernzerhof correlation, from the spec `gga/gga_c_pbe.md`. Equation labels (E1)–(E5)
# are those of the spec.

"""
    PBECorrelation{T}(β, γ, B, pw, thresholds)

Perdew–Burke–Ernzerhof correlation, Libxc's `gga_c_pbe`, in the number type `T`: the gradient
coefficient `β`, `γ`, the multiplier `B` of the A·t² term of the numerator and the A²t⁴ term of the
denominator of the bracket of H, and the uniform-gas correlation `pw`
(a [`PW92Correlation`](@ref)). The ε_ζ of rule 4, in f(ζ) and in φ(ζ), is `thresholds.zeta`; the
thresholds of `pw` are not used. Built by [`pbe_c`](@ref).
"""
struct PBECorrelation{T} <: AbstractGGA{T}
    β::T
    γ::T
    B::T
    pw::PW92Correlation{T}
    thresholds::Thresholds{T}
end

# γ = (1 − ln 2)/π² in 256-bit BigFloat, and rounded to Float64 (0.031090690869654894, the double
# nearest the exact value) and Float32 once at load.
const _PBE_GAMMA_EXACT = (1 - log(big(2))) / big(π)^2
const _PBE_GAMMA_EXACT_FLOAT64 = Float64(_PBE_GAMMA_EXACT)
const _PBE_GAMMA_EXACT_FLOAT32 = Float32(_PBE_GAMMA_EXACT)
_pbe_gamma(::Type{T}) where {T} = convert(T, _PBE_GAMMA_EXACT)
_pbe_gamma(::Type{Float64}) = _PBE_GAMMA_EXACT_FLOAT64
_pbe_gamma(::Type{Float32}) = _PBE_GAMMA_EXACT_FLOAT32

"""
    pbe_c(T = Float64)

PBE correlation, Libxc's `gga_c_pbe`: β = 0.06672455060314922, γ = (1 − ln 2)/π² rounded to `T`
and B = 1, on [`lda_c_pw_mod`](@ref), with the default thresholds, in the number type `T`.
"""
pbe_c(::Type{T} = Float64) where {T} = PBECorrelation{T}(
    convert(T, 0.06672455060314922), _pbe_gamma(T), one(T), lda_c_pw_mod(T),
    default_thresholds(T, 1.0e-12, 1.0000000000000021e-16, 2.220446049250313e-16),
)

kind(::PBECorrelation) = :correlation

_with_number_type(f::PBECorrelation, ::Type{S}) where {S} = PBECorrelation{S}(
    convert(S, f.β), convert(S, f.γ), convert(S, f.B), _with_number_type(f.pw, S), _thresholds_in(S, f.thresholds)
)

# Measured: in pure Float32 orders 0 to 3 stay within their bounds up to n_max and t² = 2^19, the
# limit at order 2 (orders 0 to 2 are always evaluated in Float32); at order 4 pure Float32 fails
# from t² = 2^14, so orders from 4 are evaluated in Float64.
_float32_max_order(::Type{<:PBECorrelation}) = 3

# Rule 1: the total density below n_thr (strict), restricted and polarized.
_cutoff(f::PBECorrelation, ρ::Real, σ::Real) = ρ < f.thresholds.density
_cutoff(f::PBECorrelation, ρ::Tuple, σ::Tuple) = ρ[1] + ρ[2] < f.thresholds.density

# x^(−7/6), with the value 1/(x·√∛x), as one power series for a `Taylor`.
_powm76(x::Real) = inv(x * fast_sqrt(fast_cbrt(x)))
_powm76(x::Taylor) = power_series(x, -convert(typeof(value(x)), 7) / 6, _powm76(value(x)))

# x^(2/3), with the value ∛x², as one power series for a `Taylor`.
_pow23(x::Real) = fast_cbrt(x)^2
_pow23(x::Taylor) = power_series(x, convert(typeof(value(x)), 2) / 3, _pow23(value(x)))

# π/(16(3π²)^(1/3)): t² = c·σ/(φ² n^(7/3)) of (E1).
_pbe_c_t2_coefficient(::Type{R}) where {R} = convert(R, π) / (16 * cbrt(3 * convert(R, π)^2))

# (1 + x)^(2/3) for φ, or the constant ε_ζ^(2/3) with no series part where 1 + x ≤ ε_ζ (rule 4).
# The argument is raised to ε_ζ first so that fast_cbrt stays in its domain where the result is
# discarded.
function _pbe_pow23(y, εζ)
    x = _raise(y, εζ)
    t = _pow23(x)
    return ifelse(value(y) <= εζ, zero(t) + _pow23(εζ), t)
end

# φ(ζ) = [(1 + ζ)^(2/3) + (1 − ζ)^(2/3)]/2 of (E1), with rule 4.
function _pbe_phi(ζ, εζ)
    return (_pbe_pow23(1 + ζ, εζ) + _pbe_pow23(1 - ζ, εζ)) / 2
end

# H of (E2) at the density n, the spin factor φ, the total σ and ε_c, with A of (E3), as
# γφ³·log(1 + X). t² = (σ·q)·q/φ² with q = n^(−7/6)·√c, so that no power of n beyond n^(±7/6) is
# formed. With y = A·t² and the multiplier B of the numerator's A·t² and the denominator's A²t⁴,
#     X = (β/γ)·t²·(1 + By)/(1 + y + By²) = E·(1 − 1/(1 + y(1 + By))),   E = exp(−ε_c/(γφ³)) − 1,
# since y(1 + By) = (1 + y + By²) − 1. Each form below is chosen by the values of y and X so that
# no series coefficient is a difference of nearly equal terms:
# - X is formed as (β/γ)·t²/(1 + w), w = y(1 − B + By)/(1 + By) (y²/(1 + y) at B = 1), for y ≤ 1
#   and as the second form above: at large y, t² times the ratio cancels to relative order 1/y²,
#   and at small y, 1 − 1/(1 + y + By²) cancels in its second-order coefficients.
# - log(1 + X) is fast_log1p(X) for X ≤ 1. Above, it is log X + log(1 + 1/X) with log X a sum of
#   the logarithms of its factors, log(β/γ) + log σ − (7/3)·log n + log c − 2·log φ − log(1 + w)
#   for y ≤ 1 and log E + log(1 − 1/(1 + y(1 + By))) above. The logarithm of a product separates
#   its variables, so its mixed coefficients in n and σ come from the small corrections alone;
#   composed with the series of X they would be differences of terms larger by the factor 1/y² or
#   X. The value is fast_log1p of the value of X.
# - For z > 1, y > 1 and X > 1, log(1 + X) is instead z + log(1 − (1 − e^(−z))·v), since
#   1 + X = e^z·(1 − (1 − e^(−z))·v), whose log1p argument lies in (−1/(2 + B), 0]. The split form
#   log X + log(1 + 1/X) is not used there: at high order its two terms are much larger than their
#   sum (measured at full polarization, n = 2^21: by about 25 times at order 9 and 40 times at
#   order 10), so their roundings remain in the sum.
# - For z = −ε_c/(γφ³) > 1, 1/E = e^(−z)/(1 − e^(−z)) and log E = z + log(1 − e^(−z)): the series of
#   1/E or log E composed with that of E would be differences of terms of the size of the powers
#   of its first coefficient.
function _pbe_H(f::PBECorrelation, n, φ, σ, ε)
    R = typeof(value(n))
    φ2 = φ * φ
    φ3 = φ2 * φ
    c = _pbe_c_t2_coefficient(R)
    q = _powm76(n) * sqrt(c)
    t2 = (σ * q) * q / φ2
    βγ = f.β / f.γ
    B = f.B
    z = -ε / (f.γ * φ3)
    E = fast_expm1(z)
    em = fast_exp(-z)
    zlarge = value(z) > 1
    rE = ifelse(zlarge, em / (-fast_expm1(-z)), inv(E))
    logE = ifelse(zlarge, z + fast_log1p(-em), fast_log(E))
    y = (βγ * rE) * t2
    # 1/t² and 1/y from their factors, φ²·n^(7/3)/(c·σ) and (E/(β/γ))·(1/t²): the inverse of the
    # series of t² or y would be a difference of terms larger by the ratio of the coefficients of
    # n^(−7/3) and n^(7/3). 1/σ is formed as (1/σ₀)·(σ₀/σ) with σ₀ the value of σ, the factor 1/σ₀
    # applied to n^(7/3) first: the coefficients of the series 1/σ itself are smaller than 1/σ₀ by
    # the factor 1/t², and underflow at large t².
    qi = _pow76(n) * inv(sqrt(c))
    rt2 = φ2 * (((qi * qi) * inv(value(σ))) * inv(σ / value(σ)))
    iy = E / βγ * rt2
    w = y * ((1 - B + B * y) / (1 + B * y))
    # v = 1/(1 + y(1 + By)) = (1/y)²/(B + (1/y)(1 + 1/y)).
    v = iy * iy / (B + iy * (1 + iy))
    small = βγ * t2 / (1 + w)
    large = E * (1 - v)
    X = ifelse(value(y) <= 1, small, large)
    # 1/X: (1 + w)/(β/γ·t²), or (1/E)·(1 + 1/(y(1 + By))) with 1/(y(1 + By)) = (1/y)²/(B + 1/y).
    rsmall = (1 + w) * rt2 / βγ
    rlarge = rE * (1 + iy * (iy / (B + iy)))
    logsmall = log(βγ * c) + fast_log(σ) - convert(R, 7) / 3 * fast_log(n) - 2 * fast_log(φ) - fast_log1p(w)
    loglarge = logE + fast_log1p(-v)
    split = ifelse(value(y) <= 1, logsmall, loglarge) + fast_log1p(ifelse(value(y) <= 1, rsmall, rlarge))
    whole = z + fast_log1p((em - 1) * v)
    above = ifelse(zlarge & (value(y) > 1), whole, split)
    L = ifelse(value(X) <= 1, fast_log1p(X), _series_with_value(above, fast_log1p(value(X))))
    return f.γ * φ3 * L
end

# x^(7/6), with the value x·√∛x, as one power series for a `Taylor`.
_pow76(x::Real) = x * fast_sqrt(fast_cbrt(x))
_pow76(x::Taylor) = power_series(x, convert(typeof(value(x)), 7) / 6, _pow76(value(x)))

# `x` with its value replaced by `v`, for a `Taylor`; `v` for a number.
_series_with_value(x::Taylor, v) = _with_value(x, v)
_series_with_value(::Real, v) = v

# (E5) with (E1): e = n[ε_c(r_s, 0) + H] at φ = 1, with σ raised to the floor σ_thr² (rule 3). The
# density is raised to n_thr only to keep the fast functions in their domains; every point that
# the raise moves is cut off.
_energy_density(f::PBECorrelation, ρ, σ) = _pbe_c_energy(f, ρ, σ, +)

# n·combine(ε_c, H): `+` for the energy; the terms alone serve to measure the cancellation in ε_c + H.
# `adjust` is applied to the series of ε_c before H is formed: `identity` for the energy; a change of
# its coefficients by one ulp measures the sensitivity of the outputs to the rounding of ε_c.
function _pbe_c_energy(f::PBECorrelation, ρ, σ, combine, adjust = identity)
    n = _raise(ρ, f.thresholds.density)
    g = _raise(σ, f.thresholds.sigma_floor)
    ε = adjust(_pw92_eps(f.pw, _seitz_radius(n)))
    return n * combine(ε, _pbe_H(f, n, one(typeof(value(n))), g, ε))
end

# The t-scale 16φ²k_F n²/π, the σ at which t² = 1, at the density n and the spin factor φ.
_pbe_t_scale(n::R, φ::R) where {R} = 16 * φ^2 * cbrt(3 * convert(R, π)^2) * fast_cbrt(n) * n^2 / convert(R, π)

# (E1) with rules 2 to 5: r_s, ζ and φ are formed from the channels raised to n_thr (rule 2), each
# σ_σσ is raised to the floor (rule 3) and so is the total σ_αα + 2σ_αβ + σ_ββ (rule 5), and the
# total density n_r of the raised channels multiplies ε_c + H, so every derivative is that of
# n_r(ε_c + H) at the raised densities. The value is n(ε_c + H) with the caller's n = n_α + n_β.
# ε_c + H is expanded in its own variables (n, ζ, σ), seeded at the scales max(n_r, m), 1 and the
# t-scale, with m of `_seed_minimum`, and composed with the series of n_r, ζ and the total σ in
# the input variables.
_energy_density(f::PBECorrelation, ρ::Tuple{Taylor, Taylor}, σ::NTuple{3, Taylor}) = _pbe_c_energy(f, ρ, σ, +)

function _pbe_c_energy(f::PBECorrelation, ρ::Tuple{Taylor{N, K}, Taylor{N, K}}, σ::NTuple{3, Taylor{N, K}}, combine, adjust = identity) where {N, K}
    thr = f.thresholds
    a = _raise(ρ[1], thr.density)
    b = _raise(ρ[2], thr.density)
    nr = a + b
    ζ = (a - b) / nr
    gt = _raise(_raise(σ[1], thr.sigma_floor) + 2 * σ[2] + _raise(σ[3], thr.sigma_floor), thr.sigma_floor)
    R = typeof(value(nr))
    εζ = convert(R, thr.zeta)
    sn = max(value(nr), _seed_minimum(R))
    st = _pbe_t_scale(value(nr), value(_pbe_phi(value(ζ), εζ)))
    n = variable(Val(3), Val(K), Val(1), sn, sn)
    z = variable(Val(3), Val(K), Val(2), value(ζ))
    g = variable(Val(3), Val(K), Val(3), value(gt), st)
    ε = adjust(_pw92_eps(f.pw, _seitz_radius(n), z, εζ))
    w = combine(ε, _pbe_H(f, n, _pbe_phi(z, εζ), g, ε))
    e = substitute(n * w, (nr / sn, ζ, gt / st))
    return _with_value(e, (value(ρ[1]) + value(ρ[2])) * value(w))
end

# ε_c + H is a function of the total density, ζ and the total σ, smooth in each channel on the
# scale of the total: both densities are seeded at max(n_r, m), with n_r the total of the raised
# channels, and every σ component at the t-scale 16φ²k_F n_r²/π at the raised channels.
function _seed_scales(f::PBECorrelation, ::Val{K}, ρ::R, σ::R) where {K, R}
    t = convert(R, f.thresholds.density)
    n = max(ρ, t)
    return (max(n, _seed_minimum(R)), _pbe_t_scale(n, one(R)))
end

function _seed_scales(f::PBECorrelation, ::Val{K}, ρ::NTuple{2, R}, σ::NTuple{3, R}) where {K, R}
    t = convert(R, f.thresholds.density)
    a, b = max(ρ[1], t), max(ρ[2], t)
    n = a + b
    s = max(n, _seed_minimum(R))
    st = _pbe_t_scale(n, _pbe_phi((a - b) / n, convert(R, f.thresholds.zeta)))
    return (s, s, st, st, st)
end

"""
    _pbe_c_t2_max(R, Val(K))

t²_max(R, K), the largest reduced gradient t² that `pbe_c` accepts at order `K` in the type `R`, a
power of two: `2^_PBE_C_T2_EXPONENT_FLOAT32[K + 1]` in Float32, `2^_PBE_C_T2_EXPONENT_FLOAT64[K + 1]`
in Float64 and the other types. Each is measured as the largest power of two at which every output
at order `K` is within its bound against BigFloat over the splits of `_lda_density_max` up to
n_max, at every power of two up to the limit (step 1 in the exponent), restricted and polarized at
orders 0 to 8 (at orders 7 and 8 over 12 of the 21 splits, including the ratios near ε_ζ), with the
errors and bounds of the tests: per entry against the largest normalized
Taylor coefficient of its order, plus for pbe_c the change of the outputs from rounding ε_c by one
ulp. At orders 9 and 10 only restricted input is measured (Float32 2^14, 2^13; Float64 no failure up
to 2^140, the end of the scan), and the limit is the smaller of that and the order-8 limit. Float32
from order 4 is evaluated in Float64. Above the limit outputs lose their accuracy: in Float64 the
total σ overflows near n_max, and in Float32 an entry beyond floatmax comes out with the wrong sign
or not finite. Orders above 10 are not measured and take the order-10 value.
"""
_pbe_c_t2_max(::Type{R}, ::Val{K}) where {R, K} = ldexp(one(R), _PBE_C_T2_EXPONENT_FLOAT64[min(K, 10) + 1])
_pbe_c_t2_max(::Type{Float32}, ::Val{K}) where {K} = ldexp(1.0f0, _PBE_C_T2_EXPONENT_FLOAT32[min(K, 10) + 1])

const _PBE_C_T2_EXPONENT_FLOAT32 = (19, 19, 19, 19, 127, 127, 16, 16, 16, 14, 13)
const _PBE_C_T2_EXPONENT_FLOAT64 = (131, 131, 131, 131, 131, 131, 131, 131, 131, 131, 131)

function _check_gga_domain(f::PBECorrelation, ::Val{K}, ρ::Real, σ::Real) where {K}
    R = _domain_type(f, (ρ, σ))
    n = convert(R, ρ)
    _check_gga_density_max(Val(K), n)
    n < f.thresholds.density && return nothing
    g = max(convert(R, σ), convert(R, f.thresholds.sigma_floor))
    _check_reduced_gradient(:t², _pbe_c_t2(n, one(R), g), _pbe_c_t2_max(R, Val(K)), Val(K))
    return nothing
end

function _check_gga_domain(f::PBECorrelation, ::Val{K}, ρ::NTuple{2, Real}, σ::NTuple{3, Real}) where {K}
    R = _domain_type(f, (ρ..., σ...))
    thr = _thresholds_in(R, f.thresholds)
    a, b = convert(R, ρ[1]), convert(R, ρ[2])
    _check_gga_density_max(Val(K), a + b)
    a + b < thr.density && return nothing
    ar, br = max(a, thr.density), max(b, thr.density)
    n = ar + br
    φ = _pbe_phi((ar - br) / n, thr.zeta)
    # t² of each component, so that the total σ is not formed: it can overflow where t² fits.
    fl = thr.sigma_floor
    s = convert(R, σ[2])
    t2 = _pbe_c_t2(n, φ, max(convert(R, σ[1]), fl)) + 2 * copysign(_pbe_c_t2(n, φ, abs(s)), s) +
        _pbe_c_t2(n, φ, max(convert(R, σ[3]), fl))
    _check_reduced_gradient(:t², max(t2, _pbe_c_t2(n, φ, fl)), _pbe_c_t2_max(R, Val(K)), Val(K))
    return nothing
end

# t² = σ/(16φ²k_F n²/π) as (√σ/√(16φ²(3π²)^(1/3) n^(7/3)/π))², for a density of at least n_thr.
function _pbe_c_t2(n::R, φ::R, σ::R) where {R}
    r = sqrt(σ) / sqrt(_pbe_t_scale(n, φ))
    return r * r
end
