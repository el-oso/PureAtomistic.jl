# Perdew–Burke–Ernzerhof exchange, from the spec `gga/gga_x_pbe.md`. Equation labels (E1)–(E4)
# are those of the spec.

"""
    PBEExchange{T}(κ, μ, thresholds)

Perdew–Burke–Ernzerhof exchange, Libxc's `gga_x_pbe`, with the parameters `κ` and `μ` of the
enhancement factor in the number type `T` and the thresholds in its value type (`T` itself, or the
coefficient type of a `Taylor` number type, so that a parameter can be a series variable). A
plain `κ` or `μ` given for a `Taylor` number type is a constant series. Built by [`pbe_x`](@ref).
"""
struct PBEExchange{T, V} <: AbstractGGA{T}
    κ::T
    μ::T
    thresholds::Thresholds{V}
    PBEExchange{T}(κ, μ, thresholds) where {T} =
        new{T, _value_type(T)}(_to_number_type(T, κ), _to_number_type(T, μ), thresholds)
end

"""
    pbe_x(T = Float64)

PBE exchange, Libxc's `gga_x_pbe`: κ = 0.804 and μ = 0.2195149727645171, with the default
thresholds, in the number type `T`.
"""
pbe_x(::Type{T} = Float64) where {T} = PBEExchange{T}(
    0.804, 0.2195149727645171, default_thresholds(T, 1.0e-15, 1.0000000000000027e-20, 2.220446049250313e-16)
)

kind(::PBEExchange) = :exchange

_with_number_type(f::PBEExchange, ::Type{S}) where {S} =
    PBEExchange{S}(convert(S, f.κ), convert(S, f.μ), _thresholds_in(S, f.thresholds))

# Rule 2: restricted input is n_α = n_β = n/2, so a channel at or below n_thr leaves nothing; this
# includes rule 1 (n < n_thr).
_cutoff(f::PBEExchange, ρ::Real, σ::Real) = ρ / 2 <= f.thresholds.density
# Rule 1: the total density below n_thr.
_cutoff(f::PBEExchange, ρ::Tuple, σ::Tuple) = ρ[1] + ρ[2] < f.thresholds.density

# x^(−4/3), with the value 1/(x·∛x), as one power series for a `Taylor`.
_powm43(x::Real) = inv(_pow43(x))
_powm43(x::Taylor) = power_series(x, -convert(typeof(value(x)), 4) / 3, _powm43(value(x)))

# s² = σ/(c·n^(8/3)) as (σ·q)·q with q = n^(−4/3)/√c, so that no power of n beyond n^(±4/3) is formed.
function _reduced_gradient2(n, σ, c)
    q = _powm43(n) * inv(sqrt(c))
    return (σ * q) * q
end

# 4(3π²)^(2/3), restricted, and 4(6π²)^(2/3), per channel: s² = σ/(c·n^(8/3)) of (E3) and (E4).
_pbe_x_c3(::Type{R}) where {R} = 4 * cbrt(3 * convert(R, π)^2)^2
_pbe_x_c6(::Type{R}) where {R} = 4 * cbrt(6 * convert(R, π)^2)^2

# n^(4/3)·F_x(s) with s² = σ/(c·n^(8/3)) and F_x = 1 + κ − κ/(1 + x), x = μs²/κ, of (E2). Two forms,
# selected by the value of x, keep the series coefficients free of differences of nearly equal
# terms:
# - x ≤ 1: n^(4/3)·(1 + μs²/(1 + x)), which has no difference of nearly equal terms in any
#   variable, κ included (1 + κ − κ/(1 + x) cancels to relative order x² in its κ-derivative).
# - x > 1: (1 + κ)·n^(4/3) − P/(1 + w), with w = 1/x = (κc/μ)·n^(4/3)·n^(4/3)/σ and
#   P = n^(4/3)·κ·w = (κ²c/μ)·n²·n²/σ. P is a polynomial of degree 4 in n, so its coefficients
#   above degree 4 are exact zeros; in the product n^(4/3)·κ/(1 + x) they would be differences of
#   terms larger than the result by the factor x.
function _pbe_x_term(f::PBEExchange, n, σ, c)
    κ, μ = f.κ, f.μ
    n43 = _pow43(n)
    μs2 = μ * _reduced_gradient2(n, σ, c)
    x = μs2 / κ
    small = n43 * (1 + μs2 / (1 + x))
    r = inv(σ)
    w = (κ * c / μ) * ((n43 * r) * n43)
    P = (κ^2 * c / μ) * ((n * n * r) * (n * n))
    large = (1 + κ) * n43 - P / (1 + w)
    return ifelse(value(x) <= 1, small, large)
end

# (E3): e(n, σ) = −(3/4)(3/π)^(1/3) n^(4/3) F_x(s), with σ raised to the floor σ_thr² (rule 3). The
# density is raised to n_thr only to keep fast_cbrt in its domain; every point that the raise moves
# is cut off.
function _energy_density(f::PBEExchange, ρ, σ)
    R = typeof(value(ρ))
    n = _raise(ρ, f.thresholds.density)
    g = _raise(σ, f.thresholds.sigma_floor)
    return -(convert(R, 3) / 4 * cbrt(3 / convert(R, π))) * _pbe_x_term(f, n, g, _pbe_x_c3(R))
end

# (E4): e = −(3/4)(6/π)^(1/3) Σ_σ n_σ^(4/3) F_x(s_σ), channel by channel; σ_αβ does not enter.
function _energy_density(f::PBEExchange, ρ::Tuple, σ::Tuple)
    R = typeof(value(ρ[1]))
    c = _pbe_x_c6(R)
    channels = _pbe_x_channel(f, ρ[1], σ[1], c) + _pbe_x_channel(f, ρ[2], σ[3], c)
    return -(convert(R, 3) / 4 * cbrt(6 / convert(R, π))) * channels
end

# n_σ^(4/3) F_x(s_σ), or zero for n_σ ≤ n_thr (rule 2), with σ_σσ raised to the floor (rule 3). The
# channel is raised to n_thr first so that fast_cbrt stays in its domain where the term is discarded.
function _pbe_x_channel(f::PBEExchange, ρσ, σσ, c)
    thr = f.thresholds
    t = _pbe_x_term(f, _raise(ρσ, thr.density), _raise(σσ, thr.sigma_floor), c)
    return ifelse(value(ρσ) <= thr.density, zero(t), t)
end

# Each channel is seeded at its own scales: the density at max(|n_σ|, n_thr, m) and σ_σσ at
# max(|σ_σσ|, r, m) with r = 4(6π²)^(2/3)·n_σr^(8/3) at the raised channel n_σr = max(n_σ, n_thr),
# the σ at which that channel's s is 1. e does not depend on σ_αβ, so its scale only needs to be
# normal.
function _seed_scales(f::PBEExchange, ::Val{K}, ρ::NTuple{2, R}, σ::NTuple{3, R}) where {K, R}
    m = _seed_minimum(R)
    t = convert(R, f.thresholds.density)
    c = _pbe_x_c6(R)
    r(x) = c * fast_cbrt(max(x, t))^8
    return (
        _density_seed(f, ρ[1], m), _density_seed(f, ρ[2], m),
        max(abs(σ[1]), r(ρ[1]), m), max(abs(σ[2]), m), max(abs(σ[3]), r(ρ[2]), m),
    )
end

# Measured: in pure Float32 every output of orders 0 to 3 stays within its bound up to n_max and
# s² = 2^66, the limit at order 2 (orders 0 to 2 are always evaluated in Float32).
# From order 4 the factors j/s of `derivatives` overflow for a σ seeded at floatmin(Float32),
# the scale of σ_σσ in a channel near n_thr, where r(n_σ) is below floatmin; such orders are
# evaluated in Float64.
_float32_max_order(::Type{<:PBEExchange}) = 3

"""
    _pbe_x_s2_max(R, Val(K))

s²_max(R, K), the largest reduced gradient s_σ² of a channel (restricted: s²) that `pbe_x` accepts
at order `K` in the type `R`, a power of two: `2^_PBE_X_S2_EXPONENT_FLOAT32[K + 1]` in Float32,
`2^_PBE_X_S2_EXPONENT_FLOAT64[K + 1]` in Float64 and the other types. Each is measured as the
largest power of two at which every output at order `K` is within its bound against BigFloat over
the splits of `_lda_density_max` up to n_max. Above it, series coefficients of size e/s² underflow
at the smallest densities. Orders above 10 are not measured and take the order-10 value.
"""
_pbe_x_s2_max(::Type{R}, ::Val{K}) where {R, K} = ldexp(one(R), _PBE_X_S2_EXPONENT_FLOAT64[min(K, 10) + 1])
_pbe_x_s2_max(::Type{Float32}, ::Val{K}) where {K} = ldexp(1.0f0, _PBE_X_S2_EXPONENT_FLOAT32[min(K, 10) + 1])

const _PBE_X_S2_EXPONENT_FLOAT32 = (127, 66, 66, 66, 127, 127, 127, 127, 127, 127, 127)
const _PBE_X_S2_EXPONENT_FLOAT64 = (1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 1023, 483, 482)

function _check_gga_domain(f::PBEExchange, ::Val{K}, ρ::Real, σ::Real) where {K}
    R = _domain_type(f, (ρ, σ))
    n = convert(R, ρ)
    _check_gga_density_max(Val(K), n)
    n / 2 > f.thresholds.density || return nothing
    _check_reduced_gradient(:s², _pbe_x_s2(n, convert(R, σ), _pbe_x_c3(R)), _pbe_x_s2_max(R, Val(K)), Val(K))
    return nothing
end

function _check_gga_domain(f::PBEExchange, ::Val{K}, ρ::NTuple{2, Real}, σ::NTuple{3, Real}) where {K}
    R = _domain_type(f, (ρ..., σ...))
    a, b = convert(R, ρ[1]), convert(R, ρ[2])
    _check_gga_density_max(Val(K), a + b)
    c, m, t = _pbe_x_c6(R), _pbe_x_s2_max(R, Val(K)), f.thresholds.density
    a > t && _check_reduced_gradient(:s²_α, _pbe_x_s2(a, convert(R, σ[1]), c), m, Val(K))
    b > t && _check_reduced_gradient(:s²_β, _pbe_x_s2(b, convert(R, σ[3]), c), m, Val(K))
    return nothing
end

# s² = σ/(c·n^(8/3)) as (√σ/(√c·n^(4/3)))², for a density above n_thr.
function _pbe_x_s2(n, σ, c)
    r = sqrt(σ) / (sqrt(c) * _pow43(n))
    return r * r
end
