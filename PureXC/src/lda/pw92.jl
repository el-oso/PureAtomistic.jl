# Perdew–Wang 1992 correlation of the uniform electron gas, from the specs `lda/lda_c_pw.md` and
# `lda/lda_c_pw_mod.md`. Equation labels (E1)–(E5) are those of `lda/lda_c_pw.md`.

"""
    PW92Correlation{T}(params, fz20, thresholds)

Perdew–Wang 1992 correlation in the number type `T`. `params` holds the three parameter sets of
the fit G, for ε_c(r_s, 0), ε_c(r_s, 1) and −α_c(r_s) in that order, each as
`(A, α₁, β₁, β₂, β₃, β₄)` with the exponent p = 1. `fz20` is f''(0) and `thresholds` gives the
density threshold and the spin-polarization threshold. Built by [`lda_c_pw`](@ref) and
[`lda_c_pw_mod`](@ref).
"""
struct PW92Correlation{T} <: AbstractLDA{T}
    params::NTuple{3, NTuple{6, T}}
    fz20::T
    thresholds::Thresholds{T}
end

kind(::PW92Correlation) = :correlation

_with_number_type(pw::PW92Correlation, ::Type{S}) where {S} = PW92Correlation{S}(
    map(p -> map(x -> convert(S, x), p), pw.params), convert(S, pw.fz20), _thresholds_in(S, pw.thresholds)
)

# Measured: in pure Float32 order 3 already fails near full polarization, where 1 − ζ is just above
# ε_ζ: at (2^94, 2^94·1e-7) the exact v3rho3_βββ is −6.2e-46 and pure Float32 gives −Inf. Orders 0
# to 2 stay within their bounds over the splits of `_lda_density_max`.
_float32_max_order(::Type{<:PW92Correlation}) = 2

# α₁, β₁, β₂, β₃, β₄ of the sets ε_c(r_s, 0), ε_c(r_s, 1), −α_c(r_s): [PW92] Table I, the same for
# both variants.
const _PW92_ALPHA_BETA = (
    (0.2137, 7.5957, 3.5876, 1.6382, 0.49294),
    (0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
    (0.11125, 10.357, 3.6231, 0.88026, 0.49671),
)

_pw92_params(::Type{T}, A) where {T} =
    ntuple(i -> convert.(T, (A[i], _PW92_ALPHA_BETA[i]...)), Val(3))

# An LDA has no σ threshold.
_pw92_thresholds(::Type{T}) where {T} = default_thresholds(T, 1.0e-15, 0.0, 2.220446049250313e-16)

# f''(0) = 4/(9(2^(1/3) − 1)) of `lda/lda_c_pw_mod.md`, in 256-bit BigFloat, and rounded to Float64
# (1.7099209341613657) and Float32 once at load, so that building the functional in these types
# converts no BigFloat.
const _FZ20_EXACT = 4 / (9 * (cbrt(big(2)) - 1))
const _FZ20_EXACT_FLOAT64 = Float64(_FZ20_EXACT)
const _FZ20_EXACT_FLOAT32 = Float32(_FZ20_EXACT)
_fz20_exact(::Type{T}) where {T} = convert(T, _FZ20_EXACT)
_fz20_exact(::Type{Float64}) = _FZ20_EXACT_FLOAT64
_fz20_exact(::Type{Float32}) = _FZ20_EXACT_FLOAT32

"""
    lda_c_pw(T = Float64)

Perdew–Wang 1992 correlation, Libxc's `lda_c_pw`: the constants as printed in [PW92] Table I and
f''(0) = 1.709921, with the default thresholds, in the number type `T`.
"""
lda_c_pw(::Type{T} = Float64) where {T} = PW92Correlation{T}(
    _pw92_params(T, (0.031091, 0.015545, 0.016887)), convert(T, 1.709921), _pw92_thresholds(T)
)

"""
    lda_c_pw_mod(T = Float64)

Perdew–Wang 1992 correlation with the constants used by PBE, Libxc's `lda_c_pw_mod`: the A values
0.0310907, 0.01554535, 0.0168869 and the exact f''(0) = 4/(9(2^(1/3) − 1)) rounded to `T`, with the
default thresholds.
"""
lda_c_pw_mod(::Type{T} = Float64) where {T} = PW92Correlation{T}(
    _pw92_params(T, (0.0310907, 0.01554535, 0.0168869)), _fz20_exact(T), _pw92_thresholds(T)
)

# Rule 1: the total density below n_thr (strict).
_cutoff(pw::PW92Correlation, ρ::Real) = ρ < pw.thresholds.density
_cutoff(pw::PW92Correlation, ρ::Tuple) = ρ[1] + ρ[2] < pw.thresholds.density

# ε_c is a function of the total density and ζ, smooth in each channel on the scale of the total,
# so both densities are seeded at max(n_r, m), with n_r the total of the channels raised to n_thr.
# Restricted, the default max(n, n_thr, m) is the same rule.
function _seed_scales(pw::PW92Correlation, ::Val{K}, ρ::NTuple{2, R}) where {K, R}
    t = convert(R, pw.thresholds.density)
    s = max(max(ρ[1], t) + max(ρ[2], t), _seed_minimum(R))
    return (s, s)
end

"""
    _seitz_radius(n)

r_s = (3/(4π n))^(1/3), [PW92] Eq. (1), for a positive normal density `n`.
"""
_seitz_radius(n) = cbrt(3 / (4 * convert(typeof(value(n)), π))) * fast_invcbrt(n)

# (E1) with p = 1: G = −2A(1 + α₁ r_s) ln[1 + 1/(2A(β₁ r_s^(1/2) + β₂ r_s + β₃ r_s^(3/2) + β₄ r_s²))],
# with `r` = r_s^(1/2).
function _pw92_G(rs, r, (A, α₁, β₁, β₂, β₃, β₄))
    q = r * (β₁ + r * (β₂ + r * (β₃ + r * β₄)))
    return -2A * (1 + α₁ * rs) * fast_log1p(inv(2A * q))
end

# (1 + x)^(4/3) for the factors of (E3), or the constant ε_ζ^(4/3) with no series part where
# 1 + x ≤ ε_ζ (rule 3). The argument is raised to ε_ζ first so that fast_cbrt stays in its domain
# where the result is discarded.
function _pw92_pow43(y, εζ)
    x = _raise(y, εζ)
    t = _pow43(x)
    return ifelse(value(y) <= εζ, zero(t) + εζ * fast_cbrt(εζ), t)
end

"""
    _pw92_eps(pw, rs, ζ[, εζ])
    _pw92_eps(pw, rs)

ε_c(r_s, ζ) of (E4), the correlation energy per electron of `pw` at the Seitz radius `rs` and the
spin polarization `ζ` ∈ [−1, 1], with the ζ-factor rule 3 applied in f(ζ) at the threshold `εζ`
(by default that of `pw`; a functional built on PW92 passes its own). Without `ζ`, the
unpolarized ε_c(r_s, 0) = G(r_s; set 0) of (E2). The arguments are numbers or `Taylor` numbers over
the same variables; `R` is the promotion of their value types, and a plain `ζ` and ε_ζ are
converted to `R`.
"""
function _pw92_eps(pw::PW92Correlation, rs, ζ, εζ = pw.thresholds.zeta)
    R = promote_type(typeof(value(rs)), typeof(value(ζ)))
    ζ = _in_value_type(R, ζ)
    r = fast_sqrt(rs)
    # (E2): the three instances of G; set 2 fits −α_c.
    ε0 = _pw92_G(rs, r, pw.params[1])
    ε1 = _pw92_G(rs, r, pw.params[2])
    αc = -_pw92_G(rs, r, pw.params[3])
    # (E3): f(ζ) = [(1 + ζ)^(4/3) + (1 − ζ)^(4/3) − 2]/(2^(4/3) − 2).
    εζ = convert(R, εζ)
    f = (_pw92_pow43(1 + ζ, εζ) + _pw92_pow43(1 - ζ, εζ) - 2) / (cbrt(convert(R, 16)) - 2)
    # (E4)
    ζ4 = ζ^4
    return ε0 + αc * f / pw.fz20 * (1 - ζ4) + (ε1 - ε0) * f * ζ4
end

_in_value_type(::Type{R}, x::Real) where {R} = convert(R, x)
_in_value_type(::Type, x::Taylor) = x

_pw92_eps(pw::PW92Correlation, rs) = _pw92_G(rs, fast_sqrt(rs), pw.params[1])

# (E5) at ζ = 0: e(n) = n G(r_s; set 0). The density is raised to n_thr only to keep fast_invcbrt in
# its domain; every point that the raise moves is cut off.
_energy_density(pw::PW92Correlation, ρ) = ρ * _pw92_eps(pw, _seitz_radius(_raise(ρ, pw.thresholds.density)))

# (E5) with rule 2: r_s and ζ are formed from the channels raised to n_thr, and so is the total
# density n_r that multiplies ε_c, so every derivative is that of n_r·ε_c at the raised densities.
# The value is n·ε_c with the caller's n = n_α + n_β. ε_c is expanded in its own variables (n, ζ),
# seeded at the scales max(n_r, m) and 1, with m of `_seed_minimum`, and composed with the series
# of n_r and ζ in the density variables.
function _energy_density(pw::PW92Correlation, ρ::Tuple{Taylor{N, K}, Taylor{N, K}}) where {N, K}
    a = _raise(ρ[1], pw.thresholds.density)
    b = _raise(ρ[2], pw.thresholds.density)
    nr = a + b
    ζ = (a - b) / nr
    s = max(value(nr), _seed_minimum(typeof(value(nr))))
    n = variable(Val(2), Val(K), Val(1), s, s)
    ε = _pw92_eps(pw, _seitz_radius(n), variable(Val(2), Val(K), Val(2), value(ζ)))
    e = substitute(n * ε, (nr / s, ζ))
    return _with_value(e, (value(ρ[1]) + value(ρ[2])) * value(ε))
end
