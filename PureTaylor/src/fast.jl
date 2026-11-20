# Scalar elementary functions for Float32 and Float64 whose compiled code has no calls (other than
# LLVM intrinsics), no throws and no data-dependent branches, so that a loop over many points can be
# vectorized. Domains are documented and not checked; outside them the result is unspecified. Every
# other `Real` falls back to the Base function.

const _FastFloat = Union{Float32,Float64}

_int(::Type{Float64}) = Int64
_int(::Type{Float32}) = Int32
_mantissa_bits(::Type{Float64}) = 52
_mantissa_bits(::Type{Float32}) = 23
_exponent_bias(::Type{Float64}) = 1023
_exponent_bias(::Type{Float32}) = 127

# ln 2 = LN2_HI + LN2_LO. LN2_HI has enough trailing zero bits that k·LN2_HI is exact for every
# exponent k of a finite float.
const LN2_HI64 = reinterpret(Float64, 0x3fe62e42fee00000)
const LN2_LO64 = Float64(log(big(2)) - big(LN2_HI64))
const LN2_HI32 = reinterpret(Float32, 0x3f317200)
const LN2_LO32 = Float32(log(big(2)) - big(LN2_HI32))
_ln2_hi(::Type{Float64}) = LN2_HI64
_ln2_lo(::Type{Float64}) = LN2_LO64
_ln2_hi(::Type{Float32}) = LN2_HI32
_ln2_lo(::Type{Float32}) = LN2_LO32
const INV_LN2_64 = Float64(1 / log(big(2)))
const INV_LN2_32 = Float32(1 / log(big(2)))
_inv_ln2(::Type{Float64}) = INV_LN2_64
_inv_ln2(::Type{Float32}) = INV_LN2_32
const SQRT_HALF64 = Float64(sqrt(big(0.5)))
const SQRT_HALF32 = Float32(sqrt(big(0.5)))
_sqrt_half(::Type{Float64}) = SQRT_HALF64
_sqrt_half(::Type{Float32}) = SQRT_HALF32

"""
    fast_sqrt(x)

`√x`. For `Float32` and `Float64` it is the correctly rounded hardware square root, without the
domain check of Base's `sqrt`. Domain: `x ≥ 0`. Other `Real`s use Base's `sqrt`.
"""
@inline fast_sqrt(x::_FastFloat) = Base.sqrt_llvm(x)
fast_sqrt(x::Real) = sqrt(x)

# x^(-1/3). Initial guess from the bits, bits(y) = M - bits(x)/3, with the division done in floating
# point so that no wide integer multiply is needed, then iterations of the third-order update
# y ← y·(1 + e/3 + 2e²/9), e = 1 - x·y³.
const INVCBRT_M64 = Int64(round((4 / 3) * (1023 - 0.0450465) * 2.0^52))
const INVCBRT_M32 = Int32(round((4 / 3) * (127 - 0.0450465) * 2.0^23))
@inline _invcbrt_guess(x::Float64) =
    reinterpret(Float64, INVCBRT_M64 - unsafe_trunc(Int64, Float64(reinterpret(Int64, x)) * (1 / 3)))
@inline _invcbrt_guess(x::Float32) =
    reinterpret(Float32, INVCBRT_M32 - unsafe_trunc(Int32, Float32(reinterpret(Int32, x)) * (1.0f0 / 3)))
@inline function _invcbrt_step(x::T, y::T) where {T}
    e = muladd(-x, y * y * y, one(T))
    return muladd(y * e, muladd(T(2 / 9), e, T(1 / 3)), y)
end
@inline _invcbrt_steps(y::Float64, x::Float64) = _invcbrt_step(x, _invcbrt_step(x, _invcbrt_step(x, y)))
@inline _invcbrt_steps(y::Float32, x::Float32) = _invcbrt_step(x, _invcbrt_step(x, y))

"""
    fast_invcbrt(x)

`x^(-1/3)`. Domain for `Float32` and `Float64`: positive normal `x`. Other `Real`s use
`inv(cbrt(x))`.
"""
@inline fast_invcbrt(x::_FastFloat) = _invcbrt_steps(_invcbrt_guess(x), x)
fast_invcbrt(x::Real) = inv(cbrt(x))

"""
    fast_cbrt(x)

`∛x`. Domain for `Float32` and `Float64`: positive normal `x`. Other `Real`s use Base's `cbrt`.
"""
@inline function fast_cbrt(x::T) where {T<:_FastFloat}
    # c = x·y² with y = x^(-1/3), then one Newton step c ← c + (x - c³)·y²/3.
    # The step is written as (x·y - c²·(c·y))·y/3 because c³ overflows for x near floatmax.
    y = fast_invcbrt(x)
    c = x * (y * y)
    return muladd((x * y - c * c * (c * y)) * y, T(1 / 3), c)
end
fast_cbrt(x::Real) = cbrt(x)

# R(s²) = Σ_{j≥1} 2·s^(2j)/(2j+1), truncated where the next term is below the rounding of log m.
const LOG_COEFFS64 = ntuple(j -> 2 / (2j + 1), 9)
const LOG_COEFFS32 = ntuple(j -> Float32(2 / (2j + 1)), 5)
@inline _log_series(s2::Float64) = s2 * evalpoly(s2, LOG_COEFFS64)
@inline _log_series(s2::Float32) = s2 * evalpoly(s2, LOG_COEFFS32)

"""
    fast_log(x)

Natural logarithm. Domain for `Float32` and `Float64`: positive normal `x`. Other `Real`s use Base's
`log`.
"""
@inline function fast_log(x::T) where {T<:_FastFloat}
    # x = 2ᵏ·m with m ∈ [√½, √2); f = m - 1, s = f/(2 + f), log m = 2·atanh(s) = f - s·(f - R(s²)).
    I = _int(T)
    p = _mantissa_bits(T)
    b = reinterpret(I, x)
    k = (b - reinterpret(I, _sqrt_half(T))) >> p
    m = reinterpret(T, b - (k << p))
    f = m - one(T)
    s = f / (T(2) + f)
    R = _log_series(s * s)
    kf = T(k)
    return kf * _ln2_hi(T) + (f - (s * (f - R) - kf * _ln2_lo(T)))
end
fast_log(x::Real) = log(x)

"""
    fast_log1p(x)

`log(1 + x)`, accurate for small `x`. Domain for `Float32` and `Float64`: `x > -1` with `1 + x`
normal. Other `Real`s use Base's `log1p`.
"""
@inline function fast_log1p(x::T) where {T<:_FastFloat}
    # u = 1 + x rounded; (x - (u - 1))/u is the first-order correction for that rounding.
    u = one(T) + x
    return fast_log(u) + (x - (u - one(T))) / u
end
fast_log1p(x::Real) = log1p(x)

# expm1(r) = Σ_{j≥1} rʲ/j! for |r| ≤ ln2/2, truncated where the next term is below the rounding.
const EXPM1_COEFFS64 = ntuple(j -> 1 / factorial(j), 13)
const EXPM1_COEFFS32 = ntuple(j -> Float32(1 / factorial(j)), 8)
@inline _expm1_series(r::Float64) = r * evalpoly(r, EXPM1_COEFFS64)
@inline _expm1_series(r::Float32) = r * evalpoly(r, EXPM1_COEFFS32)

# x = k·ln2 + r with k = round(x/ln2), |r| ≤ ln2/2; returns (expm1(r), 2ᵏ). 2ᵏ is built from its
# exponent bits, so k + bias must be a normal exponent: |x| ≤ 700 (Float64), |x| ≤ 85 (Float32).
@inline function _exp_reduce(x::T) where {T<:_FastFloat}
    k = round(x * _inv_ln2(T))
    r = muladd(-k, _ln2_lo(T), muladd(-k, _ln2_hi(T), x))
    I = _int(T)
    twok = reinterpret(T, (unsafe_trunc(I, k) + I(_exponent_bias(T))) << _mantissa_bits(T))
    return _expm1_series(r), twok
end

"""
    fast_exp(x)

`eˣ`. Domain: `|x| ≤ 700` for `Float64`, `|x| ≤ 85` for `Float32`. Other `Real`s use Base's `exp`.
"""
@inline function fast_exp(x::_FastFloat)
    p, twok = _exp_reduce(x)
    return muladd(twok, p, twok)
end
fast_exp(x::Real) = exp(x)

"""
    fast_expm1(x)

`eˣ - 1`, accurate for small `x`. Domain: `|x| ≤ 700` for `Float64`, `|x| ≤ 85` for `Float32`. Other
`Real`s use Base's `expm1`.
"""
@inline function fast_expm1(x::T) where {T<:_FastFloat}
    # eˣ - 1 = 2ᵏ·expm1(r) + (2ᵏ - 1).
    p, twok = _exp_reduce(x)
    return muladd(twok, p, twok - one(T))
end
fast_expm1(x::Real) = expm1(x)
