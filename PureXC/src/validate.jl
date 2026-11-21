@noinline _throw_input(msg) = throw(ArgumentError(msg))

# A finite, nonnegative input. −0.0 is not negative.
function _check_nonnegative(x, name)
    isfinite(x) || _throw_input(lazy"xc: $name = $x is not finite")
    x < 0 && _throw_input(lazy"xc: $name = $x is negative")
    return nothing
end

function _check_finite(x, name)
    isfinite(x) || _throw_input(lazy"xc: $name = $x is not finite")
    return nothing
end

"""
    _check_inputs(f, Val(k), ρ[, σ])

Throw an `ArgumentError` with the prefix `"xc: "` for an input the functional `f` does not accept
at derivative order `k`: a NaN or infinite input, a negative density or σ, a total density above
the domain limit of an LDA (see `_lda_density_max`), or a polarized `σαβ` with
`|σαβ| > √σαα·√σββ·(1 + 8·eps(S))` in the type `S` of the σ inputs. Otherwise return `nothing`.
"""
function _check_inputs(f::AbstractLDA, k::Val, ρ::Real)
    _check_nonnegative(ρ, :ρ)
    R = _domain_type(f, ρ)
    _check_density_max(k, convert(R, ρ))
    return nothing
end

function _check_inputs(f::AbstractLDA, k::Val, ρ::NTuple{2, Real})
    _check_nonnegative(ρ[1], :ρα)
    _check_nonnegative(ρ[2], :ρβ)
    R = _domain_type(f, ρ)
    _check_density_max(k, convert(R, ρ[1]) + convert(R, ρ[2]))
    return nothing
end

# The floating-point type in which the functional `f` evaluates the inputs `x`: the promotion of
# its number type, or the value type of a `Taylor` number type, and the input types.
_domain_type(f::AbstractFunctional, x) = float(promote_type(_value_type(_number_type(f)), _input_type(x)))
_value_type(::Type{T}) where {T} = T
_value_type(::Type{<:Taylor{N, K, M, T}}) where {N, K, M, T} = T

"""
    _lda_density_max(R, Val(K))

n_max(R, K), the largest total density an LDA accepts at derivative order `K` in the type `R`:
`2^min(⌊3e/4⌋, p(K))`, with `e = exponent(floatmax(R))` and p(K) of `_pw92_density_exponent`.

- ⌊3e/4⌋ is set by `lda_x`: e(n) ∝ n^(4/3) overflows above about 2^(3e/4).
- p(K) is set by PW92 near full polarization, where 1 − |ζ| is just above ε_ζ: there a coefficient
  of its (n, ζ) series grows like n·(1 − |ζ|)^(4/3 − K).

For Float32 and Float64 this is the measured largest power of two at which every output of
`lda_x`, `lda_c_pw` and `lda_c_pw_mod` at order `K` is within its bound against the same
functional in BigFloat (or the infinity of its sign where the exact value exceeds floatmax(R)). The
splits are n, (n, 0), (0, n), (n/2, n/2), (3n/4, n/4), (n, x) and (x, n) for x ∈ {2e-15, 1e-6, 1},
and (n, r·n) for r log-spaced over [1e-9, 1] (Float32) or [1e-17, 1] (Float64). The bound is 2100
eps(R) at K ≤ 3 and 1000 eps(R) above; for PW92 where the factor (1 ± ζ)^(4/3) of the smaller
channel is kept it adds 8κ·eps(R), κ = 2/(1 − |ζ|), the conditioning of 1 ± ζ formed from ζ (spec
`lda_c_pw`, section 8); a Float32 call evaluated in Float64 has the bound 50 eps(Float32).
Measured: Float64 2^767 for K = 0 to 6, then 2^733, 682, 630, 579 for K = 7 to 10; Float32 2^95 for
K = 0 to 10. For other types and above K = 10 it is not measured.
"""
_lda_density_max(::Type{R}, ::Val{K}) where {R, K} =
    ldexp(one(R), min(exponent(floatmax(R)) - cld(exponent(floatmax(R)), 4), _pw92_density_exponent(K)))

# p(K) for K = 0, 1, …, 10: the largest power of two at which PW92 is accurate in Float64 over the
# splits of `_lda_density_max` (measured). Above K = 10 it decreases by 53 per order, more than the
# measured step of 51 or 52 (log2 of 2/ε_ζ is 53); those orders are not measured.
const _PW92_DENSITY_EXPONENT = (1023, 1023, 987, 937, 886, 835, 784, 733, 682, 630, 579)
_pw92_density_exponent(K::Int) = K < length(_PW92_DENSITY_EXPONENT) ? _PW92_DENSITY_EXPONENT[K + 1] :
    _PW92_DENSITY_EXPONENT[end] - 53 * (K + 1 - length(_PW92_DENSITY_EXPONENT))

function _check_density_max(::Val{K}, n::R) where {K, R}
    m = _lda_density_max(R, Val(K))
    n > m && _throw_input(lazy"xc: total density n = $n exceeds $m, the largest an LDA accepts at order $K in $R")
    return nothing
end

function _check_inputs(::AbstractGGA, ::Val, ρ::Real, σ::Real)
    _check_nonnegative(ρ, :ρ)
    _check_nonnegative(σ, :σ)
    return nothing
end

# The σαβ bound |σαβ| ≤ √σαα·√σββ·(1 + 8 eps) is a rounding bound for σ's formed as 3-term dot
# products in their own type S: each σ carries at most 3 eps, a product of two at most 6 eps, and a
# factor of 2 is margin. It is compared through square roots so that nothing underflows.
function _check_inputs(::AbstractGGA, ::Val, ρ::NTuple{2, Real}, σ::NTuple{3, Real})
    _check_nonnegative(ρ[1], :ρα)
    _check_nonnegative(ρ[2], :ρβ)
    _check_nonnegative(σ[1], :σαα)
    _check_finite(σ[2], :σαβ)
    _check_nonnegative(σ[3], :σββ)
    S = float(promote_type(typeof(σ[1]), typeof(σ[2]), typeof(σ[3])))
    aa, ab, bb = convert(S, σ[1]), convert(S, σ[2]), convert(S, σ[3])
    abs(ab) > sqrt(aa) * sqrt(bb) * (1 + 8 * eps(S)) &&
        _throw_input(lazy"xc: σαβ = $(σ[2]) exceeds √(σαα·σββ) for σαα = $(σ[1]), σββ = $(σ[3])")
    return nothing
end
