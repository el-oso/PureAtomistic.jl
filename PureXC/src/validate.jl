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
`2^(e − ⌈e/4⌉ − g(K))` = `2^(⌊3e/4⌋ − g(K))`, with `e = exponent(floatmax(R))` and the loss `g(K)` of
`_LDA_DENSITY_LOSS`. It is the largest power of two at which every output of `lda_x`, `lda_c_pw`
and `lda_c_pw_mod` at order `K`, restricted and polarized at the splits (n, 0), (0, n),
(n/2, n/2) and (3n/4, n/4), stays within 2100·eps(R) of the same functional in BigFloat,
relative to the largest entry of each output. `lda_x` sets it: e(n) ∝ n^(4/3) overflows above
about 2^(3e/4).
"""
_lda_density_max(::Type{R}, ::Val{K}) where {R, K} =
    ldexp(one(R), exponent(floatmax(R)) - cld(exponent(floatmax(R)), 4) - _lda_density_loss(K))

# g(K) for K = 0, 1, …, 10. Measured largest accurate power of two, the same as the formula, for
# K = 0…10: Float64 2^767, 767, 767, 766, 764, 762, 759, 756, 752, 749, 745; Float32 2^95, 95, 95,
# 94, 92, 90, 87, 84, 80, 77, 73. Above K = 10 the loss is extended by 4 per order, the largest
# step measured; those orders are not measured.
const _LDA_DENSITY_LOSS = (0, 0, 0, 1, 3, 5, 8, 11, 15, 18, 22)
_lda_density_loss(K::Int) = K < length(_LDA_DENSITY_LOSS) ? _LDA_DENSITY_LOSS[K + 1] :
    _LDA_DENSITY_LOSS[end] + 4 * (K + 1 - length(_LDA_DENSITY_LOSS))

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
