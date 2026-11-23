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
the domain limit of an LDA (see `_lda_density_max`), an input outside the domain limits of a GGA
(see `_check_gga_domain`), or a polarized `σαβ` with
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
and (n, r·n) for r log-spaced over [1e-9, 1] (Float32) or [1e-17, 1] (Float64); for PW92 in Float64
also (n, r·n), (n/2, r·n/2) and their mirrors for r every 5e-18 over [5e-17, 4.5e-16] and every
1e-18 over [9e-17, 1.3e-16], across ε_ζ/2 and ε_ζ, and r log-spaced over [1e-17, 1e-14]. The bound
is 2100 eps(R) at K ≤ 3 and 1000 eps(R) above; for PW92 where the factor (1 ± ζ)^(4/3) of the
smaller channel is kept it adds 8κ·eps(R), κ = 2/(1 − |ζ|), the conditioning of 1 ± ζ formed from ζ
(spec `lda_c_pw`, section 8); a Float32 call evaluated in Float64 has the bound 50 eps(Float32).
Measured: Float64 2^767 for K = 0 to 6, then 2^731, 679, 627, 575 for K = 7 to 10; Float32 2^95 for
K = 0 to 10. For other types and above K = 10 it is not measured. BigFloat and the other types use
the Float64 p(K), so that, for example, BigFloat at K = 7 rejects n > 2^731.
"""
_lda_density_max(::Type{R}, ::Val{K}) where {R, K} =
    ldexp(one(R), min(exponent(floatmax(R)) - cld(exponent(floatmax(R)), 4), _pw92_density_exponent(K)))

# p(K) for K = 0, 1, …, 10: the largest power of two at which PW92 is accurate in Float64 over the
# splits of `_lda_density_max` (measured). Above K = 10 it decreases by 53 per order, more than the
# measured step of 51 or 52 (log2 of 2/ε_ζ is 53); those orders are not measured.
const _PW92_DENSITY_EXPONENT = (1023, 1023, 987, 936, 885, 834, 782, 731, 679, 627, 575)
_pw92_density_exponent(K::Int) = K < length(_PW92_DENSITY_EXPONENT) ? _PW92_DENSITY_EXPONENT[K + 1] :
    _PW92_DENSITY_EXPONENT[end] - 53 * (K + 1 - length(_PW92_DENSITY_EXPONENT))

function _check_density_max(::Val{K}, n::R) where {K, R}
    m = _lda_density_max(R, Val(K))
    n > m && _throw_input(lazy"xc: total density n = $n exceeds $m, the largest an LDA accepts at order $K in $R")
    return nothing
end

function _check_inputs(f::AbstractGGA, k::Val, ρ::Real, σ::Real)
    _check_nonnegative(ρ, :ρ)
    _check_nonnegative(σ, :σ)
    _check_gga_domain(f, k, ρ, σ)
    return nothing
end

# The σαβ bound |σαβ| ≤ √σαα·√σββ·(1 + 8 eps) is a rounding bound for σ's formed as 3-term dot
# products in their own type S: each σ carries at most 3 eps, a product of two at most 6 eps, and a
# factor of 2 is margin. It is compared through square roots so that nothing underflows.
function _check_inputs(f::AbstractGGA, k::Val, ρ::NTuple{2, Real}, σ::NTuple{3, Real})
    _check_nonnegative(ρ[1], :ρα)
    _check_nonnegative(ρ[2], :ρβ)
    _check_nonnegative(σ[1], :σαα)
    _check_finite(σ[2], :σαβ)
    _check_nonnegative(σ[3], :σββ)
    S = float(promote_type(typeof(σ[1]), typeof(σ[2]), typeof(σ[3])))
    aa, ab, bb = convert(S, σ[1]), convert(S, σ[2]), convert(S, σ[3])
    abs(ab) > sqrt(aa) * sqrt(bb) * (1 + 8 * eps(S)) &&
        _throw_input(lazy"xc: σαβ = $(σ[2]) exceeds √(σαα·σββ) for σαα = $(σ[1]), σββ = $(σ[3])")
    _check_gga_domain(f, k, ρ, σ)
    return nothing
end

"""
    _check_gga_domain(f, Val(K), ρ, σ)

Throw an `ArgumentError` with the prefix `"xc: "` for inputs outside the domain limits of the GGA
`f` at order `K` (a total density above n_max or a reduced gradient above its limit), given inputs
that pass the sign, finiteness and σαβ checks. A GGA without limits accepts every such input.
"""
_check_gga_domain(::AbstractGGA, ::Val, ρ, σ) = nothing

"""
    _gga_density_max(R, Val(K))

n_max(R, K) of the GGA family, the largest total density a PBE functional accepts at order `K` in
the type `R`: `2^p` with p = 381 in Float64 and 45 in Float32, above which the σ scale
4(6π²)^(2/3)·n^(8/3) of `pbe_x`, or σ itself at s² = 1, is beyond floatmax(R). The exponent is
⌊3(e − 6)/8⌋ with e = exponent(floatmax(R)), measured to be the largest power of two at which every
output is within its bound against BigFloat over the splits of `_lda_density_max` and the
reduced gradients up to their limits: `pbe_x` at orders 0 to 10 and `pbe_c` restricted and
polarized at orders 0 to 10 (`test/scans/pbe_c_t2_max.jl` for pbe_c). BigFloat and the other types
use the Float64 value.
"""
_gga_density_max(::Type{R}, ::Val{K}) where {R, K} = ldexp(one(R), 381)
_gga_density_max(::Type{Float32}, ::Val{K}) where {K} = ldexp(1.0f0, 45)

function _check_gga_density_max(::Val{K}, n::R) where {K, R}
    m = _gga_density_max(R, Val(K))
    n > m && _throw_input(lazy"xc: total density n = $n exceeds $m, the largest a GGA accepts at order $K in $R")
    return nothing
end

# The reduced gradient s² = σ/(c·n^(8/3)) (or t² of `pbe_c`, with its own c and power) is formed as
# (√σ/(√c·n^p))², with n^p no smaller than n_thr^(4/3), so that no subnormal power is formed; an
# overflow gives Inf, which is above every limit.
function _check_reduced_gradient(name::Symbol, x::R, m::R, ::Val{K}) where {R, K}
    x > m && _throw_input(lazy"xc: reduced gradient $name = $x exceeds $m, the largest accepted at order $K in $R")
    return nothing
end

"""
    _inputs_ok(f, Val(k), ρ[, σ])

Whether `_check_inputs(f, Val(k), ρ[, σ])` returns without throwing, computed without branches
or throws, so that a loop over many points can be vectorized. Each condition is the negation of
one throwing condition of `_check_inputs`, with the same comparison in the same type; a square
root is taken of `abs` of its argument, which equals the argument wherever the condition decides
the result (a negative input already fails its sign condition).
"""
function _inputs_ok end

# The negation of the conditions of `_check_nonnegative`.
@inline _nonnegative_ok(x) = isfinite(x) & !(x < 0)

@inline function _inputs_ok(f::AbstractLDA, ::Val{K}, ρ::Real) where {K}
    R = _domain_type(f, ρ)
    return _nonnegative_ok(ρ) & !(convert(R, ρ) > _lda_density_max(R, Val(K)))
end

@inline function _inputs_ok(f::AbstractLDA, ::Val{K}, ρ::NTuple{2, Real}) where {K}
    R = _domain_type(f, ρ)
    n = convert(R, ρ[1]) + convert(R, ρ[2])
    return _nonnegative_ok(ρ[1]) & _nonnegative_ok(ρ[2]) & !(n > _lda_density_max(R, Val(K)))
end

@inline function _inputs_ok(f::AbstractGGA, k::Val, ρ::Real, σ::Real)
    return _nonnegative_ok(ρ) & _nonnegative_ok(σ) & _gga_domain_ok(f, k, ρ, σ)
end

@inline function _inputs_ok(f::AbstractGGA, k::Val, ρ::NTuple{2, Real}, σ::NTuple{3, Real})
    S = float(promote_type(typeof(σ[1]), typeof(σ[2]), typeof(σ[3])))
    aa, ab, bb = convert(S, σ[1]), convert(S, σ[2]), convert(S, σ[3])
    signs = _nonnegative_ok(ρ[1]) & _nonnegative_ok(ρ[2]) & _nonnegative_ok(σ[1]) & isfinite(σ[2]) & _nonnegative_ok(σ[3])
    bound = !(abs(ab) > fast_sqrt(abs(aa)) * fast_sqrt(abs(bb)) * (1 + 8 * eps(S)))
    return signs & bound & _gga_domain_ok(f, k, ρ, σ)
end

"""
    _gga_domain_ok(f, Val(K), ρ, σ)

Whether `_check_gga_domain(f, Val(K), ρ, σ)` returns without throwing, without branches (see
`_inputs_ok`); it may assume nothing about the signs of the inputs, whose conditions
`_inputs_ok` combines with it.
"""
@inline _gga_domain_ok(::AbstractGGA, ::Val, ρ, σ) = true
