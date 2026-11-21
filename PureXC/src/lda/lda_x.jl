# Slater exchange of the uniform electron gas, from the spec `lda/lda_x.md`.

"""
    LDAExchange{T}(thresholds)

Slater (Dirac) exchange of the uniform electron gas, Libxc's `lda_x`, with the density threshold
`thresholds.density` in the number type `T`. Built by [`lda_x`](@ref).
"""
struct LDAExchange{T} <: AbstractLDA{T}
    thresholds::Thresholds{T}
end

"""
    lda_x(T = Float64)

Slater exchange, Libxc's `lda_x`, with its default thresholds in the number type `T`.
"""
lda_x(::Type{T} = Float64) where {T} =
    LDAExchange{T}(default_thresholds(T, 1.0e-15, 1.0000000000000027e-20, 2.220446049250313e-16))

kind(::LDAExchange) = :exchange

_with_number_type(f::LDAExchange, ::Type{S}) where {S} = LDAExchange{S}(_thresholds_in(S, f.thresholds))

# x^(4/3), with the value x·∛x. For a `Taylor` it is one power series: in the product x·∛x a
# coefficient of the factor ∛x can overflow where the coefficient of x^(4/3) fits.
_pow43(x::Real) = x * fast_cbrt(x)
_pow43(x::Taylor) = power_series(x, convert(typeof(value(x)), 4) / 3, _pow43(value(x)))

# Rule 3: restricted input is n_α = n_β = n/2, so rule 2 cuts off n/2 ≤ n_thr, which includes
# rule 1 (n < n_thr).
_cutoff(f::LDAExchange, ρ::Real) = ρ / 2 <= f.thresholds.density
# Rule 1: the total density below n_thr.
_cutoff(f::LDAExchange, ρ::Tuple) = ρ[1] + ρ[2] < f.thresholds.density

# (E2): e(n) = −(3/4)(3/π)^(1/3) n^(4/3). The density is raised to n_thr only to keep fast_cbrt in
# its domain; every point that the raise moves is cut off.
function _energy_density(f::LDAExchange, ρ)
    R = typeof(value(ρ))
    n = _raise(ρ, f.thresholds.density)
    return -(convert(R, 3) / 4 * cbrt(3 / convert(R, π))) * _pow43(n)
end

# (E3): e(n_α, n_β) = −(3/4)(6/π)^(1/3) (n_α^(4/3) + n_β^(4/3)), channel by channel.
function _energy_density(f::LDAExchange, ρ::Tuple)
    R = typeof(value(ρ[1]))
    thr = convert(R, f.thresholds.density)
    return -(convert(R, 3) / 4 * cbrt(6 / convert(R, π))) * (_lda_x_channel(ρ[1], thr) + _lda_x_channel(ρ[2], thr))
end

# n_σ^(4/3), or zero for n_σ ≤ n_thr (rule 2). The channel is raised to n_thr first so that
# fast_cbrt stays in its domain where the term is discarded.
function _lda_x_channel(ρσ, thr)
    x = _raise(ρσ, thr)
    t = _pow43(x)
    return ifelse(value(ρσ) <= thr, zero(t), t)
end
