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
    _check_inputs(f, ρ[, σ])

Throw an `ArgumentError` with the prefix `"xc: "` for an input the functional `f` does not accept:
a NaN or infinite input, a negative density or σ, or a polarized `σαβ` with
`σαβ² > σαα·σββ·(1 + 16·eps(R))` in the result type `R`. Otherwise return `nothing`.
"""
_check_inputs(::AbstractLDA, ρ::Real) = _check_nonnegative(ρ, :ρ)

function _check_inputs(::AbstractLDA, ρ::NTuple{2, Real})
    _check_nonnegative(ρ[1], :ρα)
    _check_nonnegative(ρ[2], :ρβ)
    return nothing
end

function _check_inputs(::AbstractGGA, ρ::Real, σ::Real)
    _check_nonnegative(ρ, :ρ)
    _check_nonnegative(σ, :σ)
    return nothing
end

# The σαβ bound is a rounding bound for σ's formed as 3-term dot products: each σ carries at most
# 3 eps, a product of two at most 6 eps, and a factor of 2 is margin.
function _check_inputs(f::AbstractGGA, ρ::NTuple{2, Real}, σ::NTuple{3, Real})
    _check_nonnegative(ρ[1], :ρα)
    _check_nonnegative(ρ[2], :ρβ)
    _check_nonnegative(σ[1], :σαα)
    _check_finite(σ[2], :σαβ)
    _check_nonnegative(σ[3], :σββ)
    R = _result_type(f, ρ, σ)
    aa, ab, bb = convert(R, σ[1]), convert(R, σ[2]), convert(R, σ[3])
    ab^2 > aa * bb * (1 + 16 * eps(R)) &&
        _throw_input(lazy"xc: σαβ = $(σ[2]) exceeds √(σαα·σββ) for σαα = $(σ[1]), σββ = $(σ[3])")
    return nothing
end
