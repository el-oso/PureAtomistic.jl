# Cartesian components, primitive normalization and contraction renormalization. No function here
# calls `^` with a non-integer exponent: every non-integer power is built from multiplications and
# `sqrt`, which is correctly rounded, so each result has a stated rounding bound.

"""
    ncart(l::Integer) -> Int

The number of Cartesian components of angular momentum `l`, `(l + 1)(l + 2)/2`, for every
`l ≥ 0`. Throws an `ArgumentError` for a negative `l`.
"""
function ncart(l::Integer)
    _check_l(l)
    return (Int(l) + 1) * (Int(l) + 2) ÷ 2
end

_check_l(l::Integer) = l >= 0 || throw(ArgumentError(lazy"basis: l must be ≥ 0, got $(Int(l))"))

# The Cartesian components (lx, ly, lz) of l = 0 to 5, in the oracle's order (`shell.rs:84-117`).
# No rule generates the order (l = 3 lists xyy before xxy), so it is data.
const CARTESIAN_COMPONENTS = (
    (0, 0, 0),
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1),
    (3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 2, 0), (2, 1, 0), (2, 0, 1), (1, 0, 2), (0, 1, 2), (0, 2, 1),
    (1, 1, 1),
    (4, 0, 0), (0, 4, 0), (0, 0, 4), (3, 1, 0), (3, 0, 1), (1, 3, 0), (0, 3, 1), (1, 0, 3), (0, 1, 3),
    (2, 2, 0), (2, 0, 2), (0, 2, 2), (2, 1, 1), (1, 2, 1), (1, 1, 2),
    (5, 0, 0), (0, 5, 0), (0, 0, 5), (4, 1, 0), (4, 0, 1), (1, 4, 0), (0, 4, 1), (1, 0, 4), (0, 1, 4),
    (3, 2, 0), (3, 0, 2), (2, 3, 0), (0, 3, 2), (2, 0, 3), (0, 2, 3), (3, 1, 1), (1, 3, 1), (1, 1, 3),
    (2, 2, 1), (2, 1, 2), (1, 2, 2),
)::NTuple{56, NTuple{3, Int}}

# The index in `CARTESIAN_COMPONENTS` before the first component of l = 0 to 5.
const COMPONENT_OFFSETS = (0, 1, 4, 10, 20, 35)

"""
    cartesian_component(l::Integer, i::Integer) -> NTuple{3,Int}

The exponents `(lx, ly, lz)` of the `i`-th Cartesian component of angular momentum `l`, in the
oracle's component order, for `0 ≤ l ≤ 5` and `1 ≤ i ≤ ncart(l)`. The order is that of the Molden
format for `l ≤ 4`. Throws an `ArgumentError` for `l` outside `0:5` or `i` outside `1:ncart(l)`.
"""
function cartesian_component(l::Integer, i::Integer)
    n = ncart(l)
    l <= 5 || throw(ArgumentError(lazy"basis: l = $(Int(l)) is above the supported l ≤ 5"))
    1 <= i <= n ||
        throw(ArgumentError(lazy"basis: component $(Int(i)) is outside 1:$(n) for l = $(Int(l))"))
    return CARTESIAN_COMPONENTS[COMPONENT_OFFSETS[l + 1] + i]
end

"""
    primitive_norm(α::Real, lx::Integer, ly::Integer, lz::Integer)

The normalization constant of the Cartesian Gaussian `x^lx y^ly z^lz exp(-α r²)`,
`(2α/π)^{3/4} sqrt((4α)^l / ((2lx−1)!! (2ly−1)!! (2lz−1)!!))` with `l = lx + ly + lz` and
`n!! = 1` for `n ≤ 0`, in `float(typeof(α))`. Each component has its own constant: `xy` and `xx`
differ.

Throws an `ArgumentError` for an `α` that is not finite and positive, a negative Cartesian
exponent, and a product of the double factorials above `typemax(Int)`. Every component with
`l ≤ 17` is within that limit; `(18, 0, 0)` is the first that is not.
"""
function primitive_norm(α::Real, lx::Integer, ly::Integer, lz::Integer)
    (isfinite(α) && α > 0) ||
        throw(ArgumentError(lazy"basis: the exponent is $(Float64(α)); exponents must be finite and positive"))
    (lx >= 0 && ly >= 0 && lz >= 0) || throw(
        ArgumentError(lazy"basis: Cartesian exponents must be ≥ 0, got $(Int(lx)), $(Int(ly)), $(Int(lz))")
    )
    T = float(typeof(α))
    a = T(α)
    l = Int(lx) + Int(ly) + Int(lz)
    # (2α/π)^{3/4} as sqrt(x) · sqrt(sqrt(x)).
    s = sqrt(2a / T(π))
    fourα = 4a
    q = one(T)
    for _ in 1:l
        q *= fourα
    end
    dfact = _double_factorial_product(Int(lx), Int(ly), Int(lz))
    return s * sqrt(s) * sqrt(q / T(dfact))
end

# (2lx−1)!! (2ly−1)!! (2lz−1)!! in exact `Int` arithmetic; an overflow throws rather than wraps.
function _double_factorial_product(lx::Int, ly::Int, lz::Int)
    r = 1
    for m in (lx, ly, lz)
        n = 2m - 1
        while n > 1
            r, overflow = Base.Checked.mul_with_overflow(r, n)
            overflow && throw(
                ArgumentError(lazy"basis: (2lx−1)!!(2ly−1)!!(2lz−1)!! overflows Int for the Cartesian exponents $(lx), $(ly), $(lz)")
            )
            n -= 2
        end
    end
    return r
end

"""
    contraction_renorm(exponents::AbstractVector, coefficients::AbstractVector, l::Integer)

The factor `1/sqrt(Σ_ij c_i c_j t_ij)` with `t_ij = b^{l+3/2}` and `b = 2√(α_i α_j)/(α_i + α_j)`
that normalizes the contraction of unit-normalized primitives with exponents `α` and raw
coefficients `c` at total angular momentum `l`. The result has type
`float(promote_type(eltype(exponents), eltype(coefficients)))`. The double sum runs over
`eachindex(exponents)` twice, the outer index first, so the result does not depend on the array
type.

Throws a `DimensionMismatch` when the axes differ, and an `ArgumentError` for empty vectors, a
negative `l`, an exponent that is not finite and positive, an exponent whose square in the result
type is below its `floatmin` (where the product `α_i α_j` of the base underflows), and a sum that
is not positive and finite.
"""
function contraction_renorm(exponents::AbstractVector, coefficients::AbstractVector, l::Integer)
    axes(exponents) == axes(coefficients) || throw(
        DimensionMismatch(
            lazy"basis: exponents and coefficients have different axes: $(axes(exponents)) and $(axes(coefficients))"
        )
    )
    isempty(exponents) && throw(ArgumentError("basis: a contraction needs at least one primitive"))
    _check_l(l)
    T = float(promote_type(eltype(exponents), eltype(coefficients)))
    for k in eachindex(exponents)
        x = exponents[k]
        (isfinite(x) && x > 0) || throw(
            ArgumentError(lazy"basis: exponent $(k) of a contraction is $(Float64(x)); exponents must be finite and positive")
        )
        y = T(x)
        y * y >= floatmin(T) || throw(
            ArgumentError(lazy"basis: exponent $(k) of a contraction is $(Float64(x)), whose square is below floatmin($(nameof(T)))")
        )
    end
    acc = _renorm_sum(exponents, coefficients, Int(l))
    (acc > 0 && isfinite(acc)) || throw(ArgumentError("basis: the contraction has zero or negative norm"))
    return one(acc) / sqrt(acc)
end

# The sum `Σ_ij c_i c_j t_ij` of `contraction_renorm`, without its checks: the caller has checked
# the axes and the exponents, including that each exponent squared is at least `floatmin(T)`, and
# checks the sum. Below `floatmin(T)` the product `α_i α_j` of the base loses precision or
# underflows to 0, even where `b` is exactly 1; since rounding is monotone, the smallest product is
# the smallest exponent squared. The result has type
# `float(promote_type(eltype(exponents), eltype(coefficients)))`.
function _renorm_sum(exponents::AbstractVector, coefficients::AbstractVector, L::Int)
    T = float(promote_type(eltype(exponents), eltype(coefficients)))
    acc = zero(T)
    for i in eachindex(exponents), j in eachindex(exponents)
        ai = T(exponents[i])
        aj = T(exponents[j])
        b = 2 * sqrt(ai * aj) / (ai + aj)
        # b^{l+3/2} as b^{l+1} · sqrt(b).
        t = b
        for _ in 1:L
            t *= b
        end
        t *= sqrt(b)
        acc += T(coefficients[i]) * T(coefficients[j]) * t
    end
    return acc
end
