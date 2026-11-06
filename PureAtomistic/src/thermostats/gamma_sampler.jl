"""
    randgamma(rng, shape) -> typeof(float(shape))

Draw one sample of the gamma distribution with unit scale and shape `shape > 0`, density
`x^(shape - 1) e^(-x) / Γ(shape)`, computed in the floating-point type of `shape`.

For `shape ≥ 1` this is the squeeze-and-reject method of G. Marsaglia and W. W. Tsang,
ACM Trans. Math. Softw. 26, 363 (2000): with `d = shape - 1/3`, `c = 1/√(9d)`, draw a standard
normal `x` and set `v = (1 + c x)³`; reject when `v ≤ 0`, accept `d v` when the uniform `u`
satisfies `u < 1 - 0.0331 x⁴` (the squeeze) or `log u < x²/2 + d (1 - v + log v)`, and
otherwise draw again. For `shape < 1` the sample is `g · u^(1/shape)` with `g` drawn at shape
`shape + 1` and `u` uniform on `(0, 1)`. The method is re-derived here, not transcribed from the
paper; the tests verify the moments and the quantiles of the result.
"""
function randgamma(rng::Random.AbstractRNG, shape::Real)
    a = float(shape)
    (isfinite(a) && a > 0) ||
        throw(ArgumentError(lazy"pureatomistic: the gamma shape must be positive and finite, got $shape"))
    a < 1 || return _marsaglia_tsang(rng, a)
    # u is uniform on (0, 1]: `rand` gives [0, 1), and u = 0 would give a zero sample.
    u = one(a) - rand(rng, typeof(a))
    return _marsaglia_tsang(rng, a + one(a)) * u^(one(a) / a)
end

# Each pass accepts with probability at least 0.95 (measured 0.951 at shape 1, the smallest
# shape reaching this function, and higher for larger shapes), so 10,000 rejections in a row
# have probability below 1e-13000 for a working generator: reaching the cap means the random
# numbers are not normal and uniform.
const _MARSAGLIA_TSANG_MAX_PASSES = 10_000

function _marsaglia_tsang(rng::Random.AbstractRNG, a::T) where {T <: AbstractFloat}
    d = a - one(T) / 3
    c = one(T) / sqrt(9 * d)
    for _ in 1:_MARSAGLIA_TSANG_MAX_PASSES
        x = randn(rng, T)
        w = one(T) + c * x
        w > 0 || continue
        v = w^3
        u = rand(rng, T)
        # The squeeze comes first, so `log` is evaluated only when it fails.
        (u < one(T) - T(0.0331) * x^4 || log(u) < x^2 / 2 + d * (one(T) - v + log(v))) && return d * v
    end
    throw(ArgumentError("pureatomistic: the gamma sampler rejected 10000 proposals in a row; the random number generator does not produce normal and uniform numbers"))
end
