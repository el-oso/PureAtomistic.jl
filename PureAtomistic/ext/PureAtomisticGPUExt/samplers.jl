# Normal and gamma variates on the device from Philox draws. A step's draws share its key, and
# each variate has its own counter `(i, j, stream, 0)`: `stream` names what the number is for and
# `(i, j)` where it goes, so no two variates of a step share a counter, and a variate depends only
# on the key and its counter, not on the order in which work items run or on how many draws came
# before it.

# The streams of the counter's third word.
const _STREAM_LANGEVIN = 0x00000001       # (atom, component): the noise of the BAOAB O step
const _STREAM_BUSSI_NORMAL = 0x00000002   # (0, half-step): R₁ of a Bussi half-step
const _STREAM_GAMMA_NORMAL = 0x00000003   # (pass, slot): the normal of a Marsaglia–Tsang pass
const _STREAM_GAMMA_UNIFORM = 0x00000004  # (pass, slot): the uniform of a Marsaglia–Tsang pass
const _STREAM_GAMMA_BOOST = 0x00000005    # (0, slot): the uniform of the shape < 1 boost

"""
    _PhiloxStream(key::UInt64)

The random numbers of one step: Philox4x32-10 under `key`, split into its low and high words.
"""
struct _PhiloxStream
    key::NTuple{2, UInt32}
end
_PhiloxStream(key::UInt64) = _PhiloxStream((key % UInt32, (key >> 32) % UInt32))

@inline _draw(s::_PhiloxStream, i, j, stream::UInt32) = _philox((i % UInt32, j % UInt32, stream, zero(UInt32)), s.key)

"""
    _normal2(u1, u2) -> (z1, z2)

Two independent standard normal numbers from two independent uniforms on (0, 1] by the
Box–Muller transform: `r = √(-2 log u1)`, `(z1, z2) = r (cos 2π u2, sin 2π u2)`. `u1 > 0` keeps the
logarithm finite.
"""
@inline function _normal2(u1::T, u2::T) where {T <: AbstractFloat}
    r = sqrt(-2 * log(u1))
    s, c = sincospi(2 * u2)
    return (r * c, r * s)
end

# The first normal of the Box–Muller pair from the draw at counter `(i, j, stream, 0)`; the two
# uniforms come from words 1–2 and 3–4.
@inline function _normal(s::_PhiloxStream, ::Type{T}, i, j, stream::UInt32) where {T}
    w = _draw(s, i, j, stream)
    return _normal2(_uniform(T, w[1], w[2]), _uniform(T, w[3], w[4]))[1]
end

# A uniform on (0, 1] from words 1–2 of the draw at counter `(i, j, stream, 0)`.
@inline function _uniform(s::_PhiloxStream, ::Type{T}, i, j, stream::UInt32) where {T}
    w = _draw(s, i, j, stream)
    return _uniform(T, w[1], w[2])
end

# Each pass accepts with probability at least 0.95 at the shapes it sees (≥ 1), so 10,000
# rejections in a row mean that the numbers are not normal and uniform; see the CPU sampler.
const _GAMMA_MAX_PASSES = 10_000

"""
    _device_gamma(src, a::T, slot) -> (g, accepted)

A gamma variate of unit scale and shape `a > 0` by the method of `randgamma`: Marsaglia–Tsang
squeeze and rejection at shape `b`, which is `a`, or `a + 1` when `a < 1` (`d = b - 1/3`, `c = 1/√(9d)`, `v = (1 + c x)³`,
accept `d v` when `u < 1 - 0.0331 x⁴` or `log u < x²/2 + d (1 - v + log v)`), times `u^(1/a)` when
`a < 1` (computed as `exp(log(u)/a)`). Pass `j` draws its normal and uniform at counters `(j, slot)` of the gamma streams of
`src`. After 10,000 rejected passes it returns `(0, false)`.
"""
@inline function _device_gamma(src, a::T, slot) where {T <: AbstractFloat}
    boost = a < one(T)
    b = boost ? a + one(T) : a
    d = b - one(T) / 3
    c = one(T) / sqrt(9 * d)
    for j in 1:_GAMMA_MAX_PASSES
        x = _normal(src, T, j, slot, _STREAM_GAMMA_NORMAL)
        w = one(T) + c * x
        if w > zero(T)
            v = w^3
            u = _uniform(src, T, j, slot, _STREAM_GAMMA_UNIFORM)
            # The squeeze comes first, so `log` is evaluated only when it fails. x⁴ is x² x²:
            # Base computes a Float32 to the integer power 4 in Float64.
            x2 = x * x
            if u < one(T) - T(0.0331) * x2 * x2 || log(u) < x2 / 2 + d * (one(T) - v + log(v))
                g = d * v
                if boost
                    # u^(1/a) as exp(log(u)/a): Base's Float32 power converts to Float64 in line.
                    g *= exp(log(_uniform(src, T, 0, slot, _STREAM_GAMMA_BOOST)) / a)
                end
                return g, true
            end
        end
    end
    return zero(T), false
end
