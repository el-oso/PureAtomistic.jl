"""
    Thresholds{T}(density, sigma, zeta)

The thresholds of a functional's PureXC rule, in its number type `T`: the density threshold
`density`, the reported gradient threshold `sigma` and the spin-polarization threshold `zeta`.
The floor applied to a σ is `sigma_floor`, which is `sigma^2`, or `floatmin(Float32)` where that
is larger in the thresholds `default_thresholds` builds for `Float32`.
"""
struct Thresholds{T}
    density::T
    sigma::T
    zeta::T
    sigma_floor::T
end

Thresholds{T}(density, sigma, zeta) where {T} =
    Thresholds{T}(density, sigma, zeta, convert(T, sigma)^2)

"""
    default_thresholds(T, density, sigma, zeta)

The thresholds of a functional built for the number type `T`, from the spec values `density`,
`sigma` and `zeta`. Each value is converted to `T`. For `Float32`, `zeta` is `eps(Float32)` and a
density or σ floor below `floatmin(Float32)` is raised to `floatmin(Float32)`.
"""
default_thresholds(::Type{T}, density, sigma, zeta) where {T} = Thresholds{T}(density, sigma, zeta)

function default_thresholds(::Type{Float32}, density, sigma, zeta)
    s = convert(Float32, sigma)
    tiny = floatmin(Float32)
    return Thresholds{Float32}(max(convert(Float32, density), tiny), s, eps(Float32), max(s^2, tiny))
end

"""
    _raise(x, floor)

`x` with its value raised to `floor`. For a `Taylor`, only the value moves: the series part is kept,
so the derivatives are those at the floor.
"""
_raise(x::Real, floor) = max(x, convert(typeof(x), floor))

function _raise(x::Taylor{N, K, M, T, L}, floor) where {N, K, M, T, L}
    v = max(value(x), convert(T, floor))
    return Taylor{N, K, M, T, L}(ntuple(i -> i == 1 ? v : x.c[i], Val(L)))
end

"""
    _with_value(x, v)

The `Taylor` number `x` with its value replaced by `v` and its series part kept.
"""
function _with_value(x::Taylor{N, K, M, T, L}, v) where {N, K, M, T, L}
    w = convert(T, v)
    return Taylor{N, K, M, T, L}(ntuple(i -> i == 1 ? w : x.c[i], Val(L)))
end

# The thresholds `t` converted to the type S, value by value.
_thresholds_in(::Type{S}, t::Thresholds) where {S} = Thresholds{S}(t.density, t.sigma, t.zeta, t.sigma_floor)

"""
    _to_number_type(T, x)

`x` in the number type `T` of a functional: `convert(T, x)`, or for a `Taylor` type `T` a plain
number as the constant series of that type.
"""
_to_number_type(::Type{T}, x) where {T} = convert(T, x)
_to_number_type(::Type{T}, x::Real) where {T <: Taylor} = x isa Taylor ? convert(T, x) : _constant_series(T, x)
_constant_series(::Type{Taylor{N, K, M, V, L}}, x) where {N, K, M, V, L} =
    Taylor{N, K, M, V, L}(ntuple(i -> i == 1 ? convert(V, x) : zero(V), Val(L)))
