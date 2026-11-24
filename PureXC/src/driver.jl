# The functional interface. A concrete LDA or GGA `f` has a field `thresholds::Thresholds{T}` and
# the methods `_energy_density(f, ρ)` and `_cutoff(f, ρ)` (LDA) or `_energy_density(f, ρ, σ)` and
# `_cutoff(f, ρ, σ)` (GGA). `_energy_density` takes, restricted, `Taylor` numbers ρ and σ; polarized,
# tuples ρ = (ρα, ρβ) and σ = (σαα, σαβ, σββ) of them. It returns e, the energy per volume, as a
# `Taylor` over the same variables. It is evaluated at every point, including those `_cutoff`
# selects, so it must not throw for nonnegative inputs.

"""
    _energy_density(f, ρ[, σ])

The energy per volume e of the functional `f`, as a `Taylor` over the variables of its `Taylor`
inputs: restricted, the numbers ρ and σ; polarized, the tuples ρ = (ρα, ρβ) and
σ = (σαα, σαβ, σββ). Each concrete LDA or GGA defines a method.
"""
function _energy_density end

"""
    _cutoff(f, ρ[, σ])

Whether the PureXC rule of `f` makes the energy and every derivative zero at the plain input
values `ρ` and `σ`, given in the result type (restricted: numbers; polarized: tuples). At such a
point the result of `_energy_density` is discarded and every output is an exact zero. Each
concrete LDA or GGA defines a method.
"""
function _cutoff end

@inline _number_type(::AbstractFunctional{T}) where {T} = T

# `Vararg{Any, N}`, not `x...`: Julia compiles `x...` as one wide method, in which the `map` is
# a dynamic call that allocates.
@inline _result_type(f::AbstractFunctional, x::Vararg{Any, N}) where {N} = promote_type(_number_type(f), map(_input_type, x)...)
@inline _input_type(x::Real) = typeof(x)
@inline _input_type(x::Tuple) = promote_type(map(typeof, x)...)

# The variables x[i] + s[i]·u_i of a series to order K, in input order.
@generated function _seed(::Val{K}, x::NTuple{N}, s::NTuple{N}) where {K, N}
    return Expr(:tuple, (:(variable(Val($N), Val($K), Val($i), x[$i], s[$i])) for i in 1:N)...)
end

# The flat variable tuple regrouped as the functional's arguments.
@inline _group(::Val{1}, ::Val{0}, t) = (t[1],)
@inline _group(::Val{1}, ::Val{1}, t) = (t[1], t[2])
@inline _group(::Val{2}, ::Val{0}, t) = ((t[1], t[2]),)
@inline _group(::Val{2}, ::Val{3}, t) = ((t[1], t[2]), (t[3], t[4], t[5]))

# The output fields from the partial derivatives `d` in `PureTaylor.multi_indices` order, over
# `nρ` density variables followed by `nσ` σ variables. Order 0 is `e`; order j has one field per
# split a + b = j, a descending, named v[j]rho[a]sigma[b]. With one density variable each field is
# a scalar; with two it is a tuple over the density multisets of size a (outer loop) and the σ
# multisets of size b (inner loop), each in lexicographic order.
@generated function _outputs(::Val{K}, ::Val{nρ}, ::Val{nσ}, d::NTuple{L}) where {K, nρ, nσ, L}
    N = nρ + nσ
    pos = Dict(α => j for (j, α) in enumerate(PureTaylor.multi_indices(Val(N), Val(K))))
    length(pos) == L || error("PureXC: $L derivatives given for $N variables at order $K")
    # Multisets of size m over 1:n as nondecreasing index tuples, in lexicographic order.
    multisets(n, m) = m == 0 ? [()] : [(i, r...) for i in 1:n for r in multisets(n, m - 1) if isempty(r) || r[1] >= i]
    counts(seq, n) = ntuple(i -> count(==(i), seq), n)
    names = Symbol[:e]
    vals = Any[:(d[$(pos[ntuple(_ -> 0, N)])])]
    for j in 1:K, a in j:-1:0
        b = j - a
        b > 0 && iszero(nσ) && continue
        push!(names, Symbol("v", j > 1 ? j : "", a > 0 ? "rho" : "", a > 1 ? a : "", b > 0 ? "sigma" : "", b > 1 ? b : ""))
        αs = [(counts(r, nρ)..., counts(s, nσ)...) for r in multisets(nρ, a) for s in multisets(nσ, b)]
        terms = [:(d[$(pos[α])]) for α in αs]
        push!(vals, nρ == 1 ? only(terms) : Expr(:tuple, terms...))
    end
    return :(NamedTuple{$(Tuple(names))}($(Expr(:tuple, vals...))))
end

# The derivatives to order K at the input values x, seeded with the scales s, grouped as nρ density
# and nσ σ inputs, mapped to the output fields. The scale changes only the representation of the
# series, not the derivatives.
@inline function _derive(f, ::Val{K}, gρ::Val, gσ::Val, x::NTuple{N, Real}, s::NTuple{N, Real}) where {K, N}
    e = _energy_density(f, _group(gρ, gσ, _seed(Val(K), x, s))...)
    d = derivatives(e, map(inv, s))
    cut = _cutoff(f, _group(gρ, gσ, x)...)
    return _outputs(Val(K), gρ, gσ, ntuple(i -> ifelse(cut, zero(d[i]), d[i]), Val(length(d))))
end

# The smallest seed scale m = floatmin(R): a scale is positive and normal, so 1/s is finite.
# `derivatives` multiplies the coefficients by the factors j/s in turn, so s^k itself need not be
# representable.
@inline _seed_minimum(::Type{R}) where {R} = floatmin(R)

# x^8 as three squarings: Base's integer power is an out-of-line call that keeps a loop over points
# from vectorizing.
@inline function _pow8(x)
    x2 = x * x
    x4 = x2 * x2
    return x4 * x4
end

# The natural size of a σ component at total density n: 4(3π²)^(2/3)·n^(8/3), the σ at which the
# reduced gradient is 1. The density is raised to floatmin so that fast_cbrt stays in its domain.
@inline _sigma_size(n::R) where {R <: Real} = 4 * cbrt(3 * convert(R, π)^2)^2 * _pow8(fast_cbrt(max(n, floatmin(R))))

# The seed scale of a density x evaluated as itself: max(|x|, n_thr, m).
@inline _density_seed(f::AbstractFunctional, x::R, m::R) where {R} = max(abs(x), convert(R, f.thresholds.density), m)

# The seed scale of a σ component x at total density n: max(|x|, r, m), with r = `_sigma_size(n)`;
# |x| covers a negative σαβ.
@inline _sigma_seed(x::R, n::R, m::R) where {R} = max(abs(x), _sigma_size(n), m)

"""
    _seed_scales(f, Val(K), ρ[, σ])

The seed scales of the inputs of `f` at derivative order `K`, one per input in input order:
restricted, a tuple for (ρ[, σ]); polarized, for (ρα, ρβ[, σαα, σαβ, σββ]). The inputs are given
in the result type R, and m = `_seed_minimum(R)` = floatmin(R). Every scale is at least m, so
1/s is finite.

The default suits a functional that is a sum of per-channel powers of the densities: each density
x has the scale max(|x|, n_thr, m), the value at which it is evaluated, and each σ component x has
max(|x|, r, m), with r = 4(3π²)^(2/3)·n^(8/3) at the total density n. A functional of the total
density and ζ defines a method that seeds each density at the total of the raised channels.
"""
@inline function _seed_scales(f::AbstractLDA, ::Val{K}, ρ::Real) where {K}
    return (_density_seed(f, ρ, _seed_minimum(typeof(ρ))),)
end

@inline function _seed_scales(f::AbstractLDA, ::Val{K}, ρ::NTuple{2, R}) where {K, R}
    m = _seed_minimum(R)
    return (_density_seed(f, ρ[1], m), _density_seed(f, ρ[2], m))
end

@inline function _seed_scales(f::AbstractGGA, ::Val{K}, ρ::R, σ::R) where {K, R}
    m = _seed_minimum(R)
    return (_density_seed(f, ρ, m), _sigma_seed(σ, ρ, m))
end

@inline function _seed_scales(f::AbstractGGA, ::Val{K}, ρ::NTuple{2, R}, σ::NTuple{3, R}) where {K, R}
    m = _seed_minimum(R)
    n = ρ[1] + ρ[2]
    return (
        _density_seed(f, ρ[1], m), _density_seed(f, ρ[2], m),
        _sigma_seed(σ[1], n, m), _sigma_seed(σ[2], n, m), _sigma_seed(σ[3], n, m),
    )
end

"""
    _xc_derivatives_unchecked(f, Val(k), ρ[, σ])

`xc_derivatives` without the input check. The outputs are in the result type R; they are
evaluated in `_evaluation_type(f, R, Val(k))`.
"""
@inline _xc_derivatives_unchecked(f::AbstractLDA, k::Val, ρ::Real) = _evaluate(f, k, ρ)
@inline _xc_derivatives_unchecked(f::AbstractLDA, k::Val, ρ::NTuple{2, Real}) = _evaluate(f, k, ρ)
@inline _xc_derivatives_unchecked(f::AbstractGGA, k::Val, ρ::Real, σ::Real) = _evaluate(f, k, ρ, σ)
@inline _xc_derivatives_unchecked(f::AbstractGGA, k::Val, ρ::NTuple{2, Real}, σ::NTuple{3, Real}) = _evaluate(f, k, ρ, σ)

"""
    _evaluation_type(f, R, Val(K))

The type in which the derivatives of `f` to order `K` with result type `R` are evaluated: `R`,
except Float64 for Float32 above the order `_float32_max_order(typeof(f))`. Above it the mixed
partials of a polarized point spread like (ρα/ρβ)^K, so that series coefficients on the path of an
output that fits in Float32 do not (at (1, 1e-6), K = 9, the order-9 partials of lda_c_pw range
from 19.8 to 1.4e48). It is a function of the types alone, so the choice is made when the method
is compiled.
"""
@inline _evaluation_type(f, ::Type{R}, ::Val{K}) where {R, K} = R
@inline _evaluation_type(f, ::Type{Float32}, ::Val{K}) where {K} = K > _float32_max_order(typeof(f)) ? Float64 : Float32

"""
    _float32_max_order(F)

The largest derivative order at which the functional type `F`, built for Float32 and given Float32
inputs, is evaluated in Float32: the largest order, measured, at which every output stays within
its bound in pure Float32. The default is 2; a functional measured to stay accurate at higher
orders, or not at order 2, defines a method (an `XC` defines 1).
"""
_float32_max_order(::Type{<:AbstractFunctional}) = 2

# The derivatives of `f` at the inputs `x`, evaluated in S = `_evaluation_type(f, R, k)` with the
# functional converted to S, and returned in the result type R. Converting an output beyond
# floatmax(R) gives the infinity of its sign. `Vararg{Any, N}` makes the method specialize on the
# inputs: Julia does not specialize a bare `x...` that is only passed on to other calls, and the
# unspecialized call boxes its arguments.
@inline _evaluate(f, k::Val, x::Vararg{Any, N}) where {N} = _outputs_in(_result_type(f, x...), _evaluate_unrounded(f, k, x...))

# The derivatives of `f` at the inputs `x` in its evaluation type S = `_evaluation_type(f, R, k)`,
# not rounded to the result type R.
@inline function _evaluate_unrounded(f, k::Val, x::Vararg{Any, N}) where {N}
    R = _result_type(f, x...)
    return _derivatives_at(_functional_in(f, R, _evaluation_type(f, R, k)), k, x...)
end

@inline _functional_in(f, ::Type{R}, ::Type{R}) where {R} = f
@inline _functional_in(f, ::Type, ::Type{S}) where {S} = _with_number_type(f, S)

# The outputs `r` with every entry converted to R; an entry already of type R is unchanged. Field by
# field in generated code: AllocCheck cannot prove `map` over a large NamedTuple allocation-free
# (45 fields, polarized pbe_c at order 8).
@generated function _outputs_in(::Type{R}, r::NamedTuple{names}) where {R, names}
    return :(NamedTuple{names}($(Expr(:tuple, (:(_output_in(R, r.$n)) for n in names)...))))
end
@inline _output_in(::Type{R}, x::Real) where {R} = convert(R, x)
@inline _output_in(::Type{R}, t::Tuple) where {R} = ntuple(i -> convert(R, t[i]), Val(length(t)))

"""
    _with_number_type(f, S)

The functional `f` with its parameters and thresholds converted to the number type `S`. Each
concrete functional defines a method.
"""
function _with_number_type end

# The derivatives of `f` at the inputs, in the result type of `f` and the inputs.
@inline function _derivatives_at(f::AbstractLDA, ::Val{K}, ρ::Real) where {K}
    n = convert(_result_type(f, ρ), ρ)
    return _derive(f, Val(K), Val(1), Val(0), (n,), _seed_scales(f, Val(K), n))
end

@inline function _derivatives_at(f::AbstractLDA, ::Val{K}, ρ::NTuple{2, Real}) where {K}
    R = _result_type(f, ρ)
    x = (convert(R, ρ[1]), convert(R, ρ[2]))
    return _derive(f, Val(K), Val(2), Val(0), x, _seed_scales(f, Val(K), x))
end

@inline function _derivatives_at(f::AbstractGGA, ::Val{K}, ρ::Real, σ::Real) where {K}
    R = _result_type(f, ρ, σ)
    n, g = convert(R, ρ), convert(R, σ)
    return _derive(f, Val(K), Val(1), Val(1), (n, g), _seed_scales(f, Val(K), n, g))
end

@inline function _derivatives_at(f::AbstractGGA, ::Val{K}, ρ::NTuple{2, Real}, σ::NTuple{3, Real}) where {K}
    R = _result_type(f, ρ, σ)
    r = (convert(R, ρ[1]), convert(R, ρ[2]))
    g = (convert(R, σ[1]), convert(R, σ[2]), convert(R, σ[3]))
    return _derive(f, Val(K), Val(2), Val(3), (r..., g...), _seed_scales(f, Val(K), r, g))
end
"""
    xc_derivatives(f, Val(k), ρ)
    xc_derivatives(f, Val(k), ρ, σ)

The energy per volume `e` of the functional `f` and all its partial derivatives up to order `k`,
as a `NamedTuple`. A restricted input is a number ρ (and σ for a GGA); a polarized input is
ρ = (ρα, ρβ) (and σ = (σαα, σαβ, σββ)). Order j ≥ 1 has one field per split a + b = j with a
descending, named `v[j]rho[a]sigma[b]` with exponents shown above 1 (`vrho`, `v2rhosigma`,
`v3rho2sigma`). Restricted fields are numbers. Polarized fields are tuples over the density
multisets of size a in lexicographic order over (α, β), each followed by the σ multisets of size b
in lexicographic order over (αα, αβ, ββ). The result type is the promotion of the functional's
number type and the input types. Invalid inputs throw an `ArgumentError`.
"""
function xc_derivatives(f::AbstractLDA, k::Val, ρ::Union{Real, NTuple{2, Real}})
    _check_inputs(f, k, ρ)
    return _xc_derivatives_unchecked(f, k, ρ)
end

function xc_derivatives(f::AbstractGGA, k::Val, ρ::Real, σ::Real)
    _check_inputs(f, k, ρ, σ)
    return _xc_derivatives_unchecked(f, k, ρ, σ)
end

function xc_derivatives(f::AbstractGGA, k::Val, ρ::NTuple{2, Real}, σ::NTuple{3, Real})
    _check_inputs(f, k, ρ, σ)
    return _xc_derivatives_unchecked(f, k, ρ, σ)
end

"""
    energy_density(f, ρ[, σ])

The energy per volume `e` of the functional `f`; the inputs are those of `xc_derivatives`. For a
functional whose number type is a `Taylor` (a parameter carried as a series variable), `e` is a
`Taylor` in that parameter.
"""
energy_density(f::AbstractFunctional, x...) = xc_derivatives(f, Val(0), x...).e

# A functional whose number type is a `Taylor` carries a parameter as a series variable. Its
# energy density is evaluated on the plain input values, so the result is a `Taylor` in the
# parameter. Derivatives in the inputs would need the parameter's and the inputs' variables in
# one series, which the driver does not build.
@noinline _throw_taylor_parameter() = throw(
    ArgumentError(
        "xc: xc_derivatives needs a functional whose number type is not a Taylor; " *
            "energy_density accepts one"
    )
)

function _plain_energy(f, x::Vararg{Any, N}) where {N}
    e = _energy_density(f, x...)
    return ifelse(_cutoff(f, x...), zero(e), e)
end

function energy_density(f::AbstractLDA{<:Taylor}, ρ::Union{Real, NTuple{2, Real}})
    _check_inputs(f, Val(0), ρ)
    return _plain_energy(f, ρ)
end
function energy_density(f::AbstractGGA{<:Taylor}, ρ::Real, σ::Real)
    _check_inputs(f, Val(0), ρ, σ)
    return _plain_energy(f, ρ, σ)
end
function energy_density(f::AbstractGGA{<:Taylor}, ρ::NTuple{2, Real}, σ::NTuple{3, Real})
    _check_inputs(f, Val(0), ρ, σ)
    return _plain_energy(f, ρ, σ)
end

_xc_derivatives_unchecked(::AbstractLDA{<:Taylor}, ::Val, ::Real) = _throw_taylor_parameter()
_xc_derivatives_unchecked(::AbstractLDA{<:Taylor}, ::Val, ::NTuple{2, Real}) = _throw_taylor_parameter()
_xc_derivatives_unchecked(::AbstractGGA{<:Taylor}, ::Val, ::Real, ::Real) = _throw_taylor_parameter()
_xc_derivatives_unchecked(::AbstractGGA{<:Taylor}, ::Val, ::NTuple{2, Real}, ::NTuple{3, Real}) = _throw_taylor_parameter()
