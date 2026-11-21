# The functional interface. A concrete functional `f` has a field `thresholds::Thresholds{T}` and
# a method `_energy_density(f, ρ)` (LDA) or `_energy_density(f, ρ, σ)` (GGA): restricted, ρ and σ
# are `Taylor` numbers; polarized, ρ = (ρα, ρβ) and σ = (σαα, σαβ, σββ) are tuples of them. It
# returns e, the energy per volume, as a `Taylor` over the same variables. It is evaluated at
# every point, including those `_cutoff` selects, so it must not throw for nonnegative inputs.

"""
    _energy_density(f, ρ[, σ])

The energy per volume e of the functional `f`, as a `Taylor` over the variables of its `Taylor`
inputs: restricted, the numbers ρ and σ; polarized, the tuples ρ = (ρα, ρβ) and
σ = (σαα, σαβ, σββ). Each concrete functional defines a method.
"""
function _energy_density end

"""
    _cutoff(f, ρ[, σ])

Whether the PureXC rule of `f` makes the energy and every derivative zero at the plain input
values `ρ` and `σ`, given in the result type (restricted: numbers; polarized: tuples). At such a
point the result of `_energy_density` is discarded and every output is an exact zero.
"""
_cutoff(::AbstractLDA, ρ) = false
_cutoff(::AbstractGGA, ρ, σ) = false

_number_type(::AbstractFunctional{T}) where {T} = T

_result_type(f::AbstractFunctional, x...) = promote_type(_number_type(f), map(_input_type, x)...)
_input_type(x::Real) = typeof(x)
_input_type(x::Tuple) = promote_type(map(typeof, x)...)

# The variables x[i] + s[i]·u_i of a series to order K, in input order.
@generated function _seed(::Val{K}, x::NTuple{N}, s::NTuple{N}) where {K, N}
    return Expr(:tuple, (:(variable(Val($N), Val($K), Val($i), x[$i], s[$i])) for i in 1:N)...)
end

# The flat variable tuple regrouped as the functional's arguments.
_group(::Val{1}, ::Val{0}, t) = (t[1],)
_group(::Val{1}, ::Val{1}, t) = (t[1], t[2])
_group(::Val{2}, ::Val{0}, t) = ((t[1], t[2]),)
_group(::Val{2}, ::Val{3}, t) = ((t[1], t[2]), (t[3], t[4], t[5]))

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

# The derivatives to order K at the input values x with seed floors fl, grouped as nρ density and
# nσ σ inputs, mapped to the output fields.
function _derive(f, ::Val{K}, gρ::Val, gσ::Val, x::NTuple{N}, fl::NTuple{N}) where {K, N}
    # Seed scales s = max(|x|, floor) > 0, also at x = 0; |x| covers σαβ, which may be negative.
    s = map((xi, fi) -> max(abs(xi), fi), x, fl)
    e = _energy_density(f, _group(gρ, gσ, _seed(Val(K), x, s))...)
    d = derivatives(e, map(inv, s))
    cut = _cutoff(f, _group(gρ, gσ, x)...)
    return _outputs(Val(K), gρ, gσ, ntuple(i -> ifelse(cut, zero(d[i]), d[i]), Val(length(d))))
end

"""
    _xc_derivatives_unchecked(f, Val(k), ρ[, σ])

`xc_derivatives` without the input check.
"""
function _xc_derivatives_unchecked(f::AbstractLDA, ::Val{K}, ρ::Real) where {K}
    R = _result_type(f, ρ)
    t = _thresholds(f)
    return _derive(f, Val(K), Val(1), Val(0), (convert(R, ρ),), (convert(R, t.density),))
end

function _xc_derivatives_unchecked(f::AbstractLDA, ::Val{K}, ρ::NTuple{2, Real}) where {K}
    R = _result_type(f, ρ)
    t = _thresholds(f)
    n = convert(R, t.density)
    return _derive(f, Val(K), Val(2), Val(0), (convert(R, ρ[1]), convert(R, ρ[2])), (n, n))
end

function _xc_derivatives_unchecked(f::AbstractGGA, ::Val{K}, ρ::Real, σ::Real) where {K}
    R = _result_type(f, ρ, σ)
    t = _thresholds(f)
    return _derive(
        f, Val(K), Val(1), Val(1), (convert(R, ρ), convert(R, σ)),
        (convert(R, t.density), convert(R, t.sigma_floor))
    )
end

function _xc_derivatives_unchecked(f::AbstractGGA, ::Val{K}, ρ::NTuple{2, Real}, σ::NTuple{3, Real}) where {K}
    R = _result_type(f, ρ, σ)
    t = _thresholds(f)
    n, g = convert(R, t.density), convert(R, t.sigma_floor)
    x = (convert(R, ρ[1]), convert(R, ρ[2]), convert(R, σ[1]), convert(R, σ[2]), convert(R, σ[3]))
    return _derive(f, Val(K), Val(2), Val(3), x, (n, n, g, g, g))
end

_thresholds(f::AbstractFunctional) = f.thresholds

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
    _check_inputs(f, ρ)
    return _xc_derivatives_unchecked(f, k, ρ)
end

function xc_derivatives(f::AbstractGGA, k::Val, ρ::Real, σ::Real)
    _check_inputs(f, ρ, σ)
    return _xc_derivatives_unchecked(f, k, ρ, σ)
end

function xc_derivatives(f::AbstractGGA, k::Val, ρ::NTuple{2, Real}, σ::NTuple{3, Real})
    _check_inputs(f, ρ, σ)
    return _xc_derivatives_unchecked(f, k, ρ, σ)
end

"""
    energy_density(f, ρ[, σ])

The energy per volume `e` of the functional `f`; the inputs are those of `xc_derivatives`.
"""
energy_density(f::AbstractFunctional, x...) = xc_derivatives(f, Val(0), x...).e
