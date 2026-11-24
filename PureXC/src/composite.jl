# Sums of weighted functionals, such as PBE exchange plus PBE correlation.

"""
    XC(terms)

The functional Σ wᵢ·fᵢ over `terms = ((w₁, f₁), (w₂, f₂), …)`, with real weights wᵢ. All terms
belong to one family (LDA or GGA). `energy_density`, `xc_derivatives` and `xc_evaluate!` sum the
weighted terms. Each term is evaluated by its own driver in its own evaluation type
(`_evaluation_type`), or in Float64 when the XC is Float32 above its `_float32_max_order`; the
weighted terms are summed in the widest of those types, and each output is rounded once to the
result type.
"""
struct XC{T, TS <: Tuple} <: AbstractFunctional{T}
    terms::TS
    function XC{T, TS}(terms) where {T, TS <: Tuple}
        isempty(terms) && throw(ArgumentError("xc: an XC needs at least one term"))
        fams = map(t -> family(t[2]), terms)
        all(==(first(fams)), fams) || throw(ArgumentError(lazy"xc: the terms of an XC must share one family, got $fams"))
        return new{T, TS}(terms)
    end
end
XC(terms::Tuple{Vararg{Tuple{Real, AbstractFunctional}}}) =
    XC{promote_type(map(t -> _number_type(t[2]), terms)...), typeof(terms)}(terms)

"""
    pbe(T = Float64)

PBE exchange plus PBE correlation, Libxc's `gga_x_pbe` + `gga_c_pbe`, in the number type `T`.
"""
pbe(::Type{T} = Float64) where {T} = XC(((1, pbe_x(T)), (1, pbe_c(T))))

family(xc::XC) = family(xc.terms[1][2])
kind(::XC) = :exchange_correlation
_with_number_type(xc::XC, ::Type{S}) where {S} = XC(map(t -> (t[1], _with_number_type(t[2], S)), xc.terms))
# An XC is evaluated in Float32 to order 2, the default, and every term in Float64 above it. The
# orders of its terms do not bound it: their outputs can cancel. Measured for pbe(Float32), K32 = 2:
# at order 3, where pbe_x and pbe_c are each evaluated in Float32, an entry of each term exceeds
# floatmax(Float32) with opposite signs while their sum fits, so the Float32 sum is Inf - Inf = NaN
# or an infinity (823 entries restricted, 8986 polarized on the Libxc oracle grid).
_float32_max_order(::Type{<:XC}) = 2

function _check_inputs(xc::XC, k::Val, x::Vararg{Any, N}) where {N}
    foreach(t -> _check_inputs(t[2], k, x...), xc.terms)
    return nothing
end
@inline _inputs_ok(xc::XC, k::Val, x::Vararg{Any, N}) where {N} = _terms_ok(xc.terms, k, x...)
@inline _terms_ok(::Tuple{}, k, x::Vararg{Any, N}) where {N} = true
@inline _terms_ok(terms::Tuple, k, x::Vararg{Any, N}) where {N} = _inputs_ok(terms[1][2], k, x...) & _terms_ok(Base.tail(terms), k, x...)

# Σ wᵢ·rᵢ over the NamedTuples rᵢ of the terms, field by field.
@generated function _weighted(w::Real, r::NamedTuple{names}) where {names}
    return :(NamedTuple{$names}($(Expr(:tuple, (:(_scale(w, r.$n)) for n in names)...))))
end
@generated function _sum(a::NamedTuple{names}, b::NamedTuple{names}) where {names}
    return :(NamedTuple{$names}($(Expr(:tuple, (:(_plus(a.$n, b.$n)) for n in names)...))))
end
_scale(w, x::Real) = oftype(x, w) * x
_scale(w, x::Tuple) = ntuple(i -> _scale(w, x[i]), Val(length(x)))
_plus(x::Real, y::Real) = x + y
_plus(x::Tuple, y::Tuple) = ntuple(i -> x[i] + y[i], Val(length(x)))

# The weighted sum of the terms, each evaluated in its own evaluation type and not rounded to the
# result type: `_plus` promotes, so the sum is formed in the widest evaluation type of the terms.
# Rounding each term to the result type first would turn two terms beyond floatmax(Float32) with
# opposite signs into Inf - Inf = NaN, where their sum may be finite.
@inline _term_sum(terms::Tuple{Any}, k, x::Vararg{Any, N}) where {N} = _weighted(terms[1][1], _term_outputs(terms[1][2], k, x...))
@inline function _term_sum(terms::Tuple, k, x::Vararg{Any, N}) where {N}
    head = _weighted(terms[1][1], _term_outputs(terms[1][2], k, x...))
    return _sum(head, _term_sum(Base.tail(terms), k, x...))
end
@inline _term_outputs(f, k::Val, x::Vararg{Any, N}) where {N} = _evaluate_unrounded(f, k, x...)
_term_outputs(::AbstractFunctional{<:Taylor}, ::Val, ::Vararg{Any, N}) where {N} = _throw_taylor_parameter()

# The terms in the evaluation type S of the XC: unchanged where S is the result type R, each
# converted to S otherwise (Float64 for a Float32 XC above `_float32_max_order`).
@inline function _xc_derivatives_unchecked(xc::XC, k::Val, x::Vararg{Any, N}) where {N}
    R = _result_type(xc, x...)
    return _outputs_in(R, _term_sum(_terms_in(xc, R, _evaluation_type(xc, R, k)), k, x...))
end
@inline _terms_in(xc::XC, ::Type{R}, ::Type{R}) where {R} = xc.terms
@inline _terms_in(xc::XC, ::Type, ::Type{S}) where {S} = map(t -> (t[1], _with_number_type(t[2], S)), xc.terms)

function xc_derivatives(xc::XC, k::Val, x::Vararg{Any, N}) where {N}
    _check_inputs(xc, k, x...)
    return _xc_derivatives_unchecked(xc, k, x...)
end

# An XC with a term whose number type is a `Taylor` (its own number type is then that `Taylor` or,
# mixed with plain terms, `Real`): Σ wᵢ·eᵢ over the energy densities of the terms, each a `Taylor`
# in the parameters of its own term or a plain value. A weight scales a `Taylor` in its coefficient
# type, which `*` converts it to. Every other XC sums its terms as `xc_derivatives` does.
function energy_density(xc::XC, x::Vararg{Any, N}) where {N}
    any(_taylor_term, xc.terms) || return xc_derivatives(xc, Val(0), x...).e
    return _term_energy(xc.terms, x...)
end
_taylor_term(t::Tuple) = t[2] isa AbstractFunctional{<:Taylor}
_term_energy(terms::Tuple{Any}, x::Vararg{Any, N}) where {N} = _scale_energy(terms[1][1], energy_density(terms[1][2], x...))
_term_energy(terms::Tuple, x::Vararg{Any, N}) where {N} =
    _term_energy((terms[1],), x...) + _term_energy(Base.tail(terms), x...)
_scale_energy(w, e::Real) = _scale(w, e)
_scale_energy(w, e::Taylor) = e * w

"""
    zk(e, ρ)

The energy per particle `e/ρ`, or zero where the density ρ is zero. For a polarized density
`(ρα, ρβ)` the total density is used.
"""
zk(e::Real, ρ::Real) = iszero(ρ) ? zero(e / one(ρ)) : e / ρ
zk(e::Real, ρ::Tuple) = zk(e, sum(ρ))
