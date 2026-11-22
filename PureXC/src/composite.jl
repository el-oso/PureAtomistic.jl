# Sums of weighted functionals, such as PBE exchange plus PBE correlation.

"""
    XC(terms)

The functional Σ wᵢ·fᵢ over `terms = ((w₁, f₁), (w₂, f₂), …)`, with real weights wᵢ. All terms
belong to one family (LDA or GGA). `energy_density`, `xc_derivatives` and `xc_evaluate!` sum the
weighted terms; each term is evaluated by its own driver, in its own precision.
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
_float32_max_order(::Type{XC{T, TS}}) where {T, TS} =
    minimum(i -> _float32_max_order(fieldtype(fieldtype(TS, i), 2)), 1:fieldcount(TS))

function _check_inputs(xc::XC, k::Val, x::Vararg{Any, N}) where {N}
    foreach(t -> _check_inputs(t[2], k, x...), xc.terms)
    return nothing
end

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

_term_sum(terms::Tuple{Any}, k, x::Vararg{Any, N}) where {N} = _weighted(terms[1][1], _xc_derivatives_unchecked(terms[1][2], k, x...))
function _term_sum(terms::Tuple, k, x::Vararg{Any, N}) where {N}
    head = _weighted(terms[1][1], _xc_derivatives_unchecked(terms[1][2], k, x...))
    return _sum(head, _term_sum(Base.tail(terms), k, x...))
end

_xc_derivatives_unchecked(xc::XC, k::Val, x::Vararg{Any, N}) where {N} = _term_sum(xc.terms, k, x...)

function xc_derivatives(xc::XC, k::Val, x::Vararg{Any, N}) where {N}
    _check_inputs(xc, k, x...)
    return _xc_derivatives_unchecked(xc, k, x...)
end

"""
    zk(e, ρ)

The energy per particle `e/ρ`, or zero where the density ρ is zero. For a polarized density
`(ρα, ρβ)` the total density is used.
"""
zk(e::Real, ρ::Real) = iszero(ρ) ? zero(e / one(ρ)) : e / ρ
zk(e::Real, ρ::Tuple) = zk(e, sum(ρ))
