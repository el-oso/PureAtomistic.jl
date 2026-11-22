# The array layer: a functional evaluated at every point of an array of inputs.
#
# Restricted inputs are vectors (ρ[, σ]). Polarized inputs are matrices with the spin on dimension
# 1: ρ is 2 × n and σ is 3 × n, in the order (α, β) and (αα, αβ, ββ). An output field is a matrix
# whose first dimension has one row per component of the field (1 for a scalar field) and whose
# last axis is the point axis of the inputs.

@inline function _point(x::AbstractMatrix, j, ::Val{N}) where {N}
    i = first(axes(x, 1))
    return ntuple(t -> @inbounds(x[i + t - 1, j]), Val(N))
end

# The loads and stores of the point loop use `@inbounds`: its indices come from `_points(ρ)`, and
# `_check_axes` and `_check_out` have established beforehand that every input and output field has
# those points (and the row counts the loop reads). A bounds check in the loop is an early exit that
# keeps LLVM from vectorizing it.

# The points of an input: its indices restricted, its columns polarized.
_points(x::AbstractVector) = eachindex(x)
_points(x::AbstractMatrix) = axes(x, 2)

# The inputs of point j as the arguments of `_xc_derivatives_unchecked`.
@inline _inputs(ρ::AbstractVector, j) = (@inbounds(ρ[j]),)
@inline _inputs(ρ::AbstractMatrix, j) = (_point(ρ, j, Val(2)),)
@inline _inputs(ρ::AbstractVector, σ::AbstractVector, j) = (@inbounds(ρ[j]), @inbounds(σ[j]))
@inline _inputs(ρ::AbstractMatrix, σ::AbstractMatrix, j) = (_point(ρ, j, Val(2)), _point(σ, j, Val(3)))

@noinline _throw_axes(msg) = throw(DimensionMismatch(msg))

function _check_axes(ρ::AbstractVector)
    return nothing
end
function _check_axes(ρ::AbstractMatrix)
    size(ρ, 1) == 2 || _throw_axes(lazy"xc: a polarized ρ has two rows (α, β), got $(size(ρ, 1))")
    return nothing
end
function _check_axes(ρ::AbstractVector, σ::AbstractVector)
    axes(ρ) == axes(σ) || _throw_axes(lazy"xc: ρ and σ must have the same axes: $(axes(ρ)) vs $(axes(σ))")
    return nothing
end
function _check_axes(ρ::AbstractMatrix, σ::AbstractMatrix)
    _check_axes(ρ)
    size(σ, 1) == 3 || _throw_axes(lazy"xc: a polarized σ has three rows (αα, αβ, ββ), got $(size(σ, 1))")
    axes(ρ, 2) == axes(σ, 2) || _throw_axes(lazy"xc: ρ and σ must have the same points: $(axes(ρ, 2)) vs $(axes(σ, 2))")
    return nothing
end
function _check_axes(ρ::AbstractArray, σ::AbstractArray)
    _throw_axes("xc: ρ and σ must both be vectors (restricted) or both be matrices (polarized)")
end

_ncomp(::Type{<:Real}) = 1
_ncomp(::Type{T}) where {T <: Tuple} = fieldcount(T)

# The row counts of a polarized ρ and σ are checked by `_check_axes`.
_point_input_type(x::AbstractVector, ::Val) = eltype(x)
_point_input_type(x::AbstractMatrix, ::Val{N}) where {N} = NTuple{N, eltype(x)}
_point_input_types(ρ) = (_point_input_type(ρ, Val(2)),)
_point_input_types(ρ, σ) = (_point_input_type(ρ, Val(2)), _point_input_type(σ, Val(3)))

# The result type of one point: a `NamedTuple` type.
function _point_type(f, k::Val, x::Vararg{AbstractArray, N}) where {N}
    R = Base.promote_op(_xc_derivatives_unchecked, typeof(f), typeof(k), _point_input_types(x...)...)
    isconcretetype(R) || throw(ArgumentError(lazy"xc: the result type $R of $(typeof(f)) is not concrete"))
    return R
end

# Each output field is a matrix with one row per component and the points of the inputs as columns.
function _check_out(out::NamedTuple{names}, ::Type{R}, ρ::AbstractArray) where {names, R <: NamedTuple}
    fieldnames(R) == names || throw(ArgumentError(lazy"xc: out has the fields $names, which differ from those of the derivatives $(fieldnames(R))"))
    V = fieldtypes(R)
    pts = _points(ρ)
    ntuple(i -> _check_field(names[i], out[i], V[i], ρ, pts), Val(length(names)))
    return nothing
end

function _check_field(n, a, ::Type{T}, ρ, pts) where {T}
    c = _ncomp(T)
    a isa AbstractMatrix || _throw_axes(lazy"xc: output $n must be a matrix, got $(typeof(a))")
    size(a, 1) == c || _throw_axes(lazy"xc: output $n has $(size(a, 1)) rows, the field has $c components")
    axes(a, 2) == pts || _throw_axes(lazy"xc: output $n has point axis $(axes(a, 2)), the inputs have $pts")
    return nothing
end

@inline _store!(a::AbstractMatrix, v::Real, j) = (@inbounds(a[first(axes(a, 1)), j] = v); nothing)
@inline function _store!(a::AbstractMatrix, v::Tuple, j)
    i = first(axes(a, 1))
    for t in eachindex(v)
        @inbounds a[i + t - 1, j] = v[t]
    end
    return nothing
end

# Field by field in generated code, so that no `map` over a large NamedTuple is compiled.
@generated function _store_all!(out::NamedTuple{names}, r::NamedTuple{names}, j) where {names}
    return Expr(:block, (:(_store!(out.$n, r.$n, j)) for n in names)..., :(return nothing))
end

function _validate(f, k::Val, x::Vararg{AbstractArray, N}) where {N}
    for j in _points(x[1])
        _check_inputs(f, k, _inputs(x..., j)...)
    end
    return nothing
end

"""
    xc_evaluate!(out, f, Val(k), ρ)
    xc_evaluate!(out, f, Val(k), ρ, σ)

Write the energy per volume of the functional `f` and its derivatives to order `k`
(`xc_derivatives`) at every point of the arrays ρ (and σ) into the arrays of the `NamedTuple`
`out`, which has exactly the fields of `xc_derivatives`. Restricted inputs are vectors; polarized
inputs are a `2 × n` matrix ρ and a `3 × n` matrix σ with the spin on dimension 1. Each output
field is a matrix with one row per component of the field (see `xc_derivatives`; one row for a
scalar field) and the point axis of the inputs as its columns. All inputs are validated before the first
output is written; an invalid input throws the `ArgumentError` of `xc_derivatives`, and
mismatched axes throw a `DimensionMismatch`.
"""
function xc_evaluate!(out::NamedTuple, f::AbstractFunctional, k::Val, ρ::AbstractArray, σ::Vararg{AbstractArray, M}) where {M}
    _check_axes(ρ, σ...)
    _check_out(out, _point_type(f, k, ρ, σ...), ρ)
    _validate(f, k, ρ, σ...)
    for j in _points(ρ)
        _store_all!(out, _xc_derivatives_unchecked(f, k, _inputs(ρ, σ..., j)...), j)
    end
    return out
end

"""
    xc_outputs(f, Val(k), ρ)
    xc_outputs(f, Val(k), ρ, σ)

The `NamedTuple` of arrays that `xc_evaluate!` fills, allocated with `similar` so that the array
type and the point axis follow the inputs.
"""
function xc_outputs(f::AbstractFunctional, k::Val, ρ::AbstractArray, σ::Vararg{AbstractArray, M}) where {M}
    _check_axes(ρ, σ...)
    return _outputs_like(_point_type(f, k, ρ, σ...), ρ)
end

@generated function _outputs_like(::Type{NamedTuple{names, V}}, ρ::AbstractArray) where {names, V}
    fields = map(1:length(names)) do i
        t = fieldtype(V, i)
        R = t <: Tuple ? eltype(t) : t
        c = _ncomp(t)
        :(similar(ρ, $R, (Base.OneTo($c), axes(ρ, ndims(ρ)))))
    end
    return :(NamedTuple{$names}($(Expr(:tuple, fields...))))
end
