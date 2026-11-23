# The array layer: a functional evaluated at every point of an array of inputs.
#
# Restricted inputs are vectors (ρ[, σ]). Polarized inputs are matrices with the spin on dimension
# 1: ρ is 2 × n and σ is 3 × n, in the order (α, β) and (αα, αβ, ββ). An output field is a matrix
# whose first dimension has one row per component of the field (1 for a scalar field) and whose
# last axis is the point axis of the inputs.

@inline function _point(x::AbstractMatrix, j, ::Val{N}) where {N}
    i = first(axes(x, 1))
    return ntuple(t -> x[i + t - 1, j], Val(N))
end

# The restricted loads and all stores of the point loop use `@inbounds`: its indices come from
# `_points(ρ)`, and `_check_axes` and `_check_out` have established beforehand that every input and
# output field has those points (and the row counts the loop reads). A bounds check in the loop is an
# early exit that keeps LLVM from vectorizing it. The polarized loads (`_point`) keep their checks:
# measured on Apple M-series, Julia 1.13, 4096 points, `@inbounds` there makes polarized pbe_x at
# order 1 slower (36.6 against 29.1 ns per point) and changes lda_x and pbe_x at orders 0 to 2 by at
# most 2%. On the tuple stores it makes polarized lda_x at order 1 faster (4.0 against 4.6 ns).

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

# Every point is first tested by the branch-free `_inputs_ok`, which vectorizes; only when a point
# fails are the points checked again in order by `_check_inputs`, which throws at the first invalid
# one. `_inputs_ok` is the negation of the throwing conditions of `_check_inputs`, so the second
# pass always throws; if it does not, the two disagree and that is an error of the package.
function _validate(f, k::Val, x::Vararg{AbstractArray, N}) where {N}
    _all_inputs_ok(f, k, x...) && return nothing
    for j in _points(x[1])
        _check_inputs(f, k, _inputs(x..., j)...)
    end
    throw(ErrorException("xc: _inputs_ok rejects a point that _check_inputs accepts"))
end

function _all_inputs_ok(f, k::Val, x::Vararg{AbstractArray, N}) where {N}
    ok = true
    for j in _points(x[1])
        ok &= _inputs_ok(f, k, _inputs(x..., j)...)
    end
    return ok
end

# Dense inputs read through raw pointers, as in the dense point loop; the same checks have
# established the index range.
function _all_inputs_ok(f, k::Val, ρ::Array{<:Base.IEEEFloat}, σ::Vararg{Array{<:Base.IEEEFloat}, M}) where {M}
    ok = true
    n = size(ρ, ndims(ρ))
    GC.@preserve ρ σ begin
        px = map(pointer, (ρ, σ...))
        @simd for j in Base.OneTo(n)
            ok &= _inputs_ok(f, k, _dense_inputs(px, (ρ, σ...), j)...)
        end
    end
    return ok
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

No output may share memory with an input or with another output: if one does, `xc_evaluate!`
throws `ArgumentError("xc: an output shares memory with an input or another output")` before
writing anything. Every argument is resolved through `parent` to its root array. When the roots are
`Array`s or `Memory`s of a bits type, arrays whose roots occupy disjoint address ranges are
accepted, and arrays over one storage are accepted only when the test is exact: `Array`s and views
of an `Array` indexed by integers and ranges whose address spans are disjoint, or views of the same
`Array` whose integer or range indices select disjoint sets in some dimension (an `Array` stores
each index at its own address), so disjoint rows or column blocks of one matrix are accepted. Any
other array over the same storage is rejected, even when its elements are disjoint: a `reshape`,
`transpose`, `OffsetArray` or other wrapper of one part of a buffer against another part, and views
of one `Array` that differ only in vector or logical indices. When a root is not an `Array` or
`Memory` of a bits type, the roots are tested with `Base.mightalias`; only in this case does the
test rely on the `Base.dataids` of the array type.
"""
function xc_evaluate!(out::NamedTuple, f::AbstractFunctional, k::Val, ρ::AbstractArray, σ::Vararg{AbstractArray, M}) where {M}
    _check_axes(ρ, σ...)
    _check_out(out, _point_type(f, k, ρ, σ...), ρ)
    _check_no_overlap(values(out), (ρ, σ...))
    _validate(f, k, ρ, σ...)
    _evaluate_points!(out, f, k, ρ, σ...)
    return out
end

# The point loop, after the axes, the outputs and the inputs have been checked.
_evaluate_points!(out::NamedTuple, f, k::Val, x::Vararg{AbstractArray, N}) where {N} = _evaluate_points_checked!(out, f, k, x...)

# The point loop with checked loads.
function _evaluate_points_checked!(out::NamedTuple, f, k::Val, ρ::AbstractArray, σ::Vararg{AbstractArray, M}) where {M}
    for j in _points(ρ)
        _store_all!(out, _xc_derivatives_unchecked(f, k, _inputs(ρ, σ..., j)...), j)
    end
    return nothing
end

# The dense point loop: inputs and outputs are `Array`s of IEEE floats, read and written through
# raw pointers under `@simd ivdep`, which lets LLVM vectorize the polarized loop (its checked loads
# keep it scalar). The iterations are independent: `_check_no_overlap` has established that no
# output shares memory with an input or with another output. Every index is in range: `_check_axes`
# and `_check_out` have established that restricted inputs are vectors, a polarized ρ has 2 rows
# and σ 3, and each output field has `c` rows (its number of components), all over the same
# n = size(ρ, ndims(ρ)) points; an `Array` stores column j of a c-row matrix at the linear indices
# c(j − 1) + 1 to c·j.
function _evaluate_points!(
        out::NamedTuple{names, <:Tuple{Vararg{Matrix{<:Base.IEEEFloat}}}}, f, k::Val,
        ρ::Array{<:Base.IEEEFloat}, σ::Vararg{Array{<:Base.IEEEFloat}, M}
    ) where {names, M}
    n = size(ρ, ndims(ρ))
    GC.@preserve out ρ σ begin
        po = map(pointer, values(out))
        px = map(pointer, (ρ, σ...))
        @simd ivdep for j in Base.OneTo(n)
            _store_all_dense!(po, _xc_derivatives_unchecked(f, k, _dense_inputs(px, (ρ, σ...), j)...), j)
        end
    end
    return nothing
end

# Whether an output shares memory with an input or with another output, checked pair by pair. Each
# array is first resolved through `parent`, repeatedly, to its root: the array that is its own parent,
# which holds the elements of every wrapper and view above it.
#
# - When both roots are `Array`s or `Memory`s of a bits type, each root's storage is the address
#   range [pointer, pointer + length · elsize). Roots whose ranges do not intersect share no memory.
#   When they intersect, the pair is disjoint only if one of two exact tests proves it:
#   - both arrays are strided memory of an `Array` (the `Array` itself, or a view of an `Array`
#     indexed by integers and ranges, where every element's address follows from `pointer` and
#     `strides`) and their address spans, from the lowest to the highest address of an element, do
#     not intersect;
#   - both are views of the same `Array` object with the same number of indices, and in some
#     dimension both indices are integers or ranges of integers that select disjoint sets. An
#     `Array` stores each index at its own address, so disjoint index sets select disjoint memory.
#     Through any other parent (a `reshape`, a `PermutedDimsArray`, a view with repeated indices)
#     several indices may reach one address, so index sets prove nothing there.
#   Every other pair whose root ranges intersect counts as sharing memory. Some disjoint arrays are
#   therefore rejected: a `reshape`, `transpose`, `adjoint`, `OffsetArray` or other wrapper of one
#   part of a buffer against another part of it, and views of one `Array` whose index sets differ
#   only in dimensions indexed by vectors of integers or logical masks.
# - When either root is not an `Array` or `Memory` of a bits type, `Base.mightalias` of the two
#   roots, which relies on the `Base.dataids` of their types; this is the only case that does.
#
# An empty array shares memory with nothing.
@noinline _throw_overlap() = throw(ArgumentError("xc: an output shares memory with an input or another output"))

function _check_no_overlap(outs::Tuple, ins::Tuple)
    _any_overlap(outs, ins) && _throw_overlap()
    return nothing
end

_any_overlap(::Tuple{}, ins::Tuple) = false
_any_overlap(outs::Tuple, ins::Tuple) =
    _overlaps_any(first(outs), ins) || _overlaps_any(first(outs), Base.tail(outs)) || _any_overlap(Base.tail(outs), ins)

_overlaps_any(a, ::Tuple{}) = false
_overlaps_any(a, xs::Tuple) = _overlaps(a, first(xs)) || _overlaps_any(a, Base.tail(xs))

_overlaps(a::AbstractArray, b::AbstractArray) = !(isempty(a) || isempty(b)) && _shares(a, b)

function _shares(a, b)
    ra, rb = _root(a), _root(b)
    sa, sb = _extent(ra), _extent(rb)
    (isnothing(sa) || isnothing(sb)) && return Base.mightalias(ra, rb)
    _spans_intersect(sa, sb) || return false
    _spans_disjoint(a, b) && return false
    return !_views_disjoint(a, b)
end

# The array that is its own parent, reached through `parent`.
_root(a) = _root(a, parent(a))
_root(a::T, p::T) where {T} = p === a ? a : _root(p)
_root(a, p) = _root(p)
_root(a::Union{Array, Memory}) = a

# The address range [lo, hi) of the storage of an `Array` or `Memory` of a bits type, or `nothing`;
# `sizeof` of either is its length times its element size.
_extent(a) = nothing
function _extent(a::Union{Array, Memory})
    isbitstype(eltype(a)) || return nothing
    p = UInt(pointer(a))
    return (p, p + UInt(sizeof(a)))
end

# The address span [lo, hi) of the elements of strided `Array` memory of a bits type, or `nothing`.
_span(a) = nothing
_span(a::Array) = _extent(a)
function _span(a::SubArray{<:Any, <:Any, <:Array, <:Tuple{Vararg{Union{Int, AbstractRange{Int}}}}})
    isbitstype(eltype(a)) || return nothing
    es = UInt(Base.elsize(parent(a)))
    lo = hi = UInt(pointer(a))
    for (n, s) in zip(size(a), strides(a))
        d = UInt(n - 1) * UInt(abs(s)) * es
        s < 0 ? (lo -= d) : (hi += d)
    end
    return (lo, hi + es)
end

_spans_intersect(sa::Tuple, sb::Tuple) = sa[1] < sb[2] && sb[1] < sa[2]
function _spans_disjoint(a, b)
    sa, sb = _span(a), _span(b)
    return !(isnothing(sa) || isnothing(sb)) && !_spans_intersect(sa, sb)
end

# Whether a and b are views of the same `Array` that select disjoint index sets.
_views_disjoint(a, b) = false
function _views_disjoint(a::SubArray{<:Any, <:Any, <:Array}, b::SubArray{<:Any, <:Any, <:Array})
    ia, ib = parentindices(a), parentindices(b)
    return parent(a) === parent(b) && length(ia) == length(ib) && _indices_disjoint(ia, ib)
end

# Whether index tuples of equal length select disjoint sets: disjoint in at least one dimension.
_indices_disjoint(i::Tuple{Any, Vararg}, j::Tuple{Any, Vararg}) =
    _disjoint(first(i), first(j)) || _indices_disjoint(Base.tail(i), Base.tail(j))
_indices_disjoint(::Tuple, ::Tuple) = false

const _IntIndex = Union{Integer, AbstractRange{<:Integer}}
_disjoint(i, j) = false
_disjoint(i::_IntIndex, j::_IntIndex) = isempty(intersect(_steps(i), _steps(j)))
_steps(i::Integer) = StepRange{Int, Int}(i, 1, i)
_steps(r::AbstractRange{<:Integer}) = StepRange{Int, Int}(first(r), step(r), last(r))

# The inputs of point j through the pointers p of the inputs: numbers of vectors (restricted),
# columns of a 2-row ρ and a 3-row σ (polarized).
@inline _dense_inputs(p::Tuple, ::Tuple{Vector}, j) = (unsafe_load(p[1], j),)
@inline _dense_inputs(p::Tuple, ::Tuple{Vector, Vector}, j) = (unsafe_load(p[1], j), unsafe_load(p[2], j))
@inline _dense_inputs(p::Tuple, ::Tuple{Matrix}, j) = (_dense_column(p[1], Val(2), j),)
@inline _dense_inputs(p::Tuple, ::Tuple{Matrix, Matrix}, j) = (_dense_column(p[1], Val(2), j), _dense_column(p[2], Val(3), j))
@inline _dense_column(p::Ptr, ::Val{N}, j) where {N} = ntuple(t -> unsafe_load(p, N * (j - 1) + t), Val(N))

@inline _store_dense!(p::Ptr, v::Real, j) = (unsafe_store!(p, v, j); nothing)
@inline function _store_dense!(p::Ptr, v::Tuple, j)
    c = length(v)
    for t in eachindex(v)
        unsafe_store!(p, v[t], c * (j - 1) + t)
    end
    return nothing
end

@generated function _store_all_dense!(p::Tuple, r::NamedTuple{names}, j) where {names}
    return Expr(:block, Expr(:meta, :inline), (:(_store_dense!(p[$i], r.$(names[i]), j)) for i in eachindex(names))..., :(return nothing))
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
