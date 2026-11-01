"""
    checked_eachindex(A, Bs...)

`eachindex(A, Bs...)` with a mismatch error that `juliac --trim` resolves.
"""
@inline function checked_eachindex(A, Bs::Vararg{Any, N}) where {N}
    style = IndexStyle(A, Bs...)
    inds = eachindex(style, A)
    _check_eachindex(style, inds, Bs...)
    return inds
end

_check_eachindex(style, inds) = nothing

@inline function _check_eachindex(style, inds, B, Bs::Vararg{Any, N}) where {N}
    # One argument at a time, and never a splat into the message: `Base.eachindex` splats every
    # argument into its message, and that splat is a dynamic call in the `Vararg` specialization
    # a precompiled caller refers to.
    indsB = eachindex(style, B)
    indsB == inds || _throw_eachindex_mismatch(inds, indsB)
    return _check_eachindex(style, inds, Bs...)
end

@noinline _throw_eachindex_mismatch(inds, indsB) =
    throw(DimensionMismatch(lazy"all inputs to eachindex must have the same indices, got $inds and $indsB"))
