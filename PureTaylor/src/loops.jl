# Loop forms of the coefficient kernels. A kernel whose generated straight-line code grows with the
# number of stored coefficients unrolls only up to `_UNROLL_MAX` output coefficients; above that it
# runs a loop over index tables that are computed when the method is generated and returned as
# constant tuples, so that its compiled code has a fixed size and its stack frame holds the result
# and a scratch array of O(L) entries. Each loop form performs the same arithmetic in the same order
# as the unrolled form. The two forms therefore give identical results (`==`). The coefficient kernels
# accumulate with `fma`, not `muladd`: `muladd` lets LLVM fuse each multiply-add or not, decided at
# each call site, so the same kernel would round differently inlined, out of line, unrolled and
# looped. `fma` rounds once everywhere, which keeps every call site bit-identical.

# Measured on Apple M-series, Julia 1.13: the loop product is 6 to 9 times slower than the unrolled
# one at every size from L = 28 to L = 462, so there is no crossover in run time. The threshold is
# set by compile time: one unrolled product compiles in 0.15 to 0.19 s at L = 126 and in 0.48 s at
# L = 165. Polarized pbe_c of PureXC (series of up to 165 coefficients in three variables, through
# `substitute` to five) compiles at order 8 in 5.4 s with this value and in 7.0 s with 165, for an
# evaluation 15% faster (253 against 297 µs). Every series of PureXC at orders 0 to 2 (at most 21
# coefficients) stays unrolled.
const _UNROLL_MAX = 126

# The most coefficients of a series for which the Taylor operations are forced inline (`compose`;
# see arith.jl). Forced inlining of the unrolled kernels compiles in time and memory that grow with
# the number of coefficients and the order: polarized `lda_c_pw` at order 10 (66 coefficients) took
# 104 s with `compose` forced inline at every size, against 4.8 s without. The value is the
# number of coefficients of the largest series that PureXC evaluates at orders 0 to 2.
const _INLINE_MAX = 21

# Whether a kernel with `n` output coefficients uses its loop form.
_use_loop(n::Integer) = n > _UNROLL_MAX

# The stack arrays below are `Ref`s written through a pointer. LLVM places such a `Ref` on the stack,
# which the proofs of `:noalloc` (AllocCheck, on every kernel that calls these functions) establish;
# the value-free scan of `@strict_policy` counts the pointer write as an escape, so these three
# functions are exempt from its `:noalloc` scan (see `PureTaylor.jl`). They are not inlined, so that
# each `Ref` is a statement of one of these exempt functions and of no caller.

# The tuple (f(1), f(2), …, f(L)) of type NTuple{L,T}, built by a loop. For an isbits T the entries
# are written to a stack array; any other T goes through a Vector.
@noinline function _loop_tuple(f::F, ::Type{T}, ::Val{L}) where {F, T, L}
    if isbitstype(T)
        r = Ref{NTuple{L, T}}()
        GC.@preserve r begin
            p = Ptr{T}(pointer_from_objref(r))
            for j in 1:L
                unsafe_store!(p, convert(T, f(j))::T, j)
            end
        end
        return r[]
    else
        v = Vector{T}(undef, L)
        for j in eachindex(v)
            v[j] = f(j)
        end
        return (v...,)::NTuple{L, T}
    end
end

# The tuple (x₁, …, x_L) with x₁ = x and xⱼ = g(xⱼ₋₁, j - 1), built by a loop.
@noinline function _loop_recurrence(g::G, x::T, ::Val{L}) where {G, T, L}
    if isbitstype(T)
        r = Ref{NTuple{L, T}}()
        GC.@preserve r begin
            p = Ptr{T}(pointer_from_objref(r))
            unsafe_store!(p, x, 1)
            for j in 2:L
                x = g(x, j - 1)::T
                unsafe_store!(p, x, j)
            end
        end
        return r[]
    else
        v = Vector{T}(undef, L)
        v[1] = x
        for j in 2:L
            v[j] = x = g(x, j - 1)::T
        end
        return (v...,)::NTuple{L, T}
    end
end

# A scratch array of `L` entries of type `T`: on the stack for an isbits T, a Vector otherwise. It is
# read and written with `_load` and `_store!`, which check the index, inside `_with_scratch`.
struct _StackScratch{T, L}
    p::Ptr{T}
end

@noinline function _with_scratch(f::F, ::Type{T}, ::Val{L}) where {F, T, L}
    if isbitstype(T)
        r = Ref{NTuple{L, T}}()
        return GC.@preserve r f(_StackScratch{T, L}(Ptr{T}(pointer_from_objref(r))))
    else
        return f(Vector{T}(undef, L))
    end
end

@inline function _load(s::_StackScratch{T, L}, i::Integer) where {T, L}
    1 <= i <= L || throw(BoundsError(s, i))
    return unsafe_load(s.p, i)
end
@inline function _store!(s::_StackScratch{T, L}, i::Integer, x::T) where {T, L}
    1 <= i <= L || throw(BoundsError(s, i))
    unsafe_store!(s.p, x, i)
    return nothing
end
@inline _load(v::Vector, i::Integer) = v[i]
@inline _store!(v::Vector{T}, i::Integer, x::T) where {T} = (v[i] = x; nothing)

# The entries `off + 1:off + n` of scratch `s` as an NTuple{n}, by a loop.
@inline _scratch_tuple(s, ::Type{T}, off::Int, ::Val{n}) where {T, n} = _loop_tuple(j -> _load(s, off + j), T, Val(n))

# Positions in a table are stored as Int32, which halves the size of the constant tuples.
_int32_tuple(v) = Tuple(Int32(x) for x in v)

# The total degree of each stored multi-index of a Taylor{N,K,M}.
@generated _degrees(::Val{N}, ::Val{K}, ::Val{M}) where {N, K, M} =
    _int32_tuple(sum(α) for α in stored_indices(Val(N), Val(K), Val(M)))
