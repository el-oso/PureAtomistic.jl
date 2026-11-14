# Root-mean-square deviation from a reference configuration, optionally after the best rigid
# superposition by Kabsch's method. The method is re-derived below, not taken from the source,
# and the tests check it against the minimum over rotations from an independent eigenvalue method.

# 3 × 3 matrices are column-major 9-tuples (see vacancy.jl), so the alignment of a frame
# allocates nothing.
@inline _col3(m::NTuple{9, T}, k::Int) where {T} = (m[3k - 2], m[3k - 1], m[3k])
@inline _cross3(a::NTuple{3, T}, b::NTuple{3, T}) where {T} =
    (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
@inline _scale3(a::NTuple{3, T}, s::T) where {T} = (a[1] * s, a[2] * s, a[3] * s)
@inline _cols3(a::NTuple{3, T}, b::NTuple{3, T}, c::NTuple{3, T}) where {T} =
    (a[1], a[2], a[3], b[1], b[2], b[3], c[1], c[2], c[3])

# A unit vector perpendicular to the unit vector `u`: `u` crossed with the coordinate axis it is
# least aligned with.
@inline function _perp3(u::NTuple{3, T}) where {T}
    a1, a2, a3 = abs(u[1]), abs(u[2]), abs(u[3])
    e = a1 <= a2 && a1 <= a3 ? (one(T), zero(T), zero(T)) :
        a2 <= a3 ? (zero(T), one(T), zero(T)) : (zero(T), zero(T), one(T))
    w = _cross3(u, e)
    return _scale3(w, inv(sqrt(_dot3(w, w))))
end

# The singular value decomposition `H = U diag(σ) Vᵀ` of a 3 × 3 matrix by one-sided Jacobi
# rotations (Hestenes): rotations from the right make the columns of `H V` orthogonal, and then
# they are `σₖ uₖ`. A pair of columns `(p, q)` with `α = |p|²`, `β = |q|²` and `γ = p·q` is
# rotated by the angle that zeroes `γ`: `ζ = (β - α) / 2γ`, `t = sign(ζ) / (|ζ| + √(1 + ζ²))`,
# `c = 1 / √(1 + t²)`, `s = c t`, `p ← c p - s q`, `q ← s p + c q`. Sweeps over the three pairs
# stop when no pair has `|γ| > eps(T) √(αβ)`. The columns are returned with `σ` in decreasing order.
# The third column of `U` is `u₁ × u₂`: that is `±u₃`, and the Kabsch rotation, which carries
# `d u₃` with `d = sign det(V Uᵀ)`, is the same for either sign, while `u₃ = a₃ / σ₃` is rounding
# noise for a planar configuration. A zero `σ₂` leaves `u₂` free, and a unit vector perpendicular to
# `u₁` is taken.
function _svd3(H::NTuple{9, T}) where {T}
    a = (_col3(H, 1), _col3(H, 2), _col3(H, 3))
    v = ((one(T), zero(T), zero(T)), (zero(T), one(T), zero(T)), (zero(T), zero(T), one(T)))
    for _ in 1:60
        rotated = false
        for (p, q) in ((1, 2), (1, 3), (2, 3))
            ap, aq = a[p], a[q]
            α, β, γ = _dot3(ap, ap), _dot3(aq, aq), _dot3(ap, aq)
            abs(γ) > eps(T) * sqrt(α * β) || continue
            rotated = true
            ζ = (β - α) / (2γ)
            t = (ζ >= 0 ? one(T) : -one(T)) / (abs(ζ) + sqrt(1 + ζ^2))
            c = 1 / sqrt(1 + t^2)
            s = c * t
            a = Base.setindex(Base.setindex(a, _sub3(_scale3(ap, c), _scale3(aq, s)), p), _add3(_scale3(ap, s), _scale3(aq, c)), q)
            vp, vq = v[p], v[q]
            v = Base.setindex(Base.setindex(v, _sub3(_scale3(vp, c), _scale3(vq, s)), p), _add3(_scale3(vp, s), _scale3(vq, c)), q)
        end
        rotated || break
    end
    σ = (sqrt(_dot3(a[1], a[1])), sqrt(_dot3(a[2], a[2])), sqrt(_dot3(a[3], a[3])))
    # Decreasing σ: a three-element sort by exchanges.
    order = (1, 2, 3)
    σ[order[1]] < σ[order[2]] && (order = (order[2], order[1], order[3]))
    σ[order[2]] < σ[order[3]] && (order = (order[1], order[3], order[2]))
    σ[order[1]] < σ[order[2]] && (order = (order[2], order[1], order[3]))
    s1, s2, s3 = σ[order[1]], σ[order[2]], σ[order[3]]
    a1, a2 = a[order[1]], a[order[2]]
    u1 = iszero(s1) ? (one(T), zero(T), zero(T)) : _scale3(a1, inv(s1))
    u2 = iszero(s2) ? _perp3(u1) : _scale3(a2, inv(s2))
    u3 = _cross3(u1, u2)
    U = _cols3(u1, u2, u3)
    V = _cols3(v[order[1]], v[order[2]], v[order[3]])
    return U, (s1, s2, s3), V
end

@inline _det3(m::NTuple{9, T}) where {T} =
    m[1] * (m[5] * m[9] - m[6] * m[8]) - m[4] * (m[2] * m[9] - m[3] * m[8]) + m[7] * (m[2] * m[6] - m[3] * m[5])

# X and Y share their axes. The proper rotation `R` that minimizes `Σ |R (xᵢ - x̄) - (yᵢ - ȳ)|²`, with the centroids `x̄`
# and `ȳ`. With the covariance `H = Σ (xᵢ - x̄)(yᵢ - ȳ)ᵀ = U S Vᵀ`, the optimal orthogonal map is
# `V Uᵀ`; when its determinant is -1 (a reflection), flipping the sign of the singular vector of
# the smallest singular value gives the best proper rotation, `R = V diag(1, 1, d) Uᵀ` with
# `d = sign det(V Uᵀ)`. `R` is a column-major 9-tuple.
function _kabsch(X::AbstractMatrix, Y::AbstractMatrix, ::Type{T}) where {T}
    i, j = first(axes(X, 1)), first(axes(Y, 1))
    n = size(X, 2)
    x̄ = (zero(T), zero(T), zero(T))
    ȳ = (zero(T), zero(T), zero(T))
    for a in axes(X, 2)
        x̄ = _add3(x̄, _column(T, X, i, a))
        ȳ = _add3(ȳ, _column(T, Y, j, a))
    end
    x̄ = (x̄[1] / n, x̄[2] / n, x̄[3] / n)
    ȳ = (ȳ[1] / n, ȳ[2] / n, ȳ[3] / n)
    H = ntuple(_ -> zero(T), Val(9))
    for a in axes(X, 2)
        x = _sub3(_column(T, X, i, a), x̄)
        y = _sub3(_column(T, Y, j, a), ȳ)
        H = (
            H[1] + x[1] * y[1], H[2] + x[2] * y[1], H[3] + x[3] * y[1],
            H[4] + x[1] * y[2], H[5] + x[2] * y[2], H[6] + x[3] * y[2],
            H[7] + x[1] * y[3], H[8] + x[2] * y[3], H[9] + x[3] * y[3],
        )
    end
    U, _, V = _svd3(H)
    d = sign(_det3(V) * _det3(U))
    # R = V diag(1, 1, d) Uᵀ, entry (p, q) = Σₖ V[p, k] wₖ U[q, k].
    w = (one(T), one(T), d)
    R = ntuple(Val(9)) do e
        p, q = mod1(e, 3), cld(e, 3)
        V[p] * w[1] * U[q] + V[p + 3] * w[2] * U[q + 3] + V[p + 6] * w[3] * U[q + 6]
    end
    return R, x̄, ȳ
end

function _check_rmsd_arguments(frames, reference::AbstractMatrix)
    size(reference, 1) == 3 || throw(ArgumentError(lazy"pureatomistic: reference needs 3 rows, got $(size(reference, 1))"))
    size(reference, 2) >= 1 || throw(ArgumentError("pureatomistic: rmsd needs at least one atom"))
    isempty(eachindex(frames)) && throw(ArgumentError("pureatomistic: rmsd needs at least one frame"))
    return nothing
end

"""
    rmsd(frames, reference; align = false) -> Vector

Root-mean-square deviation of each frame's `positions` from the 3 × n matrix `reference`, as a
new vector in the order of `eachindex(frames)`; see [`rmsd!`](@ref).
"""
function rmsd(frames, reference::AbstractMatrix; align::Bool = false)
    _check_rmsd_arguments(frames, reference)
    T = float(promote_type(eltype(frames[first(eachindex(frames))].positions), eltype(reference)))
    return rmsd!(Vector{T}(undef, length(eachindex(frames))), frames, reference; align)
end

"""
    rmsd!(out, frames, reference; align = false) -> out

Writes into `out`, in the order of `eachindex(frames)`, the root-mean-square deviation
`√(Σᵢ |xᵢ - yᵢ|² / n)` of each frame's `positions` (`xᵢ`) from the 3 × n matrix `reference`
(`yᵢ`). `out` has one entry per frame. With `align = true`, each frame is first superposed on the
reference by the translation and proper rotation that minimize the deviation (Kabsch's method):
both are centered on their centroids and the frame is rotated by the optimal rotation. Positions
are taken as given, with no periodic images, so a crystal should be passed with unwrapped
positions. The positions of every frame must have the axes of `reference`; views and
`OffsetArray`s are accepted. Allocates nothing for frames that are already in memory.
"""
function rmsd!(out::AbstractVector, frames, reference::AbstractMatrix; align::Bool = false)
    _check_rmsd_arguments(frames, reference)
    n = size(reference, 2)
    length(out) == length(eachindex(frames)) ||
        throw(DimensionMismatch(lazy"pureatomistic: out has $(length(out)) entries for $(length(eachindex(frames))) frames"))
    T = float(promote_type(eltype(frames[first(eachindex(frames))].positions), eltype(reference)))
    j = first(axes(reference, 1))
    for (c, k) in zip(eachindex(out), eachindex(frames))
        X = frames[k].positions
        axes(X) == axes(reference) ||
            throw(DimensionMismatch(lazy"pureatomistic: frame $k has positions $(axes(X)), the reference $(axes(reference))"))
        i = first(axes(X, 1))
        s = zero(T)
        if align
            R, x̄, ȳ = _kabsch(X, reference, T)
            for a in axes(X, 2)
                x = _sub3(_column(T, X, i, a), x̄)
                y = _sub3(_column(T, reference, j, a), ȳ)
                e = _sub3(_mul3(R, x), y)
                s += _dot3(e, e)
            end
        else
            for a in axes(X, 2)
                e = _sub3(_column(T, X, i, a), _column(T, reference, j, a))
                s += _dot3(e, e)
            end
        end
        out[c] = sqrt(s / n)
    end
    return out
end
