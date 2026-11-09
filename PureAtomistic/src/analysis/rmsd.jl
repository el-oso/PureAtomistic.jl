# Root-mean-square deviation from a reference configuration, optionally after the best rigid
# superposition (Kabsch, Acta Cryst. A 32, 922 (1976)).

# X and Y share their axes. The proper rotation `R` that minimizes `Σ |R (xᵢ - x̄) - (yᵢ - ȳ)|²`, with the centroids `x̄`
# and `ȳ`. With the covariance `H = Σ (xᵢ - x̄)(yᵢ - ȳ)ᵀ = U S Vᵀ`, the optimal orthogonal map is
# `V Uᵀ`; when its determinant is -1 (a reflection), flipping the sign of the singular vector of
# the smallest singular value gives the best proper rotation, `R = V diag(1, 1, d) Uᵀ` with
# `d = sign det(V Uᵀ)`.
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
    H = zeros(T, 3, 3)
    for a in axes(X, 2)
        x = _sub3(_column(T, X, i, a), x̄)
        y = _sub3(_column(T, Y, j, a), ȳ)
        for q in 1:3, p in 1:3
            H[p, q] += x[p] * y[q]
        end
    end
    F = LinearAlgebra.svd(H)
    d = sign(LinearAlgebra.det(F.V * F.U'))
    R = F.V * LinearAlgebra.Diagonal(T[1, 1, d]) * F.U'
    return R, x̄, ȳ
end

"""
    rmsd(frames, reference; align = false) -> Vector

Root-mean-square deviation `√(Σᵢ |xᵢ - yᵢ|² / n)` of each frame's `positions` (`xᵢ`) from the
3 × n matrix `reference` (`yᵢ`), as a vector in the order of `eachindex(frames)`. With
`align = true`, each frame is first superposed on the reference by the translation and proper
rotation that minimize the deviation (Kabsch's method): both are centered on their centroids and
the frame is rotated by the optimal rotation. Positions are taken as given, with no periodic
images, so a crystal should be passed with unwrapped positions. The positions of every frame must
have the axes of `reference`; views and `OffsetArray`s are accepted.
"""
function rmsd(frames, reference::AbstractMatrix; align::Bool = false)
    size(reference, 1) == 3 || throw(ArgumentError(lazy"pureatomistic: reference needs 3 rows, got $(size(reference, 1))"))
    n = size(reference, 2)
    n >= 1 || throw(ArgumentError("pureatomistic: rmsd needs at least one atom"))
    isempty(eachindex(frames)) && throw(ArgumentError("pureatomistic: rmsd needs at least one frame"))
    T = float(promote_type(eltype(frames[first(eachindex(frames))].positions), eltype(reference)))
    out = Vector{T}(undef, length(eachindex(frames)))
    j = first(axes(reference, 1))
    c = 0
    for k in eachindex(frames)
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
                rx = (
                    R[1, 1] * x[1] + R[1, 2] * x[2] + R[1, 3] * x[3],
                    R[2, 1] * x[1] + R[2, 2] * x[2] + R[2, 3] * x[3],
                    R[3, 1] * x[1] + R[3, 2] * x[2] + R[3, 3] * x[3],
                )
                e = _sub3(rx, y)
                s += _dot3(e, e)
            end
        else
            for a in axes(X, 2)
                e = _sub3(_column(T, X, i, a), _column(T, reference, j, a))
                s += _dot3(e, e)
            end
        end
        c += 1
        out[c] = sqrt(s / n)
    end
    return out
end
