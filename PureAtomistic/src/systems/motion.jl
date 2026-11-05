"""
    kick!(V, F, m, τ) -> nothing

Velocity update `v ← v + τ F / m` for every atom: velocities `V` in Å/fs (3 × n), forces `F` in
eV/Å (same axes as `V`), masses `m` in amu (one per column of `V`) and the time `τ` in fs. The
mass is converted with [`AMU_IN_EV_FS2_PER_A2`](@ref), so the update is in Å/fs. Throws a
`DimensionMismatch` when the axes of `F` and `V` differ.
"""
function kick!(V::AbstractMatrix{T}, F::AbstractMatrix, m::AbstractVector{T}, τ::T) where {T}
    axes(F) == axes(V) || throw(DimensionMismatch(lazy"pureatomistic: forces axes $(axes(F)) differ from velocities axes $(axes(V))"))
    axes(m, 1) == axes(V, 2) || throw(DimensionMismatch(lazy"pureatomistic: masses axes $(axes(m)) differ from the atom axis $(axes(V, 2))"))
    c = T(AMU_IN_EV_FS2_PER_A2)
    for a in axes(V, 2)
        f = τ / (m[a] * c)
        for k in axes(V, 1)
            V[k, a] += f * F[k, a]
        end
    end
    return nothing
end

"""
    drift!(X, V, h) -> nothing

Position update `x ← x + h v`: positions `X` in Å, velocities `V` in Å/fs (same axes as `X`) and
the time `h` in fs. Throws a `DimensionMismatch` when the axes of `X` and `V` differ.
"""
function drift!(X::AbstractMatrix{T}, V::AbstractMatrix{T}, h::T) where {T}
    axes(X) == axes(V) || throw(DimensionMismatch(lazy"pureatomistic: positions axes $(axes(X)) differ from velocities axes $(axes(V))"))
    for a in axes(X, 2), k in axes(X, 1)
        X[k, a] += h * V[k, a]
    end
    return nothing
end
