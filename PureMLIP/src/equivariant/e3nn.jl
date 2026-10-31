# Conversion from the native real spherical harmonics and coupling tensors to e3nn's
# conventions. For a unit vector v, e3nn's harmonics with `normalization="component"` are
# `Y_e3nn,l(v) = sqrt(4π) · Y_l(v3, v1, v2)`, ordered l-major with the native m = -l..l order
# inside each l. Equivalently `Y_e3nn,l = sqrt(4π) · M_l · Y_l(v)` with
# `M_l = wigner_D(l, R)` and `R` the rotation v ↦ (v3, v1, v2).

const E3NN_AXES = [0.0 0.0 1.0; 1.0 0.0 0.0; 0.0 1.0 0.0]   # R: v ↦ (v3, v1, v2)

"""
    e3nn_spherical_harmonics!(Y, v, lmax) -> Y

Fill `Y` (length `sh_length(lmax)`) with e3nn's spherical harmonics of `v / ‖v‖`
(`normalize=True`, `normalization="component"`), l-major in e3nn's component order:
`sqrt(4π) · Y_tb(v3, v1, v2)` with `Y_tb` the harmonics of [`spherical_harmonics!`](@ref).
"""
function e3nn_spherical_harmonics!(Y::AbstractVector{T}, v::AbstractVector{T}, lmax::Integer) where {T}
    _check_sh_sizes(Y, v, lmax)
    fv = firstindex(v)
    _real_sh!(Y, nothing, v[fv + 2], v[fv], v[fv + 1], Int(lmax))
    c = sqrt(4 * T(π))
    for i in eachindex(Y)
        Y[i] *= c
    end
    return Y
end

"""
    e3nn_spherical_harmonics_and_gradient!(Y, dY, v, lmax) -> (Y, dY)

As [`e3nn_spherical_harmonics!`](@ref), and also fill `dY` (size `(sh_length(lmax), 3)`)
with `dY[k, i] = ∂Y_k/∂v_i`, the derivatives with respect to the unnormalized `v` in its own
`(v1, v2, v3)` order, rows and columns offset from `firstindex` of each axis of `dY`.
"""
function e3nn_spherical_harmonics_and_gradient!(Y::AbstractVector{T}, dY::AbstractMatrix{T}, v::AbstractVector{T}, lmax::Integer) where {T}
    _check_sh_sizes(Y, v, lmax)
    n = sh_length(lmax)
    size(dY) == (n, 3) || throw(DimensionMismatch(lazy"dY has size $(size(dY)), expected ($n, 3)"))
    fv = firstindex(v)
    _real_sh!(Y, dY, v[fv + 2], v[fv], v[fv + 1], Int(lmax))
    c = sqrt(4 * T(π))
    for i in eachindex(Y)
        Y[i] *= c
    end
    # The kernel's columns are derivatives with respect to its arguments (v3, v1, v2).
    j = first(axes(dY, 2))
    for i in axes(dY, 1)
        d3, d1, d2 = dY[i, j], dY[i, j + 1], dY[i, j + 2]
        dY[i, j] = c * d1
        dY[i, j + 1] = c * d2
        dY[i, j + 2] = c * d3
    end
    return Y, dY
end

"""
    e3nn_basis_change(l) -> Matrix{Float64}

Orthogonal matrix `M` of size `(2l+1, 2l+1)` with `Y_e3nn,l = sqrt(4π) · M · Y_l`, where
`Y_l` is the degree-`l` block of [`spherical_harmonics!`](@ref). Rows index e3nn components
`0..2l`, columns the native `m = -l..l`. It is `wigner_D(l, R)` for the rotation
`R: v ↦ (v3, v1, v2)`.
"""
e3nn_basis_change(l::Integer) = wigner_D(l, E3NN_AXES)

# Magnitude below which a coefficient of `e3nn_wigner_3j` is rounding residue and is stored as
# an exact zero, so the sparsity pattern matches e3nn's. For every even triple with all l ≤ 8
# the residue is below 1e-15 and the smallest true nonzero coefficient is above 7e-6.
const E3NN_W3J_ZERO = 1.0e-13

"""
    e3nn_wigner_3j(l1, l2, l3) -> Array{Float64,3}

e3nn's `wigner_3j(l1, l2, l3)`: the unit-norm coupling tensor of shape
`(2l1+1, 2l2+1, 2l3+1)` indexed by e3nn components. It is
`s · Σ M1[a,a'] M2[b,b'] M3[c,c'] C[a',b',c']` with `C = real_clebsch_gordan(l1, l2, l3)`,
`Mi = e3nn_basis_change(li)` and `s = (-1)^((l1+l2-l3)/2)`. Entries with magnitude below
`1e-13` are set to zero.

Throws `ArgumentError` unless `|l1-l2| ≤ l3 ≤ l1+l2` and `l1+l2+l3` is even.
"""
function e3nn_wigner_3j(l1::Integer, l2::Integer, l3::Integer)
    C = real_clebsch_gordan(l1, l2, l3)
    M1, M2, M3 = e3nn_basis_change(l1), e3nn_basis_change(l2), e3nn_basis_change(l3)
    s = iseven((l1 + l2 - l3) ÷ 2) ? 1.0 : -1.0
    # Contract one index at a time: A = M1 on index 1, B = M2 on index 2, E = M3 on index 3.
    A = zeros(Float64, size(C))
    for c in axes(C, 3), b in axes(C, 2), a in axes(C, 1), p in axes(C, 1)
        A[a, b, c] += M1[a, p] * C[p, b, c]
    end
    B = zeros(Float64, size(C))
    for c in axes(C, 3), b in axes(C, 2), p in axes(C, 2), a in axes(C, 1)
        B[a, b, c] += M2[b, p] * A[a, p, c]
    end
    E = zeros(Float64, size(C))
    for c in axes(C, 3), p in axes(C, 3), b in axes(C, 2), a in axes(C, 1)
        E[a, b, c] += M3[c, p] * B[a, b, p]
    end
    for i in eachindex(E)
        E[i] = abs(E[i]) < E3NN_W3J_ZERO ? 0.0 : s * E[i]
    end
    return E
end
