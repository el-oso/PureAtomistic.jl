# Real spherical harmonics Y_lm, orthonormal on the unit sphere (∫ Y_lm Y_l'm' dΩ = δ δ),
# z as the polar axis, without the Condon-Shortley phase:
#   Y_l0 = N_l0 P_l(cos θ),  Y_lm = √2 N_lm P_l^m(cos θ) cos(mφ),  Y_l,-m = √2 N_lm P_l^m(cos θ) sin(mφ)
# for m > 0, with N_lm = sqrt((2l+1)/(4π) (l-m)!/(l+m)!). Degree-l values occupy the
# contiguous block m = -l..l, at position `sh_index(l, m)` in a vector of `sh_length(lmax)`.
#
# The recurrence for the unnormalized polynomials Q_lm, the prefactors F_lm, and the gradient
# identities are ported from SpheriCart (MIT), github.com/lab-cosmo/sphericart at commit
# 5db6baecce938cd971ee4f8b6eea45d0a1099287, julia/src/ka_kernels.jl and generated_kernels.jl.

"""
    sh_length(lmax) -> Int

Number of real spherical harmonics with degree `l ≤ lmax`, `(lmax + 1)^2`.
"""
sh_length(lmax::Integer) = (Int(lmax) + 1)^2

"""
    sh_index(l, m) -> Int

One-based position of `Y_lm` in a vector of real spherical harmonics, `l^2 + l + m + 1`.
"""
sh_index(l::Integer, m::Integer) = Int(l)^2 + Int(l) + Int(m) + 1

"""
    spherical_harmonics!(Y, v, lmax) -> Y

Fill `Y` with the real spherical harmonics `Y_lm(v / ‖v‖)` for `0 ≤ l ≤ lmax`, `-l ≤ m ≤ l`,
orthonormal on the unit sphere with `z` as the polar axis and no Condon-Shortley phase.
`Y[firstindex(Y) - 1 + sh_index(l, m)]` holds `Y_lm`. `v` has three entries read as x, y, z.
"""
function spherical_harmonics!(Y::AbstractVector{T}, v::AbstractVector{T}, lmax::Integer) where {T}
    _check_sh_sizes(Y, v, lmax)
    _real_sh!(Y, nothing, v, Int(lmax))
    return Y
end

"""
    spherical_harmonics_and_gradient!(Y, dY, v, lmax) -> (Y, dY)

As [`spherical_harmonics!`](@ref), and also fill `dY` (size `(sh_length(lmax), 3)`) with the
derivatives with respect to the unnormalized vector: `dY[k, i] = ∂Y_k/∂v_i`, rows and columns
offset from `firstindex` of each axis of `dY`.
"""
function spherical_harmonics_and_gradient!(Y::AbstractVector{T}, dY::AbstractMatrix{T}, v::AbstractVector{T}, lmax::Integer) where {T}
    _check_sh_sizes(Y, v, lmax)
    n = sh_length(lmax)
    size(dY) == (n, 3) || throw(DimensionMismatch(lazy"dY has size $(size(dY)), expected ($n, 3)"))
    _real_sh!(Y, dY, v, Int(lmax))
    return Y, dY
end

function _check_sh_sizes(Y::AbstractVector, v::AbstractVector, lmax::Integer)
    lmax >= 0 || throw(ArgumentError(lazy"lmax must be nonnegative, got $lmax"))
    n = sh_length(lmax)
    length(Y) == n || throw(DimensionMismatch(lazy"Y has length $(length(Y)), expected sh_length($lmax) = $n"))
    length(v) == 3 || throw(DimensionMismatch(lazy"v has length $(length(v)), expected 3"))
    return nothing
end

# `dY === nothing` computes values only. The unnormalized Q_lm (m ≥ 0) are first stored in
# the Y slots of (l, m); a second pass, by descending l, overwrites row l with Y_l,±m while
# row l - 1 still holds Q_{l-1,·}, which the gradient needs. The slots (l, -m) with m > 0
# never hold a Q, so writing Y_l,-m there is safe.
function _real_sh!(Y::AbstractVector{T}, dY, v::AbstractVector{T}, L::Int) where {T}
    fv = firstindex(v)
    return _real_sh!(Y, dY, v[fv], v[fv + 1], v[fv + 2], L)
end

function _real_sh!(Y::AbstractVector{T}, dY, vx::T, vy::T, vz::T, L::Int) where {T}
    r2 = vx * vx + vy * vy + vz * vz
    iszero(r2) && throw(ArgumentError("spherical harmonics of a zero vector are undefined"))
    r = sqrt(r2)
    x, y, z = vx / r, vy / r, vz / r
    o = firstindex(Y) - 1
    q(l, m) = Y[o + sh_index(l, m)]

    # Q_lm on the unit sphere (r² = 1); Q_00 = 1, Q_11 = -1, Q_10 = z.
    Y[o + sh_index(0, 0)] = one(T)
    if L >= 1
        Y[o + sh_index(1, 1)] = -one(T)
        Y[o + sh_index(1, 0)] = z
    end
    for l in 2:L
        qll = q(l - 1, l - 1)
        Y[o + sh_index(l, l)] = -(2l - 1) * qll
        Y[o + sh_index(l, l - 1)] = (2l - 1) * z * qll
        for m in (l - 2):-1:0
            Y[o + sh_index(l, m)] = ((2l - 1) * z * q(l - 1, m) - (l + m - 1) * q(l - 2, m)) / (l - m)
        end
    end

    rt2 = sqrt(T(2))
    for l in L:-1:0
        # F_l0 = sqrt((2l+1)/(2π)); F_lm = -F_l,m-1 / sqrt((l+m)(l+1-m)).
        F = sqrt(T(2l + 1) / (2 * T(π)))
        s, c = zero(T), one(T)            # sin(mφ), cos(mφ) times sin^m θ, at m = 0
        sp, cp = zero(T), zero(T)         # the same at m - 1
        for m in 0:l
            if m > 0
                F = -F / sqrt(T((l + m) * (l + 1 - m)))
                sp, cp = s, c
                s, c = sp * x + cp * y, cp * x - sp * y
            end
            Qlm = q(l, m)
            if !isnothing(dY)
                # ∂Q_lm/∂(x, y, z) = (x Q_{l-1,m+1}, y Q_{l-1,m+1}, (l+m) Q_{l-1,m}), with
                # Q_{l-1,k} = 0 for k > l - 1; ∂(c_m, s_m)/∂x = m (c_{m-1}, s_{m-1}),
                # ∂(c_m, s_m)/∂y = m (-s_{m-1}, c_{m-1}).
                qa = m + 1 <= l - 1 ? q(l - 1, m + 1) : zero(T)
                qb = m <= l - 1 ? (l + m) * q(l - 1, m) : zero(T)
                if iszero(m)
                    Fm = F / rt2
                    _sh_grad_row!(dY, sh_index(l, 0), Fm * x * qa, Fm * y * qa, Fm * qb, x, y, z, r)
                else
                    _sh_grad_row!(
                        dY, sh_index(l, -m),
                        F * (Qlm * m * sp + x * qa * s), F * (Qlm * m * cp + y * qa * s), F * qb * s, x, y, z, r,
                    )
                    _sh_grad_row!(
                        dY, sh_index(l, m),
                        F * (Qlm * m * cp + x * qa * c), F * (-Qlm * m * sp + y * qa * c), F * qb * c, x, y, z, r,
                    )
                end
            end
            if iszero(m)
                Y[o + sh_index(l, 0)] = F / rt2 * Qlm
            else
                Y[o + sh_index(l, -m)] = F * Qlm * s
                Y[o + sh_index(l, m)] = F * Qlm * c
            end
        end
    end
    return nothing
end

# Store the gradient with respect to the raw vector, (I - u uᵀ) g / ‖v‖, of a function whose
# gradient with respect to the unit vector u = (x, y, z) is g = (gx, gy, gz). `k` is the row
# position counted from 1 within the row axis of `dY`.
function _sh_grad_row!(dY::AbstractMatrix{T}, k::Int, gx::T, gy::T, gz::T, x::T, y::T, z::T, r::T) where {T}
    d = x * gx + y * gy + z * gz
    i = first(axes(dY, 1)) - 1 + k
    j = first(axes(dY, 2))
    dY[i, j] = (gx - d * x) / r
    dY[i, j + 1] = (gy - d * y) / r
    dY[i, j + 2] = (gz - d * z) / r
    return nothing
end
