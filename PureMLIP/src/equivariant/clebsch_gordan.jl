# Clebsch-Gordan coefficients: the Condon-Shortley complex coefficients by the Racah closed
# form, and the real-basis coupling tensor for the real spherical harmonics of
# spherical_harmonics.jl.

# FACTORIALS[n + 1] = n! in Float64; correctly rounded products, exact while n! has at most 53
# significant bits (n ≤ 22). Finite up to n = 170.
const FACTORIALS = let f = ones(Float64, 171)
    for n in 1:170
        f[n + 1] = f[n] * n
    end
    f
end

function _fact(n::Integer)
    (0 <= n < length(FACTORIALS)) || throw(ArgumentError(lazy"factorial argument out of range: $n"))
    return FACTORIALS[n + 1]
end

"""
    clebsch_gordan_complex(l1, m1, l2, m2, l3, m3) -> Float64

Condon-Shortley Clebsch-Gordan coefficient ⟨l1 m1 l2 m2 | l3 m3⟩ for integer `l`, by the Racah
closed form with Float64 factorials. Returns `0.0` when `m1 + m2 ≠ m3`, when any `|m| > l`, or
when `l1, l2, l3` violate the triangle inequality.
"""
function clebsch_gordan_complex(l1::Integer, m1::Integer, l2::Integer, m2::Integer, l3::Integer, m3::Integer)
    (m1 + m2 == m3 && abs(m1) <= l1 && abs(m2) <= l2 && abs(m3) <= l3) || return 0.0
    (abs(l1 - l2) <= l3 <= l1 + l2) || return 0.0
    pre = (2l3 + 1) * _fact(l3 + l1 - l2) * _fact(l3 - l1 + l2) * _fact(l1 + l2 - l3) / _fact(l1 + l2 + l3 + 1)
    pre *= _fact(l3 + m3) * _fact(l3 - m3) * _fact(l1 - m1) * _fact(l1 + m1) * _fact(l2 - m2) * _fact(l2 + m2)
    s = 0.0
    for k in max(0, l2 - l3 - m1, l1 - l3 + m2):min(l1 + l2 - l3, l1 - m1, l2 + m2)
        t = _fact(k) * _fact(l1 + l2 - l3 - k) * _fact(l1 - m1 - k) * _fact(l2 + m2 - k) * _fact(l3 - l2 + m1 + k) * _fact(l3 - l1 - m2 + k)
        s += (iseven(k) ? 1.0 : -1.0) / t
    end
    return sqrt(pre) * s
end

# Unitary U with Y_real = U * Y_complex, rows and columns ordered m = -l..l. Real harmonics
# (no Condon-Shortley phase) from complex ones with the Condon-Shortley phase:
#   m < 0: (i/√2)(Y^m − (−1)^m Y^{−m});  m = 0: Y^0;  m > 0: (1/√2)(Y^{−m} + (−1)^m Y^m).
function _complex_to_real(l::Int)
    U = zeros(ComplexF64, 2l + 1, 2l + 1)
    h = inv(sqrt(2.0))
    sgn(m) = iseven(m) ? 1.0 : -1.0
    for m in -l:l
        a = m + l + 1
        if m < 0
            U[a, m + l + 1] = im * h
            U[a, -m + l + 1] = -im * h * sgn(m)
        elseif m == 0
            U[a, a] = 1
        else
            U[a, -m + l + 1] = h
            U[a, m + l + 1] = h * sgn(m)
        end
    end
    return U
end

# Mode-`k` product of a 3-way array with a matrix:
# `B[.., i, ..] = Σ_p M[i, p] * A[.., p, ..]` with `i` and `p` at position `k`.
function _mode_product(A::AbstractArray{T, 3}, M::AbstractMatrix, k::Int) where {T}
    Base.require_one_based_indexing(A, M)
    1 <= k <= 3 || throw(ArgumentError(lazy"mode must be 1, 2 or 3, got $k"))
    size(M, 2) == size(A, k) || throw(DimensionMismatch(lazy"M has $(size(M, 2)) columns but mode $k of A has size $(size(A, k))"))
    perm = k == 1 ? (1, 2, 3) : k == 2 ? (2, 1, 3) : (3, 1, 2)
    Ap = permutedims(A, perm)
    sz = size(Ap)
    P = M * reshape(Ap, sz[1], sz[2] * sz[3])
    return permutedims(reshape(P, size(M, 1), sz[2], sz[3]), invperm(perm))
end

"""
    real_clebsch_gordan(l1, l2, l3) -> Array{Float64,3}

Real coupling tensor `C[a, b, c]` of shape `(2l1+1, 2l2+1, 2l3+1)` for the real spherical
harmonics of [`spherical_harmonics!`](@ref), indices ordered `m = -l..l`, normalized to unit
Frobenius norm. It is
`Σ conj(U1[a,m1]) conj(U2[b,m2]) U3[c,m3] ⟨l1 m1 l2 m2 | l3 m3⟩ / sqrt(2 l3 + 1)`
with `U_l` the complex-to-real unitary, and is real when `l1 + l2 + l3` is even.

Invariance under rotation, with `D_l = wigner_D(l, R)`:
`Σ_abc C[a,b,c] D1[a,i] D2[b,j] D3[c,k] = C[i,j,k]`.

Throws `ArgumentError` unless `|l1-l2| ≤ l3 ≤ l1+l2` and `l1+l2+l3` is even.
"""
function real_clebsch_gordan(l1::Integer, l2::Integer, l3::Integer)
    if !(abs(l1 - l2) <= l3 <= l1 + l2 && iseven(l1 + l2 + l3))
        throw(ArgumentError(lazy"real Clebsch-Gordan needs |l1-l2| ≤ l3 ≤ l1+l2 and even l1+l2+l3, got ($l1, $l2, $l3)"))
    end
    U1, U2, U3 = _complex_to_real(Int(l1)), _complex_to_real(Int(l2)), _complex_to_real(Int(l3))
    cg = zeros(ComplexF64, 2l1 + 1, 2l2 + 1, 2l3 + 1)
    for m1 in -l1:l1, m2 in -l2:l2
        m3 = m1 + m2
        abs(m3) <= l3 || continue
        cg[m1 + l1 + 1, m2 + l2 + 1, m3 + l3 + 1] = clebsch_gordan_complex(l1, m1, l2, m2, l3, m3)
    end
    C = _mode_product(_mode_product(_mode_product(cg, conj(U1), 1), conj(U2), 2), U3, 3) ./ sqrt(2l3 + 1)
    maximum(abs ∘ imag, C) < 1.0e-12 || error("real Clebsch-Gordan tensor has a nonzero imaginary part for ($l1, $l2, $l3)")
    return real.(C)
end
