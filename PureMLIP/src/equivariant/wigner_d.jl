# Real Wigner D matrices for the real spherical harmonics of spherical_harmonics.jl.

# Gauss-Legendre nodes and weights on [-1, 1] by Newton iteration on P_n.
function _gauss_legendre(n::Int)
    x = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)
    for k in 1:n
        z = cos(pi * (k - 0.25) / (n + 0.5))
        dp = 1.0
        for _ in 1:100
            p0, p1 = 1.0, z
            for j in 2:n
                p0, p1 = p1, ((2j - 1) * z * p1 - (j - 1) * p0) / j
            end
            dp = n * (z * p1 - p0) / (z^2 - 1)
            dz = p1 / dp
            z -= dz
            abs(dz) <= 2eps(abs(z)) && break
        end
        x[k] = z
        w[k] = 2 / ((1 - z^2) * dp^2)
    end
    return x, w
end

"""
    wigner_D(l, R) -> Matrix{Float64}

Real Wigner matrix `D` of size `(2l+1, 2l+1)` for the rotation `R` (3×3, `R'R = I`,
`det R = 1`, each to 1e-12), defined by `Y_l(R v) = D * Y_l(v)` for every `v`, where `Y_l` is
the length-`2l+1` block (`m = -l..l`) of [`spherical_harmonics!`](@ref). `D` is orthogonal and
`wigner_D(l, R1 * R2) = wigner_D(l, R1) * wigner_D(l, R2)`.

Computed as `D[i, j] = ∫ Y_i(R u) Y_j(u) dΩ` (the harmonics are orthonormal) by Gauss-Legendre
quadrature in `cos θ` (`l + 1` nodes) times uniform `φ` (`2l + 1` points), which is exact for
the degree-`2l` integrand up to rounding.
"""
function wigner_D(l::Integer, R::AbstractMatrix{<:Real})
    l >= 0 || throw(ArgumentError(lazy"l must be non-negative, got $l"))
    size(R) == (3, 3) || throw(ArgumentError(lazy"rotation must be 3×3, got size $(size(R))"))
    Base.require_one_based_indexing(R)
    Rf = Matrix{Float64}(R)
    if !(norm(Rf' * Rf - I, Inf) <= 1.0e-12 && abs(det(Rf) - 1) <= 1.0e-12)
        throw(ArgumentError("R is not a rotation: need R'R = I and det R = 1 to 1e-12"))
    end
    L = Int(l)
    n = 2L + 1
    nφ = n
    xs, ws = _gauss_legendre(L + 1)
    Q = length(xs) * nφ
    rows = sh_index(L, -L):sh_index(L, L)
    Y = Vector{Float64}(undef, sh_length(L))
    Ys = Matrix{Float64}(undef, n, Q)
    Yrs = Matrix{Float64}(undef, n, Q)
    w = Vector{Float64}(undef, Q)
    u = Vector{Float64}(undef, 3)
    v = Vector{Float64}(undef, 3)
    q = 0
    for k in eachindex(xs), p in 0:(nφ - 1)
        q += 1
        ct = xs[k]
        st = sqrt(1 - ct^2)
        φ = 2pi * p / nφ
        u[1] = st * cos(φ); u[2] = st * sin(φ); u[3] = ct
        mul!(v, Rf, u)
        spherical_harmonics!(Y, u, L)
        copyto!(view(Ys, :, q), view(Y, rows))
        spherical_harmonics!(Y, v, L)
        copyto!(view(Yrs, :, q), view(Y, rows))
        w[q] = ws[k] * 2pi / nφ
    end
    return (Yrs * Diagonal(w)) * Ys'
end
