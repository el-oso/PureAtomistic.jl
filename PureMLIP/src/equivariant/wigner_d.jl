# Real Wigner D matrices for the real spherical harmonics of spherical_harmonics.jl.

# n unit vectors from the Fibonacci spiral on the sphere.
function _fibonacci_sphere(n::Int)
    pts = Matrix{Float64}(undef, 3, n)
    golden = pi * (3 - sqrt(5.0))
    for i in 1:n
        z = 1 - 2 * (i - 0.5) / n
        r = sqrt(1 - z^2)
        pts[1, i] = r * cos(golden * i)
        pts[2, i] = r * sin(golden * i)
        pts[3, i] = z
    end
    return pts
end

"""
    wigner_D(l, R) -> Matrix{Float64}

Real Wigner matrix `D` of size `(2l+1, 2l+1)` for the rotation `R` (3×3, `R'R = I`,
`det R = 1`, each to 1e-12), defined by `Y_l(R v) = D * Y_l(v)` for every `v`, where `Y_l` is
the length-`2l+1` block (`m = -l..l`) of [`spherical_harmonics!`](@ref). `D` is orthogonal and
`wigner_D(l, R1 * R2) = wigner_D(l, R1) * wigner_D(l, R2)`.

It is found by least squares on `2(2l+1)` fixed, well-spread unit vectors.
"""
function wigner_D(l::Integer, R::AbstractMatrix{<:Real})
    Base.require_one_based_indexing(R)
    size(R) == (3, 3) || throw(ArgumentError(lazy"rotation must be 3×3, got size $(size(R))"))
    Rf = Matrix{Float64}(R)
    if !(maximum(abs, Rf' * Rf - I3) <= 1e-12 && abs(_det3(Rf) - 1) <= 1e-12)
        throw(ArgumentError("R is not a rotation: need R'R = I and det R = 1 to 1e-12"))
    end
    l >= 0 || throw(ArgumentError(lazy"l must be non-negative, got $l"))
    n = 2l + 1
    npts = 2n
    pts = _fibonacci_sphere(npts)
    Y = Vector{Float64}(undef, sh_length(l))
    Yr = Vector{Float64}(undef, sh_length(l))
    rows = sh_index(l, -l):sh_index(l, l)
    A = Matrix{Float64}(undef, npts, n)
    B = Matrix{Float64}(undef, npts, n)
    for p in 1:npts
        v = pts[:, p]
        spherical_harmonics!(Y, v, l)
        spherical_harmonics!(Yr, Rf * v, l)
        A[p, :] = Y[rows]
        B[p, :] = Yr[rows]
    end
    # Rows of B satisfy B[p, :] = D * A[p, :], i.e. B = A * D'.
    return Matrix{Float64}((A \ B)')
end

const I3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

_det3(M) = M[1, 1] * (M[2, 2] * M[3, 3] - M[2, 3] * M[3, 2]) -
           M[1, 2] * (M[2, 1] * M[3, 3] - M[2, 3] * M[3, 1]) +
           M[1, 3] * (M[2, 1] * M[3, 2] - M[2, 2] * M[3, 1])
