# Geometry optimization by a trust-region limited-memory BFGS method.
#
# The trust region (the box |pᵢ| ≤ Δ of largest coordinate displacement), the radius and acceptance
# rules, the rounding guard and the fixed-atom and frozen-direction handling are those of
# `TrustRegionBFGS`, and `relax!` is shared. The curvature is the limited-memory inverse BFGS
# matrix H of the last `memory` pairs (s, y) that passed the curvature test, with H₀ = γI and
# γ = sᵀy / yᵀy of the newest pair (I/alpha before the first), applied by the two-loop recursion
# (Nocedal and Wright, Numerical Optimization, 2nd ed., Springer, 2006; the recursion is re-derived
# here and verified by the tests).
#
# Every gradient, step and curvature pair is projected onto the subspace F the constraints leave
# free (the coordinates of fixed atoms zero, no component along a frozen direction), so H maps F
# into itself and the two-loop step p = -Hg lies in F: it is the minimizer of the model
# gᵀp + ½ pᵀH⁻¹p over F. When it also lies in the box it is the step, and no other solve runs.
#
# When the box binds, the model is minimized over the box within the subspace V = span{g, sᵢ, yᵢ},
# by PureDAQP. V contains -Hg and -g, and H maps V into itself (the two-loop recursion only adds
# multiples of the sᵢ and yᵢ to γv), so with an orthonormal basis Q of V the model restricted to V
# has the curvature QᵀH⁻¹Q = (QᵀHQ)⁻¹ = K⁻¹. With K = LLᵀ and p = QLw the problem handed to the
# solver is ½‖w‖² + (LᵀQᵀg)ᵀw over the rows of QL, each normalized, with the box bounds divided by
# the same norms: an identity `Diagonal` objective over at most 2·memory + 1 variables. The
# solver holds the row matrix by reference and its rows are rewritten in place, as for
# TrustRegionBFGS, so nothing is allocated per step. The step is the minimizer over the box
# restricted to V, not over the whole box: an exact box solve needs a working set of up to 3n rows,
# which is O(n²) memory. It agrees with the full-box step whenever the box does not bind, and it
# never decreases the model less than the steepest-descent point inside the box, which also lies
# in V (up to rounding when a nearly dependent vector is dropped from the basis).
#
# Storage is O(memory · n): the pairs, and the n × (2·memory + 1) basis that is also the solver's
# row matrix.

"""
    TrustRegionLBFGS{T}(natoms; memory = 10, radius = 0.1, max_radius = 0.5, alpha = 70.0,
                        eta = 1e-4, fixed = Int[], frozen = Pair{Int, NTuple{3, Float64}}[],
                        solver = ActiveSet())
    TrustRegionLBFGS(sys::System; kwargs...)

The state of a limited-memory trust-region BFGS minimizer for [`relax!`](@ref). The trust region,
`radius`, `max_radius`, `eta`, `alpha` (the starting inverse Hessian `I/alpha`), `fixed` and
`frozen` behave as for [`TrustRegionBFGS`](@ref). The inverse Hessian is that of the last
`memory` curvature pairs with initial matrix `(sᵀy / yᵀy) I` from the newest pair, applied by the
two-loop recursion; a pair is stored after every step with `sᵀy > √eps(T) |s| |y|`, kept or not.
`memory = 10` is the smallest of the values from 1 to 40 measured on two 256-atom Cu cells with
MACE-MPA-0 that took the fewest force evaluations (`bench/relax_tune.jl`,
`bench/results/relax_tune_mac.json`).

The default `solver` is `ActiveSet()` (the `:rows` working set), whereas `TrustRegionBFGS` uses
`working_set = :gram`: the subspace problems of this method, with at most `2·memory + 1`
variables, were not measured with `:gram`.

Each step is `-Hg` when that lies inside the box. Otherwise PureDAQP's dual active-set method
(`solver`) minimizes the model over the box within the span of the gradient and the stored
pairs, which contains `-Hg` and `-g`. The state holds about `4·memory + 23` numbers per coordinate,
and a relaxation with a given state is allocation-free. `boxed` counts the steps of the last
relaxation that needed the box solve.
"""
mutable struct TrustRegionLBFGS{T <: AbstractFloat, W}
    const S::Matrix{T}
    const Y::Matrix{T}
    const rho::Vector{T}
    const coef::Vector{T}
    const M::Matrix{T}
    const rownorm::Vector{T}
    const K::Matrix{T}
    const lower::Vector{T}
    const upper::Vector{T}
    const q::Vector{T}
    const c::Vector{T}
    const hv::Vector{T}
    const grad::Matrix{T}
    const gradv::Vector{T}
    const trial::Matrix{T}
    const trialv::Vector{T}
    const step::Matrix{T}
    const stepv::Vector{T}
    const y::Vector{T}
    const start::Matrix{T}
    const fixed::Vector{Bool}
    const frozen_atoms::Vector{Int}
    const frozen_dirs::Matrix{T}
    const qp::W
    const memory::Int
    const radius0::T
    const max_radius::T
    const alpha::T
    const eta::T
    radius::T
    gamma::T
    npairs::Int
    newest::Int
    rank::Int
    boxed::Int
end

function TrustRegionLBFGS{T}(
        natoms::Integer; memory::Integer = 10, radius::Real = 0.1, max_radius::Real = 0.5, alpha::Real = 70.0,
        eta::Real = 1.0e-4, fixed = Int[], frozen = Pair{Int, NTuple{3, Float64}}[], solver::ActiveSet = ActiveSet()
    ) where {T <: AbstractFloat}
    natoms >= 1 || throw(ArgumentError(lazy"pureatomistic: TrustRegionLBFGS needs at least one atom, got $natoms"))
    memory >= 1 || throw(ArgumentError(lazy"pureatomistic: TrustRegionLBFGS memory must be at least 1, got $memory"))
    _check_trust_parameters("TrustRegionLBFGS", radius, max_radius, alpha, eta)
    N = Int(natoms)
    isfixed = _fixed_mask(N, fixed)
    frozen_atoms, frozen_dirs = _frozen_rows(T, N, isfixed, frozen)
    n = 3N
    m = Int(memory)
    r = 2m + 1
    # Set up with unit coordinate rows, so the solver's row scales are exactly one; the rows are
    # rewritten before every solve.
    M = zeros(T, n, r)
    for i in 1:n
        M[i, mod1(i, r)] = one(T)
    end
    lower = fill(-T(radius), n)
    upper = fill(T(radius), n)
    q = zeros(T, r)
    # Enough iterations for every row to enter and leave the working set several times.
    qp = PureDAQP.setup(Diagonal(ones(T, r)), q, M, lower, upper, solver; max_iter = 10 * (n + r))
    grad = zeros(T, 3, N)
    trial = zeros(T, 3, N)
    step = zeros(T, 3, N)
    return TrustRegionLBFGS{T, typeof(qp)}(
        zeros(T, n, m), zeros(T, n, m), zeros(T, m), zeros(T, m), M, ones(T, n), zeros(T, r, r), lower, upper, q,
        zeros(T, r), zeros(T, n), grad, vec(grad), trial, vec(trial), step, vec(step), zeros(T, n), zeros(T, 3, N),
        isfixed, frozen_atoms, frozen_dirs, qp, m, T(radius), T(max_radius), T(alpha), T(eta), T(radius),
        inv(T(alpha)), 0, 0, 0, 0
    )
end

TrustRegionLBFGS(sys::System{T}; kwargs...) where {T} = TrustRegionLBFGS{T}(natoms(sys); kwargs...)

function _reset!(opt::TrustRegionLBFGS{T}) where {T}
    opt.radius = opt.radius0
    opt.gamma = inv(opt.alpha)
    opt.npairs = 0
    opt.newest = 0
    opt.boxed = 0
    return opt
end

# Column of S and Y holding the k-th newest pair (k = 1 is the newest).
_pair_column(opt::TrustRegionLBFGS, k::Int) = mod1(opt.newest - k + 1, opt.memory)

"""
    _two_loop!(r, opt, v) -> r

`r = Hv` for the limited-memory inverse BFGS matrix of `opt`, by the two-loop recursion.
Allocation-free.
"""
function _two_loop!(r::AbstractVector{T}, opt::TrustRegionLBFGS{T}, v::AbstractVector{T}) where {T}
    S, Y, coef = opt.S, opt.Y, opt.coef
    for i in eachindex(r, v)
        r[i] = v[i]
    end
    for k in 1:opt.npairs
        j = _pair_column(opt, k)
        a = zero(T)
        for i in eachindex(r)
            a += S[i, j] * r[i]
        end
        a *= opt.rho[j]
        coef[j] = a
        for i in eachindex(r)
            r[i] -= a * Y[i, j]
        end
    end
    for i in eachindex(r)
        r[i] *= opt.gamma
    end
    for k in opt.npairs:-1:1
        j = _pair_column(opt, k)
        b = zero(T)
        for i in eachindex(r)
            b += Y[i, j] * r[i]
        end
        b = coef[j] - opt.rho[j] * b
        for i in eachindex(r)
            r[i] += b * S[i, j]
        end
    end
    return r
end

# Store the pair (s, y) as the newest, replacing the oldest once `memory` are held, and take
# γ = sᵀy / yᵀy from it; skipped when the curvature condition sᵀy > √eps |s| |y| fails.
function _curvature_update!(opt::TrustRegionLBFGS{T}, s::AbstractVector{T}, y::AbstractVector{T}) where {T}
    sy = dot(s, y)
    yy = dot(y, y)
    sy > sqrt(eps(T)) * sqrt(dot(s, s) * yy) || return false
    length(s) == size(opt.S, 1) ||
        throw(DimensionMismatch(lazy"pureatomistic: a curvature pair of length $(length(s)) for $(size(opt.S, 1)) coordinates"))
    j = mod1(opt.newest + 1, opt.memory)
    for (i, k) in zip(axes(opt.S, 1), eachindex(s, y))
        opt.S[i, j] = s[k]
        opt.Y[i, j] = y[k]
    end
    opt.rho[j] = inv(sy)
    opt.gamma = sy / yy
    opt.newest = j
    opt.npairs = min(opt.npairs + 1, opt.memory)
    return true
end

"""
    _trust_step!(opt::TrustRegionLBFGS) -> predicted decrease

Write into `opt.step` the step of the model `gᵀp + ½ pᵀH⁻¹p` (`g = opt.grad`) for the box
`|pᵢ| ≤ opt.radius`: the two-loop step `-Hg` when it lies in the box, else the minimizer over the
box within the span of `g` and the stored pairs. Returns the model's decrease. Allocation-free.
"""
function _trust_step!(opt::TrustRegionLBFGS{T})::T where {T}
    g, p = opt.gradv, opt.stepv
    _two_loop!(p, opt, g)
    for i in eachindex(p)
        p[i] = -p[i]
    end
    _project!(opt.step, opt)
    if _max_abs(p) <= opt.radius
        # H⁻¹p = -g, so the model's decrease is -(gᵀp + ½ pᵀH⁻¹p) = -½ gᵀp.
        return -dot(g, p) / 2
    end
    opt.boxed += 1
    _subspace_problem!(opt)
    qp = opt.qp
    PureDAQP.update!(qp; q = opt.q, l = opt.lower, u = opt.upper)
    PureDAQP.cold_start!(qp)
    sol = PureDAQP.solve!(qp)
    sol.status == PureDAQP.SOLVED ||
        throw(ArgumentError(lazy"pureatomistic: the trust-region step problem was not solved: $(sol.status)"))
    return _subspace_step!(opt, qp.x)
end

# Column j of M dotted with v, and the Euclidean norm of column j, as loops over the column: a view
# of the column handed to `dot` reads to the allocation scan as an escaping allocation. `v` has the
# row axis of M.
@inline function _col_dot(M::AbstractMatrix{T}, j::Int, v::AbstractVector{T}) where {T}
    s = zero(T)
    for i in eachindex(v)
        s += M[i, j] * v[i]
    end
    return s
end

@inline function _col_norm(M::AbstractMatrix{T}, j::Int) where {T}
    s = zero(T)
    for i in axes(M, 1)
        s += M[i, j]^2
    end
    return sqrt(s)
end

# Columns 1:rank of opt.M ← an orthonormal basis of span{g, sᵢ, yᵢ} by modified Gram-Schmidt,
# applied twice to each vector; a vector whose norm falls below √eps(T) of its own is dropped.
function _subspace_basis!(opt::TrustRegionLBFGS{T}) where {T}
    M = opt.M
    rank = 0
    for k in 0:(2 * opt.npairs)
        col = rank + 1
        v = view(M, :, col)
        if iszero(k)
            for i in eachindex(v, opt.gradv)
                v[i] = opt.gradv[i]
            end
        else
            src = k <= opt.npairs ? opt.S : opt.Y
            jsrc = _pair_column(opt, k <= opt.npairs ? k : k - opt.npairs)
            for i in eachindex(v)
                v[i] = src[i, jsrc]
            end
        end
        nv0 = _col_norm(M, col)
        iszero(nv0) && continue
        for _ in 1:2, j in 1:rank
            c = _col_dot(M, j, v)
            for i in eachindex(v)
                v[i] -= c * M[i, j]
            end
        end
        nv = _col_norm(M, col)
        nv > sqrt(eps(T)) * nv0 || continue
        for i in eachindex(v)
            v[i] /= nv
        end
        rank = col
    end
    opt.rank = rank
    return rank
end

# The step problem over the span V: the basis Q in opt.M, K = QᵀHQ = LLᵀ, the rows of QL normalized
# in opt.M with their norms in opt.rownorm and the box bounds divided by them, q = LᵀQᵀg.
function _subspace_problem!(opt::TrustRegionLBFGS{T}) where {T}
    M, K, hv, c = opt.M, opt.K, opt.hv, opt.c
    r = _subspace_basis!(opt)
    fill!(K, zero(T))
    for j in 1:r
        _two_loop!(hv, opt, view(M, :, j))
        for i in 1:r
            K[i, j] = _col_dot(M, i, hv)
        end
    end
    for j in 1:r, i in (j + 1):r
        K[i, j] = K[j, i] = (K[i, j] + K[j, i]) / 2
    end
    Kr = view(K, 1:r, 1:r)
    C = cholesky!(Symmetric(Kr, :L); check = false)
    issuccess(C) || throw(ArgumentError("pureatomistic: the limited-memory inverse Hessian is not positive definite on the step subspace"))
    # c = Qᵀg, then q = Lᵀc
    for j in 1:r
        c[j] = _col_dot(M, j, opt.gradv)
    end
    fill!(opt.q, zero(T))
    for j in 1:r
        s = zero(T)
        for i in j:r
            s += K[i, j] * c[i]
        end
        opt.q[j] = s
    end
    # Rows of QL in place: column j of QL combines columns j:r of Q, which are still unchanged
    # when the columns are formed in increasing order.
    for i in axes(M, 1)
        for j in 1:r
            s = zero(T)
            for k in j:r
                s += M[i, k] * K[k, j]
            end
            M[i, j] = s
        end
        for j in (r + 1):size(M, 2)
            M[i, j] = zero(T)
        end
    end
    Δ = opt.radius
    for i in axes(M, 1)
        s = zero(T)
        for j in 1:r
            s += M[i, j]^2
        end
        nrm = sqrt(s)
        # A coordinate no basis vector moves (one of a fixed atom) keeps a zero row, which every w
        # meets; its bounds stay those of an inequality, since an equality row of zeros would be
        # singular in the working set.
        if iszero(nrm)
            opt.rownorm[i] = one(T)
        else
            opt.rownorm[i] = nrm
            for j in 1:r
                M[i, j] /= nrm
            end
        end
        opt.lower[i] = -Δ / opt.rownorm[i]
        opt.upper[i] = Δ / opt.rownorm[i]
    end
    return nothing
end

# The step p = QLw from the solution w of the subspace problem (row i of QL is rownorm[i] times
# row i of opt.M), with the frozen components removed; returns the model decrease
# -(qᵀw + ½ wᵀw).
function _subspace_step!(opt::TrustRegionLBFGS{T}, w::AbstractVector{T})::T where {T}
    M, q = opt.M, opt.q
    pred = zero(T)
    for j in eachindex(q, w)
        pred -= q[j] * w[j] + w[j]^2 / 2
    end
    p = opt.stepv
    for i in eachindex(p)
        s = zero(T)
        for j in eachindex(w)
            s += M[i, j] * w[j]
        end
        p[i] = opt.rownorm[i] * s
    end
    _project!(opt.step, opt)
    return pred
end
