# Geometry optimization by a trust-region quasi-Newton method with BFGS curvature.
#
# Every step minimizes the quadratic model m(p) = gᵀp + ½ pᵀBp of the energy, with g the
# gradient (minus the forces) and B = H⁻¹ the BFGS approximation of the Hessian, over the box
# |pᵢ| ≤ Δ of largest coordinate displacement, with p = 0 on the coordinates of fixed atoms and
# dᵀp_a = 0 for every frozen direction d of atom a. The box is the ∞-norm trust region of
# Nocedal and Wright, Numerical Optimization, 2nd ed. (Springer, 2006), Algorithm 4.1, whose
# radius and acceptance rules are used unchanged; H is updated by the inverse BFGS formula,
# their eq. (6.17), and the initial H₀ = I/α is rescaled by their eq. (6.20) before the first
# update. The equation and algorithm numbers are cited from memory and were not checked against
# the text of the book.
#
# The step problem is solved by PureDAQP's dual active-set method. With H = LLᵀ (Cholesky) and
# p = Lw the model becomes (Lᵀg)ᵀw + ½‖w‖², so the problem handed to the solver has the identity
# as its objective matrix and the rows of C L as its constraint matrix, where C holds the box
# rows (the identity) and the frozen-direction rows. Each row is normalized by its own norm,
# and its bounds with it. The solver is set up once, with an identity `Diagonal` objective;
# for that objective it keeps the constraint matrix it was given, not a copy, and reads its rows
# from there, with every row scale one because every row it was set up with is a unit
# coordinate vector. Each step rewrites the rows of that matrix in place, hands the solver the new
# linear term and bounds, and cold-starts it, since the working set it holds was factored from
# the previous rows. Nothing is allocated per step.
#
# Storage is three dense 3n × 3n matrices and one (3n + k) × 3n matrix for n atoms and k frozen
# directions, and each step factors H, which is O(n³): the method is meant for cells of up to
# about a thousand atoms.

"""
    TrustRegionBFGS{T}(natoms; radius = 0.1, max_radius = 0.5, alpha = 70.0, eta = 1e-4,
                       fixed = Int[], frozen = Pair{Int, NTuple{3, Float64}}[],
                       solver = ActiveSet())
    TrustRegionBFGS(sys::System; kwargs...)

The state of a trust-region BFGS minimizer for [`relax!`](@ref). Each step minimizes the
quadratic model of the energy, with the gradient from the forces and BFGS curvature, inside a
box of maximum coordinate displacement (the trust region, `radius` in Å), with PureDAQP's
dual active-set method (`solver`, an `ActiveSet`).

- `radius` is the starting trust radius and `max_radius` its ceiling (Å). The radius is
  quartered after a step whose energy decrease is less than a quarter of the model's, and
  doubled after one that reaches the box with a decrease above three quarters of the model's
  (Nocedal and Wright, Numerical Optimization, 2006, Algorithm 4.1).
- A step is kept when its energy decrease is more than `eta` times the model's, and undone
  otherwise. When the model's decrease is below the rounding of the energies
  (`4 eps(T) max(|E|, |E_new|)`) the ratio carries no information: the step is kept if the
  energy did not rise by more than that, and the radius is left alone.
- `alpha` (eV/Å²) sets the starting inverse Hessian `I/alpha`; it is rescaled to
  `(sᵀy / yᵀy) I` before the first update (Nocedal and Wright, eq. 6.20), and every step with
  `sᵀy > √eps(T) |s| |y|` updates it by the inverse BFGS formula (eq. 6.17), whether or not
  the step is kept.
- `fixed` lists atoms that do not move; `frozen` lists `atom => direction` pairs along which
  an atom does not move (the directions of one atom must be orthogonal). Both are equality rows
  of the step problem. Their force components are left out of the gradient and out of the
  convergence test.
- `solver` is the PureDAQP `ActiveSet` that solves each step problem; the default factors the
  working-set rows (`working_set = :rows`) and scans every row (`scan = :all`). On the step
  problems of relaxing two 256-atom Cu cells with MACE-MPA-0, at the radius the relaxation used
  and at a half and a tenth of the unconstrained step (`bench/step_problems.jl`), the working-set
  rows had condition numbers below 3.1, so `working_set = :gram` (which squares it) solved them
  1.1 to 1.5 times faster wherever the box bound; it is not the default because its solve path
  in PureDAQP 0.1.10 does not pass TrimCheck. `scan = :window` was slower in total on each of
  these sets, since the solver prices every row with one product here, and it failed on one
  problem that `:all` solves. The solve is a small part of a step: on the vacancy cell 0.33 ms
  against 3.3 ms for the factorization of H and about 0.56 s for the force evaluation. A step
  problem the solver does not solve throws.

The state holds dense `3n × 3n` matrices and every step factors one, so the method suits cells
of up to about a thousand atoms. A relaxation with a given state is allocation-free.
"""
mutable struct TrustRegionBFGS{T <: AbstractFloat, W}
    const H::Matrix{T}
    const L::Matrix{T}
    const M::Matrix{T}
    const lower::Vector{T}
    const upper::Vector{T}
    const q::Vector{T}
    const grad::Matrix{T}
    const gradv::Vector{T}
    const trial::Matrix{T}
    const trialv::Vector{T}
    const step::Matrix{T}
    const stepv::Vector{T}
    const y::Vector{T}
    const Hy::Vector{T}
    const start::Matrix{T}
    const fixed::Vector{Bool}
    const frozen_atoms::Vector{Int}
    const frozen_dirs::Matrix{T}
    const qp::W
    const radius0::T
    const max_radius::T
    const alpha::T
    const eta::T
    radius::T
    scaled::Bool
end

function _frozen_rows(::Type{T}, natoms::Int, fixed::AbstractVector{Bool}, frozen) where {T}
    atoms = Int[]
    dirs = NTuple{3, T}[]
    for (a, d) in frozen
        (1 <= a <= natoms) || throw(ArgumentError(lazy"pureatomistic: frozen atom $a is outside 1:$natoms"))
        fixed[a] && throw(ArgumentError(lazy"pureatomistic: atom $a is both fixed and frozen"))
        (length(d) == 3 && all(isfinite, d) && any(!iszero, d)) ||
            throw(ArgumentError(lazy"pureatomistic: the frozen direction of atom $a must be a finite nonzero 3-vector"))
        nd = sqrt(sum(abs2, d))
        u = (T(d[1] / nd), T(d[2] / nd), T(d[3] / nd))
        for (b, v) in zip(atoms, dirs)
            b == a || continue
            abs(u[1] * v[1] + u[2] * v[2] + u[3] * v[3]) <= sqrt(eps(T)) ||
                throw(ArgumentError(lazy"pureatomistic: the frozen directions of atom $a must be orthogonal"))
        end
        push!(atoms, a)
        push!(dirs, u)
    end
    D = zeros(T, 3, length(dirs))
    for (k, u) in pairs(dirs)
        D[1, k], D[2, k], D[3, k] = u
    end
    return atoms, D
end

function TrustRegionBFGS{T}(
        natoms::Integer; radius::Real = 0.1, max_radius::Real = 0.5, alpha::Real = 70.0, eta::Real = 1.0e-4,
        fixed = Int[], frozen = Pair{Int, NTuple{3, Float64}}[], solver::ActiveSet = ActiveSet()
    ) where {T <: AbstractFloat}
    natoms >= 1 || throw(ArgumentError(lazy"pureatomistic: TrustRegionBFGS needs at least one atom, got $natoms"))
    (isfinite(radius) && radius > 0) ||
        throw(ArgumentError(lazy"pureatomistic: TrustRegionBFGS radius must be positive and finite, got $radius"))
    (isfinite(max_radius) && max_radius >= radius) ||
        throw(ArgumentError(lazy"pureatomistic: TrustRegionBFGS max_radius must be finite and at least radius = $radius, got $max_radius"))
    (isfinite(alpha) && alpha > 0) ||
        throw(ArgumentError(lazy"pureatomistic: TrustRegionBFGS alpha must be positive and finite, got $alpha"))
    (0 <= eta < 1 / 4) || throw(ArgumentError(lazy"pureatomistic: TrustRegionBFGS eta must lie in [0, 1/4), got $eta"))
    N = Int(natoms)
    isfixed = zeros(Bool, N)
    for a in fixed
        (1 <= a <= N) || throw(ArgumentError(lazy"pureatomistic: fixed atom $a is outside 1:$N"))
        isfixed[a] = true
    end
    frozen_atoms, frozen_dirs = _frozen_rows(T, N, isfixed, frozen)
    n = 3N
    m = n + length(frozen_atoms)
    # Set up with unit coordinate rows, so the solver's row scales are exactly one; the rows are
    # rewritten before every solve.
    M = zeros(T, m, n)
    for i in 1:m
        M[i, i <= n ? i : 1] = one(T)
    end
    lower = fill(-T(radius), m)
    upper = fill(T(radius), m)
    q = zeros(T, n)
    # Enough iterations for every row to enter and leave the working set several times; the
    # active-set method needs about one iteration per row that ends up active.
    qp = PureDAQP.setup(Diagonal(ones(T, n)), q, M, lower, upper, solver; max_iter = 10 * (m + n))
    grad = zeros(T, 3, N)
    trial = zeros(T, 3, N)
    step = zeros(T, 3, N)
    return TrustRegionBFGS{T, typeof(qp)}(
        Matrix{T}(I / T(alpha), n, n), zeros(T, n, n), M, lower, upper, q,
        grad, vec(grad), trial, vec(trial), step, vec(step), zeros(T, n), zeros(T, n), zeros(T, 3, N),
        isfixed, frozen_atoms, frozen_dirs, qp, T(radius), T(max_radius), T(alpha), T(eta), T(radius), false
    )
end

TrustRegionBFGS(sys::System{T}; kwargs...) where {T} = TrustRegionBFGS{T}(natoms(sys); kwargs...)

function _reset!(opt::TrustRegionBFGS{T}) where {T}
    H = opt.H
    fill!(H, zero(T))
    for i in axes(H, 1)
        H[i, i] = inv(opt.alpha)
    end
    opt.radius = opt.radius0
    opt.scaled = false
    return opt
end

# Remove from the 3 × n matrix G the components the constraints hold: whole columns of fixed
# atoms and the component along each frozen direction (orthonormal per atom, so one pass each).
function _project!(G::AbstractMatrix{T}, opt::TrustRegionBFGS{T}) where {T}
    for a in axes(G, 2)
        if opt.fixed[a]
            G[1, a] = G[2, a] = G[3, a] = zero(T)
        end
    end
    D = opt.frozen_dirs
    for k in eachindex(opt.frozen_atoms)
        a = opt.frozen_atoms[k]
        c = D[1, k] * G[1, a] + D[2, k] * G[2, a] + D[3, k] * G[3, a]
        G[1, a] -= c * D[1, k]
        G[2, a] -= c * D[2, k]
        G[3, a] -= c * D[3, k]
    end
    return G
end

# The projected gradient -F into G.
function _gradient!(G::AbstractMatrix{T}, F::AbstractMatrix, opt::TrustRegionBFGS{T}) where {T}
    for a in axes(G, 2), d in axes(G, 1)
        G[d, a] = -F[d, a]
    end
    return _project!(G, opt)
end

"""
    _trust_step!(opt) -> predicted decrease

Write into `opt.step` the minimizer of the model `gᵀp + ½ pᵀH⁻¹p` (`g = opt.grad`, `H = opt.H`)
over the box `|pᵢ| ≤ opt.radius` and the constraint rows, and return the model's decrease
`-(gᵀp + ½ pᵀH⁻¹p)`. Allocation-free.
"""
function _trust_step!(opt::TrustRegionBFGS{T})::T where {T}
    _step_problem!(opt)
    qp = opt.qp
    PureDAQP.update!(qp; q = opt.q, l = opt.lower, u = opt.upper)
    PureDAQP.cold_start!(qp)
    sol = PureDAQP.solve!(qp)
    sol.status == PureDAQP.SOLVED ||
        throw(ArgumentError(lazy"pureatomistic: the trust-region step problem was not solved: $(sol.status)"))
    return _step_from_solution!(opt, qp.x)
end

# The step problem in the variables w of p = Lw, H = LLᵀ: the linear term q = Lᵀg, the rows of C L
# in `opt.M` and their bounds, each row normalized and its bounds divided by the same norm.
function _step_problem!(opt::TrustRegionBFGS{T}) where {T}
    H, L, M, q = opt.H, opt.L, opt.M, opt.q
    n = size(H, 1)
    copyto!(L, H)
    C = cholesky!(Symmetric(L, :L); check = false)
    issuccess(C) || throw(ArgumentError("pureatomistic: the BFGS inverse Hessian is not positive definite"))
    Δ = opt.radius
    fixed = opt.fixed
    lin = LinearIndices(opt.grad)
    cart = CartesianIndices(opt.grad)
    # Box rows: row i of L, normalized; the bounds are divided by the same norm.
    for i in 1:n
        s = zero(T)
        for j in 1:i
            s += L[i, j]^2
        end
        nrm = sqrt(s)
        for j in 1:i
            M[i, j] = L[i, j] / nrm
        end
        for j in (i + 1):n
            M[i, j] = zero(T)
        end
        bound = fixed[cart[i][2]] ? zero(T) : Δ / nrm
        opt.lower[i] = -bound
        opt.upper[i] = bound
    end
    # Frozen rows: dᵀ times the three rows of L that belong to the atom, normalized.
    D = opt.frozen_dirs
    for k in eachindex(opt.frozen_atoms)
        a = opt.frozen_atoms[k]
        r = n + k
        i1, i2, i3 = lin[1, a], lin[2, a], lin[3, a]
        s = zero(T)
        for j in 1:n
            v = (j <= i1 ? D[1, k] * L[i1, j] : zero(T)) + (j <= i2 ? D[2, k] * L[i2, j] : zero(T)) +
                (j <= i3 ? D[3, k] * L[i3, j] : zero(T))
            M[r, j] = v
            s += v^2
        end
        nrm = sqrt(s)
        for j in 1:n
            M[r, j] /= nrm
        end
        opt.lower[r] = opt.upper[r] = zero(T)
    end
    # q = Lᵀg
    g = opt.gradv
    for j in 1:n
        s = zero(T)
        for i in j:n
            s += L[i, j] * g[i]
        end
        q[j] = s
    end
    return nothing
end

# The step p = Lw from the solution w of the step problem, with the equality rows made exact; returns
# the model decrease -(qᵀw + ½ wᵀw).
function _step_from_solution!(opt::TrustRegionBFGS{T}, w::AbstractVector{T})::T where {T}
    L, q = opt.L, opt.q
    n = length(q)
    pred = zero(T)
    for j in 1:n
        pred -= q[j] * w[j] + w[j]^2 / 2
    end
    # p = L w
    p = opt.stepv
    fill!(p, zero(T))
    for j in 1:n
        wj = w[j]
        for i in j:n
            p[i] += L[i, j] * wj
        end
    end
    # The solver meets the equality rows to rounding; they are made exact here.
    _project!(opt.step, opt)
    return pred
end

# Largest absolute entry of a vector.
function _max_abs(v::AbstractVector{T}) where {T}
    m = zero(T)
    for x in v
        m = max(m, abs(x))
    end
    return m
end

# Inverse BFGS update of H with the step s and gradient change y; skipped when the curvature
# condition sᵀy > √eps |s| |y| fails. The first update replaces H₀ by (sᵀy / yᵀy) I.
function _bfgs_update!(opt::TrustRegionBFGS{T}, s::AbstractVector{T}, y::AbstractVector{T}) where {T}
    H, Hy = opt.H, opt.Hy
    sy = dot(s, y)
    yy = dot(y, y)
    sy > sqrt(eps(T)) * sqrt(dot(s, s) * yy) || return false
    if !opt.scaled
        fill!(H, zero(T))
        for i in axes(H, 1)
            H[i, i] = sy / yy
        end
        opt.scaled = true
    end
    mul!(Hy, H, y)
    ρ = inv(sy)
    c = ρ^2 * dot(y, Hy) + ρ
    for j in axes(H, 2), i in axes(H, 1)
        H[i, j] += c * s[i] * s[j] - ρ * (s[i] * Hy[j] + Hy[i] * s[j])
    end
    return true
end

function _evaluate!(potential::AbstractPotential, sys::System)
    prepare!(potential, sys.positions, sys.cell)
    energy_forces!(potential, sys.positions, sys.cell)
    return nothing
end

"""
    relax!(sys, potential, opt::TrustRegionBFGS; fmax = 1e-3, maxsteps = 10_000) -> RelaxResult

Minimize the energy of `potential` over the positions of `sys` at fixed cell with the
trust-region BFGS method of [`TrustRegionBFGS`](@ref), updating `sys.positions` in place, until
the largest atomic force the constraints leave free is below `fmax` (eV/Å) or `maxsteps` steps
are taken. Every step costs one force evaluation, whether it is kept or undone; `steps` counts
them. `max_force` in the result is that of the free force components. The state is reset at the
start, and the call is allocation-free after the potential's first evaluation. On return the
potential holds the evaluation of the returned positions.
"""
function relax!(
        sys::System{T}, potential::AbstractPotential, opt::TrustRegionBFGS{T}; fmax::Real = 1.0e-3,
        maxsteps::Integer = 10_000
    ) where {T}
    _check_relax_arguments(sys, potential, fmax, maxsteps)
    size(opt.grad, 2) == natoms(sys) ||
        throw(DimensionMismatch(lazy"pureatomistic: the TrustRegionBFGS state has $(size(opt.grad, 2)) atoms, the system $(natoms(sys))"))
    _reset!(opt)
    tol = T(fmax)
    X = sys.positions
    _evaluate!(potential, sys)
    _status(potential, 0)
    E = potential.result.energy
    _gradient!(opt.grad, potential.result.forces, opt)
    current = true    # the potential holds the evaluation of X
    for step in 0:(maxsteps - 1)
        fm = _max_force(opt.grad)
        fm < tol && return _finish(sys, potential, opt, current, true, step, E, fm)
        pred = _trust_step!(opt)
        copyto!(opt.start, X)
        for a in axes(X, 2), d in axes(X, 1)
            X[d, a] += opt.step[d, a]
        end
        _evaluate!(potential, sys)
        _status(potential, step + 1)
        Enew = potential.result.energy
        _gradient!(opt.trial, potential.result.forces, opt)
        for i in eachindex(opt.y, opt.trialv, opt.gradv)
            opt.y[i] = opt.trialv[i] - opt.gradv[i]
        end
        _bfgs_update!(opt, opt.stepv, opt.y)
        actual = E - Enew
        noise = 4 * eps(T) * max(abs(E), abs(Enew))
        if pred <= noise
            keep = actual >= -noise
        else
            ratio = actual / pred
            if ratio < T(1 // 4)
                opt.radius /= 4
            elseif ratio > T(3 // 4) && _max_abs(opt.stepv) >= opt.radius * (1 - sqrt(eps(T)))
                opt.radius = min(2 * opt.radius, opt.max_radius)
            end
            keep = ratio > opt.eta
        end
        if keep
            E = Enew
            copyto!(opt.grad, opt.trial)
            current = true
        else
            copyto!(X, opt.start)
            current = false
        end
    end
    fm = _max_force(opt.grad)
    return _finish(sys, potential, opt, current, fm < tol, Int(maxsteps), E, fm)
end

# The result, after evaluating the potential again when its last evaluation was of an undone step.
function _finish(sys::System{T}, potential, opt, current::Bool, converged::Bool, steps::Int, E::T, fm::T) where {T}
    current || _evaluate!(potential, sys)
    return RelaxResult{T}(converged, steps, E, fm)
end
