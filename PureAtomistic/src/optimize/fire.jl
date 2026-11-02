# Geometry optimization by FIRE, the fast inertial relaxation engine of Bitzek, Koskinen,
# Gähler, Moseler and Gumbsch, Phys. Rev. Lett. 97, 170201 (2006).
#
# The positions x, velocities v and forces F are treated as global 3N-dimensional vectors with
# unit masses, so v, F and the time step carry pseudo-dynamical units (the time step is a
# unitless scale, not femtoseconds). One iteration is the paper's rule set with a semi-implicit
# Euler integrator (v += dt F, then x += dt v):
#   F1  P = F · v
#   F2  if P > 0: v ← (1 - α) v + α |v| F/|F|
#   F3  if P > 0 for more than nmin consecutive steps: dt ← min(dt finc, dtmax), α ← α falpha
#   F4  if P ≤ 0: dt ← dt fdec, v ← 0, α ← alpha_start
# The displacement of one iteration is additionally capped at `maxstep` per atom, which keeps
# the first steps of a far-from-equilibrium configuration inside the range of the potential.

"""
    FIRE{T}(natoms; dt = 0.1, dtmax = 1.0, maxstep = 0.1, nmin = 5, finc = 1.1,
            fdec = 0.5, alpha_start = 0.1, falpha = 0.99)
    FIRE(sys::System; kwargs...)

The state and parameters of the FIRE minimizer (Bitzek et al., Phys. Rev. Lett. 97, 170201,
2006) for `natoms` atoms: the velocities, the current time step `dt`, the mixing parameter
`alpha` and the count `npos` of steps since the power `F · v` was last non-positive.

`nmin`, `finc`, `fdec`, `alpha_start` and `falpha` default to the values the paper reports
(`N_min = 5`, `f_inc = 1.1`, `f_dec = 0.5`, `α_start = 0.1`, `f_α = 0.99`). `dt` is the
starting time step and `dtmax` its ceiling; the paper estimates `Δt_max` as about ten times the
time step of a molecular dynamics run, and the defaults keep that ratio. All atoms have unit
mass and the forces are in eV/Å, so `dt` is a unitless scale. `maxstep` (Å) caps the largest
displacement of any atom in one step.
"""
mutable struct FIRE{T <: AbstractFloat}
    const velocity::Matrix{T}
    const dt0::T
    const dtmax::T
    const maxstep::T
    const finc::T
    const fdec::T
    const alpha_start::T
    const falpha::T
    const nmin::Int
    dt::T
    alpha::T
    npos::Int
end

function FIRE{T}(
        natoms::Integer; dt::Real = 0.1, dtmax::Real = 1.0, maxstep::Real = 0.1, nmin::Integer = 5,
        finc::Real = 1.1, fdec::Real = 0.5, alpha_start::Real = 0.1, falpha::Real = 0.99
    ) where {T <: AbstractFloat}
    natoms >= 0 || throw(ArgumentError(lazy"pureatomistic: the number of atoms must not be negative, got $natoms"))
    (isfinite(dt) && dt > 0) || throw(ArgumentError(lazy"pureatomistic: FIRE dt must be positive and finite, got $dt"))
    (isfinite(dtmax) && dtmax >= dt) ||
        throw(ArgumentError(lazy"pureatomistic: FIRE dtmax must be finite and at least dt = $dt, got $dtmax"))
    (isfinite(maxstep) && maxstep > 0) ||
        throw(ArgumentError(lazy"pureatomistic: FIRE maxstep must be positive and finite, got $maxstep"))
    nmin >= 0 || throw(ArgumentError(lazy"pureatomistic: FIRE nmin must not be negative, got $nmin"))
    (isfinite(finc) && finc > 1) || throw(ArgumentError(lazy"pureatomistic: FIRE finc must exceed 1, got $finc"))
    (0 < fdec < 1) || throw(ArgumentError(lazy"pureatomistic: FIRE fdec must lie in (0, 1), got $fdec"))
    (0 < alpha_start <= 1) ||
        throw(ArgumentError(lazy"pureatomistic: FIRE alpha_start must lie in (0, 1], got $alpha_start"))
    (0 < falpha <= 1) || throw(ArgumentError(lazy"pureatomistic: FIRE falpha must lie in (0, 1], got $falpha"))
    return FIRE{T}(
        zeros(T, 3, Int(natoms)), T(dt), T(dtmax), T(maxstep), T(finc), T(fdec), T(alpha_start), T(falpha),
        Int(nmin), T(dt), T(alpha_start), 0
    )
end

FIRE(sys::System{T}; kwargs...) where {T} = FIRE{T}(natoms(sys); kwargs...)

# Restore the state of a freshly built optimizer: no velocity, the starting step and mixing.
function _reset!(opt::FIRE{T}) where {T}
    fill!(opt.velocity, zero(T))
    opt.dt = opt.dt0
    opt.alpha = opt.alpha_start
    opt.npos = 0
    return opt
end

# Largest Euclidean norm of a column of the 3 × n force matrix F, in eV/Å. NaN propagates.
function _max_force(F::AbstractMatrix{T}) where {T}
    m = zero(T)
    for a in axes(F, 2)
        s = zero(T)
        for d in axes(F, 1)
            s += F[d, a]^2
        end
        m = max(m, sqrt(s))
    end
    return m
end

"""
    fire_step!(opt, positions, forces) -> nothing

One FIRE iteration: update the velocities of `opt` from `forces` (`3 × n`, eV/Å, minus the
gradient at `positions`) by rules F1 to F4 of the paper, then advance `positions` in place.
Allocation-free.
"""
function fire_step!(opt::FIRE{T}, X::AbstractMatrix{T}, F::AbstractMatrix{T})::Nothing where {T}
    V = opt.velocity
    axes(X) == axes(V) == axes(F) ||
        throw(DimensionMismatch(lazy"pureatomistic: positions $(axes(X)), forces $(axes(F)) and velocities $(axes(V)) must have equal axes"))
    P = zero(T)
    fn2 = zero(T)
    vn2 = zero(T)
    for a in axes(X, 2), d in axes(X, 1)
        P += F[d, a] * V[d, a]
        fn2 += F[d, a]^2
        vn2 += V[d, a]^2
    end
    if P > 0
        # F2: |F| > 0 here because P > 0.
        mix = opt.alpha * sqrt(vn2 / fn2)
        for a in axes(X, 2), d in axes(X, 1)
            V[d, a] = (1 - opt.alpha) * V[d, a] + mix * F[d, a]
        end
        opt.npos += 1
        if opt.npos > opt.nmin
            opt.dt = min(opt.dt * opt.finc, opt.dtmax)
            opt.alpha *= opt.falpha
        end
    else
        fill!(V, zero(T))
        opt.dt *= opt.fdec
        opt.alpha = opt.alpha_start
        opt.npos = 0
    end
    dt = opt.dt
    biggest = zero(T)
    for a in axes(X, 2)
        s = zero(T)
        for d in axes(X, 1)
            V[d, a] += dt * F[d, a]
            s += (dt * V[d, a])^2
        end
        biggest = max(biggest, sqrt(s))
    end
    scale = biggest > opt.maxstep ? opt.maxstep / biggest : one(T)
    for a in axes(X, 2), d in axes(X, 1)
        X[d, a] += scale * dt * V[d, a]
    end
    return nothing
end

"""
    RelaxResult{T}

The outcome of [`relax!`](@ref): `converged` (the largest atomic force fell below `fmax`),
`steps` (optimizer iterations taken), `energy` (eV) and `max_force` (largest Euclidean norm of
an atomic force, eV/Å) of the final configuration.
"""
struct RelaxResult{T <: AbstractFloat}
    converged::Bool
    steps::Int
    energy::T
    max_force::T
end

# The largest atomic force and the energy of the last evaluation of `potential`; throws when either
# is not finite, naming the iteration `step`.
function _status(potential::AbstractPotential, step::Integer)
    fm = _max_force(potential.result.forces)
    E = potential.result.energy
    (isfinite(fm) && isfinite(E)) ||
        throw(ArgumentError(lazy"pureatomistic: relax! met a non-finite energy or force after $step steps"))
    return fm, E
end

function _check_relax_arguments(sys::System, potential::AbstractPotential, fmax::Real, maxsteps::Integer)
    Base.require_one_based_indexing(sys.positions, sys.cell)
    (isfinite(fmax) && fmax > 0) ||
        throw(ArgumentError(lazy"pureatomistic: fmax must be positive and finite, got $fmax"))
    maxsteps >= 0 || throw(ArgumentError(lazy"pureatomistic: maxsteps must not be negative, got $maxsteps"))
    n_atoms(potential) == natoms(sys) ||
        throw(DimensionMismatch(lazy"pureatomistic: the potential has $(n_atoms(potential)) atoms, the system $(natoms(sys))"))
    return nothing
end

"""
    relax!(sys, potential; fmax = 1e-3, maxsteps = 10_000, kwargs...) -> RelaxResult
    relax!(sys, potential, opt::FIRE; fmax = 1e-3, maxsteps = 10_000) -> RelaxResult

Minimize the energy of `potential` over the positions of `sys` at fixed cell with FIRE
([`FIRE`](@ref)), updating `sys.positions` in place, until the largest atomic force is below
`fmax` (eV/Å) or `maxsteps` iterations are done. The keywords `dt`, `dtmax`, `maxstep`, `nmin`,
`finc`, `fdec`, `alpha_start` and `falpha` build the [`FIRE`](@ref) state; passing a state
reuses its buffer, which makes the call allocation-free after the potential's first evaluation.
Positions are not wrapped into the cell. A run that stops at `maxsteps` returns
`converged = false`; a non-finite energy or force throws. The positions must be 1-based, as the
potentials require.
"""
function relax!(
        sys::System{T}, potential::AbstractPotential, opt::FIRE{T}; fmax::Real = 1.0e-3, maxsteps::Integer = 10_000
    ) where {T}
    _check_relax_arguments(sys, potential, fmax, maxsteps)
    size(opt.velocity, 2) == natoms(sys) ||
        throw(DimensionMismatch(lazy"pureatomistic: the FIRE state has $(size(opt.velocity, 2)) atoms, the system $(natoms(sys))"))
    _reset!(opt)
    tol = T(fmax)
    X = sys.positions
    prepare!(potential, X, sys.cell)
    energy_forces!(potential, X, sys.cell)
    for step in 0:(maxsteps - 1)
        fm, E = _status(potential, step)
        fm < tol && return RelaxResult{T}(true, step, E, fm)
        fire_step!(opt, X, potential.result.forces)
        prepare!(potential, X, sys.cell)
        energy_forces!(potential, X, sys.cell)
    end
    fm, E = _status(potential, maxsteps)
    return RelaxResult{T}(fm < tol, Int(maxsteps), E, fm)
end

function relax!(
        sys::System{T}, potential::AbstractPotential; fmax::Real = 1.0e-3, maxsteps::Integer = 10_000,
        dt::Real = 0.1, dtmax::Real = 1.0, maxstep::Real = 0.1, nmin::Integer = 5, finc::Real = 1.1,
        fdec::Real = 0.5, alpha_start::Real = 0.1, falpha::Real = 0.99
    ) where {T}
    opt = FIRE(sys; dt, dtmax, maxstep, nmin, finc, fdec, alpha_start, falpha)
    return relax!(sys, potential, opt; fmax, maxsteps)
end
