# Geometry optimization by FIRE 2.0: Guénolé, Nöhring, Vaid, Houllé, Xie, Prakash and Bitzek,
# Comput. Mater. Sci. 175, 109584 (2020), their Algorithm 2 with the semi-implicit Euler integrator
# of their Algorithm 4. Against FIRE it changes four things:
#   - the mixing v ← (1 - α) v + α |v| F/|F| follows the velocity update v += dt F instead of
#     preceding it, so the position update x += dt v uses the mixed velocity; |v| is the norm of
#     the updated velocity, where the printed Algorithm 4 writes v(t);
#   - on P ≤ 0 the positions move back by ½ dt v (dt already reduced) before the velocity is
#     zeroed, which undoes part of the uphill motion that P ≤ 0 detects too late;
#   - dt is not reduced below dtmin, and during the first ndelay iterations (`initialdelay`) a
#     step with P ≤ 0 changes neither dt nor α;
#   - more than nneg_max consecutive iterations with P ≤ 0 stop the minimization.
# Masses are one and dt is a unitless scale, as for FIRE. Every displacement, the uphill correction
# included, is capped at `maxstep` for the atom that moves most. The cap is not part of the paper's
# algorithm; it keeps the first steps of a far-from-equilibrium configuration short.

"""
    FIRE2{T}(natoms; dt = 0.2, dtmax = 2.0, dtmin = 0.004, maxstep = 0.1, ndelay = 20,
             nneg_max = 2000, initialdelay = true, finc = 1.1, fdec = 0.5, alpha_start = 0.25,
             falpha = 0.99)
    FIRE2(sys::System; kwargs...)

The state and parameters of the FIRE 2.0 minimizer (Guénolé et al., Comput. Mater. Sci. 175,
109584, 2020) for `natoms` atoms: the velocities, the current time step `dt`, the mixing
parameter `alpha`, the counts `npos` and `nneg` of consecutive steps with positive and with
non-positive power `F · v`, and the iteration count `iter`.

The defaults other than `dt` and `maxstep` are the paper's LAMMPS defaults, with `dtmax` and
`dtmin` ten times and 0.02 times the starting `dt`: `ndelay` steps of positive power before `dt`
grows by `finc` and `alpha` shrinks by `falpha`, `fdec` on non-positive power,
`alpha_start = 0.25`, and a stop after more than `nneg_max` consecutive steps of non-positive
power. With `initialdelay`, a step of non-positive power within the first `ndelay` iterations
leaves `dt` and `alpha` unchanged. All atoms have unit mass and the forces are in eV/Å, so `dt`
is a unitless scale; `dt = 0.2` took the fewest force evaluations of the starting steps measured
on two 256-atom Cu cells with MACE-MPA-0 (`bench/relax_tune.jl`,
`bench/results/relax_tune_mac.json`). `maxstep` (Å) caps the largest displacement of any atom in
one move; the cap also applies to the uphill correction and is not part of the paper's algorithm.
In the mixing, `|v|` is the norm of the updated velocity, a choice of this package: the printed
Algorithm 4 writes `v(t)`.
"""
mutable struct FIRE2{T <: AbstractFloat}
    const velocity::Matrix{T}
    const dt0::T
    const dtmax::T
    const dtmin::T
    const maxstep::T
    const finc::T
    const fdec::T
    const alpha_start::T
    const falpha::T
    const ndelay::Int
    const nneg_max::Int
    const initialdelay::Bool
    dt::T
    alpha::T
    npos::Int
    nneg::Int
    iter::Int
end

function FIRE2{T}(
        natoms::Integer; dt::Real = 0.2, dtmax::Real = 2.0, dtmin::Real = 0.004, maxstep::Real = 0.1,
        ndelay::Integer = 20, nneg_max::Integer = 2000, initialdelay::Bool = true, finc::Real = 1.1,
        fdec::Real = 0.5, alpha_start::Real = 0.25, falpha::Real = 0.99
    ) where {T <: AbstractFloat}
    natoms >= 0 || throw(ArgumentError(lazy"pureatomistic: the number of atoms must not be negative, got $natoms"))
    (isfinite(dt) && dt > 0) || throw(ArgumentError(lazy"pureatomistic: FIRE2 dt must be positive and finite, got $dt"))
    (isfinite(dtmax) && dtmax >= dt) ||
        throw(ArgumentError(lazy"pureatomistic: FIRE2 dtmax must be finite and at least dt = $dt, got $dtmax"))
    (0 <= dtmin <= dt) || throw(ArgumentError(lazy"pureatomistic: FIRE2 dtmin must lie in [0, dt = $dt], got $dtmin"))
    (isfinite(maxstep) && maxstep > 0) ||
        throw(ArgumentError(lazy"pureatomistic: FIRE2 maxstep must be positive and finite, got $maxstep"))
    ndelay >= 0 || throw(ArgumentError(lazy"pureatomistic: FIRE2 ndelay must not be negative, got $ndelay"))
    nneg_max >= 0 || throw(ArgumentError(lazy"pureatomistic: FIRE2 nneg_max must not be negative, got $nneg_max"))
    (isfinite(finc) && finc > 1) || throw(ArgumentError(lazy"pureatomistic: FIRE2 finc must exceed 1, got $finc"))
    (0 < fdec < 1) || throw(ArgumentError(lazy"pureatomistic: FIRE2 fdec must lie in (0, 1), got $fdec"))
    (0 < alpha_start <= 1) ||
        throw(ArgumentError(lazy"pureatomistic: FIRE2 alpha_start must lie in (0, 1], got $alpha_start"))
    (0 < falpha <= 1) || throw(ArgumentError(lazy"pureatomistic: FIRE2 falpha must lie in (0, 1], got $falpha"))
    return FIRE2{T}(
        zeros(T, 3, Int(natoms)), T(dt), T(dtmax), T(dtmin), T(maxstep), T(finc), T(fdec), T(alpha_start),
        T(falpha), Int(ndelay), Int(nneg_max), initialdelay, T(dt), T(alpha_start), 0, 0, 0
    )
end

FIRE2(sys::System{T}; kwargs...) where {T} = FIRE2{T}(natoms(sys); kwargs...)

function _reset!(opt::FIRE2{T}) where {T}
    fill!(opt.velocity, zero(T))
    opt.dt = opt.dt0
    opt.alpha = opt.alpha_start
    opt.npos = 0
    opt.nneg = 0
    opt.iter = 0
    return opt
end

# X += c V, scaled down so that no atom moves farther than `maxstep`.
function _capped_move!(X::AbstractMatrix{T}, V::AbstractMatrix{T}, c::T, maxstep::T) where {T}
    biggest = zero(T)
    for a in axes(X, 2)
        s = zero(T)
        for d in axes(X, 1)
            s += (c * V[d, a])^2
        end
        biggest = max(biggest, sqrt(s))
    end
    scale = biggest > maxstep ? maxstep / biggest : one(T)
    for a in axes(X, 2), d in axes(X, 1)
        X[d, a] += scale * c * V[d, a]
    end
    return nothing
end

"""
    fire2_step!(opt, positions, forces) -> Bool

One FIRE 2.0 iteration: update the step size, mixing and velocities of `opt` from `forces`
(`3 × n`, eV/Å, minus the gradient at `positions`) and advance `positions` in place. Returns
`false`, without moving, when this is the iteration that exceeds `nneg_max` consecutive steps of
non-positive power, and `true` otherwise. Allocation-free.
"""
function fire2_step!(opt::FIRE2{T}, X::AbstractMatrix{T}, F::AbstractMatrix{T})::Bool where {T}
    V = opt.velocity
    axes(X) == axes(V) == axes(F) ||
        throw(DimensionMismatch(lazy"pureatomistic: positions $(axes(X)), forces $(axes(F)) and velocities $(axes(V)) must have equal axes"))
    opt.iter += 1
    P = zero(T)
    fn2 = zero(T)
    for a in axes(X, 2), d in axes(X, 1)
        P += F[d, a] * V[d, a]
        fn2 += F[d, a]^2
    end
    if P > 0
        opt.npos += 1
        opt.nneg = 0
        if opt.npos > opt.ndelay
            opt.dt = min(opt.dt * opt.finc, opt.dtmax)
            opt.alpha *= opt.falpha
        end
    else
        opt.npos = 0
        opt.nneg += 1
        opt.nneg > opt.nneg_max && return false
        if !(opt.initialdelay && opt.iter < opt.ndelay)
            if opt.dt * opt.fdec >= opt.dtmin
                opt.dt *= opt.fdec
            end
            opt.alpha = opt.alpha_start
        end
        _capped_move!(X, V, -opt.dt / 2, opt.maxstep)
        fill!(V, zero(T))
    end
    dt = opt.dt
    vn2 = zero(T)
    for a in axes(X, 2), d in axes(X, 1)
        V[d, a] += dt * F[d, a]
        vn2 += V[d, a]^2
    end
    # |F| = 0 leaves nothing to mix toward.
    if fn2 > 0
        mix = opt.alpha * sqrt(vn2 / fn2)
        for a in axes(X, 2), d in axes(X, 1)
            V[d, a] = (1 - opt.alpha) * V[d, a] + mix * F[d, a]
        end
    end
    _capped_move!(X, V, dt, opt.maxstep)
    return true
end

"""
    relax!(sys, potential, opt::FIRE2; fmax = 1e-3, maxsteps = 10_000) -> RelaxResult

Minimize the energy of `potential` over the positions of `sys` at fixed cell with FIRE 2.0
([`FIRE2`](@ref)), updating `sys.positions` in place, until the largest atomic force is below
`fmax` (eV/Å), `maxsteps` iterations are done, or more than `nneg_max` consecutive iterations
have non-positive power; the last two return `converged = false`. The state is reset at the
start and the call is allocation-free after the potential's first evaluation. Positions are not
wrapped; a non-finite energy or force throws.
"""
function relax!(
        sys::System{T}, potential::AbstractPotential, opt::FIRE2{T}; fmax::Real = 1.0e-3, maxsteps::Integer = 10_000
    ) where {T}
    _check_relax_arguments(sys, potential, fmax, maxsteps)
    size(opt.velocity, 2) == natoms(sys) ||
        throw(DimensionMismatch(lazy"pureatomistic: the FIRE2 state has $(size(opt.velocity, 2)) atoms, the system $(natoms(sys))"))
    _reset!(opt)
    tol = T(fmax)
    X = sys.positions
    prepare!(potential, X, sys.cell)
    energy_forces!(potential, X, sys.cell)
    for step in 0:(maxsteps - 1)
        fm, E = _status(potential, step)
        fm < tol && return RelaxResult{T}(true, step, E, fm)
        fire2_step!(opt, X, potential.result.forces) || return RelaxResult{T}(false, step, E, fm)
        prepare!(potential, X, sys.cell)
        energy_forces!(potential, X, sys.cell)
    end
    fm, E = _status(potential, maxsteps)
    return RelaxResult{T}(fm < tol, Int(maxsteps), E, fm)
end
