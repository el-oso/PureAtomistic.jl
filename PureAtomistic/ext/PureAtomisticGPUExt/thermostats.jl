# The Bussi and Langevin (BAOAB) thermostats on the device. A step draws one key from the
# simulation's generator on the host and passes it to the kernels, which draw their Philox numbers
# under it; the step reads back nothing beyond the status word of its force evaluation.

# Entry 3 of the status word of the device neighbor list (PureMLIP's `DeviceNeighborList`, the
# "sampler cap reached" entry) holds the thermostat's flag. The kernels set it, the list's builds
# and evaluations never clear it, and the status read of the step's evaluation, or of the next
# step's for a flag set after it, throws.
const _STATUS_THERMOSTAT = 3
const _SAMPLER_CAP = Int32(1)
const _ZERO_KINETIC_ENERGY = Int32(2)

function _check_thermostat_flag(status::AbstractVector{Int32})
    flag = status[_STATUS_THERMOSTAT]
    flag == _SAMPLER_CAP &&
        throw(ArgumentError("pureatomistic: the gamma sampler rejected 10000 proposals in a row; the random number generator does not produce normal and uniform numbers"))
    flag == _ZERO_KINETIC_ENERGY &&
        throw(ArgumentError("pureatomistic: the Bussi thermostat cannot rescale velocities with zero kinetic energy"))
    iszero(flag) || throw(ArgumentError(lazy"pureatomistic: the thermostat flag holds the unknown code $flag"))
    return nothing
end

"""
    thermostat_step!(th::BussiThermostat, sys, p::DeviceMACEPotential, dt, rng, integrate!)

The Bussi step of the CPU method on the device. One key from `rng` per step; a single work item
draws `R₁` and the gamma variate `S/2` of both half-steps under it (counters `(0, half)` and
`(pass, half)`), so the sampler's cap is in the status word before the step's force evaluation
reads it. Each half-step sums the kinetic energy on the device, computes `α` in a single work
item, and scales the velocities by it; nothing is read back. A zero kinetic energy leaves the
velocities unchanged and sets the thermostat's flag. The CPU method throws before it changes
anything; the flag reaches the host only at the next status read, which is the one inside the
force evaluation of the step (a flag set by the second half-step: the next step's). The step has
then already made its first half kick and its drift, so the exception leaves the positions and
velocities advanced by that much and `sim.step_count` unchanged. The flag stays set, so every
later status read of that potential throws too.
"""
function Thermostats.thermostat_step!(
        th::BussiThermostat, sys::System{T}, p::DeviceMACEPotential{T}, dt::Real,
        rng::Random.AbstractRNG, integrate!
    )::Nothing where {T <: AbstractFloat}
    _check_motion(sys, p)
    nf = 3 * natoms(sys) - th.removed_dof
    nf >= 1 || throw(ArgumentError(lazy"pureatomistic: the Bussi thermostat needs at least one degree of freedom, got $nf"))
    src = _PhiloxStream(rand(rng, UInt64))
    _launch!(_bussi_draw_kernel!, p.nl.backend, 1, p.draws, p.nl.status, src, T(nf - 1) / 2, nf > 1)
    t = dt / 2
    c = T(exp(-t / th.tau))
    # The numerator of q = (1 - c) K̄ / (N_f K) in the CPU's rounding; for τ = Inf it is 0 and c = 1.
    qnum = T(-expm1(-t / th.tau)) * T(nf * KB_EV_PER_K * th.temp / 2)
    _bussi_half_step!(sys, p, 1, c, qnum, T(nf))
    integrate!(sys, p)
    _bussi_half_step!(sys, p, 2, c, qnum, T(nf))
    return nothing
end

function _bussi_half_step!(sys::System{T}, p::DeviceMACEPotential{T}, half::Int, c::T, qnum::T, nf::T) where {T}
    backend = p.nl.backend
    _kinetic_sum!(sys, p)
    _launch!(_bussi_alpha_kernel!, backend, 1, p.draws, p.nl.status, p.kinetic_sum, half, c, qnum, nf, T(AMU_IN_EV_FS2_PER_A2 / 2))
    _launch!(_scale_velocities_kernel!, backend, size(sys.velocities), sys.velocities, p.draws)
    return nothing
end

@kernel function _bussi_draw_kernel!(draws, status, src, shape, with_gamma)
    _bussi_draws!(draws, status, src, shape, with_gamma)
end

# `draws[2h - 1] = R₁` and `draws[2h] = S = 2 g` of half-step `h` (S = 0 when N_f = 1); a gamma
# variate that reaches the cap sets the flag and gives S = 0.
@inline function _bussi_draws!(draws, status, src, shape::T, with_gamma::Bool) where {T}
    for half in 1:2
        draws[2 * half - 1] = _normal(src, T, 0, half, _STREAM_BUSSI_NORMAL)
        s = zero(T)
        if with_gamma
            g, accepted = _device_gamma(src, shape, half)
            accepted || (status[_STATUS_THERMOSTAT] = _SAMPLER_CAP)
            s = 2 * g
        end
        draws[2 * half] = s
    end
    return nothing
end

@kernel function _bussi_alpha_kernel!(draws, status, @Const(ksum), half, c, qnum, nf, half_amu)
    draws[5] = _bussi_alpha(status, half_amu * ksum[1], draws[2 * half - 1], draws[2 * half], c, qnum, nf)
end

# α = √((√c + √q R₁)² + q S) with q = qnum / (N_f K), the sum of squares of the CPU method, which
# rounding cannot make negative.
@inline function _bussi_alpha(status, K::T, r1::T, s::T, c::T, qnum::T, nf::T) where {T}
    if K > zero(T)
        q = qnum / (nf * K)
        return sqrt((sqrt(c) + sqrt(q) * r1)^2 + q * s)
    end
    status[_STATUS_THERMOSTAT] = _ZERO_KINETIC_ENERGY
    return one(T)
end

@kernel function _scale_velocities_kernel!(V, @Const(draws))
    k, a = @index(Global, NTuple)
    V[k, a] *= draws[5]
end

"""
    thermostat_step!(th::LangevinBAOAB, sys, p::DeviceMACEPotential, dt, rng, integrate!, after_drift! = no_hook)

The BAOAB step of the CPU method on the device, with the kick, drift and O kernels. One key from
`rng` per step; the noise of component `k` of atom `a` is the Philox normal at counter `(a, k)` of
the Langevin stream, so it does not depend on the order of the work items.
"""
function Thermostats.thermostat_step!(
        th::LangevinBAOAB, sys::System{T}, p::DeviceMACEPotential{T}, dt::Real,
        rng::Random.AbstractRNG, integrate!, after_drift!::H = no_hook
    )::Nothing where {T <: AbstractFloat, H}
    # `H` makes the method specialize on the hook, which it passes on without calling.
    _check_motion(sys, p)
    c1 = T(exp(-th.gamma * dt))
    # √((1 - c₁²) k_B T) in eV^½, as on the CPU.
    c3 = T(sqrt(-expm1(-2 * th.gamma * dt) * KB_EV_PER_K * th.temp))
    src = _PhiloxStream(rand(rng, UInt64))
    _baoab!(p.nl.backend, sys, p.result.forces, T(dt), c1, c3, src, after_drift!, _Evaluate(p))
    return nothing
end

# The force evaluation of a step: `prepare!` and `energy_forces!` of the potential.
struct _Evaluate{P}
    p::P
end
function (e::_Evaluate)(sys::System)
    Potentials.prepare!(e.p, sys.positions, sys.cell)
    Potentials.energy_forces!(e.p, sys.positions, sys.cell)
    return nothing
end

# The BAOAB splitting on the arrays of `sys` and the forces `F` on `backend`: B, A, O, A, the
# hook, `evaluate!(sys)` (which leaves the forces of the new positions in `F`), B.
function _baoab!(backend::Backend, sys::System{T}, F, h::T, c1::T, c3::T, src, after_drift!, evaluate!) where {T}
    V, X, m = sys.velocities, sys.positions, sys.masses
    _kick!(backend, V, F, m, h / 2)
    _drift!(backend, X, V, h / 2)
    _launch!(_o_step_kernel!, backend, size(V), V, m, src, c1, c3, T(AMU_IN_EV_FS2_PER_A2))
    _drift!(backend, X, V, h / 2)
    after_drift!(sys)
    evaluate!(sys)
    _kick!(backend, V, F, m, h / 2)
    return nothing
end

@kernel function _o_step_kernel!(V, @Const(m), src, c1, c3, amu)
    k, a = @index(Global, NTuple)
    V[k, a] = _o_update(V[k, a], m[a], c1, c3, amu, _normal(src, eltype(V), a, k, _STREAM_LANGEVIN))
end

# The O step of one component: c₁ v + √((1 - c₁²) k_B T / m) ξ, in the CPU's rounding.
@inline _o_update(v, m, c1, c3, amu, ξ) = c1 * v + c3 / sqrt(m * amu) * ξ
