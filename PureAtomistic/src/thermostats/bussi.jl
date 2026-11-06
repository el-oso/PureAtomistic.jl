"""
    BussiThermostat(T, τ; removed_dof = 3)

Stochastic velocity rescaling at temperature `T` (K) with coupling time `τ` (fs) of G. Bussi,
D. Donadio and M. Parrinello, J. Chem. Phys. 126, 014101 (2007). The step is a half-step of the
thermostat, a step of the simulation's integrator, and another half-step of the thermostat.
A half-step of length `t = h/2` multiplies every velocity by `α = √(α²)` with

    α² = c + (1 - c) K̄ (R₁² + S) / (N_f K) + 2 √(c (1 - c) K̄ / (N_f K)) R₁,   c = exp(-t/τ),

where `K` is the kinetic energy, `K̄ = N_f k_B T / 2` its target mean, `R₁` a standard normal
number and `S = Σᵢ₌₂^N_f Rᵢ²` the sum of `N_f - 1` further squared standard normals, drawn as
`2 g` with `g` from [`randgamma`](@ref) at shape `(N_f - 1)/2`. The kinetic energy `α² K` is
then the exact solution over `t` of the paper's stochastic equation for `K`, whose stationary
distribution is the canonical one, Gamma with shape `N_f/2` and scale `k_B T`. The update is
re-derived here from the paper's stochastic equation, not transcribed; the tests verify the
stationary distribution and the deterministic limit.

`N_f = 3n - removed_dof` for `n` atoms; the default removes the three degrees of freedom of the
center-of-mass momentum, which rescaling conserves when it is zero (as after
`maxwell_boltzmann!`). Pass `removed_dof = 0` when the forces do not conserve momentum, as for
springs tied to fixed sites. `τ = Inf` leaves the velocities unchanged, bit for bit. Each
half-step draws `R₁` and then, when `N_f > 1`, `g` from the simulation's random number
generator. A step throws when `N_f < 1` or when the kinetic energy is zero, since there is no
velocity to rescale.
"""
struct BussiThermostat <: AbstractThermostat
    temp::Float64
    tau::Float64
    removed_dof::Int
    function BussiThermostat(temp::Real, tau::Real; removed_dof::Integer = 3)
        (isfinite(temp) && temp >= 0) ||
            throw(ArgumentError(lazy"pureatomistic: the temperature must be non-negative and finite, got $temp"))
        tau > 0 || throw(ArgumentError(lazy"pureatomistic: the coupling time must be positive, got $tau"))
        removed_dof >= 0 ||
            throw(ArgumentError(lazy"pureatomistic: removed_dof must not be negative, got $removed_dof"))
        return new(Float64(temp), Float64(tau), Int(removed_dof))
    end
end

function thermostat_step!(
        th::BussiThermostat, sys::System, potential::AbstractPotential, dt::Real,
        rng::Random.AbstractRNG, integrate!
    )::Nothing
    _rescale!(th, sys, dt / 2, rng)
    integrate!(sys, potential)
    _rescale!(th, sys, dt / 2, rng)
    return nothing
end

# The step moves the positions only through `integrate!`, which carries the hook.
thermostat_step!(
    th::BussiThermostat, sys::System, potential::AbstractPotential, dt::Real,
    rng::Random.AbstractRNG, integrate!, after_drift!
)::Nothing = thermostat_step!(th, sys, potential, dt, rng, integrate!)

function _rescale!(th::BussiThermostat, sys::System{T}, t::Real, rng::Random.AbstractRNG) where {T}
    nf = 3 * natoms(sys) - th.removed_dof
    nf >= 1 || throw(ArgumentError(lazy"pureatomistic: the Bussi thermostat needs at least one degree of freedom, got $nf"))
    K = kinetic_energy(sys)
    K > 0 || throw(ArgumentError("pureatomistic: the Bussi thermostat cannot rescale velocities with zero kinetic energy"))
    c = T(exp(-t / th.tau))
    # expm1 keeps 1 - c accurate when t ≪ τ; for τ = Inf both c = 1 and 1 - c = 0 are exact.
    q = T(-expm1(-t / th.tau)) * T(nf * KB_EV_PER_K * th.temp / 2) / (nf * K)
    r1 = randn(rng, T)
    s = nf > 1 ? 2 * randgamma(rng, T(nf - 1) / 2) : zero(T)
    # α² = (√c + √q R₁)² + q S, the expression of the docstring as a sum of squares, which
    # rounding cannot make negative.
    α = sqrt((sqrt(c) + sqrt(q) * r1)^2 + q * s)
    V = sys.velocities
    for a in axes(V, 2), k in axes(V, 1)
        V[k, a] *= α
    end
    return nothing
end
