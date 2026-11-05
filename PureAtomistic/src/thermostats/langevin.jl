"""
    LangevinBAOAB(T, γ)

Langevin dynamics at temperature `T` (K) with friction `γ` (1/fs; `1/ps = 0.001/fs`), integrated
by the BAOAB splitting of B. Leimkuhler and C. Matthews, Appl. Math. Res. Express 2013, 34
(2013). With the time step `h` of the simulation's [`VelocityVerlet`](@ref) integrator, one step
is

    B: v ← v + (h/2) F(x)/m,
    A: x ← x + (h/2) v,
    O: v ← c₁ v + √((1 - c₁²) k_B T / m) ξ,   c₁ = exp(-γ h),
    A: x ← x + (h/2) v,
    B: v ← v + (h/2) F(x)/m,

with one force evaluation per step (the forces of the new positions serve the closing B of this
step and the opening B of the next). `ξ` holds independent standard normal numbers, drawn from
the simulation's random number generator atom by atom and, within an atom, in the order x, y,
z, so a seeded run reproduces bit for bit. The O step is the exact solution of the
Ornstein-Uhlenbeck part `dv = -γ v dt + √(2γ k_B T/m) dW` over `h`. The thermostat replaces the
integrator's own step. `γ = 0` gives deterministic dynamics; the friction must be finite.
"""
struct LangevinBAOAB <: AbstractThermostat
    temp::Float64
    gamma::Float64
    function LangevinBAOAB(temp::Real, gamma::Real)
        (isfinite(temp) && temp >= 0) ||
            throw(ArgumentError(lazy"pureatomistic: the temperature must be non-negative and finite, got $temp"))
        (isfinite(gamma) && gamma >= 0) ||
            throw(ArgumentError(lazy"pureatomistic: the friction must be non-negative and finite, got $gamma"))
        return new(Float64(temp), Float64(gamma))
    end
end

function thermostat_step!(
        th::LangevinBAOAB, sys::System{T}, potential::AbstractPotential, dt::Real,
        rng::Random.AbstractRNG, integrate!
    )::Nothing where {T}
    Base.require_one_based_indexing(sys.positions, sys.velocities, sys.masses)
    V, X, m = sys.velocities, sys.positions, sys.masses
    h = T(dt)
    c1 = T(exp(-th.gamma * dt))
    # √((1 - c₁²) k_B T) in eV^½; expm1 keeps 1 - c₁² accurate when γ h is small.
    c3 = T(sqrt(-expm1(-2 * th.gamma * dt) * KB_EV_PER_K * th.temp))
    amu = T(AMU_IN_EV_FS2_PER_A2)
    _kick!(V, potential.result.forces, m, h / 2)
    _drift!(X, V, h / 2)
    for a in axes(V, 2)
        s = c3 / sqrt(m[a] * amu)
        for k in axes(V, 1)
            V[k, a] = c1 * V[k, a] + s * randn(rng, T)
        end
    end
    _drift!(X, V, h / 2)
    prepare!(potential, X, sys.cell)
    energy_forces!(potential, X, sys.cell)
    _kick!(V, potential.result.forces, m, h / 2)
    return nothing
end

# v ← v + τ F / m for every atom, with the mass in amu and the force in eV/Å.
function _kick!(V::AbstractMatrix{T}, F::AbstractMatrix, m::AbstractVector{T}, τ::T) where {T}
    axes(F) == axes(V) || throw(DimensionMismatch(lazy"pureatomistic: forces axes $(axes(F)) differ from velocities axes $(axes(V))"))
    c = T(AMU_IN_EV_FS2_PER_A2)
    for a in axes(V, 2)
        f = τ / (m[a] * c)
        for k in axes(V, 1)
            V[k, a] += f * F[k, a]
        end
    end
    return nothing
end

# x ← x + h v.
function _drift!(X::AbstractMatrix{T}, V::AbstractMatrix{T}, h::T) where {T}
    for a in axes(X, 2), k in axes(X, 1)
        X[k, a] += h * V[k, a]
    end
    return nothing
end
