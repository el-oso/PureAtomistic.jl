# Radial embedding of MACE: Agnesi distance transform, Bessel basis on the transformed
# distance, polynomial cutoff on the raw distance, and the ZBL pair repulsion. Every
# function returns its value together with the derivative with respect to the distance.
# Formulas follow mace (MIT), `mace/modules/radial.py`.

# f = 1 - c1 t^p + c2 t^(p+1) - c3 t^(p+2) for t = d / scale, with the coefficients that
# make the first two derivatives vanish at t = 1; and df/dd.
@inline function _poly_envelope(t, p, inv_scale)
    c1 = oftype(t, (p + 1) * (p + 2)) / 2
    c2 = oftype(t, p * (p + 2))
    c3 = oftype(t, p * (p + 1)) / 2
    f = 1 - c1 * t^p + c2 * t^(p + 1) - c3 * t^(p + 2)
    df = (-c1 * p * t^(p - 1) + c2 * (p + 1) * t^p - c3 * (p + 2) * t^(p + 1)) * inv_scale
    return f, df
end

function polynomial_cutoff(d::S, r_max, p::Int) where {S <: Real}
    d < r_max || return zero(S), zero(S)
    f, df = _poly_envelope(d / r_max, p, inv(r_max))
    return convert(S, f), convert(S, df)
end

function agnesi_transform(d::S, r0, a, p, q) where {S <: Real}
    x = d / r0
    xq = x^q
    xqp = x^(q - p)
    den = 1 + xqp
    dt = inv(1 + a * xq / den)
    # u = a x^q / (1 + x^(q-p)); du/dx by the quotient rule, dt = 1 / (1 + u).
    du = a * (q * xq * den - xq * (q - p) * xqp) / (x * den^2)
    ddt = -dt^2 * du / r0
    return dt, ddt
end

_check_bessel(B, dB, weights) = axes(B) == axes(dB) == axes(weights) ||
    throw(DimensionMismatch(lazy"model: B, dB and weights must share axes: $(axes(B)), $(axes(dB)), $(axes(weights))"))

function bessel_basis!(B::AbstractVector{S}, dB::AbstractVector{S}, x::S, weights::AbstractVector, prefactor) where {S <: Real}
    _check_bessel(B, dB, weights)
    for n in eachindex(B)
        s, c = sincos(weights[n] * x)
        B[n] = prefactor * s / x
        dB[n] = prefactor * (weights[n] * c / x - s / x^2)
    end
    return B
end

# `bessel_basis!` for weights that are the multiples k·w₁ (k = 1, 2, ...) of the first weight.
# Then w_k x = k (w₁ x), so sin(w_k x) and cos(w_k x) follow exactly from one `sincos(w₁ x)` by
# angle addition, s_{k+1} = s_k c₁ + c_k s₁ and c_{k+1} = c_k c₁ - s_k s₁: the result is the
# function `bessel_basis!` computes, up to the rounding of the recurrence, which grows with k.
function bessel_basis_multiples!(B::AbstractVector{S}, dB::AbstractVector{S}, x::S, weights::AbstractVector, prefactor) where {S <: Real}
    _check_bessel(B, dB, weights)
    isempty(B) && return B
    s1, c1 = sincos(weights[firstindex(weights)] * x)
    s, c = s1, c1
    for n in eachindex(B)
        B[n] = prefactor * s / x
        dB[n] = prefactor * (weights[n] * c / x - s / x^2)
        s, c = s * c1 + c * s1, c * c1 - s * s1
    end
    return B
end

struct RadialEmbedding{T}
    bessel_weights::Vector{T}
    prefactor::T
    r_max::T
    cutoff_p::Int
    agnesi_a::T
    agnesi_p::T
    agnesi_q::T
    covalent_radii::Vector{T}   # indexed by atomic number + 1
    # The k-th Bessel weight equals k times the first, compared exactly in the element type the
    # constructor was given (before conversion to T); `radial_embedding!` then uses the recurrence.
    multiples::Bool
    function RadialEmbedding{T}(bessel_weights, prefactor, r_max, cutoff_p, agnesi_a, agnesi_p, agnesi_q, covalent_radii) where {T}
        return new{T}(
            bessel_weights, prefactor, r_max, cutoff_p, agnesi_a, agnesi_p, agnesi_q, covalent_radii,
            _integer_multiples(bessel_weights),
        )
    end
end

# Whether w[k] == k * w[1] exactly for every position k.
function _integer_multiples(w::AbstractVector)
    isempty(w) && return false
    w1 = w[firstindex(w)]
    for (k, i) in enumerate(eachindex(w))
        w[i] == k * w1 || return false
    end
    return true
end

function radial_embedding!(feats::AbstractVector{S}, dfeats::AbstractVector{S}, emb::RadialEmbedding, d::S, Zu::Int, Zv::Int) where {S <: Real}
    if !(d < emb.r_max)
        fill!(feats, zero(S))
        fill!(dfeats, zero(S))
        return feats
    end
    fc, dfc = polynomial_cutoff(d, emb.r_max, emb.cutoff_p)
    r0 = (emb.covalent_radii[Zu + 1] + emb.covalent_radii[Zv + 1]) / 2
    dt, ddt = agnesi_transform(d, r0, emb.agnesi_a, emb.agnesi_p, emb.agnesi_q)
    if emb.multiples
        bessel_basis_multiples!(feats, dfeats, dt, emb.bessel_weights, emb.prefactor)
    else
        bessel_basis!(feats, dfeats, dt, emb.bessel_weights, emb.prefactor)
    end
    for n in eachindex(feats)
        B = feats[n]
        dfeats[n] = dfeats[n] * ddt * fc + B * dfc
        feats[n] = B * fc
    end
    return feats
end

struct ZBLParams{T}
    c::NTuple{4, T}
    p::T
    a_exp::T
    a_prefactor::T
    covalent_radii::Vector{T}   # indexed by atomic number + 1
end

# Screened Coulomb repulsion of an edge, cut off smoothly at the sum of the covalent radii;
# half of it is credited to the receiver (each pair appears as two directed edges).
function zbl_edge_energy(d::S, Zu::Int, Zv::Int, zbl::ZBLParams) where {S <: Real}
    R = zbl.covalent_radii[Zu + 1] + zbl.covalent_radii[Zv + 1]
    U = promote_type(S, typeof(R))
    d < R || return zero(U), zero(U)
    # Ziegler-Biersack-Littmark universal screening function (mace radial.py:203-216, MIT).
    # The literals are converted to the working type `U`, so a Float32 model computes in Float32.
    ks = (U(3.2), U(0.9423), U(0.4028), U(0.2016))
    a = U(zbl.a_prefactor) * U(0.529) / (U(Zu)^U(zbl.a_exp) + U(Zv)^U(zbl.a_exp))
    ρ = d / a
    φ = zero(U)
    dφ = zero(U)
    for i in 1:4
        term = zbl.c[i] * exp(-ks[i] * ρ)
        φ += term
        dφ -= ks[i] * term / a
    end
    K = U(14.3996) * U(Zu) * U(Zv)
    v = K / d * φ
    dv = K * (dφ / d - φ / d^2)
    f, df = _poly_envelope(d / R, zbl.p, inv(R))
    h = U(0.5)
    return h * v * f, h * (dv * f + v * df)
end
