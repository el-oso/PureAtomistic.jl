# The radial stages on the device: the radial features of each edge (polynomial cutoff, Agnesi
# transform, Bessel basis), the radial MLP and its pullback, and the ZBL pair repulsion. They
# compute what `radial_embedding!`, `radial_mlp!`, `radial_mlp_pullback!` and `zbl_edge_energy` of
# PureMLIP.Model compute, in the model's element type. Integer powers use
# `Base.power_by_squaring` in that type, because Base's power of a Float32 by an integer computes in
# Float64.

# f = 1 - c1 t^p + c2 t^(p+1) - c3 t^(p+2) for t = d / scale, with the coefficients that make the
# first two derivatives vanish at t = 1; and df/dd.
@inline function _poly_envelope_device(t, p::Int, inv_scale)
    c1 = oftype(t, (p + 1) * (p + 2)) / 2
    c2 = oftype(t, p * (p + 2))
    c3 = oftype(t, p * (p + 1)) / 2
    pow(n) = Base.power_by_squaring(t, n)
    f = 1 - c1 * pow(p) + c2 * pow(p + 1) - c3 * pow(p + 2)
    df = (-c1 * p * pow(p - 1) + c2 * (p + 1) * pow(p) - c3 * (p + 2) * pow(p + 1)) * inv_scale
    return f, df
end

# The scalars of the radial embedding, passed to the features kernel by value.
function _radial_params(emb)
    emb.cutoff_p >= 1 || throw(ArgumentError(lazy"gpu: the cutoff exponent must be at least 1, got $(emb.cutoff_p)"))
    return (;
        r_max = emb.r_max, p = emb.cutoff_p, a = emb.agnesi_a, ap = emb.agnesi_p, q = emb.agnesi_q,
        prefactor = emb.prefactor, multiples = emb.multiples,
    )
end

@kernel function _radial_features_kernel!(
        feats, dfeats, @Const(d), @Const(senders), @Const(receivers), @Const(n_edges), @Const(Z), @Const(radii), @Const(weights), P, e0
    )
    j = @index(Global)
    e = e0 + j
    if e <= n_edges[1]
        _radial_column!(feats, dfeats, j, d[e], radii[Z[senders[e]] + 1], radii[Z[receivers[e]] + 1], weights, P)
    else
        _zero_features!(feats, dfeats, j)
    end
end

@inline function _zero_features!(feats, dfeats, col)
    for n in axes(feats, 1)
        feats[n, col] = zero(eltype(feats))
        dfeats[n, col] = zero(eltype(dfeats))
    end
    return nothing
end

# The radial features of an edge of length `dd` between atoms of covalent radii `ru` and `rv` into
# column `col` of `feats`, and their length derivatives into `dfeats`: the Bessel basis of the
# Agnesi-transformed length times the polynomial cutoff of the raw length.
@inline function _radial_column!(feats, dfeats, col, dd, ru, rv, weights, P)
    T = eltype(feats)
    nb = size(feats, 1)
    !(dd < P.r_max) && return _zero_features!(feats, dfeats, col)
    fc, dfc = _poly_envelope_device(dd / P.r_max, P.p, inv(P.r_max))
    # Agnesi transform: x = d / r0, u = a x^q / (1 + x^(q-p)), t = 1 / (1 + u), and dt/dd.
    r0 = (ru + rv) / 2
    x = dd / r0
    xq = x^P.q
    xqp = x^(P.q - P.ap)
    den = 1 + xqp
    t = inv(1 + P.a * xq / den)
    du = P.a * (P.q * xq * den - xq * (P.q - P.ap) * xqp) / (x * den^2)
    dt = -t^2 * du / r0
    # Bessel basis prefactor·sin(w_n t)/t. With weights w_n = n·w₁, sin and cos of w_n t follow
    # from sincos(w₁ t) by angle addition, and dB uses the frequency n·w₁.
    w1 = weights[1]
    s1, c1 = sincos(w1 * t)
    s, c = s1, c1
    for n in 1:nb
        if !P.multiples
            s, c = sincos(weights[n] * t)
        end
        w = P.multiples ? n * w1 : weights[n]
        B = P.prefactor * s / t
        dB = P.prefactor * (w * c / t - s / t^2)
        dfeats[n, col] = dB * dt * fc + B * dfc
        feats[n, col] = B * fc
        if P.multiples
            s, c = s * c1 + c * s1, c * c1 - s * s1
        end
    end
    return nothing
end

"""
    _radial_mlp!(ws, model, k, chunk) -> nothing

The radial MLP of interaction `k` on the features of the edges of `chunk`: `ws.tpw[1:W, j]` for
column `j`, with `ws.pre[:, j, l]` the logistic σ(z) of the pre-activation `z` of hidden layer
`l` and `ws.act[:, j, l]` its activation `cst·z·σ(z)`, which the pullback reads. Each layer is a
matrix product (`mul!`) followed by one elementwise kernel.
"""
function _radial_mlp!(ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, k::Integer, chunk::EdgeChunk) where {T}
    1 <= k <= 2 || throw(ArgumentError(lazy"gpu: the interaction must be 1 or 2, got $k"))
    nb = _chunk_length(ws, chunk)
    W = model.mlp[k]
    cst = model.host.mlp[k].cst
    H, Wout = size(W[1], 2), size(W[4], 2)
    pre(l) = view(ws.pre, 1:H, 1:nb, l)
    act(l) = view(ws.act, 1:H, 1:nb, l)
    silu!(l) = _launch!(_silu_kernel!, ws.backend, (H, nb), ws.act, ws.pre, l, cst)
    mul!(pre(1), transpose(W[1]), view(ws.feats, :, 1:nb))
    silu!(1)
    mul!(pre(2), transpose(W[2]), act(1))
    silu!(2)
    mul!(pre(3), transpose(W[3]), act(2))
    silu!(3)
    mul!(view(ws.tpw, 1:Wout, 1:nb), transpose(W[4]), act(3))
    return nothing
end

# act = cst·z·σ(z) of the pre-activation z = pre[q, j, l], which `pre` then holds σ(z) in place of.
@kernel function _silu_kernel!(act, pre, l, cst)
    q, j = @index(Global, NTuple)
    _silu!(act, pre, q, j, l, cst)
end

@inline function _silu!(act, pre, q, j, l, cst)
    z = pre[q, j, l]
    σ = inv(one(z) + exp(-z))
    act[q, j, l] = z * σ * cst
    pre[q, j, l] = σ
    return nothing
end

"""
    _radial_mlp_pullback!(ws, model, k, chunk) -> nothing

The vector-Jacobian product of [`_radial_mlp!`](@ref) for interaction `k` with the cotangent
`ws.dtpw[1:W, j]` of the outputs, from the `ws.pre` and `ws.act` of that call: the cotangents of
the features into `ws.dfeats_chunk[:, j]`. `ws.mlp_scratch` holds the hidden cotangents.
"""
function _radial_mlp_pullback!(ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, k::Integer, chunk::EdgeChunk) where {T}
    1 <= k <= 2 || throw(ArgumentError(lazy"gpu: the interaction must be 1 or 2, got $k"))
    nb = _chunk_length(ws, chunk)
    W = model.mlp[k]
    cst = model.host.mlp[k].cst
    H, Wout = size(W[1], 2), size(W[4], 2)
    g(s) = view(ws.mlp_scratch, 1:H, 1:nb, s)
    mul!(g(1), W[4], view(ws.dtpw, 1:Wout, 1:nb))
    _launch!(_slope_kernel!, ws.backend, (H, nb), ws.mlp_scratch, 1, ws.pre, ws.act, 3, cst)
    mul!(g(2), W[3], g(1))
    _launch!(_slope_kernel!, ws.backend, (H, nb), ws.mlp_scratch, 2, ws.pre, ws.act, 2, cst)
    mul!(g(1), W[2], g(2))
    _launch!(_slope_kernel!, ws.backend, (H, nb), ws.mlp_scratch, 1, ws.pre, ws.act, 1, cst)
    mul!(view(ws.dfeats_chunk, :, 1:nb), W[1], g(1))
    return nothing
end

# g[:, :, s] .*= ∂act/∂z = cst·σ + act·(1 - σ) of hidden layer l, which turns an activation
# cotangent into a pre-activation cotangent.
@kernel function _slope_kernel!(g, s, @Const(pre), @Const(act), l, cst)
    q, j = @index(Global, NTuple)
    _slope!(g, s, pre, act, q, j, l, cst)
end

@inline function _slope!(g, s, pre, act, q, j, l, cst)
    σ = pre[q, j, l]
    g[q, j, s] *= cst * σ + act[q, j, l] * (one(σ) - σ)
    return nothing
end

# The scalars of the ZBL repulsion, passed to its kernel by value. The envelope exponent is an
# integer, so that its powers are taken by squaring.
function _zbl_params(zbl, r_max::T) where {T}
    isinteger(zbl.p) || throw(ArgumentError(lazy"gpu: the ZBL envelope exponent must be an integer, got $(zbl.p)"))
    p = Int(zbl.p)
    p >= 1 || throw(ArgumentError(lazy"gpu: the ZBL envelope exponent must be at least 1, got $p"))
    return (; c = zbl.c, p, a_exp = T(zbl.a_exp), a_prefactor = T(zbl.a_prefactor), r_max)
end

"""
    _zbl_edges!(ws, model, chunk) -> nothing

The ZBL repulsion of the edges of `chunk` as in `zbl_edge_energy`: the energy credited to the
receiver in `ws.zbl[j]` and its length derivative in `ws.dzbl[j]` for column `j`, and their sum
over the chunk's edges of each receiver `i` added to `ws.pair[i]` in edge order. Over the chunks in
order, starting from `ws.pair` zero, `ws.pair[i]` is the sum over every edge of `i` in edge order.
Edges with `d ≥ r_max` contribute zero.
"""
function _zbl_edges!(ws::DeviceMACEWorkspace{T}, model::DeviceMACEModel{T}, chunk::EdgeChunk) where {T}
    nb = _chunk_length(ws, chunk)
    nl = chunk.nl
    P = _zbl_params(model.host.zbl, model.host.r_max)
    R = _chunk_receivers(nl, nb)
    _launch!(
        _zbl_kernel!, ws.backend, R, ws.zbl, ws.dzbl, ws.pair, ws.d, nl.senders, nl.receivers, nl.row_ptr, nl.n_edges,
        chunk.atomic_numbers, model.zbl_radii, P, first(chunk.edges) - 1, nb, R,
    )
    return nothing
end

# Work item `il` handles receivers `lo + il - 1`, `lo + il - 1 + R`, … of the chunk's
# receiver range `lo:hi` and sums each one's edges of the chunk in order.
@kernel function _zbl_kernel!(
        zbl, dzbl, pair, @Const(d), @Const(senders), @Const(receivers), @Const(row_ptr), @Const(n_edges), @Const(Z), @Const(radii),
        P, e0, nb, R
    )
    il = @index(Global)
    lo, hi = _receiver_range(receivers, n_edges[1], e0, nb)
    for i in (lo + il - 1):R:hi
        _zbl_receiver!(zbl, dzbl, pair, d, senders, row_ptr, Z, radii, P, i, e0, nb)
    end
end

# The ZBL terms of the edges of receiver `i` among slots `e0 + 1 : e0 + nb`, in columns `e - e0`,
# added to `pair[i]`.
@inline function _zbl_receiver!(zbl, dzbl, pair, d, senders, row_ptr, Z, radii, P, i, e0, nb)
    Zv = Int(Z[i])
    acc = pair[i]
    for e in _receiver_edges(row_ptr, i, e0, nb)
        Zu = Int(Z[senders[e]])
        v, dv = _zbl_edge(d[e], Zu, Zv, radii[Zu + 1] + radii[Zv + 1], P)
        zbl[e - e0], dzbl[e - e0] = v, dv
        acc += v
    end
    pair[i] = acc
    return nothing
end

# Screened Coulomb repulsion of an edge of length `d` from an atom of atomic number `Zu` to one of
# `Zv`, cut off smoothly at the sum `R` of their covalent radii; half of it is credited to the
# receiver (each pair appears as two directed edges). Returns the energy and its derivative in d.
@inline function _zbl_edge(d::T, Zu::Int, Zv::Int, R::T, P) where {T}
    (d < R && d < P.r_max) || return zero(T), zero(T)
    # Ziegler-Biersack-Littmark universal screening function (mace radial.py:203-216, MIT).
    ks = (T(3.2), T(0.9423), T(0.4028), T(0.2016))
    a = P.a_prefactor * T(0.529) / (T(Zu)^P.a_exp + T(Zv)^P.a_exp)
    ρ = d / a
    φ = zero(T)
    dφ = zero(T)
    for i in 1:4
        term = P.c[i] * exp(-ks[i] * ρ)
        φ += term
        dφ -= ks[i] * term / a
    end
    K = T(14.3996) * T(Zu) * T(Zv)
    v = K / d * φ
    dv = K * (dφ / d - φ / d^2)
    f, df = _poly_envelope_device(d / R, P.p, inv(R))
    h = T(0.5)
    return h * v * f, h * (dv * f + v * df)
end
