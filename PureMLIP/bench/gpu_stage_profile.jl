# Stage profile of the MACE evaluation on a GPU, in Float32, and the decision table that the device
# evaluation follows: per stage, the vendor matrix product (`mul!` on device arrays) or a
# KernelAbstractions kernel; which stages are fused; and the default `edge_chunk`.
#
#   GPU_BACKEND=Metal PUREMLIP_CHECKPOINT_DIR=<dir> julia --project=PureMLIP/bench/gpu/metal PureMLIP/bench/gpu_stage_profile.jl
#
#   GPU_BACKEND              Metal (default), CUDA or AMDGPU; the environment is bench/gpu/<backend in lower case>
#   PUREMLIP_CHECKPOINT_DIR  directory with mace-mpa-0-medium.model
#   STAGE_CASES              cu_bench sizes (default "5,14")
#   STAGE_MEMORY_BUDGET_GB   largest edge-chunk buffer set to allocate (default 12); a larger edge
#                            chunk is recorded as not fitting, with the bytes it needs
#   PUREMLIP_BENCH_OUT       JSON output path (default results/gpu_stage_profile_<backend>.json)
#
# On macOS, export HWLOC_COMPONENTS=-darwin first (see the KernelAbstractions rule in AGENTS.md).
#
# Every operation is synchronized and timed as the median of 20 runs after two warm-up runs. The
# per-edge and per-node kernels here are profile kernels with the arithmetic of the CPU stages
# (`src/model/`, `src/equivariant/`, `src/tensors/`); they are not compared with the CPU results.
#
# Decision rules, fixed before any measurement:
# - A matrix-shaped stage uses `mul!` when its median, including any elementwise pass it needs
#   (the activation of a radial-MLP layer, its slope in the pullback), is below the median of the
#   kernel that computes the same result in the layout the stage reads; otherwise the kernel.
#   The node stages compare `mul!` in a `(C, n, K)` layout, where each slot is one contiguous matrix,
#   with the kernel on the `(C, K, n)` layout of the CPU code. The decision is read at cu_bench(14),
#   and at the default edge chunk for the radial MLP; cu_bench(5) is reported beside it.
# - A pair of stages is fused when the fused kernel's median is below the sum of the separate ones.
# - The default `edge_chunk` is the smallest candidate whose per-edge pass (both interactions:
#   radial features, harmonics, radial MLP forward and its pullback's matrix products, messages)
#   is within 2 % of the fastest candidate, at cu_bench(14).
# - The per-edge stages are reported against a copy kernel's bandwidth: bytes each stage must read
#   and write, divided by its median, as a fraction of the copy kernel's bytes per second.
using PureMLIP, KernelAbstractions, LinearAlgebra, Statistics, JSON

const BACKEND_NAME = get(ENV, "GPU_BACKEND", "Metal")
BACKEND_NAME in ("Metal", "CUDA", "AMDGPU") || error("GPU_BACKEND must be Metal, CUDA or AMDGPU, got ", BACKEND_NAME)
@eval using $(Symbol(BACKEND_NAME))
const VENDOR = getfield(Main, Symbol(BACKEND_NAME))
VENDOR.functional() || error(BACKEND_NAME, " is not functional on this machine")
const backend = BACKEND_NAME == "Metal" ? VENDOR.MetalBackend() : BACKEND_NAME == "CUDA" ? VENDOR.CUDABackend() : VENDOR.ROCBackend()

include(joinpath(pkgdir(PureMLIP), "test", "reference", "structures.jl"))

const T = Float32
const CK = joinpath(ENV["PUREMLIP_CHECKPOINT_DIR"], "mace-mpa-0-medium.model")
const CASES = parse.(Int, split(get(ENV, "STAGE_CASES", "5,14"), ','))
const BUDGET = parse(Float64, get(ENV, "STAGE_MEMORY_BUDGET_GB", "12")) * 2^30
const OUT = get(ENV, "PUREMLIP_BENCH_OUT", joinpath(@__DIR__, "results", "gpu_stage_profile_$(lowercase(BACKEND_NAME)).json"))
const CHUNKS = (2^14, 2^16, 2^18)
const REPS = 20

# Bytes the device has allocated (vendor query).
function device_bytes()
    BACKEND_NAME == "Metal" && return Int(VENDOR.device().currentAllocatedSize)
    BACKEND_NAME == "CUDA" && return Int(VENDOR.used_memory())
    return Int(VENDOR.HIP.used_memory())
end

sync() = KernelAbstractions.synchronize(backend)

# Median, minimum and maximum of REPS synchronized runs of `f`, in microseconds.
function timed(f)
    f(); sync()
    f(); sync()
    ts = Float64[]
    for _ in 1:REPS
        t0 = time_ns()
        f()
        sync()
        push!(ts, (time_ns() - t0) / 1.0e3)
    end
    return (; median_us = median(ts), min_us = minimum(ts), max_us = maximum(ts))
end

dzeros(dims...) = KernelAbstractions.zeros(backend, T, dims...)
function drand(dims...)
    A = KernelAbstractions.allocate(backend, T, dims)
    copyto!(A, rand(T, dims...) .- T(0.5))
    return A
end
upload(A) = (D = KernelAbstractions.allocate(backend, eltype(A), size(A)); copyto!(D, A); D)

# ---------------------------------------------------------------------------------------------
# Matrix products: C[:, :, b] = α op(A[:, :, b]) op(B[:, :, b]) (+ β C), in 16 × 16 tiles, with an
# epilogue: :none, :silu (C = cst·z·σ(z), pre = σ(z), the hidden layer of the radial MLP), or
# :slope (C = z·(cst·σ + act·(1 - σ)), the pullback of that layer).
const TILE = 16

@kernel function gemm_kernel!(
        Cm, pre, @Const(act), @Const(A), @Const(B), α, β, cst, ::Val{TA}, ::Val{TB}, ::Val{EPI}
    ) where {TA, TB, EPI}
    gi, gj, b = @index(Group, NTuple)
    i, j, _ = @index(Local, NTuple)
    tileA = @localmem eltype(Cm) (TILE + 1, TILE)
    tileB = @localmem eltype(Cm) (TILE + 1, TILE)
    M = size(Cm, 1)
    N = size(Cm, 2)
    K = TA ? size(A, 1) : size(A, 2)
    I = (gi - 1) * TILE + i
    J = (gj - 1) * TILE + j
    acc = zero(eltype(Cm))
    for t in 0:(cld(K, TILE) - 1)
        ka = t * TILE + j
        tileA[i, j] = (I <= M && ka <= K) ? (TA ? A[ka, I, b] : A[I, ka, b]) : zero(eltype(Cm))
        kb = t * TILE + i
        tileB[i, j] = (kb <= K && J <= N) ? (TB ? B[J, kb, b] : B[kb, J, b]) : zero(eltype(Cm))
        @synchronize
        for k in 1:TILE
            acc += tileA[i, k] * tileB[k, j]
        end
        @synchronize
    end
    if I <= M && J <= N
        z = α * acc
        if EPI === :silu
            σ = inv(one(z) + exp(-z))
            Cm[I, J, b] = cst * z * σ
            pre[I, J, b] = σ
        elseif EPI === :slope
            σ = pre[I, J, b]
            Cm[I, J, b] = z * (cst * σ + act[I, J, b] * (one(z) - σ))
        else
            Cm[I, J, b] = iszero(β) ? z : z + β * Cm[I, J, b]
        end
    end
end

as3(A::AbstractArray{<:Any, 3}) = A
as3(A::AbstractMatrix) = reshape(A, size(A, 1), size(A, 2), 1)

function gemm!(Cm, A, B; tA = false, tB = false, α = one(T), β = zero(T), epi = :none, pre = Cm, act = Cm, cst = one(T))
    C3 = as3(Cm)
    k = gemm_kernel!(backend, (TILE, TILE, 1))
    k(C3, as3(pre), as3(act), as3(A), as3(B), T(α), T(β), T(cst), Val(tA), Val(tB), Val(epi); ndrange = (cld(size(C3, 1), TILE) * TILE, cld(size(C3, 2), TILE) * TILE, size(C3, 3)))
    return Cm
end

# Elementwise passes that `mul!` needs around the radial-MLP layers.
@kernel function silu_kernel!(act, pre, cst)
    q = @index(Global)
    z = pre[q]
    σ = inv(one(z) + exp(-z))
    act[q] = cst * z * σ
    pre[q] = σ
end
@kernel function slope_kernel!(g, @Const(pre), @Const(act), cst)
    q = @index(Global)
    σ = pre[q]
    g[q] *= cst * σ + act[q] * (one(σ) - σ)
end
silu!(act, pre, cst) = silu_kernel!(backend)(act, pre, T(cst); ndrange = length(pre))
slope!(g, pre, act, cst) = slope_kernel!(backend)(g, pre, act, T(cst); ndrange = length(g))

@kernel function copy_kernel!(dst, @Const(src))
    q = @index(Global)
    dst[q] = src[q]
end

# ---------------------------------------------------------------------------------------------
# Per-edge stages.

@kernel function edges_kernel!(r, d, @Const(pos), @Const(cell), @Const(snd), @Const(rcv), @Const(shifts))
    e = @index(Global)
    s, q = snd[e], rcv[e]
    n1, n2, n3 = T(shifts[1, e]), T(shifts[2, e]), T(shifts[3, e])
    r1 = pos[1, q] - pos[1, s] + (cell[1, 1] * n1 + cell[1, 2] * n2 + cell[1, 3] * n3)
    r2 = pos[2, q] - pos[2, s] + (cell[2, 1] * n1 + cell[2, 2] * n2 + cell[2, 3] * n3)
    r3 = pos[3, q] - pos[3, s] + (cell[3, 1] * n1 + cell[3, 2] * n2 + cell[3, 3] * n3)
    r[1, e], r[2, e], r[3, e] = r1, r2, r3
    d[e] = sqrt(r1 * r1 + r2 * r2 + r3 * r3)
end

# The radial features of `radial_embedding!` with Bessel weights k·w₁: polynomial cutoff, Agnesi
# transform, Bessel recurrence, and their length derivatives.
@inline function radial_features!(feats, dfeats, col, dd, r0, P)
    nb = size(feats, 1)
    if !(dd < P.r_max)
        for k in 1:nb
            feats[k, col] = zero(dd)
            dfeats[k, col] = zero(dd)
        end
        return nothing
    end
    p = P.p
    t = dd / P.r_max
    c1 = T((p + 1) * (p + 2)) / 2
    c2 = T(p * (p + 2))
    c3 = T(p * (p + 1)) / 2
    tp = Base.power_by_squaring(t, p)
    fc = 1 - c1 * tp + c2 * tp * t - c3 * tp * t * t
    dfc = (-c1 * p * Base.power_by_squaring(t, p - 1) + c2 * (p + 1) * tp - c3 * (p + 2) * tp * t) / P.r_max
    x = dd / r0
    xq = x^P.q
    xqp = x^(P.q - P.ap)
    den = 1 + xqp
    dt = inv(1 + P.a * xq / den)
    du = P.a * (P.q * xq * den - xq * (P.q - P.ap) * xqp) / (x * den * den)
    ddt = -dt * dt * du / r0
    s1, cc1 = sincos(P.w1 * dt)
    s, c = s1, cc1
    for k in 1:nb
        B = P.prefactor * s / dt
        dB = P.prefactor * (k * P.w1 * c / dt - s / (dt * dt))
        dfeats[k, col] = dB * ddt * fc + B * dfc
        feats[k, col] = B * fc
        s, c = s * cc1 + c * s1, c * cc1 - s * s1
    end
    return nothing
end

@kernel function radial_kernel!(feats, dfeats, @Const(d), @Const(Z), @Const(snd), @Const(rcv), @Const(radii), P, e0)
    j = @index(Global)
    e = e0 + j - 1
    r0 = (radii[Z[snd[e]] + 1] + radii[Z[rcv[e]] + 1]) / 2
    radial_features!(feats, dfeats, j, d[e], r0, P)
end

# One fused kernel: the edge vector and length, then the radial features.
@kernel function edges_radial_kernel!(
        r, d, feats, dfeats, @Const(pos), @Const(cell), @Const(snd), @Const(rcv), @Const(shifts), @Const(Z), @Const(radii), P
    )
    e = @index(Global)
    s, q = snd[e], rcv[e]
    n1, n2, n3 = T(shifts[1, e]), T(shifts[2, e]), T(shifts[3, e])
    r1 = pos[1, q] - pos[1, s] + (cell[1, 1] * n1 + cell[1, 2] * n2 + cell[1, 3] * n3)
    r2 = pos[2, q] - pos[2, s] + (cell[2, 1] * n1 + cell[2, 2] * n2 + cell[2, 3] * n3)
    r3 = pos[3, q] - pos[3, s] + (cell[3, 1] * n1 + cell[3, 2] * n2 + cell[3, 3] * n3)
    r[1, e], r[2, e], r[3, e] = r1, r2, r3
    dd = sqrt(r1 * r1 + r2 * r2 + r3 * r3)
    d[e] = dd
    radial_features!(feats, dfeats, e, dd, (radii[Z[s] + 1] + radii[Z[q] + 1]) / 2, P)
end

# e3nn spherical harmonics up to l = 3 of one vector into column `col` of `Y` (16 rows), and with
# `G` their gradient into `G[:, :, col]`: the recurrence of `_real_sh!` (src/equivariant).
shi(l, m) = l * l + l + m + 1
@inline function sh_column!(Y, G, col, vx0, vy0, vz0)
    # e3nn order: the native harmonics of (v3, v1, v2).
    vx, vy, vz = vz0, vx0, vy0
    r = sqrt(vx * vx + vy * vy + vz * vz)
    x, y, z = vx / r, vy / r, vz / r
    L = 3
    Y[shi(0, 0), col] = one(T)
    Y[shi(1, 1), col] = -one(T)
    Y[shi(1, 0), col] = z
    for l in 2:L
        qll = Y[shi(l - 1, l - 1), col]
        Y[shi(l, l), col] = -(2l - 1) * qll
        Y[shi(l, l - 1), col] = (2l - 1) * z * qll
        for m in (l - 2):-1:0
            Y[shi(l, m), col] = ((2l - 1) * z * Y[shi(l - 1, m), col] - (l + m - 1) * Y[shi(l - 2, m), col]) / (l - m)
        end
    end
    rt2 = sqrt(T(2))
    c4 = sqrt(4 * T(π))
    for l in L:-1:0
        F = sqrt(T(2l + 1) / (2 * T(π)))
        s, c = zero(T), one(T)
        sp, cp = zero(T), zero(T)
        for m in 0:l
            if m > 0
                F = -F / sqrt(T((l + m) * (l + 1 - m)))
                sp, cp = s, c
                s, c = sp * x + cp * y, cp * x - sp * y
            end
            Qlm = Y[shi(l, m), col]
            if !isnothing(G)
                qa = m + 1 <= l - 1 ? Y[shi(l - 1, m + 1), col] : zero(T)
                qb = m <= l - 1 ? (l + m) * Y[shi(l - 1, m), col] : zero(T)
                if iszero(m)
                    Fm = F / rt2
                    grad_row!(G, shi(l, 0), col, Fm * x * qa, Fm * y * qa, Fm * qb, x, y, z, r, c4)
                else
                    grad_row!(G, shi(l, -m), col, F * (Qlm * m * sp + x * qa * s), F * (Qlm * m * cp + y * qa * s), F * qb * s, x, y, z, r, c4)
                    grad_row!(G, shi(l, m), col, F * (Qlm * m * cp + x * qa * c), F * (-Qlm * m * sp + y * qa * c), F * qb * c, x, y, z, r, c4)
                end
            end
            if iszero(m)
                Y[shi(l, 0), col] = c4 * F / rt2 * Qlm
            else
                Y[shi(l, -m), col] = c4 * F * Qlm * s
                Y[shi(l, m), col] = c4 * F * Qlm * c
            end
        end
    end
    return nothing
end

# The gradient row in e3nn's column order (∂/∂v1, ∂/∂v2, ∂/∂v3).
@inline function grad_row!(G, k, col, gx, gy, gz, x, y, z, r, c4)
    dd = x * gx + y * gy + z * gz
    G[k, 3, col] = c4 * (gx - dd * x) / r
    G[k, 1, col] = c4 * (gy - dd * y) / r
    G[k, 2, col] = c4 * (gz - dd * z) / r
    return nothing
end

@kernel function sh_kernel!(Y, @Const(r), e0)
    j = @index(Global)
    e = e0 + j - 1
    sh_column!(Y, nothing, j, r[1, e], r[2, e], r[3, e])
end
@kernel function sh_grad_kernel!(Y, G, @Const(r), e0)
    j = @index(Global)
    e = e0 + j - 1
    sh_column!(Y, G, j, r[1, e], r[2, e], r[3, e])
end

# Summed messages of the receivers `a0 + 1 : a0 + n` (one work item per channel and receiver), over
# their edges, and the density of each receiver: the channel-wise tensor product of
# `tp_edge_add!` and `density_edge`. Column `e - e0 + 1` of the edge-chunk buffers is edge `e`.
@kernel function messages_kernel!(
        Mg, dens, @Const(x), @Const(Y), @Const(tpw3), @Const(feats), @Const(dw), @Const(row_ptr), @Const(snd),
        @Const(idx), @Const(val), @Const(path), a0, e0
    )
    u, il = @index(Global, NTuple)
    i = a0 + il
    for k in axes(Mg, 2)
        Mg[u, k, i] = zero(T)
    end
    ρ = zero(T)
    for e in row_ptr[i]:(row_ptr[i + 1] - 1)
        j = e - e0 + 1
        s = snd[e]
        for q in eachindex(val)
            Mg[u, idx[3, q], i] += val[q] * Y[idx[2, q], j] * tpw3[u, path[q], j] * x[u, idx[1, q], s]
        end
        if u == 1
            zz = zero(T)
            for b in axes(feats, 1)
                zz += feats[b, j] * dw[b]
            end
            ρ += tanh(zz * zz)
        end
    end
    u == 1 && (dens[i] = ρ)
end

# The same messages with the harmonics of each edge computed in the kernel from the edge vector.
@kernel function messages_sh_kernel!(
        Mg, dens, @Const(x), @Const(r), @Const(tpw3), @Const(feats), @Const(dw), @Const(row_ptr), @Const(snd),
        @Const(idx), @Const(val), @Const(path), a0, e0
    )
    u, il = @index(Global, NTuple)
    i = a0 + il
    Yp = @private T (16, 1)
    for k in axes(Mg, 2)
        Mg[u, k, i] = zero(T)
    end
    ρ = zero(T)
    for e in row_ptr[i]:(row_ptr[i + 1] - 1)
        j = e - e0 + 1
        s = snd[e]
        sh_column!(Yp, nothing, 1, r[1, e], r[2, e], r[3, e])
        for q in eachindex(val)
            Mg[u, idx[3, q], i] += val[q] * Yp[idx[2, q], 1] * tpw3[u, path[q], j] * x[u, idx[1, q], s]
        end
        if u == 1
            zz = zero(T)
            for b in axes(feats, 1)
                zz += feats[b, j] * dw[b]
            end
            ρ += tanh(zz * zz)
        end
    end
    u == 1 && (dens[i] = ρ)
end

# The ZBL edge energy and its length derivative of `zbl_edge_energy`.
@inline function zbl_edge(dd, Zu, Zv, R, Z)
    dd < R || return zero(T), zero(T)
    ks = (T(3.2), T(0.9423), T(0.4028), T(0.2016))
    a = Z.a_prefactor * T(0.529) / (T(Zu)^Z.a_exp + T(Zv)^Z.a_exp)
    ρ = dd / a
    φ = zero(T)
    dφ = zero(T)
    for k in 1:4
        term = Z.c[k] * exp(-ks[k] * ρ)
        φ += term
        dφ -= ks[k] * term / a
    end
    K = T(14.3996) * T(Zu) * T(Zv)
    v = K / dd * φ
    dv = K * (dφ / dd - φ / (dd * dd))
    t = dd / R
    p = Z.p
    c1 = (p + 1) * (p + 2) / 2
    c2 = p * (p + 2)
    c3 = p * (p + 1) / 2
    f = 1 - c1 * t^p + c2 * t^(p + 1) - c3 * t^(p + 2)
    df = (-c1 * p * t^(p - 1) + c2 * (p + 1) * t^p - c3 * (p + 2) * t^(p + 1)) / R
    h = T(0.5)
    return h * v * f, h * (dv * f + v * df)
end

@kernel function zbl_kernel!(ez, dez, @Const(d), @Const(Z), @Const(snd), @Const(rcv), @Const(radii), P)
    e = @index(Global)
    Zu, Zv = Z[snd[e]], Z[rcv[e]]
    ez[e], dez[e] = zbl_edge(d[e], Zu, Zv, radii[Zu + 1] + radii[Zv + 1], P)
end
@kernel function pair_kernel!(pair, @Const(ez), @Const(row_ptr))
    i = @index(Global)
    acc = zero(T)
    for e in row_ptr[i]:(row_ptr[i + 1] - 1)
        acc += ez[e]
    end
    pair[i] = acc
end
@kernel function zbl_pair_kernel!(pair, @Const(d), @Const(Z), @Const(snd), @Const(row_ptr), @Const(radii), P)
    i = @index(Global)
    acc = zero(T)
    Zv = Z[i]
    for e in row_ptr[i]:(row_ptr[i + 1] - 1)
        Zu = Z[snd[e]]
        acc += zbl_edge(d[e], Zu, Zv, radii[Zu + 1] + radii[Zv + 1], P)[1]
    end
    pair[i] = acc
end

# ---------------------------------------------------------------------------------------------
# Node stages.

# The symmetric-contraction features of `_features!` in a (features, atom, channel) layout, so
# that each channel's features are one contiguous matrix.
@kernel function features_kernel!(Zf, @Const(X), @Const(pairs), L)
    a, c = @index(Global, NTuple)
    np = size(pairs, 2)
    for q in 1:np
        Zf[q, a, c] = X[c, pairs[1, q], a] * X[c, pairs[2, q], a]
    end
    for xx in 1:L
        Zf[np + xx, a, c] = X[c, xx, a]
    end
    Zf[np + L + 1, a, c] = one(T)
end
# B[c, j, a] = Σ_w Y[(j - 1)L + w, a, c] X[c, w, a] (`_forward_group!`).
@kernel function contract_kernel!(B, @Const(Yc), @Const(X), L)
    c, j, a = @index(Global, NTuple)
    acc = zero(T)
    for w in 1:L
        acc += Yc[(j - 1) * L + w, a, c] * X[c, w, a]
    end
    B[c, j, a] = acc
end

# The species linear map of `species_linear!` for one instruction on the (C, K, n) layout: one
# work item per output channel, component and atom.
@kernel function species_linear_kernel!(y, @Const(W), @Const(x), @Const(species), coef, p, ri0, ro0, dd)
    w, m, i = @index(Global, NTuple)
    s = species[i]
    acc = zero(T)
    for u in axes(x, 1)
        acc += W[w, u, s, p] * x[u, ri0 + m, i]
    end
    y[w, ro0 + m, i] += coef * acc
end

@kernel function normalize_kernel!(Lx, @Const(dens))
    u, k, i = @index(Global, NTuple)
    Lx[u, k, i] /= dens[i] + one(T)
end

# ---------------------------------------------------------------------------------------------

# Receiver chunks: ranges of whole receivers with at most `chunk` edges each.
function receiver_chunks(row_ptr::Vector{Int32}, chunk::Int)
    out = Tuple{UnitRange{Int}, UnitRange{Int}}[]
    n = length(row_ptr) - 1
    a = 1
    while a <= n
        b = a
        while b < n && row_ptr[b + 2] - row_ptr[a] <= chunk
            b += 1
        end
        row_ptr[b + 1] - row_ptr[a] <= chunk || error("receiver $a has more than $chunk edges")
        push!(out, (a:b, Int(row_ptr[a]):Int(row_ptr[b + 1] - 1)))
        a = b + 1
    end
    return out
end

device_array_bytes(A) = sizeof(eltype(A)) * length(A)
function edge_chunk_bytes(ws)
    fields = (:Y, :dY, :feats, :dfeats, :tpw, :pre, :act, :dtpw, :mlp_scratch, :dfeats_chunk, :zbl, :dzbl)
    return sum(f -> device_array_bytes(getfield(ws, f)), fields)
end

stage(f) = timed(f)

function profile_case(N, dm, model, ext)
    cu = cu_bench(N)
    pos_h = T.(permutedims(cu.positions))
    cell_h = T.(permutedims(cu.cell))
    nl = neighbor_list(pos_h, cell_h, Float64(model.r_max))
    n, E = nl.n_atoms, nl.n_edges
    println("cu_bench($N): $n atoms, $E edges"); flush(stdout)
    dnl = PureMLIP.device_neighbor_list(nl, backend)
    row_ptr_h = Array(dnl.row_ptr)
    pos, cell = upload(pos_h), upload(cell_h)
    Zd = upload(Int32.(cu.atomic_numbers))
    C = model.C
    nb = length(model.radial.bessel_weights)
    rad = model.radial
    @assert rad.multiples
    P = (; r_max = rad.r_max, p = rad.cutoff_p, a = rad.agnesi_a, ap = rad.agnesi_p, q = rad.agnesi_q, w1 = rad.bessel_weights[1], prefactor = rad.prefactor)
    zb = model.zbl
    ZP = (; c = zb.c, p = zb.p, a_exp = zb.a_exp, a_prefactor = zb.a_prefactor)
    snd, rcv, shifts = dnl.senders, dnl.receivers, dnl.shifts
    r, d = dzeros(3, E), dzeros(E)
    rec = Dict{String, Any}("n_atoms" => n, "n_edges" => E)

    # Bandwidth reference: a copy of 2^26 floats.
    src, dst = drand(2^26), dzeros(2^26)
    tc = timed(() -> copy_kernel!(backend)(dst, src; ndrange = length(src)))
    copy_Bps = 2 * sizeof(T) * 2^26 / (tc.median_us * 1.0e-6)
    rec["copy_reference"] = (; bytes = 2 * sizeof(T) * 2^26, tc..., bytes_per_second = copy_Bps)
    src = dst = nothing

    # ---- per-edge stages over all edges (one chunk of every edge) and the fusion candidates
    feats, dfeats = dzeros(nb, E), dzeros(nb, E)
    Y, G = dzeros(16, E), dzeros(16, 3, E)
    per_edge = Dict{String, Any}()
    bw(bytes, t) = (; t..., bytes, bandwidth_fraction = bytes / (t.median_us * 1.0e-6) / copy_Bps)
    t_edges = stage(() -> edges_kernel!(backend)(r, d, pos, cell, snd, rcv, shifts; ndrange = E))
    per_edge["edges"] = bw(E * (2 * 12 + 12 + 8 + 16), t_edges)
    t_radial = stage(() -> radial_kernel!(backend)(feats, dfeats, d, Zd, snd, rcv, dm.radial_radii, P, 1; ndrange = E))
    per_edge["radial_features"] = bw(E * (4 + 8 + 8 + 2 * 4 * nb), t_radial)
    t_er = stage(() -> edges_radial_kernel!(backend)(r, d, feats, dfeats, pos, cell, snd, rcv, shifts, Zd, dm.radial_radii, P; ndrange = E))
    per_edge["edges_and_radial_features_fused"] = bw(E * (2 * 12 + 12 + 8 + 16 + 8 + 2 * 4 * nb), t_er)
    t_sh = stage(() -> sh_kernel!(backend)(Y, r, 1; ndrange = E))
    per_edge["spherical_harmonics"] = bw(E * (12 + 64), t_sh)
    t_shg = stage(() -> sh_grad_kernel!(backend)(Y, G, r, 1; ndrange = E))
    per_edge["spherical_harmonics_and_gradient"] = bw(E * (12 + 64 + 192), t_shg)
    ez, dez, pair = dzeros(E), dzeros(E), dzeros(n)
    t_zbl = stage(() -> zbl_kernel!(backend)(ez, dez, d, Zd, snd, rcv, dm.zbl_radii, ZP; ndrange = E))
    per_edge["zbl"] = bw(E * (4 + 8 + 8 + 8 + 8), t_zbl)
    t_pair = stage(() -> pair_kernel!(backend)(pair, ez, dnl.row_ptr; ndrange = n))
    per_edge["pair_sum"] = bw(E * 4 + n * 12, t_pair)
    t_zp = stage(() -> zbl_pair_kernel!(backend)(pair, d, Zd, snd, dnl.row_ptr, dm.zbl_radii, ZP; ndrange = n))
    per_edge["zbl_and_pair_sum_fused"] = bw(E * (4 + 8 + 4 + 8) + n * 12, t_zp)
    # Messages of both interactions over every edge, with random sender features and weights of
    # the right sizes.
    msg = Dict{String, Any}()
    for k in 1:2
        tp = model.conv[k]
        Pk = length(tp.paths)
        tpw3 = drand(C, Pk, E)
        x = drand(C, tp.K_in1, n)
        Mg = dzeros(C, tp.K_out, n)
        dens = dzeros(n)
        path = upload(Int32[p for p in eachindex(tp.paths) for _ in tp.paths[p]])
        dtp = dm.conv[k]
        dw = upload(model.density[k] ./ sqrt(T(nb)))
        t_m = stage(() -> messages_kernel!(backend)(Mg, dens, x, Y, tpw3, feats, dw, dnl.row_ptr, snd, dtp.index, dtp.value, path, 0, 1; ndrange = (C, n)))
        t_ms = stage(() -> messages_sh_kernel!(backend)(Mg, dens, x, r, tpw3, feats, dw, dnl.row_ptr, snd, dtp.index, dtp.value, path, 0, 1; ndrange = (C, n)))
        bytes = E * 4 * (C * Pk + 16 + C * tp.K_in1 + nb + 1) + n * C * tp.K_out * 4
        per_edge["messages_$k"] = bw(bytes, t_m)
        msg["messages_$k"] = (; separate_us = t_sh.median_us + t_m.median_us, fused_us = t_ms.median_us, sh = t_sh, messages = t_m, fused = t_ms)
        tpw3 = x = Mg = nothing
    end
    rec["per_edge"] = per_edge
    fusion = Dict{String, Any}()
    fusion["edges+radial_features"] = (; separate_us = t_edges.median_us + t_radial.median_us, fused_us = t_er.median_us)
    fusion["zbl+pair_sum"] = (; separate_us = t_zbl.median_us + t_pair.median_us, fused_us = t_zp.median_us)
    for k in 1:2
        fusion["spherical_harmonics+messages_$k"] = (; separate_us = msg["messages_$k"].separate_us, fused_us = msg["messages_$k"].fused_us)
    end
    for (kk, v) in fusion
        fusion[kk] = (; v..., fuse = v.fused_us < v.separate_us)
    end
    rec["fusion"] = fusion
    feats = dfeats = Y = G = ez = dez = nothing
    GC.gc(true)

    # ---- radial MLP: mul! (+ elementwise pass) against the kernel, at each edge chunk
    mlp_shapes = Dict{String, Any}()
    # Bytes of the edge-chunk buffers of a workspace for `nc` edges (see DeviceMACEWorkspace).
    Wmax = max(size(dm.mlp[1][4], 2), size(dm.mlp[2][4], 2))
    Hmax = size(dm.mlp[1][1], 2)
    chunk_need(nc) = sizeof(T) * nc * (16 + 48 + 3 * nb + 2 * Wmax + 8 * Hmax + 2)
    mlp_cands = unique([min(c, E) for c in (CHUNKS..., E)])
    for nc in mlp_cands
        rows = Dict{String, Any}()
        if chunk_need(nc) > BUDGET
            mlp_shapes[string(nc)] = rows
            continue
        end
        for k in 1:2
            W = dm.mlp[k]
            H = size(W[1], 2)
            Wout = size(W[4], 2)
            begin
                f0 = drand(nb, nc)
                a1, a2, p1, p2 = dzeros(H, nc), dzeros(H, nc), dzeros(H, nc), dzeros(H, nc)
                out = dzeros(Wout, nc)
                dout = drand(Wout, nc)
                g1, g2 = dzeros(H, nc), dzeros(H, nc)
                df = dzeros(nb, nc)
                cst = model.mlp[k].cst
                arms = Dict{String, Any}()
                k == 1 && (arms["layer1_fwd"] = (mul = stage(() -> (mul!(p1, transpose(W[1]), f0); silu!(a1, p1, cst))), kernel = stage(() -> gemm!(a1, W[1], f0; tA = true, epi = :silu, pre = p1, cst))))
                k == 1 && (arms["hidden_fwd"] = (mul = stage(() -> (mul!(p2, transpose(W[2]), a1); silu!(a2, p2, cst))), kernel = stage(() -> gemm!(a2, W[2], a1; tA = true, epi = :silu, pre = p2, cst))))
                arms["layer4_fwd"] = (mul = stage(() -> mul!(out, transpose(W[4]), a2)), kernel = stage(() -> gemm!(out, W[4], a2; tA = true)))
                arms["layer4_bwd"] = (mul = stage(() -> (mul!(g1, W[4], dout); slope!(g1, p2, a2, cst))), kernel = stage(() -> gemm!(g1, W[4], dout; epi = :slope, pre = p2, act = a2, cst)))
                k == 1 && (arms["hidden_bwd"] = (mul = stage(() -> (mul!(g2, W[3], g1); slope!(g2, p1, a1, cst))), kernel = stage(() -> gemm!(g2, W[3], g1; epi = :slope, pre = p1, act = a1, cst))))
                k == 1 && (arms["layer1_bwd"] = (mul = stage(() -> mul!(df, W[1], g2)), kernel = stage(() -> gemm!(df, W[1], g2))))
                for (name, a) in arms
                    key = k == 1 && !startswith(name, "layer4") ? name : string(name, "_interaction", k)
                    rows[key] = (; mul_us = a.mul.median_us, kernel_us = a.kernel.median_us, mul = a.mul, kernel = a.kernel)
                end
            end
        end
        mlp_shapes[string(nc)] = rows
        GC.gc(true)
    end
    rec["radial_mlp"] = mlp_shapes

    # ---- edge-chunk sweep: the per-edge pass of both interactions in receiver chunks
    sweep = Dict{String, Any}()
    for nc in mlp_cands
        tag = nc == E ? "all" : string(nc)
        GC.gc(true)
        need = chunk_need(nc)
        if need > BUDGET
            sweep[tag] = (; edge_chunk = nc, fits = false, edge_chunk_bytes = need)
            println("  edge_chunk $tag: needs $(need / 2^30) GiB, above the budget"); flush(stdout)
            continue
        end
        GC.gc(true)
        base = device_bytes()
        ws = PureMLIP.device_workspace(dm, n, E; edge_chunk = nc)
        chunks = receiver_chunks(row_ptr_h, nc)
        rows = mlp_shapes[string(nc)]
        usemul(name) = rows[name].mul_us < rows[name].kernel_us
        xs = (drand(C, model.conv[1].K_in1, n), drand(C, model.conv[2].K_in1, n))
        dens = dzeros(n)
        paths = [upload(Int32[p for p in eachindex(tp.paths) for _ in tp.paths[p]]) for tp in model.conv]
        dws_ = [upload(model.density[k] ./ sqrt(T(nb))) for k in 1:2]
        copyto!(ws.dtpw, rand(T, size(ws.dtpw)...))
        dout = ws.dtpw
        edge_chunk_bytes(ws) == need || error("the edge-chunk bytes of the workspace differ from chunk_need")
        edges_kernel!(backend)(ws.r, ws.d, pos, cell, snd, rcv, shifts; ndrange = E)
        function pass()
            for k in 1:2
                W = dm.mlp[k]
                cst = model.mlp[k].cst
                Wout = size(W[4], 2)
                Pk = Wout ÷ C
                tp = dm.conv[k]
                for (atoms, edges) in chunks
                    ne = length(edges)
                    e0 = first(edges)
                    f = view(ws.feats, :, 1:ne)
                    radial_kernel!(backend)(ws.feats, ws.dfeats, ws.d, Zd, snd, rcv, dm.radial_radii, P, e0; ndrange = ne)
                    sh_kernel!(backend)(ws.Y, ws.r, e0; ndrange = ne)
                    a1, a2, a3 = view(ws.act, :, 1:ne, 1), view(ws.act, :, 1:ne, 2), view(ws.act, :, 1:ne, 3)
                    p1, p2, p3 = view(ws.pre, :, 1:ne, 1), view(ws.pre, :, 1:ne, 2), view(ws.pre, :, 1:ne, 3)
                    for (Wl, inp, a, p, name) in ((W[1], f, a1, p1, "layer1_fwd"), (W[2], a1, a2, p2, "hidden_fwd"), (W[3], a2, a3, p3, "hidden_fwd"))
                        usemul(name) ? (mul!(p, transpose(Wl), inp); silu!(a, p, cst)) : gemm!(a, Wl, inp; tA = true, epi = :silu, pre = p, cst)
                    end
                    out = view(ws.tpw, 1:Wout, 1:ne)
                    usemul("layer4_fwd_interaction$k") ? mul!(out, transpose(W[4]), a3) : gemm!(out, W[4], a3; tA = true)
                    tpw3 = reshape(view(ws.tpw, 1:Wout, :), C, Pk, size(ws.tpw, 2))
                    messages_kernel!(backend)(ws.M[k], dens, xs[k], ws.Y, tpw3, ws.feats, dws_[k], dnl.row_ptr, snd, tp.index, tp.value, paths[k], first(atoms) - 1, e0; ndrange = (C, length(atoms)))
                    # The matrix products of the radial-MLP pullback over the same edges.
                    g1, g2 = view(ws.mlp_scratch, :, 1:ne, 1), view(ws.mlp_scratch, :, 1:ne, 2)
                    dv = view(dout, 1:Wout, 1:ne)
                    usemul("layer4_bwd_interaction$k") ? (mul!(g1, W[4], dv); slope!(g1, p3, a3, cst)) : gemm!(g1, W[4], dv; epi = :slope, pre = p3, act = a3, cst)
                    usemul("hidden_bwd") ? (mul!(g2, W[3], g1); slope!(g2, p2, a2, cst)) : gemm!(g2, W[3], g1; epi = :slope, pre = p2, act = a2, cst)
                    usemul("hidden_bwd") ? (mul!(g1, W[2], g2); slope!(g1, p1, a1, cst)) : gemm!(g1, W[2], g2; epi = :slope, pre = p1, act = a1, cst)
                    df = view(ws.dfeats_chunk, :, 1:ne)
                    usemul("layer1_bwd") ? mul!(df, W[1], g1) : gemm!(df, W[1], g1)
                end
            end
            return nothing
        end
        t = timed(pass)
        peak = device_bytes() - base
        sweep[tag] = (; edge_chunk = nc, fits = true, chunks = length(chunks), t..., edge_chunk_bytes = need, device_bytes_above_start = peak)
        println("  edge_chunk $tag: $(round(t.median_us / 1.0e3; digits = 2)) ms, $(round(need / 2^20; digits = 1)) MiB of edge-chunk buffers"); flush(stdout)
        ws = xs = dout = nothing
    end
    rec["edge_chunk_sweep"] = sweep

    # ---- node stages: e3nn linear per irrep, species linear, symmetric contraction
    lin = Dict{String, Any}()
    per_irrep = Dict{String, Any}()
    layers = (
        ("up_1", model.up[1], dm.up[1]), ("up_2", model.up[2], dm.up[2]), ("linear_1", model.linear[1], dm.linear[1]),
        ("linear_2", model.linear[2], dm.linear[2]), ("product_linear_1", model.product_linear[1], dm.product_linear[1]),
        ("product_linear_2", model.product_linear[2], dm.product_linear[2]), ("readout_0", model.readout0, dm.readout0),
        ("readout_1a", model.readout1[1], dm.readout1[1]), ("readout_1b", model.readout1[2], dm.readout1[2]),
    )
    for (name, hl, dl) in layers
        off_in, off_out = hl.in_off, hl.out_off
        # (C, K, n) layout for the kernel; (C, n, K) for mul!.
        xk, yk = drand(hl.C_in, hl.K_in, n), dzeros(hl.C_out, hl.K_out, n)
        xm, ym = drand(hl.C_in, n, hl.K_in), dzeros(hl.C_out, n, hl.K_out)
        # Each function runs the instructions `ps` of the layer, one matrix product per irrep slot.
        function kernel_fwd(ps)
            for p in ps
                i, o = hl.slots[p]
                dd = 2 * hl.in_ls[i] + 1
                xs = reshape(view(xk, :, (off_in[i] + 1):(off_in[i] + dd), :), hl.C_in, dd * n)
                ys = reshape(view(yk, :, (off_out[o] + 1):(off_out[o] + dd), :), hl.C_out, dd * n)
                gemm!(ys, view(dl.W, :, :, p), xs; α = hl.pw[p], β = one(T))
            end
            return
        end
        function mul_fwd(ps)
            for p in ps
                i, o = hl.slots[p]
                dd = 2 * hl.in_ls[i] + 1
                xs = reshape(view(xm, :, :, (off_in[i] + 1):(off_in[i] + dd)), hl.C_in, n * dd)
                ys = reshape(view(ym, :, :, (off_out[o] + 1):(off_out[o] + dd)), hl.C_out, n * dd)
                mul!(ys, view(dl.W, :, :, p), xs, hl.pw[p], true)
            end
            return
        end
        dyk, dxk = drand(hl.C_out, hl.K_out, n), dzeros(hl.C_in, hl.K_in, n)
        dym, dxm = drand(hl.C_out, n, hl.K_out), dzeros(hl.C_in, n, hl.K_in)
        function kernel_bwd(ps)
            for p in ps
                i, o = hl.slots[p]
                dd = 2 * hl.in_ls[i] + 1
                xs = reshape(view(dxk, :, (off_in[i] + 1):(off_in[i] + dd), :), hl.C_in, dd * n)
                ys = reshape(view(dyk, :, (off_out[o] + 1):(off_out[o] + dd), :), hl.C_out, dd * n)
                gemm!(xs, view(dl.W, :, :, p), ys; tA = true, α = hl.pw[p], β = one(T))
            end
            return
        end
        function mul_bwd(ps)
            for p in ps
                i, o = hl.slots[p]
                dd = 2 * hl.in_ls[i] + 1
                xs = reshape(view(dxm, :, :, (off_in[i] + 1):(off_in[i] + dd)), hl.C_in, n * dd)
                ys = reshape(view(dym, :, :, (off_out[o] + 1):(off_out[o] + dd)), hl.C_out, n * dd)
                mul!(xs, transpose(view(dl.W, :, :, p)), ys, hl.pw[p], true)
            end
            return
        end
        every = eachindex(hl.slots)
        tf_k = stage(() -> (fill!(yk, zero(T)); kernel_fwd(every)))
        tf_m = stage(() -> (fill!(ym, zero(T)); mul_fwd(every)))
        tb_k = stage(() -> (fill!(dxk, zero(T)); kernel_bwd(every)))
        tb_m = stage(() -> (fill!(dxm, zero(T)); mul_bwd(every)))
        lin[name * "_fwd"] = (; mul_us = tf_m.median_us, kernel_us = tf_k.median_us, mul = tf_m, kernel = tf_k)
        lin[name * "_bwd"] = (; mul_us = tb_m.median_us, kernel_us = tb_k.median_us, mul = tb_m, kernel = tb_k)
        # Per irrep: one instruction (slot of degree l) at a time, accumulating into the output.
        for p in every
            l = hl.in_ls[hl.slots[p][1]]
            for (dir, fk, fm) in (("fwd", kernel_fwd, mul_fwd), ("bwd", kernel_bwd, mul_bwd))
                tk, tm = stage(() -> fk(p:p)), stage(() -> fm(p:p))
                per_irrep[string(name, "_", dir, "_instruction", p, "_l", l)] = (; mul_us = tm.median_us, kernel_us = tk.median_us, mul = tm, kernel = tk)
            end
        end
    end
    # Species linear (skip_tp of interaction 0): cu_bench holds one species, so `mul!` needs one
    # product per slot over every atom; the kernel reads each atom's species.
    sl, dsl = model.skip[1], dm.skip[1]
    s_cu = model.species[30]
    spd = upload(fill(Int32(s_cu), n))
    xk, yk = drand(C, sl.K, n), dzeros(C, sl.K, n)
    xm, ym = drand(C, n, sl.K), dzeros(C, n, sl.K)
    function skip_kernel()
        fill!(yk, zero(T))
        for (p, (i, o)) in pairs(sl.slots)
            dd = 2 * sl.ls[i] + 1
            species_linear_kernel!(backend)(yk, dsl.W, xk, spd, sl.c[p], p, sl.off[i], sl.off[o], dd; ndrange = (C, dd, n))
        end
        return
    end
    function skip_mul()
        fill!(ym, zero(T))
        for (p, (i, o)) in pairs(sl.slots)
            dd = 2 * sl.ls[i] + 1
            xs = reshape(view(xm, :, :, (sl.off[i] + 1):(sl.off[i] + dd)), C, n * dd)
            ys = reshape(view(ym, :, :, (sl.off[o] + 1):(sl.off[o] + dd)), C, n * dd)
            mul!(ys, view(dsl.W, :, :, s_cu, p), xs, sl.c[p], true)
        end
        return
    end
    tk, tm = stage(skip_kernel), stage(skip_mul)
    lin["species_linear_1_fwd"] = (; mul_us = tm.median_us, kernel_us = tk.median_us, mul = tm, kernel = tk)
    rec["node_linear"] = lin
    rec["node_linear_per_irrep"] = per_irrep

    # Symmetric contraction per species over all of that species' atoms: the channel products
    # Y[:, :, c] = F[:, :, c]ᵀ Z[:, :, c], one `mul!` per channel, against one batched kernel. The
    # pullback matrices G have the shape of F, so the same comparison holds for the pullback.
    sc = Dict{String, Any}()
    for k in 1:2
        dcm = dm.contraction[k]
        j = dcm.slot[s_cu]
        nf, ncol = size(dcm.F, 1), size(dcm.F, 2)
        X = drand(C, dcm.L, n)
        Zf = dzeros(nf, n, C)
        Yc = dzeros(ncol, n, C)
        B = dzeros(C, dcm.K_out, n)
        Fj = view(dcm.F, :, :, :, j)
        t_feat = stage(() -> features_kernel!(backend)(Zf, X, dcm.pairs, dcm.L; ndrange = (n, C)))
        t_mul = stage(() -> foreach(c -> mul!(view(Yc, :, :, c), transpose(view(Fj, :, :, c)), view(Zf, :, :, c)), 1:C))
        t_ker = stage(() -> gemm!(Yc, Fj, Zf; tA = true))
        t_con = stage(() -> contract_kernel!(backend)(B, Yc, X, dcm.L; ndrange = (C, dcm.K_out, n)))
        sc["product_$k"] = (; mul_us = t_mul.median_us, kernel_us = t_ker.median_us, mul = t_mul, kernel = t_ker, features = t_feat, contract = t_con, nf, ncol)
    end
    rec["symmetric_contraction"] = sc
    # A per-node elementwise stage (the density normalization of the interaction-1 linear output).
    Lx = drand(C, model.linear[2].K_out, n)
    dn = drand(n)
    rec["node_elementwise"] = stage(() -> normalize_kernel!(backend)(Lx, dn; ndrange = size(Lx)))
    return rec
end

choose(row) = row["mul_us"] < row["kernel_us"] ? "mul!" : "kernel"

# The decision table from the measured cases (as read back from JSON).
function decide(cases)
    target = haskey(cases, "cu_bench(14)") ? "cu_bench(14)" : last(sort(collect(keys(cases))))
    tc = cases[target]

    # Default edge chunk: the smallest candidate within 2 % of the fastest that fits.
    fitting = [(v["edge_chunk"], v["median_us"]) for v in values(tc["edge_chunk_sweep"]) if v["fits"]]
    best = minimum(last, fitting)
    default_chunk = minimum(first(f) for f in fitting if last(f) <= 1.02 * best)

    matrix_rows(c, chunk) = vcat(
        [("radial_mlp_" * nm, row) for (nm, row) in c["radial_mlp"][string(chunk)]],
        [(nm, row) for (nm, row) in c["node_linear"]],
        [(nm, row) for (nm, row) in c["node_linear_per_irrep"]],
        [("symmetric_contraction_" * nm, row) for (nm, row) in c["symmetric_contraction"]],
    )
    stages = Dict{String, Any}()
    for (nm, row) in matrix_rows(tc, default_chunk)
        stages[nm] = Dict("method" => choose(row), "mul_us" => row["mul_us"], "kernel_us" => row["kernel_us"])
    end
    # Stages with no matrix form run as kernels; their medians at the target case.
    for nm in ("edges", "radial_features", "spherical_harmonics", "spherical_harmonics_and_gradient", "messages_1", "messages_2", "zbl", "pair_sum")
        stages[nm] = Dict("method" => "kernel", "kernel_us" => tc["per_edge"][nm]["median_us"], "bandwidth_fraction" => tc["per_edge"][nm]["bandwidth_fraction"])
    end
    for (nm, row) in tc["symmetric_contraction"]
        stages["symmetric_contraction_features_" * nm] = Dict("method" => "kernel", "kernel_us" => row["features"]["median_us"])
        stages["symmetric_contraction_contract_" * nm] = Dict("method" => "kernel", "kernel_us" => row["contract"]["median_us"])
    end
    stages["node_elementwise"] = Dict("method" => "kernel", "kernel_us" => tc["node_elementwise"]["median_us"])

    # The same choices at every case, so a disagreement between sizes is visible; the radial MLP of
    # a case with fewer edges than the default chunk is read at all of its edges.
    per_case = Dict{String, Any}()
    for (cname, c) in cases
        chunk = min(default_chunk, c["n_edges"])
        per_case[cname] = Dict(nm => choose(row) for (nm, row) in matrix_rows(c, chunk))
    end
    node_mul = any(stages[nm]["method"] == "mul!" for (nm, _) in tc["node_linear"])
    return Dict(
        "decided_at" => target,
        "stages" => stages,
        "choices_per_case" => per_case,
        "node_layout" => node_mul ? "(C, n, K): each slot of the node features is one contiguous C × n·(2l+1) matrix for mul!" : "(C, K, n)",
        "fused" => sort([k for (k, v) in tc["fusion"] if v["fuse"]]),
        "not_fused" => sort([k for (k, v) in tc["fusion"] if !v["fuse"]]),
        "edge_chunk" => default_chunk,
    )
end

function main()
    from = get(ENV, "STAGE_FROM", "")
    out = if isempty(from)
        ck = load_mace_checkpoint(CK)
        model = MACEModel{T}(ck)
        dm = PureMLIP.device_model(model, backend; elements = [29])
        ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
        cases = Dict{String, Any}()
        for N in CASES
            cases["cu_bench($N)"] = profile_case(N, dm, model, ext)
            GC.gc(true)
        end
        stage_shares = JSON.parsefile(joinpath(@__DIR__, "results", "stage_profile_mac.json"))["cases"]
        Dict(
            "description" => "Stage profile of the MACE evaluation on a GPU (bench/gpu_stage_profile.jl): matrix-shaped stages with mul! against a KernelAbstractions kernel, per-edge stages against a copy kernel's bandwidth, fusion candidates, and the edge-chunk sweep with peak device memory. Times in microseconds, median [min, max] of $REPS synchronized runs.",
            "backend" => BACKEND_NAME, "device" => BACKEND_NAME == "Metal" ? String(VENDOR.device().name) : "",
            "precision" => "Float32", "model" => "MACE-MPA-0 (medium)", "julia" => string(VERSION),
            "edge_chunk_candidates" => collect(CHUNKS), "memory_budget_bytes" => BUDGET,
            "cases" => JSON.parse(JSON.json(cases)),
            "cpu_work_mix" => Dict(k => stage_shares[k]["shares"] for k in keys(stage_shares) if endswith(k, "Float32")),
            "cpu_work_mix_source" => "stage_profile_mac.json, Float32 cases (shares of the samples of mace_energy_forces! on one CPU thread)",
        )
    else
        # Rebuild the decision table from the measurements of an earlier run.
        JSON.parsefile(from)
    end
    out["decision"] = decide(out["cases"])
    mkpath(dirname(OUT))
    open(io -> JSON.print(io, out, 2), OUT, "w")
    println("decision: ", JSON.json(out["decision"], 2))
    println("wrote ", OUT)
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
