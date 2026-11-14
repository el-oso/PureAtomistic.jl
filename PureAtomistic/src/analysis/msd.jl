# Mean-square displacement of atoms and the diffusion coefficient from its slope.
#
# Positions are unwrapped frame to frame: the displacement between consecutive frames is taken
# under the minimum image of the later frame's cell and added to the running unwrapped position,
# so atoms may be stored wrapped into the cell or not. The result is right only if every atom
# moves less than half the shortest cell height between frames; choosing a frame interval short
# enough for that is the caller's responsibility. `unwrap!` throws for a minimum-image
# displacement in the ambiguous range, but a longer true move maps onto a shorter image and
# passes unnoticed (in a cubic cell of side L, a move of 0.7 L along an axis reads as -0.3 L).

"""
    unwrap!(unwrapped, previous, positions, cell) -> unwrapped

Advances the unwrapped positions by one frame: for each atom adds to `unwrapped` the minimum-image
displacement from `previous` to `positions` in `cell` (lattice vectors in columns), then copies
`positions` into `previous`. The three matrices are 3 × n with equal axes. Every atom must have
moved less than half the shortest cell height since `previous`, which the caller ensures by the
frame interval: a longer move maps onto a shorter image without notice. Throws if a minimum-image
displacement is itself at least half the shortest cell height long, since its image is then
ambiguous. Allocates nothing.
"""
function unwrap!(unwrapped::AbstractMatrix, previous::AbstractMatrix, positions::AbstractMatrix, cell::AbstractMatrix)
    size(positions, 1) == 3 || throw(ArgumentError(lazy"pureatomistic: positions need 3 rows, got $(size(positions, 1))"))
    (axes(unwrapped) == axes(positions) && axes(previous) == axes(positions)) ||
        throw(
        DimensionMismatch(
            lazy"pureatomistic: unwrapped $(axes(unwrapped)), previous $(axes(previous)) and positions $(axes(positions)) must have equal axes"
        )
    )
    T = float(promote_type(eltype(unwrapped), eltype(positions), eltype(cell)))
    cm = _cell9(T, cell)
    ci = _inv3(cm)
    hmin = 1 / maximum(_inv_heights(ci))
    i = first(axes(positions, 1))
    for a in axes(positions, 2)
        x = _column(T, positions, i, a)
        d = _minimg(_sub3(x, _column(T, previous, i, a)), cm, ci)
        4 * _dot3(d, d) < hmin^2 ||
            throw(
            ArgumentError(
                lazy"pureatomistic: atom $a moved $(sqrt(_dot3(d, d))) Å between frames, at least half the shortest cell height $hmin Å"
            )
        )
        for c in 0:2
            unwrapped[i + c, a] += d[c + 1]
            previous[i + c, a] = x[c + 1]
        end
    end
    return unwrapped
end

_frame_time(f, c, dt::Real) = (c - 1) * dt
function _frame_time(f, c, ::Nothing)
    hasproperty(f, :time) ||
        throw(ArgumentError("pureatomistic: these frames carry no time; pass the frame interval as dt (fs)"))
    return f.time
end

# The lag, block and frame-interval checks shared by `msd` and `msd!`.
function _check_msd_blocks(L::Int, nblocks::Integer, dt, nf::Int)
    L >= 1 || throw(ArgumentError(lazy"pureatomistic: max_lag must be at least 1, got $L"))
    nblocks >= 1 || throw(ArgumentError(lazy"pureatomistic: nblocks must be at least 1, got $nblocks"))
    isnothing(dt) || dt > 0 || throw(ArgumentError(lazy"pureatomistic: dt must be positive, got $dt"))
    B = nf ÷ nblocks
    B > L ||
        throw(
        ArgumentError(
            lazy"pureatomistic: msd has $nf frames in $nblocks blocks of $B, too few for a lag of $L frames"
        )
    )
    return nothing
end

"""
    msd(frames; species = nothing, max_lag, nblocks = 5, dt = nothing) -> (lag_fs, msd_A2)

Mean-square displacement of the atoms of `species` over `frames` for lags `0, 1, …, max_lag`
frames in `nblocks` blocks, as a new vector of lags (fs) and a new `(max_lag + 1) × nblocks`
matrix (Å²); see [`msd!`](@ref).
"""
function msd(frames; species = nothing, max_lag::Integer, nblocks::Integer = 5, dt = nothing)
    L = Int(max_lag)
    _check_msd_blocks(L, nblocks, dt, length(eachindex(frames)))
    f1 = frames[first(eachindex(frames))]
    T = float(promote_type(eltype(f1.positions), eltype(f1.cell)))
    ns = _count_species(f1.species, species)
    return msd!(
        Vector{T}(undef, L + 1), Matrix{T}(undef, L + 1, nblocks), similar(f1.positions, T),
        similar(f1.positions, T), Array{T}(undef, 3, ns, L + 1), frames; species, dt
    )
end

"""
    msd!(lag_fs, msd_A2, unwrapped, previous, ring, frames; species = nothing, dt = nothing)
        -> (lag_fs, msd_A2)

Mean-square displacement `⟨|u(t₀ + τ) - u(t₀)|²⟩` of the atoms of `species` (`nothing`: all
atoms) over the unwrapped positions `u` (see [`unwrap!`](@ref)), for lags `τ = 0, 1, …, L`
frames with `L = length(lag_fs) - 1 ≥ 1`. The frames are split into `nblocks = size(msd_A2, 2)`
consecutive blocks of `B = nframes ÷ nblocks` frames (the last `nframes - nblocks·B` frames are
not used), and column `b` of `msd_A2` (`(L + 1) × nblocks`) averages over the atoms and every
time origin `t₀` of block `b` whose pair `(t₀, t₀ + τ)` lies inside the block, so the columns use
disjoint stretches of the trajectory. `lag_fs` receives `τ · Δt`. `lag_fs` and `msd_A2` are
1-based. The scratch `unwrapped` and `previous` have the axes of the positions, and `ring` is
`3 × nₛ × (L + 1)` with `nₛ` the number of selected atoms.

`frames` is any collection with `eachindex` and `getindex` whose elements have `positions`, `cell`
and `species` (a vector of `System`s or `Frame`s, or a `TrajectoryReader`), with the same
atoms and species in every frame (checked). The frame times come from each frame's `time` (fs),
which must be evenly spaced, or, when `dt` is given, are `dt` apart. The frames must be close
enough that no atom moves half the shortest cell height between two of them (see
[`unwrap!`](@ref)); the caller is responsible for that. Positions in Å give `msd_A2` in Å².
Allocates nothing for frames that are already in memory.
"""
function msd!(
        lag_fs::AbstractVector, msd_A2::AbstractMatrix, unwrapped::AbstractMatrix, previous::AbstractMatrix,
        ring::AbstractArray{<:Any, 3}, frames; species = nothing, dt = nothing
    )
    Base.require_one_based_indexing(lag_fs, msd_A2, ring)
    L = length(lag_fs) - 1
    nblocks = size(msd_A2, 2)
    idx = eachindex(frames)
    _check_msd_blocks(L, nblocks, dt, length(idx))
    size(msd_A2, 1) == L + 1 ||
        throw(DimensionMismatch(lazy"pureatomistic: msd_A2 has $(size(msd_A2, 1)) rows for $(L + 1) lags"))
    B = length(idx) ÷ nblocks
    f1 = frames[first(idx)]
    T = float(promote_type(eltype(f1.positions), eltype(f1.cell)))
    sp = f1.species
    _check_positions(f1.positions, sp)
    ns = _count_species(sp, species)
    iszero(ns) && throw(ArgumentError(lazy"pureatomistic: no atom has species $species"))
    size(ring) == (3, ns, L + 1) ||
        throw(DimensionMismatch(lazy"pureatomistic: ring has size $(size(ring)), expected $((3, ns, L + 1))"))
    u, prev = unwrapped, previous
    (axes(u) == axes(f1.positions) && axes(prev) == axes(f1.positions)) ||
        throw(
        DimensionMismatch(
            lazy"pureatomistic: unwrapped $(axes(u)) and previous $(axes(prev)) must have the axes $(axes(f1.positions)) of the positions"
        )
    )
    copyto!(u, f1.positions)
    copyto!(prev, f1.positions)
    i = first(axes(u, 1))
    acc = msd_A2
    fill!(acc, zero(eltype(acc)))
    t_prev = zero(T)
    Δt = zero(T)
    used = nblocks * B
    c = 0
    for k in idx
        c += 1
        c > used && break
        f = frames[k]
        t = T(_frame_time(f, c, dt))
        if c > 1
            axes(f.positions) == axes(u) ||
                throw(DimensionMismatch(lazy"pureatomistic: frame $k has positions $(axes(f.positions)), frame 1 $(axes(u))"))
            f.species == sp || throw(ArgumentError(lazy"pureatomistic: frame $k has other species than frame 1"))
            step = t - t_prev
            if c == 2
                step > 0 || throw(ArgumentError(lazy"pureatomistic: frame times must increase, got $t after $t_prev"))
                Δt = step
            else
                abs(step - Δt) <= 8 * eps(T) * max(abs(t), Δt) ||
                    throw(ArgumentError(lazy"pureatomistic: frame times are not evenly spaced: $t after $t_prev, interval $Δt"))
            end
            unwrap!(u, prev, f.positions, f.cell)
        end
        t_prev = t
        slot = mod1(c, L + 1)
        j = 0
        for a in eachindex(sp)
            _selected(sp, a, species) || continue
            j += 1
            for r in 1:3
                ring[r, j, slot] = u[i + r - 1, a]
            end
        end
        p = (c - 1) % B
        b = (c - 1) ÷ B + 1
        for τ in 1:min(L, p)
            _accumulate_lag!(acc, ring, slot, mod1(c - τ, L + 1), τ + 1, b)
        end
    end
    for b in 1:nblocks, τ in 0:L
        acc[τ + 1, b] /= T(ns) * T(B - τ)
    end
    for τ in 0:L
        lag_fs[τ + 1] = τ * Δt
    end
    return (lag_fs = lag_fs, msd_A2 = msd_A2)
end

# Adds the summed squared displacement between the ring slots `s` and `s0` to `acc[row, b]`.
function _accumulate_lag!(acc, ring, s, s0, row, b)
    total = zero(eltype(acc))
    for j in axes(ring, 2)
        dx = ring[1, j, s] - ring[1, j, s0]
        dy = ring[2, j, s] - ring[2, j, s0]
        dz = ring[3, j, s] - ring[3, j, s0]
        total += dx * dx + dy * dy + dz * dz
    end
    acc[row, b] += total
    return nothing
end

"""
    diffusion_coefficient(lag, msd; fit_window) -> (D, σ_D)

Diffusion coefficient from the Einstein relation in three dimensions, `⟨|Δr|²⟩ = 6 D τ` at long
lags. `lag` holds the lags and `msd` one column per block, as returned by [`msd`](@ref). For
each block, the least-squares line with intercept through the points with
`fit_window[1] ≤ lag ≤ fit_window[2]` gives `D_b` as its slope over 6; the intercept absorbs the
vibrational plateau. `D` is the mean of the `D_b` and `σ_D` its standard error,
`std(D_b) / √nblocks`, which treats the blocks as independent; `σ_D` is itself uncertain by a
fraction of about `1/√(2(nblocks - 1))` (29 % for 7 blocks). Needs at least two blocks and two
lags in the window. Units are those of `msd` per unit of `lag` (Å²/fs from `msd`; 1 Å²/fs is
0.1 cm²/s).
"""
function diffusion_coefficient(lag::AbstractVector, msd::AbstractMatrix; fit_window)
    axes(msd, 1) == axes(lag, 1) ||
        throw(DimensionMismatch(lazy"pureatomistic: msd rows $(axes(msd, 1)) differ from the lags $(axes(lag, 1))"))
    nb = size(msd, 2)
    nb >= 2 || throw(ArgumentError(lazy"pureatomistic: diffusion_coefficient needs at least 2 blocks for an error, got $nb"))
    T = float(promote_type(eltype(lag), eltype(msd)))
    lo, hi = T(fit_window[1]), T(fit_window[2])
    m = 0
    sx = zero(T)
    for k in eachindex(lag)
        if lo <= lag[k] <= hi
            m += 1
            sx += lag[k]
        end
    end
    m >= 2 || throw(ArgumentError(lazy"pureatomistic: the fit window ($lo, $hi) holds $m lags; it needs at least 2"))
    x̄ = sx / m
    sxx = zero(T)
    for k in eachindex(lag)
        lo <= lag[k] <= hi && (sxx += (lag[k] - x̄)^2)
    end
    mean = zero(T)
    m2 = zero(T)
    n = 0
    for b in axes(msd, 2)
        sxy = zero(T)
        for k in eachindex(lag)
            lo <= lag[k] <= hi && (sxy += (lag[k] - x̄) * msd[k, b])
        end
        x = sxy / sxx / 6
        # Welford update of the mean and sum of squared deviations of the block estimates.
        n += 1
        delta = x - mean
        mean += delta / n
        m2 += delta * (x - mean)
    end
    return (D = mean, σ_D = sqrt(m2 / (n - 1) / n))
end
