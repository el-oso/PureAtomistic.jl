# Mean-square displacement of atoms and the diffusion coefficient from its slope.
#
# Positions are unwrapped frame to frame: the displacement between consecutive frames is taken
# under the minimum image of the later frame's cell and added to the running unwrapped position,
# so atoms may be stored wrapped into the cell or not. This needs every atom to move less than
# half the shortest cell height between frames, which `unwrap!` checks.

"""
    unwrap!(unwrapped, previous, positions, cell) -> unwrapped

Advances the unwrapped positions by one frame: for each atom adds to `unwrapped` the minimum-image
displacement from `previous` to `positions` in `cell` (lattice vectors in columns), then copies
`positions` into `previous`. The three matrices are 3 × n with equal axes. Throws if a
displacement is at least half the shortest cell height long, since the image it belongs to is
then ambiguous. Allocates nothing.
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

"""
    msd(frames; species = nothing, max_lag, nblocks = 5, dt = nothing) -> (lag_fs, msd_A2)

Mean-square displacement `⟨|u(t₀ + τ) - u(t₀)|²⟩` of the atoms of `species` (`nothing`: all
atoms) over the unwrapped positions `u` (see [`unwrap!`](@ref)), for lags `τ = 0, 1, …, max_lag`
frames. The frames are split into `nblocks` consecutive blocks of `B = nframes ÷ nblocks` frames
(the last `nframes - nblocks·B` frames are not used), and column `b` of `msd_A2` averages over
the atoms and every time origin `t₀` of block `b` whose pair `(t₀, t₀ + τ)` lies inside the block,
so the columns use disjoint stretches of the trajectory. `lag_fs` holds `τ · Δt`.

`frames` is any collection with `eachindex` and `getindex` whose elements have `positions`, `cell`
and `species` (a vector of `System`s or `Frame`s, or a `TrajectoryReader`), with the same atoms
in every frame. The frame times come from each frame's `time` (fs), which must be evenly spaced,
or, when `dt` is given, are `dt` apart. Positions in Å give `msd_A2` in Å².
"""
function msd(frames; species = nothing, max_lag::Integer, nblocks::Integer = 5, dt = nothing)
    L = Int(max_lag)
    L >= 1 || throw(ArgumentError(lazy"pureatomistic: max_lag must be at least 1, got $max_lag"))
    nblocks >= 1 || throw(ArgumentError(lazy"pureatomistic: nblocks must be at least 1, got $nblocks"))
    isnothing(dt) || dt > 0 || throw(ArgumentError(lazy"pureatomistic: dt must be positive, got $dt"))
    idx = eachindex(frames)
    nf = length(idx)
    B = nf ÷ nblocks
    B > L ||
        throw(
        ArgumentError(
            lazy"pureatomistic: msd has $nf frames in $nblocks blocks of $B, too few for a lag of $L frames"
        )
    )
    f1 = frames[first(idx)]
    T = float(promote_type(eltype(f1.positions), eltype(f1.cell)))
    sp = f1.species
    _check_positions(f1.positions, sp)
    sel = [a for a in eachindex(sp) if _selected(sp, a, species)]
    isempty(sel) && throw(ArgumentError(lazy"pureatomistic: no atom has species $species"))
    ns = length(sel)
    u = similar(f1.positions, T)
    prev = similar(f1.positions, T)
    copyto!(u, f1.positions)
    copyto!(prev, f1.positions)
    i = first(axes(u, 1))
    ring = zeros(T, 3, ns, L + 1)
    acc = zeros(T, L + 1, nblocks)
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
        for j in eachindex(sel), r in 1:3
            ring[r, j, slot] = u[i + r - 1, sel[j]]
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
    return (lag_fs = T[τ * Δt for τ in 0:L], msd_A2 = acc)
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
`std(D_b) / √nblocks`, which treats the blocks as independent. Needs at least two blocks and two
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
