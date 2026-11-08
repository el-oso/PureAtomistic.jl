# Vacancy tracking on a fixed reference lattice.
#
# A frame is reduced to one vacancy site in four steps: average each atom's position over the
# last `window` frames (minimum-image displacements about the current frame, so wrapped
# positions average correctly), assign every atom to its nearest reference site, remove the
# mean displacement of the atoms from their sites and assign again, then require exactly one
# empty site and no doubly occupied site. A change of vacancy site counts as a hop once the
# new site has been the vacancy for `dwell` consecutive frames, which suppresses recrossings
# of the transition state.
#
# Cells are 3x3 with lattice vectors in columns; positions and sites are 3xN. The minimum
# image rounds fractional coordinates, which is exact for displacements shorter than half the
# cell's shortest height; an atom is always within that distance of its site.

# A 3x3 matrix as a column-major tuple, so small linear algebra stays allocation-free.
function _mat3(::Type{T}, cell::AbstractMatrix) where {T}
    axes(cell) == (Base.OneTo(3), Base.OneTo(3)) ||
        throw(ArgumentError(lazy"pureatomistic: cell must be 3x3 with 1-based axes, got axes $(axes(cell))"))
    return (
        T(cell[1, 1]), T(cell[2, 1]), T(cell[3, 1]),
        T(cell[1, 2]), T(cell[2, 2]), T(cell[3, 2]),
        T(cell[1, 3]), T(cell[2, 3]), T(cell[3, 3]),
    )
end

function _inv3(m::NTuple{9, T}) where {T}
    m1, m2, m3, m4, m5, m6, m7, m8, m9 = m
    det = m1 * (m5 * m9 - m6 * m8) - m4 * (m2 * m9 - m3 * m8) + m7 * (m2 * m6 - m3 * m5)
    (iszero(det) || !isfinite(det)) &&
        throw(ArgumentError("pureatomistic: cell is singular"))
    return (
        (m5 * m9 - m6 * m8) / det, (m3 * m8 - m2 * m9) / det, (m2 * m6 - m3 * m5) / det,
        (m6 * m7 - m4 * m9) / det, (m1 * m9 - m3 * m7) / det, (m3 * m4 - m1 * m6) / det,
        (m4 * m8 - m5 * m7) / det, (m2 * m7 - m1 * m8) / det, (m1 * m5 - m2 * m4) / det,
    )
end

@inline function _mul3(m::NTuple{9, T}, v::NTuple{3, T}) where {T}
    return (
        m[1] * v[1] + m[4] * v[2] + m[7] * v[3],
        m[2] * v[1] + m[5] * v[2] + m[8] * v[3],
        m[3] * v[1] + m[6] * v[2] + m[9] * v[3],
    )
end

# Minimum-image form of the displacement `d` in the cell `cm` with inverse `ci`.
@inline function _minimg(d::NTuple{3, T}, cm::NTuple{9, T}, ci::NTuple{9, T}) where {T}
    s = _mul3(ci, d)
    return _mul3(cm, (s[1] - round(s[1]), s[2] - round(s[2]), s[3] - round(s[3])))
end

@inline _sub3(a::NTuple{3, T}, b::NTuple{3, T}) where {T} = (a[1] - b[1], a[2] - b[2], a[3] - b[3])
@inline _add3(a::NTuple{3, T}, b::NTuple{3, T}) where {T} = (a[1] + b[1], a[2] + b[2], a[3] + b[3])
@inline _dot3(a::NTuple{3, T}, b::NTuple{3, T}) where {T} = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

@inline _col(::Type{T}, A::AbstractMatrix, j) where {T} = (T(A[1, j]), T(A[2, j]), T(A[3, j]))

function _check3(A::AbstractMatrix, name)
    axes(A, 1) == Base.OneTo(3) ||
        throw(ArgumentError(lazy"pureatomistic: $name must have 3 rows with 1-based axes, got axes $(axes(A, 1))"))
    return nothing
end

# Index of the site of `sites` nearest to `p - shift` under the minimum image.
function _nearest_site(p::NTuple{3, T}, shift::NTuple{3, T}, sites::AbstractMatrix, cm, ci) where {T}
    q = _sub3(p, shift)
    best = firstindex(sites, 2)
    bestd = typemax(T)
    for j in axes(sites, 2)
        d = _minimg(_sub3(q, _col(T, sites, j)), cm, ci)
        d2 = _dot3(d, d)
        if d2 < bestd
            bestd = d2
            best = j
        end
    end
    return best
end

# Counts the atoms of `positions` (shifted by `shift`) on each site into `occ`; when `owner` is
# not `nothing`, it receives each atom's site.
function _assign!(occ, owner, positions::AbstractMatrix, shift::NTuple{3, T}, sites, cm, ci) where {T}
    fill!(occ, 0)
    for a in axes(positions, 2)
        s = _nearest_site(_col(T, positions, a), shift, sites, cm, ci)
        owner === nothing || (owner[a] = s)
        occ[s] += 1
    end
    return nothing
end

"""
    reference_sites(cell, a, nrep)

Sites of the perfect fcc lattice with conventional lattice constant `a` (Å) in an
`nrep`×`nrep`×`nrep` cubic supercell, as a `3 × 4nrep³` matrix. `cell` must be the supercell
`nrep * a` times the identity; the check ties the sites to the cell they are tracked in.
"""
function reference_sites(cell::AbstractMatrix, a::Real, nrep::Integer)
    T = float(eltype(cell))
    nrep >= 1 || throw(ArgumentError(lazy"pureatomistic: nrep must be at least 1, got $nrep"))
    cm = _mat3(T, cell)
    L = T(a) * T(nrep)
    tol = sqrt(eps(T)) * L
    for k in 1:9
        want = k in (1, 5, 9) ? L : zero(T)
        abs(cm[k] - want) <= tol ||
            throw(ArgumentError(lazy"pureatomistic: reference_sites needs the cubic cell nrep*a = $L; cell entry $k is $(cm[k])"))
    end
    basis = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
    sites = Matrix{T}(undef, 3, 4 * nrep^3)
    n = 0
    for i in 0:(nrep - 1), j in 0:(nrep - 1), k in 0:(nrep - 1), b in basis
        n += 1
        sites[1, n] = T(a) * (i + T(b[1]) / 2)
        sites[2, n] = T(a) * (j + T(b[2]) / 2)
        sites[3, n] = T(a) * (k + T(b[3]) / 2)
    end
    return sites
end

"""
    wigner_seitz_occupancy!(occ, positions, sites, cell)

Writes into `occ` (indexed like the columns of `sites`) the number of atoms of `positions`
whose nearest site, under the minimum image in `cell`, is that site. Returns `occ`.
"""
function wigner_seitz_occupancy!(
        occ::AbstractVector{<:Integer}, positions::AbstractMatrix, sites::AbstractMatrix, cell::AbstractMatrix
    )
    _check3(positions, "positions")
    _check3(sites, "sites")
    axes(occ, 1) == axes(sites, 2) ||
        throw(DimensionMismatch(lazy"pureatomistic: occ axes $(axes(occ, 1)) do not match the sites $(axes(sites, 2))"))
    isempty(sites) && throw(ArgumentError("pureatomistic: no reference sites"))
    T = float(promote_type(eltype(positions), eltype(sites), eltype(cell)))
    cm = _mat3(T, cell)
    ci = _inv3(cm)
    _assign!(occ, nothing, positions, (zero(T), zero(T), zero(T)), sites, cm, ci)
    return occ
end

"""
    VacancyTracker(sites; cell, window, dwell, capacity = 10_000, max_transient = 25)

Tracks the single vacancy of a lattice with the reference `sites` (3×M, M-1 atoms) in the fixed
`cell`. [`update!`](@ref) feeds frames; `window` is the number of frames over which positions
are averaged before the assignment, `dwell` the number of consecutive frames a new vacancy
site must persist before a hop counts, and `capacity` the number of frames the tracker can
hold. After an error from `update!` the tracker must not be used again.

While an atom crosses the saddle of a jump, or during a concerted move of two atoms, the
Wigner-Seitz assignment can be ambiguous for a few frames: it then holds two empty sites and a
doubly occupied one, or another occupancy that is not one vacancy. `max_transient` is the number
of consecutive frames of such an occupancy that the tracker accepts (default 25, which is 500 fs
at frames 20 fs apart). It skips them: their positions do not enter the window average of later
frames, they are not stored, and the vacancy state does not change. The frame after them that
holds one vacancy continues the track. The next frame beyond the limit throws, since an
occupancy that lasts that long is a defect (a Frenkel pair, melting) rather than a jump in
progress. The first frame must hold one vacancy.

Read-only state: `nframes`, `nhops`, the frame times `t`, the unwrapped vacancy positions `R`
(3×capacity; columns `1:nframes` are valid, the first at the initial vacancy site), the
vacancy site `vsite` of each frame and the confirmation times `hop_t` of the hops; the number of
skipped frames `ntransient`, the longest run of consecutive skipped frames `longest_transient`,
the number of such runs `nepisodes` and the time `transient_t` of the first frame of each run
(entries `1:nepisodes` are valid).
"""
mutable struct VacancyTracker{T <: AbstractFloat}
    sites::Matrix{T}
    cm::NTuple{9, T}
    ci::NTuple{9, T}
    window::Int
    dwell::Int
    hist::Array{T, 3}
    avg::Matrix{T}
    owner::Vector{Int}
    occ::Vector{Int}
    nframes::Int
    nhops::Int
    site::Int
    cand::Int
    candcount::Int
    t::Vector{T}
    R::Matrix{T}
    vsite::Vector{Int}
    hop_t::Vector{T}
    max_transient::Int
    ntransient::Int
    transient_run::Int
    longest_transient::Int
    nepisodes::Int
    transient_t::Vector{T}
    tlast::T

    function VacancyTracker{T}(
            sites::AbstractMatrix; cell::AbstractMatrix, window::Integer, dwell::Integer, capacity::Integer = 10_000,
            max_transient::Integer = 25,
        ) where {T}
        _check3(sites, "sites")
        nsites = size(sites, 2)
        nsites >= 2 || throw(ArgumentError("pureatomistic: a vacancy tracker needs at least 2 sites"))
        window >= 1 || throw(ArgumentError(lazy"pureatomistic: window must be at least 1, got $window"))
        dwell >= 1 || throw(ArgumentError(lazy"pureatomistic: dwell must be at least 1, got $dwell"))
        capacity >= 1 || throw(ArgumentError(lazy"pureatomistic: capacity must be at least 1, got $capacity"))
        max_transient >= 0 ||
            throw(ArgumentError(lazy"pureatomistic: max_transient must be at least 0, got $max_transient"))
        cm = _mat3(T, cell)
        ci = _inv3(cm)
        n = nsites - 1
        return new{T}(
            Matrix{T}(sites), cm, ci, window, dwell,
            zeros(T, 3, n, window), zeros(T, 3, n), zeros(Int, n), zeros(Int, nsites),
            0, 0, 0, 0, 0,
            zeros(T, capacity), zeros(T, 3, capacity), zeros(Int, capacity), zeros(T, capacity),
            max_transient, 0, 0, 0, 0, zeros(T, capacity), T(-Inf),
        )
    end
end

function VacancyTracker(sites::AbstractMatrix; kwargs...)
    return VacancyTracker{float(eltype(sites))}(sites; kwargs...)
end

# The number of empty and of multiply occupied sites of `occ`, and the last empty site.
function _occupancy(occ::AbstractVector{Int})
    nempty = 0
    nmulti = 0
    empty = 0
    for s in eachindex(occ)
        if occ[s] == 0
            nempty += 1
            empty = s
        elseif occ[s] > 1
            nmulti += 1
        end
    end
    return nempty, nmulti, empty
end

"""
    update!(tracker, positions, t)

Adds the frame `positions` (3×N, N = number of sites - 1) taken at time `t`, which must exceed
the time of the previous frame, skipped or not. A frame whose occupancy is not exactly one empty
site and no doubly occupied site is skipped while it is one of at most `max_transient`
consecutive such frames (see [`VacancyTracker`](@ref)); the next one throws `ArgumentError`,
naming the frame (counting skipped frames) and the time since the last frame with one vacancy.
"""
function update!(tr::VacancyTracker{T}, positions::AbstractMatrix, t::Real) where {T}
    _check3(positions, "positions")
    n = length(tr.owner)
    size(positions, 2) == n ||
        throw(ArgumentError(lazy"pureatomistic: tracker has $n atoms, frame has $(size(positions, 2))"))
    f = tr.nframes + 1
    f <= length(tr.t) ||
        throw(ArgumentError(lazy"pureatomistic: tracker capacity $(length(tr.t)) is exhausted"))
    T(t) > tr.tlast ||
        throw(ArgumentError(lazy"pureatomistic: frame times must increase, got $t after $(tr.tlast) at frame $(f + tr.ntransient)"))

    slot = mod1(f, tr.window)
    j0 = firstindex(positions, 2)
    for a in 1:n
        c = _col(T, positions, j0 + a - 1)
        tr.hist[1, a, slot] = c[1]
        tr.hist[2, a, slot] = c[2]
        tr.hist[3, a, slot] = c[3]
    end

    cm, ci = tr.cm, tr.ci
    nw = min(f, tr.window)
    for a in 1:n
        ref = (tr.hist[1, a, slot], tr.hist[2, a, slot], tr.hist[3, a, slot])
        acc = (zero(T), zero(T), zero(T))
        for k in 1:nw
            h = (tr.hist[1, a, k], tr.hist[2, a, k], tr.hist[3, a, k])
            acc = _add3(acc, _minimg(_sub3(h, ref), cm, ci))
        end
        avg = _add3(ref, (acc[1] / nw, acc[2] / nw, acc[3] / nw))
        tr.avg[1, a] = avg[1]
        tr.avg[2, a] = avg[2]
        tr.avg[3, a] = avg[3]
    end

    # The first assignment finds each atom's site; the mean displacement from those sites is the
    # rigid drift of the sample, removed before the assignment that is kept.
    zero3 = (zero(T), zero(T), zero(T))
    _assign!(tr.occ, tr.owner, tr.avg, zero3, tr.sites, cm, ci)
    drift = zero3
    for a in 1:n
        d = _minimg(_sub3(_col(T, tr.avg, a), _col(T, tr.sites, tr.owner[a])), cm, ci)
        drift = _add3(drift, d)
    end
    drift = (drift[1] / n, drift[2] / n, drift[3] / n)
    _assign!(tr.occ, tr.owner, tr.avg, drift, tr.sites, cm, ci)
    nempty, nmulti, v = _occupancy(tr.occ)
    tr.tlast = T(t)
    if nempty != 1 || nmulti != 0
        _transient!(tr, f, T(t), nempty, nmulti)
        return nothing
    end
    tr.transient_run = 0

    if f == 1
        tr.site = v
        r = _col(T, tr.sites, v)
    else
        r = _col(T, tr.R, f - 1)
        if v == tr.site
            tr.cand = 0
            tr.candcount = 0
        else
            if v == tr.cand
                tr.candcount += 1
            else
                tr.cand = v
                tr.candcount = 1
            end
            if tr.candcount >= tr.dwell
                r = _add3(r, _minimg(_sub3(_col(T, tr.sites, v), _col(T, tr.sites, tr.site)), cm, ci))
                tr.site = v
                tr.cand = 0
                tr.candcount = 0
                tr.nhops += 1
                tr.hop_t[tr.nhops] = T(t)
            end
        end
    end
    tr.t[f] = T(t)
    tr.R[1, f] = r[1]
    tr.R[2, f] = r[2]
    tr.R[3, f] = r[3]
    tr.vsite[f] = tr.site
    tr.nframes = f
    return nothing
end

# Counts the frame `f` (the next frame to store) at time `t` as skipped, or throws when it is the
# first frame or one more than `max_transient` consecutive skipped frames. The window slot that the
# frame's positions went into is the one the next stored frame overwrites, so they never enter a
# later average.
function _transient!(tr::VacancyTracker{T}, f::Int, t::T, nempty::Int, nmulti::Int) where {T}
    frame = f + tr.ntransient
    f == 1 &&
        throw(
        ArgumentError(
            lazy"pureatomistic: occupancy is not one vacancy at frame $frame: $nempty empty and $nmulti multiply occupied sites; the first frame must hold one vacancy"
        )
    )
    run = tr.transient_run + 1
    run <= tr.max_transient ||
        throw(
        ArgumentError(
            lazy"pureatomistic: occupancy is not one vacancy at frame $frame: $nempty empty and $nmulti multiply occupied sites, for $run consecutive frames over $(t - tr.t[f - 1]) time units since the last frame with one vacancy (limit $(tr.max_transient) frames)"
        )
    )
    if run == 1
        tr.nepisodes += 1
        tr.transient_t[tr.nepisodes] = t
    end
    tr.transient_run = run
    tr.ntransient += 1
    tr.longest_transient = max(tr.longest_transient, run)
    return nothing
end
