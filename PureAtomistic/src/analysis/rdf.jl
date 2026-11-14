# Pair distribution function under periodic boundaries.
#
# Frames are read through a small protocol: `frames` is any collection with `eachindex` and
# `getindex` (a vector of `System`s or of `IO.Frame`s, or a `TrajectoryReader`), and each frame
# has the properties `positions` (3 × n), `cell` (3 × 3, lattice vectors in columns) and `species`.
# Positions, cells and species are read through their own axes, so views and `OffsetArray`s work.
#
# Minimum image: with `b_k` the rows of the inverse cell, a displacement `d` has the fractional
# coordinates `s_k = b_k · d`, and the cell height across face `k` is `h_k = 1 / |b_k|`. If
# `|d| < h_min / 2` then `|s_k| ≤ |d| / h_k < 1/2` for every `k`, so rounding the fractional
# coordinates of any image of `d` recovers `d` itself; and two images differ by a lattice vector,
# which is at least `h_min` long, so at most one image lies within `h_min / 2`. Pairs closer than
# `r_max ≤ h_min / 2` are therefore counted exactly once with their true distance. Beyond that
# every image within `r_max` is visited, including an atom's own images.

# The nine entries of a 3 × 3 matrix, column-major, read through its own axes.
function _cell9(::Type{T}, cell::AbstractMatrix) where {T}
    size(cell) == (3, 3) || throw(ArgumentError(lazy"pureatomistic: the cell must be 3 × 3, got $(size(cell))"))
    i, j = first(axes(cell, 1)), first(axes(cell, 2))
    return (
        T(cell[i, j]), T(cell[i + 1, j]), T(cell[i + 2, j]),
        T(cell[i, j + 1]), T(cell[i + 1, j + 1]), T(cell[i + 2, j + 1]),
        T(cell[i, j + 2]), T(cell[i + 1, j + 2]), T(cell[i + 2, j + 2]),
    )
end

# Column `a` of a 3 × n matrix whose first row index is `i`.
@inline _column(::Type{T}, P::AbstractMatrix, i, a) where {T} = (T(P[i, a]), T(P[i + 1, a]), T(P[i + 2, a]))

# `1 / h_k` for the three faces: the norms of the rows of the inverse cell `ci`.
@inline _inv_heights(ci::NTuple{9, T}) where {T} = (
    sqrt(ci[1]^2 + ci[4]^2 + ci[7]^2), sqrt(ci[2]^2 + ci[5]^2 + ci[8]^2), sqrt(ci[3]^2 + ci[6]^2 + ci[9]^2),
)

# Whether a pair search to `r_max` may use the minimum image (see the top of this file).
@inline _minimum_image_ok(r_max, ib::NTuple{3}) = 2 * r_max * maximum(ib) <= 1

# Whether atom `a` has species `s`; `nothing` selects every atom.
_selected(species, a, s) = isnothing(s) || species[a] == s

function _check_positions(P::AbstractMatrix, species::AbstractVector)
    size(P, 1) == 3 || throw(ArgumentError(lazy"pureatomistic: positions need 3 rows, got $(size(P, 1))"))
    axes(species, 1) == axes(P, 2) ||
        throw(DimensionMismatch(lazy"pureatomistic: species axes $(axes(species, 1)) differ from the atom axis $(axes(P, 2))"))
    return nothing
end

# Calls `f(a, b, r²)` for every ordered pair of an atom `a` of species `sa` and an atom `b` of
# species `sb` (`nothing` selects every atom) closer than `r_max`, once per periodic image of `b`;
# an atom is never paired with itself at zero lattice translation.
@inline function _foreach_pair(f::F, P::AbstractMatrix, cell, species, sa, sb, r_max::T) where {F, T}
    cm = _cell9(T, cell)
    ci = _inv3(cm)
    ib = _inv_heights(ci)
    r2max = r_max * r_max
    i = first(axes(P, 1))
    if _minimum_image_ok(r_max, ib)
        for a in axes(P, 2)
            _selected(species, a, sa) || continue
            pa = _column(T, P, i, a)
            for b in axes(P, 2)
                (b == a || !_selected(species, b, sb)) && continue
                d = _minimg(_sub3(_column(T, P, i, b), pa), cm, ci)
                r2 = _dot3(d, d)
                r2 < r2max && f(a, b, r2)
            end
        end
    else
        for a in axes(P, 2)
            _selected(species, a, sa) || continue
            pa = _column(T, P, i, a)
            for b in axes(P, 2)
                _selected(species, b, sb) || continue
                s = _mul3(ci, _sub3(_column(T, P, i, b), pa))
                s = (s[1] - round(s[1]), s[2] - round(s[2]), s[3] - round(s[3]))
                # An image within `r_max` has `|s_k + n_k| ≤ r_max / h_k` for each `k`. Rounding
                # of these bounds can drop only an image within a few eps of `r_max`, which
                # would fall in the last bin.
                lo1, hi1 = ceil(Int, -r_max * ib[1] - s[1]), floor(Int, r_max * ib[1] - s[1])
                lo2, hi2 = ceil(Int, -r_max * ib[2] - s[2]), floor(Int, r_max * ib[2] - s[2])
                lo3, hi3 = ceil(Int, -r_max * ib[3] - s[3]), floor(Int, r_max * ib[3] - s[3])
                for n1 in lo1:hi1, n2 in lo2:hi2, n3 in lo3:hi3
                    (b == a && iszero(n1) && iszero(n2) && iszero(n3)) && continue
                    d = _mul3(cm, (s[1] + n1, s[2] + n2, s[3] + n3))
                    r2 = _dot3(d, d)
                    r2 < r2max && f(a, b, r2)
                end
            end
        end
    end
    return nothing
end

_pair_species(::Nothing) = (nothing, nothing)
_pair_species(p::Tuple{Any, Any}) = p
_pair_species(p) = throw(ArgumentError(lazy"pureatomistic: species_pair must be nothing or a 2-tuple, got $(typeof(p))"))

"""
    rdf_histogram!(hist, positions, cell, species, r_max; species_pair = nothing) -> hist

Adds to `hist` the ordered pairs of one frame: for every atom `a` of the first species of
`species_pair` and every periodic image of every atom `b` of the second species (`nothing`
selects all atoms), except `b = a` at zero translation, a pair at distance `r < r_max` adds one
to bin `1 + floor(r / Δ)` with `Δ = r_max / length(hist)`, the bins counted from
`firstindex(hist)`. The search uses the minimum image when `r_max` is at most half the shortest
cell height and visits every image within `r_max` otherwise. `positions` is 3 × n, `cell` 3 × 3
with the lattice vectors in columns, and `species` is indexed like the atoms. Allocates nothing.
"""
function rdf_histogram!(
        hist::AbstractVector{<:Integer}, positions::AbstractMatrix, cell::AbstractMatrix, species::AbstractVector,
        r_max::Real; species_pair = nothing
    )
    _check_positions(positions, species)
    isempty(hist) && throw(ArgumentError("pureatomistic: rdf needs at least one bin"))
    T = float(promote_type(eltype(positions), eltype(cell)))
    rm = T(r_max)
    (rm > 0 && isfinite(rm)) || throw(ArgumentError(lazy"pureatomistic: r_max must be positive and finite, got $r_max"))
    sa, sb = _pair_species(species_pair)
    nb = length(hist)
    Δ = rm / nb
    h0 = firstindex(hist)
    _foreach_pair(positions, cell, species, sa, sb, rm) do _, _, r2
        k = min(nb, floor(Int, sqrt(r2) / Δ) + 1)
        hist[h0 + k - 1] += 1
    end
    return hist
end

# The number of atoms of species `s` (`nothing`: all atoms).
function _count_species(species, s)
    n = 0
    for a in eachindex(species)
        n += _selected(species, a, s)
    end
    return n
end

# The number of atoms selected by both `sa` and `sb`, which the minimum image never pairs with
# themselves.
function _count_both(species, sa, sb)
    n = 0
    for a in eachindex(species)
        n += _selected(species, a, sa) && _selected(species, a, sb)
    end
    return n
end

"""
    rdf(frames, r_max, nbins; species_pair = nothing) -> (r, g, n)

Radial distribution function of the pairs `species_pair` over `frames` in `nbins` bins up to
`r_max` (Å), as new vectors over the bins; see [`rdf!`](@ref).
"""
function rdf(frames, r_max::Real, nbins::Integer; species_pair = nothing)
    nbins >= 1 || throw(ArgumentError(lazy"pureatomistic: rdf needs at least one bin, got $nbins"))
    isempty(eachindex(frames)) && throw(ArgumentError("pureatomistic: rdf needs at least one frame"))
    f1 = frames[first(eachindex(frames))]
    T = float(promote_type(eltype(f1.positions), eltype(f1.cell)))
    return rdf!(
        Vector{T}(undef, nbins), Vector{T}(undef, nbins), Vector{T}(undef, nbins), zeros(Int, nbins),
        frames, r_max; species_pair
    )
end

"""
    rdf!(r, g, n, hist, frames, r_max; species_pair = nothing) -> (r, g, n)

Radial distribution function of the pairs `(A, B) = species_pair` (`nothing`: all atoms, either
element may be `nothing`) over `frames`, in `nbins = length(hist)` bins of width
`Δ = r_max / nbins` up to `r_max` (Å), written into `r`, `g` and `n`; `hist` is the integer
scratch of the pair counts. The four vectors share their axes. `r` holds the bin centers. With
`C_k` the ordered-pair count of bin `k` summed over the frames (see [`rdf_histogram!`](@ref),
which also states when the minimum image is used),

    g_k = C_k / (Σ_frames P / V · 4π/3 (r_hi³ - r_lo³)),

the count divided by the count an ideal gas of the same composition would give, with `P` the
number of ordered pairs the search can form in a frame. With the minimum image (`r_max` at most
half the shortest cell height) an atom never pairs with itself, so `P = N_A N_B - N_AB`, where
`N_AB` is the number of atoms selected by both `A` and `B` (`N_A` for one species twice, 0 for
two distinct species): an ideal gas gives `g = 1` on average for every selection. Beyond that
range every periodic image is counted, an atom's own images included, as in the infinite periodic
system, so `P = N_A N_B`: the images of the `B` atoms fill space at the density `N_B / V`, `g`
tends to 1 at large `r` in any cell, and a species of one atom works.

`n` is the running coordination number, the integral `∫₀^{r_hi} ρ g 4πr² dr` in histogram form:
`n_k = Σ_{j ≤ k} C_j / Σ_frames N_A`, the mean number of `B` neighbors of an `A` atom closer
than the upper edge of bin `k`.

`frames` is any collection with `eachindex` and `getindex` whose elements have `positions`,
`cell` and `species` (a vector of `System`s or `Frame`s, or a `TrajectoryReader`). Views and
`OffsetArray`s are accepted. Allocates nothing for frames that are already in memory.
"""
function rdf!(
        r::AbstractVector, g::AbstractVector, n::AbstractVector, hist::AbstractVector{<:Integer}, frames, r_max::Real;
        species_pair = nothing
    )
    (axes(r) == axes(hist) && axes(g) == axes(hist) && axes(n) == axes(hist)) ||
        throw(DimensionMismatch(lazy"pureatomistic: r $(axes(r)), g $(axes(g)), n $(axes(n)) and hist $(axes(hist)) must have equal axes"))
    nbins = length(hist)
    nbins >= 1 || throw(ArgumentError(lazy"pureatomistic: rdf needs at least one bin, got $nbins"))
    isempty(eachindex(frames)) && throw(ArgumentError("pureatomistic: rdf needs at least one frame"))
    f1 = frames[first(eachindex(frames))]
    T = float(promote_type(eltype(f1.positions), eltype(f1.cell)))
    sa, sb = _pair_species(species_pair)
    rm = T(r_max)
    fill!(hist, 0)
    weight = zero(T)
    na_total = 0
    for k in eachindex(frames)
        f = frames[k]
        na = _count_species(f.species, sa)
        nb = _count_species(f.species, sb)
        nab = _count_both(f.species, sa, sb)
        mi = _minimum_image_ok(rm, _inv_heights(_inv3(_cell9(T, f.cell))))
        pairs = mi ? T(na) * T(nb) - T(nab) : T(na) * T(nb)
        pairs >= 1 ||
            throw(
            ArgumentError(
                lazy"pureatomistic: rdf finds no pairs to count: frame $k has $na and $nb atoms of the pair, $nab in both"
            )
        )
        rdf_histogram!(hist, f.positions, f.cell, f.species, r_max; species_pair)
        weight += pairs / T(cell_volume(f.cell))
        na_total += na
    end
    Δ = T(r_max) / nbins
    cum = 0
    for (k, b) in enumerate(eachindex(hist))
        lo, hi = (k - 1) * Δ, k * Δ
        r[b] = (lo + hi) / 2
        g[b] = hist[b] / (weight * T(4π) / 3 * (hi^3 - lo^3))
        cum += hist[b]
        n[b] = T(cum) / T(na_total)
    end
    return (r = r, g = g, n = n)
end
