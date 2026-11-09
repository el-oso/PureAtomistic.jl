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

# `nothing` selects every atom, so a pair with `nothing` on both sides is a pair of one species.
_same_species(a, b) = isnothing(a) ? isnothing(b) : (!isnothing(b) && a == b)

"""
    rdf(frames, r_max, nbins; species_pair = nothing) -> (r, g, n)

Radial distribution function of the pairs `(A, B) = species_pair` (`nothing`: all atoms, either
element may be `nothing`) over `frames`, in `nbins` bins of width `Δ = r_max / nbins` up to
`r_max` (Å). `r` holds the bin centers. With `C_k` the ordered-pair count of bin `k` summed over
the frames (see [`rdf_histogram!`](@ref), which also states when the minimum image is used),

    g_k = C_k / (Σ_frames N_A (N_B - δ_AB) / V · 4π/3 (r_hi³ - r_lo³)),

the count divided by the count an ideal gas of the same composition would give: each `A` atom
sees the other `N_B - δ_AB` atoms of `B` at the uniform density `(N_B - δ_AB) / V`, with
`δ_AB = 1` when the two species are the same. An ideal gas therefore gives `g = 1` on average
when `r_max` is at most half the shortest cell height. Beyond that an atom's own periodic images
are counted as neighbors, as they are in the infinite periodic system, so a cell of few atoms
gives `g` near `N_B / (N_B - 1)` at large `r`.

`n` is the running coordination number, the integral `∫₀^{r_hi} ρ g 4πr² dr` in histogram form:
`n_k = Σ_{j ≤ k} C_j / Σ_frames N_A`, the mean number of `B` neighbors of an `A` atom closer
than the upper edge of bin `k`.

`frames` is any collection with `eachindex` and `getindex` whose elements have `positions`,
`cell` and `species` (a vector of `System`s or `Frame`s, or a `TrajectoryReader`). Views and
`OffsetArray`s are accepted; `r`, `g` and `n` are plain vectors over the bins.
"""
function rdf(frames, r_max::Real, nbins::Integer; species_pair = nothing)
    nbins >= 1 || throw(ArgumentError(lazy"pureatomistic: rdf needs at least one bin, got $nbins"))
    isempty(eachindex(frames)) && throw(ArgumentError("pureatomistic: rdf needs at least one frame"))
    f1 = frames[first(eachindex(frames))]
    T = float(promote_type(eltype(f1.positions), eltype(f1.cell)))
    sa, sb = _pair_species(species_pair)
    δ = _same_species(sa, sb) ? 1 : 0
    hist = zeros(Int, nbins)
    weight = zero(T)
    na_total = 0
    for k in eachindex(frames)
        f = frames[k]
        na = _count_species(f.species, sa)
        nb = _count_species(f.species, sb)
        (na >= 1 && nb - δ >= 1) ||
            throw(
            ArgumentError(
                lazy"pureatomistic: rdf needs atoms of both species of the pair and at least two of one species; frame $k has $na and $nb"
            )
        )
        rdf_histogram!(hist, f.positions, f.cell, f.species, r_max; species_pair)
        weight += T(na) * T(nb - δ) / T(cell_volume(f.cell))
        na_total += na
    end
    Δ = T(r_max) / nbins
    r = Vector{T}(undef, nbins)
    g = Vector{T}(undef, nbins)
    n = Vector{T}(undef, nbins)
    cum = 0
    for k in 1:nbins
        lo, hi = (k - 1) * Δ, k * Δ
        r[k] = (lo + hi) / 2
        g[k] = hist[k] / (weight * T(4π) / 3 * (hi^3 - lo^3))
        cum += hist[k]
        n[k] = T(cum) / T(na_total)
    end
    return (r = r, g = g, n = n)
end
