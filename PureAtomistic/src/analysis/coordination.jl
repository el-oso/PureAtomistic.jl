# Coordination numbers: neighbors within a cutoff, under the pair search of rdf.jl.

"""
    coordination_numbers!(counts, positions, cell, species, r_cut; species_pair = nothing) -> counts

Writes into `counts` (indexed like the atoms) the number of neighbors of each atom closer than
`r_cut` (Å): for an atom of the first species of `species_pair`, the atoms of the second species
and their periodic images, itself excluded at zero translation; `nothing` selects all atoms.
Atoms that are not of the first species get 0. Images are found as in
[`rdf_histogram!`](@ref): the minimum image when `r_cut` is at most half the shortest cell height,
every image within `r_cut` otherwise. Allocates nothing.
"""
function coordination_numbers!(
        counts::AbstractVector{<:Integer}, positions::AbstractMatrix, cell::AbstractMatrix, species::AbstractVector,
        r_cut::Real; species_pair = nothing
    )
    _check_positions(positions, species)
    axes(counts, 1) == axes(positions, 2) ||
        throw(DimensionMismatch(lazy"pureatomistic: counts axes $(axes(counts, 1)) differ from the atom axis $(axes(positions, 2))"))
    T = float(promote_type(eltype(positions), eltype(cell)))
    rc = T(r_cut)
    (rc > 0 && isfinite(rc)) || throw(ArgumentError(lazy"pureatomistic: r_cut must be positive and finite, got $r_cut"))
    sa, sb = _pair_species(species_pair)
    fill!(counts, 0)
    _foreach_pair(positions, cell, species, sa, sb, rc) do a, _, _
        counts[a] += 1
    end
    return counts
end

"""
    coordination_numbers(frame, r_cut; species_pair = nothing) -> Vector{Int}

Coordination numbers of one frame (anything with `positions`, `cell` and `species`, such as a
`System` or a `Frame`): see [`coordination_numbers!`](@ref). With `species_pair = nothing` the
result has one entry per atom, indexed like the atoms (an `OffsetArray` frame gives an
`OffsetArray`); with `species_pair = (A, B)` it has one entry per atom of species `A`, in atom
order, as a 1-based vector. Throws if no atom has species `A`. The running integral of `g(r)`,
the mean coordination number as a function of distance, is the `n` returned by [`rdf`](@ref).
"""
function coordination_numbers(frame, r_cut::Real; species_pair = nothing)
    counts = similar(frame.species, Int, (axes(frame.positions, 2),))
    coordination_numbers!(counts, frame.positions, frame.cell, frame.species, r_cut; species_pair)
    return _pick_species(counts, frame.species, species_pair)
end

_pick_species(counts, species, ::Nothing) = counts
function _pick_species(counts, species, pair)
    sa = first(_pair_species(pair))
    out = Int[counts[a] for a in eachindex(species) if _selected(species, a, sa)]
    isempty(out) && throw(ArgumentError(lazy"pureatomistic: no atom has species $sa"))
    return out
end
