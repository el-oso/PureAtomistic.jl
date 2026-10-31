"""
    NeighborList{T}

Directed edges `(sender, receiver, shift)` between atoms of a fully periodic cell, with
`r_cut` the cutoff in Å. Edge `e` with `e ≤ n_edges` carries the vector
`positions[:, receivers[e]] - positions[:, senders[e]] + cell * shifts[:, e]`, of length
`0 < ‖r‖ < r_cut`, and the edges are sorted by receiver. `senders`, `receivers` and `shifts`
have the capacity as length; only the first `n_edges` entries are valid. The remaining fields
are the scratch of the cell grid.
"""
mutable struct NeighborList{T <: AbstractFloat}
    r_cut::Float64
    n_atoms::Int
    n_edges::Int
    senders::Vector{Int}
    receivers::Vector{Int}
    shifts::Matrix{Int}
    reciprocal::Matrix{T}       # 3×3, rows are the reciprocal lattice vectors (without 2π)
    image::Matrix{Int}          # 3×n, lattice translation taking each atom into the unit cell
    bin::Matrix{Int}            # 3×n, 0-based grid cell of each atom
    dims::NTuple{3, Int}        # grid cells per lattice direction
    reach::NTuple{3, Int}       # grid cells searched on each side, per lattice direction
    bin_start::Vector{Int}      # first slot of each grid cell in `bin_atoms`, one past the last at the end
    bin_next::Vector{Int}
    bin_atoms::Vector{Int}
end

@inline _cross(u, v) = (u[2] * v[3] - u[3] * v[2], u[3] * v[1] - u[1] * v[3], u[1] * v[2] - u[2] * v[1])
@inline _dot(u, v) = u[1] * v[1] + u[2] * v[2] + u[3] * v[3]
@inline _column(A, k) = (A[1, k], A[2, k], A[3, k])

"""
    neighbor_list(positions, cell, r_cut; extra_capacity = 0) -> NeighborList

Edges within `r_cut` of a fully periodic system. `positions` is `3 × n` (one column per atom, Å),
`cell` is `3 × 3` with the lattice vectors as columns. Positions need not lie inside the cell.
The capacity is the number of edges found plus `extra_capacity`.
"""
function neighbor_list(positions::AbstractMatrix{T}, cell::AbstractMatrix{T}, r_cut::Real; extra_capacity::Integer = 0) where {T}
    Base.require_one_based_indexing(positions, cell)
    size(positions, 1) == 3 || throw(DimensionMismatch("neighbor list: positions must have 3 rows"))
    size(cell) == (3, 3) || throw(DimensionMismatch("neighbor list: the cell must be 3×3"))
    (isfinite(r_cut) && r_cut > 0) || throw(ArgumentError("neighbor list: r_cut must be positive and finite"))
    extra_capacity >= 0 || throw(ArgumentError("neighbor list: extra_capacity must not be negative"))
    n = size(positions, 2)
    nbins = max(n, 1)
    nl = NeighborList{T}(
        Float64(r_cut), n, 0, Int[], Int[], zeros(Int, 3, 0), zeros(T, 3, 3),
        zeros(Int, 3, n), zeros(Int, 3, n), (1, 1, 1), (1, 1, 1),
        zeros(Int, nbins + 1), zeros(Int, nbins), zeros(Int, n)
    )
    _bin!(nl, positions, cell)
    m = _search!(nl, positions, cell, Val(false))
    capacity = m + Int(extra_capacity)
    nl.senders = zeros(Int, capacity)
    nl.receivers = zeros(Int, capacity)
    nl.shifts = zeros(Int, 3, capacity)
    _search!(nl, positions, cell, Val(true))
    return nl
end

"""
    neighbor_list!(nl, positions, cell) -> nl

Rebuild `nl` for new `positions` and `cell` without allocating. After a capacity error `nl` must be
rebuilt with `neighbor_list`: its scratch already describes the new positions. Throws before changing the edges
if more than `length(nl.senders)` edges are found.
"""
function neighbor_list!(nl::NeighborList{T}, positions::AbstractMatrix{T}, cell::AbstractMatrix{T}) where {T}
    Base.require_one_based_indexing(positions, cell)
    size(positions) == (3, nl.n_atoms) || throw(DimensionMismatch(lazy"neighbor list: positions are $(size(positions)), expected (3, $(nl.n_atoms))"))
    size(cell) == (3, 3) || throw(DimensionMismatch("neighbor list: the cell must be 3×3"))
    _bin!(nl, positions, cell)
    m = _search!(nl, positions, cell, Val(false))
    capacity = length(nl.senders)
    m <= capacity || throw(ArgumentError(lazy"neighbor list: $m edges exceed the capacity $capacity"))
    _search!(nl, positions, cell, Val(true))
    return nl
end

# Reciprocal cell, grid dimensions, and the atoms sorted into grid cells.
#
# A grid cell is a parallelepiped of the lattice; its width along direction k, measured
# perpendicular to the other two lattice vectors, is at least `r_cut`, so every neighbor of an atom
# lies at most `reach[k]` grid cells away along k (modulo lattice translations). When the cell is
# thinner than `r_cut` along k there is one grid cell and `reach[k]` counts whole lattice translations.
# The number of grid cells never exceeds the length of `bin_next`, so the buffers never grow.
function _bin!(nl::NeighborList{T}, positions, cell) where {T}
    (all(isfinite, positions) && all(isfinite, cell)) || throw(ArgumentError("neighbor list: positions and cell must be finite"))
    a1, a2, a3 = _column(cell, 1), _column(cell, 2), _column(cell, 3)
    c1, c2, c3 = _cross(a2, a3), _cross(a3, a1), _cross(a1, a2)
    det = _dot(a1, c1)
    # The volume relative to the box of the three edge lengths is at most eps for a singular cell.
    sqrt(_dot(a1, a1) * _dot(a2, a2) * _dot(a3, a3)) * eps(T) < abs(det) || throw(ArgumentError("neighbor list: the cell is singular"))
    for (k, c) in enumerate((c1, c2, c3))
        nl.reciprocal[k, 1] = c[1] / det
        nl.reciprocal[k, 2] = c[2] / det
        nl.reciprocal[k, 3] = c[3] / det
    end
    rc = T(nl.r_cut) * (1 + 8 * eps(T))
    budget = length(nl.bin_next)
    height = (abs(det) / sqrt(_dot(c1, c1)), abs(det) / sqrt(_dot(c2, c2)), abs(det) / sqrt(_dot(c3, c3)))
    dims = map(h -> h / rc >= budget ? budget : max(1, floor(Int, h / rc)), height)
    while prod(dims) > budget
        dims = dims[1] >= dims[2] && dims[1] >= dims[3] ? (dims[1] - 1, dims[2], dims[3]) :
            dims[2] >= dims[3] ? (dims[1], dims[2] - 1, dims[3]) : (dims[1], dims[2], dims[3] - 1)
    end
    nl.dims = dims
    nl.reach = map((nb, h) -> max(1, ceil(Int, T(nl.r_cut) * nb / h)), dims, height)

    nbins = prod(dims)
    for i in 1:(nbins + 1)
        nl.bin_start[i] = 0
    end
    for i in 1:(nl.n_atoms)
        id = 0
        stride = 1
        for k in 1:3
            f = nl.reciprocal[k, 1] * positions[1, i] + nl.reciprocal[k, 2] * positions[2, i] + nl.reciprocal[k, 3] * positions[3, i]
            lattice = floor(Int, f)
            nl.image[k, i] = lattice
            b = min(floor(Int, (f - lattice) * dims[k]), dims[k] - 1)
            nl.bin[k, i] = b
            id += stride * b
            stride *= dims[k]
        end
        nl.bin_start[id + 2] += 1
    end
    nl.bin_start[1] = 1
    for b in 1:nbins
        nl.bin_start[b + 1] += nl.bin_start[b]
        nl.bin_next[b] = nl.bin_start[b]
    end
    for i in 1:(nl.n_atoms)
        id = nl.bin[1, i] + dims[1] * (nl.bin[2, i] + dims[2] * nl.bin[3, i]) + 1
        nl.bin_atoms[nl.bin_next[id]] = i
        nl.bin_next[id] += 1
    end
    return nothing
end

# Visit the candidate senders of every receiver j in order. The candidate in grid cell `b + o`
# lies in the translated copy `q = fld(b + o, dims)` of the cell, so the image seen from the
# unit-cell representatives is `-q`, and relative to the positions as given it is
# `n = image[:, sender] - image[:, receiver] - q`.
# Returns the number of edges; with `Val(true)` also stores them.
function _search!(nl::NeighborList{T}, positions, cell, ::Val{store}) where {T, store}
    rc2 = T(nl.r_cut)^2
    nx, ny, nz = nl.dims
    rx, ry, rz = nl.reach
    e = 0
    for j in 1:(nl.n_atoms)
        for oz in (-rz):rz
            iz = nl.bin[3, j] + oz
            qz = fld(iz, nz)
            for oy in (-ry):ry
                iy = nl.bin[2, j] + oy
                qy = fld(iy, ny)
                for ox in (-rx):rx
                    ix = nl.bin[1, j] + ox
                    qx = fld(ix, nx)
                    b = (ix - qx * nx) + nx * ((iy - qy * ny) + ny * (iz - qz * nz)) + 1
                    for s in nl.bin_start[b]:(nl.bin_start[b + 1] - 1)
                        i = nl.bin_atoms[s]
                        n1 = nl.image[1, i] - nl.image[1, j] - qx
                        n2 = nl.image[2, i] - nl.image[2, j] - qy
                        n3 = nl.image[3, i] - nl.image[3, j] - qz
                        (i == j && n1 == 0 && n2 == 0 && n3 == 0) && continue
                        r1 = positions[1, j] - positions[1, i] + cell[1, 1] * n1 + cell[1, 2] * n2 + cell[1, 3] * n3
                        r2 = positions[2, j] - positions[2, i] + cell[2, 1] * n1 + cell[2, 2] * n2 + cell[2, 3] * n3
                        r3 = positions[3, j] - positions[3, i] + cell[3, 1] * n1 + cell[3, 2] * n2 + cell[3, 3] * n3
                        d2 = r1 * r1 + r2 * r2 + r3 * r3
                        d2 == 0 && throw(ArgumentError(lazy"neighbor list: atoms $(min(i, j)) and $(max(i, j)) coincide"))
                        d2 < rc2 || continue
                        e += 1
                        if store
                            nl.senders[e] = i
                            nl.receivers[e] = j
                            nl.shifts[1, e] = n1
                            nl.shifts[2, e] = n2
                            nl.shifts[3, e] = n3
                        end
                    end
                end
            end
        end
    end
    store && (nl.n_edges = e)
    return e
end
