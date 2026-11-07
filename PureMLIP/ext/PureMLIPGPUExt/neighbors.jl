# The neighbor list built on the device: the cell list of the CPU's `neighbor_list` (same grid,
# same reach rule, same edges in the same order), sorted by counting on the device, and the skin
# criterion that decides on the device whether `neighbor_list!` rebuilds.
#
# A build is a fixed sequence of kernel launches. Each kernel reads the status word first and does
# nothing unless `status[STATUS_REBUILT]` is set and `status[STATUS_OVERFLOW]` is zero, so the host
# launches the sequence without reading anything from the device until the end.
#
# `status[STATUS_OVERFLOW]` holds the number of edges found when it exceeds the capacity, or one of
# three negative codes: `_INVALID_GEOMETRY` when the cell is not finite or thinner than cutoff/65536
# along a lattice direction (a singular cell included), or an atom's position is not finite or lies
# 2^30 or more cell lengths away; `_TOO_MANY_EDGES` when the edges do not fit the `Int32` edge
# indices; `_MISSING_REVERSE` when an edge's reverse is not in the list. The first two leave the
# list empty (`n_edges[1] = 0`, every row empty, every slot zero); the third is found after the
# edges are written. Every flag stays set: every later `neighbor_list!` and evaluation with the list
# throws, and the list must be built anew with `device_neighbor_list`.

const _INVALID_GEOMETRY = Int32(-1)
const _TOO_MANY_EDGES = Int32(-2)
const _MISSING_REVERSE = Int32(-3)

# Work items of the one work group of a prefix sum.
const _SCAN_GROUP = 256

# Work items that search the candidate senders of one receiver, each over a contiguous part of the
# grid-cell offsets. One work item per receiver leaves a GPU mostly idle on small systems; a rebuild on
# Metal (Apple M6, Float32) took 2.06, 1.18, 0.58 and 0.48 ms at cu_bench(5) and 4.89, 3.99, 3.78 and
# 4.18 ms at cu_bench(14) with 1, 3, 9 and 27 parts (one run each).
const _SEARCH_PARTS = 9

@inline _building(status) = !iszero(status[STATUS_REBUILT]) && iszero(status[STATUS_OVERFLOW])

@inline _cross3(u, v) = (u[2] * v[3] - u[3] * v[2], u[3] * v[1] - u[1] * v[3], u[1] * v[2] - u[2] * v[1])
@inline _dot3(u, v) = u[1] * v[1] + u[2] * v[2] + u[3] * v[3]

# Device scratch of a list of `n` atoms; the grid holds at most one grid cell per atom, as on the CPU.
function _list_scratch(backend::Backend, ::Type{T}, n::Int) where {T}
    nbins = max(n, 1)
    zi(dims...) = _device_zeros(backend, Int32, dims)
    zt(dims...) = _device_zeros(backend, T, dims)
    return (; reciprocal = zt(3, 3), grid = zi(7), image = zi(3, n), bin = zi(3, n), bin_start = zi(nbins + 1), bin_fill = zi(nbins), bin_atoms = zi(n), slots = zi(n * _SEARCH_PARTS + 1), reference = zt(3, n), reference_cell = zt(3, 3))
end

# The list with edge arrays `edges = (row_ptr, receivers, senders, shifts, reverse, n_edges)` and
# scratch `s`, with a zero status word.
function _new_list(backend::Backend, ::Type{T}, r_cut::T, skin::T, cutoff::T, n::Int, capacity::Int, edges, s) where {T}
    row_ptr, receivers, senders, shifts, reverse, n_edges = edges
    return DeviceNeighborList(
        backend, r_cut, skin, n, capacity, row_ptr, receivers, senders, shifts, reverse, n_edges,
        _device_zeros(backend, Int32, (STATUS_LENGTH,)), cutoff, s.reciprocal, s.grid, s.image, s.bin, s.bin_start, s.bin_fill,
        s.bin_atoms, s.slots, s.reference, s.reference_cell, zeros(Int32, STATUS_LENGTH),
    )
end

# Zero edge arrays for `n` atoms and `capacity` edges.
function _empty_edges(backend::Backend, n::Int, capacity::Int)
    zi(dims...) = _device_zeros(backend, Int32, dims)
    return (zi(n + 1), zi(capacity), zi(capacity), zi(3, capacity), zi(capacity), zi(1))
end

# The checks shared by the device list's entry points; returns the backend.
function _check_list_arguments(positions::AbstractMatrix, cell::AbstractMatrix)
    _require_one_based(positions, cell)
    backend = _common_backend(positions, cell)
    KernelAbstractions.supports_atomics(backend) ||
        throw(ArgumentError(lazy"gpu: the neighbor list needs atomic operations, which $(_backend_name(backend)) does not support"))
    size(positions, 1) == 3 || throw(DimensionMismatch(lazy"gpu: positions have size $(size(positions)), expected 3 rows"))
    size(cell) == (3, 3) || throw(DimensionMismatch(lazy"gpu: the cell has size $(size(cell)), expected (3, 3)"))
    return backend
end

"""
    device_neighbor_list(positions, cell, r_cut; skin = 0, extra_capacity = 0, extra_capacity_fraction = 0) -> DeviceNeighborList

The edges within `r_cut + skin` of the device arrays `positions` (`3 × n`) and `cell` (lattice
vectors as columns), built on their backend; they equal those of `neighbor_list(positions, cell,
r_cut + skin)` in the same order. With `m` edges found, the capacity is
`m + extra_capacity + ceil(extra_capacity_fraction · m)`. The build reads the edge count back to
size the list and then the status word of the sized list (two copies to the host), so this is not
for use inside an MD step; `neighbor_list!` and `device_rebuild!` rebuild an existing list there.

The default skin is 0: on Metal (Apple M6, Float32, MACE-MPA-0) a rebuild costs 0.85 % of a
`mace_energy_forces!` at cu_bench(5) (0.58 of 68 ms) and 0.29 % at cu_bench(14) (3.8 of 1307 ms),
below the 2 % at which a skin would be chosen, and in 100 MD steps at 1300 K a list without skin
is also the fastest: cu_bench(14) takes 140 s against 154, 224 and 236 s with skins of 0.5, 1.0
and 1.5 Å, which add edges to every evaluation (`bench/results/gpu_neighbor_skin_metal.json`).

Spare capacity costs evaluation time, since the per-edge stages run over every edge slot: on Metal
at cu_bench(14), `mace_energy_forces!` takes 5.6 % longer with 10 % spare slots and 24 % longer with
50 % (`bench/results/gpu_capacity_cost_metal.json`).
"""
function PureMLIP.device_neighbor_list(
        positions::AbstractMatrix, cell::AbstractMatrix, r_cut::Real; skin::Real = 0, extra_capacity::Integer = 0,
        extra_capacity_fraction::Real = 0
    )
    T = eltype(positions)
    eltype(cell) === T || throw(ArgumentError(lazy"gpu: positions and cell must share one element type, got $T and $(eltype(cell))"))
    T <: AbstractFloat || throw(ArgumentError(lazy"gpu: positions must hold floating-point numbers, got $T"))
    backend = _check_list_arguments(positions, cell)
    (isfinite(r_cut) && r_cut > 0) || throw(ArgumentError(lazy"gpu: r_cut must be positive and finite, got $r_cut"))
    (isfinite(skin) && skin >= 0) || throw(ArgumentError(lazy"gpu: the skin must not be negative, got $skin"))
    extra_capacity >= 0 || throw(ArgumentError(lazy"gpu: extra_capacity must not be negative, got $extra_capacity"))
    (isfinite(extra_capacity_fraction) && extra_capacity_fraction >= 0) ||
        throw(ArgumentError(lazy"gpu: extra_capacity_fraction must be finite and non-negative, got $extra_capacity_fraction"))
    n = size(positions, 2)
    cutoff = T(Float64(r_cut) + Float64(skin))
    s = _list_scratch(backend, T, n)
    # A list without capacity counts the edges; its overflow flag holds their number.
    probe = _new_list(backend, T, T(r_cut), T(skin), cutoff, n, 0, _empty_edges(backend, n, 0), s)
    _rebuild!(probe, positions, cell, true)
    copyto!(probe.host_status, probe.status)
    _check_list_flags(probe.host_status)
    m = Int(probe.host_status[STATUS_OVERFLOW])
    capacity = m + Int(extra_capacity) + ceil(Int, Float64(extra_capacity_fraction) * m)
    nl = _new_list(backend, T, T(r_cut), T(skin), cutoff, n, capacity, _empty_edges(backend, n, capacity), s)
    _rebuild!(nl, positions, cell, true)
    _read_list_status!(nl, positions, cell)
    return nl
end

"""
    neighbor_list!(nl::DeviceNeighborList, positions, cell) -> nl

Rebuild `nl` on the device for the device arrays `positions` and `cell` when an atom has moved by
more than `nl.skin / 2` since the last build, or when any entry of the cell differs from the cell
of the last build (the periodic images move with the cell, so a list kept across a cell change
would miss edges); with `nl.skin == 0` every call rebuilds. The decision is taken on the device and
written to `nl.status[STATUS_REBUILT]`; an overflow of the capacity is written to
`nl.status[STATUS_OVERFLOW]` as the number of edges found. The host then reads the status word once
and throws `"gpu: m edges exceed the capacity c"` on an overflow, after which `nl` holds no edges
and must be built anew with `device_neighbor_list`. Two atoms at one place throw too. A cell that
is not finite or thinner than `cutoff/65536` (a singular cell included), or a position that is not
finite or lies 2^30 or more cell lengths away, sets `nl.status[STATUS_OVERFLOW] = -1`, leaves `nl`
empty and throws; so do 2^31 - 1 edges or more (`-2`), which the `Int32` edge indices cannot hold.
An edge whose reverse is missing sets `-3` and throws. Every one of these flags stays set until
the list is built anew.
"""
function PureMLIP.neighbor_list!(nl::DeviceNeighborList, positions::AbstractMatrix, cell::AbstractMatrix)
    PureMLIP.device_rebuild!(nl, positions, cell)
    _read_list_status!(nl, positions, cell)
    return nl
end

function PureMLIP.device_rebuild!(nl::DeviceNeighborList{T}, positions::AbstractMatrix, cell::AbstractMatrix) where {T}
    _check_list_arguments(positions, cell)
    _common_backend(positions, nl.status)
    (eltype(positions) === T && eltype(cell) === T) ||
        throw(ArgumentError(lazy"gpu: positions and cell must have the list's element type $T, got $(eltype(positions)) and $(eltype(cell))"))
    size(positions, 2) == nl.n_atoms ||
        throw(DimensionMismatch(lazy"gpu: positions have size $(size(positions)), the neighbor list has $(nl.n_atoms) atoms"))
    _rebuild!(nl, positions, cell, false)
    return nl
end

# One read of the status word, and a throw for each flag a build sets.
function _read_list_status!(nl::DeviceNeighborList, positions, cell)
    status = nl.host_status
    copyto!(status, nl.status)
    _check_list_flags(status)
    m = status[STATUS_OVERFLOW]
    m > 0 && throw(ArgumentError(lazy"gpu: $m edges exceed the capacity $(nl.capacity)"))
    iszero(status[STATUS_ZERO_LENGTH]) || _throw_coincident(nl, positions, cell)
    return nothing
end

# Throws for the negative codes of `status[STATUS_OVERFLOW]`; a count of edges is left to the caller.
function _check_list_flags(status::Vector{Int32})
    flag = status[STATUS_OVERFLOW]
    flag == _INVALID_GEOMETRY &&
        throw(ArgumentError("gpu: the neighbor list needs a finite cell at least cutoff/65536 thick and finite positions within 2^30 cell lengths"))
    flag == _TOO_MANY_EDGES && throw(ArgumentError("gpu: the neighbor list has 2^31 - 1 edges or more, more than its Int32 edge indices hold"))
    flag == _MISSING_REVERSE && throw(ArgumentError("gpu: an edge of the neighbor list has no reverse edge"))
    flag < 0 && throw(ArgumentError(lazy"gpu: the neighbor list status holds the unknown code $flag"))
    return nothing
end

# Names the first pair of atoms at one place, from host copies of the list.
function _throw_coincident(nl::DeviceNeighborList, positions, cell)
    P, C = Array(positions), Array(cell)
    snd, rcv, sh = Array(nl.senders), Array(nl.receivers), Array(nl.shifts)
    for e in 1:Int(Array(nl.n_edges)[1])
        i, j = snd[e], rcv[e]
        r = ntuple(a -> P[a, j] - P[a, i] + C[a, 1] * sh[1, e] + C[a, 2] * sh[2, e] + C[a, 3] * sh[3, e], 3)
        iszero(_dot3(r, r)) && throw(ArgumentError(lazy"gpu: atoms $(min(i, j)) and $(max(i, j)) coincide"))
    end
    throw(ArgumentError("gpu: two atoms coincide"))
end

# The build sequence. `force` sets the rebuild flag; otherwise the skin criterion sets it.
function _rebuild!(nl::DeviceNeighborList{T}, positions, cell, force::Bool) where {T}
    backend, n = nl.backend, nl.n_atoms
    status = nl.status
    fill!(view(status, STATUS_ZERO_LENGTH:STATUS_ZERO_LENGTH), zero(Int32))
    if force || iszero(nl.skin)
        fill!(view(status, STATUS_REBUILT:STATUS_REBUILT), one(Int32))
    else
        fill!(view(status, STATUS_REBUILT:STATUS_REBUILT), zero(Int32))
        half = nl.skin / 2
        _launch!(_moved_kernel!, backend, n, status, positions, cell, nl.reference, nl.reference_cell, half * half)
    end
    nbins = length(nl.bin_fill)
    rc2 = nl.cutoff * nl.cutoff
    _launch!(_grid_kernel!, backend, 1, nl.reciprocal, nl.grid, status, cell, nl.cutoff, nbins)
    _launch!(_bin_start_kernel!, backend, nbins + 1, nl.bin_start, status)
    _launch!(_bin_kernel!, backend, n, nl.image, nl.bin, nl.bin_start, status, positions, nl.reciprocal, nl.grid)
    _scan!(nl.bin_start, status, nl.grid, -1, backend)
    _launch!(_bin_fill_kernel!, backend, nbins, nl.bin_fill, status, nl.bin_start)
    _launch!(_scatter_kernel!, backend, n, nl.bin_atoms, nl.bin_fill, status, nl.bin, nl.grid)
    _launch!(_sort_bins_kernel!, backend, nbins, nl.bin_atoms, status, nl.bin_start, nl.grid)
    parts = n * _SEARCH_PARTS
    _launch!(_count_kernel!, backend, parts + 1, nl.slots, status, positions, cell, nl.image, nl.bin, nl.bin_start, nl.bin_atoms, nl.grid, rc2)
    _scan!(nl.slots, status, nl.grid, parts + 1, backend)
    _launch!(_rows_kernel!, backend, n + 1, nl.row_ptr, status, nl.slots)
    _launch!(_capacity_kernel!, backend, 1, nl.n_edges, status, nl.row_ptr, n, nl.capacity)
    _launch!(_empty_rows_kernel!, backend, n + 1, nl.row_ptr, status)
    _launch!(
        _fill_kernel!, backend, parts, nl.senders, nl.receivers, nl.shifts, status, positions, cell, nl.image, nl.bin, nl.bin_start, nl.bin_atoms,
        nl.grid, nl.slots, rc2,
    )
    _launch!(_reverse_kernel!, backend, nl.capacity, nl.reverse, nl.senders, nl.receivers, nl.shifts, status, nl.row_ptr, nl.n_edges)
    _launch!(_reference_kernel!, backend, n + 1, nl.reference, nl.reference_cell, status, positions, cell)
    return nothing
end

# The skin criterion: atom `i` sets the rebuild flag when it has moved by more than half the skin
# since the last build (`half2` is the square of half the skin), and the first work item sets it
# when any entry of the cell differs from the cell of the last build. Every work item that sets the
# flag writes the same value.
@kernel function _moved_kernel!(status, @Const(positions), @Const(cell), @Const(reference), @Const(reference_cell), half2)
    i = @index(Global)
    _moved!(status, positions, cell, reference, reference_cell, half2, i)
end

@inline function _moved!(status, positions, cell, reference, reference_cell, half2, i)
    d1 = positions[1, i] - reference[1, i]
    d2 = positions[2, i] - reference[2, i]
    d3 = positions[3, i] - reference[3, i]
    # `!(≤)` so that a position that is not finite rebuilds, and the build flags it.
    moved = !(d1 * d1 + d2 * d2 + d3 * d3 <= half2)
    if i == 1
        for k in 1:9
            moved |= cell[k] != reference_cell[k]
        end
    end
    moved && (status[STATUS_REBUILT] = one(eltype(status)))
    return nothing
end

@kernel function _grid_kernel!(reciprocal, grid, status, @Const(cell), cutoff, budget)
    _grid!(reciprocal, grid, status, cell, cutoff, budget)
end

# The reciprocal cell and the grid of the CPU's `_bin!`: a grid cell is a parallelepiped of the
# lattice whose width along direction k, measured perpendicular to the other two lattice vectors, is
# at least the cutoff (widened by 8 eps), so every neighbor of an atom lies at most `reach[k]` grid
# cells away along k; a cell thinner than the cutoff along k has one grid cell there, and `reach[k]`
# counts whole lattice translations. The grid has at most `budget` cells.
@inline function _grid!(reciprocal, grid, status, cell, cutoff, budget)
    _building(status) || return nothing
    T = eltype(reciprocal)
    finite = true
    for k in 1:9
        finite &= isfinite(cell[k])
    end
    a1, a2, a3 = (cell[1, 1], cell[2, 1], cell[3, 1]), (cell[1, 2], cell[2, 2], cell[3, 2]), (cell[1, 3], cell[2, 3], cell[3, 3])
    c1, c2, c3 = _cross3(a2, a3), _cross3(a3, a1), _cross3(a1, a2)
    det = _dot3(a1, c1)
    # The volume relative to the box of the three edge lengths is at most eps for a singular cell.
    if !(finite && sqrt(_dot3(a1, a1) * _dot3(a2, a2) * _dot3(a3, a3)) * eps(T) < abs(det))
        status[STATUS_OVERFLOW] = _INVALID_GEOMETRY
        return nothing
    end
    for (k, c) in enumerate((c1, c2, c3))
        reciprocal[k, 1] = c[1] / det
        reciprocal[k, 2] = c[2] / det
        reciprocal[k, 3] = c[3] / det
    end
    rc = cutoff * (1 + 8 * eps(T))
    height = (abs(det) / sqrt(_dot3(c1, c1)), abs(det) / sqrt(_dot3(c2, c2)), abs(det) / sqrt(_dot3(c3, c3)))
    dims = map(h -> h / rc >= budget ? budget : max(1, unsafe_trunc(Int, floor(h / rc))), height)
    while dims[1] * dims[2] * dims[3] > budget
        dims = dims[1] >= dims[2] && dims[1] >= dims[3] ? (dims[1] - 1, dims[2], dims[3]) :
            dims[2] >= dims[3] ? (dims[1], dims[2] - 1, dims[3]) : (dims[1], dims[2], dims[3] - 1)
    end
    span = map((nb, h) -> cutoff * nb / h, dims, height)
    # The reach is converted with `unsafe_trunc`, which needs it finite and within Int; at most 2^16
    # grid cells on each side also keeps the number of offsets, at most (2^17 + 1)^3, times
    # `_SEARCH_PARTS` within Int.
    if !(span[1] <= 65536 && span[2] <= 65536 && span[3] <= 65536)
        status[STATUS_OVERFLOW] = _INVALID_GEOMETRY
        return nothing
    end
    for k in 1:3
        grid[k] = dims[k]
        grid[k + 3] = max(1, unsafe_trunc(Int, ceil(span[k])))
    end
    grid[7] = dims[1] * dims[2] * dims[3]
    return nothing
end

# The counts of the grid cells start at zero, behind a leading 1, so that their prefix sum gives
# the first slot of each grid cell.
@kernel function _bin_start_kernel!(bin_start, @Const(status))
    b = @index(Global)
    _building(status) && (bin_start[b] = ifelse(b == 1, one(Int32), zero(Int32)))
end

@kernel function _bin_kernel!(image, bin, bin_start, status, @Const(positions), @Const(reciprocal), @Const(grid))
    i = @index(Global)
    _bin_atom!(image, bin, bin_start, status, positions, reciprocal, grid, i)
end

# Atom `i`'s lattice image (the translation taking it into the unit cell) and grid cell, as in the
# CPU's `_bin!`, and one more atom counted in its grid cell.
@inline function _bin_atom!(image, bin, bin_start, status, positions, reciprocal, grid, i)
    _building(status) || return nothing
    T = eltype(reciprocal)
    id, stride = 0, 1
    for k in 1:3
        f = reciprocal[k, 1] * positions[1, i] + reciprocal[k, 2] * positions[2, i] + reciprocal[k, 3] * positions[3, i]
        # Images are Int32, and the search subtracts two of them and a wrap of the grid, so an image
        # below 2^30 in magnitude keeps every shift within Int32.
        if !(abs(f) < T(1073741824))
            status[STATUS_OVERFLOW] = _INVALID_GEOMETRY
            return nothing
        end
        nk = Int(grid[k])
        lattice = floor(f)
        b = min(unsafe_trunc(Int, floor((f - lattice) * nk)), nk - 1)
        image[k, i] = unsafe_trunc(Int32, lattice)
        bin[k, i] = b
        id += stride * b
        stride *= nk
    end
    KernelAbstractions.@atomic bin_start[id + 2] += one(Int32)
    return nothing
end

@kernel function _bin_fill_kernel!(bin_fill, @Const(status), @Const(bin_start))
    b = @index(Global)
    _building(status) && (bin_fill[b] = bin_start[b])
end

@kernel function _scatter_kernel!(bin_atoms, bin_fill, @Const(status), @Const(bin), @Const(grid))
    i = @index(Global)
    _scatter!(bin_atoms, bin_fill, status, bin, grid, i)
end

# Atom `i` takes a free slot of its grid cell; the slots of a cell are then sorted.
@inline function _scatter!(bin_atoms, bin_fill, status, bin, grid, i)
    _building(status) || return nothing
    id = bin[1, i] + grid[1] * (bin[2, i] + grid[2] * bin[3, i]) + 1
    slot = KernelAbstractions.@atomic bin_fill[id] += one(Int32)
    bin_atoms[slot - 1] = i
    return nothing
end

@kernel function _sort_bins_kernel!(bin_atoms, @Const(status), @Const(bin_start), @Const(grid))
    b = @index(Global)
    _sort_bin!(bin_atoms, status, bin_start, grid, b)
end

# The atoms of grid cell `b` in increasing order, the order of the CPU's scatter (insertion sort:
# a grid cell holds a few dozen atoms).
@inline function _sort_bin!(bin_atoms, status, bin_start, grid, b)
    (_building(status) && b <= grid[7]) || return nothing
    lo, hi = Int(bin_start[b]), Int(bin_start[b + 1]) - 1
    for s in (lo + 1):hi
        a = bin_atoms[s]
        t = s - 1
        while t >= lo && bin_atoms[t] > a
            bin_atoms[t + 1] = bin_atoms[t]
            t -= 1
        end
        bin_atoms[t + 1] = a
    end
    return nothing
end

"""
    _search(positions, cell, image, bin, bin_start, bin_atoms, grid, rc2, j, part, visit, acc)

The candidate senders of receiver `j` in the order of the CPU's `_search!`: grid cells by offset
(z, then y, then x), and the atoms of a grid cell in increasing order. The offsets are numbered in
that order and split into `_SEARCH_PARTS` contiguous parts; this visits part `part` (0-based), so
that the parts of a receiver, taken in turn, visit its candidates in the CPU's order. The candidate
in grid cell `b + o` lies in the translated copy `q = fld(b + o, dims)` of the cell, so its shift
relative to the positions as given is `image[:, i] - image[:, j] - q`. For every edge within
`sqrt(rc2)` the search calls `acc = visit(acc, i, n1, n2, n3, d2)` and returns `acc`.
"""
@inline function _search(positions, cell, image, bin, bin_start, bin_atoms, grid, rc2, j, part, visit::F, acc) where {F}
    nx, ny, nz = Int(grid[1]), Int(grid[2]), Int(grid[3])
    rx, ry, rz = Int(grid[4]), Int(grid[5]), Int(grid[6])
    wx, wy = 2rx + 1, 2ry + 1
    offsets = wx * wy * (2rz + 1)
    for o in fld(part * offsets, _SEARCH_PARTS):(fld((part + 1) * offsets, _SEARCH_PARTS) - 1)
        oyz, px = divrem(o, wx)
        pz, py = divrem(oyz, wy)
        iz = bin[3, j] + (pz - rz)
        qz = fld(iz, nz)
        iy = bin[2, j] + (py - ry)
        qy = fld(iy, ny)
        ix = bin[1, j] + (px - rx)
        qx = fld(ix, nx)
        b = (ix - qx * nx) + nx * ((iy - qy * ny) + ny * (iz - qz * nz)) + 1
        for s in Int(bin_start[b]):(Int(bin_start[b + 1]) - 1)
            i = Int(bin_atoms[s])
            n1 = image[1, i] - image[1, j] - qx
            n2 = image[2, i] - image[2, j] - qy
            n3 = image[3, i] - image[3, j] - qz
            (i == j && iszero(n1) && iszero(n2) && iszero(n3)) && continue
            r1 = positions[1, j] - positions[1, i] + cell[1, 1] * n1 + cell[1, 2] * n2 + cell[1, 3] * n3
            r2 = positions[2, j] - positions[2, i] + cell[2, 1] * n1 + cell[2, 2] * n2 + cell[2, 3] * n3
            r3 = positions[3, j] - positions[3, i] + cell[3, 1] * n1 + cell[3, 2] * n2 + cell[3, 3] * n3
            d2 = r1 * r1 + r2 * r2 + r3 * r3
            d2 < rc2 && (acc = visit(acc, i, n1, n2, n3, d2))
        end
    end
    return acc
end

@kernel function _count_kernel!(slots, status, @Const(positions), @Const(cell), @Const(image), @Const(bin), @Const(bin_start), @Const(bin_atoms), @Const(grid), rc2)
    w = @index(Global)
    _count_part!(slots, status, positions, cell, image, bin, bin_start, bin_atoms, grid, rc2, w)
end

# Work item `w = (j - 1)·_SEARCH_PARTS + part + 2` writes the number of edges that part `part` of
# receiver `j` finds to `slots[w]`, and work item 1 writes `slots[1] = 1`, so that the prefix sum
# gives each part its first slot. An edge of length zero (two atoms at one place) is counted and
# flagged.
@inline function _count_part!(slots, status, positions, cell, image, bin, bin_start, bin_atoms, grid, rc2, w)
    _building(status) || return nothing
    if w == 1
        slots[1] = one(Int32)
        return nothing
    end
    j, part = divrem(w - 2, _SEARCH_PARTS)
    count = _search(positions, cell, image, bin, bin_start, bin_atoms, grid, rc2, j + 1, part, _count_edge, (0, false))
    slots[w] = min(count[1], Int(typemax(Int32)))
    count[2] && (status[STATUS_ZERO_LENGTH] = one(eltype(status)))
    return nothing
end

@inline _count_edge(acc, i, n1, n2, n3, d2) = (acc[1] + 1, acc[2] | iszero(d2))

# The first slot of each receiver's row, from the first slots of the parts.
@kernel function _rows_kernel!(row_ptr, @Const(status), @Const(slots))
    j = @index(Global)
    _building(status) && (row_ptr[j] = slots[(j - 1) * _SEARCH_PARTS + 1])
end

# The edge count `row_ptr[n + 1] - 1`, or an overflow when it exceeds the capacity.
@kernel function _capacity_kernel!(n_edges, status, @Const(row_ptr), n, capacity)
    _capacity!(n_edges, status, row_ptr, n, capacity)
end

# The prefix sum saturates at typemax(Int32), so a last row pointer of typemax(Int32) means 2^31 - 1
# edges or more, which the Int32 edge slots cannot index.
@inline function _capacity!(n_edges, status, row_ptr, n, capacity)
    iszero(status[STATUS_REBUILT]) && return nothing
    if iszero(status[STATUS_OVERFLOW])
        if row_ptr[n + 1] == typemax(Int32)
            status[STATUS_OVERFLOW] = _TOO_MANY_EDGES
        else
            m = row_ptr[n + 1] - one(Int32)
            m > capacity && (status[STATUS_OVERFLOW] = m)
        end
    end
    n_edges[1] = iszero(status[STATUS_OVERFLOW]) ? row_ptr[n + 1] - one(Int32) : zero(Int32)
    return nothing
end

# After an overflow or an invalid geometry every row is empty.
@kernel function _empty_rows_kernel!(row_ptr, @Const(status))
    i = @index(Global)
    (!iszero(status[STATUS_REBUILT]) && !iszero(status[STATUS_OVERFLOW])) && (row_ptr[i] = one(Int32))
end

@kernel function _fill_kernel!(
        senders, receivers, shifts, @Const(status), @Const(positions), @Const(cell), @Const(image), @Const(bin), @Const(bin_start),
        @Const(bin_atoms), @Const(grid), @Const(slots), rc2,
    )
    w = @index(Global)
    _fill_part!(senders, receivers, shifts, status, positions, cell, image, bin, bin_start, bin_atoms, grid, slots, rc2, w)
end

# The edges of part `part` of receiver `j` (`w = (j - 1)·_SEARCH_PARTS + part + 1`), in the order of
# the search, from slot `slots[w]`.
@inline function _fill_part!(senders, receivers, shifts, status, positions, cell, image, bin, bin_start, bin_atoms, grid, slots, rc2, w)
    _building(status) || return nothing
    j, part = divrem(w - 1, _SEARCH_PARTS)
    _search(positions, cell, image, bin, bin_start, bin_atoms, grid, rc2, j + 1, part, _StoreEdge(senders, receivers, shifts, j + 1), Int(slots[w]))
    return nothing
end

struct _StoreEdge{V, M}
    senders::V
    receivers::V
    shifts::M
    j::Int
end

@inline function (store::_StoreEdge)(e, i, n1, n2, n3, d2)
    store.senders[e] = i
    store.receivers[e] = store.j
    store.shifts[1, e] = n1
    store.shifts[2, e] = n2
    store.shifts[3, e] = n3
    return e + 1
end

@kernel function _reverse_kernel!(reverse, senders, receivers, shifts, status, @Const(row_ptr), @Const(n_edges))
    e = @index(Global)
    _reverse_edge!(reverse, senders, receivers, shifts, status, row_ptr, n_edges, e)
end

# The reverse of edge `e` (from its receiver to its sender, with the negated shift), found in the
# sender's row; a slot past the edges is zeroed. The search computes the vector of the reverse edge
# from the same operands with every sign flipped, and rounding is symmetric under negation, so
# every edge's reverse is in the list; an edge without one is flagged.
@inline function _reverse_edge!(reverse, senders, receivers, shifts, status, row_ptr, n_edges, e)
    iszero(status[STATUS_REBUILT]) && return nothing
    if e > n_edges[1]
        senders[e], receivers[e], reverse[e] = zero(Int32), zero(Int32), zero(Int32)
        shifts[1, e], shifts[2, e], shifts[3, e] = zero(Int32), zero(Int32), zero(Int32)
        return nothing
    end
    i, j = senders[e], receivers[e]
    n1, n2, n3 = shifts[1, e], shifts[2, e], shifts[3, e]
    rev = zero(Int32)
    for f in row_ptr[i]:(row_ptr[i + 1] - one(Int32))
        if senders[f] == j && shifts[1, f] == -n1 && shifts[2, f] == -n2 && shifts[3, f] == -n3
            rev = f
            break
        end
    end
    reverse[e] = rev
    iszero(rev) && (status[STATUS_OVERFLOW] = _MISSING_REVERSE)
    return nothing
end

# The positions and cell of the build, for the skin criterion of the next call.
@kernel function _reference_kernel!(reference, reference_cell, @Const(status), @Const(positions), @Const(cell))
    i = @index(Global)
    _keep_reference!(reference, reference_cell, status, positions, cell, i)
end

@inline function _keep_reference!(reference, reference_cell, status, positions, cell, i)
    _building(status) || return nothing
    if i <= size(reference, 2)
        for a in 1:3
            reference[a, i] = positions[a, i]
        end
    else
        for k in 1:9
            reference_cell[k] = cell[k]
        end
    end
    return nothing
end

# The inclusive prefix sum of `a[1:len]` in place, with `len = grid[7] + 1` when `n < 0` and `n`
# otherwise, in one work group: each work item sums a contiguous segment, the first work item sums
# the segment totals, and each adds its offset.
function _scan!(a::AbstractVector{Int32}, status, grid, n::Int, backend::Backend)
    _scan_kernel!(backend, _SCAN_GROUP)(a, status, grid, n; ndrange = _SCAN_GROUP)
    return nothing
end

@kernel function _scan_kernel!(a, @Const(status), @Const(grid), n)
    t = @index(Local)
    total = @localmem Int64 (_SCAN_GROUP,)
    run = _building(status)
    len = n < 0 ? Int(grid[7]) + 1 : n
    lo, hi = _scan_segment(t, len)
    total[t] = run ? _segment_sum!(a, lo, hi) : zero(Int64)
    @synchronize
    t == 1 && _exclusive_sum!(total)
    @synchronize
    run && _add_offset!(a, lo, hi, total[t])
end

@inline function _scan_segment(t, len)
    seg = cld(len, _SCAN_GROUP)
    return (t - 1) * seg + 1, min(t * seg, len)
end

# The sums run in Int64 over non-negative Int32 entries and are stored saturated at typemax(Int32),
# so a sum that does not fit reads as typemax(Int32) from there on instead of wrapping.
@inline _saturate(x::Int64) = Int32(min(x, Int64(typemax(Int32))))

@inline function _segment_sum!(a, lo, hi)
    acc = zero(Int64)
    for i in lo:hi
        acc += a[i]
        a[i] = _saturate(acc)
    end
    return acc
end

@inline function _exclusive_sum!(total)
    acc = zero(Int64)
    for t in eachindex(total)
        v = total[t]
        total[t] = acc
        acc += v
    end
    return nothing
end

@inline function _add_offset!(a, lo, hi, off)
    for i in lo:hi
        a[i] = _saturate(Int64(a[i]) + off)
    end
    return nothing
end
