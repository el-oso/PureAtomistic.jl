# Binary trajectory, little endian, every offset a multiple of 8 so the file can be memory-mapped
# and its arrays reinterpreted in place.
#
#   header   magic "PATRAJ01" (8 bytes), version UInt32 (1), flags UInt32 (bit 0: velocities,
#            bit 1: cell), n_atoms UInt64, then n_atoms 4-byte NUL-padded element symbols,
#            padded with NULs to a multiple of 8 bytes
#   frame    time Float64 (fs), energy Float64 (eV, NaN if absent), [cell 3×3 Float64, columns =
#            lattice vectors], positions 3×n Float32 (Å), [velocities 3×n Float32],
#            NUL padding to a multiple of 8 bytes
#
# Every frame has the same size, so frame k starts at `header_bytes + (k - 1) * frame_bytes` and
# frames are appended without touching earlier bytes.

const _TRAJ_MAGIC = (0x50, 0x41, 0x54, 0x52, 0x41, 0x4a, 0x30, 0x31)  # "PATRAJ01"
const _TRAJ_VERSION = UInt32(1)
const _FLAG_VELOCITIES = UInt32(1)
const _FLAG_CELL = UInt32(2)

_pad8(n::Integer) = (8 - n % 8) % 8

function _traj_sizes(n::Int, flags::UInt32)
    header = 24 + 4n + _pad8(4n)
    body = 16 + (flags & _FLAG_CELL != 0 ? 72 : 0) + 12n + (flags & _FLAG_VELOCITIES != 0 ? 12n : 0)
    return header, body + _pad8(body)
end

function _check_little_endian(bom::UInt32 = Base.ENDIAN_BOM)
    bom == 0x04030201 ||
        throw(ArgumentError("pureatomistic: the trajectory format is little endian and this host is not"))
    return nothing
end

"""
    TrajectoryWriter(path, n_atoms, species; velocities = true, cell = true)

Create (truncating) a binary trajectory file for `n_atoms` atoms whose element symbols
(at most 4 ASCII bytes each) are `species`. `velocities` and `cell` choose which optional blocks
each frame carries; `write_frame` must then be given them. Positions and velocities are stored as
`Float32`, so a `Float32` simulation restarts bit for bit from a stored frame. Close with `close`.
"""
mutable struct TrajectoryWriter
    io::IOStream
    n::Int
    flags::UInt32
    head::Vector{Float64}
    positions::Matrix{Float32}
    velocities::Matrix{Float32}
    cell::Matrix{Float64}
    padding::Vector{UInt8}
    nframes::Int

    function TrajectoryWriter(path::AbstractString, n_atoms::Integer, species; velocities::Bool = true, cell::Bool = true)
        _check_little_endian()
        n = Int(n_atoms)
        n >= 0 || throw(ArgumentError(lazy"pureatomistic: n_atoms must be non-negative, got $n"))
        length(species) == n ||
            throw(ArgumentError(lazy"pureatomistic: $(length(species)) species for $n atoms"))
        flags = (velocities ? _FLAG_VELOCITIES : UInt32(0)) | (cell ? _FLAG_CELL : UInt32(0))
        header_bytes, frame_bytes = _traj_sizes(n, flags)
        io = open(path, "w")
        try
            write(io, collect(UInt8, _TRAJ_MAGIC))
            write(io, _TRAJ_VERSION)
            write(io, flags)
            write(io, UInt64(n))
            for s in species
                b = codeunits(String(s))
                (isempty(b) || length(b) > 4 || !all(c -> 0x20 < c < 0x7f, b)) &&
                    throw(ArgumentError(lazy"pureatomistic: species \"$s\" must be 1 to 4 printable ASCII characters"))
                write(io, b, zeros(UInt8, 4 - length(b)))
            end
            write(io, zeros(UInt8, _pad8(4n)))
            position(io) == header_bytes || error("pureatomistic: trajectory header size mismatch")
        catch
            close(io)
            rethrow()
        end
        body = 16 + (cell ? 72 : 0) + 12n + (velocities ? 12n : 0)
        return new(
            io, n, flags, zeros(Float64, 2), zeros(Float32, 3, n), zeros(Float32, 3, velocities ? n : 0),
            zeros(Float64, 3, cell ? 3 : 0), zeros(UInt8, frame_bytes - body), 0
        )
    end
end

Base.close(w::TrajectoryWriter) = (close(w.io); nothing)
Base.flush(w::TrajectoryWriter) = (flush(w.io); nothing)

# Validate one frame and copy it into the writer's buffers, converting to the stored types.
# Nothing is written, so this allocates nothing.
function pack_frame!(
        w::TrajectoryWriter, time::Real, positions::AbstractMatrix{<:Real};
        velocities::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        cell::Union{Nothing, AbstractMatrix{<:Real}} = nothing, energy::Real = NaN
    )
    n = w.n
    size(positions) == (3, n) ||
        throw(ArgumentError(lazy"pureatomistic: positions must be 3×$n, got $(size(positions))"))
    want_v = w.flags & _FLAG_VELOCITIES != 0
    want_c = w.flags & _FLAG_CELL != 0
    (velocities === nothing) == want_v &&
        throw(ArgumentError(want_v ? "pureatomistic: this trajectory stores velocities; pass them" : "pureatomistic: this trajectory stores no velocities; do not pass them"))
    (cell === nothing) == want_c &&
        throw(ArgumentError(want_c ? "pureatomistic: this trajectory stores the cell; pass it" : "pureatomistic: this trajectory stores no cell; do not pass it"))
    velocities !== nothing && size(velocities) != (3, n) &&
        throw(ArgumentError(lazy"pureatomistic: velocities must be 3×$n, got $(size(velocities))"))
    cell !== nothing && size(cell) != (3, 3) &&
        throw(ArgumentError(lazy"pureatomistic: cell must be 3×3, got $(size(cell))"))
    w.head[1] = time
    w.head[2] = energy
    if cell !== nothing
        _copy_dense!(w.cell, cell)
    end
    _copy_dense!(w.positions, positions)
    if velocities !== nothing
        _copy_dense!(w.velocities, velocities)
    end
    return nothing
end

"""
    write_frame(w::TrajectoryWriter, time, positions; velocities, cell, energy = NaN)
    write_frame(w::TrajectoryWriter, frame::Frame)

Append one frame. `positions` and `velocities` are 3×n, `cell` is 3×3; `velocities` and `cell`
are required exactly when the writer was created with them. Allocation-free after warm-up for the
first form.
"""
function write_frame(
        w::TrajectoryWriter, time::Real, positions::AbstractMatrix{<:Real};
        velocities::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        cell::Union{Nothing, AbstractMatrix{<:Real}} = nothing, energy::Real = NaN
    )
    pack_frame!(w, time, positions; velocities, cell, energy)
    write(w.io, w.head)
    w.flags & _FLAG_CELL != 0 && write(w.io, w.cell)
    write(w.io, w.positions)
    w.flags & _FLAG_VELOCITIES != 0 && write(w.io, w.velocities)
    write(w.io, w.padding)
    w.nframes += 1
    return nothing
end

function write_frame(w::TrajectoryWriter, frame::Frame)
    return write_frame(
        w, frame.time, frame.positions;
        velocities = isempty(frame.velocities) ? nothing : frame.velocities,
        cell = isempty(frame.cell) ? nothing : frame.cell, energy = frame.energy
    )
end

"""
    TrajectoryReader(path)

Open a binary trajectory for random access: `length(r)` frames, `r.species` element symbols,
`read_frame(r, k)` or `r[k]` the k-th frame (`eachindex(r)` is `1:length(r)`). Close with `close`.
"""
struct TrajectoryReader
    io::IOStream
    n::Int
    flags::UInt32
    species::Vector{String}
    header_bytes::Int
    frame_bytes::Int
    nframes::Int

    function TrajectoryReader(path::AbstractString)
        _check_little_endian()
        io = open(path, "r")
        try
            total = filesize(path)
            total >= 24 || throw(ArgumentError(lazy"pureatomistic: \"$path\" is too short for a trajectory header"))
            magic = read(io, 8)
            magic == collect(UInt8, _TRAJ_MAGIC) ||
                throw(ArgumentError(lazy"pureatomistic: \"$path\" is not a PureAtomistic trajectory (bad magic)"))
            version = read(io, UInt32)
            version == _TRAJ_VERSION ||
                throw(ArgumentError(lazy"pureatomistic: unsupported trajectory version $version"))
            flags = read(io, UInt32)
            flags & ~(_FLAG_VELOCITIES | _FLAG_CELL) == 0 ||
                throw(ArgumentError(lazy"pureatomistic: unknown trajectory flags $(repr(flags))"))
            n64 = read(io, UInt64)
            n64 <= typemax(Int32) || throw(ArgumentError(lazy"pureatomistic: implausible atom count $n64"))
            n = Int(n64)
            header_bytes, frame_bytes = _traj_sizes(n, flags)
            total >= header_bytes || throw(ArgumentError(lazy"pureatomistic: \"$path\" has a truncated header"))
            species = String[]
            for _ in 1:n
                b = read(io, 4)
                push!(species, String(b[1:(something(findfirst(==(0x00), b), 5) - 1)]))
            end
            (total - header_bytes) % frame_bytes == 0 ||
                throw(ArgumentError(lazy"pureatomistic: \"$path\" ends in a partial frame ($((total - header_bytes) % frame_bytes) extra bytes)"))
            return new(io, n, flags, species, header_bytes, frame_bytes, (total - header_bytes) ÷ frame_bytes)
        catch
            close(io)
            rethrow()
        end
    end
end

Base.close(r::TrajectoryReader) = (close(r.io); nothing)
Base.length(r::TrajectoryReader) = r.nframes

"""
    read_frame(r::TrajectoryReader, k, T = Float64) -> Frame{T}

Read frame `k` (1-based). Positions and velocities were stored as `Float32` and are converted to
`T`; for `T = Float32` and `T = Float64` the conversion is exact. `pbc` is all true when the
trajectory stores a cell, all false otherwise.
"""
function read_frame(r::TrajectoryReader, k::Integer, ::Type{T} = Float64) where {T <: AbstractFloat}
    1 <= k <= r.nframes || throw(ArgumentError(lazy"pureatomistic: frame $k out of range 1:$(r.nframes)"))
    seek(r.io, r.header_bytes + (k - 1) * r.frame_bytes)
    time = read(r.io, Float64)
    energy = read(r.io, Float64)
    cell = zeros(Float64, r.flags & _FLAG_CELL != 0 ? 3 : 0, r.flags & _FLAG_CELL != 0 ? 3 : 0)
    isempty(cell) || read!(r.io, cell)
    pos = Matrix{Float32}(undef, 3, r.n)
    read!(r.io, pos)
    vel = Matrix{Float32}(undef, 3, r.flags & _FLAG_VELOCITIES != 0 ? r.n : 0)
    isempty(vel) || read!(r.io, vel)
    return Frame{T}(
        r.species, _dense(T, pos), zeros(T, 3, 0), _dense(T, vel), _dense(T, cell),
        zeros(T, 0, 0), zeros(T, 0, 0), isempty(cell) ? (false, false, false) : (true, true, true),
        convert(T, energy), convert(T, time), Pair{String, String}[]
    )
end

# A reader is an indexable collection of its frames, so code written against `eachindex` and
# `getindex` (the trajectory analysis of `Analysis`) reads it like a vector of `Frame`s.
Base.keys(r::TrajectoryReader) = Base.OneTo(r.nframes)
Base.firstindex(r::TrajectoryReader) = first(keys(r))
Base.lastindex(r::TrajectoryReader) = r.nframes
Base.getindex(r::TrajectoryReader, k::Integer) = read_frame(r, k)
