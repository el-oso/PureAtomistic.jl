# Extended XYZ: the comment line of each frame carries `key=value` pairs, the atom lines carry
# the columns named by `Properties`. Lattice is nine numbers, the three cell vectors in order
# (a1x a1y a1z a2x ...), which fills a 3×3 matrix whose columns are the vectors.

"""
    Frame{T}

One configuration. Positions, forces and velocities are 3×n matrices; `cell` is 3×3 with the
lattice vectors as columns. An absent quantity is empty: `forces` and `velocities` have size
`(3, 0)`; `cell`, `stress` and `virial` have size `(0, 0)`; `energy` and `time` are `NaN`.
`extras` holds the comment-line pairs that have no field.

    Frame(species, positions; forces, velocities, cell, stress, virial, pbc, energy, time, extras)

Arrays already of type `Matrix{T}` are stored without copying.
"""
struct Frame{T <: AbstractFloat}
    species::Vector{String}
    positions::Matrix{T}
    forces::Matrix{T}
    velocities::Matrix{T}
    cell::Matrix{T}
    stress::Matrix{T}
    virial::Matrix{T}
    pbc::NTuple{3, Bool}
    energy::T
    time::T
    extras::Vector{Pair{String, String}}

    function Frame{T}(species, positions, forces, velocities, cell, stress, virial, pbc, energy, time, extras) where {T}
        n = length(species)
        size(positions) == (3, n) ||
            throw(ArgumentError(lazy"pureatomistic: positions must be 3×$n for $n species, got $(size(positions))"))
        for (name, a) in (("forces", forces), ("velocities", velocities))
            size(a) in ((3, n), (3, 0)) ||
                throw(ArgumentError(lazy"pureatomistic: $name must be 3×$n or empty (3×0), got $(size(a))"))
        end
        for (name, a) in (("cell", cell), ("stress", stress), ("virial", virial))
            size(a) in ((3, 3), (0, 0)) ||
                throw(ArgumentError(lazy"pureatomistic: $name must be 3×3 or empty (0×0), got $(size(a))"))
        end
        return new{T}(
            collect(String, species), positions, forces, velocities, cell, stress, virial,
            pbc, energy, time, collect(Pair{String, String}, extras)
        )
    end
end

# A dense copy of `A` in element type `T` whatever `A`'s axes; an existing `Matrix{T}` is kept.
_dense(::Type{T}, A::Matrix{T}) where {T} = A
function _dense(::Type{T}, A::AbstractMatrix) where {T}
    M = Matrix{T}(undef, size(A))
    _copy_dense!(M, A)
    return M
end

# Copy by position, so the axes of `src` need not start at 1.
function _copy_dense!(dst::Matrix, src::AbstractMatrix)
    size(dst) == size(src) || throw(DimensionMismatch(lazy"pureatomistic: cannot copy $(size(src)) into $(size(dst))"))
    for (q, j) in enumerate(axes(src, 2)), (p, i) in enumerate(axes(src, 1))
        dst[p, q] = src[i, j]
    end
    return dst
end

function Frame{T}(
        species, positions;
        forces = zeros(T, 3, 0), velocities = zeros(T, 3, 0), cell = zeros(T, 0, 0),
        stress = zeros(T, 0, 0), virial = zeros(T, 0, 0),
        pbc = size(cell, 1) == 3 ? (true, true, true) : (false, false, false),
        energy = NaN, time = NaN, extras = Pair{String, String}[]
    ) where {T}
    return Frame{T}(
        species, _dense(T, positions), _dense(T, forces), _dense(T, velocities),
        _dense(T, cell), _dense(T, stress), _dense(T, virial), NTuple{3, Bool}(pbc), convert(T, energy), convert(T, time), extras
    )
end

Frame(species, positions::AbstractMatrix{T}; kwargs...) where {T <: AbstractFloat} =
    Frame{T}(species, positions; kwargs...)

# ---------------------------------------------------------------------------------------------
# Writer

_has_space(s) = any(c -> isspace(c) || c == '=' || c == '"', s)

function _check_token(what, s)
    (isempty(s) || any(c -> isspace(c) || c == '=' || c == '"', s)) &&
        throw(ArgumentError(lazy"extxyz: $what must be non-empty without whitespace, '=' or '\"', got \"$s\""))
    return s
end

function _write_value(io::Base.IO, key::AbstractString, value::AbstractString)
    (occursin('"', value) || occursin('\n', value) || occursin('\r', value)) &&
        throw(ArgumentError(lazy"extxyz: value of '$key' must not contain a quote or a line break"))
    print(io, ' ', key, '=')
    if isempty(value) || _has_space(value)
        print(io, '"', value, '"')
    else
        print(io, value)
    end
    return nothing
end

# Float64 prints as its shortest representation, which `parse(Float64, _)` reads back exactly.
function _write_numbers(io::Base.IO, a::AbstractMatrix)
    first = true
    for x in a
        first || print(io, ' ')
        print(io, Float64(x))
        first = false
    end
    return nothing
end

function _write_array_value(io::Base.IO, key::AbstractString, a::AbstractMatrix)
    print(io, ' ', key, "=\"")
    _write_numbers(io, a)
    print(io, '"')
    return nothing
end

"""
    write_extxyz(io::Base.IO, frame::Frame)

Append `frame` to `io` in extended XYZ. Floats are written as Float64 with the shortest digits
that read back exactly, so `read_extxyz` returns the stored values bit for bit (a `Float32`
frame comes back widened to Float64). Always writes `pbc`; writes `Lattice`, `energy`,
`stress`, `virial` and `Time` only when present.
"""
function write_extxyz(io::Base.IO, frame::Frame)
    n = length(frame.species)
    has_f = !isempty(frame.forces)
    has_v = !isempty(frame.velocities)
    println(io, n)
    first_field = true
    if !isempty(frame.cell)
        print(io, "Lattice=\"")
        _write_numbers(io, frame.cell)
        print(io, '"')
        first_field = false
    end
    print(io, first_field ? "" : " ", "Properties=species:S:1:pos:R:3")
    has_f && print(io, ":forces:R:3")
    has_v && print(io, ":velocities:R:3")
    isnan(frame.energy) || (print(io, " energy="); print(io, Float64(frame.energy)))
    isempty(frame.stress) || _write_array_value(io, "stress", frame.stress)
    isempty(frame.virial) || _write_array_value(io, "virial", frame.virial)
    print(io, " pbc=\"", frame.pbc[1] ? 'T' : 'F', ' ', frame.pbc[2] ? 'T' : 'F', ' ', frame.pbc[3] ? 'T' : 'F', '"')
    isnan(frame.time) || (print(io, " Time="); print(io, Float64(frame.time)))
    for (k, v) in frame.extras
        _check_token("key", k)
        _write_value(io, k, v)
    end
    println(io)
    for j in 1:n
        print(io, _check_token("species name", frame.species[j]))
        for i in 1:3
            print(io, ' ', Float64(frame.positions[i, j]))
        end
        if has_f
            for i in 1:3
                print(io, ' ', Float64(frame.forces[i, j]))
            end
        end
        if has_v
            for i in 1:3
                print(io, ' ', Float64(frame.velocities[i, j]))
            end
        end
        println(io)
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Reader

_skipws(s::AbstractString, i::Int) = (
    while i <= lastindex(s) && isspace(s[i])
        i = nextind(s, i)
    end; i
)

# Split a comment line into `key => value` pairs. A value is a run of non-blank characters or a
# double-quoted string; a key without `=` is a boolean `T`. Blanks around `=` are allowed.
function _comment_pairs(s::AbstractString, lineno::Int)
    out = Pair{String, String}[]
    e = lastindex(s)
    i = _skipws(s, firstindex(s))
    while i <= e
        j = i
        while j <= e && !isspace(s[j]) && s[j] != '='
            j = nextind(s, j)
        end
        j == i && throw(ArgumentError(lazy"extxyz: line $lineno: '=' without a key"))
        key = String(s[i:prevind(s, j)])
        k = _skipws(s, j)
        if k <= e && s[k] == '='
            k = _skipws(s, nextind(s, k))
            k > e && throw(ArgumentError(lazy"extxyz: line $lineno: key '$key' has no value"))
            if s[k] == '"'
                m = nextind(s, k)
                start = m
                while m <= e && s[m] != '"'
                    m = nextind(s, m)
                end
                m > e && throw(ArgumentError(lazy"extxyz: line $lineno: unterminated quote in the value of '$key'"))
                val = String(s[start:prevind(s, m)])
                i = nextind(s, m)
            else
                m = k
                while m <= e && !isspace(s[m])
                    m = nextind(s, m)
                end
                val = String(s[k:prevind(s, m)])
                i = m
            end
        else
            val = "T"
            i = j
        end
        push!(out, key => val)
        i = _skipws(s, i)
    end
    return out
end

function _parse_float(s::AbstractString, what, lineno::Int)
    x = tryparse(Float64, s)
    x === nothing && throw(ArgumentError(lazy"extxyz: line $lineno: cannot parse \"$s\" as a number in $what"))
    return x
end

function _parse_bool(s::AbstractString, key, lineno::Int)
    s in ("T", "True", "true", "TRUE") && return true
    s in ("F", "False", "false", "FALSE") && return false
    throw(ArgumentError(lazy"extxyz: line $lineno: cannot parse \"$s\" as a boolean in '$key'"))
end

function _parse_matrix3(s::AbstractString, key, lineno::Int)
    parts = split(s)
    length(parts) == 9 ||
        throw(ArgumentError(lazy"extxyz: line $lineno: '$key' needs 9 numbers, got $(length(parts))"))
    m = Matrix{Float64}(undef, 3, 3)
    for (q, p) in enumerate(parts)
        m[q] = _parse_float(p, key, lineno)
    end
    return m
end

# Columns of the atom lines: a name, a type letter and a width per `Properties` entry.
# Returns (species_col, pos_col, forces_col, velocities_col, ncols), zero for an absent column.
function _parse_properties(s::AbstractString, lineno::Int)
    parts = split(s, ':')
    length(parts) % 3 == 0 && !isempty(parts) ||
        throw(ArgumentError(lazy"extxyz: line $lineno: Properties must be name:type:count triplets, got \"$s\""))
    cs = cp = cf = cv = 0
    col = 1
    for t in 1:3:length(parts)
        name, typ, cnt = parts[t], parts[t + 1], tryparse(Int, parts[t + 2])
        cnt === nothing && throw(ArgumentError(lazy"extxyz: line $lineno: bad column count \"$(parts[t + 2])\" for '$name'"))
        want_typ, want_cnt = name == "species" ? ("S", 1) : name in ("pos", "forces", "velocities") ? ("R", 3) : ("", 0)
        isempty(want_typ) && throw(ArgumentError(lazy"extxyz: line $lineno: unsupported property column '$name'"))
        (typ == want_typ && cnt == want_cnt) ||
            throw(ArgumentError(lazy"extxyz: line $lineno: property '$name' must be $want_typ:$want_cnt, got $typ:$cnt"))
        seen = name == "species" ? cs : name == "pos" ? cp : name == "forces" ? cf : cv
        seen == 0 || throw(ArgumentError(lazy"extxyz: line $lineno: property '$name' is listed twice"))
        name == "species" ? (cs = col) : name == "pos" ? (cp = col) : name == "forces" ? (cf = col) : (cv = col)
        col += cnt
    end
    (cs > 0 && cp > 0) ||
        throw(ArgumentError(lazy"extxyz: line $lineno: Properties must include species and pos"))
    return (cs, cp, cf, cv, col - 1)
end

function _read_frame(lines::Vector{SubString{String}}, i::Int)
    n = tryparse(Int, strip(lines[i]))
    (n === nothing || n < 0) &&
        throw(ArgumentError(lazy"extxyz: line $i: expected an atom count, got \"$(strip(lines[i]))\""))
    i + 1 + n <= length(lines) ||
        throw(ArgumentError(lazy"extxyz: line $i: frame of $n atoms is truncated"))
    cl = i + 1
    pairs = _comment_pairs(rstrip(lines[cl]), cl)
    cell = zeros(0, 0); stress = zeros(0, 0); virial = zeros(0, 0)
    pbc = nothing; energy = NaN; time = NaN
    props = "species:S:1:pos:R:3"
    extras = Pair{String, String}[]
    seen = String[]
    for (k, v) in pairs
        if k in ("Lattice", "Properties", "pbc", "energy", "stress", "virial", "Time")
            k in seen && throw(ArgumentError(lazy"extxyz: line $cl: key '$k' appears twice"))
            push!(seen, k)
        end
        if k == "Lattice"
            cell = _parse_matrix3(v, k, cl)
        elseif k == "Properties"
            props = v
        elseif k == "pbc"
            b = split(v)
            length(b) == 3 || throw(ArgumentError(lazy"extxyz: line $cl: 'pbc' needs 3 booleans, got $(length(b))"))
            pbc = (_parse_bool(b[1], k, cl), _parse_bool(b[2], k, cl), _parse_bool(b[3], k, cl))
        elseif k == "energy"
            energy = _parse_float(v, k, cl)
        elseif k == "stress"
            stress = _parse_matrix3(v, k, cl)
        elseif k == "virial"
            virial = _parse_matrix3(v, k, cl)
        elseif k == "Time"
            time = _parse_float(v, k, cl)
        else
            push!(extras, k => v)
        end
    end
    isnan(energy) && "energy" in seen && throw(ArgumentError(lazy"extxyz: line $cl: 'energy' is NaN"))
    isnan(time) && "Time" in seen && throw(ArgumentError(lazy"extxyz: line $cl: 'Time' is NaN"))
    cs, cp, cf, cv, ncols = _parse_properties(props, cl)
    species = Vector{String}(undef, n)
    pos = Matrix{Float64}(undef, 3, n)
    forces = cf > 0 ? Matrix{Float64}(undef, 3, n) : zeros(3, 0)
    vel = cv > 0 ? Matrix{Float64}(undef, 3, n) : zeros(3, 0)
    for j in 1:n
        ln = cl + j
        f = split(lines[ln])
        length(f) == ncols ||
            throw(ArgumentError(lazy"extxyz: line $ln: expected $ncols columns, got $(length(f))"))
        species[j] = String(f[cs])
        for q in 1:3
            pos[q, j] = _parse_float(f[cp + q - 1], "pos", ln)
            cf > 0 && (forces[q, j] = _parse_float(f[cf + q - 1], "forces", ln))
            cv > 0 && (vel[q, j] = _parse_float(f[cv + q - 1], "velocities", ln))
        end
    end
    pbc === nothing && (pbc = isempty(cell) ? (false, false, false) : (true, true, true))
    frame = Frame{Float64}(species, pos, forces, vel, cell, stress, virial, pbc, energy, time, extras)
    return frame, cl + n + 1
end

"""
    read_extxyz(path_or_io) -> Vector{Frame{Float64}}

Read every frame of an extended XYZ file. Known keys fill `Frame` fields (`Lattice`,
`Properties`, `pbc`, `energy`, `stress`, `virial`, `Time`); other pairs are kept in `extras`.
Property columns are `species`, `pos`, `forces` and `velocities`. Without `Lattice`, `pbc`
defaults to all false, otherwise to all true. Malformed input throws an `ArgumentError`
whose message starts with `"extxyz:"` and names the line.
"""
function read_extxyz(io::Base.IO)
    lines = split(read(io, String), '\n')
    isempty(last(lines)) && pop!(lines)
    frames = Frame{Float64}[]
    i = 1
    while i <= length(lines)
        if all(l -> all(isspace, l), @view lines[i:end])
            break
        end
        frame, i = _read_frame(lines, i)
        push!(frames, frame)
    end
    isempty(frames) && throw(ArgumentError("extxyz: input holds no frames"))
    return frames
end

read_extxyz(path::AbstractString) = open(read_extxyz, path)
