"""
    XYZFrame

One frame of an xyz file: its `atoms` (positions in bohr) and the `comment` of its second line,
with the surrounding whitespace stripped.
"""
struct XYZFrame
    atoms::Vector{Atom{Float64}}
    comment::String
end

"""
    read_xyz(io::Base.IO; source = "<stream>", lenient = false, warnings = InputWarning[]) -> Vector{XYZFrame}

Every frame of the xyz text in `io`. A frame is a line with the atom count, a comment line (line 2
of the frame, even when blank), and one `symbol x y z` line per atom with the coordinates in Å,
which are converted to bohr; columns after the fourth are ignored. Blank lines between the atom
lines and between frames are skipped.

Errors are `InputError`s naming `source` and the line. In lenient mode (`lenient = true`) a frame
with fewer atom lines than its count, an unknown element symbol and a non-finite coordinate are
read as the oracle reads them, and an `InputWarning` is pushed onto `warnings` for each.

Lenient mode does not reproduce the oracle for a frame with more atom lines than its count: the
oracle reads one frame and ignores the lines after it, while this reader reads the next line as
the count of the next frame, an error in both modes when it is not one.
"""
function read_xyz(
        io::Base.IO; source::AbstractString = "<stream>", lenient::Bool = false,
        warnings::Vector{InputWarning} = InputWarning[],
    )
    frames = XYZFrame[]
    # The line the reader expects next: an atom count (`:count`), a comment or an atom line.
    expect = :count
    count_line = 0
    natoms = 0
    comment = ""
    atoms = Atom{Float64}[]
    for (n, raw) in enumerate(eachline(io))
        line, tokens = input_line(raw, n)
        if expect === :comment
            comment = String(line)
            atoms = Atom{Float64}[]
            expect = natoms == 0 ? :count : :atom
            natoms == 0 && push!(frames, XYZFrame(atoms, comment))
            continue
        end
        isempty(tokens) && continue
        if expect === :count
            natoms = something(length(tokens) == 1 ? tryparse(Int, tokens[1]) : nothing, -1)
            natoms < 0 && throw(InputError(source, n, "xyz: expected the atom count"))
            count_line = n
            expect = :comment
        else
            length(tokens) >= 4 ||
                throw(InputError(source, n, "xyz: expected a symbol and three coordinates"))
            p = SVector(
                _xyz_coordinate(source, n, lenient, warnings, tokens[2]),
                _xyz_coordinate(source, n, lenient, warnings, tokens[3]),
                _xyz_coordinate(source, n, lenient, warnings, tokens[4]),
            )
            push!(atoms, _atom("xyz: ", source, n, lenient, warnings, tokens[1], p))
            if length(atoms) == natoms
                push!(frames, XYZFrame(atoms, comment))
                expect = :count
            end
        end
    end
    # A file that ends inside a frame: after its count line, or before its last atom line.
    if expect !== :count
        if expect === :comment
            comment = ""
            atoms = Atom{Float64}[]
        end
        length(atoms) < natoms && reject_or_warn(
            source, count_line, lenient, warnings, :xyz_count,
            lazy"xyz: the frame declares $(natoms) atoms but has $(length(atoms))"
        )
        push!(frames, XYZFrame(atoms, comment))
    end
    isempty(frames) && throw(InputError(source, 0, "xyz: the file has no frame"))
    return frames
end

# The coordinate token `t` of line `n`, converted from Å to bohr; in lenient mode a non-finite
# value (`nan`, `inf`, or a decimal beyond the `Float64` range) is kept.
function _xyz_coordinate(
        source::AbstractString, n::Int, lenient::Bool, warnings::Vector{InputWarning}, t::SubString{String},
    )
    x = _parse_coordinate(t)
    isnothing(x) && throw(InputError(source, n, lazy"xyz: coordinate \"$(t)\" is not a number"))
    isfinite(x) || reject_or_warn(
        source, n, lenient, warnings, :nonfinite, lazy"xyz: expected a finite number, got \"$(t)\""
    )
    return x * ANGSTROM_TO_BOHR
end

"""
    write_xyz(io::Base.IO, atoms::AbstractVector{<:Atom}; comment::AbstractString = "")

Writes one xyz frame: the atom count, the `comment` line, and `<symbol> <x> <y> <z>` per atom
with the coordinates in Å, each the shortest decimal that reads back as the same `Float64`,
separated by single spaces. Throws an `ArgumentError` for a `comment` with a line break or with
leading or trailing whitespace, neither of which `read_xyz` reads back.
"""
function write_xyz(io::Base.IO, atoms::AbstractVector{<:Atom}; comment::AbstractString = "")
    (occursin('\n', comment) || occursin('\r', comment)) &&
        throw(ArgumentError("io: an xyz comment is one line"))
    (!isempty(comment) && (isspace(first(comment)) || isspace(last(comment)))) &&
        throw(ArgumentError("io: an xyz comment has no leading or trailing whitespace"))
    print(io, length(atoms))
    write(io, '\n')
    write(io, comment)
    write(io, '\n')
    for i in eachindex(atoms)
        a = atoms[i]
        write(io, a.symbol)
        for k in eachindex(a.position)
            write(io, ' ')
            print(io, Float64(a.position[k]) * BOHR_TO_ANGSTROM)
        end
        write(io, '\n')
    end
    return nothing
end
