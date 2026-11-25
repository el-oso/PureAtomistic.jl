"""
    read_cif(io::Base.IO; source = "<stream>", lenient = false, warnings = InputWarning[]) -> Vector{Atom{Float64}}

The atoms of the `_atom_site` loop of the CIF text in `io`: a `loop_` whose tags include
`_atom_site_type_symbol` and `_atom_site_Cartn_x`, `_y` and `_z` (in Å, converted to bohr), with
one whitespace-separated row per atom. Tags are matched without regard to case, and the
characters of the type symbol that are not letters are dropped (`Fe2+` is iron). Other loops, tag
lines outside a loop, and `#` comment lines are ignored; a tag line after a loop's rows, a new
`loop_` or a `data_` line ends the loop.

Errors are `InputError`s naming `source` and the line. A loop with fractional coordinates and no
Cartesian ones throws `"cif: only Cartesian _atom_site coordinates are supported"`, and a file
without an atom is an error. In lenient mode (`lenient = true`) an `_atom_site` loop with
coordinate tags but without the type symbol or one of the Cartesian coordinates is skipped, a row
whose value count differs from the tag count is skipped, a coordinate that is not a number reads
as 0.0, a non-finite coordinate is kept, and an unknown element symbol is read as Z = 1, as the
oracle reads them, and an `InputWarning` is pushed onto `warnings` for each.
"""
function read_cif(
        io::Base.IO; source::AbstractString = "<stream>", lenient::Bool = false,
        warnings::Vector{InputWarning} = InputWarning[],
    )
    atoms = Atom{Float64}[]
    tags = String[]
    in_loop = false
    in_rows = false
    # Whether the rows of the current loop are atoms, and the columns of the type symbol and of
    # the Cartesian x, y and z; set at the loop's first row.
    atom_rows = false
    sym, x, y, z = 0, 0, 0, 0
    for (n, raw) in enumerate(eachline(io))
        line, tokens = input_line(raw, n)
        (isempty(line) || startswith(line, '#')) && continue
        head = lowercase(tokens[1])
        if head == "loop_" || startswith(head, "data_")
            in_loop = head == "loop_"
            in_rows = false
            empty!(tags)
            continue
        end
        in_loop || continue
        if startswith(line, '_')
            if in_rows
                in_loop = false
            else
                push!(tags, head)
            end
            continue
        end
        if !in_rows
            in_rows = true
            atom_rows = _is_coordinate_loop(tags)
            if atom_rows
                sym, x, y, z = _cif_columns(source, n, tags)
                if iszero(sym) || iszero(x) || iszero(y) || iszero(z)
                    reject_or_warn(
                        source, n, lenient, warnings, :cif_columns,
                        "cif: the _atom_site loop needs _atom_site_type_symbol and _atom_site_Cartn_x, _y and _z",
                    )
                    atom_rows = false
                end
            end
        end
        atom_rows || continue
        if length(tokens) != length(tags)
            reject_or_warn(
                source, n, lenient, warnings, :cif_row,
                lazy"cif: row has $(length(tokens)) values for $(length(tags)) tags"
            )
            continue
        end
        p = SVector(
            _cif_coordinate(source, n, lenient, warnings, tokens[x]),
            _cif_coordinate(source, n, lenient, warnings, tokens[y]),
            _cif_coordinate(source, n, lenient, warnings, tokens[z]),
        )
        push!(atoms, _atom("cif: ", source, n, lenient, warnings, filter(isletter, tokens[sym]), p))
    end
    isempty(atoms) && throw(InputError(source, 0, "cif: no _atom_site loop with Cartesian coordinates"))
    return atoms
end

# Whether the lowercase loop tags `tags` are those of an atom-site loop with a type symbol or
# coordinates; other `_atom_site` loops (the anisotropic displacements) carry no atoms.
_is_coordinate_loop(tags::Vector{String}) = any(
    t -> t == "_atom_site_type_symbol" || startswith(t, "_atom_site_cartn_") || startswith(t, "_atom_site_fract_"),
    tags,
)

# The columns of the type symbol and of the Cartesian x, y and z in `tags`, each 0 when missing;
# throws, naming the first row (line `n`), for fractional coordinates without Cartesian ones.
function _cif_columns(source::AbstractString, n::Int, tags::Vector{String})
    sym = something(findfirst(==("_atom_site_type_symbol"), tags), 0)
    x = something(findfirst(==("_atom_site_cartn_x"), tags), 0)
    y = something(findfirst(==("_atom_site_cartn_y"), tags), 0)
    z = something(findfirst(==("_atom_site_cartn_z"), tags), 0)
    if iszero(x) || iszero(y) || iszero(z)
        any(t -> startswith(t, "_atom_site_fract_"), tags) &&
            throw(InputError(source, n, "cif: only Cartesian _atom_site coordinates are supported"))
    end
    return sym, x, y, z
end

# The coordinate token `t` of line `n`, converted from Å to bohr; in lenient mode a token that is
# not a number reads as 0.0 and a non-finite value (`nan`, `inf`) is kept.
function _cif_coordinate(
        source::AbstractString, n::Int, lenient::Bool, warnings::Vector{InputWarning}, t::SubString{String},
    )
    x = tryparse(Float64, t)
    if isnothing(x)
        reject_or_warn(source, n, lenient, warnings, :cif_coordinate, lazy"cif: coordinate \"$(t)\" is not a number")
        return 0.0
    end
    isfinite(x) || reject_or_warn(
        source, n, lenient, warnings, :nonfinite, lazy"cif: expected a finite number, got \"$(t)\""
    )
    return x * ANGSTROM_TO_BOHR
end
