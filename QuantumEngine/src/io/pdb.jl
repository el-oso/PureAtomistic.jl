"""
    read_pdb(io::Base.IO; source = "<stream>", lenient = false, warnings = InputWarning[]) -> Vector{Atom{Float64}}

The atoms of the `ATOM  ` and `HETATM` records of the PDB text in `io`; other records are
ignored. The coordinates are the Å fields of columns 31–38, 39–46 and 47–54, converted to bohr,
and the element is read from columns 77–78, right- or left-justified.

Errors are `InputError`s naming `source` and the line; a file without a record is an error. In
lenient mode (`lenient = true`) a record shorter than 54 columns is skipped; a record without an
element in columns 77–78 takes the first character of its atom name (columns 13–16) when it is
shorter than 78 columns, and the empty symbol with Z = 1 when it is 78 columns or longer with
those columns blank; an unknown element symbol is read as Z = 1, and a non-finite coordinate is
kept. These are the oracle's values, and an `InputWarning` is pushed onto `warnings` for each.
"""
function read_pdb(
        io::Base.IO; source::AbstractString = "<stream>", lenient::Bool = false,
        warnings::Vector{InputWarning} = InputWarning[],
    )
    atoms = Atom{Float64}[]
    for (n, raw) in enumerate(eachline(io))
        line, _ = input_line(raw, n)
        record = startswith(line, "ATOM  ") ? "ATOM" : startswith(line, "HETATM") ? "HETATM" : ""
        isempty(record) && continue
        # Columns are code units; a non-ASCII character would shift them.
        isascii(line) || throw(InputError(source, n, "pdb: a record has a character outside ASCII"))
        if ncodeunits(line) < 54
            reject_or_warn(
                source, n, lenient, warnings, :pdb_short_line, lazy"pdb: $(record) record shorter than 54 columns"
            )
            continue
        end
        p = SVector(
            _pdb_coordinate(source, n, lenient, warnings, line, 31),
            _pdb_coordinate(source, n, lenient, warnings, line, 39),
            _pdb_coordinate(source, n, lenient, warnings, line, 47),
        )
        # `line` has lost its trailing blanks, so columns past its end are blank.
        w = ncodeunits(line)
        symbol = w >= 77 ? strip(SubString(line, 77, min(78, w))) : SubString(line, 1, 0)
        if isempty(symbol)
            reject_or_warn(source, n, lenient, warnings, :pdb_element_columns, "pdb: no element in columns 77–78")
            # The oracle reads blank columns 77–78 of a record of 78 columns or more, counted with
            # its trailing blanks, as the empty symbol, Z = 1, and takes the first character of the
            # atom name for a shorter record.
            if w + ncodeunits(raw) - ncodeunits(rstrip(raw)) >= 78
                push!(atoms, Atom{Float64}(1, "", p, false, element(1).mass_most_abundant))
                continue
            end
            name = strip(SubString(line, 13, 16))
            symbol = isempty(name) ? SubString("H") : SubString(name, 1, 1)
        end
        push!(atoms, _atom("pdb: ", source, n, lenient, warnings, symbol, p))
    end
    isempty(atoms) && throw(InputError(source, 0, "pdb: no ATOM or HETATM records"))
    return atoms
end

# The coordinate in columns `c:c+7` of the ASCII record `line` (line `n`), converted from Å to bohr;
# in lenient mode a non-finite value (`nan`, `inf`, or a decimal beyond the `Float64` range) is
# kept.
function _pdb_coordinate(
        source::AbstractString, n::Int, lenient::Bool, warnings::Vector{InputWarning}, line::SubString{String},
        c::Int,
    )
    t = strip(SubString(line, c, c + 7))
    x = _parse_coordinate(t)
    isnothing(x) && throw(InputError(source, n, lazy"pdb: coordinate \"$(t)\" is not a number"))
    isfinite(x) || reject_or_warn(
        source, n, lenient, warnings, :nonfinite, lazy"pdb: expected a finite number, got \"$(t)\""
    )
    return x * ANGSTROM_TO_BOHR
end
