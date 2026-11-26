"""
    input_format(path::AbstractString) -> InputFormat

The format of the input file `path`, by its extension compared without regard to case: `xyz`,
`pdb` and `cif` are the molecule files `XYZ`, `PDB` and `CIF`, `yaml` and `yml` are `YAML_INPUT`,
`json` is `JSON_INPUT`, and any other extension, or none, is `INP`. The extension is the part of
the last path component after its last `.`; a component whose only `.` is its first character
(`.json`) has none.
"""
input_format(path::AbstractString) = _format_of(String(path))
input_format(path::SubString{String}) = _format_of(path)

# The format of the UTF-8 path `s`; a `SubString` of a `String` is read in place, without a copy.
function _format_of(s::Union{String, SubString{String}})
    # The last `.` of the last path component, found by code unit: `basename` and `findlast` box
    # or make calls the trim verifier cannot resolve. In UTF-8 the bytes of `.`, `/` and `\` occur
    # only as those characters.
    i = ncodeunits(s)
    while i >= 1 && codeunit(s, i) != UInt8('.') && !_is_separator(codeunit(s, i))
        i -= 1
    end
    # No `.`, or one that starts the component.
    (i <= 1 || codeunit(s, i) != UInt8('.') || _is_separator(codeunit(s, i - 1))) && return INP
    _extension_is(s, i, "xyz") && return XYZ
    _extension_is(s, i, "pdb") && return PDB
    _extension_is(s, i, "cif") && return CIF
    (_extension_is(s, i, "yaml") || _extension_is(s, i, "yml")) && return YAML_INPUT
    _extension_is(s, i, "json") && return JSON_INPUT
    return INP
end

# Whether the byte `b` separates path components, as for `basename`.
_is_separator(b::UInt8) = b == UInt8('/') || (Sys.iswindows() && b == UInt8('\\'))

# Whether the code units of `s` after index `i` spell `word`, a lowercase ASCII word, without regard
# to case. `b | 0x20` lowercases an ASCII letter and maps no other byte to one. Unicode lowercasing,
# which the oracle applies to the extension, gives an ASCII letter for two other characters only:
# U+212A (KELVIN SIGN) becomes `k`, which no extension word holds, and U+0130 becomes `i` followed by
# U+0307. So this agrees with the oracle's comparison of the lowercased extension.
function _extension_is(s::Union{String, SubString{String}}, i::Int, word::String)
    ncodeunits(s) - i == ncodeunits(word) || return false
    for k in 1:ncodeunits(word)
        (codeunit(s, i + k) | 0x20) == codeunit(word, k) || return false
    end
    return true
end

"""
    parse_input(path::AbstractString; lenient::Bool = false) -> ParsedInput

The input file `path`, read by the reader of its `input_format`, with `path` as the source of its
errors and warnings: `parse_inp`, `parse_json`, or `parse_yaml`, which needs YAML.jl loaded (an
`InputError` in both modes without it). A molecule file (xyz, PDB or CIF) gives an `InputFile`
holding its atoms and every other field at its default, as the oracle reads it; the checks of the
other formats (that the input has atoms, and that its charge and multiplicity fit its electron
count) are not applied to it, except that strict reading rejects a file of no atoms (lenient
reading keeps it, with an `:xyz_no_atoms` warning). An xyz file must hold one frame: of several,
strict reading throws and lenient reading takes the first, with an `:xyz_frames` warning.
"""
function parse_input(path::AbstractString; lenient::Bool = false)
    source = String(path)
    fmt = input_format(source)
    fmt == INP && return _inp_file(source, lenient)
    text = _read_file(source)
    fmt == JSON_INPUT && return parse_json(text; source, lenient)
    if fmt == YAML_INPUT
        hasmethod(parse_yaml, Tuple{String}) || throw(_yaml_unavailable(source))
        return parse_yaml(text; source, lenient)::ParsedInput
    end
    return _molecule_input(fmt, text, source, lenient)
end

# The `.inp` file `source`. It is a function of its own so that the path of an `.inp` file through
# `parse_input` can be trim-checked apart from the JSON reader, which is not trim-safe.
_inp_file(source::String, lenient::Bool) = parse_inp(_read_file(source); source, lenient)

# The contents of the file `source`. It reads by `readbytes!`, since `read(path, String)` boxes the
# `Union` field of the `StatStruct` it builds.
function _read_file(source::String)
    data = UInt8[]
    chunk = Vector{UInt8}(undef, 65536)
    io = open(source)
    try
        while !eof(io)
            k = readbytes!(io, chunk)
            append!(data, view(chunk, 1:k))
        end
    finally
        close(io)
    end
    return String(data)
end

# The input of the molecule file `text` of format `fmt`: its atoms, every other setting at its
# default.
function _molecule_input(fmt::InputFormat, text::String, source::String, lenient::Bool)
    warnings = InputWarning[]
    io = IOBuffer(text)
    atoms = if fmt == XYZ
        frames = read_xyz(io; source, lenient, warnings)
        n = length(frames)
        n == 1 || reject_or_warn(
            source, 0, lenient, warnings, :xyz_frames,
            lazy"xyz: the file has $(n) frames; an input geometry needs exactly one",
        )
        first(frames).atoms
    elseif fmt == PDB
        read_pdb(io; source, lenient, warnings)
    else
        read_cif(io; source, lenient, warnings)
    end
    # The oracle accepts an xyz frame of no atoms (PDB and CIF readers reject a file of none).
    isempty(atoms) && reject_or_warn(
        source, 0, lenient, warnings, :xyz_no_atoms, "no atoms: the molecule file has no atoms",
    )
    return ParsedInput(InputFile(; atoms), warnings)
end
