"""
    Verbosity

The output levels `Silent < Low < Normal < High`. A line of level `level` is written when the
logger's verbosity is at least `level`.
"""
@enum Verbosity Silent = 0 Low = 1 Normal = 2 High = 3

"""
    verbosity_from_input(v::Integer) -> Verbosity

The logger level of the input keyword `verbosity`, as the oracle maps it: 0 is `Low`, 1 is
`Normal` and 2 or more is `High`. Throws an `ArgumentError` for a negative `v`.
"""
function verbosity_from_input(v::Integer)
    v >= 0 || throw(ArgumentError(lazy"io: verbosity must be ≥ 0, got $(v)"))
    return v == 0 ? Low : v == 1 ? Normal : High
end

"""
    Logger{A<:Base.IO, B<:Base.IO}

The output of a run: every line goes to the stream `out` (standard output in a run) and to the
log stream `log`, and both are flushed after each line.

    Logger(out, log; verbosity = Normal)

builds a logger that writes the lines of level `verbosity` and below. Pass `devnull` as `log`
for a run without a log file.
"""
struct Logger{A <: Base.IO, B <: Base.IO}
    out::A
    log::B
    verbosity::Verbosity
end

Logger(out::Base.IO, log::Base.IO; verbosity::Verbosity = Normal) = Logger(out, log, verbosity)

function _write_line(io::Base.IO, msg::AbstractString)
    write(io, msg)
    write(io, '\n')
    flush(io)
    return nothing
end

"""
    log_print(lg::Logger, level::Verbosity, msg::AbstractString) -> nothing

Writes the line `msg` to both streams of `lg` when `lg.verbosity >= level`, and flushes them.
"""
function log_print(lg::Logger, level::Verbosity, msg::AbstractString)
    lg.verbosity >= level || return nothing
    _write_line(lg.out, msg)
    _write_line(lg.log, msg)
    return nothing
end

"""
    log_success(lg::Logger, msg::AbstractString) -> nothing

Writes `"✓ " * msg` at level `Normal`.
"""
log_success(lg::Logger, msg::AbstractString) = log_print(lg, Normal, "✓ " * msg)

"""
    log_warning(lg::Logger, msg::AbstractString) -> nothing

Writes `"⚠ Warning: " * msg` at level `Normal`.
"""
log_warning(lg::Logger, msg::AbstractString) = log_print(lg, Normal, "⚠ Warning: " * msg)

"""
    log_error(lg::Logger, msg::AbstractString) -> nothing

Writes `"❌ Error: " * msg` at level `Low`.
"""
log_error(lg::Logger, msg::AbstractString) = log_print(lg, Low, "❌ Error: " * msg)

"""
    log_heading(lg::Logger, title::AbstractString) -> nothing

Writes `title` in upper case in a double-lined box, with a blank line before and after, at level
`Normal`. The box is `ncodeunits(title) + 4` characters wide inside: the oracle counts the bytes
of the title.
"""
function log_heading(lg::Logger, title::AbstractString)
    line = "═"^(ncodeunits(title) + 4)
    log_print(lg, Normal, "\n╔" * line * "╗")
    log_print(lg, Normal, "║  " * uppercase(title) * "  ║")
    log_print(lg, Normal, "╚" * line * "╝\n")
    return nothing
end

"""
    log_section(lg::Logger, title::AbstractString) -> nothing

Writes a blank line, `title`, a line of `-` as long as `title` has bytes, and a blank line, at
level `Normal`.
"""
function log_section(lg::Logger, title::AbstractString)
    log_print(lg, Normal, "\n" * title)
    log_print(lg, Normal, "-"^ncodeunits(title) * "\n")
    return nothing
end

"""
    print_atom_table(lg::Logger, atoms::AbstractVector{<:Atom}) -> nothing

Writes the oracle's atom table at level `Normal`: a header line, a rule, one row per atom with
its number (counting from 1 in iteration order), its symbol and its position in bohr and in Å,
and a blank line.
"""
function print_atom_table(lg::Logger, atoms::AbstractVector{<:Atom})
    log_print(
        lg, Normal,
        "  Atom         X [Bohr]       Y [Bohr]       Z [Bohr]      X [Ang]        Y [Ang]        Z [Ang]",
    )
    log_print(lg, Normal, "  " * "─"^95)
    for (i, a) in enumerate(atoms)
        row = "  " * fmt_int(i, 2) * " " * fmt_str(a.symbol, 4)
        for c in a.position
            row *= "   " * fmt_fixed(c, 12, 8)
        end
        for c in a.position
            row *= "   " * fmt_fixed(c * oftype(c, BOHR_TO_ANGSTROM), 12, 8)
        end
        log_print(lg, Normal, row)
    end
    log_print(lg, Normal, "")
    return nothing
end

"""
    print_total_energy(lg::Logger, e::Real) -> nothing

Writes the oracle's final energy line, `"  Total Energy:  "`, `e` in Hartree as `{:20.12}` and
`" Hartree"`, at level `Normal`.
"""
print_total_energy(lg::Logger, e::Real) =
    log_print(lg, Normal, "  Total Energy:  " * fmt_fixed(e, 20, 12) * " Hartree")

"""
    report_optimization_step(lg::Logger, step::Integer, e::Real, gnorm::Real) -> nothing

Writes the oracle's geometry-optimization line `"[OPT] Step {:3}: E = {:17.10} |grad| = {:.3e}"`
for step `step`, energy `e` and gradient norm `gnorm`, at level `Normal`.
"""
report_optimization_step(lg::Logger, step::Integer, e::Real, gnorm::Real) = log_print(
    lg, Normal, "[OPT] Step " * fmt_int(step, 3) * ": E = " * fmt_fixed(e, 17, 10) * " |grad| = " * fmt_exp(gnorm, 3)
)

# The path `path` without its extension, as Rust's `Path::with_extension("")` strips it: the
# extension starts at the last `.` of the file name, except at the file name's first character
# (`.hidden` has none). The scan reads bytes: `splitext` boxes (a runtime tuple index).
function _without_extension(path::AbstractString)
    s = String(path)
    k = ncodeunits(s)
    while k > 0
        c = codeunit(s, k)
        (c == UInt8('/') || (Sys.iswindows() && c == UInt8('\\'))) && break
        if c == UInt8('.')
            first_of_name = k == 1 || codeunit(s, k - 1) == UInt8('/') ||
                (Sys.iswindows() && codeunit(s, k - 1) == UInt8('\\'))
            first_of_name && break
            return String(SubString(s, 1, k - 1))
        end
        k -= 1
    end
    return s
end

"""
    log_path(input_path::AbstractString, output::Union{Nothing, String}) -> String

The log file of a run: `output` when the input names one, else `input_path` with its extension
replaced by `.log` (or `.log` appended when it has none), next to the input.
"""
log_path(input_path::AbstractString, output::Union{Nothing, String}) =
    isnothing(output) ? _without_extension(input_path) * ".log" : output

"""
    output_base(log_path::AbstractString) -> String

`log_path` without its extension: the base name of the other files a run writes.
"""
output_base(log_path::AbstractString) = _without_extension(log_path)
