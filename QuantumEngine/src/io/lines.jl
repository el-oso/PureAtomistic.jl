"""
    input_line(raw::AbstractString, n::Int) -> (line, tokens)

Line `n` of an input file, cleaned: a leading UTF-8 byte-order mark removed when `n == 1`, a
trailing `\\r` removed and the surrounding whitespace stripped. `tokens` are the pieces of `line`
between spaces and tabs. Both are `SubString{String}`s. Every reader and parser cleans its lines
here, so CRLF files, a byte-order mark and tab separators read as the LF, space-separated file.
"""
function input_line(raw::AbstractString, n::Int)
    s = raw isa SubString{String} ? raw : SubString(String(raw))
    n == 1 && (s = chopprefix(s, "\ufeff"))
    line = strip(chopsuffix(s, "\r"))
    return line, split(line, (' ', '\t'); keepempty = false)
end

"""
    reject_or_warn(source, line, lenient, warnings, case, msg) -> nothing

The one choice between strict and lenient reading. Strict (`lenient == false`) throws
`InputError(source, line, msg)`. Lenient pushes `InputWarning(source, line, case, msg)` onto
`warnings` and returns, and the caller then takes the value the oracle takes.
"""
function reject_or_warn(
        source::AbstractString, line::Int, lenient::Bool, warnings::Vector{InputWarning}, case::Symbol,
        msg::AbstractString,
    )
    lenient || throw(InputError(source, line, msg))
    push!(warnings, InputWarning(source, line, case, msg))
    return nothing
end

# The atom `symbol` at `position` (bohr), read on line `line` of a file of format `fmt`. An
# unknown element symbol is rejected or, in lenient mode, read as the oracle reads it: Z = 1 and
# the mass of ¹H, with the symbol as written. The `ArgumentError` of `System.element_number` for a
# bare `Bq` is rethrown as an `InputError` of that line, its `system: ` prefix replaced by `fmt`.
#
# The symbol is looked up before `Atom` is called, because a `catch` that returns would make the
# throw sites inside `Atom` reachable, and those box their arguments; the `catch` here only
# throws. Once the lookup has passed, `Atom` throws nothing for a 3-vector position and no mass,
# so any error it does throw propagates unchanged.
function _atom(
        fmt::String, source::AbstractString, line::Int, lenient::Bool, warnings::Vector{InputWarning},
        symbol::AbstractString, position::SVector{3, Float64},
    )
    z = try
        System.element_number(symbol)
    catch e
        e isa ArgumentError || rethrow()
        m = e.msg
        m isa String || rethrow()
        body = chopprefix(m, "system: ")
        throw(InputError(source, line, lazy"$(fmt)$(body)"))
    end
    if z == 0
        reject_or_warn(
            source, line, lenient, warnings, :unknown_element, lazy"$(fmt)unknown element symbol \"$(symbol)\""
        )
        return Atom{Float64}(1, String(symbol), position, false, element(1).mass_most_abundant)
    end
    return Atom(symbol, position)
end
