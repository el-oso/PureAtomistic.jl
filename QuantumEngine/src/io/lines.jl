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
# the mass of ¹H, with the symbol as written. Any other `ArgumentError` of `Atom` is rethrown as an
# `InputError` of that line, its `system: ` prefix replaced by `fmt`.
#
# The unknown symbol is found before `Atom` is called, because a `catch` that returns would make
# the throw sites inside `Atom` reachable, and those box their arguments; the `catch` here only
# throws. The other error `Atom` throws for a 3-vector position and no mass is a bare `Bq`, whose
# message is a `String`; a `LazyString` message is kept whole, as rendering it here would box.
function _atom(
        fmt::String, source::AbstractString, line::Int, lenient::Bool, warnings::Vector{InputWarning},
        symbol::AbstractString, position::SVector{3, Float64},
    )
    if _unknown_element(symbol)
        reject_or_warn(
            source, line, lenient, warnings, :unknown_element, lazy"$(fmt)unknown element symbol \"$(symbol)\""
        )
        return Atom{Float64}(1, String(symbol), position, false, element(1).mass_most_abundant)
    end
    try
        return Atom(symbol, position)
    catch e
        e isa ArgumentError || rethrow()
        m = e.msg
        if m isa String
            body = chopprefix(m, "system: ")
            throw(InputError(source, line, lazy"$(fmt)$(body)"))
        end
        m isa LazyString || rethrow()
        throw(InputError(source, line, lazy"$(fmt)$(m)"))
    end
end

# Whether `symbol` names no element under the spelling `Atom` reads: `El` is element `El` and
# `El@`, `BqEl` and `Bq-El` are ghosts of `El`, each matched without regard to case. A bare `Bq`
# is not reported here: `Atom` rejects it with a message of its own.
function _unknown_element(symbol::AbstractString)
    n = ncodeunits(symbol)
    lo, hi = 1, n
    if n >= 1 && codeunit(symbol, n) == UInt8('@')
        hi = n - 1
    elseif n >= 2 && _lower(codeunit(symbol, 1)) == UInt8('b') && _lower(codeunit(symbol, 2)) == UInt8('q')
        lo = n >= 3 && codeunit(symbol, 3) == UInt8('-') ? 4 : 3
        lo > hi && return false
    end
    for z in 1:nelements()
        _spells(symbol, lo, hi, element(z).symbol) && return false
    end
    return true
end

# Whether the code units `lo:hi` of `s` spell the ASCII symbol `ref` in any case.
function _spells(s::AbstractString, lo::Int, hi::Int, ref::String)
    hi - lo + 1 == ncodeunits(ref) || return false
    for k in 1:ncodeunits(ref)
        _lower(codeunit(s, lo + k - 1)) == _lower(codeunit(ref, k)) || return false
    end
    return true
end

_lower(c::UInt8) = UInt8('A') <= c <= UInt8('Z') ? c + (UInt8('a') - UInt8('A')) : c
