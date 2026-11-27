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
            source, line, lenient, warnings, :unknown_element, lazy"$(fmt)unknown element symbol $(repr(symbol))"
        )
        return Atom{Float64}(1, String(symbol), position, false, element(1).mass_most_abundant)
    end
    return Atom(symbol, position)
end

"""
    parse_coordinate(t::AbstractString) -> Union{Nothing,Float64}

The value of the number token `t` as Rust's `str::parse::<f64>`, which the oracle uses, reads it,
or `nothing` for a token it rejects. A decimal beyond the `Float64` range reads as ±Inf and one
below half the smallest subnormal as ±0.0, where `tryparse(Float64, t)` returns `nothing` for
both; a hexadecimal float, which `tryparse` reads and Rust rejects, is rejected. The xyz, PDB,
CIF and Molden readers and the `.inp` parser read their numbers with it.
"""
function parse_coordinate(t::AbstractString)
    n = ncodeunits(t)
    i = 1
    neg = false
    if i <= n && (codeunit(t, i) == UInt8('+') || codeunit(t, i) == UInt8('-'))
        neg = codeunit(t, i) == UInt8('-')
        i += 1
    end
    i + 1 <= n && codeunit(t, i) == UInt8('0') && (codeunit(t, i + 1) | 0x20) == UInt8('x') && return nothing
    x = tryparse(Float64, t)
    x === nothing || return x
    # `tryparse` rejected `t`, which is then out of range if it is a decimal
    # `digits[.digits][(e|E)[+|-]digits]` with at least one mantissa digit.
    ndig = 0      # mantissa digits
    nint = 0      # of which before the point
    first_nz = -1 # index among the mantissa digits of the first nonzero one
    point = false
    while i <= n
        c = codeunit(t, i)
        if c == UInt8('.') && !point
            point = true
        elseif UInt8('0') <= c <= UInt8('9')
            first_nz < 0 && c != UInt8('0') && (first_nz = ndig)
            ndig += 1
            point || (nint += 1)
        else
            break
        end
        i += 1
    end
    (ndig == 0 || first_nz < 0) && return nothing
    e = 0
    if i <= n && (codeunit(t, i) | 0x20) == UInt8('e')
        i += 1
        eneg = false
        if i <= n && (codeunit(t, i) == UInt8('+') || codeunit(t, i) == UInt8('-'))
            eneg = codeunit(t, i) == UInt8('-')
            i += 1
        end
        i <= n || return nothing
        while i <= n
            c = codeunit(t, i)
            UInt8('0') <= c <= UInt8('9') || return nothing
            # Saturates: any exponent this large is out of range either way.
            e = min(10 * e + Int(c - UInt8('0')), 10^15)
            i += 1
        end
        eneg && (e = -e)
    end
    i > n || return nothing
    # The decimal exponent of the first nonzero digit: ≥ 0 means |value| ≥ 1, so an overflow.
    lead = nint - 1 - first_nz + e
    return lead >= 0 ? (neg ? -Inf : Inf) : (neg ? -0.0 : 0.0)
end
