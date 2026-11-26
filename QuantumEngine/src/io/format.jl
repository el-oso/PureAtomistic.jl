# Numbers and strings in the layouts of Rust's `format!`, the formatter of the oracle's output.
# Rust pads to a width counted in characters, numbers on the left and strings on the right, and
# never truncates. A non-finite number prints as `inf`, `-inf` or `NaN`.

_pad_left(s::String, width::Int) = " "^max(0, width - length(s)) * s
_pad_right(s::AbstractString, width::Int) = String(s) * " "^max(0, width - length(s))

_nonfinite(x::Real) = isnan(x) ? "NaN" : x > 0 ? "inf" : "-inf"

# The precisions of `fmt_fixed` and `fmt_exp`: every precision the oracle's format strings use.
# Each has its own `@sprintf` call with a literal precision in `_printf_fixed` and `_printf_exp`:
# a run-time precision (`%.*f`) makes Printf read its arguments from the mixed tuple
# `(Int, Float64)` at a run-time index, which boxes.
const FORMAT_PRECISIONS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15)

function _precision_error(prec::Int)
    return ArgumentError(lazy"io: precision must be one of 0 to 8, 10, 12 and 15, got $(prec)")
end

function _check_precision(prec::Int)
    prec in FORMAT_PRECISIONS || throw(_precision_error(prec))
    return nothing
end

for (name, conversion) in ((:_printf_fixed, 'f'), (:_printf_exp, 'e'))
    body = :(throw(_precision_error(prec)))
    for p in reverse(FORMAT_PRECISIONS)
        spec = "%.$(p)$(conversion)"
        body = :(prec == $p ? @sprintf($spec, x) : $body)
    end
    @eval $name(x::Real, prec::Int) = $body
end

"""
    fmt_fixed(x::Real, width::Int, prec::Int) -> String

`x` with `prec` digits after the decimal point, right-aligned in `width` characters, as Rust's
`{:width.prec}`: `fmt_fixed(-76.301896228438, 20, 12) == "    -76.301896228438"`. Throws an
`ArgumentError` for a `prec` not in `FORMAT_PRECISIONS` (0 to 8, 10, 12 and 15).

The digits come from Printf's `%.<prec>f`, which rounds the exact binary value. Where that value
is an exact decimal tie, both Printf and Rust round to even on the ties the item "Rust fixed,
integer and string formats" pins; other ties are not checked, and Rust may round them the
other way. The parity harness compares parsed numbers, not text, so such a difference is cosmetic.
"""
function fmt_fixed(x::Real, width::Int, prec::Int)
    _check_precision(prec)
    isfinite(x) || return _pad_left(_nonfinite(x), width)
    return _pad_left(_printf_fixed(x, prec), width)
end

"""
    fmt_exp(x::Real, prec::Int; width::Int = 0) -> String

`x` in scientific notation with `prec` digits after the decimal point, right-aligned in `width`
characters, as Rust's `{:width.prece}`: the exponent has no `+` and no leading zeros, as in
`5.369e-1`, `0.000e0` and `1.260e0`. Throws an `ArgumentError` for a `prec` not in
`FORMAT_PRECISIONS`. Decimal ties round as in [`fmt_fixed`](@ref).
"""
function fmt_exp(x::Real, prec::Int; width::Int = 0)
    _check_precision(prec)
    isfinite(x) || return _pad_left(_nonfinite(x), width)
    s = _printf_exp(x, prec)
    # Printf writes the ASCII `<mantissa>e<sign><at least two digits>`; Rust drops a `+` sign
    # and the leading zeros of the exponent. The scans read bytes: `findlast(::Char, ::String)`
    # leaves an unresolved call under `--trim`, and `*` of a `SubString` and a `String` boxes.
    k = lastindex(s)
    while codeunit(s, k) != UInt8('e')
        k -= 1
    end
    j = k + 2
    while j < ncodeunits(s) && codeunit(s, j) == UInt8('0')
        j += 1
    end
    head = codeunit(s, k + 1) == UInt8('-') ? SubString(s, 1, k + 1) : SubString(s, 1, k)
    return _pad_left(head * SubString(s, j), width)
end

"""
    fmt_int(n::Integer, width::Int) -> String

`n` right-aligned in `width` characters, as Rust's `{:width}` for an integer.
"""
fmt_int(n::Integer, width::Int) = _pad_left(string(n), width)

"""
    fmt_str(s::AbstractString, width::Int) -> String

`s` left-aligned in `width` characters (counted as characters, not bytes), as Rust's `{:width}`
pads a string. A longer `s` is not truncated.
"""
fmt_str(s::AbstractString, width::Int) = _pad_right(s, width)
