# Numbers and strings in the layouts of Rust's `format!`, the formatter of the oracle's output.
# Rust pads to a width counted in characters, numbers on the left and strings on the right, and
# never truncates. A non-finite number prints as `inf`, `-inf` or `NaN`.

_pad_left(s::String, width::Int) = " "^max(0, width - length(s)) * s
_pad_right(s::AbstractString, width::Int) = String(s) * " "^max(0, width - length(s))

_nonfinite(x::Real) = isnan(x) ? "NaN" : x > 0 ? "inf" : "-inf"

function _check_precision(prec::Int)
    prec >= 0 || throw(ArgumentError(lazy"io: precision must be ≥ 0, got $(prec)"))
    return nothing
end

"""
    fmt_fixed(x::Real, width::Int, prec::Int) -> String

`x` with `prec` digits after the decimal point, right-aligned in `width` characters, as Rust's
`{:width.prec}`: `fmt_fixed(-76.301896228438, 20, 12) == "    -76.301896228438"`. Throws an
`ArgumentError` for a negative `prec`.

The digits come from Printf's `%.*f`, which rounds the exact binary value. Where that value is
an exact decimal tie, both Printf and Rust round to even on the ties the item "Rust fixed,
integer and string formats" pins; other ties are not checked, and Rust may round them the other
way. The parity harness compares parsed numbers, not text, so such a difference is cosmetic.
"""
function fmt_fixed(x::Real, width::Int, prec::Int)
    _check_precision(prec)
    isfinite(x) || return _pad_left(_nonfinite(x), width)
    return _pad_left(@sprintf("%.*f", prec, x), width)
end

"""
    fmt_exp(x::Real, prec::Int; width::Int = 0) -> String

`x` in scientific notation with `prec` digits after the decimal point, right-aligned in `width`
characters, as Rust's `{:width.prece}`: the exponent has no `+` and no leading zeros, as in
`5.369e-1`, `0.000e0` and `1.260e0`. Throws an `ArgumentError` for a negative `prec`. Decimal
ties round as in [`fmt_fixed`](@ref).
"""
function fmt_exp(x::Real, prec::Int; width::Int = 0)
    _check_precision(prec)
    isfinite(x) || return _pad_left(_nonfinite(x), width)
    s = @sprintf("%.*e", prec, x)
    # Printf writes the ASCII `<mantissa>e<sign><at least two digits>`. The `e` is found by a
    # byte loop: `findlast(::Char, ::String)` leaves an unresolved call under `--trim`.
    k = lastindex(s)
    while codeunit(s, k) != UInt8('e')
        k -= 1
    end
    mantissa = SubString(s, 1, k)
    minus = s[k + 1] == '-' ? "-" : ""
    exp_digits = SubString(s, k + 2)
    e10 = lstrip(exp_digits, '0')
    power = isempty(e10) ? "0" : String(e10)
    return _pad_left(mantissa * minus * power, width)
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
