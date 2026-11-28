# The Gaussian94 `.gbs` reader. One pass over the lines, numbered from 1, collects the basis blocks
# and the ECP sections in two dictionaries that record the line of each symbol's header; they are
# merged into one `ElementBasis` per symbol at EOF.

const GBS_HEADER = r"^\s*([A-Za-z]{1,2})\s+0\s*$"

# The shell letters of l = 0 to 6, in the case the messages and the ECP channel titles use.
const SHELL_LETTERS = ("S", "P", "D", "F", "G", "H", "I")
const ECP_LETTERS = ("s", "p", "d", "f", "g", "h", "i")

# The states of the reader: outside a block, after a header (before its first significant line),
# between the shells of a basis block, inside a primitive list, and at the title, the row count
# and the rows of an ECP channel.
const GBS_TOP = 0
const GBS_HEADER_SEEN = 1
const GBS_BLOCK = 2
const GBS_PRIMITIVES = 3
const GBS_ECP_TITLE = 4
const GBS_ECP_COUNT = 5
const GBS_ECP_ROWS = 6

_gbs_error(src::String, n::Int, msg) = BasisError(lazy"$(src):$(n): $(msg)")

_is_comment(line) = startswith(line, '!') || startswith(line, '#')

# The capitalized element symbol of the header `line` (already matched by `GBS_HEADER`, so it
# starts with the one or two letters of the symbol), looked up in the element table; an unknown
# symbol is a `BasisError` of line `n`. The letters are passed to `symbol_number` as a code-unit
# range and the element is taken by its number, so no `SubString` is built and nothing is caught:
# under `--check-bounds=yes` a `SubString` constructor boxes its argument for the `BoundsError` it
# may throw, and inside a `try` the message that `element` builds for an unknown symbol is
# reachable without a throw.
function _gbs_symbol(src::String, n::Int, line::SubString{String})
    k = _is_ascii_letter(codeunit(line, 2)) ? 2 : 1
    z = symbol_number(line, 1, k)
    z == 0 && throw(_unknown_element(src, n, line, k))
    return element(z).symbol
end

_unknown_element(src::String, n::Int, line::SubString{String}, k::Int) =
    _gbs_error(src, n, lazy"unknown element symbol \"$(SubString(line, 1, k))\"")

_is_ascii_letter(c::UInt8) = UInt8('a') <= ascii_lowercase(c) <= UInt8('z')

# `s` with its first code unit in ASCII upper case and the others in ASCII lower case.
function _capitalized(s::AbstractString)
    b = Vector{UInt8}(undef, ncodeunits(s))
    for k in eachindex(b)
        c = ascii_lowercase(codeunit(s, k))
        b[k] = k == 1 && UInt8('a') <= c <= UInt8('z') ? c - (UInt8('a') - UInt8('A')) : c
    end
    return String(b)
end

# Whether `t` equals the lower-case ASCII `ref` without regard to ASCII case.
function _folds_to(t::AbstractString, ref::String)
    ncodeunits(t) == ncodeunits(ref) || return false
    for k in 1:ncodeunits(ref)
        ascii_lowercase(codeunit(t, k)) == codeunit(ref, k) || return false
    end
    return true
end

# The angular momentum of a shell type: 0 to 6 for S to I, 7 for SP (read as S, then P), and -1
# for any other type; the letters are matched without regard to case.
function _shell_type(t::AbstractString)
    _folds_to(t, "sp") && return 7
    for l in eachindex(ECP_LETTERS)
        _folds_to(t, ECP_LETTERS[l]) && return l - 1
    end
    return -1
end

# A number token, with a Fortran `D` or `d` exponent read as `E`.
function _gbs_number(src::String, n::Int, t::SubString{String})
    x = occursin('D', t) || occursin('d', t) ? parse_coordinate(replace(t, 'D' => 'E', 'd' => 'E')) : parse_coordinate(t)
    x === nothing && throw(_gbs_error(src, n, lazy"\"$(t)\" is not a number"))
    return x
end

# The positive count of the token `t`, or 0 when `t` is not a decimal integer ≥ 1.
function _gbs_count(t::AbstractString)
    is_decimal_int(t) || return 0
    v = tryparse(Int, t)
    return v === nothing || v < 1 ? 0 : v
end

# `lmax` and `ncore` of an `<sym>-ECP lmax ncore` line, with `<sym>` one or two letters, or
# `(-1, -1)` when `tokens` are not such a line.
function _ecp_line(tokens::Vector{SubString{String}})
    length(tokens) == 3 || return (-1, -1)
    head = tokens[1]
    m = ncodeunits(head) - 4
    1 <= m <= 2 || return (-1, -1)
    for k in 1:4
        ascii_lowercase(codeunit(head, m + k)) == codeunit("-ecp", k) || return (-1, -1)
    end
    for k in 1:m
        _is_ascii_letter(codeunit(head, k)) || return (-1, -1)
    end
    for t in (tokens[2], tokens[3])
        (is_decimal_int(t) && !startswith(t, '-') && !startswith(t, '+')) || return (-1, -1)
    end
    lmax = tryparse(Int, tokens[2])
    ncore = tryparse(Int, tokens[3])
    (lmax === nothing || ncore === nothing) && return (-1, -1)
    return (lmax, ncore)
end

# The title of ECP channel `c` (1-based) of a section with maximum angular momentum `lmax`: the
# local channel `"<L> potential"` first, then `"<k>-<L> potential"` for k = 0 … lmax − 1.
_ecp_title(c::Int, lmax::Int) =
    c == 1 ? ECP_LETTERS[lmax + 1] * " potential" : ECP_LETTERS[c - 1] * "-" * ECP_LETTERS[lmax + 1] * " potential"

_ecp_channel_l(c::Int, lmax::Int) = c == 1 ? lmax : c - 2

# Whether `tokens` spell `title` (two words) without regard to ASCII case.
function _is_title(tokens::Vector{SubString{String}}, title::String)
    length(tokens) == 2 || return false
    sp = findfirst(' ', title)::Int
    return _folds_to(tokens[1], title[1:(sp - 1)]) && _folds_to(tokens[2], title[(sp + 1):end])
end

# The shell `l` of element `sym` with exponents `e` and coefficients `c`, checked once with
# `contraction_renorm`, which throws an `ArgumentError` when the norm sum is not positive and
# finite or an exponent squared is below `floatmin(Float64)`; that is a `BasisError` of the shell's
# line `n`. Its other `ArgumentError`s cannot occur here: the reader has checked the exponents and
# the count. The sum is zero when every coefficient is; otherwise it overflowed, underflowed or
# cancelled in Float64 (`1e300 1.0`, `1.0 1e200`, or one exponent repeated with opposite
# coefficients), or an exponent is too small for its products with the others (`1e-300 1.0`, or
# `1e-200 1.0` beside `1.0 1.0`).
function _gbs_shell(src::String, n::Int, sym::String, l::Int, e::Vector{Float64}, c::Vector{Float64})
    try
        contraction_renorm(e, c, l)
    catch err
        err isa ArgumentError || rethrow()
        letter = SHELL_LETTERS[l + 1]
        all(iszero, c) && throw(_gbs_error(src, n, lazy"the $(letter) shell of $(sym) has zero norm"))
        throw(_gbs_error(src, n, lazy"the norm of the $(letter) shell of $(sym) is not a positive finite Float64"))
    end
    prims = Vector{Primitive{Float64}}(undef, length(e))
    for k in eachindex(prims, e, c)
        prims[k] = Primitive{Float64}(e[k], c[k])
    end
    return ShellSpec{Float64}(l, prims, n)
end

"""
    parse_gbs(io::Base.IO; source::AbstractString = "<stream>") -> Dict{String,ElementBasis{Float64}}

Every element of the Gaussian94 basis file in `io`, keyed by the capitalized element symbol
(`AG 0` gives the key `"Ag"`).

The file is a sequence of basis blocks and ECP sections. Blank lines are skipped, and so are
comment lines (first character `!` or `#`) outside a primitive list and outside the rows of an ECP
channel. A `****` line closes a basis block; one outside a block is accepted. A header is a line
matching `^\\s*([A-Za-z]{1,2})\\s+0\\s*\$`. When the next line that is not blank and not a comment
is `<sym>-ECP lmax ncore`, the header opens an ECP section; otherwise it opens a basis block.

A basis block is a sequence of shells, ended by `****` or EOF. A shell is a line `<type> <count>
[<scale>]` followed by `count` primitive lines `exponent coefficient` (`exponent s-coefficient
p-coefficient` for `SP`). The types `S P D F G H I` give l = 0 to 6 and `SP` an S shell followed by
a P shell with the same exponents; they are matched without regard to case. A scale factor must
be exactly 1. A `D` or `d` exponent marker reads as `E`. Each exponent must be finite and
positive and each coefficient finite, and each shell is checked to have a norm that is positive
and finite in Float64. Each `ShellSpec` records the line of its shell header.

An ECP section holds `lmax + 1` channels: the local channel titled `<L> potential` (l = `lmax`)
first, then `<k>-<L> potential` for k = 0 … lmax − 1, where `L` and `k` are shell letters and the
titles are matched without regard to case. A channel is its title, a row count and that many rows
`power exponent coefficient`: the power is an integer ≥ 0 (the term is `r^(power - 2)`), the
exponent finite and positive and the coefficient finite. `ncore` is at most the atomic number.
The section ends after the last row of its last channel. An ECP section for an element with no basis block in the file gives an `ElementBasis`
with no shells.

Every malformed line throws a `BasisError` whose message starts with `<source>:<line>: `. A shell,
an ECP channel or an ECP section that ends early names the line where it starts; a basis block or
ECP section repeated for one element names the line of the second header and the first.
"""
function parse_gbs(io::Base.IO; source::AbstractString = "<stream>")
    src = String(source)
    blocks = Dict{String, Tuple{Int, Vector{ShellSpec{Float64}}}}()
    ecps = Dict{String, Tuple{Int, ECPPotential{Float64}}}()
    state = GBS_TOP
    sym = ""
    header_line = 0
    shells = ShellSpec{Float64}[]
    # The shell being read: its type (`_shell_type`), header line, primitive count, and columns.
    stype = 0
    shell_line = 0
    nprim = 0
    exps = Float64[]
    c1 = Float64[]
    c2 = Float64[]
    # The ECP section being read: lmax, ncore, the finished channels, and the open channel's title
    # line, row count and columns.
    lmax = 0
    ncore = 0
    channels = ECPChannel{Float64}[]
    title_line = 0
    nrows = 0
    powers = Int[]
    ecp_exps = Float64[]
    ecp_coefs = Float64[]
    n = 0
    for raw in eachline(io)
        n += 1
        line, tokens = input_line(raw, n)
        isempty(line) && continue
        header = occursin(GBS_HEADER, line)
        stars = line == "****"
        if state == GBS_PRIMITIVES
            (header || stars) && throw(_shell_ends(src, shell_line, stype, length(exps), nprim))
            ncol = stype == 7 ? 3 : 2
            if length(tokens) != ncol
                ncol == 3 && throw(
                    _gbs_error(src, n, lazy"expected 3 numbers (exponent, s-coefficient, p-coefficient), got $(length(tokens))")
                )
                throw(_gbs_error(src, n, lazy"expected 2 numbers (exponent, coefficient), got $(length(tokens))"))
            end
            a = _gbs_number(src, n, tokens[1])
            ca = _gbs_number(src, n, tokens[2])
            cb = ncol == 3 ? _gbs_number(src, n, tokens[3]) : 0.0
            (isfinite(a) && a > 0) ||
                throw(_gbs_error(src, n, lazy"exponent must be positive and finite, got $(a)"))
            isfinite(ca) || throw(_gbs_error(src, n, lazy"coefficient must be finite, got $(ca)"))
            isfinite(cb) || throw(_gbs_error(src, n, lazy"coefficient must be finite, got $(cb)"))
            push!(exps, a)
            push!(c1, ca)
            ncol == 3 && push!(c2, cb)
            if length(exps) == nprim
                if stype == 7
                    push!(shells, _gbs_shell(src, shell_line, sym, 0, exps, c1))
                    push!(shells, _gbs_shell(src, shell_line, sym, 1, exps, c2))
                else
                    push!(shells, _gbs_shell(src, shell_line, sym, stype, exps, c1))
                end
                exps = Float64[]
                c1 = Float64[]
                c2 = Float64[]
                state = GBS_BLOCK
            end
            continue
        end
        if state == GBS_ECP_ROWS
            (header || stars) && throw(_channel_ends(src, title_line, length(channels) + 1, lmax, length(powers), nrows))
            length(tokens) == 3 ||
                throw(_gbs_error(src, n, lazy"expected 3 numbers (power, exponent, coefficient), got $(length(tokens))"))
            _gbs_number(src, n, tokens[1])
            is_decimal_int(tokens[1]) || throw(_gbs_error(src, n, lazy"\"$(tokens[1])\" is not an integer power"))
            pw = tryparse(Int, tokens[1])
            pw === nothing && throw(_gbs_error(src, n, lazy"\"$(tokens[1])\" is not an integer power"))
            pw >= 0 || throw(_gbs_error(src, n, lazy"power must be non-negative, got $(pw)"))
            a = _gbs_number(src, n, tokens[2])
            ca = _gbs_number(src, n, tokens[3])
            (isfinite(a) && a > 0) ||
                throw(_gbs_error(src, n, lazy"exponent must be positive and finite, got $(a)"))
            isfinite(ca) || throw(_gbs_error(src, n, lazy"coefficient must be finite, got $(ca)"))
            push!(powers, pw)
            push!(ecp_exps, a)
            push!(ecp_coefs, ca)
            if length(powers) == nrows
                push!(channels, ECPChannel{Float64}(_ecp_channel_l(length(channels) + 1, lmax), powers, ecp_exps, ecp_coefs))
                powers = Int[]
                ecp_exps = Float64[]
                ecp_coefs = Float64[]
                if length(channels) == lmax + 1
                    ecps[sym] = (header_line, ECPPotential{Float64}(sym, ncore, lmax, channels))
                    channels = ECPChannel{Float64}[]
                    state = GBS_TOP
                else
                    state = GBS_ECP_TITLE
                end
            end
            continue
        end
        _is_comment(line) && continue
        if state == GBS_ECP_TITLE || state == GBS_ECP_COUNT
            (header || stars) && throw(_too_few_channels(src, header_line, sym, length(channels), lmax))
            c = length(channels) + 1
            if state == GBS_ECP_TITLE
                _is_title(tokens, _ecp_title(c, lmax)) ||
                    throw(_gbs_error(src, n, lazy"expected the \"$(_ecp_title(c, lmax))\" channel, got \"$(line)\""))
                title_line = n
                state = GBS_ECP_COUNT
            else
                nrows = length(tokens) == 1 ? _gbs_count(tokens[1]) : 0
                nrows == 0 && throw(
                    _gbs_error(src, n, lazy"expected the row count of the \"$(_ecp_title(c, lmax))\" channel, got \"$(line)\"")
                )
                state = GBS_ECP_ROWS
            end
            continue
        end
        if state == GBS_TOP
            stars && continue
            header || throw(_gbs_error(src, n, lazy"expected an element header \"<symbol> 0\", got \"$(line)\""))
            sym = _gbs_symbol(src, n, line)
            header_line = n
            state = GBS_HEADER_SEEN
            continue
        end
        if state == GBS_HEADER_SEEN
            stars && throw(_gbs_error(src, header_line, lazy"the basis block of $(sym) has no shells"))
            elmax, encore = _ecp_line(tokens)
            if elmax >= 0
                esym = _capitalized(SubString(tokens[1], 1, ncodeunits(tokens[1]) - 4))
                esym == sym || throw(_gbs_error(src, n, lazy"ECP section for $(esym) under the header of $(sym)"))
                if haskey(ecps, sym)
                    throw(_gbs_error(src, header_line, lazy"second ECP section for $(sym) (first at line $(ecps[sym][1]))"))
                end
                elmax <= 6 || throw(
                    _gbs_error(src, n, lazy"the ECP section of $(sym) has lmax $(elmax); channel titles stop at I (l = 6)")
                )
                z = symbol_number(sym)
                encore <= z || throw(
                    _gbs_error(src, n, lazy"the ECP section of $(sym) has ncore $(encore), above the atomic number $(z)")
                )
                lmax = elmax
                ncore = encore
                state = GBS_ECP_TITLE
                continue
            end
            if !header && haskey(blocks, sym)
                throw(_gbs_error(src, header_line, lazy"second basis block for $(sym) (first at line $(blocks[sym][1]))"))
            end
            shells = ShellSpec{Float64}[]
            state = GBS_BLOCK
        end
        # A line between the shells of a basis block.
        if stars
            blocks[sym] = (header_line, shells)
            state = GBS_TOP
            continue
        end
        header && throw(
            _gbs_error(src, n, lazy"the basis block of $(sym) is not closed by \"****\" before the header of $(_gbs_symbol(src, n, line))")
        )
        stype = _shell_type(tokens[1])
        stype < 0 && throw(_gbs_error(src, n, lazy"unknown shell type \"$(tokens[1])\""))
        2 <= length(tokens) <= 3 ||
            throw(_gbs_error(src, n, lazy"expected a shell line \"<type> <count> [<scale>]\", got \"$(line)\""))
        nprim = _gbs_count(tokens[2])
        nprim == 0 && throw(_gbs_error(src, n, lazy"expected the primitive count, got \"$(tokens[2])\""))
        if length(tokens) == 3
            scale = _gbs_number(src, n, tokens[3])
            scale == 1 || throw(_gbs_error(src, n, lazy"scale factor $(scale) is not supported"))
        end
        shell_line = n
        state = GBS_PRIMITIVES
    end
    # EOF.
    state == GBS_HEADER_SEEN && throw(_gbs_error(src, header_line, lazy"the basis block of $(sym) has no shells"))
    if state == GBS_BLOCK
        blocks[sym] = (header_line, shells)
    end
    state == GBS_PRIMITIVES && throw(_shell_ends(src, shell_line, stype, length(exps), nprim))
    (state == GBS_ECP_TITLE || state == GBS_ECP_COUNT) &&
        throw(_too_few_channels(src, header_line, sym, length(channels), lmax))
    state == GBS_ECP_ROWS && throw(_channel_ends(src, title_line, length(channels) + 1, lmax, length(powers), nrows))
    data = Dict{String, ElementBasis{Float64}}()
    for (s, (_, sh)) in blocks
        # Two calls, not one with `haskey(ecps, s) ? … : nothing`: that local would be a union with a
        # heap-allocated member.
        if haskey(ecps, s)
            data[s] = ElementBasis{Float64}(s, sh, ecps[s][2])
        else
            data[s] = ElementBasis{Float64}(s, sh, nothing)
        end
    end
    for (s, (_, ecp)) in ecps
        haskey(data, s) || (data[s] = ElementBasis{Float64}(s, ShellSpec{Float64}[], ecp))
    end
    return data
end

function _shell_ends(src::String, n::Int, stype::Int, k::Int, nprim::Int)
    letter = stype == 7 ? "SP" : SHELL_LETTERS[stype + 1]
    return _gbs_error(src, n, lazy"the $(letter) shell ends after $(k) of $(nprim) primitives")
end

_channel_ends(src::String, n::Int, c::Int, lmax::Int, k::Int, nrows::Int) =
    _gbs_error(src, n, lazy"the \"$(_ecp_title(c, lmax))\" channel ends after $(k) of $(nrows) rows")

_too_few_channels(src::String, n::Int, sym::String, k::Int, lmax::Int) =
    _gbs_error(src, n, lazy"the ECP section of $(sym) has $(k) of $(lmax + 1) channels")

"""
    read_element_basis(path::AbstractString, symbol::AbstractString) -> ElementBasis{Float64}

The basis of the element `symbol` (any ASCII case) in the Gaussian94 basis file at `path`. The
whole file is parsed with [`parse_gbs`](@ref), with `source = path`, so an error anywhere in it is
reported. Throws `BasisError("<path>: no basis for element Xe")` when the file has no entry for the
element.
"""
function read_element_basis(path::AbstractString, symbol::AbstractString)
    p = String(path)
    io = open(p)
    data = try
        parse_gbs(io; source = p)
    finally
        close(io)
    end
    key = _capitalized(symbol)
    haskey(data, key) || throw(BasisError(lazy"$(p): no basis for element $(key)"))
    return data[key]
end
