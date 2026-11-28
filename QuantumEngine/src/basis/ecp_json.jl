# The oracle's ECP JSON format: a top-level array of entries
# `{symbol, z_core, l_max, projectors: [{l, primitives: [{n, alpha, coeff}]}]}`. JSON.jl hands
# every value it reads to `_ecp_value`, which stores it as an `_ECPValue`, a concrete tagged node,
# so that the walker below reads concrete fields only. `JSON.parse` builds each object as an
# `_ECPObject` with `setindex!`, which converts the value at once and keeps the keys in the order
# of the text, a repeated key included, so that the walker can reject it: JSON.jl's own dictionary
# would keep the last of two equal keys silently. JSON.jl reads an integer as an `Int` (a `BigInt`
# beyond the Int64 range), a number with a fraction or an exponent as a `Float64` (a `BigFloat`
# beyond the Float64 range, which converts to ±Inf), a string as a `String` and an array as a
# `Vector{Any}`; every other value (`true`, `false`, `null`) is `_ECP_OTHER`.

const _ECP_INT = 0x01
const _ECP_BIGINT = 0x02
const _ECP_FLOAT = 0x03
const _ECP_STRING = 0x04
const _ECP_ARRAY = 0x05
const _ECP_OBJECT = 0x06
const _ECP_OTHER = 0x07

# `int` holds an `_ECP_INT`, `num` the value of any number (a `BigInt` and a `BigFloat` converted
# to Float64), `str` a string or the decimal digits of a `BigInt`, `items` the elements of an
# array or the values of an object, and `keys` the keys of an object, in the order of `items`.
struct _ECPValue
    kind::UInt8
    int::Int
    num::Float64
    str::String
    keys::Vector{String}
    items::Vector{_ECPValue}
end

_ecp_scalar(kind::UInt8, int::Int, num::Float64, str::String) =
    _ECPValue(kind, int, num, str, String[], _ECPValue[])

struct _ECPObject <: AbstractDict{String, Any}
    keys::Vector{String}
    values::Vector{_ECPValue}
end

_ECPObject() = _ECPObject(String[], _ECPValue[])

# JSON.jl builds an object with `dicttype()` and `setindex!` alone, and the walker reads the fields,
# so `_ECPObject` defines no other method of the dictionary interface.
function Base.setindex!(d::_ECPObject, @nospecialize(v), k::AbstractString)
    push!(d.keys, String(k))
    push!(d.values, _ecp_value(v))
    return d
end

function _ecp_value(@nospecialize(v))
    v isa Int && return _ecp_scalar(_ECP_INT, v, Float64(v), "")
    v isa BigInt && return _ecp_scalar(_ECP_BIGINT, 0, Float64(v), string(v))
    v isa Float64 && return _ecp_scalar(_ECP_FLOAT, 0, v, "")
    v isa BigFloat && return _ecp_scalar(_ECP_FLOAT, 0, Float64(v), "")
    v isa String && return _ecp_scalar(_ECP_STRING, 0, 0.0, v)
    v isa _ECPObject && return _ECPValue(_ECP_OBJECT, 0, 0.0, "", v.keys, v.values)
    if v isa Vector{Any}
        items = _ECPValue[]
        for i in eachindex(v)
            push!(items, _ecp_value(v[i]))
        end
        return _ECPValue(_ECP_ARRAY, 0, 0.0, "", String[], items)
    end
    return _ecp_scalar(_ECP_OTHER, 0, 0.0, "")
end

const ECP_ENTRY_KEYS = ("symbol", "z_core", "l_max", "projectors")
const ECP_PROJECTOR_KEYS = ("l", "primitives")
const ECP_PRIMITIVE_KEYS = ("n", "alpha", "coeff")

_ecp_error(src::String, path::String, msg) = BasisError(lazy"$(src): $(path): $(msg)")

# The object `x` at `path`, with its keys checked against `allowed` in text order (a repeated key,
# then a key outside `allowed`) and then in the order of `allowed` (a missing key).
function _ecp_object(src::String, x::_ECPValue, path::String, allowed::Tuple{Vararg{String}})
    x.kind == _ECP_OBJECT || throw(_ecp_error(src, path, "expected an object"))
    ks = x.keys
    for k in eachindex(ks)
        key = ks[k]
        for j in firstindex(ks):(k - 1)
            ks[j] == key && throw(_ecp_error(src, path, lazy"duplicate key $(repr(key))"))
        end
        key in allowed || throw(_ecp_error(src, path, lazy"unexpected key $(repr(key))"))
    end
    for key in allowed
        key in ks || throw(_ecp_error(src, path, lazy"missing key $(repr(key))"))
    end
    return x
end

# The readers of one field: the value of `key` in the object `obj` at `path`, which `_ecp_object`
# has checked to hold `key` once. An error is located at `<path>.<key>`.
_ecp_field(obj::_ECPValue, key::String) = obj.items[findfirst(==(key), obj.keys)::Int]

function _ecp_int(src::String, obj::_ECPValue, key::String, path::String)
    x = _ecp_field(obj, key)
    x.kind == _ECP_INT && return x.int
    if x.kind == _ECP_BIGINT
        throw(_ecp_error(src, path * "." * key, lazy"the integer $(x.str) is outside the Int64 range"))
    end
    throw(_ecp_error(src, path * "." * key, "expected an integer"))
end

function _ecp_nonnegative(src::String, obj::_ECPValue, key::String, path::String)
    v = _ecp_int(src, obj, key, path)
    v >= 0 || throw(_ecp_error(src, path * "." * key, lazy"must be ≥ 0, got $(v)"))
    return v
end

function _ecp_finite(src::String, obj::_ECPValue, key::String, path::String)
    x = _ecp_field(obj, key)
    x.kind in (_ECP_INT, _ECP_BIGINT, _ECP_FLOAT) || throw(_ecp_error(src, path * "." * key, "expected a number"))
    v = x.num
    isfinite(v) || throw(_ecp_error(src, path * "." * key, lazy"must be finite, got $(v)"))
    return v
end

function _ecp_positive(src::String, obj::_ECPValue, key::String, path::String)
    x = _ecp_field(obj, key)
    x.kind in (_ECP_INT, _ECP_BIGINT, _ECP_FLOAT) || throw(_ecp_error(src, path * "." * key, "expected a number"))
    v = x.num
    isfinite(v) && v > 0 || throw(_ecp_error(src, path * "." * key, lazy"must be positive and finite, got $(v)"))
    return v
end

function _ecp_string(src::String, obj::_ECPValue, key::String, path::String)
    x = _ecp_field(obj, key)
    x.kind == _ECP_STRING || throw(_ecp_error(src, path * "." * key, "expected a string"))
    return x.str
end

function _ecp_array(src::String, obj::_ECPValue, key::String, path::String)
    x = _ecp_field(obj, key)
    x.kind == _ECP_ARRAY || throw(_ecp_error(src, path * "." * key, "expected an array"))
    return x.items
end

# The location of item `k` of the array at `path`, counted from 1.
_ecp_index(path::String, k::Int) = path * "[" * string(k) * "]"

function _ecp_channel(src::String, x::_ECPValue, path::String, lmax::Int)
    obj = _ecp_object(src, x, path, ECP_PROJECTOR_KEYS)
    l = _ecp_int(src, obj, "l", path)
    0 <= l <= lmax || throw(_ecp_error(src, path * ".l", lazy"l = $(l) is outside 0:$(lmax)"))
    prims = _ecp_array(src, obj, "primitives", path)
    powers = Int[]
    exponents = Float64[]
    coefficients = Float64[]
    for i in eachindex(prims)
        ipath = _ecp_index(path * ".primitives", i)
        p = _ecp_object(src, prims[i], ipath, ECP_PRIMITIVE_KEYS)
        push!(powers, _ecp_nonnegative(src, p, "n", ipath))
        push!(exponents, _ecp_positive(src, p, "alpha", ipath))
        push!(coefficients, _ecp_finite(src, p, "coeff", ipath))
    end
    return ECPChannel{Float64}(l, powers, exponents, coefficients)
end

"""
    parse_ecp_json(io::Base.IO; source::AbstractString = "<stream>") -> Dict{String,ECPPotential{Float64}}

The effective core potentials of the ECP JSON document read from `io`, in the oracle's format: a
top-level array of objects `{"symbol", "z_core", "l_max", "projectors": [{"l", "primitives":
[{"n", "alpha", "coeff"}]}]}`. Each entry becomes one `ECPPotential` keyed by its capitalized
symbol (`"br"` is keyed `"Br"`), with `ncore = z_core` and `lmax = l_max`; each projector becomes
one `ECPChannel(l, n…, alpha…, coeff…)`, in the order of the document. The values are stored as
read.

Every key named above is required and no other is accepted. Any failure is a `BasisError` that
names `source` and the location in the document, `[2].projectors[1].primitives[3].alpha` (arrays
counted from 1; `\$` for the top level): text that is not JSON, a top level that is not an array,
an unexpected, missing or repeated key, a value of the wrong type, an integer outside the Int64
range, a negative `z_core`, `l_max` or power `n`, a `z_core` above the atomic number, a
projector `l` outside `0:l_max`, an `alpha` that is not positive and finite, a `coeff` that is
not finite (a number beyond the Float64 range reads as ±Inf, one too small for it as 0), an
unknown element symbol and a second entry for one element, printed as
`basis: t.json: [3]: second entry for Br (first at [1])`.
"""
function parse_ecp_json(io::Base.IO; source::AbstractString = "<stream>")
    src = String(source)
    x = try
        _ecp_value(JSON.parse(io; dicttype = _ECPObject))
    catch e
        e isa ArgumentError || rethrow()
        throw(_ecp_error(src, "\$", lazy"not valid JSON ($(chomp(e.msg)))"))
    end
    x.kind == _ECP_ARRAY || throw(_ecp_error(src, "\$", "expected an array of ECP entries"))
    entries = x.items
    data = Dict{String, ECPPotential{Float64}}()
    first = Dict{String, Int}()
    for k in eachindex(entries)
        path = _ecp_index("", k)
        e = _ecp_object(src, entries[k], path, ECP_ENTRY_KEYS)
        s = _ecp_string(src, e, "symbol", path)
        z = symbol_number(s)
        iszero(z) && throw(_ecp_error(src, path * ".symbol", lazy"unknown element symbol $(repr(s))"))
        sym = element(z).symbol
        haskey(first, sym) && throw(_ecp_error(src, path, lazy"second entry for $(sym) (first at [$(first[sym])])"))
        first[sym] = k
        ncore = _ecp_nonnegative(src, e, "z_core", path)
        ncore <= z || throw(_ecp_error(src, path * ".z_core", lazy"z_core $(ncore) is above the atomic number $(z)"))
        lmax = _ecp_nonnegative(src, e, "l_max", path)
        projectors = _ecp_array(src, e, "projectors", path)
        channels = ECPChannel{Float64}[]
        for j in eachindex(projectors)
            push!(channels, _ecp_channel(src, projectors[j], _ecp_index(path * ".projectors", j), lmax))
        end
        data[sym] = ECPPotential{Float64}(sym, ncore, lmax, channels)
    end
    return data
end

"""
    read_ecp_json(path::AbstractString) -> Dict{String,ECPPotential{Float64}}

The effective core potentials of the ECP JSON file at `path`, read with
[`parse_ecp_json`](@ref) with `source = path`.
"""
function read_ecp_json(path::AbstractString)
    p = String(path)
    io = open(p)
    return try
        parse_ecp_json(io; source = p)
    finally
        close(io)
    end
end
