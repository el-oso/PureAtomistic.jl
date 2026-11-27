"""
    Primitive{T<:Real}(exponent, coefficient)
    Primitive(exponent::Real, coefficient::Real)

One primitive of a contracted shell as a basis file gives it: the `exponent` in bohr⁻² and the
raw contraction `coefficient`, which multiplies a unit-normalized primitive.
"""
struct Primitive{T <: Real}
    exponent::T
    coefficient::T
    Primitive{T}(exponent, coefficient) where {T} = new{T}(exponent, coefficient)
end

Primitive(exponent::Real, coefficient::Real) =
    Primitive{promote_type(typeof(exponent), typeof(coefficient))}(exponent, coefficient)

"""
    ShellSpec{T<:Real}(l, primitives, line)
    ShellSpec(l, primitives::AbstractVector{Primitive{T}}, line)

One shell of one element as a basis file gives it: the angular momentum `l`, its `primitives`
in file order, and `line`, the line of the shell header in the file.
"""
struct ShellSpec{T <: Real}
    l::Int
    primitives::Vector{Primitive{T}}
    line::Int
    ShellSpec{T}(l, primitives, line) where {T} = new{T}(l, primitives, line)
end

ShellSpec(l, primitives::AbstractVector{Primitive{T}}, line) where {T} = ShellSpec{T}(l, primitives, line)

"""
    ECPChannel{T<:Real}(l, powers, exponents, coefficients)
    ECPChannel(l, powers, exponents::AbstractVector, coefficients::AbstractVector)

One channel of an effective core potential: the angular momentum `l` and, for each term `k` of
its radial expansion, the power `powers[k]`, the exponent `exponents[k]` and the coefficient
`coefficients[k]`, as the file gives them.
"""
struct ECPChannel{T <: Real}
    l::Int
    powers::Vector{Int}
    exponents::Vector{T}
    coefficients::Vector{T}
    ECPChannel{T}(l, powers, exponents, coefficients) where {T} = new{T}(l, powers, exponents, coefficients)
end

ECPChannel(l, powers, exponents::AbstractVector, coefficients::AbstractVector) =
    ECPChannel{promote_type(eltype(exponents), eltype(coefficients))}(l, powers, exponents, coefficients)

"""
    ECPPotential{T<:Real}(symbol, ncore, lmax, channels)
    ECPPotential(symbol, ncore, lmax, channels::AbstractVector{ECPChannel{T}})

The effective core potential of an element: its `symbol`, the number `ncore` of core electrons it
replaces, its maximum angular momentum `lmax` and its `channels`.

The channel with `l == lmax` is the local potential. It is identified by its `l`, never by its
position in `channels`. The `.gbs` reader stores the local channel first and then the
semilocal channels, whose values are those of the file: by the format's definition, a semilocal
channel `k` holds `U_k − U_L`, with `U_L` the local potential. The ECP JSON reader stores the
channels in file order with their values as read; whether those values are `U_l` or
`U_l − U_L` is not established.
"""
struct ECPPotential{T <: Real}
    symbol::String
    ncore::Int
    lmax::Int
    channels::Vector{ECPChannel{T}}
    ECPPotential{T}(symbol, ncore, lmax, channels) where {T} = new{T}(symbol, ncore, lmax, channels)
end

ECPPotential(symbol, ncore, lmax, channels::AbstractVector{ECPChannel{T}}) where {T} =
    ECPPotential{T}(symbol, ncore, lmax, channels)

"""
    ElementBasis{T<:Real}(symbol, shells, ecp)
    ElementBasis(symbol, shells::AbstractVector{ShellSpec{T}}, ecp)

The basis of one element as a basis file gives it: the capitalized element `symbol` (`"Ag"`),
its `shells` in file order, and its effective core potential `ecp`, or `nothing` when the file
has none for the element.
"""
struct ElementBasis{T <: Real}
    symbol::String
    shells::Vector{ShellSpec{T}}
    ecp::Union{Nothing, ECPPotential{T}}
    ElementBasis{T}(symbol, shells, ecp) where {T} = new{T}(symbol, shells, ecp)
end

ElementBasis(symbol, shells::AbstractVector{ShellSpec{T}}, ecp) where {T} = ElementBasis{T}(symbol, shells, ecp)

"""
    Shell{T<:Real}(center, atom, l, exponents, coefficients, ao_offset)
    Shell(center::AbstractVector, atom, l, exponents::AbstractVector, coefficients::AbstractVector, ao_offset)

A contracted Cartesian shell placed on an atom: its `center` in bohr, the index `atom` of the atom
in its molecule, the angular momentum `l`, the `exponents` in bohr⁻² and the `coefficients`, and
the 0-based offset `ao_offset` of its first component in the AO order ([`ao_range`](@ref) gives
its 1-based AO indices).

The `coefficients` are the raw coefficients of the basis file times
[`contraction_renorm`](@ref) of the shell; they do not include the per-component factor
[`primitive_norm`](@ref)`(α, lx, ly, lz)`, which the consumer applies. The constructor stores its
arguments as given and throws an `ArgumentError` for a negative `l`, exponents and coefficients of
different or zero length, an exponent that is not finite and positive, and a negative
`ao_offset`. The outer constructor takes `T` from the promoted element types of `center`,
`exponents` and `coefficients`.
"""
struct Shell{T <: Real}
    center::SVector{3, T}
    atom::Int
    l::Int
    exponents::Vector{T}
    coefficients::Vector{T}
    ao_offset::Int
    function Shell{T}(center, atom, l, exponents, coefficients, ao_offset) where {T}
        e = convert(Vector{T}, exponents)::Vector{T}
        c = convert(Vector{T}, coefficients)::Vector{T}
        _check_l(l)
        length(e) == length(c) ||
            throw(ArgumentError(lazy"basis: a shell has $(length(e)) exponents and $(length(c)) coefficients"))
        isempty(e) && throw(ArgumentError("basis: a contraction needs at least one primitive"))
        for k in eachindex(e)
            (isfinite(e[k]) && e[k] > 0) || throw(
                ArgumentError(lazy"basis: exponent $(k) of a shell is $(Float64(e[k])); exponents must be finite and positive")
            )
        end
        ao_offset >= 0 || throw(ArgumentError(lazy"basis: ao_offset must be ≥ 0, got $(Int(ao_offset))"))
        return new{T}(center, atom, l, e, c, ao_offset)
    end
end

Shell(center::AbstractVector, atom, l, exponents::AbstractVector, coefficients::AbstractVector, ao_offset) =
    Shell{promote_type(eltype(center), eltype(exponents), eltype(coefficients))}(
    center, atom, l, exponents, coefficients, ao_offset
)

"""
    BasisSet{T<:Real}(shells, nao, name)
    BasisSet(shells::AbstractVector{Shell{T}}, nao, name)

The molecular basis: its `shells` in the AO order, the number `nao` of AOs and the basis `name`.
The shells are numbered in atom order, then in file order within an atom, and the AO offsets
follow the shell order. Throws an `ArgumentError` unless the offsets are contiguous from 0 in shell
order and `nao` is their total.
"""
struct BasisSet{T <: Real}
    shells::Vector{Shell{T}}
    nao::Int
    name::String
    function BasisSet{T}(shells, nao, name) where {T}
        v = convert(Vector{Shell{T}}, shells)::Vector{Shell{T}}
        n = 0
        for k in eachindex(v)
            off = v[k].ao_offset
            off == n ||
                throw(ArgumentError(lazy"basis: shell $(k) starts at AO offset $(off), expected $(n)"))
            n += ncart(v[k].l)
        end
        nao == n || throw(ArgumentError(lazy"basis: nao is $(Int(nao)), but the shells hold $(n) AOs"))
        return new{T}(v, nao, name)
    end
end

BasisSet(shells::AbstractVector{Shell{T}}, nao, name) where {T} = BasisSet{T}(shells, nao, name)

"""
    ao_range(s::Shell) -> UnitRange{Int}

The 1-based AO indices of the components of `s`, `s.ao_offset + 1 : s.ao_offset + ncart(s.l)`.
"""
ao_range(s::Shell) = (s.ao_offset + 1):(s.ao_offset + ncart(s.l))

"""
    ShellPairs

The iterator [`shell_pairs`](@ref) returns: the shell pairs `(i, j)` with `i ≥ j` of `n` shells,
`i` ascending in the outer loop and `j` ascending from 1 to `i` in the inner loop. Its `length` is
`n(n + 1)/2`, its `eltype` `Tuple{Int,Int}`, and iterating it allocates nothing.
"""
struct ShellPairs
    n::Int
end

"""
    shell_pairs(bs::BasisSet) -> ShellPairs

The pairs `(i, j)` of shell indices of `bs` with `i ≥ j`, in the fixed order `(1, 1), (2, 1),
(2, 2), (3, 1), …`: `i` ascending in the outer loop, `j` ascending from 1 to `i` in the inner loop.
"""
shell_pairs(bs::BasisSet) = ShellPairs(length(bs.shells))

Base.length(p::ShellPairs) = p.n * (p.n + 1) ÷ 2
Base.eltype(::Type{ShellPairs}) = Tuple{Int, Int}

function Base.iterate(p::ShellPairs, state::Tuple{Int, Int} = (1, 1))
    i, j = state
    i > p.n && return nothing
    return (i, j), (j < i ? (i, j + 1) : (i + 1, 1))
end
