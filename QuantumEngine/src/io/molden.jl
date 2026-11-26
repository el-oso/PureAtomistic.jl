"""
    MoldenShell{T}(atom, l, exponents, coefficients)
    MoldenShell(atom, l, exponents, coefficients)

A contracted Gaussian shell of a Molden file: the index `atom` of the atom it sits on (in the
atom list of its `MoldenData`), its angular momentum `l` (0 to 4: `s`, `p`, `d`, `f`, `g`), and
the `exponents` and contraction `coefficients` of its primitives, one of each per primitive. The
coefficients are stored as given; their normalization is the caller's. Without a type parameter,
`T` is the floating-point type of the two vectors. Throws an `ArgumentError` for an `l` outside
0:4, an `atom` below 1, no primitive, vectors of different lengths, offset vectors, an exponent
that is not finite and positive, and a coefficient that is not finite.
"""
struct MoldenShell{T}
    atom::Int
    l::Int
    exponents::Vector{T}
    coefficients::Vector{T}
    function MoldenShell{T}(atom, l, exponents, coefficients) where {T}
        Base.require_one_based_indexing(exponents, coefficients)
        0 <= l <= 4 || throw(ArgumentError(lazy"molden: shell with l = $(l) has no Molden label"))
        atom >= 1 || throw(ArgumentError(lazy"molden: a shell's atom index is at least 1, got $(atom)"))
        ne = length(exponents)
        nc = length(coefficients)
        ne == nc || throw(ArgumentError(lazy"molden: $(ne) exponents but $(nc) coefficients"))
        ne >= 1 || throw(ArgumentError("molden: a shell has at least one primitive"))
        all(x -> isfinite(x) && x > 0, exponents) ||
            throw(ArgumentError("molden: shell exponents must be finite and positive"))
        all(isfinite, coefficients) || throw(ArgumentError("molden: shell coefficients must be finite"))
        return new{T}(atom, l, exponents, coefficients)
    end
end

MoldenShell(atom, l, exponents::AbstractVector, coefficients::AbstractVector) =
    MoldenShell{float(promote_type(eltype(exponents), eltype(coefficients)))}(atom, l, exponents, coefficients)

"""
    MoldenOrbitals{T}(energies, occupations, coefficients)
    MoldenOrbitals(energies, occupations, coefficients)

The molecular orbitals of one spin: their `energies` (hartree), `occupations`, and the
`coefficients` matrix, n_ao × n_mo, whose column `j` holds orbital `j` over the atomic orbitals
in the order of the shells. Without a type parameter, `T` is the floating-point type of the
three arrays. Throws an `ArgumentError` for offset arrays, sizes that disagree, and a value that
is not finite.
"""
struct MoldenOrbitals{T}
    energies::Vector{T}
    occupations::Vector{T}
    coefficients::Matrix{T}
    function MoldenOrbitals{T}(energies, occupations, coefficients) where {T}
        Base.require_one_based_indexing(energies, occupations, coefficients)
        ne = length(energies)
        no = length(occupations)
        nc = size(coefficients, 2)
        ne == no || throw(ArgumentError(lazy"molden: $(ne) energies but $(no) occupations"))
        ne == nc ||
            throw(ArgumentError(lazy"molden: $(ne) energies but the coefficient matrix has $(nc) columns"))
        (all(isfinite, energies) && all(isfinite, occupations) && all(isfinite, coefficients)) ||
            throw(ArgumentError("molden: orbital energies, occupations and coefficients must be finite"))
        return new{T}(energies, occupations, coefficients)
    end
end

MoldenOrbitals(energies::AbstractVector, occupations::AbstractVector, coefficients::AbstractMatrix) =
    MoldenOrbitals{float(promote_type(eltype(energies), eltype(occupations), eltype(coefficients)))}(
    energies, occupations, coefficients
)

"""
    MoldenData{T}(title, atoms, shells, alpha, beta)
    MoldenData(title, atoms, shells, alpha, beta = nothing)

The contents of a Molden file: a one-line `title`, the `atoms` (positions in bohr), the basis
`shells`, ordered by atom, the `alpha` orbitals and the `beta` orbitals (`nothing` for a
restricted wave function). The coefficient matrices have one row per Cartesian atomic orbital of
the shells, (l + 1)(l + 2)/2 per shell. Throws an `ArgumentError` for a title with a line break,
with leading or trailing whitespace or starting with `[`, none of which a Molden file can hold;
an offset vector; no atoms; an atom with a non-finite position; a shell on an atom that does not
exist; shells out of atom order; no alpha orbitals; and a coefficient matrix whose row count is
not the number of atomic orbitals. (`read_molden` rejects a file with no atoms or no alpha
orbitals.)
"""
struct MoldenData{T}
    title::String
    atoms::Vector{Atom{T}}
    shells::Vector{MoldenShell{T}}
    alpha::MoldenOrbitals{T}
    beta::Union{Nothing, MoldenOrbitals{T}}
    function MoldenData{T}(title, atoms, shells, alpha, beta) where {T}
        Base.require_one_based_indexing(atoms, shells)
        (occursin('\n', title) || occursin('\r', title)) && throw(ArgumentError("molden: a title is one line"))
        if !isempty(title)
            (isspace(first(title)) || isspace(last(title))) &&
                throw(ArgumentError("molden: a title has no leading or trailing whitespace"))
            first(title) == '[' && throw(ArgumentError("molden: a title does not start with ["))
        end
        isempty(atoms) && throw(ArgumentError("molden: the data has no atoms"))
        for i in eachindex(atoms)
            all(isfinite, atoms[i].position) ||
                throw(ArgumentError(lazy"molden: atom $(i) has a non-finite position"))
        end
        nat = length(atoms)
        prev = 0
        nao = 0
        for k in eachindex(shells)
            a = shells[k].atom
            a <= nat || throw(ArgumentError(lazy"molden: shell $(k) is on atom $(a), but there are $(nat) atoms"))
            a >= prev ||
                throw(ArgumentError(lazy"molden: shells are ordered by atom; shell $(k) on atom $(a) follows atom $(prev)"))
            prev = a
            nao += _ncartesian(shells[k].l)
        end
        isempty(alpha.energies) && throw(ArgumentError("molden: the data has no alpha orbitals"))
        _check_rows("alpha", alpha, nao)
        isnothing(beta) || _check_rows("beta", beta, nao)
        return new{T}(title, atoms, shells, alpha, beta)
    end
end

MoldenData(
    title::AbstractString, atoms::AbstractVector{Atom{T}}, shells::AbstractVector{MoldenShell{T}},
    alpha::MoldenOrbitals{T}, beta::Union{Nothing, MoldenOrbitals{T}} = nothing,
) where {T} = MoldenData{T}(title, atoms, shells, alpha, beta)

# The number of Cartesian functions of a shell of angular momentum `l`.
_ncartesian(l::Int) = (l + 1) * (l + 2) ÷ 2

function _check_rows(spin::String, orbitals::MoldenOrbitals, nao::Int)
    n = size(orbitals.coefficients, 1)
    n == nao ||
        throw(ArgumentError(lazy"molden: the shells have $(nao) atomic orbitals but the $(spin) coefficients have $(n) rows"))
    return nothing
end

const _MOLDEN_LABELS = ('s', 'p', 'd', 'f', 'g')

"""
    write_molden(io::Base.IO, d::MoldenData)

Writes `d` as a Molden file: `[Molden Format]`; `[Title]` and the title line; `[Atoms] AU` with
one line `<symbol> <index> <Z> <x> <y> <z>` per atom, in bohr; `[GTO]` with, per atom, the line
`<atom> 0`, then per shell `<s|p|d|f|g> <nprim> 1.00` and one `<exponent> <coefficient>` line per
primitive, then a blank line; and `[MO]` with, per orbital, alpha before beta, the lines
`Sym= A`, `Ene= <energy>`, `Spin= Alpha` or `Spin= Beta`, `Occup= <occupation>` and one
`<ao> <coefficient>` line per atomic orbital. Every number is written as the shortest decimal
that reads back as the same `Float64` (`string(::Float64)`), so `read_molden` reads back the
values written. The coefficients are written as given. Atomic masses are not written, since a
Molden file has no field for them; `read_molden` gives each atom its default mass.

There is no `[5D]` line, so a reader takes every shell as Cartesian. Within a shell the atomic
orbitals are expected in the order of the Molden format:

- d: xx, yy, zz, xy, xz, yz;
- f: xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz;
- g: xxxx, yyyy, zzzz, xxxy, xxxz, yyyx, yyyz, zzzx, zzzy, xxyy, xxzz, yyzz, xxyz, yyxz, zzxy.
"""
function write_molden(io::Base.IO, d::MoldenData)
    write(io, "[Molden Format]\n[Title]\n")
    write(io, d.title)
    write(io, "\n[Atoms] AU\n")
    atoms = d.atoms
    for i in eachindex(atoms)
        a = atoms[i]
        write(io, a.symbol)
        write(io, ' ')
        print(io, i)
        write(io, ' ')
        print(io, a.Z)
        for k in eachindex(a.position)
            write(io, ' ')
            _write_float(io, a.position[k])
        end
        write(io, '\n')
    end
    write(io, "[GTO]\n")
    shells = d.shells
    k = firstindex(shells)
    for i in eachindex(atoms)
        print(io, i)
        write(io, " 0\n")
        while k <= lastindex(shells) && shells[k].atom == i
            s = shells[k]
            write(io, _MOLDEN_LABELS[s.l + 1])
            write(io, ' ')
            print(io, length(s.exponents))
            write(io, " 1.00\n")
            for p in eachindex(s.exponents, s.coefficients)
                _write_float(io, s.exponents[p])
                write(io, ' ')
                _write_float(io, s.coefficients[p])
                write(io, '\n')
            end
            k += 1
        end
        write(io, '\n')
    end
    write(io, "[MO]\n")
    _write_orbitals(io, d.alpha, "Alpha")
    isnothing(d.beta) || _write_orbitals(io, d.beta, "Beta")
    return nothing
end

_write_float(io::Base.IO, x::Real) = write(io, string(Float64(x)))

function _write_orbitals(io::Base.IO, orb::MoldenOrbitals, spin::String)
    c = orb.coefficients
    for j in axes(c, 2)
        write(io, "Sym= A\nEne= ")
        _write_float(io, orb.energies[j])
        write(io, "\nSpin= ")
        write(io, spin)
        write(io, "\nOccup= ")
        _write_float(io, orb.occupations[j])
        write(io, '\n')
        for i in axes(c, 1)
            print(io, i)
            write(io, ' ')
            _write_float(io, c[i, j])
            write(io, '\n')
        end
    end
    return nothing
end

# The orbitals of one spin as they are read: energies, occupations and one coefficient column each.
struct _OrbitalLists
    energies::Vector{Float64}
    occupations::Vector{Float64}
    columns::Vector{Vector{Float64}}
end
_OrbitalLists() = _OrbitalLists(Float64[], Float64[], Vector{Float64}[])

function _orbitals(lists::_OrbitalLists, nao::Int)
    c = Matrix{Float64}(undef, nao, length(lists.columns))
    for j in eachindex(lists.columns)
        c[:, j] = lists.columns[j]
    end
    return MoldenOrbitals{Float64}(lists.energies, lists.occupations, c)
end

"""
    read_molden(io::Base.IO; source = "<stream>") -> MoldenData{Float64}

The Molden file in `io`. It starts with `[Molden Format]`; then come the sections `[Title]`
(optional), `[Atoms]`, `[GTO]` and `[MO]`, in that order, each at most once. Section names and
keywords are matched without regard to case.

- The `[Atoms]` unit is `AU`, `(AU)` or `Bohr` (bohr), or `Angs`, `Angstrom` or `(Angstrom)`
  (Å, converted to bohr). Each atom line is `<symbol> <index> <Z> <x> <y> <z>`, with the indices
  1, 2, … in order and `Z` the atomic number of the symbol (0 for a ghost). Each atom has the mass
  `Atom` gives it by default.
- In `[GTO]`, an atom block starts with `<atom> 0` and holds shells `<label> <nprim> [1.00]`,
  each followed by its `<exponent> <coefficient>` lines; the labels are `s`, `p`, `d`, `f` and
  `g`, and the scale factor, when given, is 1.
- In `[MO]`, each orbital has the keyword lines `Ene=` and `Occup=`, and optionally `Sym=` (read
  and dropped) and `Spin=` (`Alpha` or `Beta`; `Alpha` when absent), followed by `<ao> <coefficient>`
  lines with increasing indices; an atomic orbital without a line has coefficient 0. Every shell
  is read as Cartesian.

Numbers may use a Fortran `D` exponent (`1.0D-03`). Errors are `InputError`s naming `source` and
the line; among them, an unknown section, such as `[5D]`, and a number that is not finite.
"""
function read_molden(io::Base.IO; source::AbstractString = "<stream>")
    section = :start
    seen = Set{Symbol}()
    title = ""
    atoms = Atom{Float64}[]
    angstrom = false
    warnings = InputWarning[]
    shells = MoldenShell{Float64}[]
    # The [GTO] state: the atom of the current block, and the shell being read with the line it
    # starts on and the number of primitive lines it still expects.
    gto_atom = 0
    shell_l = 0
    shell_line = 0
    remaining = 0
    exps = Float64[]
    coefs = Float64[]
    # The [MO] state: the orbital being read, with the line it starts on.
    nao = 0
    alpha = _OrbitalLists()
    beta = _OrbitalLists()
    open_orbital = false
    orbital_line = 0
    have_sym = false
    have_ene = false
    have_spin = false
    have_occ = false
    ene = 0.0
    occ = 0.0
    is_beta = false
    last_ao = 0
    column = Float64[]
    n = 0
    for (k, raw) in enumerate(eachline(io))
        n = k
        line, tokens = input_line(raw, n)
        if section === :title_line
            if !startswith(line, "[")
                title = String(line)
                section = :title
                continue
            end
            section = :title
        end
        isempty(tokens) && continue
        if startswith(line, "[")
            j = findfirst(']', line)
            isnothing(j) && throw(InputError(source, n, "molden: a section header ends with ]"))
            name = SubString(line, nextind(line, 1), prevind(line, j))
            rest = strip(SubString(line, nextind(line, j)))
            s = _section(lowercase(name))
            s === :unknown && throw(InputError(source, n, lazy"molden: unknown section [$(name)]"))
            if section === :start
                s === :format || throw(InputError(source, n, "molden: the file starts with [Molden Format]"))
            else
                s === :format && throw(InputError(source, n, "molden: the file starts with [Molden Format]"))
            end
            s in seen && throw(InputError(source, n, lazy"molden: section [$(name)] is given twice"))
            push!(seen, s)
            remaining > 0 && throw(
                InputError(source, shell_line, lazy"molden: the shell declares $(length(exps) + remaining) primitives but has $(length(exps))")
            )
            if s === :atoms
                u = lowercase(rest)
                if u in ("au", "(au)", "bohr")
                    angstrom = false
                elseif u in ("angs", "angstrom", "(angstrom)")
                    angstrom = true
                else
                    throw(InputError(source, n, lazy"molden: unknown [Atoms] unit $(repr(rest))"))
                end
            else
                isempty(rest) || throw(InputError(source, n, lazy"molden: unexpected text after [$(name)]"))
            end
            if s === :gto
                :atoms in seen || throw(InputError(source, n, "molden: [GTO] comes after [Atoms]"))
            elseif s === :mo
                :gto in seen || throw(InputError(source, n, "molden: [MO] comes after [GTO]"))
                for sh in shells
                    nao += _ncartesian(sh.l)
                end
            elseif s === :title
                :atoms in seen && throw(InputError(source, n, "molden: [Title] comes before [Atoms]"))
            end
            section = s === :title ? :title_line : s
            continue
        end
        if section === :start
            throw(InputError(source, n, "molden: the file starts with [Molden Format]"))
        elseif section === :format
            throw(InputError(source, n, "molden: a line outside a section"))
        elseif section === :title
            throw(InputError(source, n, "molden: [Title] holds one line"))
        elseif section === :atoms
            length(tokens) == 6 ||
                throw(InputError(source, n, "molden: expected <symbol> <index> <Z> <x> <y> <z>"))
            idx = _molden_int(source, n, tokens[2])
            idx == length(atoms) + 1 ||
                throw(InputError(source, n, lazy"molden: atom index $(idx), expected $(length(atoms) + 1)"))
            z = _molden_int(source, n, tokens[3])
            scale = angstrom ? ANGSTROM_TO_BOHR : 1.0
            p = SVector(
                _molden_float(source, n, tokens[4]) * scale,
                _molden_float(source, n, tokens[5]) * scale,
                _molden_float(source, n, tokens[6]) * scale,
            )
            a = _atom("molden: ", source, n, false, warnings, tokens[1], p)
            a.Z == z ||
                throw(InputError(source, n, lazy"molden: atomic number $(z) does not match the symbol $(repr(a.symbol))"))
            push!(atoms, a)
        elseif section === :gto
            if remaining > 0
                length(tokens) == 2 || throw(InputError(source, n, "molden: expected an exponent and a coefficient"))
                e = _molden_float(source, n, tokens[1])
                e > 0 || throw(InputError(source, n, lazy"molden: expected a positive exponent, got $(repr(tokens[1]))"))
                push!(exps, e)
                push!(coefs, _molden_float(source, n, tokens[2]))
                remaining -= 1
                if remaining == 0
                    push!(shells, MoldenShell{Float64}(gto_atom, shell_l, exps, coefs))
                    exps = Float64[]
                    coefs = Float64[]
                end
            elseif length(tokens) == 2 && _is_decimal_int(tokens[1])
                a = _molden_int(source, n, tokens[1])
                _molden_int(source, n, tokens[2]) == 0 ||
                    throw(InputError(source, n, "molden: an atom line of [GTO] is <atom> 0"))
                1 <= a <= length(atoms) ||
                    throw(InputError(source, n, lazy"molden: [GTO] names atom $(a), but there are $(length(atoms)) atoms"))
                a > gto_atom ||
                    throw(InputError(source, n, lazy"molden: [GTO] atom $(a) follows atom $(gto_atom)"))
                gto_atom = a
            else
                gto_atom > 0 || throw(InputError(source, n, "molden: a shell line comes after its <atom> 0 line"))
                2 <= length(tokens) <= 3 || throw(InputError(source, n, "molden: expected <label> <nprim> [1.00]"))
                shell_l = _shell_l(tokens[1])
                shell_l < 0 && throw(InputError(source, n, lazy"molden: unknown shell label $(repr(tokens[1]))"))
                remaining = _molden_int(source, n, tokens[2])
                remaining >= 1 || throw(InputError(source, n, "molden: a shell has at least one primitive"))
                length(tokens) == 3 && _molden_float(source, n, tokens[3]) != 1 &&
                    throw(InputError(source, n, lazy"molden: shell scale factor $(repr(tokens[3])) is not 1"))
                shell_line = n
            end
        else # section === :mo
            eq = findfirst('=', line)
            if !isnothing(eq)
                key = lowercase(strip(SubString(line, 1, prevind(line, eq))))
                value = strip(SubString(line, nextind(line, eq)))
                kind = key == "sym" ? 1 : key == "ene" ? 2 : key == "spin" ? 3 : key == "occup" ? 4 : 0
                kind == 0 && throw(InputError(source, n, lazy"molden: unknown [MO] keyword $(repr(key))"))
                given = kind == 1 ? have_sym : kind == 2 ? have_ene : kind == 3 ? have_spin : have_occ
                if open_orbital && (last_ao > 0 || given)
                    _record_orbital(source, orbital_line, is_beta ? beta : alpha, have_ene, have_occ, ene, occ, column)
                    open_orbital = false
                end
                if !open_orbital
                    open_orbital = true
                    orbital_line = n
                    have_sym = have_ene = have_spin = have_occ = false
                    is_beta = false
                    last_ao = 0
                    column = zeros(nao)
                end
                if kind == 1
                    have_sym = true
                elseif kind == 2
                    have_ene = true
                    ene = _molden_float(source, n, value)
                elseif kind == 3
                    have_spin = true
                    v = lowercase(value)
                    v == "alpha" || v == "beta" || throw(InputError(source, n, lazy"molden: unknown spin $(repr(value))"))
                    is_beta = v == "beta"
                else
                    have_occ = true
                    occ = _molden_float(source, n, value)
                end
            else
                open_orbital ||
                    throw(InputError(source, n, "molden: a coefficient line comes after its orbital's keyword lines"))
                length(tokens) == 2 || throw(InputError(source, n, "molden: expected <ao> <coefficient>"))
                ao = _molden_int(source, n, tokens[1])
                1 <= ao <= nao ||
                    throw(InputError(source, n, lazy"molden: atomic orbital $(ao) is outside 1:$(nao)"))
                ao > last_ao ||
                    throw(InputError(source, n, lazy"molden: atomic orbital $(ao) follows $(last_ao)"))
                column[ao] = _molden_float(source, n, tokens[2])
                last_ao = ao
            end
        end
    end
    section === :start && throw(InputError(source, 0, "molden: the file starts with [Molden Format]"))
    remaining > 0 && throw(
        InputError(source, shell_line, lazy"molden: the shell declares $(length(exps) + remaining) primitives but has $(length(exps))")
    )
    open_orbital && _record_orbital(source, orbital_line, is_beta ? beta : alpha, have_ene, have_occ, ene, occ, column)
    isempty(atoms) && throw(InputError(source, 0, "molden: the file has no atoms"))
    isempty(alpha.energies) && throw(InputError(source, 0, "molden: the file has no alpha orbitals"))
    a = _orbitals(alpha, nao)
    isempty(beta.energies) && return MoldenData{Float64}(title, atoms, shells, a, nothing)
    return MoldenData{Float64}(title, atoms, shells, a, _orbitals(beta, nao))
end

function _section(name::AbstractString)
    name == "molden format" && return :format
    name == "title" && return :title
    name == "atoms" && return :atoms
    name == "gto" && return :gto
    name == "mo" && return :mo
    return :unknown
end

# The angular momentum of a shell label, or -1 for a label that is not one of s, p, d, f, g.
function _shell_l(t::AbstractString)
    ncodeunits(t) == 1 || return -1
    c = codeunit(t, 1) | 0x20
    for l in eachindex(_MOLDEN_LABELS)
        c == UInt8(_MOLDEN_LABELS[l]) && return l - 1
    end
    return -1
end

function _record_orbital(
        source::AbstractString, line::Int, lists::_OrbitalLists, have_ene::Bool, have_occ::Bool, ene::Float64,
        occ::Float64, column::Vector{Float64},
    )
    have_ene || throw(InputError(source, line, "molden: the orbital has no Ene= line"))
    have_occ || throw(InputError(source, line, "molden: the orbital has no Occup= line"))
    push!(lists.energies, ene)
    push!(lists.occupations, occ)
    push!(lists.columns, column)
    return nothing
end

# Whether `t` is a decimal integer: an optional sign, then one or more ASCII digits. `tryparse(Int, _)`
# alone also accepts base prefixes such as `0x1`.
function _is_decimal_int(t::AbstractString)
    n = ncodeunits(t)
    i = 1
    n >= 1 && (codeunit(t, 1) == UInt8('+') || codeunit(t, 1) == UInt8('-')) && (i = 2)
    i <= n || return false
    for k in i:n
        UInt8('0') <= codeunit(t, k) <= UInt8('9') || return false
    end
    return true
end

function _molden_int(source::AbstractString, n::Int, t::AbstractString)
    x = _is_decimal_int(t) ? tryparse(Int, t) : nothing
    isnothing(x) && throw(InputError(source, n, lazy"molden: expected an integer, got $(repr(t))"))
    return x
end

# The number `t`, with a Fortran `D` exponent read as `e`; throws for a token that is not a finite
# decimal number.
function _molden_float(source::AbstractString, n::Int, t::AbstractString)
    x = _parse_coordinate(replace(t, 'D' => 'e', 'd' => 'e'))
    isnothing(x) && throw(InputError(source, n, lazy"molden: expected a number, got $(repr(t))"))
    isfinite(x) || throw(InputError(source, n, lazy"molden: expected a finite number, got $(repr(t))"))
    return x
end
