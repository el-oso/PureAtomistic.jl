# The molecular basis built from a basis name: the file lookup, the whole-file read, and the
# placement of each element's shells on its atoms.

"""
    build_basis(mol::Molecule, name::AbstractString; paths = basis_search_paths(), logger = nothing) -> BasisSet{Float64}

The molecular basis of `mol` in the basis `name`, in Float64; the same as
`build_basis(mol, name, Float64; paths, logger)`.
"""
build_basis(
    mol::Molecule, name::AbstractString;
    paths::AbstractVector{<:AbstractString} = basis_search_paths(), logger::Union{Nothing, Logger} = nothing,
) = build_basis(mol, name, Float64; paths, logger)

"""
    build_basis(mol::Molecule, name::AbstractString, ::Type{T}; paths = basis_search_paths(), logger = nothing) -> BasisSet{T}

The molecular basis of `mol` in the basis `name`, with element type `T <: AbstractFloat`.

The file is `locate_basis_file(name; paths)`, read whole with [`parse_gbs`](@ref). Each atom of
`mol.atoms`, in its iteration order, gets the shells of the element `basis_element(atom)` in file
order (a ghost atom gets those of the element it stands for), centered at its position converted
to `T`. A shell's `atom` is the atom's index in `mol.atoms`, whatever the axes of `mol.atoms`. Its
exponents are those of the file converted to `T`, and its coefficients are those of the file
converted to `T` times [`contraction_renorm`](@ref) of the shell computed in `T`. The AO offsets
run from 0 in shell order, and `BasisSet.name` is `name` as given.

With a `logger`, writes the `Normal` line `basis <name>: <file>`. When `name`, stripped and
lowercased, is `def2-svp-jkfit` (the oracle's placeholder, which resolves to
`def2-universal-jkfit`), the note `basis <name>: the oracle's def2-SVP-JKFIT is a placeholder;
using def2-universal-jkfit` comes before it.

Throws a `BasisError` when the name has no file, when the file does not parse, when the file has no
shells for an element of `mol` (`<file>: no basis for element Xe`), when a shell placed on an atom
has `l > 5` (`<file>:<line>: I shell (l = 6) for C; this build supports l ≤ 5`), and when an
exponent of a placed shell is not finite and positive in `T`, or the norm of its contraction is not
a positive finite `T`, naming the shell's line.
"""
function build_basis(
        mol::Molecule, name::AbstractString, ::Type{T};
        paths::AbstractVector{<:AbstractString} = basis_search_paths(), logger::Union{Nothing, Logger} = nothing,
    ) where {T <: AbstractFloat}
    file = locate_basis_file(name; paths)
    data = open(io -> parse_gbs(io; source = file), file)
    bs = _build_basis(mol, data, T, file, String(name))
    if logger !== nothing
        if lowercase(strip(name)) == "def2-svp-jkfit"
            log_print(
                logger, Normal,
                "basis " * name * ": the oracle's def2-SVP-JKFIT is a placeholder; using def2-universal-jkfit",
            )
        end
        log_print(logger, Normal, "basis " * name * ": " * file)
    end
    return bs
end

# The molecular basis of `mol` from `data`, the elements `parse_gbs` read from `file`, in `T`, with
# `BasisSet.name` `name`. `file` names the source in the error messages. It reads no file and
# resolves no name, and keeps every guarantee but `:noalloc`.
function _build_basis(mol::Molecule, data::AbstractDict{String, ElementBasis{Float64}}, ::Type{T}, file::String, name::String) where {T}
    shells = Shell{T}[]
    offset = 0
    for (i, atom) in pairs(mol.atoms)
        sym = element(basis_element(atom)).symbol
        eb = get(data, sym, nothing)
        (eb === nothing || isempty(eb.shells)) && throw(BasisError(lazy"$(file): no basis for element $(sym)"))
        center = SVector{3, T}(atom.position)
        for spec in eb.shells
            push!(shells, _place_shell(T, file, sym, spec, center, i, offset))
            offset += ncart(spec.l)
        end
    end
    return BasisSet{T}(shells, offset, name)
end

# The shell `spec` of element `sym` from `file`, placed at `center` on atom `atom` at AO offset
# `offset`, in `T`. The file's values are finite and positive in Float64, which the reader checked;
# what this checks is that they stay so in `T`.
function _place_shell(::Type{T}, file::String, sym::String, spec::ShellSpec{Float64}, center::SVector{3, T}, atom::Int, offset::Int) where {T}
    l = spec.l
    l <= 5 || throw(
        BasisError(lazy"$(file):$(spec.line): $(SHELL_LETTERS[l + 1]) shell (l = $(l)) for $(sym); this build supports l ≤ 5")
    )
    prims = spec.primitives
    e = Vector{T}(undef, length(prims))
    c = Vector{T}(undef, length(prims))
    for k in eachindex(e, c, prims)
        x = T(prims[k].exponent)
        (isfinite(x) && x > 0) || throw(
            BasisError(lazy"$(file):$(spec.line): exponent $(prims[k].exponent) of the $(SHELL_LETTERS[l + 1]) shell of $(sym) is not finite and positive in $(nameof(T))")
        )
        e[k] = x
        c[k] = T(prims[k].coefficient)
    end
    # `_renorm_sum` needs each exponent squared to be at least `floatmin(T)`; see its comment.
    s = all(x -> x * x >= floatmin(T), e) ? _renorm_sum(e, c, l) : zero(T)
    (s > 0 && isfinite(s)) || throw(
        BasisError(lazy"$(file):$(spec.line): the norm of the $(SHELL_LETTERS[l + 1]) shell of $(sym) is not a positive finite $(nameof(T))")
    )
    r = one(T) / sqrt(s)
    for k in eachindex(c)
        c[k] *= r
    end
    return Shell{T}(center, atom, l, e, c, offset)
end
