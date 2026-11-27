# The `.inp` parser: one pass over the lines, keyword lines outside blocks and content lines inside
# them. Each place where the oracle accepts an input silently calls `_reject`, which throws in
# strict mode and records an `InputWarning` of the named case in lenient mode; each place where
# the oracle itself fails throws an `InputError` in both modes.

# The keywords: the word, the keyword it names (its first word), and the fewest and most values it
# takes on its line, -1 for no limit. `method`, `oniom_high` and `oniom_low` count their values
# themselves (one word, or three for the Libxc forms).
const INP_KEYWORDS = (
    ("title", :title, 1, -1),
    ("run", :run, 1, 1), ("run_mode", :run, 1, 1), ("optimize", :optimize, 0, 0),
    ("charge", :charge, 1, 1), ("multiplicity", :multiplicity, 1, 1),
    ("basis", :basis, 1, 1), ("method", :method, 1, 3),
    ("scf_convergence", :scf_convergence, 1, 1), ("max_iterations", :max_iterations, 1, 1),
    ("diis_method", :diis_method, 1, 1), ("conv_tol_density", :conv_tol_density, 1, 1),
    ("diis_reset", :diis_reset, 1, 1), ("diis_reset_threshold", :diis_reset, 1, 1),
    ("damping", :damping, 1, 1), ("smearing", :smearing, 1, 1), ("smear", :smearing, 1, 1),
    ("grid", :grid, 1, 1), ("grid_level", :grid, 1, 1), ("xc_grid", :grid, 1, 1),
    ("level_shift", :level_shift, 1, 1),
    ("auto_stabilize", :auto_stabilize, 1, 1), ("autostabilize", :auto_stabilize, 1, 1),
    ("robust_convergence", :robust_convergence, 1, 1), ("robust", :robust_convergence, 1, 1),
    ("field", :field, 3, 3), ("electric_field", :field, 3, 3),
    ("density_fitting", :density_fitting, 1, 1), ("ri_j", :density_fitting, 1, 1), ("ri_k", :ri_k, 1, 1),
    ("aux_basis", :aux_basis, 1, 1), ("gpu", :gpu, 1, 1), ("compute_mode", :compute_mode, 1, 1),
    ("checkpoint", :checkpoint, 1, 1),
    ("load_checkpoint", :load_checkpoint, 1, 1), ("restart", :load_checkpoint, 1, 1),
    ("ts_target_mode", :ts_target_mode, 1, 1), ("ts_trust_radius", :ts_trust_radius, 1, 1),
    ("neb_k", :neb_k, 1, 1), ("neb_images", :neb_images, 1, 1), ("climbing_image", :climbing_image, 1, 1),
    ("temperature", :temperature, 1, 1), ("temp", :temperature, 1, 1),
    ("pressure", :pressure, 1, 1), ("press", :pressure, 1, 1),
    ("symmetry_number", :symmetry_number, 1, 1), ("sigma", :symmetry_number, 1, 1),
    ("dispersion", :dispersion, 1, 1), ("zora", :zora, 1, 1), ("zora_relativistic", :zora, 1, 1),
    ("f12", :f12, 1, 1), ("explicit_correlation", :f12, 1, 1),
    ("f12_gamma", :f12_gamma, 1, 1), ("gamma", :f12_gamma, 1, 1), ("gamma_f12", :f12_gamma, 1, 1),
    ("f12_model", :f12_model, 1, 1), ("f12_approximation", :f12_model, 1, 1),
    ("nbo", :nbo, 1, 1), ("admm", :admm, 1, 1), ("admm_basis", :admm_basis, 1, 1),
    ("ecp", :ecp, 1, 1), ("use_ecp", :ecp, 1, 1), ("ecp_name", :ecp_name, 1, 1),
    ("vdw_kernel", :vdw_kernel, 1, 1),
    ("solvation", :solvation, 1, 1), ("solvent", :solvent, 1, 1),
    ("epsilon", :epsilon, 1, 1), ("dielectric", :epsilon, 1, 1),
    ("pcm_radii", :pcm_radii, 1, 1), ("radii_scaling", :pcm_radii, 1, 1),
    ("pcm_grid", :pcm_grid, 1, 1), ("smd_grid", :pcm_grid, 1, 1),
    ("output", :output, 1, 1), ("verbosity", :verbosity, 1, 1),
    ("export_molden", :export_molden, 1, 1), ("export_cube", :export_cube, 1, 1), ("export_wfx", :export_wfx, 1, 1),
    ("population", :population, 1, -1), ("pop_analysis", :population, 1, -1),
    ("pw_cutoff", :pw_cutoff, 1, 1), ("cutoff", :pw_cutoff, 1, 1),
    ("kpoints", :kpoints, 3, 3), ("k_points", :kpoints, 3, 3), ("kgrid", :kpoints, 3, 3),
    ("dos_k_grid", :dos_k_grid, 3, 3), ("dos_kgrid", :dos_k_grid, 3, 3), ("dos_sigma", :dos_sigma, 1, 1),
    ("kpath", :kpath, 1, -1), ("k_path", :kpath, 1, -1),
    ("oniom_high", :oniom_high, 1, 3), ("oniom_low", :oniom_low, 1, 3), ("oniom_model", :oniom_model, 1, -1),
    ("oniom_model_charge", :oniom_model_charge, 1, 1), ("oniom_model_multiplicity", :oniom_model_multiplicity, 1, 1),
    ("freeze_atom", :freeze_atom, 1, -1), ("freeze", :freeze_atom, 1, -1),
    ("freeze_bond", :freeze_bond, 2, 3), ("freeze_angle", :freeze_angle, 3, 4),
    ("freeze_dihedral", :freeze_dihedral, 4, 5), ("freeze_torsion", :freeze_dihedral, 4, 5),
    ("irc_step_size", :irc_step_size, 1, 1), ("irc_step", :irc_step_size, 1, 1),
    ("irc_max_points", :irc_max_points, 1, 1), ("irc_points", :irc_max_points, 1, 1),
    ("irc_direction", :irc_direction, 1, 1),
    ("tddft_states", :tddft_states, 1, 1), ("num_tddft_states", :tddft_states, 1, 1), ("nstates", :tddft_states, 1, 1),
    ("tddft_method", :tddft_method, 1, 1),
    ("compute_soc", :compute_soc, 1, 1), ("soc", :compute_soc, 1, 1),
    ("tddft_triplets", :tddft_triplets, 1, 1), ("triplets", :tddft_triplets, 1, 1),
    ("dmft", :dmft, 1, 1), ("use_dmft", :dmft, 1, 1), ("dmft_temp", :dmft_temp, 1, 1),
    ("dmft_n_freq", :dmft_n_freq, 1, 1), ("dmft_matsubara", :dmft_n_freq, 1, 1),
    ("dmft_n_bath", :dmft_n_bath, 1, 1), ("dmft_bath", :dmft_n_bath, 1, 1),
    ("geometry", :geometry, 0, 1), ("product_geometry", :product_geometry, 0, 1), ("charges", :charges, 0, 1),
    ("hubbard", :hubbard, 0, 0), ("fragments", :fragments, 0, -1), ("cell", :cell, 0, 0), ("pbc", :pbc, 0, 0),
    ("oniom", :oniom, 0, 0),
)

const ACCEPTED_KEYWORDS = sort!(collect(map(first, INP_KEYWORDS)))

# The keywords whose missing or malformed value the oracle rejects (`ok_or(...)?`), an error in
# both modes.
const REQUIRED_KEYWORDS = (
    :charge, :multiplicity, :basis, :method, :scf_convergence, :max_iterations, :ts_target_mode,
    :ts_trust_radius, :neb_k, :neb_images, :temperature, :pressure, :symmetry_number, :oniom_high, :oniom_low,
)

# The keywords that may be given more than once. The geometry blocks check their repetition
# themselves.
const REPEATABLE_KEYWORDS = (
    :freeze_atom, :freeze_bond, :freeze_angle, :freeze_dihedral, :geometry, :product_geometry, :charges, :hubbard,
)

# The keywords with a fixed list of values, whose short list is `:too_few_values`.
const LIST_KEYWORDS = (:field, :kpoints, :dos_k_grid, :freeze_bond, :freeze_angle, :freeze_dihedral)

# The block openers that take no value on their line.
const BARE_BLOCKS = (:hubbard, :cell, :pbc, :oniom)

# The Boolean keywords, whose missing value is `:bool_fallback`.
const BOOL_KEYWORDS = (
    :climbing_image, :density_fitting, :ri_k, :gpu, :load_checkpoint, :dispersion, :zora, :f12, :nbo,
    :auto_stabilize, :robust_convergence, :dmft, :admm, :export_molden, :export_cube, :export_wfx,
    :compute_soc, :tddft_triplets, :ecp,
)

# The Boolean keywords whose missing or malformed value the oracle reads as false
# (`unwrap_or(false)`, `parser.rs:258-266`).
const FALSE_KEYWORDS = (:export_molden, :export_cube, :export_wfx)

# The block openers. In lenient mode an opener inside an open block closes that block and opens
# its own, as the oracle's `geometry`, `product_geometry`, `charges`, `hubbard`, `fragments` and
# `cell` lines end a geometry block (`parser.rs:280-345`).
const BLOCK_KEYWORDS = (:geometry, :product_geometry, :charges, :hubbard, :fragments, :cell, :pbc, :oniom)

"""
    accepted_keywords() -> Vector{String}

Every keyword of an `.inp` file, aliases and block openers included, sorted.
"""
accepted_keywords() = copy(ACCEPTED_KEYWORDS)

# The index in `INP_KEYWORDS` of the lowercase word `w`, or 0.
function _keyword_index(w::AbstractString)
    for i in eachindex(INP_KEYWORDS)
        INP_KEYWORDS[i][1] == w && return i
    end
    return 0
end

# The keyword a repetition is checked under: `optimize` repeats `run`, and `gpu` repeats
# `compute_mode`, because each pair sets one field. The ONIOM block keywords share the keys of
# their top-level forms.
function _duplicate_key(c::Symbol)
    c === :optimize && return :run
    c === :gpu && return :compute_mode
    return c
end

const DEFAULT_METHOD = MethodSpec(HF, AutoReference, (0, 0), 0.0)

# The state of one parse: the input being filled, whether a `title` keyword has set its title, the
# open block and the line it opened at, the first line of each keyword seen, and the values that
# are resolved after the pass (solvation and ONIOM). `seq` is the solvation the oracle reads, one
# line after the other, which lenient mode returns.
mutable struct _InpState
    source::String
    lenient::Bool
    warnings::Vector{InputWarning}
    d::_InputDraft
    titled::Bool
    seen::Dict{Symbol, Int}
    block::Symbol
    block_line::Int
    block_lines::Int
    bohr::Bool
    first_geometry::Int
    first_product::Int
    cell::Vector{SVector{3, Float64}}
    hubbard_lines::Vector{Int}
    solvation::SolvationModel
    solvent::Symbol
    epsilon::Float64
    radii::Float64
    grid::Int
    seq::SolvationSpec
    oniom::Bool
    high::MethodSpec
    low::MethodSpec
    model_atoms::Vector{Int}
    model_charge::Union{Nothing, Int}
    model_multiplicity::Union{Nothing, Int}
    sites::_IndexSites
end

function _InpState(source::String, lenient::Bool, warnings::Vector{InputWarning})
    s = SolvationSpec()
    return _InpState(
        source, lenient, warnings, _draft(InputFile()), false, Dict{Symbol, Int}(), :none, 0, 0, false, 0, 0,
        SVector{3, Float64}[], Int[], NoSolvation, s.solvent, s.epsilon, s.radii_scaling, s.grid_order,
        s, false, DEFAULT_METHOD, DEFAULT_METHOD, Int[], nothing, nothing, _IndexSites(),
    )
end

_reject(st::_InpState, n::Int, case::Symbol, msg::AbstractString) =
    reject_or_warn(st.source, n, st.lenient, st.warnings, case, msg)

"""
    parse_inp(text::AbstractString; source = "<string>", lenient = false) -> ParsedInput

The input file `text` in the `.inp` format, read strictly by default. `source` names the file in
error messages and warnings. Lines are `keyword value…`, case-insensitive in the keyword, with
blocks (`geometry [unit]`, `product_geometry [unit]`, `charges [unit]`, `hubbard`, `fragments`,
`cell`, `pbc`, `oniom`) closed by `end`; a `#` starts a comment, and the first comment line gives
the title unless a `title` keyword does. The grammar, its deviations from the oracle and every
keyword are documented on the Input page of the manual.

In strict mode every input the oracle accepts silently but that is not valid is an `InputError`
naming the line. In lenient mode (`lenient = true`) such an input is read as the oracle reads it
and recorded as an `InputWarning` in the result. Atom positions are converted to bohr.
"""
function parse_inp(text::AbstractString; source::AbstractString = "<string>", lenient::Bool = false)
    st = _InpState(String(source), lenient, InputWarning[])
    for (n, raw) in enumerate(eachsplit(String(text), '\n'))
        _line!(st, n, raw)
    end
    if st.block !== :none
        l = st.block_line
        _reject(st, l, :unclosed_block, lazy"the $(st.block) block opened at line $(l) is not closed by `end`")
        _close_block!(st)
    end
    _resolve_solvation!(st)
    _check_f12(st)
    oniom = st.oniom ? ONIOMSpec(st.high, st.low, st.model_atoms, st.model_charge, st.model_multiplicity) : nothing
    inp = _finalize_input(
        _input(st.d, oniom); source = st.source, lenient, warnings = st.warnings,
        method_line = get(st.seen, :method, 0), multiplicity_line = get(st.seen, :multiplicity, 0),
        sites = st.sites,
    )
    return ParsedInput(inp, st.warnings)
end

_is_comment(t::AbstractString) = startswith(t, '#')

# Line `n`: the title of a comment line, a type-confusion check, then a block content line or a
# keyword line.
function _line!(st::_InpState, n::Int, raw::SubString{String})
    line, tokens = input_line(raw, n)
    isempty(line) && return nothing
    if startswith(line, '#')
        st.titled || !isempty(st.d.title) || (st.d.title = String(strip(line[nextind(line, 1):end])))
        return nothing
    end
    k = findfirst(_is_comment, tokens)
    isnothing(k) || resize!(tokens, k - 1)
    isempty(tokens) && return nothing
    t1 = tokens[1]
    if endswith(t1, ':') || startswith(line, "- ") || startswith(line, "-\t")
        _reject(
            st, n, :yaml_in_inp,
            lazy"`$(t1)` looks like YAML; .inp files use `keyword value` lines, and YAML input needs a .yaml or .yml extension",
        )
        return nothing
    end
    kw = lowercase(t1)
    if st.block === :none
        _keyword_line!(st, n, tokens, kw)
    else
        _block_line!(st, n, tokens, kw)
    end
    return nothing
end

# Value readers. `kw` names the keyword or block in the message. A `_required` reader throws for a
# token that does not parse (the oracle fails there too); an `_optional` reader rejects it as
# `case` and returns `nothing`. Every float reader rejects a non-finite value as `:nonfinite` and,
# in lenient mode, returns it.

# Whether `t` is an optional sign followed by one or more ASCII digits.
function _is_decimal_int(t::AbstractString)
    n = ncodeunits(t)
    i = n >= 1 && (codeunit(t, 1) == UInt8('+') || codeunit(t, 1) == UInt8('-')) ? 2 : 1
    i <= n || return false
    for j in i:n
        UInt8('0') <= codeunit(t, j) <= UInt8('9') || return false
    end
    return true
end

# The words `words[i:end]` joined by `delim`. Written out with `write` calls on the concrete word
# and delimiter types, which `Base.join` reaches only through a runtime dispatch.
function _join(words::Vector{S}, delim::Union{Char, String}, i::Int) where {S <: AbstractString}
    io = IOBuffer()
    for j in i:lastindex(words)
        j > i && write(io, delim)
        write(io, words[j])
    end
    return String(take!(io))
end

_parse_int(t::AbstractString) = _is_decimal_int(t) ? tryparse(Int, t) : nothing

function _finite(st::_InpState, n::Int, kw, t::AbstractString, x::Float64)
    isfinite(x) || _reject(st, n, :nonfinite, lazy"`$(kw)`: expected a finite number, got $(repr(t))")
    return x
end

function _float_required(st::_InpState, n::Int, kw, t::AbstractString)
    x = parse_coordinate(t)
    isnothing(x) && throw(InputError(st.source, n, lazy"`$(kw)`: expected a number, got $(repr(t))"))
    return _finite(st, n, kw, t, x)
end

function _float_optional(st::_InpState, n::Int, kw, t::AbstractString, case::Symbol)
    x = parse_coordinate(t)
    if isnothing(x)
        _reject(st, n, case, lazy"`$(kw)`: expected a number, got $(repr(t))")
        return nothing
    end
    return _finite(st, n, kw, t, x)
end

# A float, or `fallback` for a token that does not parse: the oracle's `unwrap_or(fallback)`.
function _float_or(st::_InpState, n::Int, kw, t::AbstractString, case::Symbol, fallback::Float64)
    x = _float_optional(st, n, kw, t, case)
    return isnothing(x) ? fallback : x
end

# `t` as Rust's `parse::<usize>` reads it (a decimal with an optional `+`), or `nothing`.
_parse_usize(t::AbstractString) = startswith(t, '-') ? nothing : _parse_int(t)

# The message for a token that `_parse_usize` rejects.
function _uint_msg(kw, t::AbstractString)
    isnothing(_parse_int(t)) && return lazy"`$(kw)`: expected an integer, got $(repr(t))"
    return lazy"`$(kw)`: expected a non-negative integer, got $(repr(t))"
end

# The readers of the fields the oracle stores as `usize`, where a negative token fails to parse.
function _uint_required(st::_InpState, n::Int, kw, t::AbstractString)
    x = _parse_usize(t)
    isnothing(x) && throw(InputError(st.source, n, _uint_msg(kw, t)))
    return x
end

function _uint_optional(st::_InpState, n::Int, kw, t::AbstractString, case::Symbol)
    x = _parse_usize(t)
    isnothing(x) && _reject(st, n, case, _uint_msg(kw, t))
    return x
end

# An unsigned integer, or `fallback` for a token that does not parse: the oracle's `unwrap_or(fallback)`.
function _uint_or(st::_InpState, n::Int, kw, t::AbstractString, case::Symbol, fallback::Int)
    x = _uint_optional(st, n, kw, t, case)
    return isnothing(x) ? fallback : x
end

function _int_required(st::_InpState, n::Int, kw, t::AbstractString)
    x = _parse_int(t)
    isnothing(x) && throw(InputError(st.source, n, lazy"`$(kw)`: expected an integer, got $(repr(t))"))
    return x
end

# The readers of the fields the oracle stores as `i32`, where a token outside the `Int32` range
# fails to parse: `charge` is then an oracle error (`ok_or`), and an ONIOM model charge the
# oracle's `parse().ok()` fallback, `nothing`.
_int32_msg(kw, t::AbstractString) = lazy"`$(kw)`: expected an integer in the Int32 range, got $(repr(t))"

function _int32_required(st::_InpState, n::Int, kw, t::AbstractString)
    x = _int_required(st, n, kw, t)
    typemin(Int32) <= x <= typemax(Int32) || throw(InputError(st.source, n, _int32_msg(kw, t)))
    return x
end

function _int32_optional(st::_InpState, n::Int, kw, t::AbstractString, case::Symbol)
    x = _int_optional(st, n, kw, t, case)
    (isnothing(x) || typemin(Int32) <= x <= typemax(Int32)) && return x
    _reject(st, n, case, _int32_msg(kw, t))
    return nothing
end

function _int_optional(st::_InpState, n::Int, kw, t::AbstractString, case::Symbol)
    x = _parse_int(t)
    isnothing(x) && _reject(st, n, case, lazy"`$(kw)`: expected an integer, got $(repr(t))")
    return x
end

# A 0-based atom index of the file, returned 1-based.
function _index(st::_InpState, n::Int, kw, t::AbstractString, case::Symbol)
    x = _parse_int(t)
    if isnothing(x) || x < 0
        _reject(st, n, case, lazy"`$(kw)`: expected an atom index (an integer ≥ 0), got $(repr(t))")
        return nothing
    end
    # The 1-based index of `typemax(Int)` is not an `Int`; it fails in both modes.
    x == typemax(Int) &&
        throw(InputError(st.source, n, lazy"`$(kw)`: expected an atom index (an integer ≥ 0), got $(repr(t))"))
    return x + 1
end

function _bool(st::_InpState, n::Int, kw, t::AbstractString)
    t == "true" && return true
    t == "false" && return false
    _reject(st, n, :bool_fallback, lazy"`$(kw)`: expected true or false, got $(repr(t))")
    return nothing
end

# The oracle's value for Boolean keyword `c` whose token `t` is not `true` or `false`: Rust's
# `parse::<bool>` fails, and `unwrap_or` gives true, or false for `FALSE_KEYWORDS`;
# `compute_soc` and `tddft_triplets` compare the token with `"true"` and `"1"`
# (`parser.rs:515-520`).
function _bool_fallback(c::Symbol, t::AbstractString)
    c in FALSE_KEYWORDS && return false
    (c === :compute_soc || c === :tddft_triplets) && return t == "1"
    return true
end

# Sets the field of Boolean keyword `c` to `x`; `gpu` sets the compute mode.
function _set_bool!(st::_InpState, c::Symbol, x::Bool)
    d = st.d
    if c === :climbing_image
        d.climbing_image = x
    elseif c === :density_fitting
        d.use_ri_j = x
    elseif c === :ri_k
        d.use_ri_k = x
    elseif c === :gpu
        d.compute_mode = x ? GPU : CPU
    elseif c === :load_checkpoint
        d.load_checkpoint = x
    elseif c === :dispersion
        d.dispersion = x
    elseif c === :zora
        d.zora = x
    elseif c === :f12
        d.use_f12 = x
    elseif c === :nbo
        d.nbo = x
    elseif c === :auto_stabilize
        d.auto_stabilize = x
    elseif c === :robust_convergence
        d.robust_convergence = x
    elseif c === :dmft
        d.use_dmft = x
    elseif c === :admm
        d.admm = x
    elseif c === :export_molden
        d.export_molden = x
    elseif c === :export_cube
        d.export_cube = x
    elseif c === :export_wfx
        d.export_wfx = x
    elseif c === :compute_soc
        d.compute_soc = x
    elseif c === :tddft_triplets
        d.tddft_triplets = x
    else
        d.use_ecp = x
    end
    return nothing
end

function _unknown_msg(kw, t::AbstractString, accepted::Vector{String})
    words = _join(accepted, ", ", 1)
    return lazy"unknown value `$(t)` for `$(kw)`; accepted: $(words)"
end

# Rejects the unknown word `t` of keyword `kw` (the vocabulary function returned `nothing`).
_unknown(st::_InpState, n::Int, kw, t::AbstractString, accepted::Vector{String}) =
    _reject(st, n, :unknown_value, _unknown_msg(kw, t, accepted))

# Whether the `nv` values of keyword `kw` are as many as it takes; rejects the line otherwise. An
# excess is read up to the count in lenient mode (`true`), as the oracle ignores it; a shortfall is
# an error in both modes for a required keyword and is skipped in lenient mode (`false`).
function _count_ok(st::_InpState, n::Int, kw::String, c::Symbol, nv::Int, lo::Int, hi::Int)
    nv >= lo && (hi < 0 || nv <= hi) && return true
    if hi == 0 && c in BARE_BLOCKS
        _reject(st, n, c === :hubbard ? :hubbard_inline_argument : :extra_values, lazy"`$(kw)` takes no value on its line")
        return true
    end
    msg = if hi < 0
        lazy"`$(kw)` needs a value"
    elseif lo == hi
        lo == 1 ? lazy"`$(kw)` takes 1 value, got $(nv)" : lazy"`$(kw)` takes $(lo) values, got $(nv)"
    else
        lazy"`$(kw)` takes $(lo) or $(hi) values, got $(nv)"
    end
    if hi >= 0 && nv > hi
        _reject(st, n, :extra_values, msg)
        return true
    end
    c in REQUIRED_KEYWORDS && throw(InputError(st.source, n, msg))
    case = c in LIST_KEYWORDS ? :too_few_values : c in BOOL_KEYWORDS ? :bool_fallback : :missing_value
    _reject(st, n, case, msg)
    return false
end

# Whether `w` starts with `casscf` in any ASCII case.
function _is_casscf(w::AbstractString)
    ncodeunits(w) >= 6 || return false
    for (i, c) in enumerate(codeunits("casscf"))
        _lower(codeunit(w, i)) == c || return false
    end
    return true
end

_is_paren(c::Char) = c == '(' || c == ')'

# The active space the oracle reads from a `casscf` word that `method_word` rejects
# (`parser.rs:1154-1163`): the word lowercased, every leading `casscf` and every `(` and `)` at
# either end removed, and the rest split at `,`. Two parts are the electron and orbital counts,
# which must be unsigned integers; any other number of parts gives (2, 2). `_casscf_counts`
# returns `(status, e, o)`: status 0 for counts read, 1 for a malformed electron count and 2 for a
# malformed orbital count.
function _casscf_counts(w::AbstractString)
    s = SubString(lowercase(w))
    while startswith(s, "casscf")
        s = chopprefix(s, "casscf")
    end
    parts = split(strip(_is_paren, s), ',')
    length(parts) == 2 || return (0, 2, 2)
    e = _parse_usize(strip(parts[1]))
    isnothing(e) && return (1, 0, 0)
    o = _parse_usize(strip(parts[2]))
    isnothing(o) && return (2, 0, 0)
    return (0, e, o)
end

function _casscf_oracle(st::_InpState, n::Int, w::AbstractString)
    status, e, o = _casscf_counts(w)
    status == 1 && throw(InputError(st.source, n, "Invalid CASSCF electron count"))
    status == 2 && throw(InputError(st.source, n, "Invalid CASSCF orbital count"))
    return (e, o)
end

# The method of a `method`, `high`, `low`, `oniom_high` or `oniom_low` line.
function _method_value(st::_InpState, n::Int, kw::String, tokens::Vector{SubString{String}})
    nv = length(tokens) - 1
    nv == 0 && throw(InputError(st.source, n, lazy"`$(kw)` needs a value"))
    w = tokens[2]
    lx = _lookup(w, LIBXC_WORDS)
    if isnothing(lx)
        nv > 1 && _reject(st, n, :extra_values, lazy"`$(kw)` takes 1 value, got $(nv)")
    else
        nv < 3 && throw(InputError(st.source, n, lazy"`$(kw)` takes 3 values, got $(nv)"))
        nv > 3 && _reject(st, n, :extra_values, lazy"`$(kw)` takes 3 values, got $(nv)")
    end
    m = method_word(view(tokens, 2:lastindex(tokens)))
    if isnothing(m)
        if _is_casscf(w)
            _reject(st, n, :casscf_shape, lazy"casscf needs (electrons,orbitals), got `$(w)`")
            return MethodSpec(CASSCF, AutoReference, _casscf_oracle(st, n, w), 0.0)
        end
        isnothing(lx) && throw(InputError(st.source, n, _unknown_msg(kw, w, accepted_methods())))
        msg = lx[2] ? lazy"`$(kw)`: `$(lowercase(w))` takes an integer Libxc ID and the exact-exchange fraction" :
            lazy"`$(kw)`: `$(lowercase(w))` takes 2 integer Libxc IDs"
        throw(InputError(st.source, n, msg))
    end
    (m.kind == LIBXC_HGGA || m.kind == LIBXC_HMGGA) && _finite(st, n, kw, tokens[4], m.exx)
    return m
end

# A keyword line outside blocks.
function _keyword_line!(st::_InpState, n::Int, tokens::Vector{SubString{String}}, kw::String)
    i = _keyword_index(kw)
    if i == 0
        if kw == "end"
            _reject(st, n, :unknown_keyword, "`end` closes no block")
            return nothing
        end
        words = _join(ACCEPTED_KEYWORDS, ", ", 1)
        _reject(st, n, :unknown_keyword, lazy"unknown keyword `$(tokens[1])`; accepted keywords: $(words)")
        return nothing
    end
    _, c, lo, hi = INP_KEYWORDS[i]
    if !(c in REPEATABLE_KEYWORDS)
        key = _duplicate_key(c)
        first = get(st.seen, key, 0)
        first == 0 || _reject(st, n, :duplicate_keyword, lazy"keyword `$(kw)` given twice (lines $(first) and $(n))")
        st.seen[key] = n
    end
    if c === :method || c === :oniom_high || c === :oniom_low
        m = _method_value(st, n, kw, tokens)
        isnothing(m) && return nothing
        if c === :method
            st.d.method = m
        else
            st.oniom = true
            c === :oniom_high ? (st.high = m) : (st.low = m)
        end
        return nothing
    end
    _count_ok(st, n, kw, c, length(tokens) - 1, lo, hi) || return _missing!(st, c)
    _keyword_value!(st, n, kw, c, tokens)
    return nothing
end

# The value the oracle takes for keyword `c` given without its value, or for a list keyword with
# too few values; reached in lenient mode only. Every keyword not named keeps its value.
function _missing!(st::_InpState, c::Symbol)
    d = st.d
    if c in BOOL_KEYWORDS
        _set_bool!(st, c, !(c in FALSE_KEYWORDS))
    elseif c === :checkpoint
        d.checkpoint = nothing
    elseif c === :aux_basis
        d.aux_basis = nothing
    elseif c === :admm_basis
        d.admm_basis = nothing
    elseif c === :f12_gamma
        d.f12_gamma = 1.0
    elseif c === :f12_model
        d.f12_model = "F12b"
    elseif c === :damping
        d.damping = 0.0
    elseif c === :smearing
        d.smearing = 0.0
    elseif c === :level_shift
        d.level_shift = 0.0
    elseif c === :diis_reset
        d.diis_reset_threshold = 0.1
    elseif c === :dos_sigma
        d.dos_sigma = 0.1
    elseif c === :verbosity
        d.verbosity = 1
    elseif c === :grid
        d.grid_level = 1
    elseif c === :conv_tol_density
        d.conv_tol_density = nothing
    elseif c === :pw_cutoff
        d.pw_cutoff = nothing
    elseif c === :solvent
        _set_solvent!(st, :water)
    elseif c === :epsilon
        _set_epsilon!(st, 78.39)
    elseif c === :pcm_radii
        _set_radii!(st, 1.2)
    elseif c === :pcm_grid
        _set_pcm_grid!(st, 110)
    elseif c === :output
        d.output = ""
    elseif c === :vdw_kernel
        d.vdw_kernel = nothing
    elseif c === :kpath
        d.kpath = String[]
    elseif c === :population
        d.population = PopulationKind[]
    elseif c === :oniom_model
        st.oniom = true
    elseif c === :oniom_model_charge
        st.oniom = true
        st.model_charge = nothing
    elseif c === :oniom_model_multiplicity
        st.oniom = true
        st.model_multiplicity = nothing
    elseif c === :ecp_name
        d.use_ecp = true
        d.ecp_name = nothing
    end
    return nothing
end

# The solvation settings, kept twice: by keyword for the order-independent reading, and in `seq`
# as the oracle reads the lines in order (`parser.rs:202-251`).

# `solvation <model>`: the oracle replaces the settings by the model's defaults.
function _set_solvation!(st::_InpState, x::SolvationModel)
    st.solvation = x
    st.seq = x == NoSolvation ? SolvationSpec() : SolvationSpec(; model = x)
    return nothing
end

# `solvent <name>`: the oracle sets the solvent of an SMD model, or replaces any other model by
# SMD with its defaults and that solvent.
function _set_solvent!(st::_InpState, x::Symbol)
    st.solvent = x
    s = st.seq
    st.seq = s.model == SMD ?
        SolvationSpec(SMD, x, 78.39, false, s.radii_scaling, s.grid_order, s.surface, s.probe_radius) :
        SolvationSpec(; model = SMD, solvent = x)
    return nothing
end

# `epsilon <x>`: the oracle sets the dielectric of PCM, replaces that of the SMD solvent, or
# without a model makes PCM with its defaults and that dielectric.
function _set_epsilon!(st::_InpState, x::Float64)
    st.epsilon = x
    s = st.seq
    st.seq = s.model == NoSolvation ? SolvationSpec(; model = PCM, epsilon = x) :
        SolvationSpec(s.model, s.solvent, x, s.model == SMD, s.radii_scaling, s.grid_order, s.surface, s.probe_radius)
    return nothing
end

# `pcm_radii <x>`: the oracle applies it under PCM only.
function _set_radii!(st::_InpState, x::Float64)
    st.radii = x
    s = st.seq
    s.model == PCM &&
        (st.seq = SolvationSpec(PCM, s.solvent, s.epsilon, s.custom_epsilon, x, s.grid_order, s.surface, s.probe_radius))
    return nothing
end

# `pcm_grid <g>`: the oracle applies it under PCM or SMD.
function _set_pcm_grid!(st::_InpState, g::Int)
    st.grid = g
    s = st.seq
    s.model == NoSolvation ||
        (st.seq = SolvationSpec(s.model, s.solvent, s.epsilon, s.custom_epsilon, s.radii_scaling, g, s.surface, s.probe_radius))
    return nothing
end

# The value of keyword `c` (written `kw`) on line `n`, whose value count is checked.
function _keyword_value!(st::_InpState, n::Int, kw::String, c::Symbol, tokens::Vector{SubString{String}})
    d = st.d
    nv = length(tokens) - 1
    t = nv >= 1 ? tokens[2] : tokens[1]
    if c in BOOL_KEYWORDS && c !== :ecp
        x = _bool(st, n, kw, t)
        _set_bool!(st, c, isnothing(x) ? _bool_fallback(c, t) : x)
    elseif c === :title
        title = _join(tokens, ' ', 2)
        if ncodeunits(title) >= 2 && startswith(title, '"') && endswith(title, '"')
            title = title[2:prevind(title, lastindex(title))]
        end
        d.title = title
        st.titled = true
    elseif c === :run
        x = run_mode_word(t)
        isnothing(x) && throw(InputError(st.source, n, _unknown_msg(kw, t, accepted_run_modes())))
        d.run_mode = x
    elseif c === :optimize
        d.run_mode = Optimization
    elseif c === :charge
        d.charge = _int32_required(st, n, kw, t)
    elseif c === :multiplicity
        d.multiplicity = _uint_required(st, n, kw, t)
    elseif c === :basis
        d.basis = lowercase(t)
    elseif c === :scf_convergence
        d.scf_convergence = _float_required(st, n, kw, t)
    elseif c === :max_iterations
        d.max_iterations = _uint_required(st, n, kw, t)
    elseif c === :diis_method
        x = diis_word(t)
        if isnothing(x)
            # The oracle stores the word, and no code reads it: the extrapolation stays CDIIS.
            _unknown(st, n, kw, t, accepted_diis_methods())
            d.diis_method = CDIIS
        else
            d.diis_method = x
        end
    elseif c === :conv_tol_density
        d.conv_tol_density = _float_optional(st, n, kw, t, :number_fallback)
    elseif c === :diis_reset
        d.diis_reset_threshold = _float_or(st, n, kw, t, :number_fallback, 0.1)
    elseif c === :damping
        d.damping = _float_or(st, n, kw, t, :number_fallback, 0.0)
    elseif c === :smearing
        d.smearing = _float_or(st, n, kw, t, :number_fallback, 0.0)
    elseif c === :grid
        x = grid_word(t)
        if isnothing(x)
            # The oracle reads any other unsigned integer as the level, and 1 for the rest
            # (`parser.rs:168-178`).
            _unknown(st, n, kw, t, accepted_grids())
            g = _parse_usize(t)
            d.grid_level = isnothing(g) ? 1 : g
        else
            d.grid_level = x
        end
    elseif c === :level_shift
        d.level_shift = _float_or(st, n, kw, t, :number_fallback, 0.0)
    elseif c === :field
        ex = _float_or(st, n, kw, tokens[2], :bad_list_entry, 0.0)
        ey = _float_or(st, n, kw, tokens[3], :bad_list_entry, 0.0)
        ez = _float_or(st, n, kw, tokens[4], :bad_list_entry, 0.0)
        d.electric_field = SVector(ex, ey, ez)
    elseif c === :aux_basis
        d.aux_basis = String(t)
    elseif c === :compute_mode
        x = compute_mode_word(t)
        isnothing(x) ? _unknown(st, n, kw, t, accepted_compute_modes()) : (d.compute_mode = x)
    elseif c === :checkpoint
        d.checkpoint = String(t)
    elseif c === :ts_target_mode
        d.ts_target_mode = _uint_required(st, n, kw, t)
    elseif c === :ts_trust_radius
        d.ts_trust_radius = _float_required(st, n, kw, t)
    elseif c === :neb_k
        d.neb_k = _float_required(st, n, kw, t)
    elseif c === :neb_images
        d.neb_images = _uint_required(st, n, kw, t)
    elseif c === :temperature
        d.temperature = _float_required(st, n, kw, t)
    elseif c === :pressure
        d.pressure = _float_required(st, n, kw, t)
    elseif c === :symmetry_number
        d.symmetry_number = _float_required(st, n, kw, t)
    elseif c === :f12_gamma
        d.f12_gamma = _float_or(st, n, kw, t, :number_fallback, 1.0)
    elseif c === :f12_model
        d.f12_model = String(t)
    elseif c === :admm_basis
        d.admm_basis = String(t)
    elseif c === :ecp
        if t == "true" || t == "false"
            d.use_ecp = t == "true"
        else
            d.use_ecp = true
            d.ecp_name = String(t)
        end
    elseif c === :ecp_name
        d.use_ecp = true
        d.ecp_name = String(t)
    elseif c === :vdw_kernel
        x = vdw_kernel_word(t)
        # The oracle stores an unknown word and computes no dispersion for it.
        isnothing(x) && _unknown(st, n, kw, t, accepted_vdw_kernels())
        d.vdw_kernel = x
    elseif c === :solvation
        x = solvation_word(t)
        isnothing(x) ? _unknown(st, n, kw, t, accepted_solvation_models()) : _set_solvation!(st, x)
    elseif c === :solvent
        x = solvent_word(t)
        isnothing(x) ? _unknown(st, n, kw, t, accepted_solvents()) : _set_solvent!(st, x)
    elseif c === :epsilon
        _set_epsilon!(st, _float_or(st, n, kw, t, :number_fallback, 78.39))
    elseif c === :pcm_radii
        _set_radii!(st, _float_or(st, n, kw, t, :number_fallback, 1.2))
    elseif c === :pcm_grid
        _set_pcm_grid!(st, _uint_or(st, n, kw, t, :number_fallback, 110))
    elseif c === :output
        d.output = String(t)
    elseif c === :verbosity
        d.verbosity = _uint_or(st, n, kw, t, :number_fallback, 1)
    elseif c === :population
        pop = PopulationKind[]
        bad = false
        for j in 2:lastindex(tokens)
            x = population_word(tokens[j])
            if isnothing(x)
                _unknown(st, n, kw, tokens[j], accepted_populations())
                bad = true
            else
                push!(pop, x)
            end
        end
        # The oracle stores the words, and no code reads them: the analyses stay the default.
        d.population = bad ? PopulationKind[Mulliken, Lowdin] : pop
    elseif c === :pw_cutoff
        d.pw_cutoff = _float_optional(st, n, kw, t, :number_fallback)
    elseif c === :kpoints || c === :dos_k_grid
        f = c === :kpoints ? 1 : 10
        k3 = (
            _uint_or(st, n, kw, tokens[2], :bad_list_entry, f), _uint_or(st, n, kw, tokens[3], :bad_list_entry, f),
            _uint_or(st, n, kw, tokens[4], :bad_list_entry, f),
        )
        c === :kpoints ? (d.kpoints = k3) : (d.dos_k_grid = k3)
    elseif c === :dos_sigma
        d.dos_sigma = _float_or(st, n, kw, t, :number_fallback, 0.1)
    elseif c === :kpath
        path = String[]
        for j in 2:lastindex(tokens)
            push!(path, String(tokens[j]))
        end
        d.kpath = path
    elseif c === :oniom_model
        st.oniom = true
        _model_atoms!(st, n, kw, tokens)
    elseif c === :oniom_model_charge
        st.oniom = true
        st.model_charge = _int32_optional(st, n, kw, t, :number_fallback)
    elseif c === :oniom_model_multiplicity
        st.oniom = true
        st.model_multiplicity = _uint_optional(st, n, kw, t, :number_fallback)
    elseif c === :freeze_atom
        for j in 2:lastindex(tokens)
            a = _index(st, n, kw, tokens[j], :bad_list_entry)
            isnothing(a) && continue
            push!(d.constraints, Constraint(FreezeAtom, (a, 0, 0, 0), nothing))
            _constraint_site!(st, n, kw)
        end
    elseif c === :freeze_bond || c === :freeze_angle || c === :freeze_dihedral
        _freeze!(st, n, kw, c, tokens)
    elseif c === :irc_step_size
        x = _float_optional(st, n, kw, t, :number_fallback)
        isnothing(x) || (d.irc_step_size = x)
    elseif c === :irc_max_points
        x = _uint_optional(st, n, kw, t, :number_fallback)
        isnothing(x) || (d.irc_max_points = x)
    elseif c === :irc_direction
        x = irc_direction_word(t)
        # The oracle reads every word but the forward and backward ones as Both (`parser.rs:466-474`).
        isnothing(x) && _unknown(st, n, kw, t, accepted_irc_directions())
        d.irc_direction = isnothing(x) ? Both : x
    elseif c === :tddft_states
        x = _uint_optional(st, n, kw, t, :number_fallback)
        isnothing(x) || (d.tddft_states = x)
    elseif c === :tddft_method
        x = tddft_method_word(t)
        # The oracle stores the word, and its TDDFT driver runs TDA for an unknown one.
        isnothing(x) && _unknown(st, n, kw, t, accepted_tddft_methods())
        d.tddft_method = isnothing(x) ? TDA : x
    elseif c === :dmft_temp
        x = _float_optional(st, n, kw, t, :number_fallback)
        isnothing(x) || (d.dmft_temperature = x)
    elseif c === :dmft_n_freq
        x = _uint_optional(st, n, kw, t, :number_fallback)
        isnothing(x) || (d.dmft_n_freq = x)
    elseif c === :dmft_n_bath
        x = _uint_optional(st, n, kw, t, :number_fallback)
        isnothing(x) || (d.dmft_n_bath = x)
    elseif c === :fragments
        _fragments_keyword!(st, n, tokens)
    else
        _open_block!(st, n, c, tokens)
    end
    return nothing
end

# The atom indices of an `oniom_model`, `model_atoms` or `model_indices` line.
function _model_atoms!(st::_InpState, n::Int, kw::String, tokens::Vector{SubString{String}})
    for j in 2:lastindex(tokens)
        a = _index(st, n, kw, tokens[j], :bad_list_entry)
        isnothing(a) && continue
        push!(st.model_atoms, a)
        push!(st.sites.model_lines, n)
        push!(st.sites.model_labels, "`" * kw * "`")
    end
    return nothing
end

# A `freeze_bond`, `freeze_angle` or `freeze_dihedral` line: 2, 3 or 4 atom indices and an
# optional target, in bohr for a bond and in degrees (stored in radians) for an angle.
function _freeze!(st::_InpState, n::Int, kw::String, c::Symbol, tokens::Vector{SubString{String}})
    k = c === :freeze_bond ? 2 : c === :freeze_angle ? 3 : 4
    atoms = (0, 0, 0, 0)
    for j in 1:k
        a = _index(st, n, kw, tokens[j + 1], :bad_list_entry)
        isnothing(a) && return nothing
        atoms = Base.setindex(atoms, a, j)
    end
    target = nothing
    if length(tokens) >= k + 2
        target = _float_optional(st, n, kw, tokens[k + 2], :bad_list_entry)
        isnothing(target) || c === :freeze_bond || (target = deg2rad(target))
    end
    kind = c === :freeze_bond ? FreezeBond : c === :freeze_angle ? FreezeAngle : FreezeDihedral
    push!(st.d.constraints, Constraint(kind, atoms, target))
    _constraint_site!(st, n, kw)
    return nothing
end

function _constraint_site!(st::_InpState, n::Int, kw::String)
    push!(st.sites.constraint_lines, n)
    push!(st.sites.constraint_labels, "`" * kw * "`")
    return nothing
end

# Appends the fragments `frags` of line `n`.
function _add_fragments!(st::_InpState, n::Int, frags::Vector{Vector{Int}})
    append!(st.d.fragments, frags)
    for _ in eachindex(frags)
        push!(st.sites.fragment_lines, n)
    end
    return nothing
end

# Splits `text` into fragments at `;` and each into 0-based atom indices at `,`, returned 1-based.
# An empty fragment and an index that is not an integer ≥ 0 are rejected and left out.
function _parse_fragments(st::_InpState, n::Int, text::AbstractString)
    frags = Vector{Int}[]
    for part in eachsplit(text, ';')
        if isempty(strip(part))
            _reject(st, n, :fragment_index, "`fragments`: an empty fragment")
            continue
        end
        frag = Int[]
        for idx in eachsplit(part, ',')
            s = strip(idx)
            x = _parse_int(s)
            if isnothing(x)
                _reject(st, n, :fragment_index, lazy"fragment index `$(s)` is not an integer")
            elseif x < 0
                _reject(st, n, :fragment_index, lazy"fragment index `$(s)` is not an integer ≥ 0")
            else
                push!(frag, x + 1)
            end
        end
        isempty(frag) || push!(frags, frag)
    end
    return frags
end

# The `fragments` keyword: two or more fragments on its line, or none, which opens a block.
function _fragments_keyword!(st::_InpState, n::Int, tokens::Vector{SubString{String}})
    if length(tokens) == 1
        _open_block!(st, n, :fragments, tokens)
        return nothing
    end
    frags = _parse_fragments(st, n, _join(tokens, ' ', 2))
    if length(frags) >= 2
        _add_fragments!(st, n, frags)
        return nothing
    end
    _reject(st, n, :too_few_values, lazy"`fragments` on its line needs at least 2 fragments, got $(length(frags))")
    _open_block!(st, n, :fragments, tokens)
    return nothing
end

# Opens the block of keyword `c` on line `n`; a geometry block reads its unit.
function _open_block!(st::_InpState, n::Int, c::Symbol, tokens::Vector{SubString{String}})
    if c === :geometry
        st.first_geometry == 0 ? (st.first_geometry = n) :
            _reject(st, n, :duplicate_block, lazy"second geometry block (first at line $(st.first_geometry))")
    elseif c === :product_geometry
        st.first_product == 0 ? (st.first_product = n) :
            _reject(st, n, :duplicate_block, lazy"second product_geometry block (first at line $(st.first_product))")
    elseif c === :oniom
        st.oniom = true
    end
    st.block = c
    st.block_line = n
    st.block_lines = 0
    st.bohr = false
    if (c === :geometry || c === :product_geometry || c === :charges) && length(tokens) >= 2
        u = lowercase(tokens[2])
        if u == "bohr" || u == "au"
            st.bohr = true
        elseif !(u == "angstrom" || u == "ang")
            throw(InputError(st.source, n, lazy"unknown value `$(tokens[2])` for the $(c) unit; accepted: angstrom, ang, bohr, au"))
        end
    end
    return nothing
end

_inside(st::_InpState, t::AbstractString) =
    lazy"`$(t)` inside the $(st.block) block opened at line $(st.block_line); close the block with `end`"

# A line inside the open block that is neither the block's content nor a keyword. Its first token
# may be a misspelled keyword, so the message lists the keywords, as an unknown keyword's does.
function _not_keyword(st::_InpState, t::AbstractString)
    words = _join(ACCEPTED_KEYWORDS, ", ", 1)
    return lazy"`$(t)` inside the $(st.block) block opened at line $(st.block_line) is not a keyword; the accepted keywords are $(words); close the block with `end`"
end

# A line inside the open block.
function _block_line!(st::_InpState, n::Int, tokens::Vector{SubString{String}}, kw::String)
    if kw == "end"
        length(tokens) == 1 || _reject(st, n, :extra_values, lazy"`end` takes no value, got $(length(tokens) - 1)")
        _close_block!(st)
        return nothing
    end
    i = _keyword_index(kw)
    if i != 0
        # Lenient mode reads the line as its keyword, as the oracle matches the keyword before
        # it reads a block line (`parser.rs:56-563`); the block stays open unless the keyword
        # opens a block.
        _reject(st, n, :block_keyword, _inside(st, tokens[1]))
        if INP_KEYWORDS[i][2] in BLOCK_KEYWORDS
            l = st.block_line
            _reject(st, n, :unclosed_block, lazy"the $(st.block) block opened at line $(l) is not closed by `end`")
            _close_block!(st)
        end
        _keyword_line!(st, n, tokens, kw)
        return nothing
    end
    b = st.block
    if b === :geometry || b === :product_geometry
        _geometry_line!(st, n, tokens, kw)
    elseif b === :charges
        _charges_line!(st, n, tokens)
    elseif b === :hubbard
        _hubbard_line!(st, n, tokens)
    elseif b === :fragments
        _add_fragments!(st, n, _parse_fragments(st, n, _join(tokens, ' ', 1)))
    elseif b === :cell
        _cell_line!(st, n, tokens)
    elseif b === :pbc
        _pbc_line!(st, n, tokens)
    else
        _oniom_line!(st, n, tokens, kw)
    end
    return nothing
end

# The position of the coordinate tokens `i:i+2`, in bohr.
function _position(st::_InpState, n::Int, tokens::Vector{SubString{String}}, i::Int)
    b = st.block
    x = _float_required(st, n, b, tokens[i])
    y = _float_required(st, n, b, tokens[i + 1])
    z = _float_required(st, n, b, tokens[i + 2])
    st.bohr && return SVector(x, y, z)
    return SVector(x * ANGSTROM_TO_BOHR, y * ANGSTROM_TO_BOHR, z * ANGSTROM_TO_BOHR)
end

# Whether the oracle reads `s` as a ghost atom: a prefix `Bq` or `bq`, or a suffix `@`
# (`periodic_table.rs:11`). The test is case-sensitive, so `BQx` is no ghost.
_ghost_form(s::AbstractString) = startswith(s, "Bq") || startswith(s, "bq") || endswith(s, '@')

# `Sym x y z`, an atom, or `Bq x y z q`, a point charge, in a geometry or product_geometry block.
function _geometry_line!(st::_InpState, n::Int, tokens::Vector{SubString{String}}, kw::String)
    nt = length(tokens)
    nt < 4 && return _reject(st, n, :dropped_block_line, _not_keyword(st, tokens[1]))
    if kw == "bq" && nt >= 5
        nt > 5 && _reject(st, n, :extra_values, lazy"a Bq line takes 5 tokens (Bq, 3 coordinates and a charge), got $(nt)")
        p = _position(st, n, tokens, 2)
        q = _float_required(st, n, st.block, tokens[5])
        push!(st.d.point_charges, PointCharge{Float64}(q, p))
        return nothing
    end
    nt > 4 && _reject(st, n, :extra_values, lazy"a geometry line takes 4 tokens (a symbol and 3 coordinates), got $(nt)")
    p = _position(st, n, tokens, 2)
    sym = tokens[1]
    atoms = st.block === :geometry ? st.d.atoms : st.d.product_atoms
    # The oracle reads any symbol of ghost form as a ghost of Z = 0 (`periodic_table.rs:5-12`),
    # which lenient mode builds: the symbol as written, mass 0.
    if kw == "bq" || kw == "bq-"
        msg = "a Bq atom needs an element (BqO, O@); a point charge is `Bq x y z q`"
        _ghost_form(sym) || throw(InputError(st.source, n, msg))
        _reject(st, n, :ghost_unchecked, msg)
        push!(atoms, Atom{Float64}(0, String(sym), p, true, 0.0))
        return nothing
    end
    if element_number(sym) == 0
        msg = lazy"unknown element symbol $(repr(sym))"
        _ghost_form(sym) || throw(InputError(st.source, n, msg))
        _reject(st, n, :ghost_unchecked, msg)
        push!(atoms, Atom{Float64}(0, String(sym), p, true, 0.0))
        return nothing
    end
    push!(atoms, Atom(sym, p))
    return nothing
end

# `q x y z` in a charges block.
function _charges_line!(st::_InpState, n::Int, tokens::Vector{SubString{String}})
    nt = length(tokens)
    nt < 4 && return _reject(st, n, :dropped_block_line, _not_keyword(st, tokens[1]))
    nt > 4 && _reject(st, n, :extra_values, lazy"a charges line takes 4 tokens (a charge and 3 coordinates), got $(nt)")
    q = _float_required(st, n, :charges, tokens[1])
    push!(st.d.point_charges, PointCharge{Float64}(q, _position(st, n, tokens, 2)))
    return nothing
end

# `Sym U` in a hubbard block; an element is given once.
function _hubbard_line!(st::_InpState, n::Int, tokens::Vector{SubString{String}})
    nt = length(tokens)
    nt < 2 && return _reject(st, n, :dropped_block_line, _not_keyword(st, tokens[1]))
    nt > 2 && _reject(st, n, :extra_values, lazy"a hubbard line takes 2 tokens (a symbol and U), got $(nt)")
    sym = tokens[1]
    _check_hubbard_symbol(st.source, n, st.lenient, st.warnings, "`hubbard`", sym)
    # A U that does not parse fails in the oracle too (`parser.rs:566`); lenient mode gives its message.
    st.lenient && isnothing(parse_coordinate(tokens[2])) &&
        throw(InputError(st.source, n, lazy"Invalid U value: $(tokens[2])"))
    u = _float_required(st, n, :hubbard, tokens[2])
    h = st.d.hubbard
    for i in eachindex(h, st.hubbard_lines)
        if lowercase(h[i].symbol) == lowercase(sym)
            _reject(st, n, :duplicate_keyword, lazy"`hubbard` gives `$(sym)` twice (lines $(st.hubbard_lines[i]) and $(n))")
            break
        end
    end
    # The oracle's map keeps the last U of a symbol and tells symbols apart by case (`parser.rs:567`).
    for i in eachindex(h, st.hubbard_lines)
        if h[i].symbol == sym
            h[i] = HubbardU(String(sym), u)
            st.hubbard_lines[i] = n
            return nothing
        end
    end
    push!(h, HubbardU(String(sym), u))
    push!(st.hubbard_lines, n)
    return nothing
end

# One of the three lattice vectors (Å) of a cell block.
function _cell_line!(st::_InpState, n::Int, tokens::Vector{SubString{String}})
    st.block_lines >= 3 && return _reject(st, n, :extra_values, "the cell block takes 3 lines")
    nt = length(tokens)
    nt == 3 || return _reject(st, n, :dropped_block_line, lazy"a cell line takes 3 numbers, got $(nt)")
    x = _float_optional(st, n, :cell, tokens[1], :bad_list_entry)
    y = _float_optional(st, n, :cell, tokens[2], :bad_list_entry)
    z = _float_optional(st, n, :cell, tokens[3], :bad_list_entry)
    (isnothing(x) || isnothing(y) || isnothing(z)) && return nothing
    push!(st.cell, SVector(x::Float64, y::Float64, z::Float64))
    st.block_lines += 1
    return nothing
end

_pbc_flag(t::AbstractString) = t == "true" || t == "1" || t == "T"

# The three periodicity flags of a pbc block.
function _pbc_line!(st::_InpState, n::Int, tokens::Vector{SubString{String}})
    st.block_lines >= 1 && return _reject(st, n, :extra_values, "the pbc block takes 1 line")
    nt = length(tokens)
    nt < 3 && return _reject(st, n, :too_few_values, lazy"a pbc line takes 3 values, got $(nt)")
    # Lenient mode reads the first three tokens, true for `true`, `1` or `T` and false for any
    # other token, as the oracle does (`parser.rs:645-653`); one warning covers the line.
    nt > 3 && _reject(st, n, :extra_values, lazy"a pbc line takes 3 values, got $(nt)")
    for j in 1:3
        t = tokens[j]
        if !(t == "true" || t == "false")
            _reject(st, n, :bool_fallback, lazy"`pbc`: expected true or false, got $(repr(t))")
            break
        end
    end
    st.d.pbc = (_pbc_flag(tokens[1]), _pbc_flag(tokens[2]), _pbc_flag(tokens[3]))
    st.block_lines += 1
    return nothing
end

# `high`, `low`, `model_atoms`/`model_indices`, `model_charge` or `model_multiplicity` in an oniom
# block.
function _oniom_line!(st::_InpState, n::Int, tokens::Vector{SubString{String}}, kw::String)
    key = kw == "high" ? :oniom_high : kw == "low" ? :oniom_low :
        kw == "model_atoms" || kw == "model_indices" ? :oniom_model :
        kw == "model_charge" ? :oniom_model_charge : kw == "model_multiplicity" ? :oniom_model_multiplicity : :none
    key === :none && return _reject(st, n, :dropped_block_line, _not_keyword(st, tokens[1]))
    first = get(st.seen, key, 0)
    first == 0 || _reject(st, n, :duplicate_keyword, lazy"keyword `$(kw)` given twice (lines $(first) and $(n))")
    st.seen[key] = n
    if key === :oniom_high || key === :oniom_low
        m = _method_value(st, n, kw, tokens)
        isnothing(m) && return nothing
        key === :oniom_high ? (st.high = m) : (st.low = m)
    elseif key === :oniom_model
        _count_ok(st, n, kw, key, length(tokens) - 1, 1, -1) && _model_atoms!(st, n, kw, tokens)
    else
        ok = _count_ok(st, n, kw, key, length(tokens) - 1, 1, 1)
        if key === :oniom_model_charge
            st.model_charge = ok ? _int32_optional(st, n, kw, tokens[2], :number_fallback) : nothing
        else
            st.model_multiplicity = ok ? _uint_optional(st, n, kw, tokens[2], :number_fallback) : nothing
        end
    end
    return nothing
end

# Closes the open block: a cell block needs its 3 lines and a pbc block its line.
function _close_block!(st::_InpState)
    b = st.block
    l = st.block_line
    k = st.block_lines
    if b === :cell
        if k < 3
            _reject(st, l, :too_few_values, lazy"the cell block opened at line $(l) has $(k) lines; it takes 3")
        else
            r = st.cell
            st.d.lattice = SMatrix{3, 3, Float64, 9}(r[1][1], r[2][1], r[3][1], r[1][2], r[2][2], r[3][2], r[1][3], r[2][3], r[3][3])
        end
    elseif b === :pbc && k == 0
        _reject(st, l, :too_few_values, lazy"the pbc block opened at line $(l) has no line; it takes 1")
    end
    st.block = :none
    return nothing
end

# Whether a warning of `ws` on solvation line `l` already accounts for a lenient result that differs
# from the order-independent one: a solvation-order case, or `:unknown_value`, the word of a
# `solvation` or `solvent` line that the oracle ignores and the order-independent reading counts.
function _order_warned_at(ws::Vector{InputWarning}, l::Int)
    for w in ws
        w.line == l || continue
        c = w.case
        (c === :order_dependent || c === :option_without_model || c === :unknown_value) && return true
    end
    return false
end

# The solvation of the input. Strict mode reads it independently of the order of the lines; lenient
# mode takes the oracle's reading in file order, `seq`, and records an `:order_dependent` warning
# when the two differ and no warning on a solvation line accounts for it (`_order_warned_at`).
function _resolve_solvation!(st::_InpState)
    _solvation_by_keyword!(st)
    st.lenient || return nothing
    s = st.seq
    if s !== st.d.solvation
        l = 0
        warned = false
        for key in (:solvation, :solvent, :epsilon, :pcm_radii, :pcm_grid)
            lk = get(st.seen, key, 0)
            l = max(l, lk)
            lk == 0 || (warned |= _order_warned_at(st.warnings, lk))
        end
        warned || push!(
            st.warnings,
            InputWarning(st.source, l, :order_dependent, "the solvation lines give another result when read in file order, as the oracle reads them"),
        )
    end
    st.d.solvation = s
    return nothing
end

# The solvation model and its options, independent of the order of their lines. The model is the
# `solvation` word if given; otherwise `solvent` gives SMD and `epsilon` alone gives PCM.
function _solvation_by_keyword!(st::_InpState)
    seen = st.seen
    l_model = get(seen, :solvation, 0)
    l_solvent = get(seen, :solvent, 0)
    l_eps = get(seen, :epsilon, 0)
    l_radii = get(seen, :pcm_radii, 0)
    l_grid = get(seen, :pcm_grid, 0)
    base = SolvationSpec()
    model = l_model != 0 ? st.solvation : l_solvent != 0 ? SMD : l_eps != 0 ? PCM : NoSolvation
    if model == NoSolvation
        if l_model != 0
            for (l, name) in ((l_solvent, "solvent"), (l_eps, "epsilon"), (l_radii, "pcm_radii"), (l_grid, "pcm_grid"))
                l == 0 || _reject(st, l, :option_without_model, lazy"`$(name)` is given with `solvation none`")
            end
        else
            l_radii == 0 || _reject(st, l_radii, :option_without_model, "`pcm_radii` needs `solvation pcm`")
            l_grid == 0 || _reject(st, l_grid, :option_without_model, "`pcm_grid` needs `solvation pcm`")
        end
        return nothing
    end
    eps = l_eps != 0 ? st.epsilon : base.epsilon
    grid = l_grid != 0 ? st.grid : base.grid_order
    if model == PCM
        l_solvent == 0 || _reject(st, l_solvent, :order_dependent, "`solvent` needs `solvation smd`")
        radii = l_radii != 0 ? st.radii : base.radii_scaling
        st.d.solvation = SolvationSpec(; model = PCM, epsilon = eps, radii_scaling = radii, grid_order = grid)
    else
        l_radii == 0 || _reject(st, l_radii, :option_without_model, "`pcm_radii` needs `solvation pcm`")
        solvent = l_solvent != 0 ? st.solvent : base.solvent
        st.d.solvation = SolvationSpec(; model = SMD, solvent, epsilon = eps, custom_epsilon = l_eps != 0, grid_order = grid)
    end
    return nothing
end

# `f12_gamma` and `f12_model` need `f12 true` or an F12 method.
function _check_f12(st::_InpState)
    k = st.d.method.kind
    (st.d.use_f12 || k == MP2_F12 || k == CCSD_F12 || k == CCSD_T_F12) && return nothing
    for (key, name) in ((:f12_gamma, "f12_gamma"), (:f12_model, "f12_model"))
        l = get(st.seen, key, 0)
        l == 0 || _reject(st, l, :f12_option_alone, lazy"`$(name)` needs `f12 true` or an F12 method")
    end
    return nothing
end
