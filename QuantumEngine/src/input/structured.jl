# The schema walker of structured input: one pass over a `Node` tree, in the oracle's key order
# (`parse_yaml_string`). Each place where the oracle ignores a key or a value silently calls
# `_reject`, which throws in strict mode and records an `InputWarning` of the named `:yaml_*` case
# in lenient mode; each place where the oracle itself fails throws an `InputError` in both modes.
# Every message names the key path (`scf.max_iter`, `geometry[2].position[3]`, with 1-based list
# positions) and line 0.

# The keys each map accepts. A key of `scf` with a top-level fallback is accepted at both levels.
const TOP_KEYS = [
    "title", "charge", "multiplicity", "basis", "method", "dispersion", "zora", "mode", "task", "hubbard",
    "vdw_kernel", "scf", "max_iterations", "scf_convergence", "gpu", "compute_mode", "pw_cutoff", "lattice",
    "pbc", "k_grid", "kpoints", "k_path", "use_dmft", "dmft", "solvation", "oniom", "tddft", "exports", "f12",
    "f12_gamma", "gamma", "f12_model", "geometry", "atoms", "unit", "irc_step_size", "irc_step",
    "irc_max_points", "irc_points", "irc_direction", "constraints",
]
const TASK_KEYS = ["mode"]
const SCF_KEYS = [
    "max_iter", "conv_energy", "use_ri_j", "use_ri_k", "aux_basis", "gpu", "compute_mode", "diis_method",
    "verbosity", "checkpoint", "load_checkpoint", "damping", "smearing", "grid", "auto_stabilize",
    "robust_convergence", "level_shift", "field", "pw_cutoff", "lattice", "pbc", "admm", "admm_basis",
    "population", "kpoints", "kpath", "use_dmft", "dmft_options", "dmft",
]
const DMFT_KEYS = ["temperature", "n_matsubara", "n_bath", "u_map", "j_map"]
const SOLVATION_KEYS = ["model", "pcm", "smd", "solvent", "grid_order", "probe_radius", "epsilon", "radii_scaling", "surface"]
const SMD_KEYS = ["model", "pcm", "smd", "solvent", "grid_order", "probe_radius"]
const PCM_KEYS = ["model", "pcm", "smd", "epsilon", "radii_scaling", "grid_order", "surface"]
const ONIOM_KEYS = ["high", "low", "model_atoms", "model_charge", "model_multiplicity"]
const TDDFT_KEYS = ["num_states", "method", "soc", "triplets"]
const EXPORT_KEYS = ["molden", "cube", "wfx"]
const F12_KEYS = ["gamma", "f12_gamma", "model", "approximation"]
const ATOM_KEYS = ["symbol", "position", "x", "y", "z"]
const CONSTRAINT_KEYS = ["freeze_atom", "freeze_bond", "freeze_angle", "target"]

# The method words of structured input: every `.inp` method word but the Libxc forms, which take
# numbers after the word.
const STRUCTURED_METHODS = ACCEPTED_METHODS[.!startswith.(ACCEPTED_METHODS, "libxc_")]
const SOLVATION_MODEL_WORDS = ["pcm", "smd"]

# The source, mode and warnings of one walk.
struct _Walk
    source::String
    lenient::Bool
    warnings::Vector{InputWarning}
end

_reject(w::_Walk, case::Symbol, msg::AbstractString) = reject_or_warn(w.source, 0, w.lenient, w.warnings, case, msg)

# A value and its key path; `present` is false, and `node` a null node, for a key the map lacks.
struct _At
    path::String
    node::Node
    present::Bool
end

const _ABSENT = Node(nothing)

_child(path::String, key::String) = isempty(path) ? key : path * "." * key
_list_path(path::String, k::Int) = path * "[" * string(k) * "]"

# The value of `key` in the map `m`; absent when `m` is not a map or lacks the key.
function _at(m::_At, key::String)
    n = m.node
    if n.kind == MapNode
        for j in eachindex(n.keys)
            n.keys[j] == key && return _At(_child(m.path, key), n.items[j], true)
        end
    end
    return _At(_child(m.path, key), _ABSENT, false)
end

# The `k`-th (1-based) item of the list `a`.
_item(a::_At, k::Int) = _At(_list_path(a.path, k), a.node.items[firstindex(a.node.items) + k - 1], true)

function _kind_word(k::NodeKind)
    k == NullNode && return "null"
    k == BoolNode && return "a Boolean"
    k == IntNode && return "an integer"
    k == FloatNode && return "a number"
    k == StringNode && return "a string"
    k == ArrayNode && return "a list"
    return "a map"
end

# Whether the present value `a` is of `kind`, rejecting it otherwise with `case`. An integer for a
# number is converted in strict mode; in lenient mode it is `:yaml_int_for_float`, ignored as the
# oracle ignores it.
function _ok(w::_Walk, a::_At, kind::NodeKind, case::Symbol = :yaml_wrong_type)
    a.present || return false
    k = a.node.kind
    k == kind && return true
    if kind == FloatNode && k == IntNode
        w.lenient || return true
        _reject(w, :yaml_int_for_float, lazy"$(a.path): expected a number, got an integer")
        return false
    end
    _reject(w, case, lazy"$(a.path): expected $(_kind_word(kind)), got $(_kind_word(k))")
    return false
end

_sbool(w::_Walk, a::_At) = _ok(w, a, BoolNode) ? a.node.b : nothing
_sint(w::_Walk, a::_At) = _ok(w, a, IntNode) ? a.node.i : nothing

# An integer the oracle stores unsigned. A negative value, which the oracle wraps to a count near
# 2^64, is an error in both modes.
function _suint(w::_Walk, a::_At)
    x = _sint(w, a)
    isnothing(x) && return nothing
    x < 0 && throw(InputError(w.source, 0, lazy"$(a.path): expected a non-negative integer, got $(x)"))
    return x
end

function _sfloat(w::_Walk, a::_At, case::Symbol = :yaml_wrong_type)
    _ok(w, a, FloatNode, case) || return nothing
    n = a.node
    x = n.kind == IntNode ? Float64(n.i) : n.f
    isfinite(x) || _reject(w, :nonfinite, lazy"$(a.path): expected a finite number, got $(x)")
    return x
end

# A number of a list, or `fallback` for an entry the oracle reads with `unwrap_or(fallback)`.
function _number(w::_Walk, a::_At, fallback::Float64)
    x = _sfloat(w, a, :yaml_bad_number)
    return isnothing(x) ? fallback : x
end

_sstring(w::_Walk, a::_At) = _ok(w, a, StringNode)
_smap(w::_Walk, a::_At) = _ok(w, a, MapNode)
_slist(w::_Walk, a::_At) = _ok(w, a, ArrayNode)

# Rejects the keys of the map `m` that are not `accepted`.
function _check_keys(w::_Walk, m::_At, accepted::Vector{String})
    for k in m.node.keys
        k in accepted && continue
        words = _join(accepted, ", ", 1)
        _reject(w, :yaml_unknown_key, lazy"$(_child(m.path, k)): unknown key; accepted keys: $(words)")
    end
    return nothing
end

_both(w::_Walk, a::_At, b::_At) =
    _reject(w, :yaml_both_keys, lazy"`$(a.path)` and `$(b.path)` are both given; give one")

# Of a key and its fallback, the one the oracle reads when it tests for presence: `a` when given.
function _first(w::_Walk, a::_At, b::_At)
    a.present && b.present && _both(w, a, b)
    return a.present ? a : b
end

# Of a key and its fallback, the one the oracle reads with `.as_<kind>().or(...)`: `a` when given
# with the value type `kind`, otherwise `b`.
function _alt(w::_Walk, a::_At, b::_At, kind::NodeKind)
    (a.present && b.present) || return a.present ? a : b
    _both(w, a, b)
    return _ok(w, a, kind) ? a : b
end

# Rejects the unknown word of `a` with `case`.
_unknown_word(w::_Walk, a::_At, case::Symbol, accepted::Vector{String}) =
    _reject(w, case, _unknown_msg(a.path, a.node.s, accepted))

# A list of `Sym: value` pairs (`hubbard`, `u_map`, `j_map`), in the order given.
function _hubbard_map(w::_Walk, a::_At)
    out = HubbardU[]
    _smap(w, a) || return out
    n = a.node
    for j in eachindex(n.keys)
        u = _sfloat(w, _At(_child(a.path, n.keys[j]), n.items[j], true))
        isnothing(u) || push!(out, HubbardU(n.keys[j], u))
    end
    return out
end

# The method of the string `a`, or `nothing` when it names none and `required` is false (the
# top-level `method`, which the oracle ignores then). A `casscf` word that `method_word` rejects is
# `:yaml_unknown_method`, and lenient reading takes the oracle's active space. For an ONIOM method
# (`required`), an unknown word and a malformed active space are errors in both modes.
function _structured_method(w::_Walk, a::_At, required::Bool)
    s = a.node.s
    m = method_word([s])
    isnothing(m) || return m
    if _is_casscf(s)
        _reject(w, :yaml_unknown_method, lazy"$(a.path): casscf needs (electrons,orbitals), got `$(s)`")
        status, e, o = _casscf_counts(s)
        if status != 0
            required || return nothing
            throw(InputError(w.source, 0, status == 1 ? "Invalid CASSCF electron count" : "Invalid CASSCF orbital count"))
        end
        return MethodSpec(CASSCF, AutoReference, (e, o), 0.0)
    end
    msg = _unknown_msg(a.path, s, STRUCTURED_METHODS)
    required && throw(InputError(w.source, 0, msg))
    _reject(w, :yaml_unknown_method, msg)
    return nothing
end

"""
    parse_structured(root::Node; source::AbstractString, lenient::Bool = false) -> ParsedInput

The structured input whose tree is `root` (JSON, YAML or any `AbstractDict` once converted to a
`Node`), read strictly by default; `source` names it in error messages and warnings, whose line is
0. Keys are case-sensitive and read as the oracle reads them; the schema and its deviations from
the oracle are documented on the Input page of the manual. The checks shared with `.inp` (atoms
given, a restricted reference only for a singlet, a whole number of electrons per spin) run last.

In strict mode every input the oracle accepts silently but that is not valid (an unknown key, a
value of the wrong type, an unknown word) is an `InputError` naming the key path. In lenient mode
such an input is read as the oracle reads it and recorded as an `InputWarning` of a `:yaml_*`
case. Atom positions are converted to bohr.
"""
function parse_structured(root::Node; source::AbstractString, lenient::Bool = false)
    w = _Walk(String(source), lenient, InputWarning[])
    root.kind == NullNode && throw(InputError(w.source, 0, "the document is empty"))
    root.kind == MapNode ||
        throw(InputError(w.source, 0, lazy"the top-level value is $(_kind_word(root.kind)), not a map"))
    doc = _At("", root, true)
    _check_keys(w, doc, TOP_KEYS)
    d = _draft(InputFile())
    _top!(w, d, doc)
    _scf!(w, d, doc)
    _solvation!(w, d, doc)
    o = _at(doc, "oniom")
    _tddft!(w, d, doc)
    _exports!(w, d, doc)
    _f12!(w, d, doc)
    _geometry!(w, d, doc)
    _irc!(w, d, doc)
    _constraints!(w, d, doc)
    inp = o.present ? _input(d, _oniom(w, o)) : _input(d, nothing)
    return ParsedInput(_finalize_input(inp; source = w.source, lenient, warnings = w.warnings), w.warnings)
end

# The run mode and the top-level options.
function _top!(w::_Walk, d::_InputDraft, doc::_At)
    task = _at(doc, "task")
    task.present && _smap(w, task) && _check_keys(w, task, TASK_KEYS)
    mode = _first(w, _at(task, "mode"), _at(doc, "mode"))
    if _sstring(w, mode)
        x = run_mode_word(mode.node.s)
        isnothing(x) ? _unknown_word(w, mode, :yaml_unknown_mode, accepted_run_modes()) : (d.run_mode = x)
    end
    a = _at(doc, "title")
    _sstring(w, a) && (d.title = a.node.s)
    x = _sint(w, _at(doc, "charge"))
    isnothing(x) || (d.charge = x)
    x = _suint(w, _at(doc, "multiplicity"))
    isnothing(x) || (d.multiplicity = x)
    a = _at(doc, "basis")
    _sstring(w, a) && (d.basis = lowercase(a.node.s))
    b = _sbool(w, _at(doc, "dispersion"))
    isnothing(b) || (d.dispersion = b)
    b = _sbool(w, _at(doc, "zora"))
    isnothing(b) || (d.zora = b)
    a = _at(doc, "method")
    if _sstring(w, a)
        m = _structured_method(w, a, false)
        isnothing(m) || (d.method = m)
    end
    a = _at(doc, "hubbard")
    a.present && (d.hubbard = _hubbard_map(w, a))
    a = _at(doc, "vdw_kernel")
    if _sstring(w, a)
        k = vdw_kernel_word(a.node.s)
        # The oracle stores an unknown word and computes no dispersion for it.
        isnothing(k) && _unknown_word(w, a, :yaml_unknown_value, accepted_vdw_kernels())
        d.vdw_kernel = k
    end
    return nothing
end

# The `scf` map and the top-level fallbacks of its keys.
function _scf!(w::_Walk, d::_InputDraft, doc::_At)
    scf = _at(doc, "scf")
    scf.present && _smap(w, scf) && _check_keys(w, scf, SCF_KEYS)
    x = _suint(w, _alt(w, _at(scf, "max_iter"), _at(doc, "max_iterations"), IntNode))
    isnothing(x) || (d.max_iterations = x)
    f = _sfloat(w, _alt(w, _at(scf, "conv_energy"), _at(doc, "scf_convergence"), FloatNode))
    isnothing(f) || (d.scf_convergence = f)
    b = _sbool(w, _at(scf, "use_ri_j"))
    isnothing(b) || (d.use_ri_j = b)
    b = _sbool(w, _at(scf, "use_ri_k"))
    isnothing(b) || (d.use_ri_k = b)
    a = _at(scf, "aux_basis")
    _sstring(w, a) && (d.aux_basis = a.node.s)
    # `gpu: true` selects the GPU and `gpu: false` changes nothing; `compute_mode` is read after
    # it, so the two set one field and are not given together.
    g = _alt(w, _at(scf, "gpu"), _at(doc, "gpu"), BoolNode)
    b = _sbool(w, g)
    b === true && (d.compute_mode = GPU)
    c = _alt(w, _at(scf, "compute_mode"), _at(doc, "compute_mode"), StringNode)
    g.present && c.present && _both(w, g, c)
    if _sstring(w, c)
        m = compute_mode_word(c.node.s)
        isnothing(m) ? _unknown_word(w, c, :yaml_unknown_value, accepted_compute_modes()) : (d.compute_mode = m)
    end
    a = _at(scf, "diis_method")
    if _sstring(w, a)
        m = diis_word(a.node.s)
        # The oracle stores the word, and no code reads it: the extrapolation stays CDIIS.
        isnothing(m) && _unknown_word(w, a, :yaml_unknown_value, accepted_diis_methods())
        d.diis_method = isnothing(m) ? CDIIS : m
    end
    x = _suint(w, _at(scf, "verbosity"))
    isnothing(x) || (d.verbosity = x)
    a = _at(scf, "checkpoint")
    _sstring(w, a) && (d.checkpoint = a.node.s)
    b = _sbool(w, _at(scf, "load_checkpoint"))
    isnothing(b) || (d.load_checkpoint = b)
    f = _sfloat(w, _at(scf, "damping"))
    isnothing(f) || (d.damping = f)
    f = _sfloat(w, _at(scf, "smearing"))
    isnothing(f) || (d.smearing = f)
    x = _suint(w, _at(scf, "grid"))
    isnothing(x) || (d.grid_level = x)
    b = _sbool(w, _at(scf, "auto_stabilize"))
    isnothing(b) || (d.auto_stabilize = b)
    b = _sbool(w, _at(scf, "robust_convergence"))
    isnothing(b) || (d.robust_convergence = b)
    f = _sfloat(w, _at(scf, "level_shift"))
    isnothing(f) || (d.level_shift = f)
    _field!(w, d, _at(scf, "field"))
    f = _sfloat(w, _alt(w, _at(scf, "pw_cutoff"), _at(doc, "pw_cutoff"), FloatNode))
    isnothing(f) || (d.pw_cutoff = f)
    _lattice!(w, d, _first(w, _at(scf, "lattice"), _at(doc, "lattice")))
    _pbc!(w, d, _first(w, _at(scf, "pbc"), _at(doc, "pbc")))
    b = _sbool(w, _at(scf, "admm"))
    isnothing(b) || (d.admm = b)
    a = _at(scf, "admm_basis")
    _sstring(w, a) && (d.admm_basis = a.node.s)
    _population!(w, d, _at(scf, "population"))
    _kpoints!(w, d, _first(w, _at(scf, "kpoints"), _first(w, _at(doc, "k_grid"), _at(doc, "kpoints"))))
    _kpath!(w, d, _first(w, _at(scf, "kpath"), _at(doc, "k_path")))
    b = _sbool(w, _alt(w, _at(scf, "use_dmft"), _at(doc, "use_dmft"), BoolNode))
    isnothing(b) || (d.use_dmft = b)
    _dmft!(w, d, _first(w, _at(scf, "dmft_options"), _first(w, _at(doc, "dmft"), _at(scf, "dmft"))))
    return nothing
end

# Whether the list `a` has exactly `n` items; rejects it with `case` otherwise.
function _length_ok(w::_Walk, a::_At, n::Int, what::String, case::Symbol)
    k = length(a.node.items)
    k == n && return true
    _reject(w, case, lazy"$(a.path): expected $(n) $(what), got $(k)")
    return false
end

# `field`: the oracle reads the first 3 entries of a list of 3 or more, each `unwrap_or(0.0)`.
function _field!(w::_Walk, d::_InputDraft, a::_At)
    _slist(w, a) || return nothing
    _length_ok(w, a, 3, "numbers", :yaml_list_length)
    length(a.node.items) >= 3 || return nothing
    d.electric_field = SVector(_number(w, _item(a, 1), 0.0), _number(w, _item(a, 2), 0.0), _number(w, _item(a, 3), 0.0))
    return nothing
end

# `lattice`: 3 rows of 3 numbers in Å. The oracle reads a list of another length as no lattice,
# a row that is not 3 entries as zeros and an entry that is not a number as 0.0.
function _lattice!(w::_Walk, d::_InputDraft, a::_At)
    _slist(w, a) || return nothing
    _length_ok(w, a, 3, "rows", :yaml_list_length) || return nothing
    r1, r2, r3 = _lattice_row(w, _item(a, 1)), _lattice_row(w, _item(a, 2)), _lattice_row(w, _item(a, 3))
    d.lattice = SMatrix{3, 3, Float64, 9}(r1[1], r2[1], r3[1], r1[2], r2[2], r3[2], r1[3], r2[3], r3[3])
    return nothing
end

function _lattice_row(w::_Walk, r::_At)
    if r.node.kind != ArrayNode || length(r.node.items) != 3
        _reject(w, :yaml_lattice_row, lazy"$(r.path): expected 3 numbers")
        return (0.0, 0.0, 0.0)
    end
    return (_number(w, _item(r, 1), 0.0), _number(w, _item(r, 2), 0.0), _number(w, _item(r, 3), 0.0))
end

# `pbc`: 3 Booleans; the oracle ignores a list of another length and reads an entry that is not a
# Boolean as true.
function _pbc!(w::_Walk, d::_InputDraft, a::_At)
    _slist(w, a) || return nothing
    _length_ok(w, a, 3, "Booleans", :yaml_list_length) || return nothing
    p1, p2, p3 = _item(a, 1), _item(a, 2), _item(a, 3)
    d.pbc = (_ok(w, p1, BoolNode) ? p1.node.b : true, _ok(w, p2, BoolNode) ? p2.node.b : true, _ok(w, p3, BoolNode) ? p3.node.b : true)
    return nothing
end

# `population`: a list of analysis words. The oracle stores the words, and no code reads them, so
# a list with an entry that is not a known word keeps the default analyses.
function _population!(w::_Walk, d::_InputDraft, a::_At)
    _slist(w, a) || return nothing
    pop = PopulationKind[]
    bad = false
    for k in 1:length(a.node.items)
        e = _item(a, k)
        if !_sstring(w, e)
            bad = true
            continue
        end
        x = population_word(e.node.s)
        if isnothing(x)
            _unknown_word(w, e, :yaml_unknown_value, accepted_populations())
            bad = true
        else
            push!(pop, x)
        end
    end
    d.population = bad ? PopulationKind[Mulliken, Lowdin] : pop
    return nothing
end

# A count of a list, or `fallback` for an entry that is not an integer.
function _count(w::_Walk, a::_At, fallback::Int)
    _ok(w, a, IntNode, :yaml_bad_number) || return fallback
    x = a.node.i
    x < 0 && throw(InputError(w.source, 0, lazy"$(a.path): expected a non-negative integer, got $(x)"))
    return x
end

# `kpoints`: 3 counts; the oracle ignores a list of another length and reads an entry that is not
# an integer as 1.
function _kpoints!(w::_Walk, d::_InputDraft, a::_At)
    _slist(w, a) || return nothing
    _length_ok(w, a, 3, "integers", :yaml_list_length) || return nothing
    d.kpoints = (_count(w, _item(a, 1), 1), _count(w, _item(a, 2), 1), _count(w, _item(a, 3), 1))
    return nothing
end

# `kpath`: the point labels; the oracle drops an entry that is not a string.
function _kpath!(w::_Walk, d::_InputDraft, a::_At)
    _slist(w, a) || return nothing
    path = String[]
    for k in 1:length(a.node.items)
        e = _item(a, k)
        _sstring(w, e) && push!(path, e.node.s)
    end
    d.kpath = path
    return nothing
end

function _dmft!(w::_Walk, d::_InputDraft, a::_At)
    (a.present && _smap(w, a)) || return nothing
    _check_keys(w, a, DMFT_KEYS)
    f = _sfloat(w, _at(a, "temperature"))
    isnothing(f) || (d.dmft_temperature = f)
    x = _suint(w, _at(a, "n_matsubara"))
    isnothing(x) || (d.dmft_n_freq = x)
    x = _suint(w, _at(a, "n_bath"))
    isnothing(x) || (d.dmft_n_bath = x)
    u = _at(a, "u_map")
    u.present && append!(d.dmft_u, _hubbard_map(w, u))
    j = _at(a, "j_map")
    j.present && append!(d.dmft_j, _hubbard_map(w, j))
    return nothing
end

# `solvation`: the model is `model` (`pcm` or `smd`), else `pcm: true`, else `smd: true`, and the
# options read are those of that model.
function _solvation!(w::_Walk, d::_InputDraft, doc::_At)
    s = _at(doc, "solvation")
    (s.present && _smap(w, s)) || return nothing
    model, pcm, smd = _at(s, "model"), _at(s, "pcm"), _at(s, "smd")
    if model.present + pcm.present + smd.present > 1
        _reject(w, :yaml_both_keys, lazy"$(s.path): give one of `model`, `pcm` and `smd`")
    end
    kind = NoSolvation
    if model.present && _ok(w, model, StringNode)
        t = lowercase(model.node.s)
        if t == "pcm"
            kind = PCM
        elseif t == "smd"
            kind = SMD
        else
            _unknown_word(w, model, :yaml_unknown_value, SOLVATION_MODEL_WORDS)
        end
    elseif pcm.present && _ok(w, pcm, BoolNode)
        pcm.node.b && (kind = PCM)
    elseif smd.present && _ok(w, smd, BoolNode)
        smd.node.b && (kind = SMD)
    elseif !(model.present || pcm.present || smd.present)
        _reject(w, :yaml_solvation_without_model, lazy"$(s.path): give model, pcm or smd")
    end
    _check_keys(w, s, kind == SMD ? SMD_KEYS : kind == PCM ? PCM_KEYS : SOLVATION_KEYS)
    kind == NoSolvation && return nothing
    grid = _suint(w, _at(s, "grid_order"))
    grid_order = isnothing(grid) ? 110 : grid
    if kind == SMD
        a = _at(s, "solvent")
        solvent = :water
        if _sstring(w, a)
            x = solvent_word(a.node.s)
            isnothing(x) ? _unknown_word(w, a, :yaml_unknown_solvent, accepted_solvents()) : (solvent = x)
        end
        r = _sfloat(w, _at(s, "probe_radius"))
        probe = isnothing(r) ? 0.4 * ANGSTROM_TO_BOHR : r * ANGSTROM_TO_BOHR
        d.solvation = SolvationSpec(; model = SMD, solvent, grid_order, probe_radius = probe)
        return nothing
    end
    e = _sfloat(w, _at(s, "epsilon"))
    r = _sfloat(w, _at(s, "radii_scaling"))
    a = _at(s, "surface")
    surface = VdwSurface
    if _sstring(w, a)
        x = surface_word(a.node.s)
        isnothing(x) ? _unknown_word(w, a, :yaml_unknown_surface, accepted_surfaces()) : (surface = x)
    end
    d.solvation = SolvationSpec(;
        model = PCM, epsilon = isnothing(e) ? 78.39 : e, radii_scaling = isnothing(r) ? 1.2 : r, grid_order, surface,
    )
    return nothing
end

# `oniom`: given at all, it asks for an ONIOM calculation.
function _oniom(w::_Walk, o::_At)
    _smap(w, o) && _check_keys(w, o, ONIOM_KEYS)
    a = _at(o, "high")
    high = _sstring(w, a) ? _structured_method(w, a, true) : DEFAULT_METHOD
    a = _at(o, "low")
    low = _sstring(w, a) ? _structured_method(w, a, true) : DEFAULT_METHOD
    atoms = Int[]
    a = _at(o, "model_atoms")
    if _slist(w, a)
        for k in 1:length(a.node.items)
            x = _suint(w, _item(a, k))
            isnothing(x) || push!(atoms, x + 1)
        end
    end
    return ONIOMSpec(
        something(high, DEFAULT_METHOD), something(low, DEFAULT_METHOD), atoms,
        _sint(w, _at(o, "model_charge")), _suint(w, _at(o, "model_multiplicity")),
    )
end

function _tddft!(w::_Walk, d::_InputDraft, doc::_At)
    t = _at(doc, "tddft")
    (t.present && _smap(w, t)) || return nothing
    _check_keys(w, t, TDDFT_KEYS)
    x = _suint(w, _at(t, "num_states"))
    isnothing(x) || (d.tddft_states = x)
    a = _at(t, "method")
    if _sstring(w, a)
        m = tddft_method_word(a.node.s)
        # The oracle stores the word, and its TDDFT driver runs TDA for an unknown one.
        isnothing(m) && _unknown_word(w, a, :yaml_unknown_value, accepted_tddft_methods())
        d.tddft_method = isnothing(m) ? TDA : m
    end
    b = _sbool(w, _at(t, "soc"))
    isnothing(b) || (d.compute_soc = b)
    b = _sbool(w, _at(t, "triplets"))
    isnothing(b) || (d.tddft_triplets = b)
    return nothing
end

function _exports!(w::_Walk, d::_InputDraft, doc::_At)
    e = _at(doc, "exports")
    (e.present && _smap(w, e)) || return nothing
    _check_keys(w, e, EXPORT_KEYS)
    b = _sbool(w, _at(e, "molden"))
    isnothing(b) || (d.export_molden = b)
    b = _sbool(w, _at(e, "cube"))
    isnothing(b) || (d.export_cube = b)
    b = _sbool(w, _at(e, "wfx"))
    isnothing(b) || (d.export_wfx = b)
    return nothing
end

# `f12`, a Boolean or a map of options (any value but a Boolean turns F12 on, as in the oracle),
# then the top-level `f12_gamma`/`gamma`, which turns F12 on in the oracle and needs it in strict
# mode, and `f12_model`.
function _f12!(w::_Walk, d::_InputDraft, doc::_At)
    f = _at(doc, "f12")
    if f.present
        if f.node.kind == BoolNode
            d.use_f12 = f.node.b
        else
            _smap(w, f) && _check_keys(w, f, F12_KEYS)
            d.use_f12 = true
            g = _sfloat(w, _alt(w, _at(f, "gamma"), _at(f, "f12_gamma"), FloatNode))
            isnothing(g) || (d.f12_gamma = g)
            m = _alt(w, _at(f, "model"), _at(f, "approximation"), StringNode)
            _sstring(w, m) && (d.f12_model = m.node.s)
        end
    end
    a = _alt(w, _at(doc, "f12_gamma"), _at(doc, "gamma"), FloatNode)
    g = _sfloat(w, a)
    if !isnothing(g)
        d.use_f12 || _reject(w, :yaml_f12_gamma_enables, lazy"$(a.path) needs f12")
        d.f12_gamma = g
        d.use_f12 = true
    end
    a = _at(doc, "f12_model")
    _sstring(w, a) && (d.f12_model = a.node.s)
    return nothing
end

# Whether `unit` names bohr: `bohr` or `au`, in any ASCII case; `angstrom` or `ang`, the default,
# name Å.
function _bohr(w::_Walk, doc::_At)
    u = _at(doc, "unit")
    _sstring(w, u) || return false
    s = u.node.s
    (_word_eq(s, "bohr") || _word_eq(s, "au")) && return true
    (_word_eq(s, "angstrom") || _word_eq(s, "ang")) && return false
    _reject(w, :yaml_unknown_unit, lazy"$(u.path): unknown value `$(s)`; accepted: angstrom, ang, bohr, au")
    return false
end

# `geometry`, else `atoms`: a list of `{symbol, position: [x, y, z]}` or `{symbol, x, y, z}`.
function _geometry!(w::_Walk, d::_InputDraft, doc::_At)
    g = _first(w, _at(doc, "geometry"), _at(doc, "atoms"))
    bohr = _bohr(w, doc)
    (g.present && _slist(w, g)) || return nothing
    for k in 1:length(g.node.items)
        push!(d.atoms, _structured_atom(w, _item(g, k), bohr))
    end
    return nothing
end

# One atom of the geometry list. A missing or non-string `symbol`, a `position` of fewer than 3
# numbers and an unknown element are errors in both modes, as in the oracle.
function _structured_atom(w::_Walk, e::_At, bohr::Bool)
    e.node.kind == MapNode ||
        throw(InputError(w.source, 0, lazy"$(e.path): expected a map, got $(_kind_word(e.node.kind))"))
    _check_keys(w, e, ATOM_KEYS)
    sy = _at(e, "symbol")
    sy.present || throw(InputError(w.source, 0, lazy"$(e.path): missing key `symbol`"))
    sy.node.kind == StringNode ||
        throw(InputError(w.source, 0, lazy"$(sy.path): expected a string, got $(_kind_word(sy.node.kind))"))
    pos = _at(e, "position")
    cx, cy, cz = _at(e, "x"), _at(e, "y"), _at(e, "z")
    if pos.present && (cx.present || cy.present || cz.present)
        _both(w, pos, cx.present ? cx : cy.present ? cy : cz)
    end
    if pos.present && pos.node.kind == ArrayNode
        n = length(pos.node.items)
        n < 3 && throw(InputError(w.source, 0, lazy"$(pos.path): expected 3 numbers, got $(n)"))
        _length_ok(w, pos, 3, "numbers", :yaml_list_length)
        x, y, z = _number(w, _item(pos, 1), 0.0), _number(w, _item(pos, 2), 0.0), _number(w, _item(pos, 3), 0.0)
    else
        pos.present && _ok(w, pos, ArrayNode)
        x, y, z = _coordinate(w, e, cx, "x"), _coordinate(w, e, cy, "y"), _coordinate(w, e, cz, "z")
    end
    s = sy.node.s
    zn = try
        element_number(s)
    catch err
        err isa ArgumentError || rethrow()
        m = err.msg
        m isa String || rethrow()
        body = chopprefix(m, "system: ")
        throw(InputError(w.source, 0, lazy"$(sy.path): $(body)"))
    end
    zn == 0 && throw(InputError(w.source, 0, lazy"$(sy.path): unknown element symbol $(repr(s))"))
    p = SVector(x, y, z)
    return Atom(s, bohr ? p : p * ANGSTROM_TO_BOHR)
end

# Coordinate `name` of the `{symbol, x, y, z}` form; the oracle reads a missing one as 0.0.
function _coordinate(w::_Walk, e::_At, c::_At, name::String)
    c.present && return _number(w, c, 0.0)
    _reject(w, :yaml_missing_coordinate, lazy"$(e.path): missing key `$(name)`")
    return 0.0
end

function _irc!(w::_Walk, d::_InputDraft, doc::_At)
    f = _sfloat(w, _alt(w, _at(doc, "irc_step_size"), _at(doc, "irc_step"), FloatNode))
    isnothing(f) || (d.irc_step_size = f)
    x = _suint(w, _alt(w, _at(doc, "irc_max_points"), _at(doc, "irc_points"), IntNode))
    isnothing(x) || (d.irc_max_points = x)
    a = _at(doc, "irc_direction")
    if _sstring(w, a)
        r = irc_direction_word(a.node.s)
        # The oracle reads every word but the forward and backward ones as Both.
        isnothing(r) && _unknown_word(w, a, :yaml_unknown_value, accepted_irc_directions())
        d.irc_direction = isnothing(r) ? Both : r
    end
    return nothing
end

# The 0-based atom indices of `freeze_bond` (`n = 2`) or `freeze_angle` (`n = 3`), returned 1-based
# with unused slots 0, or `nothing` for a list of fewer than `n`, which the oracle ignores. An
# entry that is not an integer is the oracle's `unwrap_or` fallback, 0, 1 or 2 by position.
function _indices(w::_Walk, a::_At, n::Int)
    k = length(a.node.items)
    if k != n
        _reject(w, :yaml_constraint_index, lazy"$(a.path): expected $(n) indices, got $(k)")
        k < n && return nothing
    end
    i1 = _constraint_index(w, _item(a, 1), 0)
    i2 = _constraint_index(w, _item(a, 2), 1)
    i3 = n == 3 ? _constraint_index(w, _item(a, 3), 2) : 0
    return (i1, i2, i3, 0)
end

function _constraint_index(w::_Walk, a::_At, fallback::Int)
    if a.node.kind != IntNode
        _reject(w, :yaml_constraint_index, lazy"$(a.path): expected an atom index (an integer ≥ 0), got $(_kind_word(a.node.kind))")
        return fallback + 1
    end
    x = a.node.i
    x < 0 && throw(InputError(w.source, 0, lazy"$(a.path): expected an atom index (an integer ≥ 0), got $(x)"))
    return x + 1
end

# `constraints`: a list of `{freeze_atom: i}`, `{freeze_bond: [i, j], target}` and
# `{freeze_angle: [i, j, k], target}`, 0-based in the file. A bond target is in bohr; an angle
# target is in degrees and stored in radians.
function _constraints!(w::_Walk, d::_InputDraft, doc::_At)
    c = _at(doc, "constraints")
    (c.present && _slist(w, c)) || return nothing
    for k in 1:length(c.node.items)
        e = _item(c, k)
        _smap(w, e) || continue
        _check_keys(w, e, CONSTRAINT_KEYS)
        fa, fb, fg, t = _at(e, "freeze_atom"), _at(e, "freeze_bond"), _at(e, "freeze_angle"), _at(e, "target")
        if fa.present + fb.present + fg.present != 1
            _reject(w, :yaml_constraint_index, lazy"$(e.path): give one of `freeze_atom`, `freeze_bond` and `freeze_angle`")
        end
        if fa.present && _ok(w, fa, IntNode)
            t.present && _reject(w, :yaml_unknown_key, lazy"$(t.path): a target applies to freeze_bond and freeze_angle")
            push!(d.constraints, Constraint(FreezeAtom, (_constraint_index(w, fa, 0), 0, 0, 0), nothing))
        elseif fb.present && _ok(w, fb, ArrayNode)
            ids = _indices(w, fb, 2)
            isnothing(ids) || push!(d.constraints, Constraint(FreezeBond, ids, t.present ? _sfloat(w, t) : nothing))
        elseif fg.present && _ok(w, fg, ArrayNode)
            ids = _indices(w, fg, 3)
            if !isnothing(ids)
                deg = t.present ? _sfloat(w, t) : nothing
                push!(d.constraints, Constraint(FreezeAngle, ids, isnothing(deg) ? nothing : deg2rad(deg)))
            end
        end
    end
    return nothing
end
