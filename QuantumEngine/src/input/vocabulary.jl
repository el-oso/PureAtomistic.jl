# The words of the input keywords, each table a tuple of `(word, value)` pairs with the word in
# lowercase ASCII, scanned in order. A tuple of one pair type keeps every lookup concrete. The
# words are those the oracle's `.inp` parser accepts for the keyword.

_lower(c::UInt8) = UInt8('A') <= c <= UInt8('Z') ? c + 0x20 : c

# Whether `w` spells the table word `t` without regard to ASCII case, compared byte by byte.
function _word_eq(w::AbstractString, t::String)
    n = ncodeunits(t)
    ncodeunits(w) == n || return false
    for i in 1:n
        _lower(codeunit(w, i)) == codeunit(t, i) || return false
    end
    return true
end

_solvent_skip(c::UInt8) = c == UInt8('-') || c == UInt8('_') || c == UInt8(' ')

# Whether `w` spells the table word `t` without regard to ASCII case once the bytes `-`, `_` and
# space are skipped on both sides.
function _solvent_eq(w::AbstractString, t::String)
    nw = ncodeunits(w)
    nt = ncodeunits(t)
    i = _solvent_next(w, 1)
    j = _solvent_next(t, 1)
    while i <= nw && j <= nt
        _lower(codeunit(w, i)) == _lower(codeunit(t, j)) || return false
        i = _solvent_next(w, i + 1)
        j = _solvent_next(t, j + 1)
    end
    return i > nw && j > nt
end

# The first code unit of `s` at or after `i` that is not skipped, or `ncodeunits(s) + 1`.
function _solvent_next(s::AbstractString, i::Int)
    n = ncodeunits(s)
    while i <= n && _solvent_skip(codeunit(s, i))
        i += 1
    end
    return i
end

# The value of the first pair of `table` whose word `w` spells, or `nothing`.
function _lookup(w::AbstractString, table::Tuple)
    for (t, v) in table
        _word_eq(w, t) && return v
    end
    return nothing
end

const RUN_MODE_WORDS = (
    ("sp", SinglePoint), ("single_point", SinglePoint),
    ("opt", Optimization), ("optimization", Optimization),
    ("ts", TransitionState), ("saddle", TransitionState),
    ("neb", NEB), ("path", NEB),
    ("freq", Frequency), ("frequency", Frequency),
    ("tddft", TDDFT), ("excited", TDDFT),
    ("spectroscopy", Spectroscopy), ("spec", Spectroscopy),
    ("optfreq", OptFreq), ("optimize_frequency", OptFreq),
    ("bsse", BSSE), ("counterpoise", BSSE),
    ("nbo", NBO),
    ("bands", Bands), ("band_structure", Bands),
    ("dos", DOS),
    ("nmr", NMR), ("shielding", NMR),
    ("qtaim", QTAIM), ("topology", QTAIM), ("aim", QTAIM),
    ("irc", IRC), ("intrinsic_reaction_coordinate", IRC),
)

# The methods named by one word, with the reference each asks for. A leading `r` asks for a
# restricted and a leading `u` for an unrestricted reference.
const METHOD_WORDS = (
    ("rhf", (HF, Restricted)), ("uhf", (HF, Unrestricted)), ("hf", (HF, AutoReference)),
    ("mp2", (MP2, AutoReference)), ("rmp2", (MP2, Restricted)), ("ump2", (MP2, Unrestricted)),
    ("mp2-f12", (MP2_F12, AutoReference)), ("mp2_f12", (MP2_F12, AutoReference)),
    ("rmp2-f12", (MP2_F12, Restricted)),
    ("bws2", (BWS2, AutoReference)), ("bw-s2", (BWS2, AutoReference)), ("bwpt2", (BWS2, AutoReference)),
    ("dlpno", (DLPNO_MP2, AutoReference)), ("dlpno-mp2", (DLPNO_MP2, AutoReference)),
    ("dlpno_mp2", (DLPNO_MP2, AutoReference)),
    ("ccsd", (CCSD, AutoReference)), ("rccsd", (CCSD, Restricted)), ("uccsd", (CCSD, Unrestricted)),
    ("ccsd(t)", (CCSD_T, AutoReference)), ("uccsd(t)", (CCSD_T, Unrestricted)),
    ("ccsd-f12", (CCSD_F12, AutoReference)), ("ccsd_f12", (CCSD_F12, AutoReference)),
    ("rccsd-f12", (CCSD_F12, Restricted)),
    ("ccsd(t)-f12", (CCSD_T_F12, AutoReference)), ("ccsd(t)_f12", (CCSD_T_F12, AutoReference)),
    ("ccsd_t_f12", (CCSD_T_F12, AutoReference)),
    ("lda", (LDA, AutoReference)), ("pbe", (PBE, AutoReference)), ("blyp", (BLYP, AutoReference)),
    ("scan", (SCAN, AutoReference)), ("tpss", (TPSS, AutoReference)),
    ("m06l", (M06L, AutoReference)), ("m06-l", (M06L, AutoReference)),
    ("m062x", (M062X, AutoReference)), ("m06-2x", (M062X, AutoReference)),
    ("pbe0", (PBE0, AutoReference)), ("b3lyp", (B3LYP, AutoReference)),
    ("rpa", (RPA, AutoReference)), ("rirpa", (RPA, AutoReference)),
    ("b88-vdw", (B88_VDW, AutoReference)), ("b88_vdw", (B88_VDW, AutoReference)),
    ("pbe-vdw", (PBE_VDW, AutoReference)), ("pbe_vdw", (PBE_VDW, AutoReference)),
    ("b97x-d", (B97X_D, AutoReference)), ("b97xd", (B97X_D, AutoReference)),
    ("wb97m-v", (WB97M_V, AutoReference)), ("wb97mv", (WB97M_V, AutoReference)),
    ("b2plyp", (B2PLYP, AutoReference)), ("mpw2plyp", (MPW2PLYP, AutoReference)),
    ("dsd-pbep86", (DSD_PBEP86, AutoReference)), ("dsdpbep86", (DSD_PBEP86, AutoReference)),
    ("eom-ccsd", (EOM_CCSD, AutoReference)), ("eom_ccsd", (EOM_CCSD, AutoReference)),
    ("pwdft", (PWDFT, AutoReference)), ("pw-dft", (PWDFT, AutoReference)), ("pw_dft", (PWDFT, AutoReference)),
)

# The Libxc forms: the word, then two Int IDs (`false`) or one Int ID and a fraction (`true`).
const LIBXC_WORDS = (
    ("libxc_gga", (LIBXC_GGA, false)), ("libxc_mgga", (LIBXC_MGGA, false)),
    ("libxc_hgga", (LIBXC_HGGA, true)), ("libxc_hmgga", (LIBXC_HMGGA, true)),
)

const SOLVATION_WORDS = (
    ("pcm", PCM), ("true", PCM), ("smd", SMD), ("none", NoSolvation), ("false", NoSolvation),
)

# Each solvent is named by the first of its words.
const SOLVENT_WORDS = (
    ("water", :water), ("h2o", :water),
    ("acetonitrile", :acetonitrile), ("mecn", :acetonitrile), ("ch3cn", :acetonitrile),
    ("methanol", :methanol), ("meoh", :methanol), ("ch3oh", :methanol),
    ("ethanol", :ethanol), ("etoh", :ethanol),
    ("dmso", :dmso), ("dimethylsulfoxide", :dmso),
    ("chloroform", :chloroform), ("chcl3", :chloroform),
    ("dcm", :dcm), ("dichloromethane", :dcm), ("ch2cl2", :dcm),
    ("toluene", :toluene), ("phme", :toluene),
    ("benzene", :benzene), ("phh", :benzene),
    ("thf", :thf), ("tetrahydrofuran", :thf),
    ("hexane", :hexane), ("nhexane", :hexane),
    ("cyclohexane", :cyclohexane),
    ("acetone", :acetone),
    ("dmf", :dmf), ("dimethylformamide", :dmf),
    ("carbontetrachloride", :carbontetrachloride), ("ccl4", :carbontetrachloride),
    ("diethylether", :diethylether), ("ether", :diethylether), ("et2o", :diethylether),
    ("dioxane", :dioxane), ("14dioxane", :dioxane),
    ("nitrobenzene", :nitrobenzene),
    ("pyridine", :pyridine),
)

const VDW_KERNEL_WORDS = (
    ("d2", D2), ("grimme-d2", D2), ("d3", D3), ("grimme-d3", D3), ("d3bj", D3BJ),
    ("b97x-d", B97X_D_DISP), ("vv10", VV10), ("nl-vdw", VV10),
)

const TDDFT_METHOD_WORDS = (
    ("tda", TDA), ("rpa", RPA_TD),
    ("spin_flip", SpinFlip), ("spinflip", SpinFlip), ("sf", SpinFlip), ("sf-tddft", SpinFlip),
)

const GRID_WORDS = (
    ("coarse", 0), ("standard", 1), ("medium", 1), ("fine", 2), ("ultrafine", 3), ("superfine", 4),
    ("0", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4),
)

const IRC_DIRECTION_WORDS = (
    ("forward", Forward), ("fwd", Forward),
    ("backward", Backward), ("bwd", Backward), ("back", Backward),
    ("both", Both),
)

const COMPUTE_MODE_WORDS = (("cpu", CPU), ("gpu", GPU), ("hybrid", Hybrid))

const DIIS_WORDS = (("cdiis", CDIIS), ("ediis", EDIIS))

const POPULATION_WORDS = (("mulliken", Mulliken), ("lowdin", Lowdin))

const SURFACE_WORDS = (("vdw", VdwSurface), ("sas", SASSurface), ("smd", SMDSurface))

# The word lists are built from Base functions alone at load time, so the package owns no
# allocating function that only builds them.
const ACCEPTED_RUN_MODES = collect(map(first, RUN_MODE_WORDS))
const ACCEPTED_METHODS = [collect(map(first, METHOD_WORDS)); collect(map(first, LIBXC_WORDS)); "casscf(e,o)"]
const ACCEPTED_SOLVATION_MODELS = collect(map(first, SOLVATION_WORDS))
const ACCEPTED_SOLVENTS = collect(map(first, SOLVENT_WORDS))
const ACCEPTED_VDW_KERNELS = collect(map(first, VDW_KERNEL_WORDS))
const ACCEPTED_TDDFT_METHODS = collect(map(first, TDDFT_METHOD_WORDS))
const ACCEPTED_GRIDS = collect(map(first, GRID_WORDS))
const ACCEPTED_IRC_DIRECTIONS = collect(map(first, IRC_DIRECTION_WORDS))
const ACCEPTED_COMPUTE_MODES = collect(map(first, COMPUTE_MODE_WORDS))
const ACCEPTED_DIIS_METHODS = collect(map(first, DIIS_WORDS))
const ACCEPTED_POPULATIONS = collect(map(first, POPULATION_WORDS))
const ACCEPTED_SURFACES = collect(map(first, SURFACE_WORDS))

"""
    run_mode_word(w::AbstractString) -> Union{Nothing,RunMode}

The run mode `w` names, in any ASCII case, or `nothing` for an unknown word. The words are
`accepted_run_modes()`.
"""
run_mode_word(w::AbstractString) = _lookup(w, RUN_MODE_WORDS)

"""
    solvation_word(w::AbstractString) -> Union{Nothing,SolvationModel}

The solvation model `w` names: `pcm` and `true` are `PCM`, `smd` is `SMD`, `none` and `false`
are `NoSolvation`; `nothing` for any other word.
"""
solvation_word(w::AbstractString) = _lookup(w, SOLVATION_WORDS)

"""
    solvent_word(w::AbstractString) -> Union{Nothing,Symbol}

The solvent `w` names, as the `Symbol` of its first word in `accepted_solvents()`, or `nothing`.
The comparison ignores ASCII case and the characters `-`, `_` and space, so `Di-chloro_methane`
is `:dcm`.
"""
function solvent_word(w::AbstractString)
    for (t, v) in SOLVENT_WORDS
        _solvent_eq(w, t) && return v
    end
    return nothing
end

"""
    vdw_kernel_word(w::AbstractString) -> Union{Nothing,VdwKernel}

The van der Waals kernel `w` names, or `nothing`. The words are `accepted_vdw_kernels()`.
"""
vdw_kernel_word(w::AbstractString) = _lookup(w, VDW_KERNEL_WORDS)

"""
    tddft_method_word(w::AbstractString) -> Union{Nothing,TDDFTMethod}

The excited-state method `w` names (`tda`; `rpa`; `spin_flip`, `spinflip`, `sf`, `sf-tddft`), or
`nothing`.
"""
tddft_method_word(w::AbstractString) = _lookup(w, TDDFT_METHOD_WORDS)

"""
    grid_word(w::AbstractString) -> Union{Nothing,Int}

The integration grid level `w` names: `coarse` 0, `standard` and `medium` 1, `fine` 2,
`ultrafine` 3, `superfine` 4, or one of the digits `0` to `4`; `nothing` for anything else.
"""
grid_word(w::AbstractString) = _lookup(w, GRID_WORDS)

"""
    irc_direction_word(w::AbstractString) -> Union{Nothing,IRCDirection}

The IRC direction `w` names (`forward`, `fwd`; `backward`, `bwd`, `back`; `both`), or `nothing`.
"""
irc_direction_word(w::AbstractString) = _lookup(w, IRC_DIRECTION_WORDS)

"""
    compute_mode_word(w::AbstractString) -> Union{Nothing,ComputeMode}

The compute mode `w` names (`cpu`, `gpu`, `hybrid`), or `nothing`.
"""
compute_mode_word(w::AbstractString) = _lookup(w, COMPUTE_MODE_WORDS)

"""
    diis_word(w::AbstractString) -> Union{Nothing,DIISMethod}

The DIIS method `w` names (`cdiis`, `ediis`), or `nothing`.
"""
diis_word(w::AbstractString) = _lookup(w, DIIS_WORDS)

"""
    population_word(w::AbstractString) -> Union{Nothing,PopulationKind}

The population analysis `w` names (`mulliken`, `lowdin`), or `nothing`.
"""
population_word(w::AbstractString) = _lookup(w, POPULATION_WORDS)

"""
    surface_word(w::AbstractString) -> Union{Nothing,PCMSurface}

The cavity surface `w` names (`vdw`, `sas`, `smd`), or `nothing`.
"""
surface_word(w::AbstractString) = _lookup(w, SURFACE_WORDS)

# The decimal integer spelled by the code units `lo:hi` of `s` (an optional sign when `signed`,
# then one or more ASCII digits) when it lies in `-limit:limit`, or `nothing`.
function _decimal_int(s::AbstractString, lo::Int, hi::Int, signed::Bool, limit::Int)
    neg = false
    if signed && lo <= hi && (codeunit(s, lo) == UInt8('+') || codeunit(s, lo) == UInt8('-'))
        neg = codeunit(s, lo) == UInt8('-')
        lo += 1
    end
    lo <= hi || return nothing
    v = 0
    for i in lo:hi
        c = codeunit(s, i)
        UInt8('0') <= c <= UInt8('9') || return nothing
        v = 10 * v + Int(c - UInt8('0'))
        v > limit && return nothing
    end
    return neg ? -v : v
end

# A Libxc ID: a signed 32-bit decimal integer.
_libxc_id(s::AbstractString) = _decimal_int(s, 1, ncodeunits(s), true, Int(typemax(Int32)))

# A fraction as Parsers.jl reads a `Float64`, except that a hexadecimal float is rejected.
function _fraction(s::AbstractString)
    i = 1
    n = ncodeunits(s)
    i <= n && (codeunit(s, i) == UInt8('+') || codeunit(s, i) == UInt8('-')) && (i += 1)
    i + 1 <= n && codeunit(s, i) == UInt8('0') && _lower(codeunit(s, i + 1)) == UInt8('x') && return nothing
    return Parsers.tryparse(Float64, s)
end

# `casscf(e,o)` in any ASCII case, with `e` and `o` unsigned decimal integers.
function _casscf(w::AbstractString)
    n = ncodeunits(w)
    n >= 11 || return nothing
    for (i, c) in enumerate(codeunits("casscf("))
        _lower(codeunit(w, i)) == c || return nothing
    end
    codeunit(w, n) == UInt8(')') || return nothing
    comma = 0
    for i in 8:(n - 1)
        if codeunit(w, i) == UInt8(',')
            comma = i
            break
        end
    end
    iszero(comma) && return nothing
    limit = Int(typemax(Int32))
    e = _decimal_int(w, 8, comma - 1, false, limit)
    o = _decimal_int(w, comma + 1, n - 1, false, limit)
    (isnothing(e) || isnothing(o)) && return nothing
    return MethodSpec(CASSCF, AutoReference, (e, o), 0.0)
end

"""
    method_word(tokens::AbstractVector{<:AbstractString}) -> Union{Nothing,MethodSpec}

The method that the tokens after the `method` keyword name, or `nothing` when they name none.
The first token is the method word, in any ASCII case (`accepted_methods()`); `rhf`, `rmp2`,
`rmp2-f12`, `rccsd` and `rccsd-f12` ask for a restricted reference, `uhf`, `ump2`, `uccsd` and
`uccsd(t)` for an unrestricted one, and every other word for `AutoReference`. `casscf(e,o)` is one
token. `libxc_gga x c` and `libxc_mgga x c` read the two following tokens as Libxc IDs, and
`libxc_hgga id exx` and `libxc_hmgga id exx` read an ID and the exact-exchange fraction; a missing
or malformed number gives `nothing`. Tokens after those the method reads are left to the caller.
`tokens` may have any indices.
"""
function method_word(tokens::AbstractVector{<:AbstractString})
    isempty(tokens) && return nothing
    i = firstindex(tokens)
    w = tokens[i]
    kr = _lookup(w, METHOD_WORDS)
    isnothing(kr) || return MethodSpec(kr[1], kr[2], (0, 0), 0.0)
    kh = _lookup(w, LIBXC_WORDS)
    isnothing(kh) && return _casscf(w)
    length(tokens) >= 3 || return nothing
    kind, hybrid = kh
    id = _libxc_id(tokens[i + 1])
    isnothing(id) && return nothing
    if hybrid
        exx = _fraction(tokens[i + 2])
        isnothing(exx) && return nothing
        return MethodSpec(kind, AutoReference, (id, 0), exx)
    end
    id2 = _libxc_id(tokens[i + 2])
    isnothing(id2) && return nothing
    return MethodSpec(kind, AutoReference, (id, id2), 0.0)
end

"""
    accepted_run_modes() -> Vector{String}

The words `run_mode_word` accepts, for error messages. The same vector is returned on every call;
it must not be modified. `accepted_methods`, `accepted_solvation_models`, `accepted_solvents`,
`accepted_vdw_kernels`, `accepted_tddft_methods`, `accepted_grids`, `accepted_irc_directions`,
`accepted_compute_modes`, `accepted_diis_methods`, `accepted_populations` and `accepted_surfaces`
list the words of the other vocabulary functions in the same way; `accepted_methods` lists the
CASSCF form as `casscf(e,o)`.
"""
accepted_run_modes() = ACCEPTED_RUN_MODES
accepted_methods() = ACCEPTED_METHODS
accepted_solvation_models() = ACCEPTED_SOLVATION_MODELS
accepted_solvents() = ACCEPTED_SOLVENTS
accepted_vdw_kernels() = ACCEPTED_VDW_KERNELS
accepted_tddft_methods() = ACCEPTED_TDDFT_METHODS
accepted_grids() = ACCEPTED_GRIDS
accepted_irc_directions() = ACCEPTED_IRC_DIRECTIONS
accepted_compute_modes() = ACCEPTED_COMPUTE_MODES
accepted_diis_methods() = ACCEPTED_DIIS_METHODS
accepted_populations() = ACCEPTED_POPULATIONS
accepted_surfaces() = ACCEPTED_SURFACES
