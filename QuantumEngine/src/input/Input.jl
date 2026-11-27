"""
    QuantumEngine.Input

The typed input file. `InputFile` holds every setting with the oracle's default, `ParsedInput`
pairs it with the warnings of lenient reading, and the enumerations and structs of its fields
name the methods, run modes and options. The vocabulary functions (`method_word`,
`run_mode_word`, `solvent_word` and the others) map the words an input file may use to those
values, without regard to ASCII case, and return `nothing` for an unknown word; each has an
`accepted_*` companion that lists its words. `parse_inp` reads an `.inp` file into a `ParsedInput`,
and `molecule` builds the `Molecule` of an `InputFile`. Structured input is a tree of `Node`s:
`json_node` reads JSON text into one and `node` any `AbstractDict`, `parse_structured` reads the tree
into a `ParsedInput`, and `parse_json` and `parse_dict` do both. `parse_yaml` reads YAML text; its
method is in the extension `QuantumEngineYAMLExt`, which loads with YAML.jl. `parse_input` reads a
file with the reader its extension names (`input_format`), a molecule file into an `InputFile` that
holds only its atoms.
"""
module Input

using JSON: JSON
using Parsers: Parsers
using StaticArrays: SVector, SMatrix
using StrictMode: @strict_exempt, @strict_policy
using ..Errors: InputError, InputWarning
using ..Units: ANGSTROM_TO_BOHR
using ..System: Atom, PointCharge, Molecule, nalpha, element_number
using ..IO: input_line, reject_or_warn, parse_coordinate, read_xyz, read_pdb, read_cif

include("types.jl")
include("vocabulary.jl")
include("inp.jl")
include("node.jl")
include("structured.jl")
include("json.jl")
include("dict.jl")
include("dispatch.jl")

export RunMode, SinglePoint, Optimization, TransitionState, NEB, Frequency, TDDFT, Spectroscopy,
    OptFreq, BSSE, NBO, Bands, DOS, NMR, QTAIM, IRC
export Reference, AutoReference, Restricted, Unrestricted
export MethodKind, HF, MP2, MP2_F12, BWS2, DLPNO_MP2, CCSD, CCSD_T, CCSD_F12, CCSD_T_F12, LDA, PBE,
    BLYP, SCAN, TPSS, M06L, M062X, PBE0, B3LYP, RPA, B88_VDW, PBE_VDW, B97X_D, WB97M_V, B2PLYP,
    MPW2PLYP, DSD_PBEP86, EOM_CCSD, PWDFT, CASSCF, LIBXC_GGA, LIBXC_MGGA, LIBXC_HGGA, LIBXC_HMGGA
export ComputeMode, CPU, GPU, Hybrid
export DIISMethod, CDIIS, EDIIS
export SolvationModel, NoSolvation, PCM, SMD
export PCMSurface, VdwSurface, SASSurface, SMDSurface
export VdwKernel, D2, D3, D3BJ, B97X_D_DISP, VV10
export TDDFTMethod, TDA, RPA_TD, SpinFlip
export IRCDirection, Forward, Backward, Both
export PopulationKind, Mulliken, Lowdin
export ConstraintKind, FreezeAtom, FreezeBond, FreezeAngle, FreezeDihedral
export InputFormat, INP, JSON_INPUT, YAML_INPUT, XYZ, PDB, CIF
export MethodSpec, Constraint, HubbardU, ONIOMSpec, SolvationSpec, InputFile, ParsedInput,
    use_fock_damping
export run_mode_word, method_word, solvation_word, solvent_word, vdw_kernel_word, tddft_method_word,
    grid_word, irc_direction_word, compute_mode_word, diis_word, population_word, surface_word
export accepted_run_modes, accepted_methods, accepted_solvation_models, accepted_solvents,
    accepted_vdw_kernels, accepted_tddft_methods, accepted_grids, accepted_irc_directions,
    accepted_compute_modes, accepted_diis_methods, accepted_populations, accepted_surfaces
export parse_inp, molecule, accepted_keywords
export NodeKind, NullNode, BoolNode, IntNode, FloatNode, StringNode, ArrayNode, MapNode, Node
export parse_structured, json_node, parse_json, node, parse_dict, parse_yaml
export input_format, parse_input

# The keyword body of `InputFile` allocates only when it stores an `ONIOMSpec`, which it boxes into
# the `Union{Nothing,ONIOMSpec}` field; every other call is held to `:noalloc`.
@strict_exempt :noalloc types = (
    ntuple(Returns(Any), Base.fieldindex(InputFile, :oniom) - 1)..., ONIOMSpec,
    ntuple(Returns(Any), fieldcount(InputFile) - Base.fieldindex(InputFile, :oniom))..., Type{InputFile},
) InputFile reason = "entry point: keyword body of the InputFile constructor, boxes an ONIOMSpec"
@strict_exempt :noalloc ONIOMSpec reason = "entry point: ONIOMSpec constructor"
@strict_exempt :noalloc ParsedInput reason = "entry point: ParsedInput constructor"
@strict_exempt :noalloc HubbardU reason = "entry point: HubbardU constructor"
@strict_exempt :typestable types = (AbstractString,) solvent_word reason = "returns a Symbol or nothing for an unknown word; the union is split at every call site and allocates nothing"
@strict_exempt :noalloc types = (AbstractString,) _fraction reason = "Parsers' exact float path allocates (an 800-digit buffer); remove this exemption when a Parsers release contains JuliaData/Parsers.jl#244"
@strict_exempt :noalloc types = (AbstractVector{<:AbstractString},) method_word reason = "Parsers' exact float path allocates (an 800-digit buffer); remove this exemption when a Parsers release contains JuliaData/Parsers.jl#244"

@strict_exempt :noalloc parse_inp reason = "entry point: .inp parser"
@strict_exempt :noalloc accepted_keywords reason = "entry point: keyword list, returns a copy"
@strict_exempt :noalloc _finalize_input reason = "entry point helper: input checks of parse_inp"
@strict_exempt :noalloc _IndexSites reason = "entry point helper: index-site constructor of parse_inp and parse_structured"
@strict_exempt :noalloc _check_atom_indices reason = "entry point helper: atom index checks of _finalize_input"
@strict_exempt :noalloc _site_label reason = "entry point helper: entry label of _finalize_input"
@strict_exempt :noalloc _index_in_range reason = "entry point helper: atom index check of _finalize_input"
@strict_exempt :noalloc _repeated_atom reason = "entry point helper: repeated atom check of _finalize_input"
@strict_exempt :noalloc _check_hubbard_symbol reason = "entry point helper: hubbard symbol check of parse_inp and parse_structured"
@strict_exempt :noalloc _constraint_site! reason = "entry point helper: constraint site recorder of parse_inp"
@strict_exempt :noalloc _add_fragments! reason = "entry point helper: fragment reader of parse_inp"
@strict_exempt :noalloc _InputDraft reason = "entry point: input draft constructor"
@strict_exempt :noalloc _draft reason = "entry point helper: input draft of parse_inp and _finalize_input"
@strict_exempt :noalloc types = (_InputDraft, ONIOMSpec) _build_input reason = "entry point helper: InputFile builder of parse_inp and _finalize_input"
@strict_exempt :noalloc types = (_InputDraft, ONIOMSpec) _input reason = "entry point helper: InputFile builder of parse_inp and _finalize_input"
@strict_exempt :noalloc _InpState reason = "entry point: parser state constructor of parse_inp"
@strict_exempt :noalloc _reject reason = "entry point helper: strict-or-lenient choice of parse_inp and parse_structured"
@strict_exempt :noalloc _line! reason = "entry point helper: line reader of parse_inp"
@strict_exempt :noalloc _join reason = "entry point helper: word joiner of parse_inp and parse_structured, builds a String"
@strict_exempt :noalloc _finite reason = "entry point helper: number check of parse_inp"
@strict_exempt :noalloc _float_required reason = "entry point helper: number reader of parse_inp"
@strict_exempt :noalloc _float_optional reason = "entry point helper: number reader of parse_inp"
@strict_exempt :noalloc _int_optional reason = "entry point helper: integer reader of parse_inp"
@strict_exempt :noalloc _int32_optional reason = "entry point helper: Int32 reader of parse_inp"
@strict_exempt :noalloc _int32_msg reason = "entry point helper: Int32 range message of parse_inp"
@strict_exempt :noalloc _float_or reason = "entry point helper: number reader with the oracle's fallback, of parse_inp"
@strict_exempt :noalloc _uint_msg reason = "entry point helper: unsigned-integer message of parse_inp"
@strict_exempt :noalloc _uint_optional reason = "entry point helper: unsigned integer reader of parse_inp"
@strict_exempt :noalloc _uint_or reason = "entry point helper: unsigned integer reader with the oracle's fallback, of parse_inp"
@strict_exempt :noalloc _casscf_oracle reason = "entry point helper: CASSCF active-space reader of parse_inp"
@strict_exempt :noalloc _casscf_counts reason = "entry point helper: CASSCF active-space reader of parse_inp and parse_structured"
@strict_exempt :noalloc _missing! reason = "entry point helper: missing-value reader of parse_inp"
@strict_exempt :noalloc _index reason = "entry point helper: atom index reader of parse_inp"
@strict_exempt :noalloc _bool reason = "entry point helper: Boolean reader of parse_inp"
@strict_exempt :noalloc _unknown_msg reason = "entry point helper: unknown-word message of parse_inp and parse_structured"
@strict_exempt :noalloc _unknown reason = "entry point helper: unknown-word rejection of parse_inp"
@strict_exempt :noalloc _count_ok reason = "entry point helper: value count check of parse_inp"
@strict_exempt :noalloc _method_value reason = "entry point helper: method reader of parse_inp"
@strict_exempt :noalloc _keyword_line! reason = "entry point helper: keyword line reader of parse_inp"
@strict_exempt :noalloc _keyword_value! reason = "entry point helper: keyword value reader of parse_inp"
@strict_exempt :noalloc _model_atoms! reason = "entry point helper: ONIOM model atom reader of parse_inp"
@strict_exempt :noalloc _freeze! reason = "entry point helper: constraint reader of parse_inp"
@strict_exempt :noalloc _parse_fragments reason = "entry point helper: fragment reader of parse_inp"
@strict_exempt :noalloc _fragments_keyword! reason = "entry point helper: fragments keyword reader of parse_inp"
@strict_exempt :noalloc _open_block! reason = "entry point helper: block opener of parse_inp"
@strict_exempt :noalloc _inside reason = "entry point helper: block line message of parse_inp"
@strict_exempt :noalloc _block_line! reason = "entry point helper: block line reader of parse_inp"
@strict_exempt :noalloc _position reason = "entry point helper: position reader of parse_inp"
@strict_exempt :noalloc _geometry_line! reason = "entry point helper: geometry line reader of parse_inp"
@strict_exempt :noalloc _charges_line! reason = "entry point helper: charges line reader of parse_inp"
@strict_exempt :noalloc _hubbard_line! reason = "entry point helper: hubbard line reader of parse_inp"
@strict_exempt :noalloc _cell_line! reason = "entry point helper: cell line reader of parse_inp"
@strict_exempt :noalloc _pbc_line! reason = "entry point helper: pbc line reader of parse_inp"
@strict_exempt :noalloc _oniom_line! reason = "entry point helper: oniom line reader of parse_inp"
@strict_exempt :noalloc _close_block! reason = "entry point helper: block closer of parse_inp"
@strict_exempt :noalloc _resolve_solvation! reason = "entry point helper: solvation resolution of parse_inp"
@strict_exempt :noalloc _solvation_by_keyword! reason = "entry point helper: order-independent solvation of parse_inp"
@strict_exempt :noalloc _check_f12 reason = "entry point helper: F12 option check of parse_inp"

@strict_exempt :noalloc Node reason = "entry point: Node constructor"
@strict_exempt :noalloc parse_structured reason = "entry point: structured-input reader"
@strict_exempt :typestable :noalloc :noboxing types = (AbstractString,) parse_json reason = "reads JSON.jl's untyped values (JSON.parse returns Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (AbstractString, Bool, typeof(parse_json), AbstractString) parse_json reason = "reads JSON.jl's untyped values (JSON.parse returns Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (AbstractString,) json_node reason = "reads JSON.jl's untyped values (JSON.parse returns Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (AbstractString, typeof(json_node), AbstractString) json_node reason = "reads JSON.jl's untyped values (JSON.parse returns Any); not on the trimmed path"
@strict_exempt :noalloc _JSONObject reason = "entry point: JSON object constructor of json_node"
@strict_exempt :typestable :noalloc :noboxing types = (_JSONObject, Any, AbstractString) setindex! reason = "reads JSON.jl's untyped values (JSON.parse returns Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (_JSONObject,) iterate reason = "reads JSON.jl's untyped values (JSON.parse returns Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (_JSONObject, Int) iterate reason = "reads JSON.jl's untyped values (JSON.parse returns Any); not on the trimmed path"
@strict_exempt :noalloc _Walk reason = "entry point helper: walk state constructor of parse_structured"
@strict_exempt :noalloc _At reason = "entry point helper: key-path constructor of parse_structured"
@strict_exempt :noalloc _child reason = "entry point helper: key path builder of parse_structured"
@strict_exempt :noalloc _list_path reason = "entry point helper: key path builder of parse_structured"
@strict_exempt :noalloc _at reason = "entry point helper: map lookup of parse_structured"
@strict_exempt :noalloc _item reason = "entry point helper: list lookup of parse_structured"
@strict_exempt :noalloc _ok reason = "entry point helper: type check of parse_structured"
@strict_exempt :noalloc _sbool reason = "entry point helper: Boolean reader of parse_structured"
@strict_exempt :noalloc _sint reason = "entry point helper: integer reader of parse_structured"
@strict_exempt :noalloc _scharge reason = "entry point helper: charge reader of parse_structured"
@strict_exempt :noalloc _suint reason = "entry point helper: unsigned integer reader of parse_structured"
@strict_exempt :noalloc _sfloat reason = "entry point helper: number reader of parse_structured"
@strict_exempt :noalloc _number reason = "entry point helper: list number reader of parse_structured"
@strict_exempt :noalloc _sstring reason = "entry point helper: string check of parse_structured"
@strict_exempt :noalloc _smap reason = "entry point helper: map check of parse_structured"
@strict_exempt :noalloc _slist reason = "entry point helper: list check of parse_structured"
@strict_exempt :noalloc _check_keys reason = "entry point helper: unknown-key check of parse_structured"
@strict_exempt :noalloc _both reason = "entry point helper: fallback check of parse_structured"
@strict_exempt :noalloc _first reason = "entry point helper: fallback reader of parse_structured"
@strict_exempt :noalloc _alt reason = "entry point helper: fallback reader of parse_structured"
@strict_exempt :noalloc _unknown_word reason = "entry point helper: unknown-word rejection of parse_structured"
@strict_exempt :noalloc _hubbard_map reason = "entry point helper: element map reader of parse_structured"
@strict_exempt :noalloc _structured_method reason = "entry point helper: method reader of parse_structured"
@strict_exempt :noalloc _top! reason = "entry point helper: top-level reader of parse_structured"
@strict_exempt :noalloc _scf! reason = "entry point helper: scf reader of parse_structured"
@strict_exempt :noalloc _length_ok reason = "entry point helper: list length check of parse_structured"
@strict_exempt :noalloc _field! reason = "entry point helper: field reader of parse_structured"
@strict_exempt :noalloc _lattice! reason = "entry point helper: lattice reader of parse_structured"
@strict_exempt :noalloc _lattice_row reason = "entry point helper: lattice row reader of parse_structured"
@strict_exempt :noalloc _pbc! reason = "entry point helper: pbc reader of parse_structured"
@strict_exempt :noalloc _population! reason = "entry point helper: population reader of parse_structured"
@strict_exempt :noalloc _count reason = "entry point helper: list count reader of parse_structured"
@strict_exempt :noalloc _kpoints! reason = "entry point helper: k-point reader of parse_structured"
@strict_exempt :noalloc _kpath! reason = "entry point helper: k-path reader of parse_structured"
@strict_exempt :noalloc _dmft! reason = "entry point helper: DMFT reader of parse_structured"
@strict_exempt :noalloc _solvation! reason = "entry point helper: solvation reader of parse_structured"
@strict_exempt :noalloc _oniom reason = "entry point helper: ONIOM reader of parse_structured"
@strict_exempt :noalloc _tddft! reason = "entry point helper: TDDFT reader of parse_structured"
@strict_exempt :noalloc _exports! reason = "entry point helper: exports reader of parse_structured"
@strict_exempt :noalloc _f12! reason = "entry point helper: F12 reader of parse_structured"
@strict_exempt :noalloc _bohr reason = "entry point helper: unit reader of parse_structured"
@strict_exempt :noalloc _geometry! reason = "entry point helper: geometry reader of parse_structured"
@strict_exempt :noalloc _structured_atom reason = "entry point helper: atom reader of parse_structured"
@strict_exempt :noalloc _coordinate reason = "entry point helper: coordinate reader of parse_structured"
@strict_exempt :noalloc _irc! reason = "entry point helper: IRC reader of parse_structured"
@strict_exempt :noalloc _indices reason = "entry point helper: constraint index reader of parse_structured"
@strict_exempt :noalloc _constraint_index reason = "entry point helper: constraint index reader of parse_structured"
@strict_exempt :noalloc _constraints! reason = "entry point helper: constraint reader of parse_structured"

@strict_exempt :typestable :noalloc :noboxing types = (AbstractDict,) node reason = "reads a dictionary's untyped values (an AbstractDict holds Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (AbstractString, typeof(node), AbstractDict) node reason = "reads a dictionary's untyped values (an AbstractDict holds Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (Any, String, String) _dict_node reason = "reads a dictionary's untyped values (an AbstractDict holds Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (AbstractDict, String, String) _dict_map reason = "reads a dictionary's untyped values (an AbstractDict holds Any); not on the trimmed path"
@strict_exempt :noalloc _path_prefix reason = "entry point helper: key path message of node"
@strict_exempt :typestable :noalloc :noboxing types = (AbstractDict,) parse_dict reason = "reads a dictionary's untyped values (an AbstractDict holds Any); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (AbstractString, Bool, typeof(parse_dict), AbstractDict) parse_dict reason = "reads a dictionary's untyped values (an AbstractDict holds Any); not on the trimmed path"
@strict_exempt :noalloc _yaml_unavailable reason = "entry point: error constructor"
@strict_exempt :typestable :noalloc :noboxing types = (AbstractString,) parse_input reason = "dispatches to the readers of untyped values (parse_json, parse_yaml); not on the trimmed path"
@strict_exempt :typestable :noalloc :noboxing types = (Bool, typeof(parse_input), AbstractString) parse_input reason = "dispatches to the readers of untyped values (parse_json, parse_yaml); not on the trimmed path"
@strict_exempt :noalloc _read_file reason = "entry point helper: file reader of parse_input, builds a String"
@strict_exempt :noalloc _inp_file reason = "entry point helper: .inp file reader of parse_input"
@strict_exempt :noalloc _molecule_input reason = "entry point helper: molecule file reader of parse_input"

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
