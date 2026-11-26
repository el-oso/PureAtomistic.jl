"""
    QuantumEngine.Input

The typed input file. `InputFile` holds every setting with the oracle's default, `ParsedInput`
pairs it with the warnings of lenient reading, and the enumerations and structs of its fields
name the methods, run modes and options. The vocabulary functions (`method_word`,
`run_mode_word`, `solvent_word` and the others) map the words an input file may use to those
values, without regard to ASCII case, and return `nothing` for an unknown word; each has an
`accepted_*` companion that lists its words.
"""
module Input

using Parsers: Parsers
using StaticArrays: SVector, SMatrix
using StrictMode: @strict_exempt, @strict_policy
using ..Errors: InputWarning
using ..Units: ANGSTROM_TO_BOHR
using ..System: Atom, PointCharge

include("types.jl")
include("vocabulary.jl")

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

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
