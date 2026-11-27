"""
    RunMode

What a run computes: `SinglePoint`, `Optimization`, `TransitionState`, `NEB`, `Frequency`,
`TDDFT`, `Spectroscopy`, `OptFreq`, `BSSE`, `NBO`, `Bands`, `DOS`, `NMR`, `QTAIM` or `IRC`.
"""
@enum RunMode SinglePoint Optimization TransitionState NEB Frequency TDDFT Spectroscopy OptFreq BSSE NBO Bands DOS NMR QTAIM IRC

@doc "`SinglePoint::RunMode`: one energy at the input geometry." SinglePoint
@doc "`Optimization::RunMode`: a geometry optimization to a minimum." Optimization
@doc "`TransitionState::RunMode`: a transition-state (saddle point) search." TransitionState
@doc "`NEB::RunMode`: a nudged elastic band reaction path." NEB
@doc "`Frequency::RunMode`: harmonic vibrational frequencies at the input geometry." Frequency
@doc "`TDDFT::RunMode`: excited states by time-dependent density functional theory." TDDFT
@doc "`Spectroscopy::RunMode`: a spectroscopy run." Spectroscopy
@doc "`OptFreq::RunMode`: a geometry optimization followed by the frequencies at the optimized geometry." OptFreq
@doc "`BSSE::RunMode`: a counterpoise correction of the basis-set superposition error." BSSE
@doc "`NBO::RunMode`: a natural bond orbital analysis." NBO
@doc "`Bands::RunMode`: a band structure." Bands
@doc "`DOS::RunMode`: a density of states." DOS
@doc "`NMR::RunMode`: NMR shieldings." NMR
@doc "`QTAIM::RunMode`: the topology of the electron density (quantum theory of atoms in molecules)." QTAIM
@doc "`IRC::RunMode`: an intrinsic reaction coordinate from a transition state." IRC

"""
    Reference

The reference wave function a method asks for: `Restricted`, `Unrestricted`, or
`AutoReference`, which chooses restricted for a singlet and unrestricted otherwise.
"""
@enum Reference AutoReference Restricted Unrestricted

@doc "`AutoReference::Reference`: restricted for a singlet, unrestricted otherwise." AutoReference
@doc "`Restricted::Reference`: a restricted reference." Restricted
@doc "`Unrestricted::Reference`: an unrestricted reference." Unrestricted

"""
    MethodKind

The electronic-structure method. `CCSD_T` is CCSD(T); `LIBXC_GGA`, `LIBXC_MGGA`, `LIBXC_HGGA` and
`LIBXC_HMGGA` are functionals named by their Libxc IDs.
"""
@enum MethodKind HF MP2 MP2_F12 BWS2 DLPNO_MP2 CCSD CCSD_T CCSD_F12 CCSD_T_F12 LDA PBE BLYP SCAN TPSS M06L M062X PBE0 B3LYP RPA B88_VDW PBE_VDW B97X_D WB97M_V B2PLYP MPW2PLYP DSD_PBEP86 EOM_CCSD PWDFT CASSCF LIBXC_GGA LIBXC_MGGA LIBXC_HGGA LIBXC_HMGGA

@doc "`HF::MethodKind`: Hartree–Fock." HF
@doc "`MP2::MethodKind`: second-order Møller–Plesset perturbation theory." MP2
@doc "`MP2_F12::MethodKind`: explicitly correlated MP2-F12." MP2_F12
@doc "`BWS2::MethodKind`: Brillouin–Wigner second-order perturbation theory (BW-s2)." BWS2
@doc "`DLPNO_MP2::MethodKind`: domain-based local pair natural orbital MP2." DLPNO_MP2
@doc "`CCSD::MethodKind`: coupled cluster with single and double excitations." CCSD
@doc "`CCSD_T::MethodKind`: CCSD with perturbative triples, CCSD(T)." CCSD_T
@doc "`CCSD_F12::MethodKind`: explicitly correlated CCSD-F12." CCSD_F12
@doc "`CCSD_T_F12::MethodKind`: explicitly correlated CCSD(T)-F12." CCSD_T_F12
@doc "`LDA::MethodKind`: the local density approximation." LDA
@doc "`PBE::MethodKind`: the PBE generalized gradient approximation." PBE
@doc "`BLYP::MethodKind`: the BLYP generalized gradient approximation." BLYP
@doc "`SCAN::MethodKind`: the SCAN meta-GGA." SCAN
@doc "`TPSS::MethodKind`: the TPSS meta-GGA." TPSS
@doc "`M06L::MethodKind`: the M06-L meta-GGA." M06L
@doc "`M062X::MethodKind`: the M06-2X hybrid meta-GGA." M062X
@doc "`PBE0::MethodKind`: the PBE0 hybrid." PBE0
@doc "`B3LYP::MethodKind`: the B3LYP hybrid." B3LYP
@doc "`RPA::MethodKind`: the random phase approximation." RPA
@doc "`B88_VDW::MethodKind`: the van der Waals functional of input word `b88-vdw`." B88_VDW
@doc "`PBE_VDW::MethodKind`: the van der Waals functional of input word `pbe-vdw`." PBE_VDW
@doc "`B97X_D::MethodKind`: the range-separated hybrid ωB97X-D." B97X_D
@doc "`WB97M_V::MethodKind`: the range-separated hybrid meta-GGA ωB97M-V." WB97M_V
@doc "`B2PLYP::MethodKind`: the B2PLYP double hybrid." B2PLYP
@doc "`MPW2PLYP::MethodKind`: the mPW2PLYP double hybrid." MPW2PLYP
@doc "`DSD_PBEP86::MethodKind`: the DSD-PBEP86 double hybrid." DSD_PBEP86
@doc "`EOM_CCSD::MethodKind`: equation-of-motion CCSD." EOM_CCSD
@doc "`PWDFT::MethodKind`: plane-wave density functional theory." PWDFT
@doc "`CASSCF::MethodKind`: complete active space SCF; `MethodSpec.ints` holds the active electrons and orbitals." CASSCF
@doc "`LIBXC_GGA::MethodKind`: a GGA given by its exchange and correlation Libxc IDs." LIBXC_GGA
@doc "`LIBXC_MGGA::MethodKind`: a meta-GGA given by its exchange and correlation Libxc IDs." LIBXC_MGGA
@doc "`LIBXC_HGGA::MethodKind`: a hybrid GGA given by its Libxc ID and exact-exchange fraction." LIBXC_HGGA
@doc "`LIBXC_HMGGA::MethodKind`: a hybrid meta-GGA given by its Libxc ID and exact-exchange fraction." LIBXC_HMGGA

"""
    ComputeMode

Where a run computes: `CPU`, `GPU` or `Hybrid`.
"""
@enum ComputeMode CPU GPU Hybrid

@doc "`CPU::ComputeMode`: on the CPU." CPU
@doc "`GPU::ComputeMode`: on the GPU." GPU
@doc "`Hybrid::ComputeMode`: on the CPU and the GPU together." Hybrid

"""
    DIISMethod

The SCF extrapolation: `CDIIS` or `EDIIS`.
"""
@enum DIISMethod CDIIS EDIIS

@doc "`CDIIS::DIISMethod`: commutator DIIS." CDIIS
@doc "`EDIIS::DIISMethod`: energy DIIS." EDIIS

"""
    SolvationModel

The continuum solvation model: `NoSolvation`, `PCM` or `SMD`.
"""
@enum SolvationModel NoSolvation PCM SMD

@doc "`NoSolvation::SolvationModel`: no solvation model." NoSolvation
@doc "`PCM::SolvationModel`: the polarizable continuum model." PCM
@doc "`SMD::SolvationModel`: the SMD solvation model." SMD

"""
    PCMSurface

The cavity surface: the van der Waals surface `VdwSurface`, the solvent-accessible surface
`SASSurface`, or the SMD Coulomb cavity `SMDSurface`.
"""
@enum PCMSurface VdwSurface SASSurface SMDSurface

@doc "`VdwSurface::PCMSurface`: the van der Waals surface." VdwSurface
@doc "`SASSurface::PCMSurface`: the solvent-accessible surface." SASSurface
@doc "`SMDSurface::PCMSurface`: the SMD Coulomb cavity." SMDSurface

"""
    VdwKernel

The van der Waals correction: `D2`, `D3`, `D3BJ`, `B97X_D_DISP` (the dispersion of ωB97X-D) or
`VV10`.
"""
@enum VdwKernel D2 D3 D3BJ B97X_D_DISP VV10

@doc "`D2::VdwKernel`: Grimme's D2 dispersion correction." D2
@doc "`D3::VdwKernel`: Grimme's D3 dispersion correction." D3
@doc "`D3BJ::VdwKernel`: Grimme's D3 with Becke–Johnson damping." D3BJ
@doc "`B97X_D_DISP::VdwKernel`: the dispersion correction of ωB97X-D." B97X_D_DISP
@doc "`VV10::VdwKernel`: the VV10 nonlocal correlation." VV10

"""
    TDDFTMethod

The excited-state equations: the Tamm–Dancoff approximation `TDA`, the full linear response
`RPA_TD`, or `SpinFlip`.
"""
@enum TDDFTMethod TDA RPA_TD SpinFlip

@doc "`TDA::TDDFTMethod`: the Tamm–Dancoff approximation." TDA
@doc "`RPA_TD::TDDFTMethod`: the full linear response." RPA_TD
@doc "`SpinFlip::TDDFTMethod`: spin-flip TDDFT." SpinFlip

"""
    IRCDirection

The direction an IRC follows from the transition state: `Forward`, `Backward` or `Both`.
"""
@enum IRCDirection Forward Backward Both

@doc "`Forward::IRCDirection`: forward from the transition state." Forward
@doc "`Backward::IRCDirection`: backward from the transition state." Backward
@doc "`Both::IRCDirection`: both directions from the transition state." Both

"""
    PopulationKind

A population analysis: `Mulliken` or `Lowdin`.
"""
@enum PopulationKind Mulliken Lowdin

@doc "`Mulliken::PopulationKind`: the Mulliken population analysis." Mulliken
@doc "`Lowdin::PopulationKind`: the Löwdin population analysis." Lowdin

"""
    ConstraintKind

An optimization constraint: `FreezeAtom`, `FreezeBond`, `FreezeAngle` or `FreezeDihedral`.
"""
@enum ConstraintKind FreezeAtom FreezeBond FreezeAngle FreezeDihedral

@doc "`FreezeAtom::ConstraintKind`: the position of one atom held fixed." FreezeAtom
@doc "`FreezeBond::ConstraintKind`: a bond length held fixed." FreezeBond
@doc "`FreezeAngle::ConstraintKind`: a bond angle held fixed." FreezeAngle
@doc "`FreezeDihedral::ConstraintKind`: a dihedral angle held fixed." FreezeDihedral

"""
    InputFormat

The format of an input file: `INP`, `JSON_INPUT`, `YAML_INPUT`, or the molecule files `XYZ`, `PDB`
and `CIF`.
"""
@enum InputFormat INP JSON_INPUT YAML_INPUT XYZ PDB CIF

@doc "`INP::InputFormat`: the `.inp` input format." INP
@doc "`JSON_INPUT::InputFormat`: a JSON input file." JSON_INPUT
@doc "`YAML_INPUT::InputFormat`: a YAML input file." YAML_INPUT
@doc "`XYZ::InputFormat`: an xyz molecule file." XYZ
@doc "`PDB::InputFormat`: a PDB molecule file." PDB
@doc "`CIF::InputFormat`: a CIF molecule file." CIF

"""
    MethodSpec(kind::MethodKind, reference::Reference, ints::NTuple{2,Int}, exx::Float64)

A method with its parameters. `ints` holds the active electrons and orbitals of CASSCF, the
exchange and correlation Libxc IDs of `LIBXC_GGA` and `LIBXC_MGGA`, or the one Libxc ID of
`LIBXC_HGGA` and `LIBXC_HMGGA` in `ints[1]`; `exx` is the exact-exchange fraction of the last two.
Unused slots are 0.
"""
struct MethodSpec
    kind::MethodKind
    reference::Reference
    ints::NTuple{2, Int}
    exx::Float64
end

"""
    Constraint(kind::ConstraintKind, atoms::NTuple{4,Int}, target::Union{Nothing,Float64})

An optimization constraint on the 1-based atoms `atoms` (unused slots 0): one atom for
`FreezeAtom`, two for `FreezeBond`, three for `FreezeAngle` (the vertex second) and four for
`FreezeDihedral`. `target` is the value to hold, in bohr for a bond and in radians for an angle or
a dihedral, or `nothing` to hold the starting value.
"""
struct Constraint
    kind::ConstraintKind
    atoms::NTuple{4, Int}
    target::Union{Nothing, Float64}
end

"""
    HubbardU(symbol::String, U::Float64)

A Hubbard parameter `U` for the atoms of element `symbol`.
"""
struct HubbardU
    symbol::String
    U::Float64
end

"""
    ONIOMSpec(high, low, model_atoms, model_charge, model_multiplicity)

A two-layer ONIOM calculation: the `high` and `low` methods, the 1-based indices of the
`model_atoms`, and the charge and multiplicity of the model system (`nothing` for those of the
whole system).
"""
struct ONIOMSpec
    high::MethodSpec
    low::MethodSpec
    model_atoms::Vector{Int}
    model_charge::Union{Nothing, Int}
    model_multiplicity::Union{Nothing, Int}
end

"""
    SolvationSpec(; model, solvent, epsilon, custom_epsilon, radii_scaling, grid_order, surface, probe_radius)

The continuum solvation settings: the `model` (default `NoSolvation`), the named `solvent`
(default `:water`), the PCM dielectric constant `epsilon` (78.39), the scale of the atomic radii
`radii_scaling` (1.2), the Lebedev order of each atomic sphere `grid_order` (110), the cavity
`surface` (`VdwSurface`) and the SMD probe radius `probe_radius` in bohr (0.40 Å). Under SMD,
`custom_epsilon == true` means that `epsilon` was given and replaces the dielectric constant of
`solvent`.
"""
Base.@kwdef struct SolvationSpec
    model::SolvationModel = NoSolvation
    solvent::Symbol = :water
    epsilon::Float64 = 78.39
    custom_epsilon::Bool = false
    radii_scaling::Float64 = 1.2
    grid_order::Int = 110
    surface::PCMSurface = VdwSurface
    probe_radius::Float64 = 0.4 * ANGSTROM_TO_BOHR
end

"""
    InputFile(; kwargs...)

Every setting of an input file, each keyword argument a field with the oracle's default. Atom
positions are in bohr, and every atom index (`constraints`, `fragments`, the ONIOM model atoms) is
1-based. `lattice` holds the cell vectors as rows, in Å whatever the geometry unit; `freeze_bond`
targets are in bohr and angle targets in radians. `basis` holds the basis-set name in lowercase;
alias resolution belongs to the basis-set reader. `dos_sigma` is in eV.
"""
Base.@kwdef struct InputFile
    atoms::Vector{Atom{Float64}} = Atom{Float64}[]
    product_atoms::Vector{Atom{Float64}} = Atom{Float64}[]
    point_charges::Vector{PointCharge{Float64}} = PointCharge{Float64}[]
    title::String = ""
    run_mode::RunMode = SinglePoint
    charge::Int = 0
    multiplicity::Int = 1
    basis::String = "sto-3g"
    method::MethodSpec = MethodSpec(HF, AutoReference, (0, 0), 0.0)
    scf_convergence::Float64 = 1.0e-6
    max_iterations::Int = 50
    diis_method::DIISMethod = CDIIS
    conv_tol_density::Union{Nothing, Float64} = nothing
    diis_reset_threshold::Float64 = 0.1
    damping::Float64 = 0.0
    smearing::Float64 = 0.0
    grid_level::Int = 1
    level_shift::Float64 = 0.0
    auto_stabilize::Bool = true
    robust_convergence::Bool = false
    electric_field::Union{Nothing, SVector{3, Float64}} = nothing
    use_ri_j::Bool = false
    use_ri_k::Bool = false
    aux_basis::Union{Nothing, String} = nothing
    compute_mode::ComputeMode = CPU
    checkpoint::Union{Nothing, String} = nothing
    load_checkpoint::Bool = false
    ts_target_mode::Int = 0
    ts_trust_radius::Float64 = 0.1
    neb_k::Float64 = 0.1
    neb_images::Int = 7
    climbing_image::Bool = true
    temperature::Float64 = 298.15
    pressure::Float64 = 1.0
    symmetry_number::Float64 = 1.0
    dispersion::Bool = false
    zora::Bool = false
    use_f12::Bool = false
    f12_gamma::Float64 = 1.0
    f12_model::String = "F12b"
    nbo::Bool = false
    admm::Bool = false
    admm_basis::Union{Nothing, String} = nothing
    use_ecp::Bool = false
    ecp_name::Union{Nothing, String} = nothing
    vdw_kernel::Union{Nothing, VdwKernel} = nothing
    solvation::SolvationSpec = SolvationSpec()
    output::Union{Nothing, String} = nothing
    verbosity::Int = 1
    export_molden::Bool = false
    export_cube::Bool = false
    export_wfx::Bool = false
    population::Vector{PopulationKind} = [Mulliken, Lowdin]
    pw_cutoff::Union{Nothing, Float64} = nothing
    kpoints::NTuple{3, Int} = (1, 1, 1)
    dos_k_grid::NTuple{3, Int} = (10, 10, 10)
    dos_sigma::Float64 = 0.1
    kpath::Vector{String} = String[]
    lattice::Union{Nothing, SMatrix{3, 3, Float64, 9}} = nothing
    pbc::NTuple{3, Bool} = (true, true, true)
    oniom::Union{Nothing, ONIOMSpec} = nothing
    constraints::Vector{Constraint} = Constraint[]
    irc_step_size::Float64 = 0.1
    irc_max_points::Int = 30
    irc_direction::IRCDirection = Both
    tddft_states::Int = 10
    tddft_method::TDDFTMethod = TDA
    compute_soc::Bool = false
    tddft_triplets::Bool = false
    use_dmft::Bool = false
    dmft_temperature::Float64 = 300.0
    dmft_n_freq::Int = 1024
    dmft_n_bath::Int = 4
    dmft_u::Vector{HubbardU} = HubbardU[]
    dmft_j::Vector{HubbardU} = HubbardU[]
    hubbard::Vector{HubbardU} = HubbardU[]
    fragments::Vector{Vector{Int}} = Vector{Int}[]
end

"""
    use_fock_damping(inp::InputFile) -> Bool

Whether the SCF damps the Fock matrix: `inp.damping > 1e-10`.
"""
use_fock_damping(inp::InputFile) = inp.damping > 1.0e-10

"""
    ParsedInput(input::InputFile, warnings::Vector{InputWarning})

A parsed input file and the warnings lenient reading recorded for it (empty in strict mode).
"""
struct ParsedInput
    input::InputFile
    warnings::Vector{InputWarning}
end

"""
    molecule(inp::InputFile) -> Molecule{Float64, Vector{Atom{Float64}}}

The molecule of `inp`: its atoms, charge, multiplicity and point charges. The atom and
point-charge vectors are those of `inp`, not copies.
"""
molecule(inp::InputFile) =
    Molecule(inp.atoms; charge = inp.charge, multiplicity = inp.multiplicity, point_charges = inp.point_charges)

# A mutable copy of every field of `InputFile`, which a parser fills one keyword at a time and
# turns into an `InputFile` once, with `_input`. `_draft(inp)` copies the fields of `inp`, sharing
# its vectors. The definitions are generated from the field list, so a field added to `InputFile`
# is carried without an edit here.
let fields = fieldnames(InputFile), decls = Expr[], copies = Expr[], kwargs = Expr[]
    for f in fields
        push!(decls, :($f::$(fieldtype(InputFile, f))))
        push!(copies, :(inp.$f))
        push!(kwargs, Expr(:kw, f, f === :oniom ? :oniom : :(d.$f)))
    end
    @eval mutable struct _InputDraft
        $(decls...)
    end
    @eval _draft(inp::InputFile) = _InputDraft($(copies...))
    # `oniom` is passed apart from the draft with its concrete type, so the keyword body of
    # `InputFile` is compiled for `Nothing` and for `ONIOMSpec` there, never for the union.
    @eval _build_input(d::_InputDraft, oniom::Union{Nothing, ONIOMSpec}) = InputFile(; $(kwargs...))
end

_input(d::_InputDraft, oniom::Union{Nothing, ONIOMSpec}) =
    isnothing(oniom) ? _build_input(d, nothing) : _build_input(d, oniom)

# The method word that asks for the restricted reference of `kind`.
function _restricted_word(kind::MethodKind)
    kind == HF && return "rhf"
    kind == MP2 && return "rmp2"
    kind == MP2_F12 && return "rmp2-f12"
    kind == CCSD && return "rccsd"
    kind == CCSD_F12 && return "rccsd-f12"
    return "restricted"
end

"""
    _finalize_input(inp::InputFile; source = "<input>", lenient = false, warnings = InputWarning[], method_line = 0, multiplicity_line = 0) -> InputFile

The checks every input format shares, run once after the parse of `inp` from `source`: the input
has atoms; a method with a restricted reference has multiplicity 1 (in lenient mode the reference
becomes `AutoReference`, as the oracle chooses it, with a `:rhf_open_shell` warning); and the
charge and multiplicity give a whole number of α and β electrons. The errors name the line of
`method` and of `multiplicity` (0 for a structured input or a keyword not given); an
`ArgumentError` of the electron counts is rethrown as an `InputError` with its message. Every
parser passes `source`, `lenient` and `warnings`; the defaults give the method a call without
keywords, which the type-stability check of the module judges.
"""
function _finalize_input(
        inp::InputFile; source::AbstractString = "<input>", lenient::Bool = false,
        warnings::Vector{InputWarning} = InputWarning[], method_line::Int = 0, multiplicity_line::Int = 0,
    )
    isempty(inp.atoms) && throw(InputError(source, 0, "no atoms: the input has no geometry block"))
    m = inp.method
    if m.reference == Restricted && inp.multiplicity != 1
        reject_or_warn(
            source, method_line, lenient, warnings, :rhf_open_shell,
            lazy"method `$(_restricted_word(m.kind))` requests a restricted reference, but multiplicity is $(inp.multiplicity)",
        )
        d = _draft(inp)
        d.method = MethodSpec(m.kind, AutoReference, m.ints, m.exx)
        inp = _input(d, inp.oniom)
    end
    try
        nalpha(molecule(inp))
    catch e
        e isa ArgumentError || rethrow()
        msg = e.msg
        msg isa LazyString && throw(InputError(source, multiplicity_line, msg))
        msg isa String && throw(InputError(source, multiplicity_line, msg))
        rethrow()
    end
    return inp
end
