"""
    RunMode

What a run computes: `SinglePoint`, `Optimization`, `TransitionState`, `NEB`, `Frequency`,
`TDDFT`, `Spectroscopy`, `OptFreq`, `BSSE`, `NBO`, `Bands`, `DOS`, `NMR`, `QTAIM` or `IRC`.
"""
@enum RunMode SinglePoint Optimization TransitionState NEB Frequency TDDFT Spectroscopy OptFreq BSSE NBO Bands DOS NMR QTAIM IRC

"""
    Reference

The reference wave function a method asks for: `Restricted`, `Unrestricted`, or
`AutoReference`, which chooses restricted for a singlet and unrestricted otherwise.
"""
@enum Reference AutoReference Restricted Unrestricted

"""
    MethodKind

The electronic-structure method. `CCSD_T` is CCSD(T); `LIBXC_GGA`, `LIBXC_MGGA`, `LIBXC_HGGA` and
`LIBXC_HMGGA` are functionals named by their Libxc IDs.
"""
@enum MethodKind HF MP2 MP2_F12 BWS2 DLPNO_MP2 CCSD CCSD_T CCSD_F12 CCSD_T_F12 LDA PBE BLYP SCAN TPSS M06L M062X PBE0 B3LYP RPA B88_VDW PBE_VDW B97X_D WB97M_V B2PLYP MPW2PLYP DSD_PBEP86 EOM_CCSD PWDFT CASSCF LIBXC_GGA LIBXC_MGGA LIBXC_HGGA LIBXC_HMGGA

"""
    ComputeMode

Where a run computes: `CPU`, `GPU` or `Hybrid`.
"""
@enum ComputeMode CPU GPU Hybrid

"""
    DIISMethod

The SCF extrapolation: `CDIIS` or `EDIIS`.
"""
@enum DIISMethod CDIIS EDIIS

"""
    SolvationModel

The continuum solvation model: `NoSolvation`, `PCM` or `SMD`.
"""
@enum SolvationModel NoSolvation PCM SMD

"""
    PCMSurface

The cavity surface: the van der Waals surface `VdwSurface`, the solvent-accessible surface
`SASSurface`, or the SMD Coulomb cavity `SMDSurface`.
"""
@enum PCMSurface VdwSurface SASSurface SMDSurface

"""
    VdwKernel

The van der Waals correction: `D2`, `D3`, `D3BJ`, `B97X_D_DISP` (the dispersion of ωB97X-D) or
`VV10`.
"""
@enum VdwKernel D2 D3 D3BJ B97X_D_DISP VV10

"""
    TDDFTMethod

The excited-state equations: the Tamm–Dancoff approximation `TDA`, the full linear response
`RPA_TD`, or `SpinFlip`.
"""
@enum TDDFTMethod TDA RPA_TD SpinFlip

"""
    IRCDirection

The direction an IRC follows from the transition state: `Forward`, `Backward` or `Both`.
"""
@enum IRCDirection Forward Backward Both

"""
    PopulationKind

A population analysis: `Mulliken` or `Lowdin`.
"""
@enum PopulationKind Mulliken Lowdin

"""
    ConstraintKind

An optimization constraint: `FreezeAtom`, `FreezeBond`, `FreezeAngle` or `FreezeDihedral`.
"""
@enum ConstraintKind FreezeAtom FreezeBond FreezeAngle FreezeDihedral

"""
    InputFormat

The format of an input file: `INP`, `JSON_INPUT`, `YAML_INPUT`, or the molecule files `XYZ`, `PDB`
and `CIF`.
"""
@enum InputFormat INP JSON_INPUT YAML_INPUT XYZ PDB CIF

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
targets are in bohr and angle targets in radians. `basis` is lowercased as written; alias
resolution belongs to the basis-set reader. `dos_sigma` is in eV.
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
