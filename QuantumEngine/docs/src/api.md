# API

## Errors

```@docs
QuantumEngineError
InputError
BasisError
SCFNotConverged
OptimizationFailed
NotImplemented
FunctionalUnavailable
NumericalError
InputWarning
```

## Molden files

```@docs
QuantumEngine.IO.MoldenShell
QuantumEngine.IO.MoldenOrbitals
QuantumEngine.IO.MoldenData
QuantumEngine.IO.write_molden
QuantumEngine.IO.read_molden
```

## Input types

```@docs
QuantumEngine.Input.InputFile
QuantumEngine.Input.ParsedInput
QuantumEngine.Input.MethodSpec
QuantumEngine.Input.Constraint
QuantumEngine.Input.HubbardU
QuantumEngine.Input.ONIOMSpec
QuantumEngine.Input.SolvationSpec
QuantumEngine.Input.use_fock_damping
QuantumEngine.Input.RunMode
QuantumEngine.Input.SinglePoint
QuantumEngine.Input.Optimization
QuantumEngine.Input.TransitionState
QuantumEngine.Input.NEB
QuantumEngine.Input.Frequency
QuantumEngine.Input.TDDFT
QuantumEngine.Input.Spectroscopy
QuantumEngine.Input.OptFreq
QuantumEngine.Input.BSSE
QuantumEngine.Input.NBO
QuantumEngine.Input.Bands
QuantumEngine.Input.DOS
QuantumEngine.Input.NMR
QuantumEngine.Input.QTAIM
QuantumEngine.Input.IRC
QuantumEngine.Input.Reference
QuantumEngine.Input.AutoReference
QuantumEngine.Input.Restricted
QuantumEngine.Input.Unrestricted
QuantumEngine.Input.MethodKind
QuantumEngine.Input.HF
QuantumEngine.Input.MP2
QuantumEngine.Input.MP2_F12
QuantumEngine.Input.BWS2
QuantumEngine.Input.DLPNO_MP2
QuantumEngine.Input.CCSD
QuantumEngine.Input.CCSD_T
QuantumEngine.Input.CCSD_F12
QuantumEngine.Input.CCSD_T_F12
QuantumEngine.Input.LDA
QuantumEngine.Input.PBE
QuantumEngine.Input.BLYP
QuantumEngine.Input.SCAN
QuantumEngine.Input.TPSS
QuantumEngine.Input.M06L
QuantumEngine.Input.M062X
QuantumEngine.Input.PBE0
QuantumEngine.Input.B3LYP
QuantumEngine.Input.RPA
QuantumEngine.Input.B88_VDW
QuantumEngine.Input.PBE_VDW
QuantumEngine.Input.B97X_D
QuantumEngine.Input.WB97M_V
QuantumEngine.Input.B2PLYP
QuantumEngine.Input.MPW2PLYP
QuantumEngine.Input.DSD_PBEP86
QuantumEngine.Input.EOM_CCSD
QuantumEngine.Input.PWDFT
QuantumEngine.Input.CASSCF
QuantumEngine.Input.LIBXC_GGA
QuantumEngine.Input.LIBXC_MGGA
QuantumEngine.Input.LIBXC_HGGA
QuantumEngine.Input.LIBXC_HMGGA
QuantumEngine.Input.ComputeMode
QuantumEngine.Input.CPU
QuantumEngine.Input.GPU
QuantumEngine.Input.Hybrid
QuantumEngine.Input.DIISMethod
QuantumEngine.Input.CDIIS
QuantumEngine.Input.EDIIS
QuantumEngine.Input.SolvationModel
QuantumEngine.Input.NoSolvation
QuantumEngine.Input.PCM
QuantumEngine.Input.SMD
QuantumEngine.Input.PCMSurface
QuantumEngine.Input.VdwSurface
QuantumEngine.Input.SASSurface
QuantumEngine.Input.SMDSurface
QuantumEngine.Input.VdwKernel
QuantumEngine.Input.D2
QuantumEngine.Input.D3
QuantumEngine.Input.D3BJ
QuantumEngine.Input.B97X_D_DISP
QuantumEngine.Input.VV10
QuantumEngine.Input.TDDFTMethod
QuantumEngine.Input.TDA
QuantumEngine.Input.RPA_TD
QuantumEngine.Input.SpinFlip
QuantumEngine.Input.IRCDirection
QuantumEngine.Input.Forward
QuantumEngine.Input.Backward
QuantumEngine.Input.Both
QuantumEngine.Input.PopulationKind
QuantumEngine.Input.Mulliken
QuantumEngine.Input.Lowdin
QuantumEngine.Input.ConstraintKind
QuantumEngine.Input.FreezeAtom
QuantumEngine.Input.FreezeBond
QuantumEngine.Input.FreezeAngle
QuantumEngine.Input.FreezeDihedral
QuantumEngine.Input.InputFormat
QuantumEngine.Input.INP
QuantumEngine.Input.JSON_INPUT
QuantumEngine.Input.YAML_INPUT
QuantumEngine.Input.XYZ
QuantumEngine.Input.PDB
QuantumEngine.Input.CIF
```

## Input words

```@docs
QuantumEngine.Input.method_word
QuantumEngine.Input.run_mode_word
QuantumEngine.Input.solvation_word
QuantumEngine.Input.solvent_word
QuantumEngine.Input.vdw_kernel_word
QuantumEngine.Input.tddft_method_word
QuantumEngine.Input.grid_word
QuantumEngine.Input.irc_direction_word
QuantumEngine.Input.compute_mode_word
QuantumEngine.Input.diis_word
QuantumEngine.Input.population_word
QuantumEngine.Input.surface_word
QuantumEngine.Input.accepted_run_modes
QuantumEngine.Input.accepted_methods
QuantumEngine.Input.accepted_solvation_models
QuantumEngine.Input.accepted_solvents
QuantumEngine.Input.accepted_vdw_kernels
QuantumEngine.Input.accepted_tddft_methods
QuantumEngine.Input.accepted_grids
QuantumEngine.Input.accepted_irc_directions
QuantumEngine.Input.accepted_compute_modes
QuantumEngine.Input.accepted_diis_methods
QuantumEngine.Input.accepted_populations
QuantumEngine.Input.accepted_surfaces
```

## Modules

```@docs
QuantumEngine.QuantumEngine
QuantumEngine.Errors
QuantumEngine.Units
QuantumEngine.System
QuantumEngine.IO
QuantumEngine.Input
```

## Units and elements

```@docs
QuantumEngine.Units.Element
QuantumEngine.Units.element
QuantumEngine.Units.nelements
QuantumEngine.Units.covalent_radius
QuantumEngine.Units.bondi_radius
QuantumEngine.Units.bragg_slater_radius
QuantumEngine.Units.symbol_number
QuantumEngine.Units.ascii_lowercase
```

## Physical constants

```@docs
QuantumEngine.Units.BOHR_TO_ANGSTROM
QuantumEngine.Units.ANGSTROM_TO_BOHR
QuantumEngine.Units.HARTREE_TO_EV
QuantumEngine.Units.HARTREE_TO_KCAL
QuantumEngine.Units.HARTREE_TO_JOULE
QuantumEngine.Units.HARTREE_TO_CM1
QuantumEngine.Units.BOHR_TO_METER
QuantumEngine.Units.AMU_TO_KG
QuantumEngine.Units.ELEMENTARY_CHARGE
QuantumEngine.Units.BOLTZMANN_K
QuantumEngine.Units.PLANCK_H
QuantumEngine.Units.AVOGADRO_NA
QuantumEngine.Units.SPEED_OF_LIGHT
QuantumEngine.Units.GAS_CONSTANT_R
QuantumEngine.Units.AU_TO_DEBYE
```

## Atoms and molecules

```@docs
QuantumEngine.System.Atom
QuantumEngine.System.PointCharge
QuantumEngine.System.Molecule
QuantumEngine.System.element_number
QuantumEngine.System.basis_element
QuantumEngine.System.nelectrons
QuantumEngine.System.nalpha
QuantumEngine.System.nbeta
QuantumEngine.System.nuclear_repulsion
```

## Molecule files

```@docs
QuantumEngine.IO.XYZFrame
QuantumEngine.IO.read_xyz
QuantumEngine.IO.write_xyz
QuantumEngine.IO.read_pdb
QuantumEngine.IO.read_cif
```

## Reading lines

```@docs
QuantumEngine.IO.input_line
QuantumEngine.IO.reject_or_warn
QuantumEngine.IO.parse_coordinate
QuantumEngine.IO.is_path_separator
```

## Number formats

```@docs
QuantumEngine.IO.fmt_fixed
QuantumEngine.IO.fmt_exp
QuantumEngine.IO.fmt_int
QuantumEngine.IO.fmt_str
QuantumEngine.IO.FORMAT_PRECISIONS
```

## Logging

```@docs
QuantumEngine.IO.Logger
QuantumEngine.IO.Verbosity
QuantumEngine.IO.Silent
QuantumEngine.IO.Low
QuantumEngine.IO.Normal
QuantumEngine.IO.High
QuantumEngine.IO.verbosity_from_input
QuantumEngine.IO.log_print
QuantumEngine.IO.log_heading
QuantumEngine.IO.log_section
QuantumEngine.IO.log_success
QuantumEngine.IO.log_warning
QuantumEngine.IO.log_error
QuantumEngine.IO.print_atom_table
QuantumEngine.IO.print_total_energy
QuantumEngine.IO.report_optimization_step
QuantumEngine.IO.log_path
QuantumEngine.IO.output_base
```

## Structured-input node kinds

The tree type `Node` and its kind `NodeKind` are listed with the structured-input readers in
[Input files](input.md); the kinds are:

```@docs
QuantumEngine.Input.NullNode
QuantumEngine.Input.BoolNode
QuantumEngine.Input.IntNode
QuantumEngine.Input.FloatNode
QuantumEngine.Input.StringNode
QuantumEngine.Input.ArrayNode
QuantumEngine.Input.MapNode
```
