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
QuantumEngine.Input.Reference
QuantumEngine.Input.MethodKind
QuantumEngine.Input.ComputeMode
QuantumEngine.Input.DIISMethod
QuantumEngine.Input.SolvationModel
QuantumEngine.Input.PCMSurface
QuantumEngine.Input.VdwKernel
QuantumEngine.Input.TDDFTMethod
QuantumEngine.Input.IRCDirection
QuantumEngine.Input.PopulationKind
QuantumEngine.Input.ConstraintKind
QuantumEngine.Input.InputFormat
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
```

## Number formats

```@docs
QuantumEngine.IO.fmt_fixed
QuantumEngine.IO.fmt_exp
QuantumEngine.IO.fmt_int
QuantumEngine.IO.fmt_str
```

## Logging

```@docs
QuantumEngine.IO.Logger
QuantumEngine.IO.Verbosity
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
