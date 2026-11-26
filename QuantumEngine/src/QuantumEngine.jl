"""
    QuantumEngine

A quantum-chemistry program in pure Julia. Its submodules are `Errors` (the error types),
`Units` (physical constants and the element table), `System` (atoms, molecules, electron
counts and the nuclear repulsion), `IO` (the xyz, PDB, CIF and Molden files, the logger and the
number formats) and `Input` (the typed input file, the words of its keywords, the `.inp` parser and
the JSON and dictionary readers; YAML files are read through the extension `QuantumEngineYAMLExt`).
"""
module QuantumEngine

using StrictMode: @strict_policy

include("errors/Errors.jl")
include("units/Units.jl")
include("system/System.jl")
include("io/IO.jl")
include("input/Input.jl")

using .Errors: QuantumEngineError, InputError, BasisError, SCFNotConverged, OptimizationFailed,
    NotImplemented, FunctionalUnavailable, NumericalError, InputWarning
using .Units: Element, element
using .System: Atom, PointCharge, Molecule, basis_element, nelectrons, nalpha, nbeta,
    nuclear_repulsion
using .IO: XYZFrame, read_xyz, write_xyz, read_pdb, read_cif
using .Input: InputFile, ParsedInput, parse_input, parse_inp, parse_json, parse_dict, parse_yaml, molecule

export QuantumEngineError, InputError, BasisError, SCFNotConverged, OptimizationFailed,
    NotImplemented, FunctionalUnavailable, NumericalError, InputWarning
export Element, element
export Atom, PointCharge, Molecule, basis_element, nelectrons, nalpha, nbeta, nuclear_repulsion
export XYZFrame, read_xyz, write_xyz, read_pdb, read_cif
export InputFile, ParsedInput, parse_input, parse_inp, parse_json, parse_dict, parse_yaml, molecule

public Units, System, IO, Input

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
