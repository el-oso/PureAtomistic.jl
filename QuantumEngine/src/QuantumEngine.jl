"""
    QuantumEngine

A quantum-chemistry program in pure Julia. Its submodules are `Errors` (the error types),
`Units` (physical constants and the element table), `System` (atoms, molecules, electron
counts and the nuclear repulsion) and `IO` (the xyz, PDB and CIF molecule files).
"""
module QuantumEngine

using StrictMode: @strict_policy

include("errors/Errors.jl")
include("units/Units.jl")
include("system/System.jl")
include("io/IO.jl")

using .Errors: QuantumEngineError, InputError, BasisError, SCFNotConverged, OptimizationFailed,
    NotImplemented, FunctionalUnavailable, NumericalError, InputWarning
using .Units: Element, element
using .System: Atom, PointCharge, Molecule, basis_element, nelectrons, nalpha, nbeta,
    nuclear_repulsion
using .IO: XYZFrame, read_xyz, write_xyz, read_pdb, read_cif

export QuantumEngineError, InputError, BasisError, SCFNotConverged, OptimizationFailed,
    NotImplemented, FunctionalUnavailable, NumericalError, InputWarning
export Element, element
export Atom, PointCharge, Molecule, basis_element, nelectrons, nalpha, nbeta, nuclear_repulsion
export XYZFrame, read_xyz, write_xyz, read_pdb, read_cif

public Units, System, IO

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
