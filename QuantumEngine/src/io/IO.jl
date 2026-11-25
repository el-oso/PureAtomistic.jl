"""
    QuantumEngine.IO

Molecule files: the xyz reader (every frame) and writer, and the PDB and CIF readers. Each reader
reads in strict mode by default, where every input the oracle accepts silently but that is not
valid is an `InputError`, or in lenient mode, where it takes the oracle's value and records an
`InputWarning`. Positions are in bohr; the files hold Å.
"""
module IO

using StaticArrays: SVector
using StrictMode: @strict_exempt, @strict_policy
using ..Errors: InputError, InputWarning
using ..Units: ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, element
using ..System: System, Atom

include("lines.jl")
include("xyz.jl")
include("pdb.jl")
include("cif.jl")

export XYZFrame, read_xyz, write_xyz, read_pdb, read_cif
public input_line, reject_or_warn

@strict_exempt :noalloc read_xyz reason = "entry point: xyz reader"
@strict_exempt :noalloc write_xyz reason = "entry point: xyz writer"
@strict_exempt :noalloc read_pdb reason = "entry point: PDB reader"
@strict_exempt :noalloc read_cif reason = "entry point: CIF reader"
@strict_exempt :noalloc input_line reason = "entry point helper: line reader of read_xyz, read_pdb, read_cif and parse_inp"
@strict_exempt :noalloc reject_or_warn reason = "entry point helper: strict-or-lenient choice of read_xyz, read_pdb, read_cif, parse_inp and parse_structured"
@strict_exempt :noalloc _atom reason = "entry point helper: atom builder of read_xyz, read_pdb and read_cif"
@strict_exempt :noalloc _xyz_coordinate reason = "entry point helper: coordinate parser of read_xyz"
@strict_exempt :noalloc _pdb_coordinate reason = "entry point helper: coordinate parser of read_pdb"
@strict_exempt :noalloc _cif_coordinate reason = "entry point helper: coordinate parser of read_cif"

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
