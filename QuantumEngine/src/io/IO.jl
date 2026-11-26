"""
    QuantumEngine.IO

Molecule files and output. The xyz reader (every frame) and writer, and the PDB and CIF readers:
each reader reads in strict mode by default, where every input the oracle accepts silently but
that is not valid is an `InputError`, or in lenient mode, where it takes the oracle's value and
records an `InputWarning`. Positions are in bohr; the files hold Å. The `Logger`, which writes
each line to standard output and to the log file, the oracle's fixed output lines, and the
Rust-compatible number formats they are built from.
"""
module IO

using StaticArrays: SVector
using Printf: @sprintf
using StrictMode: @strict_exempt, @strict_policy
using ..Errors: InputError, InputWarning
using ..Units: ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, element
using ..System: System, Atom

include("lines.jl")
include("xyz.jl")
include("pdb.jl")
include("cif.jl")
include("format.jl")
include("logger.jl")

export XYZFrame, read_xyz, write_xyz, read_pdb, read_cif
export Verbosity, Silent, Low, Normal, High, verbosity_from_input, Logger, log_print, log_success,
    log_warning, log_error, log_heading, log_section, print_atom_table, print_total_energy,
    report_optimization_step
public input_line, reject_or_warn, fmt_fixed, fmt_exp, fmt_int, fmt_str, log_path, output_base

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
@strict_exempt :noalloc _parse_coordinate reason = "entry point helper: coordinate token reader of read_xyz, read_pdb and read_cif"
@strict_exempt :noalloc fmt_fixed reason = "entry point: Rust-compatible formatter, builds a String"
@strict_exempt :noalloc fmt_exp reason = "entry point: Rust-compatible formatter, builds a String"
@strict_exempt :noalloc fmt_int reason = "entry point: Rust-compatible formatter, builds a String"
@strict_exempt :noalloc fmt_str reason = "entry point: Rust-compatible formatter, builds a String"
@strict_exempt :noalloc _pad_left reason = "entry point helper: padding of fmt_fixed, fmt_exp and fmt_int"
@strict_exempt :noalloc _pad_right reason = "entry point helper: padding of fmt_str"
@strict_exempt :noalloc types = (IOBuffer, AbstractString) _write_line reason = "entry point helper: line writer of log_print"
@strict_exempt :noalloc log_print reason = "entry point: logger output line"
@strict_exempt :noalloc log_success reason = "entry point: logger output line, builds a String"
@strict_exempt :noalloc log_warning reason = "entry point: logger output line, builds a String"
@strict_exempt :noalloc log_error reason = "entry point: logger output line, builds a String"
@strict_exempt :noalloc log_heading reason = "entry point: logger output lines, builds Strings"
@strict_exempt :noalloc log_section reason = "entry point: logger output lines, builds Strings"
@strict_exempt :noalloc print_atom_table reason = "entry point: logger output lines, builds Strings"
@strict_exempt :noalloc print_total_energy reason = "entry point: logger output line, builds a String"
@strict_exempt :noalloc report_optimization_step reason = "entry point: logger output line, builds a String"
@strict_exempt :noalloc types = (AbstractString, Nothing) log_path reason = "entry point: log file path, builds a String"
@strict_exempt :noalloc output_base reason = "entry point: output base path, builds a String"
@strict_exempt :noalloc _without_extension reason = "entry point helper: extension stripping of log_path and output_base"

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
