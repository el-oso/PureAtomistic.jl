# The rendering of every error type and of `InputWarning` into an `IOBuffer`, which the juliac entry
# prints; the element lookups by symbol and by atomic number; and the atom and molecule
# constructors, the element-number lookup, the electron counts and the nuclear repulsion in Float64
# and Float32; and the xyz writer and the xyz, PDB and CIF readers, through their default and their
# full keyword calls, with the line helpers they share. The constructors are validated through the functions
# `new_atom` and `new_molecule`: the verifier builds no entry point for a call whose callee is a
# `UnionAll` type (`unresolved ccallable for Tuple{UnionAll, ...}`).
@testitem "TrimCheck: error rendering, element lookup, System and molecule files" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using QuantumEngine
            new_atom(symbol, position) = Atom(symbol, position)
            new_atom_mass(symbol, position, mass) = Atom(symbol, position; mass)
            new_molecule(atoms) = Molecule(atoms)
            read_xyz_kw(io, source, lenient, warnings) = read_xyz(io; source, lenient, warnings)
            read_pdb_kw(io, source, lenient, warnings) = read_pdb(io; source, lenient, warnings)
            read_cif_kw(io, source, lenient, warnings) = read_cif(io; source, lenient, warnings)
            write_xyz_kw(io, atoms, comment) = write_xyz(io, atoms; comment)
        end,
        Base.showerror(IOBuffer, InputError),
        Base.showerror(IOBuffer, BasisError),
        Base.showerror(IOBuffer, SCFNotConverged),
        Base.showerror(IOBuffer, OptimizationFailed),
        Base.showerror(IOBuffer, NotImplemented),
        Base.showerror(IOBuffer, FunctionalUnavailable),
        Base.showerror(IOBuffer, NumericalError),
        Base.show(IOBuffer, InputWarning),
        QuantumEngine.element(String),
        QuantumEngine.element(Int),
        new_atom(String, Vector{Float64}),
        new_atom(String, Vector{Float32}),
        new_atom(SubString{String}, Vector{Float64}),
        new_atom(SubString{String}, Vector{Float32}),
        new_atom_mass(String, Vector{Float64}, Float64),
        new_atom_mass(String, Vector{Float32}, Float32),
        new_molecule(Vector{Atom{Float64}}),
        new_molecule(Vector{Atom{Float32}}),
        QuantumEngine.System.element_number(String),
        QuantumEngine.System.element_number(SubString{String}),
        QuantumEngine.basis_element(Atom{Float64}),
        QuantumEngine.basis_element(Atom{Float32}),
        QuantumEngine.nuclear_repulsion(Molecule{Float64, Vector{Atom{Float64}}}),
        QuantumEngine.nuclear_repulsion(Molecule{Float32, Vector{Atom{Float32}}}),
        QuantumEngine.nelectrons(Molecule{Float64, Vector{Atom{Float64}}}),
        QuantumEngine.nelectrons(Molecule{Float32, Vector{Atom{Float32}}}),
        QuantumEngine.nalpha(Molecule{Float64, Vector{Atom{Float64}}}),
        QuantumEngine.nalpha(Molecule{Float32, Vector{Atom{Float32}}}),
        QuantumEngine.nbeta(Molecule{Float64, Vector{Atom{Float64}}}),
        QuantumEngine.nbeta(Molecule{Float32, Vector{Atom{Float32}}}),
        QuantumEngine.read_xyz(IOBuffer),
        QuantumEngine.read_pdb(IOBuffer),
        QuantumEngine.read_cif(IOBuffer),
        read_xyz_kw(IOBuffer, String, Bool, Vector{InputWarning}),
        read_pdb_kw(IOBuffer, String, Bool, Vector{InputWarning}),
        read_cif_kw(IOBuffer, String, Bool, Vector{InputWarning}),
        QuantumEngine.write_xyz(IOBuffer, Vector{Atom{Float64}}),
        QuantumEngine.write_xyz(IOBuffer, Vector{Atom{Float32}}),
        write_xyz_kw(IOBuffer, Vector{Atom{Float64}}, String),
        write_xyz_kw(IOBuffer, Vector{Atom{Float32}}, String),
        QuantumEngine.IO.input_line(String, Int),
        QuantumEngine.IO.reject_or_warn(String, Int, Bool, Vector{InputWarning}, Symbol, String),
    )
end

# The formatters, in Float64 and Float32, and the logger's line writer and two of its fixed output
# lines, on a logger that writes to an `IOBuffer` with no log file. `fmt_exp` with a width is
# validated through `fmt_exp_kw`.
@testitem "TrimCheck: formatters and logger" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using QuantumEngine
            fmt_exp_kw(x, prec, width) = QuantumEngine.IO.fmt_exp(x, prec; width)
        end,
        QuantumEngine.IO.fmt_fixed(Float64, Int, Int),
        QuantumEngine.IO.fmt_fixed(Float32, Int, Int),
        QuantumEngine.IO.fmt_exp(Float64, Int),
        QuantumEngine.IO.fmt_exp(Float32, Int),
        fmt_exp_kw(Float64, Int, Int),
        QuantumEngine.IO.print_total_energy(QuantumEngine.IO.Logger{IOBuffer, Base.DevNull}, Float64),
        QuantumEngine.IO.report_optimization_step(QuantumEngine.IO.Logger{IOBuffer, Base.DevNull}, Int, Float64, Float64),
        QuantumEngine.IO.log_print(QuantumEngine.IO.Logger{IOBuffer, Base.DevNull}, QuantumEngine.IO.Verbosity, String),
    )
end
