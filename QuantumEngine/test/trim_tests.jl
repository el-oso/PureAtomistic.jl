# The rendering of every error type and of `InputWarning` into an `IOBuffer`, which the juliac entry
# prints; the element lookups by symbol and by atomic number; and the atom and molecule
# constructors, the electron counts and the nuclear repulsion in Float64 and Float32. The
# constructors are validated through the functions `new_atom` and `new_molecule`: the verifier builds
# no entry point for a call whose callee is a `UnionAll` type (`unresolved ccallable for
# Tuple{UnionAll, ...}`).
@testitem "TrimCheck: error rendering, element lookup and System" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using QuantumEngine
            new_atom(symbol, position) = Atom(symbol, position)
            new_atom_mass(symbol, position, mass) = Atom(symbol, position; mass)
            new_molecule(atoms) = Molecule(atoms)
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
    )
end
