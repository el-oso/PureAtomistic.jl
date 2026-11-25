# The calls whose compiled specializations the StrictMode gates judge. `check.jl` includes this file
# in a child process that loads only the package, so no test fixture adds a specialization of its
# own; the items "test_calls sweep finds no unresolved call" and "every exemption is used and
# reasoned" include it as well.
using QuantumEngine

# Every error type and `InputWarning`, each built from a `String` and from a `LazyString` message,
# rendered into an `IOBuffer` (`showerror` for the errors, `show` for the warning); then the
# element lookups and radius accessors of `Units`; then the atom and molecule constructors, the
# electron counts and the nuclear repulsion of `System` in Float64 and Float32.
function run_workload()
    io = IOBuffer()
    sub = SubString("ab", 1, 1)
    for msg in ("m", lazy"m $(1) $(:s) $(2.5) $('c') $(true) $(sub)")
        for e in (
                InputError("a.inp", 3, msg), InputError("a.inp", 0, msg), BasisError(msg),
                SCFNotConverged(msg), OptimizationFailed(msg), NotImplemented("01 §Errors", msg),
                FunctionalUnavailable(msg), NumericalError(msg),
            )
            showerror(io, e)
        end
        show(io, InputWarning("a.inp", 3, :case, msg))
        show(io, InputWarning("a.inp", 0, :case, msg))
    end
    # The element lookups by symbol (String and SubString) and by Z, and the radius accessors.
    Units = QuantumEngine.Units
    for e in (element("Fe"), element("fe"), element(SubString("xCu", 2, 3)), element(26), element(Int32(6)))
        Units.covalent_radius(e)
        Units.bragg_slater_radius(e)
    end
    Units.bondi_radius(element("C"))
    Units.nelements()
    # Real atoms and each ghost spelling, built from Float64 and Float32 positions, and a molecule
    # of each with the default keywords and with a charge and multiplicity. The symbols are
    # `SubString`s, as a reader passes its tokens: their conversion to `String` is the allocation
    # the exemption of `Atom` names.
    sym(s) = SubString(s, 1, ncodeunits(s))
    for T in (Float64, Float32)
        atoms = [
            Atom(sym("O"), T[0, 0, 0.2]), Atom(sym("H"), T[0, 1.4, -0.9]),
            Atom(sym("H"), T[0, -1.4, -0.9]; mass = 2), Atom(sym("O@"), T[0, 0, 3]),
            Atom(sym("BqH"), T[0, 1, 3]), Atom(sym("Bq-H"), T[0, -1, 3]),
        ]
        foreach(basis_element, atoms)
        for mol in (Molecule(atoms), Molecule(atoms; charge = 1, multiplicity = 2))
            nuclear_repulsion(mol)
            nelectrons(mol)
            nalpha(mol)
            nbeta(mol)
        end
    end
    return take!(io)
end
