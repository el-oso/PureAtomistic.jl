# The calls whose compiled specializations the StrictMode gates judge. `check.jl` includes this file
# in a child process that loads only the package, so no test fixture adds a specialization of its
# own; the items "test_calls sweep finds no unresolved call" and "every exemption is used and
# reasoned" include it as well.
using QuantumEngine

# Every error type and `InputWarning`, each built from a `String` and from a `LazyString` message,
# rendered into an `IOBuffer` (`showerror` for the errors, `show` for the warning); then the
# element lookups and radius accessors of `Units`; then the atom and molecule constructors, the
# electron counts and the nuclear repulsion of `System` in Float64 and Float32; then the xyz
# writer and the xyz, PDB and CIF readers of `IO`.
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
    # the exemption of `Atom` names. A real atom, one with a mass and a ghost are also built from
    # `String` symbols, whose keyword body allocates nothing and is held to `:noalloc`.
    sym(s) = SubString(s, 1, ncodeunits(s))
    for T in (Float64, Float32)
        atoms = [
            Atom(sym("O"), T[0, 0, 0.2]), Atom(sym("H"), T[0, 1.4, -0.9]),
            Atom(sym("H"), T[0, -1.4, -0.9]; mass = 2), Atom(sym("O@"), T[0, 0, 3]),
            Atom(sym("BqH"), T[0, 1, 3]), Atom(sym("Bq-H"), T[0, -1, 3]),
        ]
        foreach(basis_element, atoms)
        foreach(basis_element, (Atom("O", T[0, 0, 0]), Atom("H", T[0, 0, 1]; mass = 2), Atom("O@", T[0, 0, 2])))
        for mol in (Molecule(atoms), Molecule(atoms; charge = 1, multiplicity = 2))
            nuclear_repulsion(mol)
            nelectrons(mol)
            nalpha(mol)
            nbeta(mol)
        end
    end
    # The molecule files: an xyz frame written from Float64 and from Float32 atoms; an xyz, a PDB
    # and a CIF file read strictly and, with every lenient case, leniently; then the error of an
    # unknown element in strict mode, rendered.
    for T in (Float64, Float32)
        write_xyz(io, [Atom("O", T[0, 0, 0.2]), Atom("H", T[0, 1.4, -0.9])]; comment = "w")
    end
    pdb_record(name, el) = "ATOM      1 " * rpad(name, 4) * " WAT A   1       0.000   0.000   0.117  1.00  0.00" * " "^10 * el
    pdb = join([pdb_record(" O", " O"), pdb_record(" H", " H")], "\n")
    pdb_lenient = join(
        [
            first(pdb_record(" O", " O"), 40), first(pdb_record("CL", "CL"), 66), pdb_record(" X", "Xx"),
            pdb_record(" X", "  "), replace(pdb_record(" O", "O "), "0.117" => "  nan"),
        ], "\n"
    )
    cif = "data_t\nloop_\n_atom_site_type_symbol\n_atom_site_Cartn_x\n_atom_site_Cartn_y\n_atom_site_Cartn_z\n"
    xyz = "2\r\nw\r\nO\t0 0 0.2\r\nH 0 1.4 -0.9\r\n"
    read_xyz(IOBuffer("\ufeff" * xyz * xyz))
    read_pdb(IOBuffer(pdb))
    read_cif(IOBuffer(cif * "Fe2+ 0.0 0.0 0.0\nO 1.5 0 0\n"))
    w = InputWarning[]
    read_xyz(IOBuffer("3\n\nXx 0 0 0\nH inf 0 1\n"); source = "a.xyz", lenient = true, warnings = w)
    read_pdb(IOBuffer(pdb_lenient); source = "a.pdb", lenient = true, warnings = w)
    read_cif(IOBuffer(cif * "O 0 0\nO x 0 0\nXx 0 nan 0\n"); source = "a.cif", lenient = true, warnings = w)
    read_cif(
        IOBuffer(cif * "O 0 0 0\nloop_\n_atom_site_label\n_atom_site_Cartn_x\nO1 0\n"); source = "a.cif", lenient = true,
        warnings = w,
    )
    try
        read_xyz(IOBuffer("1\nw\nXx 0 0 0\n"))
    catch e
        e isa InputError || rethrow()
        showerror(io, e)
    end
    return take!(io)
end
