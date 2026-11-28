# The calls whose compiled specializations the StrictMode gates judge. `check.jl` includes this file
# in a child process that loads only the package and YAML.jl (which loads the extension
# `QuantumEngineYAMLExt`), so no test fixture adds a specialization of its
# own; the items "test_calls sweep finds no unresolved call" and "every exemption is used and
# reasoned" include it as well.
using QuantumEngine
using YAML
using StaticArrays: SVector

# Every error type and `InputWarning`, each built from a `String` and from a `LazyString` message,
# rendered into an `IOBuffer` (`showerror` for the errors, `show` for the warning); then the
# element lookups and radius accessors of `Units`; then the atom and molecule constructors, the
# electron counts and the nuclear repulsion of `System` in Float64 and Float32; then the xyz
# writer and the xyz, PDB and CIF readers of `IO`; then the formatters and the logger of `IO`; then
# the Molden writer and reader of `IO`; then the input types and the word lookups of `Input`; then
# the `.inp` parser; then the structured readers and the YAML extension; then `parse_input` on a
# file of each format; then the shell geometry and the basis types of `Basis` in Float64 and
# Float32; then the `.gbs` reader.
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
    # The Rust-compatible formatters; a logger built with the default and with a given verbosity,
    # writing to an `IOBuffer` with no log file (`devnull`), through each of its functions; the
    # verbosity of the input keyword; and the log and output paths.
    IOm = QuantumEngine.IO
    for x in (-76.301896228438, 2.5f-4)
        IOm.fmt_fixed(x, 20, 12)
        IOm.fmt_exp(x, 3)
        IOm.fmt_exp(x, 3; width = 8)
    end
    IOm.fmt_int(3, 3)
    IOm.fmt_str("O", 4)
    IOm.fmt_str(sym("O"), 4)
    for lg in (IOm.Logger(io, devnull), IOm.Logger(io, devnull; verbosity = IOm.High))
        IOm.log_print(lg, IOm.Normal, "line")
        IOm.log_success(lg, "ok")
        IOm.log_warning(lg, "w")
        IOm.log_error(lg, "e")
        IOm.log_heading(lg, "Final Results")
        IOm.log_section(lg, "Section")
        IOm.print_atom_table(lg, [Atom("O", [0, 0, 0.2]), Atom("H", [0, 1.4, -0.9])])
        IOm.print_atom_table(lg, [Atom("O", Float32[0, 0, 0.2])])
        IOm.print_total_energy(lg, -76.301896228438)
        IOm.report_optimization_step(lg, 3, -1.1167593075, 2.5e-4)
    end
    foreach(IOm.verbosity_from_input, (0, 1, 2))
    IOm.log_path("dir/h2.inp", nothing)
    IOm.log_path("h2.inp", "h2_test")
    IOm.output_base("dir/h2.log")
    # The Molden data types in Float64 and Float32, written with and without beta orbitals and read
    # back, through the default and the `source` keyword; then a reader error, rendered.
    for T in (Float64, Float32)
        atoms = [Atom("O", T[0, 0, 0.2]), Atom("H", T[0, 1.4, -0.9])]
        shells = [IOm.MoldenShell(1, 0, T[130.7, 23.8], T[0.15, 0.53]), IOm.MoldenShell(1, 1, T[5.0], T[1.0]), IOm.MoldenShell(2, 0, T[3.4], T[0.15])]
        alpha = IOm.MoldenOrbitals(T[-20.2, 0.3], T[2, 0], ones(T, 5, 2))
        for beta in (nothing, alpha)
            m = IOBuffer()
            IOm.write_molden(m, IOm.MoldenData("w", atoms, shells, alpha, beta))
            seekstart(m)
            IOm.read_molden(m)
            seekstart(m)
            IOm.read_molden(m; source = "w.molden")
        end
    end
    try
        IOm.read_molden(IOBuffer("[Molden Format]\n[Atoms] nm\n"))
    catch e
        e isa InputError || rethrow()
        showerror(io, e)
    end
    # The input types, built with their defaults and with keywords; each word lookup on a `String`
    # and on a `SubString` token, as the parsers pass it, and `method_word` on vectors and views of
    # each; and the word lists.
    In = QuantumEngine.Input
    m = In.MethodSpec(In.CASSCF, In.AutoReference, (4, 4), 0.0)
    sv = In.SolvationSpec(; model = In.SMD, solvent = :dcm, epsilon = 9.0, custom_epsilon = true)
    oniom = In.ONIOMSpec(m, In.method_word(["hf"]), [1, 2], nothing, 1)
    for inp in (
            In.InputFile(),
            In.InputFile(;
                title = "t", method = m, solvation = sv, oniom, damping = 0.2,
                constraints = [In.Constraint(In.FreezeBond, (1, 2, 0, 0), 2.0)], hubbard = [In.HubbardU("Fe", 4.0)],
            ),
        )
        In.use_fock_damping(inp)
        In.ParsedInput(inp, InputWarning[])
    end
    In.SolvationSpec()
    words = (
        In.run_mode_word, In.solvation_word, In.solvent_word, In.vdw_kernel_word, In.tddft_method_word,
        In.grid_word, In.irc_direction_word, In.compute_mode_word, In.diis_word, In.population_word,
        In.surface_word,
    )
    for w in ("Opt", sym("Di-chloro_methane"), "unknown")
        foreach(f -> f(w), words)
    end
    for toks in (["casscf(4,4)"], ["libxc_gga", "101", "130"], ["libxc_hgga", "402", "0.2"], ["UHF"], ["dft"])
        In.method_word(toks)
        In.method_word(view(toks, firstindex(toks):lastindex(toks)))
        st = sym.(toks)
        In.method_word(st)
        In.method_word(view(st, firstindex(st):lastindex(st)))
    end
    foreach(
        f -> f(), (
            In.accepted_run_modes, In.accepted_methods, In.accepted_solvation_models, In.accepted_solvents,
            In.accepted_vdw_kernels, In.accepted_tddft_methods, In.accepted_grids, In.accepted_irc_directions,
            In.accepted_compute_modes, In.accepted_diis_methods, In.accepted_populations, In.accepted_surfaces,
        )
    )
    # The `.inp` parser on an input with every keyword and block, through its default and its
    # keyword call and on a `SubString`; the molecule of the result; the keyword list; and a parser
    # error, rendered.
    inp_text = """
    # water
    title "Water" # inline comment
    charge 0
    multiplicity 1
    run freq
    basis 6-31G*
    method libxc_hgga 402 0.2
    scf_convergence 1e-8
    max_iterations 75
    diis_method ediis
    conv_tol_density 1e-7
    diis_reset 0.2
    damping 0.1
    smear 0.01
    grid fine
    level_shift 0.3
    autostabilize false
    robust true
    field 0 0 0.01
    ri_j true
    ri_k true
    aux_basis def2-SVP-JKFIT
    gpu false
    checkpoint run.chk
    restart true
    ts_target_mode 2
    ts_trust_radius 0.2
    neb_k 0.3
    neb_images 9
    climbing_image false
    temp 310
    press 2
    sigma 2
    dispersion true
    zora true
    f12 true
    f12_gamma 0.9
    f12_model F12a
    nbo true
    admm true
    admm_basis cfit
    ecp def2
    vdw_kernel d3bj
    solvation smd
    solvent dcm
    epsilon 9.0
    smd_grid 194
    output out.log
    verbosity 2
    export_molden true
    export_cube true
    export_wfx true
    population mulliken lowdin
    cutoff 400
    kpoints 2 2 2
    dos_k_grid 4 4 4
    dos_sigma 0.2
    kpath G X
    oniom_model_charge 0
    oniom_model_multiplicity 1
    freeze_atom 0
    freeze_bond 0 1 1.8
    freeze_angle 1 0 2 104.5
    freeze_dihedral 0 1 2 3
    irc_step 0.05
    irc_points 12
    irc_direction fwd
    nstates 5
    tddft_method rpa
    soc true
    triplets true
    dmft true
    dmft_temp 100
    dmft_n_freq 512
    dmft_n_bath 6
    geometry angstrom
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    He 0 0 3
    Bq 0 0 2 0.5
    end
    product_geometry bohr
    O 0 0 0
    end
    charges
    0.5 0 0 3
    end
    hubbard
    O 4.0
    end
    fragments
    0 ; 1,2
    end
    cell
    5 0 0
    0 5 0
    0 0 5
    end
    pbc
    true true false
    end
    oniom
    high pbe0
    low hf
    model_atoms 0
    end
    """
    for p in (parse_inp(inp_text), parse_inp(inp_text; source = "w.inp", lenient = false), parse_inp(SubString(inp_text, 1)))
        molecule(p.input)
    end
    parse_inp("title t\nsolvation pcm\nepsilon 4\npcm_radii 1.1\nfragments 0;1,2\ngeometry\nO 0 0 0\nH 0 0 1\nH 0 1 0\nend\n")
    # Lenient reading: fallbacks, missing values, the solvation read in order, and a keyword and a
    # block opener inside a block.
    lenient_text = """
    dispersion False
    damping abc
    grid 7
    field 0.1 x 0.3
    kpoints 2 x 2
    population foo
    tddft_method foo
    pcm_radii 1.1
    solvation pcm
    solvent
    epsilon
    pcm_grid
    method casscf(4)
    nbo
    output
    oniom_model_charge
    geometry
    Foo@ 0 0 0
    O 0 0 0
    charge 0
    H 0 0 1
    H 0 1 0
    pbc
    1 T F
    end
    hubbard
    O 4.0
    O 5.0
    """
    molecule(parse_inp(lenient_text; source = "w.inp", lenient = true).input)
    In.accepted_keywords()
    try
        parse_inp("charge 0\n")
    catch e
        e isa InputError || rethrow()
        showerror(io, e)
    end
    # Structured input: a JSON text with every section, read through the default and the keyword
    # calls; a tree built with each `Node` constructor and read strictly and, with every lenient
    # case, leniently; and a walker error, rendered.
    json_text = """
    {"title": "w", "charge": 0, "multiplicity": 1, "basis": "STO-3G", "method": "b3lyp", "mode": "opt",
     "dispersion": true, "zora": false, "hubbard": {"O": 4.0}, "vdw_kernel": "d3bj", "unit": "angstrom",
     "scf": {"max_iter": 75, "conv_energy": 1e-8, "use_ri_j": true, "use_ri_k": false, "aux_basis": "jk",
             "gpu": true, "diis_method": "ediis", "verbosity": 2, "checkpoint": "a.chk",
             "load_checkpoint": false, "damping": 0.1, "smearing": 0.0, "grid": 2, "auto_stabilize": true,
             "robust_convergence": false, "level_shift": 0.2, "field": [0.0, 0.0, 0.01], "pw_cutoff": 300.0,
             "lattice": [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]], "pbc": [true, true, false], "admm": true,
             "admm_basis": "cfit", "population": ["mulliken"], "kpoints": [2, 2, 2], "kpath": ["G", "X"],
             "use_dmft": true, "dmft_options": {"temperature": 500.0, "n_matsubara": 64, "n_bath": 2,
             "u_map": {"O": 4.0}, "j_map": {"O": 0.9}}},
     "solvation": {"model": "smd", "solvent": "water", "grid_order": 110, "probe_radius": 0.4},
     "oniom": {"high": "pbe0", "low": "hf", "model_atoms": [0], "model_charge": 0, "model_multiplicity": 1},
     "tddft": {"num_states": 3, "method": "tda", "soc": false, "triplets": true},
     "exports": {"molden": true, "cube": false, "wfx": true}, "f12": {"gamma": 1.1, "model": "F12a"},
     "f12_model": "F12b", "irc_step_size": 0.1, "irc_max_points": 20, "irc_direction": "both",
     "constraints": [{"freeze_atom": 0}, {"freeze_bond": [0, 1], "target": 1.8}, {"freeze_angle": [1, 0, 2], "target": 104.5}],
     "geometry": [{"symbol": "O", "position": [0.0, 0.0, 0.1173]}, {"symbol": "H", "x": 0.0, "y": 0.7572, "z": -0.4692},
                  {"symbol": "H", "position": [0.0, -0.7572, -0.4692]}]}
    """
    for p in (parse_json(json_text), parse_json(json_text; source = "w.json", lenient = false))
        molecule(p.input)
    end
    In.json_node(SubString(json_text, 1))
    N = In.Node
    atom(s, z) = N(["symbol", "position"], [N(s), N(N[N(0.0), N(0), N(z)])])
    tree = N(
        ["geometry", "unit", "title", "f12", "solvation", "scf", "mode", "constraints", "tddft", "method", "lattice"],
        [
            N(N[atom("H", -0.7), N(["symbol", "x", "y"], [N("H"), N(0.0), N("y")])]), N("nm"), N(1), N(1.5),
            N(["model", "surface", "epsilon", "radii_scaling"], [N("pcm"), N("blob"), N(4), N(nothing)]),
            N(["max_iter", "foo", "field", "kpoints", "damping"], [N(true), N(1), N(N[N(1.0)]), N(N[N(1), N(-0.5), N(1)]), N(NaN)]),
            N("zzz"), N(N[N(["freeze_bond"], [N(N[N("a")])]), N(["freeze_angle", "target"], [N(N[N(0), N(1), N(2), N(3)]), N(90.0)])]),
            N(["method"], [N("cis")]), N("casscf"), N(N[N(N[N(1.0), N(0.0)]), N(N[N(0.0), N(1.0), N(0.0)]), N(N[N(0.0), N(0.0), N(1.0)])]),
        ],
    )
    for lenient in (false, true)
        try
            In.parse_structured(tree; source = "w.json", lenient)
        catch e
            e isa InputError || rethrow()
            showerror(io, e)
        end
    end
    In.parse_structured(N(["geometry"], [N(N[atom("H", -0.7), atom("H", 0.7)])]); source = "w.json")
    # Structured input from a `Dict`: every value kind, through `parse_dict` in both modes and
    # `node` with and without a source; each `node` error rendered; and the error of a YAML input
    # read without YAML.jl.
    h2 = Any[Dict{String, Any}("symbol" => "H", "position" => Any[0.0, 0, Int32(-1)]), Dict("symbol" => "H", "x" => 0.0f0, "y" => 0.0, "z" => 0.7)]
    d = Dict{String, Any}(
        "title" => SubString("xw", 2), "charge" => Int8(0), "zora" => false, "unit" => "bohr", "f12_model" => nothing,
        "aux" => missing, "hubbard" => Dict("O" => 4.0), "geometry" => h2,
    )
    for lenient in (false, true)
        try
            parse_dict(d; source = "w.yaml", lenient)
        catch e
            e isa InputError || rethrow()
            showerror(io, e)
        end
    end
    parse_dict(Dict{String, Any}("unit" => "bohr", "geometry" => h2); source = "w.yaml")
    In.node(d)
    In.node(d; source = "w.yaml")
    for bad in (
            Dict{Any, Any}(1 => 2), Dict("a" => Dict(:k => 1)), Dict("a" => 'c'), Dict("a" => Int128(2)^70),
            IdDict{Any, Any}("a" => 1, SubString("ab", 1, 1) => 2),
        )
        try
            In.node(bad)
        catch e
            e isa InputError || rethrow()
            showerror(io, e)
        end
    end
    showerror(io, In._yaml_unavailable("w.yaml"))
    # YAML text through the extension, in both modes and from a `String` and a `SubString`: a
    # mapping, a document that is not a mapping (with and without the `.inp` hint, and after a byte
    # order mark), an empty document, a parser error, a scanner error and a merge key.
    yh2 = "unit: bohr\ngeometry:\n  - symbol: H\n    position: [0.0, 0.0, -0.7]\n  - symbol: H\n    position: [0.0, 0.0, 0.7]\n"
    parse_yaml(yh2)
    for text in (yh2, "charge 0\n", "\ufeffcharge 0\n", "- 1\n", "", "a: [1\n", "a: \"x\n", "b: &b {x: 1}\nc: {<<: *b}\n"), lenient in (false, true)
        for t in (text, SubString(text))
            try
                parse_yaml(t; source = "w.yaml", lenient)
            catch e
                e isa InputError || rethrow()
                showerror(io, e)
            end
        end
    end
    # Input files by extension: `.inp`, JSON, YAML and the molecule files, a multi-frame xyz in both
    # modes, an xyz frame of no atoms in both modes, a directory, and the format lookup on a
    # `SubString`.
    dir = mktempdir()
    files = (
        "w.inp" => "geometry bohr\nH 0 0 -0.7\nH 0 0 0.7\nend\n",
        "w.json" => json_text,
        "w.yaml" => yh2,
        "w.xyz" => "2\nh2\nH 0 0 0\nH 0 0 0.74\n",
        "f.xyz" => "1\na\nH 0 0 0\n1\nb\nH 0 0 1\n",
        "e.xyz" => "0\nempty\n",
        "w.pdb" => "ATOM      1  H   HOH A   1       0.000   0.000   0.000  1.00  0.00           H\n",
        "w.cif" => "data_w\nloop_\n_atom_site_type_symbol\n_atom_site_Cartn_x\n_atom_site_Cartn_y\n_atom_site_Cartn_z\nH 0 0 0\n",
    )
    for (name, text) in files
        path = joinpath(dir, name)
        write(path, text)
        for lenient in (false, true)
            try
                parse_input(path; lenient)
            catch e
                e isa InputError || rethrow()
                showerror(io, e)
            end
        end
        name in ("f.xyz", "e.xyz") || parse_input(path)
    end
    # A directory, which is rejected before it is read.
    sub = mkpath(joinpath(dir, "d.xyz"))
    for path in (sub, joinpath(dir, "d.xyz", "")), lenient in (false, true)
        try
            parse_input(path; lenient)
        catch e
            e isa InputError || rethrow()
            showerror(io, e)
        end
    end
    In.input_format(SubString("xw.JSON", 2))
    rm(dir; recursive = true)
    # The shell geometry helpers, each of the seven basis types built in Float64 and Float32, the AO
    # ranges, and the shell pairs iterated, counted and typed.
    B = QuantumEngine.Basis
    for T in (Float64, Float32)
        B.ncart(2)
        B.cartesian_component(3, 5)
        B.primitive_norm(T(0.8), 1, 1, 0)
        e, c = T[3.425250914, 0.6239137298], T[0.1543289673, 0.5353281423]
        B.contraction_renorm(e, c, 1)
        p = B.Primitive(T(0.8), T(0.5))
        sp = B.ShellSpec(1, [p], 3)
        ch = B.ECPChannel(0, [1], T[1.5], T[-2.0])
        B.ElementBasis("Na", [sp], B.ECPPotential("Na", 10, 0, [ch]))
        r = SVector{3, T}(0, 0, 0)
        shells = [B.Shell(r, 1, 0, e, c, 0), B.Shell(r, 1, 1, e, c, 1)]
        bs = B.BasisSet(shells, 4, "w")
        foreach(B.ao_range, bs.shells)
        pairs = B.shell_pairs(bs)
        length(pairs)
        eltype(pairs)
        foreach(identity, pairs)
    end
    # The `.gbs` reader on an `IOBuffer` (a file with S, SP and D shells and an ECP section) and on a
    # temporary file, through the default and the `source` keyword; then one input of each reader
    # error, rendered.
    gbs = "****\n! c\nH 0\nS 2 1.00\n 0.34D+01 0.15D+00\n 0.62 0.53\n****\nNA 0\nS 1 1.00\n 0.5 1.0\n" *
        "SP 1 1.00\n 0.5 0.3 0.7\nD 1 1.00\n 0.8 1.0\n****\nNA 0\nNA-ECP 1 10\np potential\n1\n" *
        "1 175.5 -10.0\ns-p potential\n1\n0 243.4 3.0\n"
    B.parse_gbs(IOBuffer(gbs))
    B.parse_gbs(IOBuffer(gbs); source = "w.gbs")
    gdir = mktempdir()
    gpath = joinpath(gdir, "w.gbs")
    write(gpath, gbs)
    B.read_element_basis(gpath, "Na")
    ecp_head = "NA 0\nNA-ECP 1 10\n"
    for bad in (
            "Xx 0\n", "H 0\nQ 1\n", "H 0\nS\n", "H 0\nS x\n", "H 0\nS 1 1.2\n", "H 0\nS 1\n 1.0\n",
            "H 0\nS 1\n x 1\n", "H 0\nS 2\n 1 1\n****\n", "H 0\nS 1\n -1 1\n", "H 0\nS 1\n 1 NaN\n",
            "C 0\nSP 1\n 1 1 Inf\n", "H 0\nS 1\n 1 0\n****\n", "S 3 1.00\n", "C 0\n****\n", "C 0\n",
            "C 0\nS 1\n 1 1\nN 0\n", "C 0\nS 1\n 1 1\n****\nC 0\nS 1\n 1 1\n****\n", "C 0\nP 2\n 1 1\n",
            "NA 0\nMG-ECP 1 10\n", "NA 0\nNA-ECP 7 10\n", ecp_head * "x\n", ecp_head * "p potential\nx\n",
            ecp_head * "p potential\n1\n1 1\n", ecp_head * "p potential\n1\n1.5 1 1\n",
            ecp_head * "p potential\n1\n1 x 1\n", ecp_head * "p potential\n2\n1 1 1\n****\n",
            ecp_head * "p potential\n2\n1 1 1\n", ecp_head * "p potential\n1\n1 1 1\n****\n",
            ecp_head * "p potential\n1\n1 1 1\n", ecp_head * "p potential\n", ecp_head * "p potential\n1\n1 1 1\n" * ecp_head,
        )
        try
            B.parse_gbs(IOBuffer(bad); source = "w.gbs")
        catch e
            e isa BasisError || rethrow()
            showerror(io, e)
        end
    end
    try
        B.read_element_basis(gpath, "Xe")
    catch e
        e isa BasisError || rethrow()
        showerror(io, e)
    end
    rm(gdir; recursive = true)
    return take!(io)
end
