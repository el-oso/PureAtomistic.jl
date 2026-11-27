# The vendored oracle input files read through `parse_input`, each compared field by field with an
# `InputFile` built from the values the oracle's `.inp` parser (`src/input/parser.rs`) gives them:
# the fields listed in each item, every other field at its default.
@testmodule FixtureFiles begin
    using QuantumEngine
    using QuantumEngine.Input
    using QuantumEngine.Units: ANGSTROM_TO_BOHR

    # parser.rs:583 converts Å coordinates to bohr by multiplying each by ANGSTROM_TO_BOHR.
    ang(symbol, x, y, z) = Atom(symbol, [x, y, z] .* ANGSTROM_TO_BOHR)

    # The fields of `a` and `b` that differ, for a readable failure next to `fields_equal`.
    differing(a::InputFile, b::InputFile) = [f for f in fieldnames(InputFile) if getfield(a, f) != getfield(b, f)]
end

@testitem "fixture: examples/h2.inp" setup = [Fixtures, FixtureFiles] begin
    using QuantumEngine
    using QuantumEngine.Input
    p = parse_input(Fixtures.oracle_input_path("examples/h2.inp"))
    @test p isa ParsedInput && isempty(p.warnings)
    # parser.rs:44-46: the first `#` line is the title; parser.rs:383-385 with parser.rs:1127: `rhf`;
    # parser.rs:270-279, 377-382, 539-548: charge, multiplicity, basis, convergence and iterations
    # as written, all equal to their defaults; parser.rs:307-317 with :581-584: a bohr geometry is
    # stored as written.
    expected = InputFile(;
        title = "Hydrogen molecule",
        method = MethodSpec(HF, Restricted, (0, 0), 0.0),
        atoms = [Atom("H", [0.0, 0.0, -0.7]), Atom("H", [0.0, 0.0, 0.7])],
    )
    @test FixtureFiles.differing(p.input, expected) == Symbol[]
    @test Fixtures.fields_equal(p.input, expected)
end

@testitem "fixture: examples/h2o_631g_sp.inp" setup = [Fixtures, FixtureFiles] begin
    using QuantumEngine
    using QuantumEngine.Input
    p = parse_input(Fixtures.oracle_input_path("examples/h2o_631g_sp.inp"))
    @test isempty(p.warnings)
    # parser.rs:44-46: the title; parser.rs:377-382: the basis lowercased, `631-g` kept as written
    # (the alias is resolved by the basis-set reader); parser.rs:1142: `pbe0`; parser.rs:539-548: the
    # convergence and the iteration count; parser.rs:307-317 with :583: an Å geometry.
    expected = InputFile(;
        title = "Water single-point PBE0/6-31G",
        basis = "631-g",
        method = MethodSpec(PBE0, AutoReference, (0, 0), 0.0),
        scf_convergence = 1.0e-10,
        max_iterations = 100,
        atoms = [
            FixtureFiles.ang("O", 0.0, 0.0, 0.1173),
            FixtureFiles.ang("H", 0.0, 0.7572, -0.4692),
            FixtureFiles.ang("H", 0.0, -0.7572, -0.4692),
        ],
    )
    @test FixtureFiles.differing(p.input, expected) == Symbol[]
    @test Fixtures.fields_equal(p.input, expected)
    @test Fixtures.fields_equal(molecule(p.input), Fixtures.water())
end

@testitem "fixture: examples/o2_triplet.inp" setup = [Fixtures, FixtureFiles] begin
    using QuantumEngine
    using QuantumEngine.Input
    path = Fixtures.oracle_input_path("examples/o2_triplet.inp")
    # `method rhf` (line 5) asks for a restricted reference at multiplicity 3.
    @test_throws "o2_triplet.inp:5: method `rhf` requests a restricted reference, but multiplicity is 3" parse_input(path)
    p = parse_input(path; lenient = true)
    @test [(w.case, w.line) for w in p.warnings] == [(:rhf_open_shell, 5)]
    # parser.rs:1127 reads `rhf` as the oracle's one Hartree-Fock method, whose reference the
    # multiplicity decides; parser.rs:275-279: multiplicity 3; parser.rs:44-46: the title.
    expected = InputFile(;
        title = "Oxygen molecule - triplet ground state (UHF)",
        multiplicity = 3,
        method = MethodSpec(HF, AutoReference, (0, 0), 0.0),
        atoms = [FixtureFiles.ang("O", 0.0, 0.0, 0.0), FixtureFiles.ang("O", 1.208, 0.0, 0.0)],
    )
    @test FixtureFiles.differing(p.input, expected) == Symbol[]
    @test Fixtures.fields_equal(p.input, expected)
    @test Fixtures.fields_equal(molecule(p.input), Fixtures.o2_triplet())
end

@testitem "fixture: examples/h2o_tddft.inp" setup = [Fixtures, FixtureFiles] begin
    using QuantumEngine
    using QuantumEngine.Input
    p = parse_input(Fixtures.oracle_input_path("examples/h2o_tddft.inp"))
    @test isempty(p.warnings)
    # parser.rs:1135: `lda`; parser.rs:61-62 with :70: `run tddft`; parser.rs:505-509:
    # `num_tddft_states 3`; parser.rs:44-46: the title; parser.rs:307-317 with :583: an Å geometry.
    expected = InputFile(;
        title = "Water TD-DFT excited states and NTO analysis",
        method = MethodSpec(LDA, AutoReference, (0, 0), 0.0),
        run_mode = TDDFT,
        tddft_states = 3,
        atoms = [
            FixtureFiles.ang("O", 0.0, 0.0, 0.0),
            FixtureFiles.ang("H", 0.757, 0.586, 0.0),
            FixtureFiles.ang("H", -0.757, 0.586, 0.0),
        ],
    )
    @test FixtureFiles.differing(p.input, expected) == Symbol[]
    @test Fixtures.fields_equal(p.input, expected)
end

@testitem "fixture: tests/h2o_dimer_bsse.inp" setup = [Fixtures, FixtureFiles] begin
    using QuantumEngine
    using QuantumEngine.Input
    p = parse_input(Fixtures.oracle_input_path("tests/h2o_dimer_bsse.inp"))
    @test isempty(p.warnings)
    # parser.rs:84-88: `title` joins the words after it; parser.rs:61-62 with :73: `run_mode bsse`;
    # parser.rs:1127: `rhf`; parser.rs:289-306: `fragments 0,1,2 ; 3,4,5` with 0-based indices,
    # stored 1-based; parser.rs:158-160: `nbo true`; parser.rs:307-317 with :583: an Å geometry.
    expected = InputFile(;
        title = "Water Dimer BSSE Test",
        run_mode = BSSE,
        method = MethodSpec(HF, Restricted, (0, 0), 0.0),
        fragments = [[1, 2, 3], [4, 5, 6]],
        nbo = true,
        atoms = [
            FixtureFiles.ang("O", 0.0, 0.0, 0.0),
            FixtureFiles.ang("H", 0.757, 0.586, 0.0),
            FixtureFiles.ang("H", -0.757, 0.586, 0.0),
            FixtureFiles.ang("O", 0.0, 0.0, 2.9),
            FixtureFiles.ang("H", 0.757, 0.586, 2.9),
            FixtureFiles.ang("H", -0.757, 0.586, 2.9),
        ],
    )
    @test length(p.input.atoms) == 6
    @test FixtureFiles.differing(p.input, expected) == Symbol[]
    @test Fixtures.fields_equal(p.input, expected)
end

@testitem "fixture: tests/o2_hubbard.in" setup = [Fixtures, FixtureFiles] begin
    using QuantumEngine
    using QuantumEngine.Input
    path = Fixtures.oracle_input_path("tests/o2_hubbard.in")
    # The geometry block (line 6) has no `end`; `hubbard` on line 10 ends it (parser.rs:280-285).
    @test_throws "o2_hubbard.in:10: `hubbard` inside the geometry block opened at line 6" parse_input(path)
    p = parse_input(path; lenient = true)
    # The cases of the item "lenient: o2_hubbard.in cases".
    @test [(w.case, w.line) for w in p.warnings] == [(:block_keyword, 10), (:unclosed_block, 10)]
    # parser.rs:84-88: the title; parser.rs:377-382: the basis lowercased; parser.rs:1136: `pbe`;
    # parser.rs:275-279: multiplicity 3; parser.rs:564-567: the hubbard line `O 4.0`;
    # parser.rs:307-317 with :583: an Å geometry.
    expected = InputFile(;
        title = "O2 Hubbard Test",
        basis = "6-31g",
        method = MethodSpec(PBE, AutoReference, (0, 0), 0.0),
        multiplicity = 3,
        hubbard = [HubbardU("O", 4.0)],
        atoms = [FixtureFiles.ang("O", 0.0, 0.0, 0.0), FixtureFiles.ang("O", 0.0, 0.0, 1.208)],
    )
    @test length(p.input.atoms) == 2
    @test FixtureFiles.differing(p.input, expected) == Symbol[]
    @test Fixtures.fields_equal(p.input, expected)
end

@testitem "dispatch: formats" setup = [Fixtures, FixtureFiles] begin
    using QuantumEngine
    using QuantumEngine.Input
    using QuantumEngine.Input: _yaml_unavailable
    # parser.rs:15-26: the extension, lowercased, picks the reader; anything else is `.inp`.
    for (path, fmt) in (
            ("a.xyz", XYZ), ("a.XYZ", XYZ), ("d/a.Xyz", XYZ), ("a.pdb", PDB), ("a.PDB", PDB), ("a.cif", CIF),
            ("a.CiF", CIF), ("a.yaml", YAML_INPUT), ("a.YAML", YAML_INPUT), ("a.yml", YAML_INPUT),
            ("a.YmL", YAML_INPUT), ("a.json", JSON_INPUT), ("a.JSON", JSON_INPUT), ("a.inp", INP),
            ("a.in", INP), ("a", INP), ("a.", INP), ("a.xyz.bak", INP), ("d.xyz/a", INP),
            (".xyz", INP), ("d/.json", INP), ("a..yaml", YAML_INPUT), ("..xyz", XYZ), (SubString("xa.pdb", 2), PDB),
            # The oracle lowercases `İ` (U+0130) to `i` and U+0307, and `K` (U+212A) to `k`.
            ("a.c\u0130f", INP), ("a.\u212ayz", INP),
        )
        @test input_format(path) === fmt
    end
    # A geometry file gives an input holding only its atoms (parser.rs:662-673).
    p = parse_input(Fixtures.oracle_input_path("examples/water.xyz"))
    @test isempty(p.warnings)
    expected = InputFile(;
        atoms = [
            FixtureFiles.ang("O", 0.0, 0.0, 0.0),
            FixtureFiles.ang("H", 0.757, 0.586, 0.0),
            FixtureFiles.ang("H", -0.757, 0.586, 0.0),
        ],
    )
    @test FixtureFiles.differing(p.input, expected) == Symbol[]
    @test Fixtures.fields_equal(p.input, expected)
    # PDB and CIF files, and an upper-case extension.
    dir = mktempdir()
    pdb = joinpath(dir, "w.PDB")
    write(
        pdb, """
        ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
        ATOM      2  H   HOH A   1       0.757   0.586   0.000  1.00  0.00           H
        ATOM      3  H   HOH A   1      -0.757   0.586   0.000  1.00  0.00           H
        """
    )
    @test Fixtures.fields_equal(parse_input(pdb).input, expected)
    cif = joinpath(dir, "w.cif")
    write(
        cif, """
        data_w
        loop_
        _atom_site_type_symbol
        _atom_site_Cartn_x
        _atom_site_Cartn_y
        _atom_site_Cartn_z
        O 0.000 0.000 0.000
        H 0.757 0.586 0.000
        H -0.757 0.586 0.000
        """
    )
    @test Fixtures.fields_equal(parse_input(cif).input, expected)
    # A JSON file goes to `parse_json`, with the path as its source.
    json = joinpath(dir, "w.json")
    write(json, """{"title": "w", "geometry": [{"symbol": "H", "position": [0.0, 0.0, 0.0]}], "multiplicity": 2}""")
    p = parse_input(json)
    @test (p.input.title, p.input.multiplicity, length(p.input.atoms)) == ("w", 2, 1)
    write(json, """{"geometry": []}""")
    @test_throws "input: $json: no atoms" parse_input(json)
    # Any other extension is read as `.inp`.
    inp = joinpath(dir, "w.txt")
    write(inp, Fixtures.oracle_input("examples/h2.inp"))
    @test Fixtures.fields_equal(parse_input(inp).input, parse_input(Fixtures.oracle_input_path("examples/h2.inp")).input)
    # A molecule file keeps the default charge and multiplicity whatever its electron count, as the
    # oracle does (parser.rs:670-672).
    xyz = joinpath(dir, "w.xyz")
    write(xyz, "1\nH atom\nH 0 0 0\n")
    p = parse_input(xyz)
    @test (length(p.input.atoms), p.input.charge, p.input.multiplicity) == (1, 0, 1)
    # Errors of the geometry readers name the path.
    write(xyz, "x\n")
    @test_throws "input: $xyz:1: xyz: expected the atom count" parse_input(xyz)
    # The oracle reads an xyz frame of no atoms (io/molecule.rs:39-51); strict reading rejects it.
    write(xyz, "0\nempty\n")
    @test_throws "input: $xyz: no atoms: the molecule file has no atoms" parse_input(xyz)
    p = parse_input(xyz; lenient = true)
    @test isempty(p.input.atoms)
    @test [(w.case, w.line) for w in p.warnings] == [(:xyz_no_atoms, 0)]
    @test Fixtures.fields_equal(p.input, InputFile())
    # The message of a YAML file read without YAML.jl loaded.
    e = _yaml_unavailable("a.yaml")
    @test e isa InputError
    @test sprint(showerror, e) == "input: a.yaml: reading YAML needs `using YAML` (QuantumEngineYAMLExt)"
end

@testitem "dispatch: yaml" setup = [Fixtures] begin
    using QuantumEngine
    # Each run is a child process, which loads YAML.jl (and with it the extension) or not, so that
    # loading it here does not change what the other items of this process see.
    dir = mktempdir()
    good = joinpath(dir, "w.yml")
    write(good, "multiplicity: 2\ngeometry:\n  - symbol: H\n    position: [0.0, 0.0, 0.0]\n")
    inp = joinpath(dir, "w.YAML")
    write(inp, "charge 0\n")
    code = """
    ARGS[1] == "yaml" && @eval using YAML
    using QuantumEngine
    for path in ARGS[2:end], lenient in (false, true)
        try
            p = parse_input(path; lenient)
            println("ok ", p.input.multiplicity, " ", length(p.input.atoms))
        catch e
            e isa InputError || rethrow()
            showerror(stdout, e)
            println()
        end
    end
    """
    env = ("JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing)
    run_child(mode) = split(
        read(addenv(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code $mode $good $inp`, env...), String),
        '\n'; keepempty = false,
    )
    # With YAML.jl loaded, a YAML file goes to `parse_yaml`, with the path as its source.
    out = run_child("yaml")
    @test length(out) == 4
    @test out[1] == "ok 2 1" && out[2] == "ok 2 1"
    @test startswith(out[3], "input: $inp: ") && occursin("looks like .inp syntax", out[3])
    @test startswith(out[4], "input: $inp: no atoms")
    # Without YAML.jl, a YAML file is an error in both modes.
    msg(path) = "input: $path: reading YAML needs `using YAML` (QuantumEngineYAMLExt)"
    @test run_child("none") == [msg(good), msg(good), msg(inp), msg(inp)]
end

@testitem "dispatch: xyz frames" setup = [Fixtures, FixtureFiles] begin
    using QuantumEngine
    using QuantumEngine.Input
    path = Fixtures.oracle_input_path("examples/legacy_inputs/neb_test.neb.xyz")
    @test_throws "xyz: the file has 5 frames; an input geometry needs exactly one" parse_input(path)
    try
        parse_input(path)
    catch e
        @test e isa InputError && e.line == 0
    end
    # The oracle reads the atom lines of the first frame and stops (io/molecule.rs:50-51).
    p = parse_input(path; lenient = true)
    @test [(w.case, w.line) for w in p.warnings] == [(:xyz_frames, 0)]
    expected = InputFile(;
        atoms = [
            FixtureFiles.ang("H", 0.0, 0.0, 0.0),
            FixtureFiles.ang("H", 0.0, 0.0, 0.39159114),
            FixtureFiles.ang("H", 0.0, 0.0, 1.44994556),
        ],
    )
    @test FixtureFiles.differing(p.input, expected) == Symbol[]
    @test Fixtures.fields_equal(p.input, expected)
end

@testitem "input that is not valid UTF-8" begin
    using QuantumEngine
    h2 = "geometry bohr\nH 0 0 -0.7\nH 0 0 0.7\nend\n"
    latin1 = "# caf\xe9\n" * h2
    title = "title a\xffb\n" * h2
    for lenient in (false, true)
        @test_throws "input: t.inp:1: the file is not valid UTF-8" parse_inp(latin1; source = "t.inp", lenient)
        @test_throws "input: t.inp:1: the file is not valid UTF-8" parse_inp(title; source = "t.inp", lenient)
        @test_throws "input: t.inp:3: the file is not valid UTF-8" parse_inp("geometry bohr\nH 0 0 -0.7\nH\xff 0 0 0.7\nend\n"; source = "t.inp", lenient)
    end
    mktempdir() do dir
        for (name, text) in (("a.inp", latin1), ("b.inp", title), ("c.json", "{\"title\": \"a\xffb\"}"), ("d.xyz", "1\n\xe9\nH 0 0 0\n"))
            path = joinpath(dir, name)
            write(path, text)
            line = name == "d.xyz" ? 2 : 1
            for lenient in (false, true)
                @test_throws "input: $(path):$(line): the file is not valid UTF-8" parse_input(path; lenient)
            end
        end
    end
end

@testitem "dispatch: a directory" begin
    using QuantumEngine
    # Opening a directory succeeds and can read as empty text, so the path is rejected before it is
    # read, whatever its extension names.
    dir = mktempdir()
    for name in ("sub", "sub.xyz", "sub.json"), lenient in (false, true)
        path = joinpath(dir, name)
        mkpath(path)
        @test_throws "input: $path: io: the path is a directory" parse_input(path; lenient)
        # With a trailing separator the path has no extension and is read as `.inp`.
        slash = path * "/"
        @test_throws "input: $slash: io: the path is a directory" parse_input(slash; lenient)
    end
end

# The settings the oracle echoes for each fixture file (references.toml [fixture] tables, from
# its captured standard output) against the parse of the same file. The oracle prints its own
# method name (engine/config.rs:252-275) and the convergence as Rust's `{:.1e}`
# (engine/mod.rs:88); the tests/h2o_dimer_bsse.inp run (run mode bsse) prints no `Atoms:` line
# (engine/electronic.rs:260-264).
@testitem "fixture: echoed settings match the oracle" setup = [Fixtures, OracleHarness] begin
    using QuantumEngine
    using QuantumEngine.Input
    using QuantumEngine.IO: fmt_exp
    oracle_name = Dict(HF => "RHF", PBE => "PBE", PBE0 => "PBE0", LDA => "LDA")
    tables = get(OracleHarness.references(), "fixture", Dict{String, Any}())
    @test sort(collect(keys(tables))) == sort(first.(OracleHarness.FIXTURE_FILES))
    without_atoms = String[]
    @testset "$rel" for (rel, lenient) in OracleHarness.FIXTURE_FILES
        # An absent table errors here and fails the key-set test above.
        t = get(tables, rel, Dict{String, Any}())
        inp = parse_input(Fixtures.oracle_input_path(rel); lenient).input
        @test t["exitcode"] == 0
        @test t["title"] == inp.title
        @test t["method"] == oracle_name[inp.method.kind]
        @test t["basis"] == inp.basis
        @test t["multiplicity"] == inp.multiplicity
        @test t["charge"] == inp.charge
        @test t["convergence"] == fmt_exp(inp.scf_convergence, 1)
        if haskey(t, "atoms")
            @test t["atoms"] == length(inp.atoms)
        else
            push!(without_atoms, rel)
        end
    end
    @test without_atoms == ["tests/h2o_dimer_bsse.inp"]
end
