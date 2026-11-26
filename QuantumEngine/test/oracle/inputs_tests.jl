# The vendored oracle inputs read through `parse_input` in both modes, each against its row of
# `inputs/EXPECTED.toml`. The rows cite the oracle's source for each outcome.

@testmodule OracleInputs begin
    using QuantumEngine
    using TOML

    const ROOT = joinpath(pkgdir(QuantumEngine), "test", "oracle", "inputs")
    const EXTENSIONS = (".inp", ".in", ".yaml", ".yml", ".json", ".xyz")

    # The vendored input files, as `/`-separated paths relative to `ROOT`.
    files() = sort(
        [
            join(splitpath(relpath(joinpath(dir, f), ROOT)), "/") for (dir, _, fs) in walkdir(ROOT) for
                f in fs if splitext(f)[2] in EXTENSIONS
        ]
    )

    rows() = TOML.parsefile(joinpath(ROOT, "EXPECTED.toml"))

    path(rel) = joinpath(ROOT, rel)

    cases(p) = sort([String(w.case) for w in p.warnings])
end

@testitem "every vendored input has an expected outcome" setup = [OracleInputs] begin
    using QuantumEngine
    using YAML
    files = OracleInputs.files()
    rows = OracleInputs.rows()
    @test length(files) == 72
    # A file without a row, and a row without a file.
    @test setdiff(files, keys(rows)) == String[]
    @test sort(collect(setdiff(keys(rows), files))) == String[]
    @testset "$rel" for rel in files
        haskey(rows, rel) || continue
        row = rows[rel]
        path = OracleInputs.path(rel)
        @test row["strict"] in ("ok", "error")
        @test row["lenient"] in ("ok", "error")
        if row["strict"] == "ok"
            @test !haskey(row, "strict_message")
            @test parse_input(path) isa ParsedInput
        else
            @test_throws row["strict_message"] parse_input(path)
        end
        if row["lenient"] == "ok"
            @test !haskey(row, "lenient_message")
            @test issorted(row["lenient_warnings"])
            @test OracleInputs.cases(parse_input(path; lenient = true)) == row["lenient_warnings"]
        else
            @test !haskey(row, "lenient_warnings")
            @test_throws row["lenient_message"] parse_input(path; lenient = true)
        end
    end
end

@testitem "vendored inputs: outcomes known now" setup = [OracleInputs] begin
    using QuantumEngine
    using QuantumEngine.Input
    rows = OracleInputs.rows()
    path = OracleInputs.path
    diis = [
        "examples/h2_ediis.inp", "examples/h2o.inp", "examples/h2o_test.inp", "examples/h2o_cdiis.inp",
        "examples/h2o_ediis.inp", "examples/h2o_nodiss.inp", "examples/legacy_inputs/water_ediis.inp",
    ]
    for rel in diis
        @test (rows[rel]["strict"], rows[rel]["lenient"], rows[rel]["lenient_warnings"]) == ("error", "ok", ["unknown_keyword"])
        @test occursin("unknown keyword `use_diis`", rows[rel]["strict_message"])
    end
    for rel in ("examples/ch3_radical.inp", "examples/o2_triplet.inp")
        @test (rows[rel]["strict"], rows[rel]["lenient_warnings"]) == ("error", ["rhf_open_shell"])
    end

    # `convergence` (line 9) and `maxiter` (line 10) are unknown keywords.
    p = parse_input(path("examples/legacy_inputs/h2o_opt.inp"); lenient = true)
    @test [(w.case, w.line) for w in p.warnings] == [(:unknown_keyword, 9), (:unknown_keyword, 10)]

    # Strict mode stops at `END_GEOMETRY` (line 14); lenient mode reads both geometries.
    rel = "examples/legacy_inputs/neb_test.inp"
    @test rows[rel]["strict"] == "error" && occursin("`END_GEOMETRY`", rows[rel]["strict_message"])
    p = parse_input(path(rel); lenient = true)
    inp = p.input
    @test (length(inp.atoms), length(inp.product_atoms)) == (3, 3)
    @test (inp.neb_images, inp.run_mode, inp.charge) == (5, NEB, 1)
    @test [(w.case, w.line) for w in p.warnings if w.case !== :unclosed_block] == [
        (:dropped_block_line, 14), (:block_keyword, 17), (:dropped_block_line, 21),
        (:block_keyword, 23), (:block_keyword, 24), (:dropped_block_line, 25),
    ]
    @test count(w -> w.case === :unclosed_block, p.warnings) == 2

    # The unclosed geometry blocks of the tests/ inputs.
    for rel in ("tests/argon_vdw.in", "tests/h2o.in", "tests/h2o_vdw.in", "tests/h2o_vdw_d2.in")
        @test rows[rel]["strict"] == "error"
        @test Set(rows[rel]["lenient_warnings"]) == Set(["block_keyword", "unclosed_block"])
    end

    # YAML text in `.inp` files: strict names the mismatch, lenient finds no atoms.
    for rel in ("molecules/h2_ccsd.inp", "molecules/oh_uccsd.inp")
        @test_throws "looks like YAML" parse_input(path(rel))
        @test_throws "no atoms" parse_input(path(rel); lenient = true)
    end

    # `basis 631-g` is the oracle's 6-31G alias and parses as written.
    for rel in ("examples/h2o_opt.inp", "examples/h2o_631g_sp.inp")
        @test parse_input(path(rel)).input.basis == "631-g"
    end

    # A trajectory of five frames.
    rel = "examples/legacy_inputs/neb_test.neb.xyz"
    @test rows[rel]["lenient_warnings"] == ["xyz_frames"]
    @test length(parse_input(path(rel); lenient = true).input.atoms) == 3
end
