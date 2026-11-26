# YAML input through the extension `QuantumEngineYAMLExt`, which loads with YAML.jl. The oracle
# (`quantum_engine` at 4a78a1d) reads `.yaml`, `.yml` and `.json` files with yaml-rust2 0.9.0 in
# `parse_yaml_string` (`src/input/parser.rs:677-1076`); the values below are the ones it reads, from
# that source, with the lines cited.

@testmodule YAMLText begin
    using QuantumEngine, YAML
    # H2 in bohr, with `extra` lines appended to the top-level mapping.
    h2(extra = "") = """
        unit: bohr
        geometry:
          - symbol: H
            position: [0.0, 0.0, -0.7]
          - symbol: H
            position: [0.0, 0.0, 0.7]
        """ * extra
    strict(text) = parse_yaml(text; source = "t.yaml")
    lenient(text) = parse_yaml(text; source = "t.yaml", lenient = true)
    cases(p) = [w.case for w in p.warnings]
end

# Without YAML.jl, in a fresh process, `parse_yaml` has no method; with it, the extension defines one.
@testitem "yaml: the extension loads with YAML" begin
    using QuantumEngine
    script = "using QuantumEngine; print(hasmethod(parse_yaml, Tuple{AbstractString}), \" \", " *
        "isnothing(Base.get_extension(QuantumEngine, :QuantumEngineYAMLExt)))"
    project = dirname(Base.active_project())
    @test read(`$(Base.julia_cmd()) --startup-file=no --project=$project -e $script`, String) == "false true"
    using YAML
    @test hasmethod(parse_yaml, Tuple{AbstractString})
    @test !isnothing(Base.get_extension(QuantumEngine, :QuantumEngineYAMLExt))
end

@testitem "yaml: tests/si.yaml" setup = [Fixtures] begin
    using QuantumEngine, YAML
    using QuantumEngine.Input: Bands
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    using StaticArrays: SMatrix
    text = Fixtures.oracle_input("tests/si.yaml")
    for lenient in (false, true)
        p = parse_yaml(text; source = "si.yaml", lenient)
        i = p.input
        @test isempty(p.warnings)
        @test i.title == "Silicon Crystal Band Structure"                         # parser.rs:713
        @test i.run_mode == Bands                                                # parser.rs:683-707
        @test i.max_iterations == 50                                             # parser.rs:741
        @test i.scf_convergence == 1.0e-6                                        # parser.rs:742
        @test i.pw_cutoff == 6.0                                                 # parser.rs:782
        # parser.rs:787-799, rows in Å whatever the geometry unit.
        @test i.lattice == SMatrix{3, 3}(0.0, 5.13, 5.13, 5.13, 0.0, 5.13, 5.13, 5.13, 0.0)
        @test i.kpoints == (1, 1, 1)                                             # parser.rs:820-835
        @test i.kpath == ["G", "X", "W", "L", "G"]                               # parser.rs:838-846
        # parser.rs:1006-1028: `atoms` when `geometry` is absent; no `unit`, so Å.
        @test [a.symbol for a in i.atoms] == ["Si", "Si"]
        @test i.atoms[1].position == [0.0, 0.0, 0.0]
        @test i.atoms[2].position == fill(2.565 * ANGSTROM_TO_BOHR, 3)
    end
end

# Neither file has a geometry, so the oracle reads no atoms (parser.rs:1006-1029); every key it does
# not know it ignores. Strict mode stops at the first unknown key, `dft`; lenient mode reads what the
# oracle reads and ends in the error of an input without atoms.
@testitem "yaml: input_hf.yaml and input.yaml" setup = [Fixtures] begin
    using QuantumEngine, YAML
    for f in ("examples/legacy_inputs/input_hf.yaml", "src/input/input.yaml")
        text = Fixtures.oracle_input(f)
        @test_throws "input: $f: dft: unknown key; accepted keys: title, charge" parse_yaml(text; source = f)
        @test_throws "input: $f: no atoms: the input has no geometry block" parse_yaml(text; source = f, lenient = true)
    end
end

@testitem "yaml: the CASSCF string of parser.rs:1257" begin
    using QuantumEngine, YAML
    using QuantumEngine.Input: MethodSpec, CASSCF, AutoReference, OptFreq
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    text = """
    title: CASSCF & ZORA YAML Test
    charge: 0
    multiplicity: 1
    basis: def2-svp
    method: casscf(4,4)
    zora: true
    dispersion: true
    task:
      mode: optfreq
    f12:
      gamma: 1.4
      model: F12b
    geometry:
      - symbol: H
        position: [0.0, 0.0, 0.0]
      - symbol: H
        position: [0.0, 0.0, 0.74]
    """
    for lenient in (false, true)
        p = parse_yaml(text; source = "t.yaml", lenient)
        i = p.input
        @test isempty(p.warnings)
        # The assertions of parser.rs:1276-1284.
        @test i.title == "CASSCF & ZORA YAML Test"
        @test i.method == MethodSpec(CASSCF, AutoReference, (4, 4), 0.0)
        @test i.run_mode == OptFreq
        @test i.zora && i.dispersion
        @test i.use_f12 && i.f12_gamma == 1.4
        @test length(i.atoms) == 2
        # parser.rs:714-716, 995 and 1018-1026.
        @test (i.charge, i.multiplicity, i.basis, i.f12_model) == (0, 1, "def2-svp", "F12b")
        @test i.atoms[2].position == [0.0, 0.0, 0.74 * ANGSTROM_TO_BOHR]
    end
end

# `calculation` is no key of the oracle's reader, so it ignores the method and basis under it.
@testitem "yaml: the SMD string of tests/test_smd_solvation.rs:239" begin
    using QuantumEngine, YAML
    using QuantumEngine.Input: SMD, HF, AutoReference
    text = """
    calculation:
      method: hf
      basis: sto-3g
    solvation:
      model: smd
      solvent: chloroform
    geometry:
      - symbol: O
        x: 0.0
        y: 0.0
        z: 0.0
    """
    @test_throws "input: t.yaml: calculation: unknown key; accepted keys: title" parse_yaml(text; source = "t.yaml")
    p = parse_yaml(text; source = "t.yaml", lenient = true)
    @test [w.case for w in p.warnings] == [:yaml_unknown_key]
    i = p.input
    # The assertion of tests/test_smd_solvation.rs:255-260; parser.rs:902-915.
    @test i.solvation.model == SMD && i.solvation.solvent == :chloroform
    @test [a.symbol for a in i.atoms] == ["O"] && i.atoms[1].position == [0.0, 0.0, 0.0]
    @test (i.method.kind, i.method.reference, i.basis) == (HF, AutoReference, "sto-3g")
end

# The two files in `.inp` syntax with a `.yaml` extension. YAML.jl reads h2_io_test.yaml as one
# plain scalar: strict mode names the mismatch, and lenient mode, as the oracle, reads a document
# without keys and so without atoms. test_neb.yaml is not YAML: the comment after its first lines ends
# the scalar, and the next line starts a second document without `---`; YAML.jl throws its parser
# error, as yaml-rust2 does ("did not find expected <document start>", parser.rs:603 of yaml-rust2).
@testitem "yaml: .inp syntax in a .yaml file" setup = [Fixtures] begin
    using QuantumEngine, YAML
    msg = "top-level YAML document is not a mapping; this file looks like .inp syntax"
    f = "examples/legacy_inputs/h2_io_test.yaml"
    text = Fixtures.oracle_input(f)
    @test_throws "input: $f: $msg" parse_yaml(text; source = f)
    @test_throws "input: $f: no atoms: the input has no geometry block" parse_yaml(text; source = f, lenient = true)
    f = "examples/legacy_inputs/test_neb.yaml"
    text = Fixtures.oracle_input(f)
    for lenient in (false, true)
        @test_throws "expected '<document start>'" parse_yaml(text; source = f, lenient)
        @test_broken try
            parse_yaml(text; source = f, lenient)
            false
        catch e
            occursin(msg, sprint(showerror, e))
        end
    end
end

@testitem "yaml: documents that are not a mapping" setup = [YAMLText] begin
    using QuantumEngine, YAML
    using .YAMLText: strict, lenient
    for text in ("", "# only a comment\n", "---\n...\n")
        @test_throws "input: t.yaml: the document is empty" strict(text)
        @test_throws "input: t.yaml: the document is empty" lenient(text)
    end
    @test_throws "input: t.yaml: top-level YAML document is not a mapping; this file looks like .inp syntax" strict("charge 0\n")
    for text in ("- 1\n- 2\n", "42\n")
        @test_throws "input: t.yaml: top-level YAML document is not a mapping" strict(text)
        @test_throws "input: t.yaml: no atoms: the input has no geometry block" lenient(text)
    end
    # The oracle reads the first document of several (parser.rs:678-680), and so does YAML.jl.
    p = strict(YAMLText.h2("charge: 0\n") * "---\ncharge: 1\n")
    @test p.input.charge == 0
end

# yaml-rust2 rejects a repeated key (yaml.rs:223-228 of yaml-rust2 0.9.0); YAML.jl logs an error and
# keeps the last value, and the extension reads the mapping as written, so the key is an error in
# both modes.
@testitem "yaml: a repeated key" setup = [YAMLText] begin
    using QuantumEngine, YAML
    using .YAMLText: h2, strict, lenient
    for run in (strict, lenient)
        @test_logs (:error, "Duplicate key detected in mapping") match_mode = :any @test_throws(
            "input: t.yaml: duplicate key \"charge\"", run(h2("charge: 0\ncharge: 1\n")),
        )
        @test_logs (:error, "Duplicate key detected in mapping") match_mode = :any @test_throws(
            "input: t.yaml: scf: duplicate key \"max_iter\"", run(h2("scf:\n  max_iter: 5\n  max_iter: 6\n")),
        )
    end
end

@testitem "yaml: keys keep the order of the file" setup = [YAMLText] begin
    using QuantumEngine, YAML
    using .YAMLText: h2, strict
    # The first unknown key of the file is the one reported.
    @test_throws "input: t.yaml: zz: unknown key" strict(h2("zz: 1\naa: 2\nmm: 3\n"))
    @test_throws "input: t.yaml: aa: unknown key" strict(h2("aa: 1\nzz: 2\nmm: 3\n"))
end

# Where YAML.jl types a scalar differently from the oracle's yaml-rust2 0.9.0 (`Yaml::from_str` and
# `parse_f64`, yaml.rs:66-73 and 656-685: an integer by Rust's `i64` parser after a `0x`, `0o` or `+`
# prefix, else a float by Rust's `f64` parser, else a string; only `true`, `false`, `~`, `null` and
# the empty value are keywords). JSON.jl rejects each of these forms as invalid JSON, so the
# difference is YAML's alone. One test per difference states what QuantumEngine reads; the oracle's
# value is in the comment.
@testitem "yaml: typing differences from the oracle's reader" setup = [YAMLText] begin
    using QuantumEngine, YAML
    using .YAMLText: h2, strict, lenient, cases
    # `017`: YAML.jl reads an octal 15; yaml-rust2 a decimal 17. (`-017` and `+017` are -17 and 17 in both.)
    @test strict(h2("max_iterations: 017\n")).input.max_iterations == 15
    @test strict(h2("charge: -017\nmultiplicity: 2\n")).input.charge == -17
    # `08`: YAML.jl reads a string; yaml-rust2 the integer 8.
    @test_throws "input: t.yaml: max_iterations: expected an integer, got a string" strict(h2("max_iterations: 08\n"))
    p = lenient(h2("max_iterations: 08\n"))
    @test p.input.max_iterations == 50 && cases(p) == [:yaml_wrong_type]
    # `True`, `TRUE`, `False`: YAML.jl reads Booleans; yaml-rust2 strings, which `as_bool` ignores (zora stays false).
    @test strict(h2("zora: True\n")).input.zora
    @test strict(h2("zora: TRUE\n")).input.zora
    @test !strict(h2("zora: False\ndispersion: TRUE\n")).input.zora
    # `nan`, `NaN`, `inf`, `infinity`: YAML.jl reads strings; yaml-rust2 floats by Rust's parser (NaN, Inf).
    for word in ("nan", "NaN", "inf", "infinity")
        @test_throws "input: t.yaml: scf_convergence: expected a number, got a string" strict(h2("scf_convergence: $word\n"))
        q = lenient(h2("scf_convergence: $word\n"))
        @test q.input.scf_convergence == 1.0e-6 && cases(q) == [:yaml_wrong_type]
    end
    # `.NaN`: YAML.jl reads NaN; yaml-rust2 a string (its list holds `.nan`, `NaN` and `.NAN` only).
    @test_throws "input: t.yaml: scf_convergence: expected a finite number, got NaN" strict(h2("scf_convergence: .NaN\n"))
    p = lenient(h2("scf_convergence: .NaN\n"))
    @test isnan(p.input.scf_convergence) && cases(p) == [:nonfinite]
    # `0b101`: YAML.jl reads the integer 5; yaml-rust2 a string, so the oracle keeps 50 iterations.
    @test strict(h2("max_iterations: 0b101\n")).input.max_iterations == 5
    # `1_000` and `1_000.5`: YAML.jl reads 1000 and 1000.5; yaml-rust2 strings.
    @test strict(h2("max_iterations: 1_000\n")).input.max_iterations == 1000
    @test strict(h2("scf:\n  damping: 1_000.5\n")).input.damping == 1000.5
    # `Null`, `NULL`: YAML.jl reads null; yaml-rust2 strings (the oracle's title would be "Null").
    @test_throws "input: t.yaml: title: expected a string, got null" strict(h2("title: Null\n"))
    p = lenient(h2("title: NULL\n"))
    @test p.input.title == "" && cases(p) == [:yaml_wrong_type]
    # `2001-12-14`: YAML.jl reads a `Date`, which no node holds, in both modes; yaml-rust2 a string.
    @test_throws "input: t.yaml: title: unsupported value of type" strict(h2("title: 2001-12-14\n"))
    @test_throws "input: t.yaml: title: unsupported value of type" lenient(h2("title: 2001-12-14\n"))
    # An integer beyond Int64: YAML.jl reads an Int128, an error in both modes; yaml-rust2 a float,
    # which `as_i64` ignores (the oracle's charge stays 0).
    @test_throws "input: t.yaml: charge: the integer 9223372036854775808 is outside the Int64 range" strict(
        h2("charge: 9223372036854775808\n"),
    )
    @test_throws "input: t.yaml: charge: the integer 9223372036854775808 is outside the Int64 range" lenient(
        h2("charge: 9223372036854775808\n"),
    )
    # A key that is not a string: YAML.jl reads the integer key 1, an error in both modes; the
    # oracle looks keys up by string and ignores it.
    @test_throws "input: t.yaml: the key 1 of type Int64 is not a string" strict(h2("1: x\n"))
    @test_throws "input: t.yaml: the key 1 of type Int64 is not a string" lenient(h2("1: x\n"))
end
