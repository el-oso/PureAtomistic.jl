# The expected lines are copied verbatim from the oracle's `examples/h2o_pbe0_631g.log`.
@testitem "oracle output lines" setup = [Fixtures] begin
    using QuantumEngine
    using QuantumEngine.IO: Logger, print_total_energy, print_atom_table, log_heading, report_optimization_step
    out = IOBuffer(); lg = Logger(out, devnull)
    print_total_energy(lg, -76.301896228438)
    @test String(take!(out)) == "  Total Energy:      -76.301896228438 Hartree\n"
    print_atom_table(lg, Fixtures.water().atoms)
    @test split(String(take!(out)), '\n')[3] ==
        "   1 O        0.00000000     0.00000000     0.22166487     0.00000000     0.00000000     0.11730000"
    log_heading(lg, "Final Results")
    @test String(take!(out)) == "\n╔═════════════════╗\n║  FINAL RESULTS  ║\n╚═════════════════╝\n\n"
    # io/writer.rs:41: "[OPT] Step {:3}: E = {:17.10} |grad| = {:.3e}"; -1.1167593075 is 13
    # characters, so {:17.10} pads it with 4 spaces after "E = ".
    report_optimization_step(lg, 3, -1.1167593075, 2.5e-4)
    @test String(take!(out)) == "[OPT] Step   3: E =     -1.1167593075 |grad| = 2.500e-4\n"
end

# The header, rule and rows of the same log's atom table, and the section, success, warning and
# error lines in the oracle's layout (`io/writer.rs`).
@testitem "logger: tables and lines" setup = [Fixtures] begin
    using QuantumEngine.IO: Logger, print_atom_table, log_section, log_success, log_warning, log_error, log_heading
    out = IOBuffer(); lg = Logger(out, devnull)
    print_atom_table(lg, Fixtures.water().atoms)
    @test String(take!(out)) == join(
        [
            "  Atom         X [Bohr]       Y [Bohr]       Z [Bohr]      X [Ang]        Y [Ang]        Z [Ang]",
            "  " * "─"^95,
            "   1 O        0.00000000     0.00000000     0.22166487     0.00000000     0.00000000     0.11730000",
            "   2 H        0.00000000     1.43090062    -0.88665950     0.00000000     0.75720000    -0.46920000",
            "   3 H        0.00000000    -1.43090062    -0.88665950     0.00000000    -0.75720000    -0.46920000",
            "",
            "",
        ], '\n'
    )
    log_section(lg, "ORBITAL ENERGIES")
    @test String(take!(out)) == "\nORBITAL ENERGIES\n----------------\n\n"
    log_success(lg, "done")
    log_warning(lg, "w")
    log_error(lg, "e")
    @test String(take!(out)) == "✓ done\n⚠ Warning: w\n❌ Error: e\n"
    # The box and the rule count the title's bytes, as the oracle's `len()` does: "Å" is 2 bytes.
    log_heading(lg, "Å")
    @test String(take!(out)) == "\n╔══════╗\n║  Å  ║\n╚══════╝\n\n"
    log_section(lg, "Å")
    @test String(take!(out)) == "\nÅ\n--\n\n"
end

@testitem "logger: generic indexing" setup = [Fixtures] begin
    using OffsetArrays
    using QuantumEngine.IO: Logger, print_atom_table
    atoms = Fixtures.water().atoms
    table(a) = (out = IOBuffer(); print_atom_table(Logger(out, devnull), a); String(take!(out)))
    expected = table(atoms)
    @test table(OffsetArray(atoms, -1:1)) == expected
    @test table(view(atoms, 1:3)) == expected
    rows = split(expected, '\n')[3:5]
    @test [first(r, 4) for r in rows] == ["   1", "   2", "   3"]
    # A Float32 molecule prints its Float32 coordinates and their Float32 Å values; the expected row
    # is Rust's `{:12.8}` of the f32 product 1.0 * 0.529177210903.
    a32 = [Atom("H", Float32[0, 0, 1])]
    @test split(table(a32), '\n')[3] ==
        "   1 H        0.00000000     0.00000000     1.00000000     0.00000000     0.00000000     0.52917719"
end

@testitem "logger: levels" begin
    using QuantumEngine.IO: Logger, Silent, Low, Normal, High, log_print, log_error, log_warning, log_heading,
        print_total_energy
    function run(v)
        out, log = IOBuffer(), IOBuffer()
        lg = Logger(out, log; verbosity = v)
        log_print(lg, High, "high")
        log_print(lg, Normal, "normal")
        log_warning(lg, "w")
        log_heading(lg, "h")
        print_total_energy(lg, 1.0)
        log_error(lg, "e")
        log_print(lg, Low, "low")
        return String(take!(out)), String(take!(log))
    end
    o, l = run(Low)
    @test o == l == "❌ Error: e\nlow\n"
    o, l = run(Silent)
    @test o == l == ""
    o, l = run(Normal)
    @test o == l
    @test !occursin("high", o) && occursin("normal\n", o) && endswith(o, "❌ Error: e\nlow\n")
    o, l = run(High)
    @test o == l
    @test startswith(o, "high\nnormal\n")
    # The default verbosity is Normal.
    @test Logger(IOBuffer(), devnull).verbosity == Normal
end

@testitem "logger: verbosity mapping" begin
    using QuantumEngine.IO: verbosity_from_input, Low, Normal, High
    @test verbosity_from_input(0) == Low
    @test verbosity_from_input(1) == Normal
    @test verbosity_from_input(2) == High
    @test verbosity_from_input(5) == High
    @test_throws "io: verbosity must be ≥ 0, got -1" verbosity_from_input(-1)
end

@testitem "logger: paths" begin
    using QuantumEngine.IO: log_path, output_base
    @test log_path("dir/h2.inp", nothing) == "dir/h2.log"
    @test log_path("h2.inp", "h2_test") == "h2_test"
    @test output_base("dir/h2.log") == "dir/h2"
    # Rust's `Path::with_extension("log")` and `with_extension("")` on the same paths (rustc 1.100):
    # a name without an extension gets one, and a leading dot does not start an extension.
    for (p, lp, base) in (
            ("h2", "h2.log", "h2"), (".hidden", ".hidden.log", ".hidden"), ("a.tar.gz", "a.tar.log", "a.tar"),
            ("foo.", "foo.log", "foo"), ("dir.d/h2", "dir.d/h2.log", "dir.d/h2"), ("dir/.h2.inp", "dir/.h2.log", "dir/.h2"),
            ("dir/.hidden", "dir/.hidden.log", "dir/.hidden"), ("a..b", "a..log", "a."),
        )
        @test log_path(p, nothing) == lp
        @test output_base(p) == base
    end
    @test log_path(SubString("xdir/h2.inp", 2), nothing) == "dir/h2.log"
    # A path with no file name has no extension to replace.
    for p in ("", "dir/", "a/b.c/", ".", "..", "dir/..", "/")
        @test_throws "io: path has no file name" log_path(p, nothing)
        @test_throws "io: path has no file name" output_base(p)
    end
    @test log_path("dir/", "out.log") == "out.log"
end
