# The parity harness against the stored oracle outputs in `captured/` and the stored values in
# `references.toml`. These items need no oracle binary.

# `==`: the printed decimal is parsed, and capture.jl stored the same parse.
@testitem "harness: parses the stored water PBE log" setup = [OracleHarness] begin
    using .OracleHarness: captured, total_energy, reference
    @test total_energy(read(captured("water_pbe_631g.log"), String)) == reference("water_pbe_631g", "total_energy")
    @test occursin(r"^[0-9a-f]{40}$", OracleHarness.references()["meta"]["oracle_commit"])
end

# The literals are the "Reference energies measured from the oracle" paragraph of the rewrite
# specs' 00-overview.md (lines 189-195), which quotes the oracle's printed 12-decimal values; both
# sides parse the same decimal, so `==`.
@testitem "harness: captured energies equal the 00-overview values" setup = [OracleHarness] begin
    using .OracleHarness: captured, total_energy
    energy(case) = total_energy(read(captured(case * ".log"), String))
    @test energy("water_pbe_631g") == -76.29926974315  # 00-overview: water PBE/6-31G
    @test energy("water_rhf_631g") == -75.983974472322  # 00-overview: water RHF/6-31G
    @test energy("o2_uhf_631g") == -149.54555366252    # 00-overview: O2 triplet UHF/6-31G
end

@testitem "harness: the other reference energies" setup = [OracleHarness] begin
    using .OracleHarness: captured, total_energy, reference
    for case in ("water_rhf_631g", "o2_uhf_631g")
        @test total_energy(read(captured(case * ".log"), String)) == reference(case, "total_energy")
    end
end

@testitem "harness: case inputs parse strictly" setup = [OracleHarness] begin
    using QuantumEngine
    using QuantumEngine.Input
    names = OracleHarness.case_names()
    @test names == sort(
        [
            "water_pbe_631g", "water_rhf_631g", "water_mp2", "water_ccsd", "h2_opt", "water_optfreq",
            "o2_uhf_631g", "h2_qtaim",
        ]
    )
    @testset "$case" for case in names
        t = OracleHarness.case_table(case)
        p = parse_inp(OracleHarness.case_input(case); source = case * ".inp")
        @test p isa ParsedInput && isempty(p.warnings)
        @test t["source"] isa String && !isempty(t["source"])
        # Every compared quantity's extractor is listed.
        @test all(OracleHarness.quantity_extractor(k) in t["extract"] for k in keys(t["tolerance"]))
    end
    # The derived inputs carry what their source line says.
    method(case) = parse_inp(OracleHarness.case_input(case)).input.method
    @test method("water_pbe_631g").kind == PBE
    @test parse_inp(OracleHarness.case_input("water_pbe_631g")).input.export_molden
    @test method("water_rhf_631g") == MethodSpec(HF, Restricted, (0, 0), 0.0)
    @test method("o2_uhf_631g") == MethodSpec(HF, Unrestricted, (0, 0), 0.0)
    @test parse_inp(OracleHarness.case_input("o2_uhf_631g")).input.basis == "6-31g"
    @test parse_inp(OracleHarness.case_input("h2_qtaim")).input.run_mode == QTAIM
end

@testitem "harness: references.toml is in the writer's form" setup = [OracleHarness] begin
    text = read(OracleHarness.REFERENCES, String)
    @test sprint(OracleHarness.write_references, OracleHarness.references()) == text
end

@testitem "harness: extractors" setup = [OracleHarness, Fixtures] begin
    using QuantumEngine.IO: fmt_fixed
    using .OracleHarness: captured, reference, reference_vector
    log(case) = read(captured(case * ".log"), String)
    steps = OracleHarness.opt_steps(log("h2_opt"))
    @test !isempty(steps)
    # The steps are numbered consecutively, and the last energy is the one printed
    # (`io/writer.rs:41`, `{:17.10}`).
    @test [s.step for s in steps] == first(steps).step .+ (0:(length(steps) - 1))
    @test last(steps).energy == reference("h2_opt", "opt_energy")
    @test occursin("E = " * fmt_fixed(last(steps).energy, 17, 10) * " |grad|", log("h2_opt"))
    mp2 = OracleHarness.mp2_correlation(read(captured("water_mp2.stdout"), String))
    @test mp2 < 0 && mp2 == reference("water_mp2", "mp2_correlation")
    # Water has 3N - 6 = 3 vibrations, at a minimum all real.
    ν = OracleHarness.frequencies(log("water_optfreq"))
    @test length(ν) == 3 && all(≥(0), ν)
    @test ν == reference_vector("water_optfreq", "frequencies")
    ext = OracleHarness.ext_report(captured("h2_qtaim.ext"))
    @test ext["n_nacp"] == 2
    @test ext["n_bcp"] == reference("h2_qtaim", "ext_report.n_bcp")
end

# Kind (a): the two printed values have 12 decimals, each ±5e-13, plus the Float64 sum of the 3
# pair terms (~6e-15), within 2e-12.
@testitem "harness: nuclear repulsion cross-check" setup = [OracleHarness, Fixtures] begin
    using QuantumEngine
    e_nuc = OracleHarness.nuclear_repulsion_from(read(OracleHarness.captured("water_pbe_631g.log"), String))
    @test abs(nuclear_repulsion(Fixtures.water()) - e_nuc) ≤ 2.0e-12
end

@testitem "harness: captured files match their SHA-256" setup = [OracleHarness] begin
    using .OracleHarness: captured, file_sha256
    @testset "$case" for case in OracleHarness.case_names()
        hashes = OracleHarness.case_table(case)["sha256"]
        @test sort(collect(keys(hashes))) == sort(["log"; "stdout"; OracleHarness.case_table(case)["exports"]])
        for (ext, h) in hashes
            @test file_sha256(captured(case * "." * ext)) == h
        end
    end
    for (rel, _) in OracleHarness.FIXTURE_FILES
        @test file_sha256(OracleHarness.fixture_stdout(rel)) == OracleHarness.references()["fixture"][rel]["sha256"]
    end
end

@testitem "harness: missing tolerance" setup = [OracleHarness] begin
    using .OracleHarness: tolerance
    # The DFT row is added with its measured defect; until then the comparison throws.
    @test_throws "oracle: no tolerance row dft_closed.energy for water_pbe_631g.total_energy" tolerance(
        "water_pbe_631g", "total_energy"
    )
    @test_throws "oracle: no tolerance row (no map entry) for water_pbe_631g.mp2_correlation" tolerance(
        "water_pbe_631g", "mp2_correlation"
    )
    @test_throws "oracle: no case nosuch in references.toml" tolerance("nosuch", "total_energy")
    @test tolerance("water_rhf_631g", "total_energy") == 1.0e-9
    @test tolerance("water_mp2", "mp2_correlation") == 1.0e-9
    @test tolerance("o2_uhf_631g", "total_energy") == 1.0e-7
    @test tolerance("water_optfreq", "frequencies") == 1.0
    @test tolerance("water_pbe_631g", "nuclear_repulsion") == 5.0e-13
end

@testitem "harness: locating the oracle" setup = [OracleHarness] begin
    withenv("QUANTUM_ENGINE_ORACLE" => "quantum_engine", "QUANTUM_ENGINE_ORACLE_BASIS" => nothing) do
        @test OracleHarness.oracle_binary() == "quantum_engine"
        @test_throws "oracle: QUANTUM_ENGINE_ORACLE is set but QUANTUM_ENGINE_ORACLE_BASIS (the oracle's basis_sets directory) is not" OracleHarness.oracle_basis_dir()
    end
    withenv("QUANTUM_ENGINE_ORACLE" => "", "QUANTUM_ENGINE_ORACLE_BASIS" => nothing) do
        @test OracleHarness.oracle_binary() === nothing
        @test_throws "oracle: QUANTUM_ENGINE_ORACLE_BASIS (the oracle's basis_sets directory) is not set" OracleHarness.oracle_basis_dir()
        @test_throws "oracle: QUANTUM_ENGINE_ORACLE (the oracle binary) is not set" OracleHarness.run_oracle("h2_opt")
    end
    withenv("QUANTUM_ENGINE_ORACLE_BASIS" => joinpath(mktempdir(), "absent")) do
        @test_throws "oracle: QUANTUM_ENGINE_ORACLE_BASIS is not a directory" OracleHarness.oracle_basis_dir()
    end
end

# Lines written from the oracle's format strings (each cited in the extractor's docstring), not
# oracle output: they check the patterns and the missing-line errors without a capture.
@testitem "harness: extractors read the oracle's line formats" setup = [OracleHarness] begin
    using QuantumEngine.IO: fmt_fixed, fmt_exp
    H = OracleHarness
    # engine/results.rs:18 and scf/scf_cycle.rs:591.
    log = join(
        [
            "SCF iter   1  E = " * fmt_fixed(-75.5, 16, 12) * "  ΔE = " * fmt_exp(1.0e-2, 3; width = 8),
            "SCF converged in 9 iterations  E = -85.123456789012",
            "  Total Energy:  " * fmt_fixed(-76.0, 20, 12) * " Hartree",
            "  Total Energy:  " * fmt_fixed(-76.25, 20, 12) * " Hartree",
        ], '\n'
    )
    @test H.total_energy(log) == -76.25
    @test H.scf_iterations(log) == [-75.5]
    @test H.nuclear_repulsion_from(log) == -76.25 - -85.123456789012
    @test H.scf_iterations("UHF iter   2  E = " * fmt_fixed(-1.5, 18, 12) * "  ΔE = 1.000e-3") == [-1.5]
    # io/writer.rs:41.
    opt = "[OPT] Step   0: E = " * fmt_fixed(-1.1, 17, 10) * " |grad| = 2.500e-4\n" *
        "[OPT] Step   1: E = " * fmt_fixed(-1.2, 17, 10) * " |grad| = 1.000e-5"
    @test H.opt_steps(opt) == [(step = 0, energy = -1.1, gnorm = 2.5e-4), (step = 1, energy = -1.2, gnorm = 1.0e-5)]
    # scf/mp2.rs:74 and scf/ccsd.rs:365.
    @test H.mp2_correlation("  Total MP2 Correlation: -0.1280755678 Hartree") == -0.1280755678
    @test H.ccsd_correlation("  FINAL CCSD CORRELATION:  " * fmt_fixed(-0.2, 18, 10) * " Hartree") == -0.2
    # vibrations/mod.rs:154-178: the T/R rows are left out.
    row(i, ν, type, λ) = "║ " * lpad(i, 4) * " │ " * fmt_fixed(ν, 16, 4) * " │ " * rpad(type, 8) * " │ " * fmt_fixed(λ, 13, 6) * " ║"
    table = join(
        [
            "║            VIBRATIONAL FREQUENCIES                 ║",
            "║ Mode │ Frequency (cm^-1) │  Type    │  Eigenvalue   ║",
            row(1, 12.5, "T/R   ", 0.0), row(2, -120.0, "Im    ", -0.001), row(3, 1650.25, "Vib   ", 0.2),
            "╚══════╧══════════════════╧══════════╧═══════════════╝",
        ], '\n'
    )
    @test H.frequencies(table) == [-120.0, 1650.25]
    # analysis/aim.rs:226-229 and 252-255.
    ext = joinpath(mktempdir(), "h2.ext")
    write(
        ext, """
         Number of Nuclear Attractors (3,-3) [NACP]:      2
         Number of Bond Critical Points (3,-1) [BCP]:      1
         Number of Ring Critical Points (3,+1) [RCP]:      0
         Number of Cage Critical Points (3,+3) [CCP]:      0
         1     H1 - H2           1.40000      1.40000      0.00000      0.265000    -1.050000     0.00000
         1     H1 - H2          0.010000    -0.300000    -0.290000    3.0000    -1.0943   -100.000   Shared (Covalent)
        """
    )
    d = H.ext_report(ext)
    @test d == Dict(
        "n_nacp" => 2, "n_bcp" => 1, "n_rcp" => 0, "n_ccp" => 0, "bcp1.r_e" => 1.4, "bcp1.r_b" => 1.4,
        "bcp1.delta_r" => 0.0, "bcp1.rho" => 0.265, "bcp1.laplacian" => -1.05, "bcp1.ellipticity" => 0.0,
    )
    # engine/mod.rs:76-88 and engine/electronic.rs:24.
    out = join(
        [
            "Title: t", "║  CALCULATION SUMMARY  ║", "Title:        Water", "Method:       RHF", "Basis Set:    6-31g",
            "Electrons:    10", "Multiplicity: 1", "Charge:       0", "Convergence:  1.0e-6", "  Atoms:        3",
        ], '\n'
    )
    @test H.echoed_settings(out) == Dict(
        "title" => "Water", "method" => "RHF", "basis" => "6-31g", "multiplicity" => 1, "charge" => 0,
        "convergence" => "1.0e-6", "atoms" => 3,
    )
    @test_throws "oracle stdout: no `charge` line in the Calculation Summary" H.echoed_settings(
        "║  CALCULATION SUMMARY  ║\nTitle: a\nMethod: RHF\nBasis Set: b\nMultiplicity: 1\nConvergence:  1.0e-6"
    )
    @test_throws "oracle stdout: no `Calculation Summary` heading" H.echoed_settings("Title: a")
    # The missing-line errors.
    @test_throws "oracle log: no `Total Energy:` line" H.total_energy("SCF converged in 9 iterations  E = -1.0")
    @test_throws "oracle log: no `SCF converged … E =` line" H.nuclear_repulsion_from(
        "UHF converged in 9 iterations\n  Total Energy:  " * fmt_fixed(-1.0, 20, 12) * " Hartree"
    )
    @test_throws "oracle log: no `SCF iter` or `UHF iter` line" H.scf_iterations("")
    @test_throws "oracle log: no `[OPT] Step` line" H.opt_steps("")
    @test_throws "oracle stdout: no `Total MP2 Correlation:` line" H.mp2_correlation("")
    @test_throws "oracle stdout: no `FINAL CCSD CORRELATION:` line" H.ccsd_correlation("")
    @test_throws "oracle log: no `VIBRATIONAL FREQUENCIES` line" H.frequencies("")
    @test_throws "oracle log: the `VIBRATIONAL FREQUENCIES` table has no closing line" H.frequencies(
        "║            VIBRATIONAL FREQUENCIES                 ║"
    )
    empty = joinpath(mktempdir(), "e.ext")
    write(empty, "")
    @test_throws "oracle ext: no critical-point counts in e.ext" H.ext_report(empty)
    # The scrubbed lines: local paths rewritten, timing lines dropped, leftover paths rejected.
    dir = mktempdir()
    raw = "Reading input: $dir/w.inp\nTitle: w\nLogging to: $dir/w.log\n\n║ QTAIM topological elapsed: 1.234 s\nIQA completed in 12.30ms\nCalculation completed successfully.\n"
    @test H.scrub(raw, "w.inp", dir) == "Reading input: w.inp\nTitle: w\nLogging to: w.log\n\nCalculation completed successfully.\n"
    @test_throws "oracle: the output of w.inp names the run directory" H.scrub("✓ Exported Molden to: $dir/w.molden", "w.inp", dir)
    @test_throws "oracle: the output of w.inp names the home directory" H.scrub(homedir() * "/x", "w.inp", dir)
end
