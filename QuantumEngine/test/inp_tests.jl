@testmodule InpText begin
    using QuantumEngine

    # The water geometry of `examples/h2o_631g_sp.inp` as a block with the opening line `header`.
    water(header = "geometry angstrom") = """
    $header
    O  0.0000  0.0000  0.1173
    H  0.0000  0.7572 -0.4692
    H  0.0000 -0.7572 -0.4692
    end
    """

    # The input `text` parsed strictly from the source `t.inp`.
    parse(text; lenient = false) = parse_inp(text; source = "t.inp", lenient).input
end

@testitem "inp: h2.inp values" setup = [Fixtures] begin
    using QuantumEngine
    using QuantumEngine.Input
    p = parse_inp(Fixtures.oracle_input("examples/h2.inp"); source = "h2.inp")
    @test p isa ParsedInput && isempty(p.warnings)
    inp = p.input
    @test inp.title == "Hydrogen molecule"
    @test (inp.charge, inp.multiplicity, inp.basis) == (0, 1, "sto-3g")
    @test inp.method.kind == HF && inp.method.reference == Restricted
    @test inp.scf_convergence == 1.0e-6 && inp.max_iterations == 50
    # The geometry block is in bohr, so the coordinates are stored as written.
    @test [a.symbol for a in inp.atoms] == ["H", "H"]
    @test inp.atoms[1].position == [0.0, 0.0, -0.7] && inp.atoms[2].position == [0.0, 0.0, 0.7]
    # Every other field keeps its default.
    d = InputFile()
    for f in fieldnames(InputFile)
        f in (:atoms, :title, :method) && continue
        @test getfield(inp, f) == getfield(d, f)
    end
    mol = molecule(inp)
    @test mol isa Molecule{Float64, Vector{Atom{Float64}}}
    @test mol.atoms == inp.atoms && (mol.charge, mol.multiplicity) == (0, 1)
    @test Fixtures.fields_equal(mol, Fixtures.h2())
    # A `SubString` text reads as the `String` it is cut from.
    text = Fixtures.oracle_input("examples/h2.inp")
    sub = SubString("xx" * text * "yy", 3, 2 + ncodeunits(text))
    @test Fixtures.fields_equal(parse_inp(sub; source = "h2.inp").input, inp)
end

@testitem "inp: units" setup = [InpText] begin
    using QuantumEngine
    using StaticArrays: SVector
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    inp = InpText.parse(InpText.water())
    @test inp.atoms[1].position[3] == 0.1173 * ANGSTROM_TO_BOHR
    @test inp.atoms[2].position == [0.0, 0.7572 * ANGSTROM_TO_BOHR, -0.4692 * ANGSTROM_TO_BOHR]
    for u in ("ang", "ANGSTROM", "Ang")
        @test InpText.parse(InpText.water("geometry $u")).atoms == inp.atoms
    end
    # The default unit is Å.
    @test InpText.parse(InpText.water("geometry")).atoms == inp.atoms
    for u in ("bohr", "au", "AU", "Bohr")
        @test InpText.parse(InpText.water("geometry $u")).atoms[1].position[3] == 0.1173
    end
    @test_throws "unknown value `furlong` for the geometry unit; accepted: angstrom, ang, bohr, au" InpText.parse(InpText.water("geometry furlong"))
    # No unit carries over from one block to the next: a `product_geometry` or `charges` block
    # without a unit after `geometry bohr` is in Å.
    inp = InpText.parse(InpText.water("geometry bohr") * "product_geometry\nH 0 0 1.5\nend\ncharges\n0.5 0 0 2.0\nend\n")
    @test inp.atoms[1].position[3] == 0.1173
    @test inp.product_atoms[1].position[3] == 1.5 * ANGSTROM_TO_BOHR
    @test inp.point_charges == [PointCharge(0.5, SVector(0.0, 0.0, 2.0 * ANGSTROM_TO_BOHR))]
    inp = InpText.parse(InpText.water() * "product_geometry bohr\nH 0 0 1.5\nend\ncharges au\n0.5 0 0 2.0\nend\n")
    @test inp.product_atoms[1].position[3] == 1.5
    @test inp.point_charges[1].position[3] == 2.0
end

@testitem "inp: unknown keyword" setup = [InpText] begin
    using QuantumEngine
    @test_throws "input: t.inp:1: unknown keyword `foo_bar`; accepted keywords: " InpText.parse("foo_bar 1\n" * InpText.water())
    @test_throws "scf_convergence" InpText.parse("foo_bar 1\n" * InpText.water())
    @test_throws "unknown keyword `use_diis`" InpText.parse("use_diis true\n" * InpText.water())
    words = QuantumEngine.Input.accepted_keywords()
    @test issorted(words) && allunique(words)
    # Every alias is listed, and the block openers.
    for w in ("temp", "temperature", "smd_grid", "pcm_grid", "freeze", "freeze_torsion", "k_path", "geometry", "cell", "pbc", "oniom")
        @test w in words
    end
    msg = sprint(
        showerror, try
            InpText.parse("foo_bar 1\n" * InpText.water())
        catch e
            e
        end
    )
    @test endswith(msg, "accepted keywords: " * join(words, ", "))
    # The list is a copy: changing it changes neither the next list nor the message.
    push!(words, "zzz_bogus")
    @test !("zzz_bogus" in QuantumEngine.Input.accepted_keywords())
    @test QuantumEngine.Input.accepted_keywords() == words[1:(end - 1)]
    msg2 = sprint(
        showerror, try
            InpText.parse("foo_bar 1\n" * InpText.water())
        catch e
            e
        end
    )
    @test msg2 == msg && !occursin("zzz_bogus", msg2)
    # Keywords are read without regard to case.
    @test InpText.parse("CHARGE 0\nMultiplicity 1\n" * InpText.water()).charge == 0
end

@testitem "inp: value counts" setup = [InpText] begin
    using QuantumEngine
    w = InpText.water()
    @test_throws "`charge` takes 1 value, got 2" InpText.parse("charge 0 1\n" * w)
    @test_throws "`charge` takes 1 value, got 0" InpText.parse("charge\n" * w)
    @test_throws "`field` takes 3 values, got 2" InpText.parse("field 0.1 0.2\n" * w)
    @test_throws "`freeze_bond` takes 2 or 3 values, got 1" InpText.parse("freeze_bond 1\n" * w)
    @test_throws "`freeze_dihedral` takes 4 or 5 values, got 6" InpText.parse("freeze_dihedral 0 1 2 3 4.0 5\n" * w)
    @test_throws "`kpath` needs a value" InpText.parse("kpath\n" * w)
    @test_throws "`title` needs a value" InpText.parse("title\n" * w)
    @test_throws "`optimize` takes 0 values, got 1" InpText.parse("optimize now\n" * w)
    @test_throws "`geometry` takes 0 or 1 values, got 2" InpText.parse(InpText.water("geometry bohr extra"))
    for kw in ("hubbard", "cell", "pbc", "oniom")
        @test_throws "`$kw` takes no value on its line" InpText.parse("$kw x\nend\n" * w)
    end
    @test_throws "`method` takes 3 values, got 2" InpText.parse("method libxc_gga 101\n" * w)
    @test_throws "`method` takes 1 value, got 2" InpText.parse("method pbe0 d3\n" * w)
    @test_throws "`method` needs a value" InpText.parse("method\n" * w)
    # A missing Boolean value is `:bool_fallback` in lenient mode; other missing values are `:missing_value`.
    for (kw, case) in (("zora", :bool_fallback), ("ecp", :bool_fallback), ("gpu", :bool_fallback), ("export_cube", :bool_fallback), ("checkpoint", :missing_value))
        @test_throws "`$kw` takes 1 value, got 0" InpText.parse("$kw\n" * w)
        p = parse_inp("$kw\n" * w; source = "t.inp", lenient = true)
        @test [x.case for x in p.warnings] == [case]
    end
end

@testitem "inp: value types" setup = [InpText] begin
    using QuantumEngine
    using QuantumEngine.Input
    w = InpText.water()
    # Booleans are exactly `true` or `false`.
    @test_throws "`zora`: expected true or false, got \"yes\"" InpText.parse("zora yes\n" * w)
    @test_throws "`zora`: expected true or false, got \"True\"" InpText.parse("zora True\n" * w)
    @test InpText.parse("zora true\nri_j false\n" * w).zora
    # Integers are decimal.
    @test_throws "`charge`: expected an integer, got \"x\"" InpText.parse("charge x\n" * w)
    @test_throws "`charge`: expected an integer, got \"0x1\"" InpText.parse("charge 0x1\n" * w)
    @test_throws "`max_iterations`: expected an integer, got \"1.0\"" InpText.parse("max_iterations 1.0\n" * w)
    @test InpText.parse("charge +0\nmax_iterations 75\n" * w).max_iterations == 75
    # Floats are finite decimals.
    @test_throws "`damping`: expected a number, got \"abc\"" InpText.parse("damping abc\n" * w)
    @test_throws "`damping`: expected a number, got \"0x1p-2\"" InpText.parse("damping 0x1p-2\n" * w)
    @test_throws "`scf_convergence`: expected a finite number, got \"nan\"" InpText.parse("scf_convergence nan\n" * w)
    @test_throws "`damping`: expected a finite number, got \"inf\"" InpText.parse("damping inf\n" * w)
    @test_throws "`geometry`: expected a finite number, got \"NaN\"" InpText.parse("geometry\nH 0 0 NaN\nH 0 0 1\nend\n")
    @test_throws "`geometry`: expected a number, got \"x\"" InpText.parse("geometry\nH 0 0 x\nH 0 0 1\nend\n")
    @test_throws "`method`: expected a finite number, got \"nan\"" InpText.parse("method libxc_hgga 402 nan\n" * w)
    @test InpText.parse("damping 0.25\nscf_convergence 1E-8\n" * w).damping == 0.25
    # Out of range: an overflow is non-finite, an underflow is 0.0, as Rust reads them.
    @test_throws "`scf_convergence`: expected a finite number, got \"1e500\"" InpText.parse("scf_convergence 1e500\n" * w)
    @test_throws "`geometry`: expected a finite number, got \"1e500\"" InpText.parse("geometry\nO 0 0 1e500\nH 0 0 1\nH 0 1 0\nend\n")
    @test InpText.parse("damping 1e-400\n" * w).damping === 0.0
    p = parse_inp("scf_convergence 1e500\n" * w; source = "t.inp", lenient = true)
    @test p.input.scf_convergence == Inf
    @test [x.case for x in p.warnings] == [:nonfinite]
    # Words go through the vocabulary of their keyword.
    @test_throws "unknown value `tpu` for `compute_mode`; accepted: cpu, gpu, hybrid" InpText.parse("compute_mode tpu\n" * w)
    @test_throws "unknown value `fast` for `run`; accepted: sp, single_point" InpText.parse("run fast\n" * w)
    @test_throws "unknown value `zzz` for `method`; accepted: rhf, uhf, hf" InpText.parse("method zzz\n" * w)
    @test_throws "unknown value `pbe` for `population`; accepted: mulliken, lowdin" InpText.parse("population mulliken pbe\n" * w)
    # Each keyword stores its value.
    inp = InpText.parse(
        """
        run_mode freq
        basis 6-31G*
        diis_method EDIIS
        conv_tol_density 1e-7
        diis_reset 0.2
        smear 0.01
        grid fine
        level_shift 0.3
        autostabilize false
        robust true
        electric_field 0.0 0.0 0.01
        density_fitting true
        ri_k true
        aux_basis def2-SVP-JKFIT
        compute_mode Hybrid
        checkpoint run.chk
        restart true
        ts_target_mode 2
        ts_trust_radius 0.2
        neb_k 0.3
        neb_images 9
        climbing_image false
        temp 310.0
        press 2.0
        sigma 2
        dispersion true
        zora_relativistic true
        nbo true
        admm true
        admm_basis cfit
        ecp_name def2-ecp
        vdw_kernel d3bj
        output out.log
        verbosity 2
        export_molden true
        export_cube true
        export_wfx true
        pop_analysis lowdin
        cutoff 400
        kgrid 2 3 4
        dos_kgrid 5 6 7
        dos_sigma 0.2
        k_path G X M
        freeze 0 2
        freeze_bond 0 1 1.8
        freeze_angle 1 0 2 104.5
        freeze_torsion 0 1 2 3
        irc_step 0.05
        irc_points 12
        irc_direction fwd
        nstates 5
        tddft_method sf
        soc true
        triplets true
        use_dmft true
        dmft_temp 100.0
        dmft_matsubara 512
        dmft_bath 6
        """ * w
    )
    @test inp.run_mode == Frequency && inp.basis == "6-31g*" && inp.diis_method == EDIIS
    @test (inp.conv_tol_density, inp.diis_reset_threshold, inp.smearing, inp.grid_level, inp.level_shift) == (1.0e-7, 0.2, 0.01, 2, 0.3)
    @test (inp.auto_stabilize, inp.robust_convergence, inp.electric_field) == (false, true, [0.0, 0.0, 0.01])
    @test (inp.use_ri_j, inp.use_ri_k, inp.aux_basis, inp.compute_mode) == (true, true, "def2-SVP-JKFIT", Hybrid)
    @test (inp.checkpoint, inp.load_checkpoint, inp.ts_target_mode, inp.ts_trust_radius) == ("run.chk", true, 2, 0.2)
    @test (inp.neb_k, inp.neb_images, inp.climbing_image) == (0.3, 9, false)
    @test (inp.temperature, inp.pressure, inp.symmetry_number) == (310.0, 2.0, 2.0)
    @test (inp.dispersion, inp.zora, inp.nbo, inp.admm, inp.admm_basis) == (true, true, true, true, "cfit")
    @test (inp.use_ecp, inp.ecp_name, inp.vdw_kernel) == (true, "def2-ecp", D3BJ)
    @test (inp.output, inp.verbosity, inp.export_molden, inp.export_cube, inp.export_wfx) == ("out.log", 2, true, true, true)
    @test inp.population == [Lowdin]
    @test (inp.pw_cutoff, inp.kpoints, inp.dos_k_grid, inp.dos_sigma, inp.kpath) == (400.0, (2, 3, 4), (5, 6, 7), 0.2, ["G", "X", "M"])
    # Atom indices are 0-based in the file and 1-based in `InputFile`; angles are read in degrees.
    @test inp.constraints == [
        Constraint(FreezeAtom, (1, 0, 0, 0), nothing), Constraint(FreezeAtom, (3, 0, 0, 0), nothing),
        Constraint(FreezeBond, (1, 2, 0, 0), 1.8), Constraint(FreezeAngle, (2, 1, 3, 0), deg2rad(104.5)),
        Constraint(FreezeDihedral, (1, 2, 3, 4), nothing),
    ]
    @test (inp.irc_step_size, inp.irc_max_points, inp.irc_direction) == (0.05, 12, Forward)
    @test (inp.tddft_states, inp.tddft_method, inp.compute_soc, inp.tddft_triplets) == (5, SpinFlip, true, true)
    @test (inp.use_dmft, inp.dmft_temperature, inp.dmft_n_freq, inp.dmft_n_bath) == (true, 100.0, 512, 6)
    @test InpText.parse("gpu true\n" * w).compute_mode == GPU
    @test InpText.parse("gpu false\n" * w).compute_mode == CPU
    @test InpText.parse("run optimization\n" * w).run_mode == Optimization
    @test InpText.parse("optimize\n" * w).run_mode == Optimization
    @test InpText.parse("grid 3\n" * w).grid_level == 3
    @test_throws "unknown value `9` for `grid`" InpText.parse("grid 9\n" * w)
    @test_throws "`freeze_atom`: expected an atom index (an integer ≥ 0), got \"-1\"" InpText.parse("freeze_atom -1\n" * w)
    @test_throws "`freeze_bond`: expected a number, got \"x\"" InpText.parse("freeze_bond 0 1 x\n" * w)
    @test_throws "`kpoints`: expected an integer, got \"x\"" InpText.parse("kpoints 2 2 x\n" * w)
end

@testitem "inp: duplicates" setup = [InpText] begin
    using QuantumEngine
    w = InpText.water()
    @test_throws "input: t.inp:9: keyword `charge` given twice (lines 2 and 9)" InpText.parse("# c\ncharge 0\n" * w * "basis sto-3g\ncharge 1\n")
    # Aliases name one keyword.
    @test_throws "keyword `temperature` given twice (lines 1 and 2)" InpText.parse("temp 300\ntemperature 310\n" * w)
    @test_throws "keyword `run` given twice (lines 1 and 2)" InpText.parse("optimize\nrun sp\n" * w)
    @test_throws "keyword `gpu` given twice (lines 1 and 2)" InpText.parse("compute_mode cpu\ngpu true\n" * w)
    # The `freeze_*` keywords are repeatable.
    @test length(InpText.parse("freeze_atom 0\nfreeze_atom 1\nfreeze_bond 0 1\nfreeze_bond 1 2\n" * w).constraints) == 4
    second = "charge 0\nmultiplicity 1\nbasis sto-3g\n" * InpText.water() * InpText.water()
    @test_throws "input: t.inp:9: second geometry block (first at line 4)" InpText.parse(second)
    second = "\ncharge 0\nproduct_geometry\nH 0 0 0\nH 0 0 1\nend\n" * w * "product_geometry\nH 0 0 0\nH 0 0 1\nend\n"
    @test_throws "input: t.inp:12: second product_geometry block (first at line 3)" InpText.parse(second)
    @test_throws "keyword `cell` given twice" InpText.parse("cell\n1 0 0\n0 1 0\n0 0 1\nend\ncell\n1 0 0\n0 1 0\n0 0 1\nend\n" * w)
    @test_throws "`hubbard` gives `o` twice (lines 2 and 3)" InpText.parse("hubbard\nO 4.0\no 5.0\nend\n" * w)
end

@testitem "inp: blocks" setup = [InpText] begin
    using QuantumEngine
    @test_throws "input: t.inp:3: `charge` inside the geometry block opened at line 1; close the block with `end`" InpText.parse("geometry\nH 0 0 0\ncharge 0\nH 0 0 1\nend\n")
    @test_throws "`END_GEOMETRY` inside the geometry block opened at line 1" InpText.parse("geometry\nH 0 0 0\nH 0 0 1\nEND_GEOMETRY\n")
    @test_throws "`H` inside the geometry block opened at line 1" InpText.parse("geometry\nH 0 0\nend\n")
    @test_throws "input: t.inp:2: a geometry line takes 4 tokens (a symbol and 3 coordinates), got 5" InpText.parse("geometry\nH 0 0 0 1\nH 0 0 1\nend\n")
    @test_throws "a Bq line takes 5 tokens (Bq, 3 coordinates and a charge), got 6" InpText.parse("geometry\nH 0 0 0\nH 0 0 1\nBq 0 0 0 1 2\nend\n")
    @test_throws "input: t.inp:2: the geometry block opened at line 2 is not closed by `end`" InpText.parse("\ngeometry\nH 0 0 0\nH 0 0 1\n")
    @test_throws "the hubbard block opened at line 1 is not closed by `end`" InpText.parse("hubbard\nO 4.0\n")
    @test_throws "`end` closes no block" InpText.parse(InpText.water() * "end\n")
    @test_throws "`end` takes no value" InpText.parse("geometry\nH 0 0 0\nH 0 0 1\nend geometry\n")
    @test_throws "`basis` inside the oniom block opened at line 1" InpText.parse("oniom\nbasis sto-3g\nend\n" * InpText.water())
    @test_throws "`zzz` inside the oniom block opened at line 1" InpText.parse("oniom\nzzz 1\nend\n" * InpText.water())
    @test_throws "`0.5` inside the charges block opened at line 1" InpText.parse("charges\n0.5 0 0\nend\n" * InpText.water())
    @test_throws "`charges`: expected a number, got \"q\"" InpText.parse("charges\nq 0 0 0\nend\n" * InpText.water())
    @test_throws "a charges line takes 4 tokens (a charge and 3 coordinates), got 5" InpText.parse("charges\n1 0 0 0 0\nend\n" * InpText.water())
    @test_throws "`O` inside the hubbard block opened at line 1" InpText.parse("hubbard\nO\nend\n" * InpText.water())
    @test_throws "a hubbard line takes 2 tokens (a symbol and U), got 3" InpText.parse("hubbard\nO 4.0 1\nend\n" * InpText.water())
    @test_throws "`hubbard`: expected a number, got \"x\"" InpText.parse("hubbard\nO x\nend\n" * InpText.water())
    @test_throws "a cell line takes 3 numbers, got 2" InpText.parse("cell\n1 0\nend\n" * InpText.water())
    @test_throws "the cell block takes 3 lines" InpText.parse("cell\n1 0 0\n0 1 0\n0 0 1\n1 1 1\nend\n" * InpText.water())
    @test_throws "the cell block opened at line 1 has 2 lines; it takes 3" InpText.parse("cell\n1 0 0\n0 1 0\nend\n" * InpText.water())
    @test_throws "`cell`: expected a number, got \"x\"" InpText.parse("cell\n1 0 x\n0 1 0\n0 0 1\nend\n" * InpText.water())
    @test_throws "a pbc line takes 3 values, got 2" InpText.parse("pbc\ntrue false\nend\n" * InpText.water())
    @test_throws "the pbc block takes 1 line" InpText.parse("pbc\ntrue true true\ntrue true true\nend\n" * InpText.water())
    @test_throws "the pbc block opened at line 1 has no line; it takes 1" InpText.parse("pbc\nend\n" * InpText.water())
    @test_throws "`pbc`: expected true or false, got \"1\"" InpText.parse("pbc\n1 T F\nend\n" * InpText.water())
    @test_throws "`fragments`: an empty fragment" InpText.parse(InpText.water() * "fragments\n0,1;\nend\n")
    @test_throws "fragment index `x` is not an integer" InpText.parse(InpText.water() * "fragments 0,x;1,2\n")
    @test_throws "fragment index `-1` is not an integer ≥ 0" InpText.parse(InpText.water() * "fragments 0,-1;1,2\n")
    @test_throws "`fragments` on its line needs at least 2 fragments, got 1" InpText.parse(InpText.water() * "fragments 0,1\n3\nend\n")
    # The ONIOM block and keywords.
    inp = InpText.parse("oniom\nhigh pbe0\nlow hf\nmodel_atoms 0 1\nmodel_charge -1\nmodel_multiplicity 2\nend\n" * InpText.water())
    o = inp.oniom
    @test (o.high.kind, o.low.kind, o.model_atoms, o.model_charge, o.model_multiplicity) == (QuantumEngine.Input.PBE0, QuantumEngine.Input.HF, [1, 2], -1, 2)
    inp = InpText.parse("oniom_high b3lyp\noniom_model 2\noniom_model_charge 0\noniom_model_multiplicity 1\n" * InpText.water())
    o = inp.oniom
    @test (o.high.kind, o.low.kind, o.model_atoms, o.model_charge, o.model_multiplicity) == (QuantumEngine.Input.B3LYP, QuantumEngine.Input.HF, [3], 0, 1)
    @test InpText.parse("oniom_low pbe\n" * InpText.water()).oniom.low.kind == QuantumEngine.Input.PBE
    @test_throws "keyword `high` given twice (lines 2 and 3)" InpText.parse("oniom\nhigh pbe0\nhigh pbe\nend\n" * InpText.water())
    @test_throws "keyword `high` given twice (lines 1 and 3)" InpText.parse("oniom_high pbe0\noniom\nhigh pbe\nend\n" * InpText.water())
    @test isnothing(InpText.parse(InpText.water()).oniom)
end

@testitem "inp: no atoms" setup = [InpText] begin
    using QuantumEngine
    for text in ("", "\n\n", "# only a comment\n# another\n", "charge 0\nmultiplicity 1\nbasis sto-3g\n", "geometry\nend\n", "charges\n1 0 0 0\nend\n")
        @test_throws "input: t.inp: no atoms: the input has no geometry block" InpText.parse(text)
    end
end

@testitem "inp: methods and references" setup = [InpText] begin
    using QuantumEngine
    using QuantumEngine.Input
    w = InpText.water()
    @test InpText.parse(w).method == MethodSpec(HF, AutoReference, (0, 0), 0.0)
    @test InpText.parse("method UHF\n" * w).method == MethodSpec(HF, Unrestricted, (0, 0), 0.0)
    @test InpText.parse("method casscf(4,4)\n" * w).method == MethodSpec(CASSCF, AutoReference, (4, 4), 0.0)
    @test InpText.parse("method libxc_gga 101 130\n" * w).method == MethodSpec(LIBXC_GGA, AutoReference, (101, 130), 0.0)
    @test InpText.parse("method libxc_hgga 402 0.2\n" * w).method == MethodSpec(LIBXC_HGGA, AutoReference, (402, 0), 0.2)
    @test_throws "`method`: `libxc_gga` takes 2 integer Libxc IDs" InpText.parse("method libxc_gga x 130\n" * w)
    @test_throws "`method`: `libxc_hgga` takes an integer Libxc ID and the exact-exchange fraction" InpText.parse("method libxc_hgga 402 x\n" * w)
    @test_throws "casscf needs (electrons,orbitals), got `casscf(4)`" InpText.parse("method casscf(4)\n" * w)
    # A restricted reference needs a singlet; the error names the method line.
    o2 = "geometry\nO 0 0 0\nO 0 0 1.208\nend\n"
    for word in ("rhf", "rmp2", "rccsd", "rmp2-f12", "rccsd-f12")
        @test_throws "input: t.inp:2: method `$word` requests a restricted reference, but multiplicity is 3" InpText.parse("multiplicity 3\nmethod $word\n" * o2)
    end
    @test InpText.parse("multiplicity 3\nmethod uhf\n" * o2).method.reference == Unrestricted
    @test InpText.parse("multiplicity 3\n" * o2).method.reference == AutoReference
    # The electron parity is checked at the multiplicity line, or at line 0 without one.
    @test_throws "input: t.inp:1: system: 16 electrons cannot have multiplicity 2" InpText.parse("multiplicity 2\n" * o2)
    @test_throws "input: t.inp: system: 1 electrons cannot have multiplicity 1" InpText.parse("geometry\nH 0 0 0\nend\n")
    @test_throws "input: t.inp: system: charge 3 exceeds the nuclear charge 2" InpText.parse("charge 3\ngeometry\nH 0 0 0\nH 0 0 1\nend\n")
    @test_throws "input: t.inp:1: system: multiplicity must be at least 1, got 0" InpText.parse("multiplicity 0\n" * w)
    @test InpText.parse("charge 1\nmultiplicity 2\ngeometry\nH 0 0 0\nH 0 0 1\nend\n").multiplicity == 2
end

@testitem "inp: solvation" setup = [InpText] begin
    using QuantumEngine
    using QuantumEngine.Input
    w = InpText.water()
    both(a, b) = (InpText.parse(a * "\n" * b * "\n" * w).solvation, InpText.parse(b * "\n" * a * "\n" * w).solvation)
    # Each pair gives one result in both line orders.
    s1, s2 = both("epsilon 4.0", "solvation pcm")
    @test s1 == s2 && (s1.model, s1.epsilon, s1.custom_epsilon) == (PCM, 4.0, false)
    s1, s2 = both("solvation smd", "solvent dcm")
    @test s1 == s2 && (s1.model, s1.solvent, s1.custom_epsilon) == (SMD, :dcm, false)
    s1, s2 = both("solvation smd\nsolvent dcm", "epsilon 9.0")
    @test s1 == s2 && (s1.model, s1.solvent, s1.epsilon, s1.custom_epsilon) == (SMD, :dcm, 9.0, true)
    s1, s2 = both("pcm_radii 1.1", "solvation pcm")
    @test s1 == s2 && (s1.model, s1.radii_scaling) == (PCM, 1.1)
    s1, s2 = both("pcm_grid 194", "solvation smd")
    @test s1 == s2 && (s1.model, s1.grid_order) == (SMD, 194)
    # Without `solvation`, `solvent` gives SMD and `epsilon` alone gives PCM.
    s = InpText.parse("solvent Di-chloro_methane\n" * w).solvation
    @test (s.model, s.solvent) == (SMD, :dcm)
    s = InpText.parse("dielectric 2.5\n" * w).solvation
    @test (s.model, s.epsilon, s.custom_epsilon) == (PCM, 2.5, false)
    s = InpText.parse("solvation true\n" * w).solvation
    @test s == SolvationSpec(; model = PCM)
    @test InpText.parse("solvation none\n" * w).solvation == SolvationSpec()
    @test InpText.parse(w).solvation == SolvationSpec()
    # Errors.
    @test_throws "input: t.inp:2: `solvent` needs `solvation smd`" InpText.parse("solvation pcm\nsolvent dcm\n" * w)
    @test_throws "input: t.inp:1: `solvent` needs `solvation smd`" InpText.parse("solvent dcm\nsolvation true\n" * w)
    @test_throws "input: t.inp:2: `epsilon` is given with `solvation none`" InpText.parse("solvation false\nepsilon 4\n" * w)
    @test_throws "`solvent` is given with `solvation none`" InpText.parse("solvent dcm\nsolvation none\n" * w)
    @test_throws "input: t.inp:2: `pcm_radii` needs `solvation pcm`" InpText.parse("solvation smd\npcm_radii 1.1\n" * w)
    @test_throws "input: t.inp:1: `pcm_radii` needs `solvation pcm`" InpText.parse("pcm_radii 1.1\n" * w)
    @test_throws "`pcm_grid` needs `solvation pcm`" InpText.parse("smd_grid 194\n" * w)
    @test_throws "unknown value `cosmo` for `solvation`; accepted: pcm, true, smd, none, false" InpText.parse("solvation cosmo\n" * w)
    @test_throws "unknown value `brine` for `solvent`" InpText.parse("solvent brine\n" * w)
end

@testitem "inp: f12 options" setup = [InpText] begin
    using QuantumEngine
    using QuantumEngine.Input
    w = InpText.water()
    inp = InpText.parse("f12 true\nf12_gamma 0.9\nf12_model F12a\n" * w)
    @test (inp.use_f12, inp.f12_gamma, inp.f12_model) == (true, 0.9, "F12a")
    for m in ("mp2-f12", "ccsd-f12", "ccsd(t)-f12")
        inp = InpText.parse("method $m\ngamma 1.1\nf12_approximation F12a\n" * w)
        @test (inp.use_f12, inp.f12_gamma, inp.f12_model) == (false, 1.1, "F12a")
    end
    @test_throws "input: t.inp:1: `f12_gamma` needs `f12 true` or an F12 method" InpText.parse("f12_gamma 0.9\n" * w)
    @test_throws "input: t.inp:2: `f12_model` needs `f12 true` or an F12 method" InpText.parse("f12 false\nf12_model F12a\nmethod mp2\n" * w)
end

@testitem "inp: type confusion" setup = [InpText, Fixtures] begin
    using QuantumEngine
    msg = "`basis:` looks like YAML; .inp files use `keyword value` lines, and YAML input needs a .yaml or .yml extension"
    @test_throws msg InpText.parse("basis: sto-3g\n" * InpText.water())
    @test_throws "input: t.inp:2: `-` looks like YAML" InpText.parse("geometry\n- symbol: H\nend\n")
    @test_throws "input: t.inp:4: `molecule:` looks like YAML" InpText.parse(Fixtures.oracle_input("molecules/h2_ccsd.inp"))
end

@testitem "inp: line endings" setup = [InpText, Fixtures] begin
    using QuantumEngine
    lf = "title water\ncharge 0\n" * InpText.water() * "basis sto-3g\n"
    ref = InpText.parse(lf)
    for text in (
            replace(lf, "\n" => "\r\n"), "﻿" * lf, "﻿" * replace(lf, "\n" => "\r\n"),
            replace(lf, " " => "\t"), replace(lf, "  " => " \t "), "  " * replace(lf, "\n" => "  \n\t"),
        )
        @test Fixtures.fields_equal(InpText.parse(text), ref)
    end
    # A byte-order mark is removed from the first line only.
    @test_throws "unknown keyword `﻿charge`" InpText.parse("title t\n﻿charge 0\n" * InpText.water())
end

@testitem "inp: comments, title and quotes" setup = [InpText] begin
    using QuantumEngine
    w = InpText.water()
    inp = InpText.parse("charge 0 # neutral\nmultiplicity 1 #singlet\n" * w)
    @test (inp.charge, inp.multiplicity) == (0, 1)
    @test InpText.parse("# just a comment line\n" * replace(w, "end" => "end # geometry")).atoms == InpText.parse(w).atoms
    @test InpText.parse("title \"H2O MP2-F12 Test\"\n" * w).title == "H2O MP2-F12 Test"
    @test InpText.parse("title  H2O   MP2-F12\tTest # comment\n" * w).title == "H2O MP2-F12 Test"
    @test InpText.parse("title \"a\" \"b\"\n" * w).title == "a\" \"b"
    @test InpText.parse("title \"\"x\"\"\n" * w).title == "\"x\""
    @test InpText.parse("# first\n# second\n" * w).title == "first"
    @test InpText.parse("#first\n" * w).title == "first"
    @test InpText.parse("# first\ntitle second\n" * w).title == "second"
    @test InpText.parse("title second\n# first\n" * w).title == "second"
    @test InpText.parse(w * "#  late #title\n").title == "late #title"
    @test InpText.parse("#\n# second\n" * w).title == "second"
    @test InpText.parse("title \"\"\n# c\n" * w).title == ""
    @test_throws "keyword `title` given twice" InpText.parse("title a\ntitle b\n" * w)
end

@testitem "inp: atom lines and blocks" setup = [InpText, Fixtures] begin
    using QuantumEngine
    using StaticArrays: SVector
    using QuantumEngine.Input
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    w = InpText.water()
    inp = InpText.parse(replace(w, "end" => "Bq 0 0 1 0.5\nbq 0 0 2 -0.5\nend"))
    @test length(inp.atoms) == 3
    @test inp.point_charges == [
        PointCharge(0.5, SVector(0.0, 0.0, ANGSTROM_TO_BOHR)), PointCharge(-0.5, SVector(0.0, 0.0, 2 * ANGSTROM_TO_BOHR)),
    ]
    # Ghost atoms.
    inp = InpText.parse(replace(w, "end" => "O@ 0 0 3\nBqH 0 0 4\nend"))
    @test [(a.symbol, a.Z, a.ghost) for a in inp.atoms[4:5]] == [("O@", 0, true), ("BqH", 0, true)]
    @test_throws "input: t.inp:2: unknown element symbol \"Xx\"" InpText.parse("geometry\nXx 0 0 0\nH 0 0 1\nend\n")
    @test_throws "input: t.inp:5: unknown element symbol \"Foo@\"" InpText.parse(replace(w, "end" => "Foo@ 0 0 3\nend"))
    @test_throws "input: t.inp:5: a Bq atom needs an element (BqO, O@); a point charge is `Bq x y z q`" InpText.parse(replace(w, "end" => "Bq 0 0 3\nend"))
    # The cell block holds the lattice vectors as rows, in Å whatever the geometry unit.
    inp = InpText.parse("cell\n5.0 0 0\n0.5 6.0 0\n0 0 7.0\nend\npbc\ntrue false true\nend\n" * InpText.water("geometry bohr"))
    @test inp.lattice == [5.0 0 0; 0.5 6.0 0; 0 0 7.0]
    @test inp.lattice[2, :] == [0.5, 6.0, 0.0]
    @test inp.pbc == (true, false, true)
    # The fragments block of an oracle test input, 0-based in the file.
    inp = parse_inp(Fixtures.oracle_input("tests/stress_dimer_bsse.in"); source = "stress_dimer_bsse.in").input
    @test inp.fragments == [[1, 2, 3], [4, 5, 6]]
    @test (inp.title, inp.run_mode, inp.method.kind, inp.basis, inp.dispersion) == ("H2O Dimer BSSE Stress Test", BSSE, B3LYP, "6-31g", true)
    @test InpText.parse(w * "fragments 0,1 ; 2\n").fragments == [[1, 2], [3]]
    # `ecp` takes true, false or a name.
    inp = InpText.parse("ecp def2\n" * w)
    @test (inp.use_ecp, inp.ecp_name) == (true, "def2")
    @test (InpText.parse("ecp true\n" * w).use_ecp, InpText.parse("use_ecp false\n" * w).use_ecp) == (true, false)
    @test isnothing(InpText.parse("ecp true\n" * w).ecp_name)
    # The hubbard block.
    inp = InpText.parse("hubbard\nO 4.0\nFe 5.5\nend\n" * w)
    @test inp.hubbard == [HubbardU("O", 4.0), HubbardU("Fe", 5.5)]
    @test_throws "input: t.inp:1: `hubbard` takes no value on its line" InpText.parse("hubbard O:5.0\nend\n" * w)
end

@testitem "inp: class B readings" setup = [InpText] begin
    using QuantumEngine
    # The oracle scans the whole text for the first `cell` and the first `pbc`, wherever they occur
    # (`parser.rs:622-657`): after the `cell` in "supercell" it reads the first three lines with
    # exactly three numbers as the lattice, here the coordinates of the first three atoms (an atom
    # line's symbol is not a number), and after the `pbc` of the title it reads the first line with
    # three or more tokens, here the first atom line, as the three flags, all false. Neither text
    # opens a block here.
    text = "title supercell pbc test\ngeometry angstrom\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\nH 0.0 0.0 1.48\nH 0.0 0.0 2.22\nend\n"
    for lenient in (false, true)
        p = parse_inp(text; source = "t.inp", lenient)
        @test isempty(p.warnings)
        @test isnothing(p.input.lattice) && p.input.pbc == (true, true, true)
        @test p.input.title == "supercell pbc test"
    end
    # A multi-line fragments block appends its lines; the oracle keeps only the last
    # (`parser.rs:575-582` replaces `input.fragments` on each line).
    text = InpText.water() * "fragments\n0,1\n2\nend\n"
    for lenient in (false, true)
        p = parse_inp(text; source = "t.inp", lenient)
        @test isempty(p.warnings)
        @test p.input.fragments == [[1, 2], [3]]
    end
    text = "geometry\nO 0 0 0\nH 0 0 1\nH 0 1 0\nO 0 0 3\nH 0 0 4\nH 0 1 3\nend\nfragments\n0,1,2\n3,4,5\nend\n"
    for lenient in (false, true)
        p = parse_inp(text; source = "t.inp", lenient)
        @test isempty(p.warnings)
        @test p.input.fragments == [[1, 2, 3], [4, 5, 6]]
    end
end

@testitem "inp: juliac entry" setup = [Fixtures] begin
    using QuantumEngine
    root = pkgdir(QuantumEngine)
    entry = joinpath(root, "juliac", "entry.jl")
    dir = mktempdir()
    h2 = joinpath(dir, "h2.inp")
    write(h2, Fixtures.oracle_input("examples/h2.inp"))
    env = ("JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing)
    cmd = addenv(`$(Base.julia_cmd()) --startup-file=no --project=$root $entry $h2`, env...)
    out = read(cmd, String)
    @test out == join(["Hydrogen molecule", "2", "1", "1", QuantumEngine.IO.fmt_fixed(1 / 1.4, 20, 12)], "\n") * "\n"
    @test split(out, '\n'; keepempty = false) |> length == 5
    bad = joinpath(dir, "bad.inp")
    write(bad, "charge 0\n")
    p = run(pipeline(ignorestatus(addenv(`$(Base.julia_cmd()) --startup-file=no --project=$root $entry $bad`, env...)); stdout = joinpath(dir, "out.txt")))
    @test p.exitcode == 1
    @test read(joinpath(dir, "out.txt"), String) == "input: $bad: no atoms: the input has no geometry block\n"
    # Two atoms at one position pass the parser; the nuclear repulsion rejects them.
    coinc = joinpath(dir, "coinc.inp")
    write(coinc, "geometry\nH 0 0 0\nH 0 0 0\nend\n")
    p = run(pipeline(ignorestatus(addenv(`$(Base.julia_cmd()) --startup-file=no --project=$root $entry $coinc`, env...)); stdout = joinpath(dir, "out.txt")))
    @test p.exitcode == 1
    @test read(joinpath(dir, "out.txt"), String) == "input: $coinc: system: atoms 1 and 2 coincide\n"
end
