# Each input the oracle (`quantum_engine` at 4a78a1d) accepts silently: strict mode throws, and
# lenient mode gives the oracle's value with one warning of the case. The `parser.rs` lines cited
# are those of the oracle's `src/input/parser.rs`.

@testmodule Lenient begin
    using QuantumEngine

    # The water geometry, closed by `end`.
    const W = "geometry\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\nend\n"

    strict(text) = parse_inp(text; source = "t.inp").input
    lenient(text) = parse_inp(text; source = "t.inp", lenient = true)
    cases(p) = [w.case for w in p.warnings]
end

@testitem "lenient: per-keyword table" setup = [Lenient] begin
    using QuantumEngine
    using QuantumEngine.Input
    using StaticArrays: SVector
    # One row per field: the keyword and its aliases, the lines around the keyword line (its
    # context), the field read, the lenient value for the bad input and its case, and the lenient
    # value for the missing value and its case. `:error` is an oracle error, an error in both
    # modes; `:none` is no such input. The bad input is the keyword followed by `bad`.
    E = :error
    N = :none
    oniom = ("oniom\n", "end\n")
    plain = ("", "")
    pcm = ("solvation pcm\n", "")
    f12 = ("f12 true\n", "")
    const_atoms(i) = [c.atoms[1] for c in i.constraints]
    rows = [
        # parser.rs:61-83 (an unknown mode is `Err`), 84-88
        (["run", "run_mode"], plain, i -> i.run_mode, "zzz", E, nothing, SinglePoint, :missing_value),
        (["title"], plain, i -> i.title, N, nothing, nothing, "", :missing_value),
        # parser.rs:89-110, 270-278, 539-547: `ok_or(...)?`
        (
            [
                "charge", "multiplicity", "scf_convergence", "max_iterations", "ts_target_mode", "ts_trust_radius",
                "neb_k", "neb_images", "temperature", "temp", "pressure", "press", "symmetry_number", "sigma",
            ], plain, identity, "zzz", E, nothing, E, nothing,
        ),
        # parser.rs:377-382, 383-409, 1079-1167
        (["basis"], plain, identity, N, nothing, nothing, E, nothing),
        (["method", "oniom_high", "oniom_low"], plain, identity, "zzz", E, nothing, E, nothing),
        (["high", "low"], oniom, identity, "zzz", E, nothing, E, nothing),
        # parser.rs:307-330
        (["geometry", "product_geometry"], ("", "end\n"), identity, "zzz", E, nothing, N, nothing),
        # parser.rs:101, 113-122, 140-150, 158, 185-190, 521, 554: `unwrap_or(true)`
        (["climbing_image"], plain, i -> i.climbing_image, "zzz", true, :bool_fallback, true, :bool_fallback),
        (["density_fitting", "ri_j"], plain, i -> i.use_ri_j, "zzz", true, :bool_fallback, true, :bool_fallback),
        (["ri_k"], plain, i -> i.use_ri_k, "zzz", true, :bool_fallback, true, :bool_fallback),
        (["gpu"], plain, i -> i.compute_mode, "zzz", GPU, :bool_fallback, GPU, :bool_fallback),
        (["load_checkpoint", "restart"], plain, i -> i.load_checkpoint, "zzz", true, :bool_fallback, true, :bool_fallback),
        (["dispersion"], plain, i -> i.dispersion, "zzz", true, :bool_fallback, true, :bool_fallback),
        (["zora", "zora_relativistic"], plain, i -> i.zora, "zzz", true, :bool_fallback, true, :bool_fallback),
        (["f12", "explicit_correlation"], plain, i -> i.use_f12, "zzz", true, :bool_fallback, true, :bool_fallback),
        (["nbo"], plain, i -> i.nbo, "zzz", true, :bool_fallback, true, :bool_fallback),
        (["auto_stabilize", "autostabilize"], plain, i -> i.auto_stabilize, "zzz", true, :bool_fallback, true, :bool_fallback),
        (["robust_convergence", "robust"], plain, i -> i.robust_convergence, "zzz", true, :bool_fallback, true, :bool_fallback),
        (["dmft", "use_dmft"], plain, i -> i.use_dmft, "zzz", true, :bool_fallback, true, :bool_fallback),
        (["admm"], plain, i -> i.admm, "zzz", true, :bool_fallback, true, :bool_fallback),
        # parser.rs:258-266: `unwrap_or(false)`
        (["export_molden"], plain, i -> i.export_molden, "zzz", false, :bool_fallback, false, :bool_fallback),
        (["export_cube"], plain, i -> i.export_cube, "zzz", false, :bool_fallback, false, :bool_fallback),
        (["export_wfx"], plain, i -> i.export_wfx, "zzz", false, :bool_fallback, false, :bool_fallback),
        # parser.rs:515-520: `s == "true" || s == "1"`, `unwrap_or(true)`
        (["compute_soc", "soc"], plain, i -> i.compute_soc, "zzz", false, :bool_fallback, true, :bool_fallback),
        (["tddft_triplets", "triplets"], plain, i -> i.tddft_triplets, "zzz", false, :bool_fallback, true, :bool_fallback),
        # parser.rs:126-136: an unknown word matches `_ => {}`
        (["compute_mode"], plain, i -> i.compute_mode, "zzz", CPU, :unknown_value, CPU, :missing_value),
        # parser.rs:137, 267, 557: `parts.get(1).map(...)`
        (["checkpoint"], plain, i -> i.checkpoint, N, nothing, nothing, nothing, :missing_value),
        (["aux_basis"], plain, i -> i.aux_basis, N, nothing, nothing, nothing, :missing_value),
        (["admm_basis"], plain, i -> i.admm_basis, N, nothing, nothing, nothing, :missing_value),
        # parser.rs:152, 155 (with `f12 true`, which the options need in strict mode)
        (["f12_gamma", "gamma", "gamma_f12"], f12, i -> i.f12_gamma, "zzz", 1.0, :number_fallback, 1.0, :missing_value),
        (["f12_model", "f12_approximation"], f12, i -> i.f12_model, N, nothing, nothing, "F12b", :missing_value),
        # parser.rs:161, 165, 191, 182, 363, 255: `unwrap_or(x)`
        (["damping"], plain, i -> i.damping, "zzz", 0.0, :number_fallback, 0.0, :missing_value),
        (["smearing", "smear"], plain, i -> i.smearing, "zzz", 0.0, :number_fallback, 0.0, :missing_value),
        (["level_shift"], plain, i -> i.level_shift, "zzz", 0.0, :number_fallback, 0.0, :missing_value),
        (["diis_reset", "diis_reset_threshold"], plain, i -> i.diis_reset_threshold, "zzz", 0.1, :number_fallback, 0.1, :missing_value),
        (["dos_sigma"], plain, i -> i.dos_sigma, "zzz", 0.1, :number_fallback, 0.1, :missing_value),
        (["verbosity"], plain, i -> i.verbosity, "zzz", 1, :number_fallback, 1, :missing_value),
        # parser.rs:168-178: `val_str.parse().unwrap_or(1)`
        (["grid", "grid_level", "xc_grid"], plain, i -> i.grid_level, "zzz", 1, :unknown_value, 1, :missing_value),
        # parser.rs:179, 346: `.ok()`
        (["conv_tol_density"], plain, i -> i.conv_tol_density, "zzz", nothing, :number_fallback, nothing, :missing_value),
        (["pw_cutoff", "cutoff"], plain, i -> i.pw_cutoff, "zzz", nothing, :number_fallback, nothing, :missing_value),
        # parser.rs:194-201, 349-355, 356-362: `parts.len() >= 4`, each entry `unwrap_or(x)`
        (["field", "electric_field"], plain, i -> i.electric_field, "0.1 zzz 0.3", SVector(0.1, 0.0, 0.3), :bad_list_entry, nothing, :too_few_values),
        (["kpoints", "k_points", "kgrid"], plain, i -> i.kpoints, "2 zzz 2", (2, 1, 2), :bad_list_entry, (1, 1, 1), :too_few_values),
        (["dos_k_grid", "dos_kgrid"], plain, i -> i.dos_k_grid, "2 zzz 2", (2, 10, 2), :bad_list_entry, (10, 10, 10), :too_few_values),
        # parser.rs:202-211: an unknown or missing word sets nothing
        (["solvation"], plain, i -> i.solvation, "zzz", SolvationSpec(), :unknown_value, SolvationSpec(), :missing_value),
        # parser.rs:212-223: `unwrap_or(&"water")`, and an unknown name sets nothing
        (["solvent"], plain, i -> i.solvation, "zzz", SolvationSpec(), :unknown_value, SolvationSpec(; model = SMD), :missing_value),
        # parser.rs:224-237: `unwrap_or(78.39)`, then without a model a PCM model
        (["epsilon", "dielectric"], plain, i -> i.solvation, "zzz", SolvationSpec(; model = PCM), :number_fallback, SolvationSpec(; model = PCM), :missing_value),
        # parser.rs:238-251: `unwrap_or(1.2)`, `unwrap_or(110)`, under PCM
        (["pcm_radii", "radii_scaling"], pcm, i -> i.solvation.radii_scaling, "zzz", 1.2, :number_fallback, 1.2, :missing_value),
        (["pcm_grid", "smd_grid"], pcm, i -> i.solvation.grid_order, "zzz", 110, :number_fallback, 110, :missing_value),
        # parser.rs:252-254: `unwrap_or(&"")`
        (["output"], plain, i -> i.output, N, nothing, nothing, "", :missing_value),
        # parser.rs:286-288: the word is stored, and no kernel matches it
        (["vdw_kernel"], plain, i -> i.vdw_kernel, "zzz", nothing, :unknown_value, nothing, :missing_value),
        # parser.rs:366, 560: every token after the keyword; no code reads `pop_analysis`
        (["kpath", "k_path"], plain, i -> i.kpath, N, nothing, nothing, String[], :missing_value),
        (["population", "pop_analysis"], plain, i -> i.population, "zzz", [Mulliken, Lowdin], :unknown_value, PopulationKind[], :missing_value),
        # parser.rs:396-402, 411-425: an entry that is not a `usize` is dropped
        (["model_atoms", "model_indices"], oniom, i -> i.oniom.model_atoms, "0 zzz 2", [1, 3], :bad_list_entry, Int[], :missing_value),
        (["oniom_model"], plain, i -> i.oniom.model_atoms, "0 zzz 2", [1, 3], :bad_list_entry, Int[], :missing_value),
        (["freeze_atom", "freeze"], plain, const_atoms, "0 zzz 2", [1, 3], :bad_list_entry, Int[], :missing_value),
        # parser.rs:426-455: a bad index drops the constraint, and too few indices too
        (["freeze_bond"], plain, i -> i.constraints, "0 zzz", Constraint[], :bad_list_entry, Constraint[], :too_few_values),
        (["freeze_angle"], plain, i -> i.constraints, "0 1 zzz", Constraint[], :bad_list_entry, Constraint[], :too_few_values),
        (["freeze_dihedral", "freeze_torsion"], plain, i -> i.constraints, "0 1 2 zzz", Constraint[], :bad_list_entry, Constraint[], :too_few_values),
        # parser.rs:456-465, 505-509, 524-538: `if let Some(...)` sets nothing otherwise
        (["irc_step_size", "irc_step"], plain, i -> i.irc_step_size, "zzz", 0.1, :number_fallback, 0.1, :missing_value),
        (["irc_max_points", "irc_points"], plain, i -> i.irc_max_points, "zzz", 30, :number_fallback, 30, :missing_value),
        (["tddft_states", "num_tddft_states", "nstates"], plain, i -> i.tddft_states, "zzz", 10, :number_fallback, 10, :missing_value),
        (["dmft_temp"], plain, i -> i.dmft_temperature, "zzz", 300.0, :number_fallback, 300.0, :missing_value),
        (["dmft_n_freq", "dmft_matsubara"], plain, i -> i.dmft_n_freq, "zzz", 1024, :number_fallback, 1024, :missing_value),
        (["dmft_n_bath", "dmft_bath"], plain, i -> i.dmft_n_bath, "zzz", 4, :number_fallback, 4, :missing_value),
        # parser.rs:466-474 (`_ => Both`), 510-514 (`engine/electronic.rs:292-296` runs TDA for an
        # unknown word), 549-553 (no code reads `diis_method`)
        (["irc_direction"], plain, i -> i.irc_direction, "zzz", Both, :unknown_value, Both, :missing_value),
        (["tddft_method"], plain, i -> i.tddft_method, "zzz", TDA, :unknown_value, TDA, :missing_value),
        (["diis_method"], plain, i -> i.diis_method, "zzz", CDIIS, :unknown_value, CDIIS, :missing_value),
        # parser.rs:475-488: `.ok()`
        (["model_charge"], oniom, i -> i.oniom.model_charge, "zzz", nothing, :number_fallback, nothing, :missing_value),
        (["model_multiplicity"], oniom, i -> i.oniom.model_multiplicity, "zzz", nothing, :number_fallback, nothing, :missing_value),
        (["oniom_model_charge"], plain, i -> i.oniom.model_charge, "zzz", nothing, :number_fallback, nothing, :missing_value),
        (["oniom_model_multiplicity"], plain, i -> i.oniom.model_multiplicity, "zzz", nothing, :number_fallback, nothing, :missing_value),
        # parser.rs:489-504: a bare `ecp` sets `use_ecp`, and `ecp_name` sets it too
        (["ecp", "use_ecp"], plain, i -> i.use_ecp, N, nothing, nothing, true, :bool_fallback),
        (["ecp_name"], plain, i -> (i.use_ecp, i.ecp_name), N, nothing, nothing, (true, nothing), :missing_value),
    ]
    for (kws, (pre, post), get, bad, badval, badcase, missval, misscase) in rows, kw in kws
        for (args, val, case) in ((bad, badval, badcase), (nothing, missval, misscase))
            (args === N || val === N) && continue
            text = pre * (isnothing(args) ? kw : "$kw $args") * "\n" * post * Lenient.W
            @test_throws InputError Lenient.strict(text)
            if val === E
                @test_throws InputError Lenient.lenient(text)
            else
                p = Lenient.lenient(text)
                @test isequal(get(p.input), val)
                @test Lenient.cases(p) == [case]
            end
        end
    end
    # `ecp zzz` names an ECP in both modes (parser.rs:489-500).
    for lenient in (false, true)
        p = parse_inp("ecp zzz\n" * Lenient.W; source = "t.inp", lenient)
        @test (p.input.use_ecp, p.input.ecp_name) == (true, "zzz") && isempty(p.warnings)
    end
    # The inputs of this table that the oracle's input files use.
    L(text) = Lenient.lenient(text * Lenient.W).input
    @test L("dispersion False\n").dispersion
    @test !L("export_molden 1\n").export_molden
    @test L("compute_soc 1\n").compute_soc
    @test L("nbo\n").nbo
    @test L("damping abc\n").damping == 0.0
    @test L("grid abc\n").grid_level == 1
    @test L("irc_step abc\n").irc_step_size == 0.1
    @test isnothing(L("conv_tol_density abc\n").conv_tol_density)
    @test isnothing(L("pw_cutoff abc\n").pw_cutoff)
    @test isnothing(L("field 0.1 0.2\n").electric_field)
    @test L("kpoints 2 2\n").kpoints == (1, 1, 1)
    @test isempty(L("freeze_bond 1\n").constraints)
    @test L("kpoints 2 2 x\n").kpoints == (2, 2, 1)
    @test L("oniom\nmodel_atoms 0 x 2\nend\n").oniom.model_atoms == [1, 3]
    @test L("compute_mode tpu\n").compute_mode == CPU
    @test L("solvation cosmo\n").solvation == SolvationSpec()
    @test L("solvent foo\n").solvation == SolvationSpec()
    @test L("irc_direction sideways\n").irc_direction == Both
    @test L("tddft_method foo\n").tddft_method == TDA
    @test isnothing(L("vdw_kernel pbe-vdw\n").vdw_kernel)
    @test L("diis_method foo\n").diis_method == CDIIS
    @test L("population foo\n").population == [Mulliken, Lowdin]
    @test L("title\n").title == ""
    @test L("run\n").run_mode == SinglePoint
    @test L("output\n").output == ""
end

@testitem "lenient: per-keyword details" setup = [Lenient] begin
    using QuantumEngine
    using QuantumEngine.Input
    L(text) = Lenient.lenient(text * Lenient.W)
    # parser.rs:168-178: an unsigned integer outside the vocabulary is the level itself.
    p = L("grid 7\n")
    @test p.input.grid_level == 7 && Lenient.cases(p) == [:unknown_value]
    @test L("grid -1\n").input.grid_level == 1
    # parser.rs:426-436: a bad target keeps the constraint without a target.
    p = L("freeze_bond 0 1 zzz\n")
    @test p.input.constraints == [Constraint(FreezeBond, (1, 2, 0, 0), nothing)] && Lenient.cases(p) == [:bad_list_entry]
    # parser.rs:560: one bad entry gives the default list, however many entries are good.
    @test L("population mulliken zzz\n").input.population == [Mulliken, Lowdin]
    # parser.rs:212-223: under SMD, `solvent` replaces the solvent and any custom dielectric.
    p = L("solvation smd\nepsilon 4.0\npcm_grid 50\nsolvent dcm\n")
    @test p.input.solvation == SolvationSpec(; model = SMD, solvent = :dcm, grid_order = 50)
    @test Lenient.cases(p) == [:order_dependent]
    # parser.rs:224-237, 244-251: under SMD, `epsilon` replaces the dielectric and `pcm_grid` applies.
    p = L("solvent dcm\nepsilon 4.0\npcm_grid 50\n")
    @test p.input.solvation == SolvationSpec(; model = SMD, solvent = :dcm, epsilon = 4.0, custom_epsilon = true, grid_order = 50)
    @test isempty(p.warnings)
    # parser.rs:204-209: `solvation none` drops the model the lines before it set.
    p = L("epsilon 4.0\nsolvation none\n")
    @test p.input.solvation == SolvationSpec() && Lenient.cases(p) == [:option_without_model]
end

@testitem "lenient: unknown_keyword" setup = [Lenient] begin
    using QuantumEngine
    # parser.rs:563-618: an unknown keyword outside a block matches no branch.
    text = "use_diis true\n" * Lenient.W
    @test_throws "unknown keyword `use_diis`" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test Lenient.cases(p) == [:unknown_keyword] && length(p.input.atoms) == 3
end

@testitem "lenient: dropped_block_line" setup = [Lenient] begin
    using QuantumEngine
    # parser.rs:583: a geometry line of fewer than 4 tokens is dropped (`neb_test.inp:14`).
    text = "geometry\nH 0 0 0\nH 0 0 0.74\nEND_GEOMETRY\nend\n"
    @test_throws "`END_GEOMETRY` inside the geometry block opened at line 1 is not a keyword; the accepted keywords are" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test length(p.input.atoms) == 2 && Lenient.cases(p) == [:dropped_block_line]
end

@testitem "lenient: block_keyword" setup = [Lenient] begin
    using QuantumEngine
    # parser.rs:56-563: the keyword is matched before the block line, and `charge` leaves the
    # geometry block open (parser.rs:270-273).
    text = "geometry\nH 0 0 0\nH 0 0 1\ncharge 1\nH 0 1 0\nend\n"
    @test_throws "inside the geometry block opened at line" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test p.input.charge == 1 && length(p.input.atoms) == 3 && Lenient.cases(p) == [:block_keyword]
end

@testitem "lenient: unclosed_block" setup = [Lenient] begin
    using QuantumEngine
    using QuantumEngine.Input
    # parser.rs:563-600: the atom lines are read whether or not `end` follows.
    text = "geometry\nH 0 0 0\nH 0 0 1\n"
    @test_throws "is not closed by `end`" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test length(p.input.atoms) == 2 && Lenient.cases(p) == [:unclosed_block]
    # A block opener ends an unclosed block (parser.rs:280-284 clears the geometry block) and is
    # out of place in it: both cases.
    p = Lenient.lenient("geometry\nO 0 0 0\nO 0 0 1.2\nhubbard\nO 4.0\nend\nmultiplicity 3\n")
    @test Lenient.cases(p) == [:block_keyword, :unclosed_block]
    @test length(p.input.atoms) == 2 && p.input.hubbard == [HubbardU("O", 4.0)]
end

@testitem "lenient: hubbard_inline_argument" setup = [Lenient] begin
    using QuantumEngine
    # parser.rs:280-285 ignores the value and opens the block; the next line of two tokens is
    # read as a U line (parser.rs:564-567), so this fails in both modes.
    text = "hubbard O:5.0\nuse_diis true\n" * Lenient.W
    @test_throws "`hubbard` takes no value on its line" Lenient.strict(text)
    @test_throws "Invalid U value: true" Lenient.lenient(text)
    p = Lenient.lenient("hubbard O:5.0\nend\n" * Lenient.W)
    @test isempty(p.input.hubbard) && Lenient.cases(p) == [:hubbard_inline_argument]
end

@testitem "lenient: fragment_index" setup = [Lenient] begin
    using QuantumEngine
    # parser.rs:289-306: an index that is not a `usize` is dropped (`filter_map`).
    text = Lenient.W * "fragments 0,x;1,2\n"
    @test_throws "fragment index `x` is not an integer" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test p.input.fragments == [[1], [2, 3]] && Lenient.cases(p) == [:fragment_index]
end

@testitem "lenient: atom_index_range" setup = [Lenient] begin
    using QuantumEngine
    using QuantumEngine.Input: Constraint, FreezeAtom, FreezeDihedral
    # parser.rs:289-306, 396-455: the oracle stores every index as written and checks none against
    # the atoms. The water geometry has 3 atoms (lines 1-5).
    range_msg(kw, i) = "`$(kw)`: atom index $(i) is out of range; the input has 3 atoms (indices 0 to 2)"
    rows = [
        ("freeze_atom 99\n", 6, range_msg("freeze_atom", 99), i -> i.constraints, [Constraint(FreezeAtom, (100, 0, 0, 0), nothing)]),
        ("freeze 3\n", 6, range_msg("freeze", 3), i -> i.constraints, [Constraint(FreezeAtom, (4, 0, 0, 0), nothing)]),
        ("freeze_torsion 0 1 2 3\n", 6, range_msg("freeze_torsion", 3), i -> i.constraints, [Constraint(FreezeDihedral, (1, 2, 3, 4), nothing)]),
        ("fragments 0,1;5,2\n", 6, range_msg("fragments", 5), i -> i.fragments, [[1, 2], [6, 3]]),
        ("fragments\n0,1\n2,7\nend\n", 8, range_msg("fragments", 7), i -> i.fragments, [[1, 2], [3, 8]]),
        ("oniom_model 0 99\n", 6, range_msg("oniom_model", 99), i -> i.oniom.model_atoms, [1, 100]),
        ("oniom\nmodel_atoms 0 9\nend\n", 7, range_msg("model_atoms", 9), i -> i.oniom.model_atoms, [1, 10]),
    ]
    for (line, n, msg, get, val) in rows
        text = Lenient.W * line
        @test_throws "input: t.inp:$(n): $(msg)" Lenient.strict(text)
        p = Lenient.lenient(text)
        @test get(p.input) == val
        @test Lenient.cases(p) == [:atom_index_range]
        @test p.warnings[1].line == n
    end
    # Each index out of range is one warning.
    p = Lenient.lenient(Lenient.W * "fragments 0,5;6,1\n")
    @test Lenient.cases(p) == [:atom_index_range, :atom_index_range]
    # The check runs after the whole file is read, so a constraint may come before the geometry.
    @test Lenient.strict("freeze_atom 2\n" * Lenient.W).constraints == [Constraint(FreezeAtom, (3, 0, 0, 0), nothing)]
    @test_throws "input: t.inp:1: `freeze_atom`: atom index 3 is out of range" Lenient.strict("freeze_atom 3\n" * Lenient.W)
end

@testitem "lenient: charges outside the Int32 range" setup = [Lenient] begin
    using QuantumEngine
    # parser.rs:270-274: `charge` is an `i32` read with `ok_or("Invalid charge")?`, an oracle error;
    # parser.rs:475-488: a model charge is an `Option<i32>` read with `parse().ok()`, so `nothing`.
    for lenient in (false, true), t in ("-3000000000", "2147483648", "-2147483649")
        @test_throws "input: t.inp:1: `charge`: expected an integer in the Int32 range, got \"$(t)\"" parse_inp(
            "charge $(t)\n" * Lenient.W; source = "t.inp", lenient,
        )
    end
    for (text, n, kw) in (("oniom_model_charge 2147483648\n", 6, "oniom_model_charge"), ("oniom\nmodel_charge -3000000000\nend\n", 7, "model_charge"))
        t = Lenient.W * text
        @test_throws "input: t.inp:$(n): `$(kw)`: expected an integer in the Int32 range" Lenient.strict(t)
        p = Lenient.lenient(t)
        @test isnothing(p.input.oniom.model_charge)
        @test Lenient.cases(p) == [:number_fallback]
    end
    # The ends of the range are read.
    @test Lenient.strict(Lenient.W * "oniom_model_charge -2147483648\n").oniom.model_charge == -2147483648
    @test Lenient.strict(Lenient.W * "oniom_model_charge 2147483647\n").oniom.model_charge == 2147483647
end

@testitem "lenient: atom indices of an InputFile built in code" begin
    using QuantumEngine
    using QuantumEngine.Input: Constraint, FreezeBond, ONIOMSpec, MethodSpec, HF, AutoReference
    # Without the sites a parser records, the messages give line 0 and the field's key path.
    m = MethodSpec(HF, AutoReference, (0, 0), 0.0)
    atoms = [Atom("H", [0.0, 0.0, -0.7]), Atom("H", [0.0, 0.0, 0.7])]
    fin(inp; lenient = false) = QuantumEngine.Input._finalize_input(inp; source = "code", lenient, warnings = InputWarning[])
    inp = InputFile(; atoms, constraints = [Constraint(FreezeBond, (1, 3, 0, 0), nothing)])
    @test_throws "input: code: constraints[1]: atom index 2 is out of range; the input has 2 atoms (indices 0 to 1)" fin(inp)
    inp = InputFile(; atoms, oniom = ONIOMSpec(m, m, [1, 2, 5], nothing, nothing))
    @test_throws "input: code: oniom.model_atoms[3]: atom index 4 is out of range" fin(inp)
    @test fin(inp; lenient = true).oniom.model_atoms == [1, 2, 5]
    @test_throws "input: code: `fragments`: atom index 1 is given twice" fin(InputFile(; atoms, fragments = [[1, 2], [2]]))
end

@testitem "lenient: repeated_atom" setup = [Lenient] begin
    using QuantumEngine
    using QuantumEngine.Input: Constraint, FreezeBond, FreezeAngle
    # The oracle stores a constraint on one atom twice, and an atom in two fragments, as written.
    rows = [
        ("freeze_bond 0 0 1.4\n", "`freeze_bond`: atom index 0 is given twice", i -> i.constraints, [Constraint(FreezeBond, (1, 1, 0, 0), 1.4)]),
        ("freeze_angle 0 1 0\n", "`freeze_angle`: atom index 0 is given twice", i -> i.constraints, [Constraint(FreezeAngle, (1, 2, 1, 0), nothing)]),
        ("fragments 0;0,1\n", "`fragments`: atom index 0 is given twice", i -> i.fragments, [[1], [1, 2]]),
        ("fragments 0,0;1\n", "`fragments`: atom index 0 is given twice", i -> i.fragments, [[1, 1], [2]]),
    ]
    for (line, msg, get, val) in rows
        text = Lenient.W * line
        @test_throws "input: t.inp:6: $(msg)" Lenient.strict(text)
        p = Lenient.lenient(text)
        @test get(p.input) == val
        @test Lenient.cases(p) == [:repeated_atom]
        @test p.warnings[1].line == 6
    end
end

@testitem "lenient: hubbard_element" setup = [Lenient] begin
    using QuantumEngine
    using QuantumEngine.Input: HubbardU
    # parser.rs:563-568: the oracle stores the symbol of a U line as written.
    text = Lenient.W * "hubbard\nXx 4.0\nO 5.0\nend\n"
    @test_throws "input: t.inp:7: `hubbard`: unknown element symbol \"Xx\"" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test p.input.hubbard == [HubbardU("Xx", 4.0), HubbardU("O", 5.0)]
    @test Lenient.cases(p) == [:hubbard_element]
    @test p.warnings[1].line == 7
    # A symbol of any ASCII case names its element.
    @test Lenient.strict(Lenient.W * "hubbard\nfe 4.0\nend\n").hubbard == [HubbardU("fe", 4.0)]
    # A bare Bq is malformed, an error in both modes.
    for lenient in (false, true)
        @test_throws "input: t.inp:7: `hubbard`: a Bq atom needs an element" parse_inp(
            Lenient.W * "hubbard\nBq 4.0\nend\n"; source = "t.inp", lenient,
        )
    end
end

@testitem "lenient: nonfinite" setup = [Lenient] begin
    using QuantumEngine
    # parser.rs:161, 539-543: Rust's `f64` parses `nan` and `inf`.
    for (line, get, val) in (("scf_convergence nan", i -> i.scf_convergence, NaN), ("damping inf", i -> i.damping, Inf))
        text = line * "\n" * Lenient.W
        @test_throws "expected a finite number" Lenient.strict(text)
        p = Lenient.lenient(text)
        @test isequal(get(p.input), val) && Lenient.cases(p) == [:nonfinite]
    end
end

@testitem "lenient: extra_values" setup = [Lenient] begin
    using QuantumEngine
    using QuantumEngine.Input
    # parser.rs:270-273 and 59-60 read `parts.get(1)` or nothing, so the tokens after are ignored.
    text = "charge 0 1\n" * Lenient.W
    @test_throws "`charge` takes 1 value, got 2" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test p.input.charge == 0 && Lenient.cases(p) == [:extra_values]
    text = "optimize now\n" * Lenient.W
    @test_throws "`optimize` takes 0 values, got 1" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test p.input.run_mode == Optimization && Lenient.cases(p) == [:extra_values]
end

@testitem "lenient: duplicate_keyword" setup = [Lenient] begin
    using QuantumEngine
    using QuantumEngine.Input
    # parser.rs:270-273: each line overwrites the charge.
    text = "charge 0\nmultiplicity 2\ncharge 1\n" * Lenient.W
    @test_throws "given twice" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test p.input.charge == 1 && Lenient.cases(p) == [:duplicate_keyword]
    # parser.rs:564-567: the hubbard map keeps the last U of a symbol and tells symbols apart by case.
    p = Lenient.lenient("hubbard\nO 4.0\nO 5.0\no 6.0\nend\n" * Lenient.W)
    @test p.input.hubbard == [HubbardU("O", 5.0), HubbardU("o", 6.0)]
    @test Lenient.cases(p) == [:duplicate_keyword, :duplicate_keyword]
end

@testitem "lenient: duplicate_block" setup = [Lenient] begin
    using QuantumEngine
    # parser.rs:307-317 reopens the geometry block without clearing the atoms, and 595-596 appends
    # each atom: the atoms of both blocks are kept, in file order.
    text = "title t\n\nmultiplicity 2\ngeometry\nH 0 0 0\nH 0 0 1\nend\n\ngeometry\nH 0 0 2\nend\n"
    @test_throws "input: t.inp:9: second geometry block (first at line 4)" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test [a.position[3] for a in p.input.atoms] == [0.0, 1.0, 2.0] .* QuantumEngine.Units.ANGSTROM_TO_BOHR
    @test Lenient.cases(p) == [:duplicate_block]
    text = Lenient.W * "product_geometry\nH 0 0 3\nend\nproduct_geometry\nH 0 0 4\nend\n"
    @test_throws "second product_geometry block (first at line" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test length(p.input.product_atoms) == 2 && Lenient.cases(p) == [:duplicate_block]
end

@testitem "lenient: order_dependent" setup = [Lenient] begin
    using QuantumEngine
    using QuantumEngine.Input
    W = Lenient.W
    # (1) parser.rs:238-243 applies `pcm_radii` only under PCM; `solvation pcm` comes after.
    text = "pcm_radii 1.1\nsolvation pcm\n" * W
    @test Lenient.strict(text).solvation.radii_scaling == 1.1
    @test isempty(parse_inp(text; source = "t.inp").warnings)
    p = Lenient.lenient(text)
    @test p.input.solvation.radii_scaling == 1.2 && Lenient.cases(p) == [:order_dependent]
    # (2) parser.rs:205: `solvation pcm` resets the options.
    text = "epsilon 4.0\nsolvation pcm\n" * W
    @test Lenient.strict(text).solvation.epsilon == 4.0
    @test isempty(parse_inp(text; source = "t.inp").warnings)
    p = Lenient.lenient(text)
    @test p.input.solvation == SolvationSpec(; model = PCM) && Lenient.cases(p) == [:order_dependent]
    # (3) parser.rs:207: `solvation smd` resets the solvent.
    text = "solvent dcm\nsolvation smd\n" * W
    s = Lenient.strict(text).solvation
    @test (s.model, s.solvent) == (SMD, :dcm)
    @test isempty(parse_inp(text; source = "t.inp").warnings)
    p = Lenient.lenient(text)
    @test (p.input.solvation.model, p.input.solvation.solvent) == (SMD, :water) && Lenient.cases(p) == [:order_dependent]
    # (4) parser.rs:218-220: `solvent` replaces the PCM model by SMD.
    text = "solvation pcm\nsolvent dcm\n" * W
    @test_throws "`solvent` needs `solvation smd`" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test p.input.solvation == SolvationSpec(; model = SMD, solvent = :dcm) && Lenient.cases(p) == [:order_dependent]
    # (5) parser.rs:205: the later `solvation pcm` replaces the SMD model.
    text = "solvent dcm\nsolvation pcm\n" * W
    @test_throws "`solvent` needs `solvation smd`" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test p.input.solvation == SolvationSpec(; model = PCM) && Lenient.cases(p) == [:order_dependent]
    # A warning of another case on a solvation line does not name the order, so both are given.
    p = Lenient.lenient("epsilon 4.0 x\nsolvation pcm\n" * W)
    @test p.input.solvation == SolvationSpec(; model = PCM) && Lenient.cases(p) == [:extra_values, :order_dependent]
    # parser.rs:212-223: `solvent` replaces the PCM model that `epsilon` set.
    p = Lenient.lenient("epsilon zzz\nsolvent dcm\n" * W)
    @test p.input.solvation == SolvationSpec(; model = SMD, solvent = :dcm) && Lenient.cases(p) == [:number_fallback, :order_dependent]
end

@testitem "lenient: option_without_model" setup = [Lenient] begin
    using QuantumEngine
    using QuantumEngine.Input
    # parser.rs:238-243: without PCM, `pcm_radii` sets nothing.
    text = "pcm_radii 1.1\n" * Lenient.W
    @test_throws "`pcm_radii` needs `solvation pcm`" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test p.input.solvation == SolvationSpec() && Lenient.cases(p) == [:option_without_model]
end

@testitem "lenient: casscf_shape" setup = [Lenient] begin
    using QuantumEngine
    using QuantumEngine.Input
    # parser.rs:1154-1163: a word that does not split into two counts is CASSCF(2, 2).
    text = "method casscf(4)\n" * Lenient.W
    @test_throws "casscf needs (electrons,orbitals)" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test p.input.method == MethodSpec(CASSCF, AutoReference, (2, 2), 0.0) && Lenient.cases(p) == [:casscf_shape]
end

@testitem "lenient: casscf oracle reading" setup = [Lenient] begin
    using QuantumEngine
    using QuantumEngine.Input
    # parser.rs:1155-1160: the parentheses at the ends are optional, and two parts that are not
    # unsigned integers fail in the oracle too.
    m(text) = Lenient.lenient("method $text\n" * Lenient.W).input.method.ints
    @test m("CASSCF4,6") == (4, 6) && m("casscf((4,6") == (4, 6) && m("casscf") == (2, 2)
    @test_throws "Invalid CASSCF electron count" m("casscf(-1,2)")
    @test_throws "Invalid CASSCF orbital count" m("casscf(2,x)")
end

@testitem "lenient: ghost_unchecked" setup = [Lenient] begin
    using QuantumEngine
    # periodic_table.rs:5-12: a symbol of ghost form is a ghost of Z = 0, whatever it names.
    for (sym, msg) in (("Foo@", "unknown element symbol"), ("Bq", "a Bq atom needs an element"))
        text = "geometry\n$sym 0 0 0\nH 0 0 0.7\nH 0 0 1.4\nend\n"
        @test_throws msg Lenient.strict(text)
        p = Lenient.lenient(text)
        a = p.input.atoms[1]
        @test (a.Z, a.symbol, a.ghost, a.mass) == (0, sym, true, 0.0) && Lenient.cases(p) == [:ghost_unchecked]
        @test_throws "system: ghost atom $(repr(sym)) stands for no element" basis_element(a)
    end
    # periodic_table.rs:11 tests the prefix `Bq` or `bq` case-sensitively: `BQx` and `BQ` are no
    # ghosts, and name no element, so both modes throw.
    for (sym, msg) in (("BQx", "unknown element symbol \"BQx\""), ("BQ", "a Bq atom needs an element"), ("bQ", "a Bq atom needs an element"))
        text = "geometry\n$sym 0 0 0\nH 0 0 0.7\nH 0 0 1.4\nend\n"
        @test_throws msg Lenient.strict(text)
        @test_throws msg Lenient.lenient(text)
    end
    # The product geometry block reads ghosts the same way.
    p = Lenient.lenient(Lenient.W * "product_geometry\nFoo@ 0 0 0\nend\n")
    @test p.input.product_atoms[1].Z == 0
end

@testitem "lenient: rhf_open_shell" setup = [Lenient] begin
    using QuantumEngine
    using QuantumEngine.Input
    # types.rs:236 and the engine choose RHF or UHF by multiplicity; `rhf` names the same method.
    text = "method rhf\nmultiplicity 3\ngeometry\nO 0 0 0\nO 0 0 1.208\nend\n"
    @test_throws "requests a restricted reference" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test p.input.method.reference == AutoReference && Lenient.cases(p) == [:rhf_open_shell]
end

@testitem "lenient: f12_option_alone" setup = [Lenient] begin
    using QuantumEngine
    # parser.rs:152: `f12_gamma` sets the exponent and leaves `use_f12` off.
    text = "f12_gamma 0.9\n" * Lenient.W
    @test_throws "needs `f12 true`" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test (p.input.f12_gamma, p.input.use_f12) == (0.9, false) && Lenient.cases(p) == [:f12_option_alone]
end

@testitem "lenient: yaml_in_inp" setup = [Lenient, Fixtures] begin
    using QuantumEngine
    # parser.rs:563-618 reads no YAML line as a keyword, so the file has no atoms.
    text = Fixtures.oracle_input("molecules/h2_ccsd.inp")
    @test_throws "looks like YAML" Lenient.strict(text)
    @test_throws "no atoms" Lenient.lenient(text)
end

@testitem "lenient: bool_fallback (pbc)" setup = [Lenient] begin
    using QuantumEngine
    # parser.rs:645-653: `true`, `1` and `T` are true, every other token false.
    text = "pbc\n1 T F\nend\n" * Lenient.W
    @test_throws "expected true or false" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test p.input.pbc == (true, true, false) && Lenient.cases(p) == [:bool_fallback]
    # parser.rs:647 reads the first three tokens of a longer line.
    p = Lenient.lenient("pbc\ntrue false true x\nend\n" * Lenient.W)
    @test p.input.pbc == (true, false, true) && Lenient.cases(p) == [:extra_values]
end

@testitem "lenient: o2_hubbard.in cases" setup = [Lenient, Fixtures] begin
    using QuantumEngine
    using QuantumEngine.Input
    # The geometry block (line 6) has no `end`. `hubbard` on line 10 ends it (parser.rs:280-285
    # clears the geometry block) and opens the hubbard block, which `end` closes (parser.rs:369-376).
    text = Fixtures.oracle_input("tests/o2_hubbard.in")
    @test_throws "input: o2_hubbard.in:10: `hubbard` inside the geometry block opened at line 6" parse_inp(text; source = "o2_hubbard.in")
    p = parse_inp(text; source = "o2_hubbard.in", lenient = true)
    @test [(w.case, w.line) for w in p.warnings] == [(:block_keyword, 10), (:unclosed_block, 10)]
    @test length(p.input.atoms) == 2 && p.input.hubbard == [HubbardU("O", 4.0)] && p.input.multiplicity == 3
end

@testitem "lenient: stress_h2o_complex.in cases" setup = [Lenient, Fixtures] begin
    using QuantumEngine
    using QuantumEngine.Input
    # `hubbard O:5.0` (line 12) opens a hubbard block that no `end` closes (parser.rs:280-285).
    # Lines 13-17 are keywords, which the oracle matches before the block line (parser.rs:56-563):
    # `vdw_kernel pbe-vdw` stores a word no kernel matches (parser.rs:286-288), and the others set
    # their values.
    text = Fixtures.oracle_input("tests/stress_h2o_complex.in")
    @test_throws "input: stress_h2o_complex.in:12: `hubbard` takes no value on its line" parse_inp(text; source = "stress_h2o_complex.in")
    p = parse_inp(text; source = "stress_h2o_complex.in", lenient = true)
    @test [(w.case, w.line) for w in p.warnings] == [
        (:hubbard_inline_argument, 12), (:block_keyword, 13), (:unknown_value, 13), (:block_keyword, 14),
        (:block_keyword, 15), (:block_keyword, 16), (:block_keyword, 17), (:unclosed_block, 12),
    ]
    inp = p.input
    @test isempty(inp.hubbard) && isnothing(inp.vdw_kernel) && inp.nbo
    @test (inp.max_iterations, inp.damping, inp.verbosity) == (200, 0.1, 1)
end

@testitem "lenient: unsigned fields" setup = [Lenient] begin
    using QuantumEngine
    using QuantumEngine.Input
    # types.rs stores these fields as `usize`, so a negative token fails Rust's `parse` and takes
    # the keyword's fallback (parser.rs:89-110, 244-255, 349-362, 456-465, 475-488, 505-509,
    # 524-538, 539-547). One row per field: the keyword, its context, the field read, the negative
    # input, the lenient value and its case; `:error` is an oracle error, an error in both modes.
    E = :error
    plain = ("", "")
    oniom = ("oniom\n", "end\n")
    pcm = ("solvation pcm\n", "")
    rows = [
        ("multiplicity", plain, identity, "-1", E, nothing),
        ("max_iterations", plain, identity, "-1", E, nothing),
        ("ts_target_mode", plain, identity, "-1", E, nothing),
        ("neb_images", plain, identity, "-1", E, nothing),
        ("verbosity", plain, i -> i.verbosity, "-3", 1, :number_fallback),
        ("pcm_grid", pcm, i -> i.solvation.grid_order, "-1", 110, :number_fallback),
        ("kpoints", plain, i -> i.kpoints, "-2 2 2", (1, 2, 2), :bad_list_entry),
        ("dos_k_grid", plain, i -> i.dos_k_grid, "2 -0 2", (2, 10, 2), :bad_list_entry),
        ("irc_max_points", plain, i -> i.irc_max_points, "-1", 30, :number_fallback),
        ("tddft_states", plain, i -> i.tddft_states, "-1", 10, :number_fallback),
        ("dmft_n_freq", plain, i -> i.dmft_n_freq, "-1", 1024, :number_fallback),
        ("dmft_n_bath", plain, i -> i.dmft_n_bath, "-1", 4, :number_fallback),
        ("model_multiplicity", oniom, i -> i.oniom.model_multiplicity, "-1", nothing, :number_fallback),
        ("oniom_model_multiplicity", plain, i -> i.oniom.model_multiplicity, "-1", nothing, :number_fallback),
    ]
    for (kw, (pre, post), get, args, val, case) in rows
        text = pre * "$kw $args\n" * post * Lenient.W
        @test_throws "`$kw`: expected a non-negative integer, got" Lenient.strict(text)
        if val === E
            @test_throws "`$kw`: expected a non-negative integer, got" Lenient.lenient(text)
        else
            p = Lenient.lenient(text)
            @test isequal(get(p.input), val)
            @test Lenient.cases(p) == [case]
        end
    end
    # A `+` sign is Rust's unsigned syntax too, and `charge` is signed (`i32`).
    @test Lenient.strict("verbosity +2\n" * Lenient.W).verbosity == 2
    @test Lenient.strict("charge -1\nmultiplicity 2\n" * Lenient.W).charge == -1
end

@testitem "lenient: tokens split on spaces and tabs only" setup = [Lenient] begin
    using QuantumEngine
    # The oracle splits on every Unicode white space (`split_whitespace`); QuantumEngine splits on
    # ' ' and '\t', so a no-break space joins the keyword to its value (docs/src/parity.md).
    text = "charge 1\n" * Lenient.W
    @test_throws "unknown keyword `charge 1`" Lenient.strict(text)
    p = Lenient.lenient(text)
    @test Lenient.cases(p) == [:unknown_keyword] && p.input.charge == 0
end

@testitem "lenient: dropped_block_line in cell, pbc and fragments" setup = [Lenient] begin
    using QuantumEngine
    # A line that cannot start the block's content is no block line; the block's own content and
    # the lines of the right shape keep their other cases.
    for (block, line, content, n) in (
            ("cell", "MAX_ITER 3", "1 0 0\n0 1 0\n0 0 1\n", 5),
            ("cell", "MAX_ITER", "1 0 0\n0 1 0\n0 0 1\n", 5),
            ("pbc", "MAX_ITER", "true true false\n", 3),
            ("pbc", "MAX_ITER 3", "true true false\n", 3),
            ("fragments", "MAX_ITER", "0,1;2\n", 3),
            ("fragments", "MAX_ITER 3", "0,1;2\n", 3),
        )
        text = Lenient.W * "$(block)\n$(content)$(line)\nend\n"
        l = count(==('\n'), Lenient.W) + 1 + count(==('\n'), content) + 1
        word = split(line)[1]
        @test_throws "input: t.inp:$(l): `$(word)` inside the $(block) block opened at line $(l - count(==('\n'), content) - 1) is not a keyword; the accepted keywords are" Lenient.strict(text)
        p = Lenient.lenient(text)
        @test Lenient.cases(p) == [:dropped_block_line]
    end
    # The content survives the dropped line.
    p = Lenient.lenient(Lenient.W * "cell\n1 0 0\nMAX_ITER 3\n0 1 0\n0 0 1\nend\n")
    @test p.input.lattice !== nothing
    p = Lenient.lenient(Lenient.W * "pbc\nMAX_ITER\ntrue true false\nend\n")
    @test p.input.pbc == (true, true, false)
    p = Lenient.lenient(Lenient.W * "fragments\nMAX_ITER\n0,1;2\nend\n")
    @test p.input.fragments == [[1, 2], [3]]
    # Lines of the right first token keep their own cases.
    @test_throws "a cell line takes 3 numbers, got 2" Lenient.strict(Lenient.W * "cell\n1 0\nend\n")
    @test_throws "a pbc line takes 3 values, got 2" Lenient.strict(Lenient.W * "pbc\ntrue true\nend\n")
    @test_throws "fragment index `x` is not an integer" Lenient.strict(Lenient.W * "fragments\n0,x;1\nend\n")
end
