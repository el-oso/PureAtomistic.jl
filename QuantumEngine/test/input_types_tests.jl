@testitem "input types: defaults" begin
    using QuantumEngine
    using QuantumEngine.Input
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    inp = InputFile()
    # Geometry and its blocks.
    @test inp.atoms == Atom{Float64}[] && inp.atoms isa Vector{Atom{Float64}}
    @test inp.product_atoms == Atom{Float64}[] && inp.point_charges == PointCharge{Float64}[]
    # The system and the method.
    @test (inp.title, inp.run_mode, inp.charge, inp.multiplicity, inp.basis) == ("", SinglePoint, 0, 1, "sto-3g")
    @test inp.method == MethodSpec(HF, AutoReference, (0, 0), 0.0)
    # SCF.
    @test (inp.scf_convergence, inp.max_iterations, inp.diis_method) == (1.0e-6, 50, CDIIS)
    @test isnothing(inp.conv_tol_density)
    @test (inp.diis_reset_threshold, inp.damping, inp.smearing, inp.grid_level, inp.level_shift) == (0.1, 0.0, 0.0, 1, 0.0)
    @test (inp.auto_stabilize, inp.robust_convergence) == (true, false)
    @test isnothing(inp.electric_field)
    @test !use_fock_damping(inp)
    @test use_fock_damping(InputFile(; damping = 0.3))
    # Density fitting, compute mode and checkpoints.
    @test (inp.use_ri_j, inp.use_ri_k, inp.aux_basis, inp.compute_mode) == (false, false, nothing, CPU)
    @test (inp.checkpoint, inp.load_checkpoint) == (nothing, false)
    # Transition states, NEB and thermochemistry.
    @test (inp.ts_target_mode, inp.ts_trust_radius) == (0, 0.1)
    @test (inp.neb_k, inp.neb_images, inp.climbing_image) == (0.1, 7, true)
    @test (inp.temperature, inp.pressure, inp.symmetry_number) == (298.15, 1.0, 1.0)
    # Corrections and analyses.
    @test (inp.dispersion, inp.zora, inp.use_f12, inp.f12_gamma, inp.f12_model) == (false, false, false, 1.0, "F12b")
    @test (inp.nbo, inp.admm, inp.admm_basis, inp.use_ecp, inp.ecp_name) == (false, false, nothing, false, nothing)
    @test isnothing(inp.vdw_kernel)
    # Solvation.
    s = inp.solvation
    @test (s.model, s.solvent, s.epsilon, s.custom_epsilon) == (NoSolvation, :water, 78.39, false)
    @test (s.radii_scaling, s.grid_order, s.surface) == (1.2, 110, VdwSurface)
    @test s.probe_radius == 0.4 * ANGSTROM_TO_BOHR
    # Output.
    @test (inp.output, inp.verbosity, inp.export_molden, inp.export_cube, inp.export_wfx) == (nothing, 1, false, false, false)
    @test inp.population == [Mulliken, Lowdin]
    # Periodic systems.
    @test (inp.pw_cutoff, inp.kpoints, inp.dos_k_grid, inp.dos_sigma) == (nothing, (1, 1, 1), (10, 10, 10), 0.1)
    @test inp.kpath == String[] && inp.kpath isa Vector{String}
    @test (inp.lattice, inp.pbc) == (nothing, (true, true, true))
    # ONIOM, constraints and IRC.
    @test isnothing(inp.oniom)
    @test inp.constraints == Constraint[]
    @test (inp.irc_step_size, inp.irc_max_points, inp.irc_direction) == (0.1, 30, Both)
    # Excited states.
    @test (inp.tddft_states, inp.tddft_method, inp.compute_soc, inp.tddft_triplets) == (10, TDA, false, false)
    # DMFT, Hubbard and fragments.
    @test (inp.use_dmft, inp.dmft_temperature, inp.dmft_n_freq, inp.dmft_n_bath) == (false, 300.0, 1024, 4)
    @test inp.dmft_u == HubbardU[] && inp.dmft_j == HubbardU[] && inp.hubbard == HubbardU[]
    @test inp.fragments == Vector{Int}[] && inp.fragments isa Vector{Vector{Int}}
    # Each default vector is a new one per call.
    @test InputFile().population !== InputFile().population
    p = ParsedInput(inp, InputWarning[])
    @test p.input === inp && isempty(p.warnings)
end

@testitem "input types: method words" begin
    using QuantumEngine.Input
    @test method_word(["uhf"]).reference == Unrestricted
    @test method_word(["ccsd(t)"]).kind == CCSD_T
    @test method_word(["casscf(4,4)"]).ints == (4, 4)
    @test method_word(["libxc_hgga", "402", "0.2"]) == MethodSpec(LIBXC_HGGA, AutoReference, (402, 0), 0.2)
    @test method_word(["dft"]) === nothing
    @test method_word(["casscf(4)"]) === nothing
    # References.
    @test method_word(["rhf"]) == MethodSpec(HF, Restricted, (0, 0), 0.0)
    @test method_word(["HF"]) == MethodSpec(HF, AutoReference, (0, 0), 0.0)
    @test method_word(["RMP2"]).reference == Restricted
    @test method_word(["ump2"]).reference == Unrestricted
    @test method_word(["uccsd(t)"]) == MethodSpec(CCSD_T, Unrestricted, (0, 0), 0.0)
    @test method_word(["rccsd"]).reference == Restricted
    @test method_word(["b3lyp"]).reference == AutoReference
    # The parameterized forms.
    @test method_word(["CASSCF(10,8)"]) == MethodSpec(CASSCF, AutoReference, (10, 8), 0.0)
    @test method_word(["libxc_gga", "101", "130"]) == MethodSpec(LIBXC_GGA, AutoReference, (101, 130), 0.0)
    @test method_word(["LIBXC_MGGA", "263", "267"]) == MethodSpec(LIBXC_MGGA, AutoReference, (263, 267), 0.0)
    @test method_word(["libxc_hmgga", "+450", "1e-1"]) == MethodSpec(LIBXC_HMGGA, AutoReference, (450, 0), 0.1)
    # The number of the following tokens is the caller's to check.
    @test method_word(["hf", "extra"]) == MethodSpec(HF, AutoReference, (0, 0), 0.0)
    # Malformed parameterized forms and an empty token list.
    for toks in (
            String[], ["casscf"], ["casscf()"], ["casscf(4,)"], ["casscf(,4)"], ["casscf(4,4"],
            ["casscf(+4,4)"], ["casscf(4,4,4)"], ["casscf(a,4)"], ["casscf(99999999999,4)"],
            ["libxc_gga", "101"], ["libxc_gga", "101", "x"], ["libxc_gga", "0x65", "130"],
            ["libxc_gga", "3000000000", "1"], ["libxc_hgga", "402"], ["libxc_hgga", "402", "0x1p-2"],
            ["libxc_hgga", "4.5", "0.2"], ["libxc_hgga", "402", "a"],
        )
        @test method_word(toks) === nothing
    end
end

@testitem "input types: method words on a view" begin
    using QuantumEngine.Input, OffsetArrays
    v = method_word(["libxc_hgga", "402", "0.2"])
    @test method_word(view(["x", "libxc_hgga", "402", "0.2"], 2:4)) == v
    @test method_word(OffsetArray(["libxc_hgga", "402", "0.2"], -1:1)) == v
    @test method_word(view(["x", "uhf"], 2:2)) == method_word(["uhf"])
    @test method_word(split("libxc_hgga 402 0.2")) == v
end

@testitem "input types: other words" begin
    using QuantumEngine.Input
    @test run_mode_word("counterpoise") == BSSE
    @test solvation_word("true") == PCM
    @test solvation_word("false") == NoSolvation
    @test solvation_word("cosmo") === nothing
    @test solvent_word("Di-chloro_methane") == :dcm
    @test vdw_kernel_word("grimme-d2") == D2
    @test vdw_kernel_word("pbe-vdw") === nothing
    @test grid_word("ultrafine") == 3
    # Case, length and the solvent normalization.
    @test run_mode_word("OPT") == Optimization
    @test run_mode_word("op") === nothing && run_mode_word("optt") === nothing && run_mode_word("") === nothing
    @test solvent_word("  1,4") === nothing
    @test solvent_word("1-4-dioxane") == :dioxane
    @test solvent_word("1,4-dioxane") === nothing
    @test solvent_word("14-Dioxane") == :dioxane
    @test solvent_word("n-hexane") == :hexane
    @test solvent_word("-_ ") === nothing
    @test solvent_word("water-") == :water
    @test solvent_word("wat") === nothing
    @test grid_word("5") === nothing && grid_word("01") === nothing
    @test population_word("Löwdin") === nothing
    @test run_mode_word(SubString("xsp", 2, 3)) == SinglePoint
end

# Every alias the oracle's `.inp` parser accepts for each keyword, and the value it maps to.
@testitem "input types: every oracle alias maps" begin
    using QuantumEngine.Input
    run_modes = [
        "sp" => SinglePoint, "single_point" => SinglePoint, "opt" => Optimization,
        "optimization" => Optimization, "ts" => TransitionState, "saddle" => TransitionState,
        "neb" => NEB, "path" => NEB, "freq" => Frequency, "frequency" => Frequency,
        "tddft" => TDDFT, "excited" => TDDFT, "spectroscopy" => Spectroscopy, "spec" => Spectroscopy,
        "optfreq" => OptFreq, "optimize_frequency" => OptFreq, "bsse" => BSSE, "counterpoise" => BSSE,
        "nbo" => NBO, "bands" => Bands, "band_structure" => Bands, "dos" => DOS, "nmr" => NMR,
        "shielding" => NMR, "qtaim" => QTAIM, "topology" => QTAIM, "aim" => QTAIM, "irc" => IRC,
        "intrinsic_reaction_coordinate" => IRC,
    ]
    methods = [
        "rhf" => (HF, Restricted), "uhf" => (HF, Unrestricted), "hf" => (HF, AutoReference),
        "mp2" => (MP2, AutoReference), "rmp2" => (MP2, Restricted), "ump2" => (MP2, Unrestricted),
        "mp2-f12" => (MP2_F12, AutoReference), "mp2_f12" => (MP2_F12, AutoReference),
        "rmp2-f12" => (MP2_F12, Restricted),
        "bws2" => (BWS2, AutoReference), "bw-s2" => (BWS2, AutoReference), "bwpt2" => (BWS2, AutoReference),
        "dlpno" => (DLPNO_MP2, AutoReference), "dlpno-mp2" => (DLPNO_MP2, AutoReference),
        "dlpno_mp2" => (DLPNO_MP2, AutoReference),
        "ccsd" => (CCSD, AutoReference), "rccsd" => (CCSD, Restricted), "uccsd" => (CCSD, Unrestricted),
        "ccsd(t)" => (CCSD_T, AutoReference), "uccsd(t)" => (CCSD_T, Unrestricted),
        "ccsd-f12" => (CCSD_F12, AutoReference), "ccsd_f12" => (CCSD_F12, AutoReference),
        "rccsd-f12" => (CCSD_F12, Restricted),
        "ccsd(t)-f12" => (CCSD_T_F12, AutoReference), "ccsd(t)_f12" => (CCSD_T_F12, AutoReference),
        "ccsd_t_f12" => (CCSD_T_F12, AutoReference),
        "lda" => (LDA, AutoReference), "pbe" => (PBE, AutoReference), "blyp" => (BLYP, AutoReference),
        "scan" => (SCAN, AutoReference), "tpss" => (TPSS, AutoReference), "pbe0" => (PBE0, AutoReference),
        "b3lyp" => (B3LYP, AutoReference), "m06l" => (M06L, AutoReference), "m06-l" => (M06L, AutoReference),
        "m062x" => (M062X, AutoReference), "m06-2x" => (M062X, AutoReference),
        "rpa" => (RPA, AutoReference), "rirpa" => (RPA, AutoReference),
        "b88-vdw" => (B88_VDW, AutoReference), "b88_vdw" => (B88_VDW, AutoReference),
        "pbe-vdw" => (PBE_VDW, AutoReference), "pbe_vdw" => (PBE_VDW, AutoReference),
        "b97x-d" => (B97X_D, AutoReference), "b97xd" => (B97X_D, AutoReference),
        "wb97m-v" => (WB97M_V, AutoReference), "wb97mv" => (WB97M_V, AutoReference),
        "b2plyp" => (B2PLYP, AutoReference), "mpw2plyp" => (MPW2PLYP, AutoReference),
        "dsd-pbep86" => (DSD_PBEP86, AutoReference), "dsdpbep86" => (DSD_PBEP86, AutoReference),
        "eom-ccsd" => (EOM_CCSD, AutoReference), "eom_ccsd" => (EOM_CCSD, AutoReference),
        "pwdft" => (PWDFT, AutoReference), "pw-dft" => (PWDFT, AutoReference), "pw_dft" => (PWDFT, AutoReference),
    ]
    solvation = ["pcm" => PCM, "true" => PCM, "smd" => SMD, "none" => NoSolvation, "false" => NoSolvation]
    solvents = [
        "water" => :water, "h2o" => :water, "acetonitrile" => :acetonitrile, "mecn" => :acetonitrile,
        "ch3cn" => :acetonitrile, "methanol" => :methanol, "meoh" => :methanol, "ch3oh" => :methanol,
        "ethanol" => :ethanol, "etoh" => :ethanol, "dmso" => :dmso, "dimethylsulfoxide" => :dmso,
        "chloroform" => :chloroform, "chcl3" => :chloroform, "dcm" => :dcm, "dichloromethane" => :dcm,
        "ch2cl2" => :dcm, "toluene" => :toluene, "phme" => :toluene, "benzene" => :benzene,
        "phh" => :benzene, "thf" => :thf, "tetrahydrofuran" => :thf, "hexane" => :hexane,
        "nhexane" => :hexane, "cyclohexane" => :cyclohexane, "acetone" => :acetone, "dmf" => :dmf,
        "dimethylformamide" => :dmf, "carbontetrachloride" => :carbontetrachloride,
        "ccl4" => :carbontetrachloride, "diethylether" => :diethylether, "ether" => :diethylether,
        "et2o" => :diethylether, "dioxane" => :dioxane, "14dioxane" => :dioxane,
        "nitrobenzene" => :nitrobenzene, "pyridine" => :pyridine,
    ]
    kernels = [
        "d2" => D2, "grimme-d2" => D2, "d3" => D3, "grimme-d3" => D3, "d3bj" => D3BJ,
        "b97x-d" => B97X_D_DISP, "vv10" => VV10, "nl-vdw" => VV10,
    ]
    tddft = ["tda" => TDA, "rpa" => RPA_TD, "spin_flip" => SpinFlip, "spinflip" => SpinFlip, "sf" => SpinFlip, "sf-tddft" => SpinFlip]
    grids = [
        "coarse" => 0, "standard" => 1, "medium" => 1, "fine" => 2, "ultrafine" => 3, "superfine" => 4,
        "0" => 0, "1" => 1, "2" => 2, "3" => 3, "4" => 4,
    ]
    irc = ["forward" => Forward, "fwd" => Forward, "backward" => Backward, "bwd" => Backward, "back" => Backward, "both" => Both]
    compute = ["cpu" => CPU, "gpu" => GPU, "hybrid" => Hybrid]
    diis = ["cdiis" => CDIIS, "ediis" => EDIIS]
    population = ["mulliken" => Mulliken, "lowdin" => Lowdin]
    surfaces = ["vdw" => VdwSurface, "sas" => SASSurface, "smd" => SMDSurface]

    for (f, list, accepted) in (
            (run_mode_word, run_modes, accepted_run_modes()),
            (solvation_word, solvation, accepted_solvation_models()),
            (solvent_word, solvents, accepted_solvents()),
            (vdw_kernel_word, kernels, accepted_vdw_kernels()),
            (tddft_method_word, tddft, accepted_tddft_methods()),
            (grid_word, grids, accepted_grids()),
            (irc_direction_word, irc, accepted_irc_directions()),
            (compute_mode_word, compute, accepted_compute_modes()),
            (diis_word, diis, accepted_diis_methods()),
            (population_word, population, accepted_populations()),
            (surface_word, surfaces, accepted_surfaces()),
        )
        for (w, v) in list
            @test f(w) == v
            @test f(uppercase(w)) == v
        end
        @test sort(accepted) == sort(first.(list))
    end
    for (w, (k, r)) in methods
        @test method_word([w]) == MethodSpec(k, r, (0, 0), 0.0)
        @test method_word([uppercase(w)]) == MethodSpec(k, r, (0, 0), 0.0)
    end
    @test sort(accepted_methods()) ==
        sort([first.(methods); "libxc_gga"; "libxc_mgga"; "libxc_hgga"; "libxc_hmgga"; "casscf(e,o)"])
    @test accepted_methods() === accepted_methods()
end

@testitem "input types: allocation-free words" begin
    using QuantumEngine.Input, StrictModeTest
    # @test_noalloc throws a StrictViolation unless AllocCheck proves the call allocation-free.
    @test (@test_noalloc run_mode_word("Counterpoise")) == BSSE
    @test (@test_noalloc solvent_word("Di-chloro_methane")) == :dcm
    # `solvent_word` returns `Union{Nothing,Symbol}`; neither branch allocates.
    @test (@test_noalloc solvent_word("cosmo")) === nothing
    @test (@test_noalloc solvent_word(SubString("xDMSO", 2))) == :dmso
    @test (@test_noalloc accepted_methods()) === accepted_methods()
end

@testitem "input types: concrete fields" begin
    using QuantumEngine
    using QuantumEngine.Input
    concrete(T) = isconcretetype(T) || (T isa Union && all(isconcretetype, Base.uniontypes(T)))
    for S in (InputFile, SolvationSpec, ONIOMSpec, MethodSpec, Constraint, ParsedInput)
        @test all(concrete, fieldtypes(S))
    end
    @test !all(isconcretetype, fieldtypes(InputFile))
end
