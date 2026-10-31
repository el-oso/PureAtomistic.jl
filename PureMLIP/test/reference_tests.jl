@testitem "LAMMPS reference: structures, metadata, sanity" setup = [LammpsReference, Checkpoints] begin
    using SHA
    include(pkgdir(PureMLIP, "test", "reference", "structures.jl"))
    gen = reference_structures()
    for name in (:mpa0_f64, :mpa0_f32, :mp0b3_f64)
        ref = LammpsReference.load(name)
        ϵ = name == :mpa0_f32 ? Float64(eps(Float32)) : eps(Float64)
        @test ref["schema"] == "puremlip-lammps-mace-v1"
        @test [s["name"] for s in ref["structures"]] == [g.name for g in gen]
        for g in gen
            s = LammpsReference.structure(ref, g.name)
            # Written with 17 significant digits: the round trip is exact.
            @test s.positions == permutedims(g.positions) && s.cell == permutedims(g.cell)
            @test s.atomic_numbers == g.atomic_numbers
            # Newton's third law: rounding bound 64·ϵ·max(Σ|F|, n·1 eV/Å) in the export's precision
            # (n ≤ 32 atoms, ≤ 100 edge terms per atom). ΣF is a sum of per-edge pair forces, so its
            # rounding error scales with their magnitudes, which are of order 1 eV/Å per atom in these
            # metallic cells even when the net forces cancel; isolated_cu has no edges and its forces
            # are exactly zero.
            @test maximum(abs, sum(s.forces; dims = 2)) <= 64 * ϵ * max(sum(abs, s.forces), length(s.atomic_numbers) * 1.0)
        end
        src = name == :mp0b3_f64 ? :mp0b3 : :mpa0
        @test ref["model"]["source_sha256"] == bytes2hex(open(sha256, Checkpoints.fetch(src)))
    end
    # isolated_cu has no edges: its LAMMPS energy is E0(Cu) exactly when node_energy includes E0.
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0))
    e0cu = ck.state.f64["atomic_energies_fn.atomic_energies"].data[findfirst(==(29), ck.hyperparameters.atomic_numbers)]
    @test LammpsReference.structure(LammpsReference.load(:mpa0_f64), "isolated_cu").energy == e0cu
end
