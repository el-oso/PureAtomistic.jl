@testitem "PureBLAS: forces and virial match LAMMPS-MACE" tags = [:pureblas] setup = [LammpsReference, Checkpoints] begin
    using PureBLAS
    @test PureBLAS.is_active()
    # Float64: the 1e-6 LAMMPS targets of the main suite. Float32: the main suite's tolerances for
    # the Float32 LAMMPS run (measured margins 1.55e-5 eV, 9.4e-6 eV/Å and 8.0e-5 eV).
    for (T, ref, src, tole, tolf, tolv) in (
            (Float64, :mpa0_f64, :mpa0, 1.0e-6, 1.0e-6, 1.0e-6),
            (Float64, :mp0b3_f64, :mp0b3, 1.0e-6, 1.0e-6, 1.0e-6),
            (Float32, :mpa0_f32, :mpa0, 2.5e-4, 1.5e-4, 1.2e-3),
        )
        model = MACEModel{T}(load_mace_checkpoint(Checkpoints.fetch(src)))
        for s in LammpsReference.structures(LammpsReference.load(ref))
            r = mace_energy_forces(model, T.(s.positions), T.(s.cell), s.atomic_numbers)
            @test abs(r.energy - s.energy) <= tole
            @test maximum(abs, r.forces - s.forces; init = 0.0) <= tolf
            @test maximum(abs, r.virial - s.virial) <= tolv
        end
    end
end

@testitem "PureBLAS: results agree with OpenBLAS" tags = [:pureblas] setup = [Checkpoints] begin
    using PureBLAS, Serialization
    @test PureBLAS.is_active()
    include(pkgdir(PureMLIP, "test", "reference", "structures.jl"))
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0))
    # The OpenBLAS values come from a child process that never loads PureBLAS.
    out = tempname()
    script = joinpath(@__DIR__, "openblas_reference.jl")
    run(`$(Base.julia_cmd()) --project=$(Base.active_project()) $script $(Checkpoints.fetch(:mpa0)) $out`)
    ref = deserialize(out)
    rm(out)
    for T in (Float64, Float32)
        m = MACEModel{T}(ck)
        for g in reference_structures()
            P, C = T.(permutedims(g.positions)), T.(permutedims(g.cell))
            r = mace_energy_forces(m, P, C, g.atomic_numbers)
            o = ref[(T, g.name)]
            # Oracle: the same evaluation with OpenBLAS. Rounding bound c·eps(T)·N relative to the
            # largest magnitude of the quantity, with c = 10 and N = 1280 accumulated terms.
            # Measured over the five structures: the largest relative force difference is
            # 3.7e-13 (Float64) and 2.7e-4 (Float32), 7.7 and 5.7 times below the bound.
            tol = 10 * eps(T) * 1280
            @test abs(r.energy - o.energy) <= tol * abs(o.energy)
            @test maximum(abs, r.forces - o.forces; init = zero(T)) <= tol * maximum(abs, o.forces; init = zero(T))
            @test maximum(abs, r.virial - o.virial) <= tol * maximum(abs, o.virial)
        end
    end
end

@testitem "PureBLAS: evaluation is allocation-free" tags = [:pureblas] setup = [Checkpoints] begin
    using PureBLAS, StrictModeTest, StrictMode
    @test PureBLAS.is_active()
    ck = load_mace_checkpoint(Checkpoints.fetch(:mp0b3))
    P = [0.0 2.5 1.2; 0.0 0.0 1.9; 0.0 0.0 0.3]; C = [7.0 0.0 0.0; 0.0 7.0 0.0; 0.0 0.0 7.0]; Z = [29, 29, 8]
    @noinline f!(res, ws, m, P, C, Z, nl) = mace_energy!(res, ws, m, P, C, Z, nl)
    @noinline g!(res, ws, m, P, C, Z, nl) = mace_energy_forces!(res, ws, m, P, C, Z, nl)
    for T in (Float64, Float32)
        m = MACEModel{T}(ck)
        PT, CT = T.(P), T.(C)
        nl = neighbor_list(PT, CT, 6.0)
        # A chunk smaller than the edge count exercises the chunked radial MLP and its pullback.
        ws = MACEWorkspace{T}(m, 3, nl.n_edges; edge_chunk = 7)
        res = MACEResult{T}(3)
        f!(res, ws, m, PT, CT, Z, nl)
        g!(res, ws, m, PT, CT, Z, nl)
        @test nl.n_edges > 7
        @test_noalloc f!(res, ws, m, PT, CT, Z, nl)
        @test_noalloc g!(res, ws, m, PT, CT, Z, nl)
    end
end
