@testitem "PureBLAS: forces and virial match LAMMPS-MACE" tags = [:pureblas] setup = [LammpsReference, Checkpoints] begin
    using PureBLAS
    @test PureBLAS.is_active()
    # Float64: the 1e-6 LAMMPS targets of the main suite. Float32: the main suite's tolerances for
    # the Float32 LAMMPS run (measured margins 1.55e-5 eV, 1.0e-5 eV/Å and 4.7e-5 eV).
    for (T, ref, src, tole, tolf, tolv) in (
            (Float64, :mpa0_f64, :mpa0, 1.0e-6, 1.0e-6, 1.0e-6),
            (Float64, :mp0b3_f64, :mp0b3, 1.0e-6, 1.0e-6, 1.0e-6),
            (Float32, :mpa0_f32, :mpa0, 2.5e-4, 1.5e-4, 1.2e-3),
        )
        model = MACEModel{T}(load_mace_checkpoint(Checkpoints.fetch(src)))
        for s in LammpsReference.structures(LammpsReference.load(ref))
            r = mace_energy_forces(model, T.(s.positions), T.(s.cell), s.atomic_numbers)
            @test abs(r.energy - s.energy) <= tole
            # Node energies: the main suite's 1e-6 eV against LAMMPS-MACE, Float64 models only.
            T === Float64 && @test maximum(abs, r.node_energies - s.node_energies) <= 1.0e-6
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
        ws = MACEWorkspace{T}(m, 3, nl.n_edges; edge_chunk = 7, elements = Z)
        res = MACEResult{T}(3)
        f!(res, ws, m, PT, CT, Z, nl)
        g!(res, ws, m, PT, CT, Z, nl)
        @test nl.n_edges > 7
        @test_noalloc f!(res, ws, m, PT, CT, Z, nl)
        @test_noalloc g!(res, ws, m, PT, CT, Z, nl)
    end
end

# The item "forces: threaded chunks give the serial chunk result" with PureBLAS routing the matrix
# products of the child and of this process: PureBLAS keeps its matrix-product scratch per task,
# and the workers of a workspace keep theirs from one evaluation to the next, so a threaded
# evaluation after the first allocates nothing here either. PureBLAS's own thread pool stays off
# (`PureBLAS.get_num_threads() == 1`), so the only threads are Julia's four.
@testitem "PureBLAS: threaded chunks give the serial chunk result" tags = [:pureblas] setup = [Checkpoints, LammpsReference] begin
    using PureBLAS, LinearAlgebra
    @test PureBLAS.is_active() && PureBLAS.get_num_threads() == 1
    include(pkgdir(PureMLIP, "test", "threads", "threaded_child.jl"))
    include(pkgdir(PureMLIP, "test", "threads", "threaded_parent.jl"))
    checkpoint = Checkpoints.fetch(:mpa0)
    child = joinpath(pkgdir(PureMLIP), "test", "threads", "threaded_child.jl")
    cmd = `$(Base.julia_cmd()) -t 4,0 --gcthreads=1,0 --project=$(Base.active_project()) $child $checkpoint pureblas`
    lines = split(read(cmd, String), '\n')
    @test any(==("PUREBLAS_THREADS\t1"), lines)
    check_threaded_child(lines, load_mace_checkpoint(checkpoint), LammpsReference.load(:mpa0_f64), LammpsReference.structure)
end
