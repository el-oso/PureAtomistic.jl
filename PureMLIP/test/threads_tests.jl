# Oracle: the serial evaluation with the same receiver chunks, which sums every quantity in the
# same order as the tasks of `ThreadedChunks()`, so the two must agree bit for bit; and the
# one-chunk evaluation, within a rounding bound. The test process runs one thread, so a child
# process under `julia -t 4,0` evaluates the cases of `threads/threaded_child.jl`.
@testitem "forces: threaded chunks give the serial chunk result" setup = [Checkpoints] begin
    using LinearAlgebra
    include(pkgdir(PureMLIP, "test", "threads", "threaded_child.jl"))
    checkpoint = Checkpoints.fetch(:mpa0)
    child = joinpath(pkgdir(PureMLIP), "test", "threads", "threaded_child.jl")
    cmd = `$(Base.julia_cmd()) -t 4,0 --gcthreads=1,0 --project=$(Base.active_project()) $child $checkpoint`
    lines = split(read(setenv(cmd, merge(ENV, Dict("OPENBLAS_NUM_THREADS" => "1"))), String), '\n')
    @test any(==("NTHREADS\t4"), lines)
    rows = Dict(parse(Int, f[2]) => f for f in (split(l, '\t') for l in lines) if f[1] == "CASE")
    cases = threaded_cases()
    @test sort(collect(keys(rows))) == eachindex(cases)
    ck = load_mace_checkpoint(checkpoint)
    models = Dict(T => MACEModel{T}(ck) for T in (Float64, Float32))
    # The child runs one BLAS thread, and so do the serial evaluations here.
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    # With OpenBLAS, the BLAS of this environment, each `Threads.@spawn` allocates its task (224
    # bytes), the closure it runs (1,808 bytes: the closure holds the workspace struct inline) and the
    # task's wait condition (96 bytes). Each evaluation runs two parallel phases, with one task fewer
    # than chunks in each, and allocates nothing else. A BLAS with task-local scratch (PureBLAS)
    # allocates that scratch in every new task as well.
    task_bytes = 2128
    for (k, (name, T, nchunks, P, C, Z)) in enumerate(cases)
        f = rows[k]
        m = models[T]
        nl = neighbor_list(P, C, m.r_max)
        n = nl.n_atoms
        r = mace_energy_forces!(MACEResult{T}(n), MACEWorkspace{T}(m, n, nl.n_edges; nchunks, elements = Z), m, P, C, Z, nl)
        @testset "$name, $T, $nchunks chunks" begin
            @test frombits(T, f[3]) == r.energy
            @test frombits(T, f[4], n) == r.node_energies
            @test frombits(T, f[5], (3, n)) == r.forces
            @test frombits(T, f[6], (3, 3)) == r.virial
            @test frombits(T, f[7], (3, 3)) == r.stress
            @test frombits(T, f[8]) == r.energy           # mace_energy! with ThreadedChunks()
            @test f[9] == "true"                           # a second threaded evaluation, same bits
            @test parse(Int, f[10]) <= 2 * (nchunks - 1) * task_bytes
            if name == "one atom"
                @test nl.n_edges == 0
                @test r.energy == m.e0[findfirst(==(29), m.z_table)]   # no edges: E0 exactly
                @test r.forces == zeros(T, 3, 1) && r.virial == zeros(T, 3, 3)
            else
                ref = mace_energy_forces!(MACEResult{T}(n), MACEWorkspace{T}(m, n, nl.n_edges; elements = Z), m, P, C, Z, nl)
                # Rounding bound 10·eps(T)·N relative to the largest magnitude of each quantity,
                # with N = 1280 the largest inner dimension of any matrix product in the model.
                # Measured over these cases: at most 12.3·eps(T) in Float64 (35 chunks) and
                # 5.3·eps(T) in Float32.
                tol = 10 * eps(T) * 1280
                @test abs(r.energy - ref.energy) <= tol * abs(ref.energy)
                @test maximum(abs, r.node_energies - ref.node_energies) <= tol * maximum(abs, ref.node_energies)
                @test maximum(abs, r.forces - ref.forces) <= tol * maximum(abs, ref.forces)
                @test maximum(abs, r.virial - ref.virial) <= tol * maximum(abs, ref.virial)
            end
        end
    end
    BLAS.set_num_threads(blas_threads)
end

# Oracle: the serial evaluation with the same receiver chunks. On one thread the tasks of
# `ThreadedChunks()` run one after another on the calling thread, in the order the scheduler picks.
@testitem "forces: threaded chunks on one thread give the serial chunk result" setup = [Checkpoints] begin
    include(pkgdir(PureMLIP, "test", "reference", "structures.jl"))
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0))
    cu = cu_bench(2)
    for T in (Float64, Float32)
        m = MACEModel{T}(ck)
        P, C, Z = Matrix{T}(cu.positions'), Matrix{T}(cu.cell'), cu.atomic_numbers
        n = length(Z)
        nl = neighbor_list(P, C, m.r_max)
        for nchunks in (1, 3)
            ws = MACEWorkspace{T}(m, n, nl.n_edges; nchunks, elements = Z)
            r = mace_energy_forces!(MACEResult{T}(n), ws, m, P, C, Z, nl)
            rt = mace_energy_forces!(MACEResult{T}(n), ws, m, P, C, Z, nl, ThreadedChunks())
            @test rt.energy == r.energy && rt.node_energies == r.node_energies
            @test rt.forces == r.forces && rt.virial == r.virial && rt.stress == r.stress
            re = mace_energy!(MACEResult{T}(n), ws, m, P, C, Z, nl, ThreadedChunks())
            @test re.energy == r.energy && re.node_energies == r.node_energies
        end
    end
end

# A chunk that throws: the call rethrows its error only after every other chunk has finished, so
# no task still writes the workspace when the caller sees the error. The chunks here are test
# functions handed to the internal task runner; the others sleep first, so they are still
# running when the failing chunk throws.
@testitem "threaded chunks: a failing chunk is rethrown after every task has finished" begin
    spawn_chunks! = PureMLIP.Model._spawn_chunks!
    n = 4
    for failing in 1:n
        done = falses(n)
        work(c) = c == failing ? error("chunk $c failed") : (sleep(0.05); done[c] = true; nothing)
        if failing == n
            # The last chunk runs on the calling task, so its own error propagates.
            @test_throws "chunk $n failed" spawn_chunks!(work, 1, n)
        else
            @test_throws TaskFailedException spawn_chunks!(work, 1, n)
        end
        @test all(done[c] for c in 1:n if c != failing)
    end
end
