# Oracle: the serial evaluation with the same receiver chunks, which sums every quantity in the
# same order as the tasks of `ThreadedChunks()`, so the two must agree bit for bit; the one-chunk
# evaluation, within a rounding bound; the LAMMPS-MACE references; and finite differences of the
# threaded energy. The test process runs one thread, so a child process under `julia -t 4,0`
# evaluates the cases of `threads/threaded_child.jl`, and `threads/threaded_parent.jl` checks them.
@testitem "forces: threaded chunks give the serial chunk result" setup = [Checkpoints, LammpsReference] begin
    using LinearAlgebra
    include(pkgdir(PureMLIP, "test", "threads", "threaded_child.jl"))
    include(pkgdir(PureMLIP, "test", "threads", "threaded_parent.jl"))
    checkpoint = Checkpoints.fetch(:mpa0)
    child = joinpath(pkgdir(PureMLIP), "test", "threads", "threaded_child.jl")
    cmd = `$(Base.julia_cmd()) -t 4,0 --gcthreads=1,0 --project=$(Base.active_project()) $child $checkpoint`
    lines = split(read(setenv(cmd, merge(ENV, Dict("OPENBLAS_NUM_THREADS" => "1"))), String), '\n')
    # The child runs one BLAS thread, and so do the serial evaluations here.
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    check_threaded_child(lines, load_mace_checkpoint(checkpoint), LammpsReference.load(:mpa0_f64), LammpsReference.structure)
    BLAS.set_num_threads(blas_threads)
end

# Oracle: the serial evaluation with the same receiver chunks. On one thread the workers of
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
            @test_throws "model: no receiver-chunk phase 5" PureMLIP.Model._chunk_phase!(5, re, ws, m, Z, nl, 1)
            close(ws)
        end
        # The workers take the job's neighbor-list type from the model, so a list of another
        # element type is refused.
        nl64 = neighbor_list(Matrix{Float64}(P), Matrix{Float64}(C), m.r_max)
        if T == Float32
            ws = MACEWorkspace{T}(m, n, nl.n_edges; nchunks = 2, elements = Z)
            @test_throws "model: ThreadedChunks() needs a neighbor list of the element type of the model" mace_energy!(
                MACEResult{T}(n), ws, m, P, C, Z, nl64, ThreadedChunks()
            )
            close(ws)
        end
    end
end

# A test job for the chunk workers: chunk `c` of phase `p` records `p` in `seen[c]` after `delay`
# seconds, or throws when `c` is `failing`.
@testmodule ChunkJobs begin
    import PureMLIP
    struct TestJob
        seen::Vector{Int}
        failing::Int
        delay::Float64
    end
    function PureMLIP.Model._chunk!(job::TestJob, phase::Int, c::Int)
        c == job.failing && error("chunk $c failed")
        sleep(job.delay)
        job.seen[c] = phase
        return nothing
    end
    workers() = PureMLIP.Model.ChunkWorkers{TestJob}()
    run!(w, job, phase, n) = PureMLIP.Model._run_chunks!(w, job, phase, n)
    tasks(w) = w.pool.tasks
end

# A chunk that throws: the call rethrows its error only after every other chunk has finished, so
# no worker still writes the job's buffers when the caller sees the error; the failed worker is
# replaced by the next call.
@testitem "threaded chunks: a failing chunk is rethrown after every chunk has finished" setup = [ChunkJobs] begin
    n = 4
    w = ChunkJobs.workers()
    for failing in 1:n
        job = ChunkJobs.TestJob(zeros(Int, n), failing, 0.05)
        if failing == n
            # The last chunk runs on the calling task, so its own error propagates.
            @test_throws "chunk $n failed" ChunkJobs.run!(w, job, 1, n)
        else
            @test_throws TaskFailedException ChunkJobs.run!(w, job, 1, n)
            @test istaskfailed(ChunkJobs.tasks(w)[failing])
        end
        @test all(job.seen[c] == 1 for c in 1:n if c != failing)
        # The next call runs every chunk, on a new worker where one failed.
        ok = ChunkJobs.TestJob(zeros(Int, n), 0, 0.0)
        ChunkJobs.run!(w, ok, 2, n)
        @test ok.seen == fill(2, n)
    end
    @test_throws "model: a chunk phase must be positive, got 0" ChunkJobs.run!(w, ChunkJobs.TestJob(zeros(Int, n), 0, 0.0), 0, n)
    close(w)
    @test all(istaskdone, ChunkJobs.tasks(w))
end

# The workers outlive a call and are reused; `close` stops them, a later call starts them again,
# and the workers of a workspace that is no longer reachable stop without `close`.
@testitem "threaded chunks: workers are reused, closed and stopped with their owner" setup = [ChunkJobs, Checkpoints] begin
    n = 3
    w = ChunkJobs.workers()
    @test isempty(ChunkJobs.tasks(w))
    ChunkJobs.run!(w, ChunkJobs.TestJob(zeros(Int, n), 0, 0.0), 1, n)
    first_tasks = copy(ChunkJobs.tasks(w))
    @test length(first_tasks) == n - 1 && !any(istaskdone, first_tasks)
    job = ChunkJobs.TestJob(zeros(Int, n), 0, 0.0)
    ChunkJobs.run!(w, job, 3, n)
    @test job.seen == fill(3, n)
    @test ChunkJobs.tasks(w) == first_tasks                 # the same tasks ran the second call
    @test isempty(w.pool.job)                               # the job is not kept after the call
    close(w)
    @test all(istaskdone, first_tasks) && !any(istaskfailed, first_tasks)
    ChunkJobs.run!(w, job, 4, n)
    @test job.seen == fill(4, n) && !any(istaskdone, ChunkJobs.tasks(w))
    close(w)
    close(w)                                                # closing stopped workers does nothing
    @test all(istaskdone, ChunkJobs.tasks(w))

    # A workspace that is dropped after a threaded evaluation: its finalizer stops the workers.
    include(pkgdir(PureMLIP, "test", "reference", "structures.jl"))
    cu = cu_bench(2)
    m = MACEModel{Float64}(load_mace_checkpoint(Checkpoints.fetch(:mpa0)))
    P, C, Z = Matrix(cu.positions'), Matrix(cu.cell'), cu.atomic_numbers
    nl = neighbor_list(P, C, m.r_max)
    function evaluate_and_drop()
        ws = MACEWorkspace{Float64}(m, length(Z), nl.n_edges; nchunks = 3, elements = Z)
        mace_energy!(MACEResult{Float64}(length(Z)), ws, m, P, C, Z, nl, ThreadedChunks())
        return copy(ws.workers.pool.tasks)
    end
    dropped = evaluate_and_drop()
    @test length(dropped) == 2 && !any(istaskdone, dropped)
    deadline = time() + 30
    while !all(istaskdone, dropped) && time() < deadline
        GC.gc(true)
        sleep(0.05)
    end
    @test all(istaskdone, dropped) && !any(istaskfailed, dropped)
end
