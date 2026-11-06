# Runs the juliac --trim build and the resulting executable on MP-0b3. Slow, so it is skipped
# unless PUREMLIP_JULIAC_BUILD=1 (see test/runtests.jl). Unlike TrimCheck's @validate, which
# infers each entry point in isolation, the build is the whole-program trim verification.
@testitem "juliac --trim build of mace_inspect" tags = [:juliac] setup = [Checkpoints] begin
    root = pkgdir(PureMLIP)
    buildjl = joinpath(root, "juliac", "build.jl")
    exe = joinpath(root, "juliac", "build", "bin", "mace_inspect" * (Sys.iswindows() ? ".exe" : ""))
    # A ReTestItems/TestItemRunner worker restricts JULIA_LOAD_PATH; the juliac subprocess
    # needs the default path (including @stdlib).
    cmd = addenv(
        `$(Base.julia_cmd()) --startup-file=no --project=$root $buildjl`,
        "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing
    )
    @test success(pipeline(cmd; stdout = stderr, stderr = stderr))
    @test isfile(exe)

    out = read(`$exe $(Checkpoints.fetch(:mp0b3))`, String)
    @test occursin("model_class=mace.modules.models.ScaleShiftMACE\n", out)
    @test occursin("state_tensors=89\n", out)
    @test occursin("num_parameters=9063204\n", out)

    err = IOBuffer()
    @test !success(pipeline(ignorestatus(`$exe`); stderr = err))
    @test occursin("usage:", String(take!(err)))
end

# The trimmed executable evaluates forces with `ThreadedChunks()` on two chunks and two threads.
# `-t 2` would reach the program as an argument, so the thread count comes from the environment.
# The untrimmed run of the same entry point is the oracle, bit for bit.
@testitem "juliac --trim build of mace_threads" tags = [:juliac] begin
    root = pkgdir(PureMLIP)
    buildjl = joinpath(root, "juliac", "build.jl")
    exe = joinpath(root, "juliac", "build", "bin", "mace_threads" * (Sys.iswindows() ? ".exe" : ""))
    entry = joinpath(root, "juliac", "entry_threads.jl")
    cmd = addenv(
        `$(Base.julia_cmd()) --startup-file=no --project=$root $buildjl mace_threads entry_threads.jl`,
        "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing
    )
    @test success(pipeline(cmd; stdout = stderr, stderr = stderr))
    @test isfile(exe)

    mktempdir() do dir
        # The program reads the checkpoint from a file; the in-memory builder is not trimmable.
        ck = joinpath(dir, "tiny.model")
        write(ck, PureMLIP.tiny_mace_checkpoint())
        # A run that does not finish in 120 s counts as a hang.
        function run_with_timeout(c::Cmd)
            out, err = joinpath(dir, "out.txt"), joinpath(dir, "err.txt")
            p = run(pipeline(ignorestatus(c); stdout = out, stderr = err); wait = false)
            t0 = time()
            while process_running(p) && time() - t0 < 120
                sleep(0.1)
            end
            hung = process_running(p)
            hung && kill(p)
            wait(p)
            return (; hung, ok = !hung && success(p), out = read(out, String), err = read(err, String))
        end
        for mode in ("close", "open", "dropped")
            @testset "$mode" begin
                trimmed = run_with_timeout(addenv(`$exe $mode $ck`, "JULIA_NUM_THREADS" => "2"))
                untrimmed = run_with_timeout(
                    addenv(
                        `$(Base.julia_cmd()) -t 2 --startup-file=no --project=$root $entry $mode $ck`,
                        "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing
                    )
                )
                @test !trimmed.hung && trimmed.ok
                @test untrimmed.ok
                @test isempty(trimmed.err)
                @test startswith(trimmed.out, "threads=2\nenergy=")
                @test trimmed.out == untrimmed.out
                # The finalizer of a dropped workspace stops its workers in the trimmed program.
                mode == "dropped" && @test occursin("workers_stopped=true\n", trimmed.out)
            end
        end
    end
end
