# Speed-up of `mace_energy_forces!` and `mace_energy!` with `ThreadedChunks()` over the one-thread
# serial evaluation, at several Julia thread counts, on the PureMLIP of the active environment.
#
#   julia --project=<env> PureMLIP/bench/thread_scaling.jl
#
# The environment needs PureMLIP, JSON, Statistics and LinearAlgebra, and PureBLAS for
# SCALING_BLAS=pureblas. Each measurement runs in its own child process under
# `julia -t <n>,0 --gcthreads=1,0` (no interactive thread, one GC thread), with one BLAS thread;
# a PureBLAS child asserts that PureBLAS's own thread pool is off. One thread runs the default
# one-chunk workspace without `ThreadedChunks()`; n threads run `n · SCALING_CHUNKS_PER_THREAD`
# receiver chunks with it. macOS cannot pin threads to cores, so every round runs each thread
# count once, in an order that rotates from round to round, and a speed-up is taken within one
# round: the one-thread time of that round over the n-thread time of that round.
#
#   PUREMLIP_CHECKPOINT_DIR     directory with mace-mpa-0-medium.model
#   SCALING_CASE                "N:precision" (default "5:64")
#   SCALING_THREADS             thread counts (default "1,2,4,6,8"; must include 1)
#   SCALING_CHUNKS_PER_THREAD   receiver chunks per thread (default 1)
#   SCALING_ROUNDS              rounds to add in this run (default 5)
#   SCALING_ENTRIES             entry points (default "mace_energy_forces!,mace_energy!")
#   SCALING_WARMUP, SCALING_SAMPLES   evaluations per entry point before timing (default 3) and
#                               timed (default 3; a child reports their median)
#   SCALING_BLAS                "openblas" (default) or "pureblas"
#   SCALING_OUT                 JSON path (default: a new file in the system temp directory). An
#                               existing file with the same settings is extended by the new rounds.
using JSON, LinearAlgebra, Printf, Statistics

swap_used_mb() = parse(Float64, match(r"used = ([0-9.]+)M", read(`sysctl -n vm.swapusage`, String))[1])
sysctl(key) = strip(read(`sysctl -n $key`, String))

# ---------------------------------------------------------------------------------------------
# Child: prints `RESULT <entry> <seconds>` per entry point and `BLAS <library file names>`.
# ---------------------------------------------------------------------------------------------
function child_main(N, T, threads, nchunks, entries, warmup, samples, blas)
    PM = Base.require(Main, :PureMLIP)
    include(joinpath(@__DIR__, "..", "test", "reference", "structures.jl"))
    BLAS.set_num_threads(1)
    Threads.nthreads() == threads || error("expected ", threads, " threads, got ", Threads.nthreads())
    if blas == "pureblas"
        pb = Base.require(Main, :PureBLAS)
        Base.invokelatest(pb.activate)
        Base.invokelatest(pb.is_active) || error("PureBLAS did not become active")
        BLAS.set_num_threads(1)
        Base.invokelatest(pb.get_num_threads) == 1 || error("the PureBLAS thread pool is on")
    end
    Base.invokelatest(child_measure, PM, N, T, threads, nchunks, entries, warmup, samples)
    println("BLAS\t", join((basename(l.libname) for l in BLAS.get_config().loaded_libs), ","))
    return
end

function child_measure(PM, N, ::Type{T}, threads, nchunks, entries, warmup, samples) where {T}
    ck = PM.load_mace_checkpoint(joinpath(ENV["PUREMLIP_CHECKPOINT_DIR"], "mace-mpa-0-medium.model"))
    cu = Main.cu_bench(N)
    model = PM.MACEModel{T}(ck)
    P, C, z = Matrix{T}(cu.positions'), Matrix{T}(cu.cell'), cu.atomic_numbers
    nl = PM.neighbor_list(P, C, model.r_max)
    ws = PM.MACEWorkspace{T}(model, length(z), nl.n_edges; elements = z, nchunks)
    res = PM.MACEResult{T}(length(z))
    extra = threads == 1 ? () : (PM.ThreadedChunks(),)
    for name in entries
        f = name == "mace_energy_forces!" ? PM.mace_energy_forces! : PM.mace_energy!
        for _ in 1:warmup
            f(res, ws, model, P, C, z, nl, extra...)
        end
        t = median([@elapsed(f(res, ws, model, P, C, z, nl, extra...)) for _ in 1:samples])
        println("RESULT\t", name, "\t", t)
    end
    return
end

# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------
function run_child(N, T, threads, nchunks, entries, warmup, samples, blas)
    tflags = `-t $threads,0 --gcthreads=1,0`
    spec = "$N:$(T === Float64 ? 64 : 32)"
    cmd = `$(Base.julia_cmd()) $tflags --project=$(Base.active_project()) $(@__FILE__) --child $spec $threads $nchunks $(join(entries, ',')) $warmup $samples $blas`
    out = read(setenv(cmd, merge(ENV, Dict("OPENBLAS_NUM_THREADS" => "1"))), String)
    times = Dict{String, Float64}()
    libs = String[]
    for line in split(out, '\n')
        f = split(line, '\t')
        f[1] == "RESULT" && (times[f[2]] = parse(Float64, f[3]))
        f[1] == "BLAS" && (libs = split(f[2], ','))
    end
    return times, libs
end

function driver_main()
    a, b = split(get(ENV, "SCALING_CASE", "5:64"), ':')
    N, T = parse(Int, a), b == "64" ? Float64 : Float32
    threads = parse.(Int, split(get(ENV, "SCALING_THREADS", "1,2,4,6,8"), ','))
    1 in threads || error("SCALING_THREADS must include 1")
    per = parse(Int, get(ENV, "SCALING_CHUNKS_PER_THREAD", "1"))
    rounds = parse(Int, get(ENV, "SCALING_ROUNDS", "5"))
    entries = split(get(ENV, "SCALING_ENTRIES", "mace_energy_forces!,mace_energy!"), ',')
    warmup = parse(Int, get(ENV, "SCALING_WARMUP", "3"))
    samples = parse(Int, get(ENV, "SCALING_SAMPLES", "3"))
    blas = get(ENV, "SCALING_BLAS", "openblas")
    blas in ("openblas", "pureblas") || error("SCALING_BLAS must be openblas or pureblas")
    out = get(ENV, "SCALING_OUT", tempname() * "_thread_scaling.json")
    nchunks(n) = n == 1 ? 1 : n * per
    tree = dirname(dirname(@__DIR__))
    commit = readchomp(`git -C $tree rev-parse --short=12 HEAD`)
    isempty(readchomp(`git -C $tree status --porcelain --untracked-files=no`)) || (commit *= "-dirty")
    settings = Dict(
        "case" => "cu_bench($N)", "precision" => string(T), "threads" => threads, "chunks_per_thread" => per,
        "entries" => entries, "warmup" => warmup, "samples" => samples, "blas" => blas, "commit" => commit,
    )
    done = isfile(out) ? JSON.parsefile(out) : nothing
    if done !== nothing
        all(k -> done["settings"][k] == settings[k], keys(settings)) || error("$out holds other settings")
    end
    kept = done === nothing ? Any[] : Any[r for r in done["rounds"]]
    libs = String[]
    sw0 = swap_used_mb()
    for r in (length(kept) + 1):(length(kept) + rounds)
        # Rotate the order by one each round and reverse it every other round, so each thread count
        # takes every position.
        order = circshift(threads, r - 1)
        iseven(r) && reverse!(order)
        times = Dict{String, Any}()
        for n in order
            t, libs = run_child(N, T, n, nchunks(n), entries, warmup, samples, blas)
            times[string(n)] = t
        end
        push!(kept, Dict("round" => r, "order" => order, "seconds" => times))
        @printf("round %d (%s): %s\n", r, join(order, ","), join((@sprintf("%s %s %.3fs", n, e, times[string(n)][e]) for n in threads for e in entries), "  "))
    end
    sw1 = swap_used_mb()
    summary = [
        Dict(
            "threads" => n, "nchunks" => nchunks(n), "entry_point" => e,
            "speedups" => [k["seconds"]["1"][e] / k["seconds"][string(n)][e] for k in kept],
        ) for e in entries for n in threads if n != 1
    ]
    for s in summary
        x = s["speedups"]
        s["median"], s["min"], s["max"] = median(x), minimum(x), maximum(x)
    end
    doc = Dict(
        "schema" => "puremlip-thread-scaling-v1", "settings" => settings,
        "machine" => sysctl("machdep.cpu.brand_string"),
        "cores" => Dict(
            sysctl("hw.perflevel$l.name") => parse(Int, sysctl("hw.perflevel$l.physicalcpu")) for l in 0:(parse(Int, sysctl("hw.nperflevels")) - 1)
        ),
        "julia_version" => string(VERSION), "julia_flags" => "-t <n>,0 --gcthreads=1,0", "blas_threads" => 1,
        "pureblas_threads" => blas == "pureblas" ? 1 : nothing, "blas_libraries" => libs,
        "speedup" => "one-thread serial seconds over n-thread seconds of the same round",
        "swap_mb" => vcat(done === nothing ? Float64[] : done["swap_mb"], [sw0, sw1]),
        "rounds" => kept, "summary" => summary,
    )
    open(io -> (JSON.json(io, doc; pretty = true); println(io)), out, "w")
    for s in summary
        @printf("%-20s %d threads (%d chunks): median %.2f  min %.2f  max %.2f  (%d rounds)\n", s["entry_point"], s["threads"], s["nchunks"], s["median"], s["min"], s["max"], length(kept))
    end
    @printf("swap %.1f -> %.1f MB\n", sw0, sw1)
    println("results written to ", out)
    return
end

if !isempty(ARGS) && ARGS[1] == "--child"
    a, b = split(ARGS[2], ':')
    child_main(
        parse(Int, a), b == "64" ? Float64 : Float32, parse(Int, ARGS[3]), parse(Int, ARGS[4]), split(ARGS[5], ','),
        parse(Int, ARGS[6]), parse(Int, ARGS[7]), ARGS[8]
    )
else
    driver_main()
end
