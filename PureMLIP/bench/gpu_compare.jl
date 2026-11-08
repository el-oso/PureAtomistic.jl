# Interleaved speed comparison of the device MACE evaluation and MD step against threaded CPU
# PureMLIP and against an external program, with the cost of the per-step readback.
#
#   GPU_BACKEND=Metal PUREMLIP_CHECKPOINT_DIR=<dir> julia --project=PureAtomistic/bench/gpu/metal PureMLIP/bench/gpu_compare.jl
#
# Each arm runs in its own child process, once per round, and the order of the arms rotates from
# round to round so that a steady drift of the machine does not favor one arm. The arms:
#   gpu       this checkout on the device (GPU_BACKEND), environment PureAtomistic/bench/gpu/<backend>
#   cpu       this checkout, `-t <n>,0 --gcthreads=1,0` with `ThreadedChunks()` and one BLAS thread,
#             optionally with PureBLAS (PUREMLIP_GPU_COMPARE_PUREBLAS); a temporary environment
#   external  the command in PUREMLIP_GPU_EXTERNAL_CMD (a program outside the repositories); it
#             reads the cell and precision from PUREMLIP_GPU_EXTERNAL_N and
#             PUREMLIP_GPU_EXTERNAL_PRECISION and prints one JSON line `{"seconds": ...}`, with
#             `"peak_bytes"` when it knows its peak memory, or `{"fits": false, "peak_bytes": ...}`
#             when the cell does not fit its memory limit
#
# Modes (PUREMLIP_GPU_COMPARE_MODE):
#   eval      (default) one energy + forces + stress evaluation, `mace_energy_forces!`; the device
#             evaluation is synchronized before the clock stops
#   md        one velocity Verlet step of a `Simulation` (gpu, cpu)
#   readback  the cost of the status read of a device step: blocks of device steps with and without
#             it, interleaved in one process (gpu only)
#
#   GPU_BACKEND                    Metal (default), CUDA or AMDGPU
#   PUREMLIP_CHECKPOINT_DIR        directory with mace-mpa-0-medium.model
#   PUREMLIP_GPU_COMPARE_CELLS     "N:arm+arm,..." with N the cu_bench size (default
#                                  "5:gpu+cpu+external,14:gpu+cpu"; readback "5:gpu,14:gpu")
#   PUREMLIP_GPU_COMPARE_ROUNDS    rounds (default 7)
#   PUREMLIP_GPU_COMPARE_THREADS   Julia threads of the cpu arm (default 6)
#   PUREMLIP_GPU_COMPARE_PUREBLAS  PureBLAS checkout (a directory) for the cpu arm; without it OpenBLAS
#   PUREMLIP_GPU_EXTERNAL_CMD      command of the external arm
#   PUREMLIP_GPU_COMPARE_LOCK      script with `acquire` (prints a token) and `release <token>`; held
#                                  around each round
#   PUREMLIP_GPU_COMPARE_CAL_TOLERANCE  calibration tolerances "<within>,<pair>" (default "0.03,0.01", the
#                                  values of ab_compare.jl)
#   PUREMLIP_GPU_COMPARE_CAL_THREADED  within-child tolerance of the multi-threaded cpu arm (default 0.5);
#                                  its calibration is not compared with the other children
#   PUREMLIP_GPU_COMPARE_OUT       JSON output path
#
# The precision is Float32. The speed-up of a round is the reference time over the time of the gpu
# arm (above 1: the gpu arm is faster); the record gives the minimum, median and maximum over the
# rounds. A round is discarded and measured again when swap usage grew while it ran or when the
# calibration loop (a fixed scalar loop independent of PureMLIP, run on one thread before and after
# each measurement) shows the children ran at different speeds.
#
# Peak memory: the gpu arm reports the vendor's allocated device memory (Metal:
# `device().currentAllocatedSize`, the memory the device holds for all its resources, checked
# against a known allocation at start-up) sampled after setup and after every timed call, and the
# maximum resident set size; the cpu arm reports the maximum resident set size.
using JSON, LinearAlgebra, Printf, Statistics

const SAMPLES = 3
const WARMUP_EVALS = 5
const CAL_REPEATS = 30
const MAX_DISCARDS = 20
const CAL_WITHIN_TOLERANCE, CAL_PAIR_TOLERANCE = parse.(Float64, split(get(ENV, "PUREMLIP_GPU_COMPARE_CAL_TOLERANCE", "0.03,0.01"), ','))
const CAL_THREADED_TOLERANCE = parse(Float64, get(ENV, "PUREMLIP_GPU_COMPARE_CAL_THREADED", "0.5"))
const THREADS = parse(Int, get(ENV, "PUREMLIP_GPU_COMPARE_THREADS", "6"))
const BACKEND_NAME = get(ENV, "GPU_BACKEND", "Metal")
const MD_STEPS = 5
const READBACK_BLOCK = 20

function swap_used_mb()
    if Sys.isapple()
        return parse(Float64, match(r"used = ([0-9.]+)M", read(`sysctl -n vm.swapusage`, String))[1])
    elseif Sys.islinux()
        kb(key) = parse(Float64, match(Regex("^$key:\\s+(\\d+) kB", "m"), read("/proc/meminfo", String))[1])
        return (kb("SwapTotal") - kb("SwapFree")) / 1024
    end
    return error("swap usage is read on macOS and Linux only")
end

# Time of a fixed scalar loop (minimum over repeats); it moves only with the speed of the core.
function calibrate()
    best = Inf
    for _ in 1:CAL_REPEATS
        t = time_ns()
        x = 1.0
        for i in 1:1_000_000
            x = muladd(x, 0.9999999, 1.0e-9 * i)
        end
        best = min(best, (time_ns() - t) / 1.0e9 + 0 * x)
    end
    return best
end

# ---------------------------------------------------------------------------------------------
# Children
# ---------------------------------------------------------------------------------------------
function load_backend()
    vendor = getfield(Main, Symbol(BACKEND_NAME))
    Base.invokelatest(vendor.functional) || error(BACKEND_NAME, " is not functional on this machine")
    backend = BACKEND_NAME == "Metal" ? vendor.MetalBackend() : BACKEND_NAME == "CUDA" ? vendor.CUDABackend() : vendor.ROCBackend()
    return vendor, backend
end

# Bytes the device holds now. Metal: the documented `currentAllocatedSize` of the device.
function device_bytes(vendor)
    BACKEND_NAME == "Metal" && return Int(Base.invokelatest(() -> vendor.device().currentAllocatedSize))
    error("no device-memory query is implemented for ", BACKEND_NAME)
end

# Allocates a known buffer and checks that the query moves by at least that much.
function check_device_query(vendor, backend)
    before = device_bytes(vendor)
    n = 256 * 2^20
    A = Base.invokelatest(() -> Main.KernelAbstractions.zeros(backend, UInt8, n))
    Base.invokelatest(() -> Main.KernelAbstractions.synchronize(backend))
    delta = device_bytes(vendor) - before
    delta >= n || error("the device-memory query moved by ", delta, " bytes for a ", n, "-byte allocation")
    println("QUERY_CHECK\t", n, "\t", delta)
    A = nothing
    return
end

function setup_cpu_threads(pureblas)
    BLAS.set_num_threads(1)
    if pureblas
        pb = Base.require(Main, :PureBLAS)
        Base.invokelatest(pb.activate)
        Base.invokelatest(pb.is_active) || error("PureBLAS did not become active")
        BLAS.set_num_threads(1)
        Base.invokelatest(pb.get_num_threads) == 1 || error("the PureBLAS thread pool is on")
    end
    return
end

function child_main(mode, arm, N, threads, pureblas)
    @eval Main using Random
    PM = Base.require(Main, :PureMLIP)
    include(joinpath(pkgdir(PM), "test", "reference", "structures.jl"))
    arm == "cpu" && (Threads.nthreads() == threads || error("expected ", threads, " threads, got ", Threads.nthreads()))
    arm == "cpu" && setup_cpu_threads(pureblas)
    if mode == "eval"
        Base.invokelatest(arm == "gpu" ? eval_gpu : eval_cpu, PM, N)
    else
        PA = Base.require(Main, :PureAtomistic)
        Base.invokelatest(mode == "readback" ? readback_gpu : arm == "gpu" ? md_gpu : md_cpu, PM, PA, N)
    end
    println("MAXRSS\t", Sys.maxrss())
    println("BLAS\t", join((basename(l.libname) for l in BLAS.get_config().loaded_libs), ","))
    return
end

checkpoint(PM) = PM.load_mace_checkpoint(joinpath(ENV["PUREMLIP_CHECKPOINT_DIR"], "mace-mpa-0-medium.model"))

# Median of `SAMPLES` timed calls after warm-up, with calibration before and after.
function timed(f, after = () -> nothing; samples = SAMPLES, onsample = () -> nothing)
    for _ in 1:WARMUP_EVALS
        f()
    end
    after()
    cal0 = calibrate()
    ts = Float64[]
    for _ in 1:samples
        t0 = time_ns()
        f()
        after()
        push!(ts, (time_ns() - t0) / 1.0e9)
        onsample()
    end
    cal1 = calibrate()
    println("RESULT\t", median(ts), "\t", cal0, "\t", cal1)
    return
end

function eval_cpu(PM, N)
    T = Float32
    cu = Main.cu_bench(N)
    model = PM.MACEModel{T}(checkpoint(PM))
    P, C, z = Matrix{T}(cu.positions'), Matrix{T}(cu.cell'), cu.atomic_numbers
    nl = PM.neighbor_list(P, C, model.r_max; extra_capacity = 64)
    nch = Threads.nthreads()
    ws = PM.MACEWorkspace{T}(model, length(z), nl.n_edges; elements = z, nchunks = nch)
    res = PM.MACEResult{T}(length(z))
    println("INFO\t", JSON.json(Dict("atoms" => length(z), "edges" => nl.n_edges, "nchunks" => nch, "edge_chunk" => ws.edge_chunk)))
    timed(() -> PM.mace_energy_forces!(res, ws, model, P, C, z, nl, PM.ThreadedChunks()))
    return
end

function eval_gpu(PM, N)
    T = Float32
    vendor, backend = load_backend()
    KA = Main.KernelAbstractions
    check_device_query(vendor, backend)
    up(A) = (D = KA.allocate(backend, eltype(A), size(A)); copyto!(D, A); D)
    cu = Main.cu_bench(N)
    model = PM.MACEModel{T}(checkpoint(PM))
    z = cu.atomic_numbers
    dm = PM.device_model(model, backend; elements = sort!(unique(z)))
    P, C = up(Matrix{T}(cu.positions')), up(Matrix{T}(cu.cell'))
    Z = up(Int32.(z))
    nl = PM.device_neighbor_list(P, C, Float64(model.r_max); extra_capacity_fraction = 0.05)
    ws = PM.device_workspace(dm, length(z), nl.capacity)
    res = PM.device_result(dm, length(z))
    peak = Ref(device_bytes(vendor))
    println("INFO\t", JSON.json(Dict("atoms" => length(z), "capacity" => nl.capacity, "edge_chunk" => ws.edge_chunk)))
    f = () -> PM.mace_energy_forces!(res, ws, dm, P, C, Z, nl)
    timed(f, () -> KA.synchronize(backend); onsample = () -> (peak[] = max(peak[], device_bytes(vendor))))
    println("PEAK_DEVICE\t", peak[])
    return
end

# The system of an MD step: a Cu crystal of `N` cells per edge at 300 K.
function md_system(PA, N)
    host = PA.Systems.fcc_cell(3.615f0, N, :Cu)
    PA.Systems.maxwell_boltzmann!(host, 300.0, Xoshiro(1))
    return host
end

# A MACEPotential evaluated on `ThreadedChunks()`: its workspace has one receiver chunk per thread.
function threaded_potential(PM, PA, model, host)
    T = Float32
    inner = PA.MACEPotential(model, host.positions, host.cell, host.species)
    inner.ws = PM.MACEWorkspace{T}(model, size(host.positions, 2), length(inner.nl.senders); elements = host.species, nchunks = Threads.nthreads())
    return inner
end

function md_cpu(PM, PA, N)
    T = Float32
    model = PM.MACEModel{T}(checkpoint(PM))
    host = md_system(PA, N)
    p = threaded_potential(PM, PA, model, host)
    # `step!` evaluates `energy_forces!` of the potential; this wrapper evaluates it on `ThreadedChunks()`.
    include_string(
        Main, """
        import PureAtomistic.Potentials, PureMLIP
        struct ThreadedMACE{P} <: Potentials.AbstractStressPotential
            p::P
        end
        Base.getproperty(q::ThreadedMACE, s::Symbol) = s === :p ? getfield(q, :p) : getproperty(getfield(q, :p), s)
        Potentials.n_atoms(q::ThreadedMACE) = Potentials.n_atoms(q.p)
        Potentials.cutoff(q::ThreadedMACE) = Potentials.cutoff(q.p)
        Potentials.stress(q::ThreadedMACE) = Potentials.stress(q.p)
        Potentials.node_energies(q::ThreadedMACE) = Potentials.node_energies(q.p)
        Potentials.prepare!(q::ThreadedMACE, x::AbstractMatrix, c::AbstractMatrix) = Potentials.prepare!(q.p, x, c)
        function Potentials.energy_forces!(q::ThreadedMACE, x::AbstractMatrix, c::AbstractMatrix)::Nothing
            PureMLIP.mace_energy_forces!(q.p.result, q.p.ws, q.p.model, x, c, q.p.atomic_numbers, q.p.nl, PureMLIP.ThreadedChunks())
            return nothing
        end
        """
    )
    Base.invokelatest(md_cpu_run, PA, p, host)
    return
end

function md_cpu_run(PA, p, host)
    tp = Main.ThreadedMACE(p)
    sim = PA.Simulation(host, tp, PA.VelocityVerlet(2.0); rng = Xoshiro(0))
    println("INFO\t", JSON.json(Dict("atoms" => size(host.positions, 2), "edges" => p.nl.n_edges, "nchunks" => Threads.nthreads())))
    timed(() -> PA.step!(sim); samples = MD_STEPS)
    p.n_regrows == 0 || error("the potential regrew during the timed steps")
    return
end

function md_gpu(PM, PA, N)
    T = Float32
    vendor, backend = load_backend()
    KA = Main.KernelAbstractions
    check_device_query(vendor, backend)
    model = PM.MACEModel{T}(checkpoint(PM))
    sys = PA.to_device(md_system(PA, N), backend)
    p = PA.device_potential(model, sys; skin = 0)
    sim = PA.Simulation(sys, p, PA.VelocityVerlet(2.0); rng = Xoshiro(0))
    peak = Ref(device_bytes(vendor))
    println("INFO\t", JSON.json(Dict("atoms" => size(sys.positions, 2), "capacity" => p.nl.capacity)))
    timed(() -> PA.step!(sim), () -> KA.synchronize(backend); samples = MD_STEPS, onsample = () -> (peak[] = max(peak[], device_bytes(vendor))))
    p.n_regrows == 0 || error("the potential regrew during the timed steps")
    println("PEAK_DEVICE\t", peak[])
    return
end

# Device steps with and without the status read, as blocks of `READBACK_BLOCK` steps closed by a
# synchronization, alternating inside one process. The step without the read launches the same
# kernels in the same order; it leaves the status word unread, so the host runs ahead of the device.
function readback_gpu(PM, PA, N)
    T = Float32
    vendor, backend = load_backend()
    KA = Main.KernelAbstractions
    ext = Base.get_extension(PA, :PureAtomisticGPUExt)
    model = PM.MACEModel{T}(checkpoint(PM))
    sys = PA.to_device(md_system(PA, N), backend)
    p = PA.device_potential(model, sys; skin = 0)
    sim = PA.Simulation(sys, p, PA.VelocityVerlet(2.0); rng = Xoshiro(0))
    h = T(2.0)
    function step(read)
        ext._kick!(sys, p, h / 2)
        ext._drift!(sys, p, h)
        PA.Potentials.prepare!(p, sys.positions, sys.cell)
        ext._evaluate!(p, sys.positions, sys.cell)
        read && (ext._read_status!(p); PM.check_device_status(p.ws, p.model, p.numbers, p.nl) == 0 || error("list overflow"))
        ext._kick!(sys, p, h / 2)
        return
    end
    block(read) = (
        t0 = time_ns(); for _ in 1:READBACK_BLOCK
            step(read)
        end; KA.synchronize(backend); (time_ns() - t0) / 1.0e9 / READBACK_BLOCK
    )
    for _ in 1:WARMUP_EVALS
        step(true)
    end
    KA.synchronize(backend)
    # A normal step, for reference against the step with the read written out here.
    reference() = (
        t0 = time_ns(); for _ in 1:READBACK_BLOCK
            PA.step!(sim)
        end; KA.synchronize(backend); (time_ns() - t0) / 1.0e9 / READBACK_BLOCK
    )
    cal0 = calibrate()
    with, without, ref = Float64[], Float64[], Float64[]
    for k in 1:SAMPLES
        # Alternate the order of the three variants from sample to sample.
        order = k % 2 == 1 ? (:with, :without, :ref) : (:ref, :without, :with)
        for v in order
            t = v === :with ? block(true) : v === :without ? block(false) : reference()
            push!(v === :with ? with : v === :without ? without : ref, t)
        end
    end
    cal1 = calibrate()
    PM.check_device_status(p.ws, p.model, p.numbers, p.nl) == 0 || error("the unread steps overflowed the list")
    println("INFO\t", JSON.json(Dict("atoms" => size(sys.positions, 2), "block_steps" => READBACK_BLOCK, "ref" => ref)))
    println("RESULT\t", median(with), "\t", cal0, "\t", cal1, "\t", median(without))
    return
end

# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------
git(dir, args...) = readchomp(`git -C $dir $args`)

function parse_cells(spec)
    return [(parse(Int, a), String.(split(b, '+'))) for (a, b) in split.(split(spec, ','), ':')]
end

# A temporary environment for the cpu arm: this checkout's PureMLIP and PureAtomistic, and PureBLAS.
function make_cpu_env(root, pureblas)
    dir = get(ENV, "PUREMLIP_GPU_COMPARE_CPU_ENV", "")
    isempty(dir) ? (dir = mktempdir()) : mkpath(dir)
    uuids = Dict(
        "JSON" => "682c06a0-de6a-54ab-a142-c8b1cf79cde6", "LinearAlgebra" => "37e2e46d-f89d-539d-b4ee-838fcccc9c8e",
        "Statistics" => "10745b16-79ce-11e8-11f9-7d13ad32a3b2", "Random" => "9a3f8284-a2c9-5f02-9a11-845980a1fd5c",
        "PureMLIP" => "e31974d8-f9c1-4d27-8220-1506e2a12d0a", "PureAtomistic" => "48bcfeca-9cba-41eb-9343-fd702b0553a7",
    )
    isempty(pureblas) || (uuids["PureBLAS"] = "cc9e14db-574f-4602-bf53-1167cc4b26d2")
    sources = Dict("PureMLIP" => joinpath(root, "PureMLIP"), "PureAtomistic" => joinpath(root, "PureAtomistic"))
    isempty(pureblas) || (sources["PureBLAS"] = abspath(pureblas))
    open(joinpath(dir, "Project.toml"), "w") do io
        println(io, "[deps]")
        foreach(((k, v),) -> println(io, k, " = \"", v, "\""), uuids)
        println(io, "\n[sources]")
        foreach(((k, v),) -> println(io, k, " = {path = \"", v, "\"}"), sources)
        println(io, "\n[compat]\nJSON = \"1.6\"")
    end
    run(`$(Base.julia_cmd()) --project=$dir -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"`)
    return dir
end

function run_child(env, mode, arm, N, threads, pureblas)
    tflags = arm == "cpu" ? `-t $threads,0 --gcthreads=1,0` : `-t 1`
    extra = Dict("OPENBLAS_NUM_THREADS" => "1")
    Sys.isapple() && !haskey(ENV, "HWLOC_COMPONENTS") && (extra["HWLOC_COMPONENTS"] = "-darwin")
    cmd = setenv(
        `$(Base.julia_cmd()) $tflags --project=$env $(@__FILE__) --child $mode $arm $N $threads $pureblas`,
        merge(ENV, extra),
    )
    out = read(cmd, String)
    r = Dict{String, Any}()
    for line in split(out, '\n')
        f = split(line, '\t')
        f[1] == "RESULT" && (r["seconds"] = parse(Float64, f[2]); r["cal"] = [parse(Float64, f[3]), parse(Float64, f[4])]; length(f) > 4 && (r["seconds_without_read"] = parse(Float64, f[5])))
        f[1] == "INFO" && (r["info"] = JSON.parse(f[2]))
        f[1] == "MAXRSS" && (r["maxrss_bytes"] = parse(Int, f[2]))
        f[1] == "PEAK_DEVICE" && (r["peak_device_bytes"] = parse(Int, f[2]))
        f[1] == "BLAS" && (r["blas"] = String(f[2]))
        f[1] == "QUERY_CHECK" && (r["query_check"] = [parse(Int, f[2]), parse(Int, f[3])])
    end
    haskey(r, "seconds") || error("child printed no result:\n", out)
    return r
end

function run_external(N)
    cmd = get(ENV, "PUREMLIP_GPU_EXTERNAL_CMD", "")
    isempty(cmd) && error("the external arm needs PUREMLIP_GPU_EXTERNAL_CMD")
    out = read(setenv(`sh -c $cmd`, merge(ENV, Dict("PUREMLIP_GPU_EXTERNAL_N" => string(N), "PUREMLIP_GPU_EXTERNAL_PRECISION" => "32"))), String)
    r = JSON.parse(last(filter(!isempty, split(out, '\n'))))
    r["fits"] = get(r, "fits", true)
    if r["fits"]
        r["seconds"] = Float64(r["seconds"])
    end
    return r
end

with_lock(f) = begin
    lock = get(ENV, "PUREMLIP_GPU_COMPARE_LOCK", "")
    isempty(lock) && return f()
    tok = readchomp(`$lock acquire`)
    try
        f()
    finally
        run(`$lock release $tok`)
    end
end

function round_discard_reason(results, sw0, sw1)
    sw1 > sw0 + 1 && return "swap usage grew during the round"
    meds = Float64[]
    for (arm, r) in results
        haskey(r, "cal") || continue
        spread = maximum(r["cal"]) / minimum(r["cal"]) - 1
        spread > (arm == "cpu" && THREADS > 1 ? CAL_THREADED_TOLERANCE : CAL_WITHIN_TOLERANCE) && return @sprintf("calibration varied by %.1f%% within the %s child", 100 * spread, arm)
        # The calibration loop of a multi-threaded child runs right after the threads' burst and
        # reflects where the scheduler places it; it is not compared with the other children.
        arm == "cpu" && THREADS > 1 || push!(meds, median(r["cal"]))
    end
    length(meds) > 1 && maximum(meds) / minimum(meds) - 1 > CAL_PAIR_TOLERANCE &&
        return @sprintf("calibration medians differ by %.1f%% between the children", 100 * (maximum(meds) / minimum(meds) - 1))
    return nothing
end

function driver_main()
    root = git(@__DIR__, "rev-parse", "--show-toplevel")
    mode = get(ENV, "PUREMLIP_GPU_COMPARE_MODE", "eval")
    mode in ("eval", "md", "readback") || error("PUREMLIP_GPU_COMPARE_MODE must be eval, md or readback")
    default_cells = mode == "eval" ? "5:gpu+cpu+external,14:gpu+cpu" : mode == "md" ? "5:gpu+cpu,14:gpu+cpu" : "5:gpu,14:gpu"
    cells = parse_cells(get(ENV, "PUREMLIP_GPU_COMPARE_CELLS", default_cells))
    rounds = parse(Int, get(ENV, "PUREMLIP_GPU_COMPARE_ROUNDS", "7"))
    threads = parse(Int, get(ENV, "PUREMLIP_GPU_COMPARE_THREADS", "6"))
    pureblas = get(ENV, "PUREMLIP_GPU_COMPARE_PUREBLAS", "")
    out = get(ENV, "PUREMLIP_GPU_COMPARE_OUT", joinpath(@__DIR__, "results", "gpu_compare_$(mode)_$(lowercase(BACKEND_NAME)).json"))
    gpu_env = joinpath(root, "PureAtomistic", "bench", "gpu", lowercase(BACKEND_NAME))
    arms_used = unique(reduce(vcat, last.(cells)))
    cpu_env = "cpu" in arms_used ? make_cpu_env(root, pureblas) : ""
    env_of(arm) = arm == "gpu" ? gpu_env : cpu_env
    head = git(root, "rev-parse", "--short=12", "HEAD") * (isempty(git(root, "status", "--porcelain", "--untracked-files=normal")) ? "" : "-dirty")
    pb_commit = isempty(pureblas) ? nothing : git(pureblas, "rev-parse", "--short=12", "HEAD")
    doc_cells = Any[]
    sw_first = swap_used_mb()
    write_doc = function ()
        doc = (;
            schema = "puremlip-gpu-compare-v1", mode, precision = "Float32", backend = BACKEND_NAME,
            machine = Sys.isapple() ? strip(read(`sysctl -n machdep.cpu.brand_string`, String)) : string(Sys.CPU_NAME),
            julia_version = string(VERSION), commit = head, cpu_threads = threads, cpu_threads_flags = "-t $threads,0 --gcthreads=1,0",
            blas_threads = 1, cpu_blas = isempty(pureblas) ? "OpenBLAS" : "PureBLAS", pureblas_commit = pb_commit, pureblas_active = !isempty(pureblas),
            device_memory_method = "largest currentAllocatedSize sampled after timed calls; not a high-water mark",
            samples_per_measurement = SAMPLES, rounds, swap_first_start_mb = sw_first, swap_last_end_mb = swap_used_mb(),
            speed_up = "reference time / gpu time per round; above 1 the gpu arm is faster",
            calibration_tolerance = (; within = CAL_WITHIN_TOLERANCE, pair = CAL_PAIR_TOLERANCE, threaded_within = CAL_THREADED_TOLERANCE),
            cells = doc_cells,
        )
        mkpath(dirname(out))
        open(io -> (JSON.json(io, doc; pretty = true); println(io)), out, "w")
        return
    end
    for (N, arms) in cells
        println("cell cu_bench($N): ", join(arms, ", "))
        # Untimed pass: pays compilation and finds out whether the external arm fits.
        fits = true
        ext_result = nothing
        for arm in arms
            if arm == "external"
                ext_result = r = run_external(N)
                r["fits"] || (fits = false; println("external arm does not fit at cu_bench($N): ", r))
            else
                run_child(env_of(arm), mode, arm, N, threads, !isempty(pureblas))
            end
        end
        cell_arms = filter(a -> a != "external" || fits, arms)
        kept = Any[]
        discarded = Any[]
        attempt = 0
        while length(kept) < rounds
            attempt += 1
            rotation = mod(attempt - 1, length(cell_arms))
            order = circshift(cell_arms, -rotation)
            sw0 = swap_used_mb()
            results = Dict{String, Any}()
            with_lock() do
                for arm in order
                    results[arm] = arm == "external" ? run_external(N) : run_child(env_of(arm), mode, arm, N, threads, !isempty(pureblas))
                end
            end
            sw1 = swap_used_mb()
            reason = round_discard_reason(results, sw0, sw1)
            if reason !== nothing
                println("round discarded: ", reason)
                push!(discarded, (; round = attempt, order, reason, seconds = Dict(a => get(r, "seconds", nothing) for (a, r) in results)))
                length(discarded) >= MAX_DISCARDS && error(length(discarded), " discards: the machine is too busy")
                continue
            end
            push!(kept, (; round = attempt, order, swap_before_mb = sw0, swap_after_mb = sw1, arms = results))
            println(
                "round ", length(kept), " (", join(order, ","), ")  ",
                join((@sprintf("%s %.4f s", a, r["seconds"]) for (a, r) in sort!(collect(results); by = first) if haskey(r, "seconds")), "  ")
            )
            flush(stdout)
        end
        push!(doc_cells, summarize(N, mode, cell_arms, fits, ext_result, kept, discarded))
        write_doc()
    end
    write_doc()
    println("results written to ", out)
    return
end

function stats(x)
    return (; min = minimum(x), median = median(x), max = maximum(x), values = x)
end

function summarize(N, mode, arms, fits, ext_result, kept, discarded)
    secs(a) = [r.arms[a]["seconds"] for r in kept]
    per_arm = Dict{String, Any}()
    for a in arms
        rs = [r.arms[a] for r in kept]
        d = Dict{String, Any}("seconds" => stats(secs(a)))
        for key in ("maxrss_bytes", "peak_device_bytes", "peak_bytes", "info", "query_check", "seconds_without_read")
            haskey(rs[1], key) && (d[key] = key == "seconds_without_read" ? stats([r[key] for r in rs]) : key in ("info", "query_check") ? rs[1][key] : maximum(r[key] for r in rs))
        end
        per_arm[a] = d
    end
    speedups = Dict{String, Any}()
    if "gpu" in arms
        for a in arms
            a == "gpu" && continue
            speedups[a] = stats(secs(a) ./ secs("gpu"))
        end
    end
    return (; cu_bench = N, arms = per_arm, speed_up_over_gpu = speedups, external_fits = fits, external_untimed = ext_result, rounds = [(; r.round, r.order, r.swap_before_mb, r.swap_after_mb, seconds = Dict(a => get(x, "seconds", nothing) for (a, x) in r.arms), calibration_s = Dict(a => x["cal"] for (a, x) in r.arms if haskey(x, "cal"))) for r in kept], discarded_rounds = discarded)
end

if length(ARGS) >= 1 && ARGS[1] == "--child"
    # The device packages load at top level, so that the children see them in a newer world age.
    if ARGS[3] == "gpu"
        @eval using KernelAbstractions
        @eval using $(Symbol(BACKEND_NAME))
    end
    child_main(ARGS[2], ARGS[3], parse(Int, ARGS[4]), parse(Int, ARGS[5]), ARGS[6] == "true")
else
    driver_main()
end
