# Interleaved A/B timing of two PureAtomistic versions on the trajectory analysis functions.
#
#   PUREATOMISTIC_AB_BASE=<git ref> julia --project=PureAtomistic/bench PureAtomistic/bench/ab_analysis.jl
#
# The method is that of PureMLIP/bench/ab_compare.jl. macOS only: it reads `sysctl` (swap usage,
# CPU brand). Two PureAtomistic versions cannot load into one process (same UUID), so each side runs
# in its own child process, alternating the order within consecutive pairs (base, head, then head,
# base, ...) so that a steady drift of the machine does not favor one side. The head side is the
# working tree of this checkout; the base side is a `git worktree` of PUREATOMISTIC_AB_BASE in a
# temporary directory outside the repository, removed at exit. Each side gets its own temporary
# environment whose `[sources]` point PureAtomistic and PureMLIP at that side's tree; both pin the
# Chairmarks version of PureMLIP/bench/Manifest.toml. A child warms up, then reports the median of
# SAMPLES Chairmarks samples per case. The speed-up of a pair is base time over head time, so a
# value above 1 means the head is faster.
#
# A pair is discarded, and measured again, when swap usage grew while it ran, or when a fixed
# scalar calibration loop (independent of PureAtomistic) shows the children ran at different
# speeds: by more than 1 % between their medians, or by more than 3 % within one child. One thread
# (`-t 1`) and one BLAS thread in every child.
#
#   PUREATOMISTIC_AB_BASE   git ref of the base side (required)
#   PUREATOMISTIC_AB_PAIRS  kept pairs to collect (default 7)
#   PUREATOMISTIC_AB_OUT    JSON output path (default: a new file under the system temp directory,
#                           whose path is printed; pass PureAtomistic/bench/results/... to keep a
#                           record)
#
# Cases: `rmsd(frames, reference; align = true)` and `align = false` over 60 frames of the 256-atom
# fcc Cu cell (a = 3.61 Å), each frame the reference rigidly rotated and translated with Gaussian
# displacements of 0.05 Å, in Float64 and Float32; the frames come from a fixed seed, so both
# sides see the same input.
using JSON, LinearAlgebra, Pkg, Printf, Statistics

const SAMPLES = 15
const WARMUP_EVALS = 5
const MAX_DISCARDS = 20
const CAL_REPEATS = 30
const CAL_WITHIN_TOLERANCE, CAL_PAIR_TOLERANCE = 0.03, 0.01
const CASES = ("rmsd align=true Float64", "rmsd align=false Float64", "rmsd align=true Float32", "rmsd align=false Float32")
const NFRAMES = 60

swap_used_mb() = parse(Float64, match(r"used = ([0-9.]+)M", read(`sysctl -n vm.swapusage`, String))[1])

# ---------------------------------------------------------------------------------------------
# Child
# ---------------------------------------------------------------------------------------------

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

# A proper rotation about the unit axis `u` by `θ` (Rodrigues).
function rotation(u, θ)
    u = u / norm(u)
    K = [0 -u[3] u[2]; u[3] 0 -u[1]; -u[2] u[1] 0]
    return I + sin(θ) * K + (1 - cos(θ)) * K * K
end

# A linear congruential stream, so that the frames do not depend on the Random version.
mutable struct Stream
    s::UInt64
end
next!(r::Stream) = (r.s = r.s * 6364136223846793005 + 1442695040888963407; (r.s >> 11) / 2.0^53)
gauss!(r::Stream) = sqrt(-2 * log(1 - next!(r))) * cos(2π * next!(r))

function frames(PA, ::Type{T}) where {T}
    ref = PA.Systems.fcc_cell(3.61, 4, :Cu)
    Y = Matrix{Float64}(ref.positions)
    r = Stream(0x5eed)
    out = map(1:NFRAMES) do _
        R = rotation([gauss!(r), gauss!(r), gauss!(r)], 2π * next!(r))
        t = [gauss!(r), gauss!(r), gauss!(r)]
        X = R * Y .+ t .+ 0.05 .* [gauss!(r) for _ in 1:3, _ in axes(Y, 2)]
        (; positions = Matrix{T}(X))
    end
    return out, Matrix{T}(Y)
end

function child_main()
    PA = Base.require(Main, :PureAtomistic)
    CM = Base.require(Main, :Chairmarks)
    BLAS.set_num_threads(1)
    Threads.nthreads() == 1 || error("expected 1 thread")
    for T in (Float64, Float32)
        fr, Y = Base.invokelatest(frames, PA, T)
        for align in (true, false)
            f = () -> Base.invokelatest(PA.Analysis.rmsd, fr, Y; align)
            for _ in 1:WARMUP_EVALS
                f()
            end
            cal0 = calibrate()
            b = Base.invokelatest(CM.benchmark, f; evals = 1, samples = SAMPLES, seconds = 600)
            t = median([s.time for s in b.samples])
            cal1 = calibrate()
            println("RESULT\trmsd align=", align, " ", T, "\t", t, "\t", cal0, "\t", cal1)
        end
    end
    return
end

# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------
git(dir, args...) = readchomp(`git -C $dir $args`)

function commit_label(tree)
    sha = git(tree, "rev-parse", "--short=12", "HEAD")
    return isempty(git(tree, "status", "--porcelain", "--untracked-files=no")) ? sha : sha * "-dirty"
end

function chairmarks_version(root)
    manifest = Base.parsed_toml(joinpath(root, "PureMLIP", "bench", "Manifest.toml"))
    return only(manifest["deps"]["Chairmarks"])["version"]
end

function make_env(dir, tree, chairmarks)
    mkpath(dir)
    uuids = Dict(
        "Chairmarks" => "0ca39b1e-fe0b-4e98-acfc-b1656634c4de",
        "LinearAlgebra" => "37e2e46d-f89d-539d-b4ee-838fcccc9c8e",
        "JSON" => "682c06a0-de6a-54ab-a142-c8b1cf79cde6",
        "Printf" => "de0858da-6303-5e67-8744-51eddeeeb8d7",
        "Statistics" => "10745b16-79ce-11e8-11f9-7d13ad32a3b2",
        "PureMLIP" => "e31974d8-f9c1-4d27-8220-1506e2a12d0a",
        "PureAtomistic" => "48bcfeca-9cba-41eb-9343-fd702b0553a7",
    )
    open(joinpath(dir, "Project.toml"), "w") do io
        println(io, "[deps]")
        for (name, uuid) in uuids
            println(io, name, " = \"", uuid, "\"")
        end
        println(io, "\n[sources]")
        println(io, "PureMLIP = {path = \"", joinpath(tree, "PureMLIP"), "\"}")
        println(io, "PureAtomistic = {path = \"", joinpath(tree, "PureAtomistic"), "\"}")
        println(io, "\n[compat]\nChairmarks = \"=", chairmarks, "\"\nJSON = \"1.6\"")
    end
    run(`$(Base.julia_cmd()) --project=$dir -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"`)
    return dir
end

function run_child(env)
    cmd = setenv(`$(Base.julia_cmd()) -t 1 --project=$env $(@__FILE__) --child`, merge(ENV, Dict("OPENBLAS_NUM_THREADS" => "1")))
    results = Dict{String, NTuple{3, Float64}}()
    for line in split(read(cmd, String), '\n')
        f = split(line, '\t')
        f[1] == "RESULT" && (results[f[2]] = (parse(Float64, f[3]), parse(Float64, f[4]), parse(Float64, f[5])))
    end
    Set(keys(results)) == Set(CASES) || error("a child reported ", collect(keys(results)))
    return results
end

function discard_reason(rb, rh, sw0, sw1)
    sw1 > sw0 + 1 && return "swap usage grew during the pair"
    cals(r) = [c for v in values(r) for c in v[2:3]]
    for (name, c) in (("base", cals(rb)), ("head", cals(rh)))
        spread = maximum(c) / minimum(c) - 1
        spread > CAL_WITHIN_TOLERANCE && return @sprintf("calibration varied by %.1f%% within the %s child", 100 * spread, name)
    end
    d = median(cals(rh)) / median(cals(rb)) - 1
    abs(d) > CAL_PAIR_TOLERANCE && return @sprintf("calibration medians differ by %.1f%% between the children", 100 * d)
    return nothing
end

function driver_main()
    root = git(@__DIR__, "rev-parse", "--show-toplevel")
    base_ref = get(ENV, "PUREATOMISTIC_AB_BASE", "")
    isempty(base_ref) && error("set PUREATOMISTIC_AB_BASE to a git ref")
    npairs = parse(Int, get(ENV, "PUREATOMISTIC_AB_PAIRS", "7"))
    out = get(ENV, "PUREATOMISTIC_AB_OUT", tempname() * "_ab_analysis.json")
    chairmarks = chairmarks_version(root)
    scratch = mktempdir()
    base_tree = joinpath(scratch, "base")
    git(root, "worktree", "add", "--detach", base_tree, base_ref)
    try
        base_sha, head_sha = commit_label(base_tree), commit_label(root)
        println("base ", base_sha, "  head ", head_sha)
        base_env = make_env(joinpath(scratch, "env_base"), base_tree, chairmarks)
        head_env = make_env(joinpath(scratch, "env_head"), root, chairmarks)
        # Untimed pair first: it pays the compilation of both environments once.
        run_child(base_env)
        run_child(head_env)
        kept, discarded = Any[], Any[]
        sw_first = swap_used_mb()
        attempt = 0
        while length(kept) < npairs
            attempt += 1
            base_first = isodd(attempt)
            sw0 = swap_used_mb()
            r = Dict{Symbol, Any}()
            for side in (base_first ? (:base, :head) : (:head, :base))
                r[side] = run_child(side === :base ? base_env : head_env)
            end
            sw1 = swap_used_mb()
            order = base_first ? "base,head" : "head,base"
            reason = discard_reason(r[:base], r[:head], sw0, sw1)
            pair = (;
                pair = attempt, order, swap_before_mb = sw0, swap_after_mb = sw1,
                base_s = Dict(k => v[1] for (k, v) in r[:base]), head_s = Dict(k => v[1] for (k, v) in r[:head]),
                base_calibration_s = Dict(k => [v[2], v[3]] for (k, v) in r[:base]),
                head_calibration_s = Dict(k => [v[2], v[3]] for (k, v) in r[:head]),
            )
            if reason !== nothing
                println("pair discarded: ", reason)
                push!(discarded, merge(pair, (; reason)))
                length(discarded) >= MAX_DISCARDS && error(length(discarded), " discards: the machine is too busy")
                continue
            end
            push!(kept, pair)
            @printf("pair %d (%s)  %s\n", length(kept), order, join((@sprintf("%s %.3f", c, pair.base_s[c] / pair.head_s[c]) for c in CASES), "  "))
        end
        cases = map(CASES) do c
            s = [p.base_s[c] / p.head_s[c] for p in kept]
            (;
                case = c, speedup_per_pair = s, median = median(s), min = minimum(s), max = maximum(s),
                base_s = [p.base_s[c] for p in kept], head_s = [p.head_s[c] for p in kept],
            )
        end
        doc = (;
            schema = "pureatomistic-ab-analysis-v1",
            machine = strip(read(`sysctl -n machdep.cpu.brand_string`, String)),
            julia_version = string(VERSION), chairmarks_version = chairmarks,
            base_ref, base_commit = base_sha, head_commit = head_sha,
            workload = "rmsd over $NFRAMES frames of fcc_cell(3.61, 4, :Cu) (256 atoms), rigidly moved with 0.05 Å Gaussian displacements",
            threads = 1, blas_threads = 1, samples = SAMPLES, warmup_evaluations = WARMUP_EVALS,
            speedup = "base time / head time per pair; above 1 means the head is faster",
            order = "odd pairs run base then head, even pairs head then base",
            calibration = "seconds of a fixed scalar loop before and after each measurement; pairs are discarded when base and head medians differ by more than $(100 * CAL_PAIR_TOLERANCE)% or one child varies by more than $(100 * CAL_WITHIN_TOLERANCE)%",
            swap_first_start_mb = sw_first, swap_last_end_mb = swap_used_mb(),
            pairs = kept, discarded_pairs = discarded, cases,
        )
        open(io -> (JSON.json(io, doc; pretty = true); println(io)), out, "w")
        for c in cases
            @printf("%-26s speed-up median %.3f  min %.3f  max %.3f\n", c.case, c.median, c.min, c.max)
        end
        @printf("%d pairs kept, %d discarded\n", length(kept), length(discarded))
        println("results written to ", out)
    finally
        try
            git(root, "worktree", "remove", "--force", base_tree)
        catch err
            @warn "could not remove the base worktree" err
        end
        rm(scratch; recursive = true, force = true)
    end
    return
end

if !isempty(ARGS) && ARGS[1] == "--child"
    child_main()
else
    driver_main()
end
