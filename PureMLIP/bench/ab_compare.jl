# Interleaved A/B timing of two PureMLIP versions on `mace_energy_forces!` and `mace_energy!`.
#
#   PUREMLIP_AB_BASE=<git ref> PUREMLIP_BENCH_CASES="5:64,5:32" \
#       julia --project=PureMLIP/bench PureMLIP/bench/ab_compare.jl
#
# macOS only: it reads `sysctl` (swap usage, CPU brand). Two PureMLIP versions cannot load into
# one process (same UUID), so each side runs in its own child process, alternating
# the order within consecutive pairs (base, head, then head, base, ...) so that a steady drift of the
# machine does not favor one side. The head side is the working tree of this checkout; the base side
# is a `git worktree` of PUREMLIP_AB_BASE in a temporary directory outside the repository, removed
# at exit. Each side gets its own temporary environment whose `[sources]` point PureMLIP at that
# side's tree; both pin the same Chairmarks version (the one in the bench Manifest). A child
# warms up, then reports the median of three one-evaluation Chairmarks samples per case and entry
# point. The ratio of a pair is head/base, so a value below 1 means the head is faster.
#
#   PUREMLIP_AB_BASE         git ref of the base side (required)
#   PUREMLIP_CHECKPOINT_DIR  directory with mace-mpa-0-medium.model
#   PUREMLIP_BENCH_CASES     cases as "N:precision,..." (default "5:64,5:32")
#   PUREMLIP_AB_PAIRS        kept pairs to collect (default 5)
#   PUREMLIP_AB_OUT          JSON output path (default: a new file under the system temp directory,
#                            whose path is printed; pass PureMLIP/bench/results/... to keep a record)
#   PUREMLIP_AB_LOAD         "PureBLAS": the head side runs `using PureBLAS; PureBLAS.activate()`
#                            before warm-up; the base side stays on OpenBLAS
#   PUREMLIP_AB_PUREBLAS     PureBLAS checkout (a directory or a URL), required with
#                            PUREMLIP_AB_LOAD=PureBLAS
#
# A pair is discarded, and measured again, when swap usage grew while it ran, or when a fixed
# scalar calibration loop (independent of PureMLIP) shows the children ran at different speeds,
# as when another process shares the core or the scheduler moves a child between core types. One thread
# everywhere (`-t 1`, `BLAS.set_num_threads(1)`).
using Chairmarks, JSON, LinearAlgebra, Pkg, Printf, Statistics

const SAMPLES = 3
const WARMUP_EVALS = 5
const MAX_DISCARDS = 20
const CAL_REPEATS = 30
# A pair is discarded when the calibration medians of its two children differ by more than
# CAL_PAIR_TOLERANCE, or when one child's calibration varies by more than CAL_WITHIN_TOLERANCE.
const CAL_WITHIN_TOLERANCE = 0.03
const CAL_PAIR_TOLERANCE = 0.01
const ENTRIES = ("mace_energy_forces!", "mace_energy!")

swap_used_mb() = parse(Float64, match(r"used = ([0-9.]+)M", read(`sysctl -n vm.swapusage`, String))[1])

# ---------------------------------------------------------------------------------------------
# Child: measures every case on the PureMLIP of the active environment and prints one line per
# result: `RESULT <case> <entry> <seconds>` and `BLAS <library file names>`.
# ---------------------------------------------------------------------------------------------
function child_main(structures, cases, load)
    PM = Base.require(Main, :PureMLIP)
    include(structures)
    BLAS.set_num_threads(1)
    Threads.nthreads() == 1 || error("run with julia -t 1")
    if load == "PureBLAS"
        pb = Base.require(Main, :PureBLAS)
        Base.invokelatest(pb.activate)
        Base.invokelatest(pb.is_active) || error("PureBLAS did not become active")
        BLAS.set_num_threads(1)
    elseif load != ""
        error("unknown PUREMLIP_AB_LOAD: ", load)
    end
    Base.invokelatest(measure, PM, cases)
    println("PKG\t", join(("$(v.name)=$(v.version)" for (_, v) in Pkg.dependencies() if v.version !== nothing), ","))
    println("BLAS\t", join((basename(l.libname) for l in BLAS.get_config().loaded_libs), ","))
    return
end

# Time of a fixed scalar loop (minimum over repeats). It does not touch PureMLIP, so it moves
# only with the speed of the core the process is on, not with the code under test.
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

function measure(PM, cases)
    ck = PM.load_mace_checkpoint(joinpath(ENV["PUREMLIP_CHECKPOINT_DIR"], "mace-mpa-0-medium.model"))
    for (N, T) in cases
        cu = Main.cu_bench(N)
        model = PM.MACEModel{T}(ck)
        positions = Matrix{T}(cu.positions')
        cell = Matrix{T}(cu.cell')
        z = cu.atomic_numbers
        nl = PM.neighbor_list(positions, cell, model.r_max; extra_capacity = 64)
        ws = PM.MACEWorkspace{T}(model, length(z), nl.n_edges)
        res = PM.MACEResult{T}(length(z))
        evals = (
            "mace_energy_forces!" => () -> PM.mace_energy_forces!(res, ws, model, positions, cell, z, nl),
            "mace_energy!" => () -> PM.mace_energy!(res, ws, model, positions, cell, z, nl),
        )
        for (name, f) in evals
            for _ in 1:WARMUP_EVALS
                f()
            end
            cal0 = calibrate()
            t = median([s.time for s in (@be f() evals = 1 samples = SAMPLES seconds = 600).samples])
            cal1 = calibrate()
            println("RESULT\t", N, ":", T === Float64 ? 64 : 32, "\t", name, "\t", t, "\t", cal0, "\t", cal1)
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

function chairmarks_version()
    manifest = Base.parsed_toml(joinpath(@__DIR__, "Manifest.toml"))
    return only(manifest["deps"]["Chairmarks"])["version"]
end

# Writes and instantiates an environment whose PureMLIP is the one in `tree`. `extra` maps
# further package names to `(uuid, source)` where source is a `[sources]` entry.
function make_env(dir, tree, extra)
    mkpath(dir)
    uuids = Dict(
        "Chairmarks" => "0ca39b1e-fe0b-4e98-acfc-b1656634c4de",
        "LinearAlgebra" => "37e2e46d-f89d-539d-b4ee-838fcccc9c8e",
        "JSON" => "682c06a0-de6a-54ab-a142-c8b1cf79cde6",
        "Printf" => "de0858da-6303-5e67-8744-51eddeeeb8d7",
        "Statistics" => "10745b16-79ce-11e8-11f9-7d13ad32a3b2",
        "PureMLIP" => "e31974d8-f9c1-4d27-8220-1506e2a12d0a",
    )
    open(joinpath(dir, "Project.toml"), "w") do io
        println(io, "[deps]")
        for (name, uuid) in uuids
            println(io, name, " = \"", uuid, "\"")
        end
        for (name, (uuid, _)) in extra
            println(io, name, " = \"", uuid, "\"")
        end
        println(io, "\n[sources]")
        println(io, "PureMLIP = {path = \"", joinpath(tree, "PureMLIP"), "\"}")
        for (name, (_, source)) in extra
            println(io, name, " = ", source)
        end
        println(io, "\n[compat]\nChairmarks = \"=", chairmarks_version(), "\"\nJSON = \"1.6\"")
    end
    code = "using Pkg; Pkg.instantiate(); Pkg.precompile()"
    run(`$(Base.julia_cmd()) --project=$dir -e $code`)
    return dir
end

function run_child(env, structures, cases, load)
    spec = join(("$N:$(T === Float64 ? 64 : 32)" for (N, T) in cases), ",")
    cmd = setenv(
        `$(Base.julia_cmd()) -t 1 --project=$env $(@__FILE__) --child $structures $spec $load`,
        merge(ENV, Dict("OPENBLAS_NUM_THREADS" => "1")),
    )
    out = read(cmd, String)
    results = Dict{Tuple{String, String}, NTuple{3, Float64}}()
    blas = String[]
    pkgs = Dict{String, String}()
    for line in split(out, '\n')
        f = split(line, '\t')
        if f[1] == "RESULT"
            results[(f[2], f[3])] = (parse(Float64, f[4]), parse(Float64, f[5]), parse(Float64, f[6]))
        elseif f[1] == "BLAS"
            blas = split(f[2], ',')
        elseif f[1] == "PKG"
            for kv in split(f[2], ',')
                name, ver = split(kv, '=')
                pkgs[name] = ver
            end
        end
    end
    return (; results, blas, pkgs)
end

first_times(r) = Dict(string(k[1], " ", k[2]) => v[1] for (k, v) in r)
calibrations(r) = Dict(string(k[1], " ", k[2]) => [v[2], v[3]] for (k, v) in r)

function pair_discard_reason(rb, rh, sw0, sw1)
    sw1 > sw0 + 1 && return "swap usage grew during the pair"
    cals(r) = [c for v in values(r) for c in v[2:3]]
    cb, ch = cals(rb), cals(rh)
    for (name, c) in (("base", cb), ("head", ch))
        spread = maximum(c) / minimum(c) - 1
        spread > CAL_WITHIN_TOLERANCE && return @sprintf("calibration varied by %.1f%% within the %s child", 100 * spread, name)
    end
    d = median(ch) / median(cb) - 1
    abs(d) > CAL_PAIR_TOLERANCE && return @sprintf("calibration medians differ by %.1f%% between the children", 100 * d)
    return nothing
end

function pureblas_source(spec)
    return isdir(spec) ? "{path = \"$(abspath(spec))\"}" : "{url = \"$spec\"}"
end

function summarize(cases, kept)
    entries = []
    isempty(kept) && return entries
    for (N, T) in cases
        key = "$N:$(T === Float64 ? 64 : 32)"
        per_entry = map(ENTRIES) do e
            ratios = [p.head_s[(key, e)][1] / p.base_s[(key, e)][1] for p in kept]
            (;
                entry_point = e, pair_ratios = ratios, median = median(ratios), min = minimum(ratios), max = maximum(ratios),
                base_s = [p.base_s[(key, e)][1] for p in kept], head_s = [p.head_s[(key, e)][1] for p in kept],
            )
        end
        push!(entries, (; system = "cu_bench($N)", precision = string(T), entries = per_entry))
    end
    return entries
end

function driver_main()
    root = git(@__DIR__, "rev-parse", "--show-toplevel")
    base_ref = get(ENV, "PUREMLIP_AB_BASE", "")
    isempty(base_ref) && error("set PUREMLIP_AB_BASE to a git ref")
    cases = [
        (parse(Int, a), b == "64" ? Float64 : Float32) for (a, b) in
            split.(split(get(ENV, "PUREMLIP_BENCH_CASES", "5:64,5:32"), ','), ':')
    ]
    npairs = parse(Int, get(ENV, "PUREMLIP_AB_PAIRS", "5"))
    out = get(ENV, "PUREMLIP_AB_OUT", tempname() * "_ab_compare.json")
    load = get(ENV, "PUREMLIP_AB_LOAD", "")
    load in ("", "PureBLAS") || error("PUREMLIP_AB_LOAD must be empty or PureBLAS")
    head_extra = Dict{String, Tuple{String, String}}()
    pureblas_commit = nothing
    if load == "PureBLAS"
        spec = get(ENV, "PUREMLIP_AB_PUREBLAS", "")
        isempty(spec) && error("PUREMLIP_AB_LOAD=PureBLAS needs PUREMLIP_AB_PUREBLAS (a checkout path or URL)")
        head_extra["PureBLAS"] = ("cc9e14db-574f-4602-bf53-1167cc4b26d2", pureblas_source(spec))
        isdir(spec) && (pureblas_commit = commit_label(spec))
    end
    structures = joinpath(root, "PureMLIP", "test", "reference", "structures.jl")

    scratch = mktempdir()
    base_tree = joinpath(scratch, "base")
    git(root, "worktree", "add", "--detach", base_tree, base_ref)
    try
        base_sha, head_sha = commit_label(base_tree), commit_label(root)
        println("base ", base_sha, "  head ", head_sha)
        base_env = make_env(joinpath(scratch, "env_base"), base_tree, Dict{String, Tuple{String, String}}())
        head_env = make_env(joinpath(scratch, "env_head"), root, head_extra)

        # Untimed pair first: it pays the compilation of both sides' environments once.
        run_child(base_env, structures, cases, "")
        run_child(head_env, structures, cases, load)

        kept = Any[]
        discarded = Any[]
        last = Dict{String, Any}()
        sw_first = swap_used_mb()
        write_doc = function (sw_last)
            doc = (;
                schema = "puremlip-ab-compare-v2",
                machine = Sys.isapple() ? strip(read(`sysctl -n machdep.cpu.brand_string`, String)) : string(Sys.CPU_NAME),
                julia_version = string(VERSION), chairmarks_version = chairmarks_version(),
                base_ref, base_commit = base_sha, head_commit = head_sha,
                pureblas_commit, load = isempty(load) ? "none" : load,
                threads = 1, blas_threads = 1,
                blas_libraries = Dict(k => v.blas for (k, v) in last),
                package_versions = Dict(k => v.pkgs for (k, v) in last),
                samples_per_evaluation_point = SAMPLES, warmup_evaluations = WARMUP_EVALS,
                ratio = "head/base per pair; below 1 means the head is faster",
                order = "odd pairs run base then head, even pairs head then base",
                calibration = "seconds of a fixed scalar loop before and after each measurement; pairs are discarded when base and head medians differ by more than $(100 * CAL_PAIR_TOLERANCE)% or one child varies by more than $(100 * CAL_WITHIN_TOLERANCE)%",
                swap_first_start_mb = sw_first, swap_last_end_mb = sw_last,
                pairs = [
                    (; p.pair, p.order, p.swap_before_mb, p.swap_after_mb, base_calibration_s = calibrations(p.base_s), head_calibration_s = calibrations(p.head_s))
                        for p in kept
                ],
                discarded_pairs = discarded, cases = summarize(cases, kept),
            )
            open(io -> (JSON.json(io, doc; pretty = true); println(io)), out, "w")
            return doc
        end
        attempt = 0
        while length(kept) < npairs
            attempt += 1
            base_first = isodd(attempt)
            sw0 = swap_used_mb()
            sides = base_first ? (:base, :head) : (:head, :base)
            r = Dict{Symbol, Any}()
            for side in sides
                r[side] = side === :base ? run_child(base_env, structures, cases, "") : run_child(head_env, structures, cases, load)
            end
            last["base"], last["head"] = r[:base], r[:head]
            rb, rh = r[:base].results, r[:head].results
            sw1 = swap_used_mb()
            order = base_first ? "base,head" : "head,base"
            pair = (; pair = attempt, order, swap_before_mb = sw0, swap_after_mb = sw1, base_s = rb, head_s = rh)
            reason = pair_discard_reason(rb, rh, sw0, sw1)
            if reason !== nothing
                println("pair discarded: ", reason)
                push!(
                    discarded, (;
                        pair.pair, pair.order, reason, pair.swap_before_mb, pair.swap_after_mb,
                        base_s = first_times(rb), head_s = first_times(rh), base_calibration_s = calibrations(rb), head_calibration_s = calibrations(rh),
                    )
                )
                if length(discarded) >= MAX_DISCARDS
                    write_doc(swap_used_mb())
                    println("results written to ", out)
                    error(length(discarded), " discards (swap growth or calibration): the machine is too busy")
                end
                continue
            end
            push!(kept, pair)
            @printf(
                "pair %d (%s)  %s\n", length(kept), order,
                join((@sprintf("%s %s %.3f", k[1], k[2], rh[k][1] / rb[k][1]) for k in sort(collect(keys(rb)))), "  ")
            )
        end
        sw_last = swap_used_mb()
        doc = write_doc(sw_last)
        for e in doc.cases, x in e.entries
            @printf("%s %-7s %-20s median %.4f  min %.4f  max %.4f\n", e.system, e.precision, x.entry_point, x.median, x.min, x.max)
        end
        @printf("%d pairs kept, %d discarded\n", length(kept), length(discarded))
        @printf("swap at start %.2f MB, at end %.2f MB\n", sw_first, sw_last)
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
    structures, spec, load = ARGS[2], ARGS[3], ARGS[4]
    cases = [(parse(Int, a), b == "64" ? Float64 : Float32) for (a, b) in split.(split(spec, ','), ':')]
    child_main(structures, cases, load)
else
    driver_main()
end
