# Mutation smoke check. Each entry of mutations.toml names a source file, an exact substring
# (`find`, which must occur exactly once), a deliberate bug (`replace`) and a regex (`expect`)
# matched against the names of the test items expected to fail. The script first runs the
# unmutated package as a control and stops if the suite fails. For each selected entry it then
# copies PureMLIP to a temporary directory, applies the one edit, runs the full test suite in a
# fresh Julia process and classifies the run:
#
#   invalid           the package did not precompile or load, or the suite never reported
#   killed            a test item failed, and a failing item matches `expect`
#   killed_timeout    the run exceeded the time limit and was killed with its whole process
#                     tree: the mutant makes the code loop (or run) without end, which a test
#                     suite detects by never finishing; it counts as killed
#   killed_elsewhere  test items failed, but none matches `expect`
#   survived          the suite passed
#
# The exit status is nonzero when the control fails or any mutation is invalid, survives or is
# killed elsewhere: `expect` names the item that must catch the bug, so a different item failing
# means the intended test no longer does its job.
#
# MUTATION_SUITE=gpu judges the extension files instead: the entries come from mutations_gpu.toml and
# each run executes test/gpu/runtests.jl on the KernelAbstractions CPU backend, in the environment
# test/gpu/cpu.
#
# Usage: julia run.jl [filter ...]
# A filter is a regular expression matched against mutation names; with none, all run.
# Set PUREMLIP_CHECKPOINT_DIR to the checkpoint cache the suite reads, and
# MUTATION_LOG_DIR to keep each run's log as <name>.log. Set MUTATION_TIMEOUT_S to the time limit
# in seconds for one mutated run (default 900); the unmutated control has no limit.
# Set MUTATION_NO_CONTROL=1 to skip the control run when running the entries in several batches
# after one passing control.

using TOML

const GPU_SUITE = get(ENV, "MUTATION_SUITE", "") == "gpu"
# KernelAbstractions 0.10's CPU backend hangs at hwloc's topology detection on Apple M-series chips
# with performance levels; hwloc's Darwin component is skipped. Child processes inherit this.
Sys.isapple() && get!(ENV, "HWLOC_COMPONENTS", "-darwin")

const PKG_DIR = dirname(dirname(@__DIR__))
const DEFAULT_TIMEOUT_S = 900.0
const SKIP_DIRS = ("juliac/build", "docs/build")

# Copy the package tree, leaving out build outputs and coverage files.
function copy_package(src, dst)
    mkpath(dst)
    for name in readdir(src)
        s, d = joinpath(src, name), joinpath(dst, name)
        rel = relpath(s, PKG_DIR)
        (rel in SKIP_DIRS || endswith(name, ".cov")) && continue
        if isdir(s) && !islink(s)
            copy_package(s, d)
        else
            cp(s, d; follow_symlinks = false)
        end
    end
    return nothing
end

function apply_mutation(dst, m)
    path = joinpath(dst, m["file"])
    isfile(path) || error("mutation $(m["name"]): no file $(m["file"])")
    src = read(path, String)
    n = count(m["find"], src)
    n == 1 || error("mutation $(m["name"]): `find` occurs $n times in $(m["file"]), expected exactly 1")
    write(path, replace(src, m["find"] => m["replace"]; count = 1))
    return nothing
end

# Names of the test items that reported a failure or an error in `log`.
failed_items(log::AbstractString) = unique(
    [m.captures[1] for m in eachmatch(r"^(.+?): (?:Test Failed|Error During Test) at "m, log)]
)

# Verdict for a run with exit status `ok` and output `log`; `expect` is a regex for item names.
function classify(ok::Bool, log::AbstractString, expect::AbstractString)
    items = failed_items(log)
    # A LoadError or ParseError counts only when no test item reported: failing items print
    # their own stack traces, which contain LoadError lines.
    invalid = occursin("Failed to precompile", log) || !occursin("Test Summary:", log) ||
        (isempty(items) && occursin(r"LoadError|ParseError", replace(log, "LoadError: Some tests did not pass" => "")))
    invalid && return :invalid, items
    ok && return :survived, items
    isempty(items) && return :invalid, items
    return any(i -> occursin(Regex(expect), i), items) ? :killed : :killed_elsewhere, items
end

# Run `cmd` with output to `io`. The child is its own process-group leader, so on timeout one
# SIGKILL to the group ends it and every process it started (Pkg.test spawns further Julia
# processes). Returns `(ok, timed_out)`.
function run_limited(cmd, io, limit)
    proc = run(pipeline(detach(cmd); stdout = io, stderr = io); wait = false)
    pid = getpid(proc)
    deadline = time() + limit
    while process_running(proc)
        if time() > deadline
            run(ignorestatus(`kill -KILL -- -$pid`))
            wait(proc)
            return false, true
        end
        sleep(0.5)
    end
    return success(proc), false
end

# Time limit in seconds for one mutated run.
function timeout_s()
    t = tryparse(Float64, get(ENV, "MUTATION_TIMEOUT_S", string(DEFAULT_TIMEOUT_S)))
    (isnothing(t) || t <= 0) && error("MUTATION_TIMEOUT_S must be a positive number of seconds")
    return t
end

# Run the suite on a copy of the package, after applying `m` when it is given.
function run_suite(m, logdir, name)
    dir = mktempdir()
    try
        dst = joinpath(dir, "PureMLIP")
        copy_package(PKG_DIR, dst)
        # The precompile workload loads and evaluates the tiny checkpoint, so a mutation that breaks
        # either would fail package precompilation and read as invalid instead of being caught by a
        # test. The workload is removed from every copy, the control included, so that all runs share
        # one configuration; the import stays because Aqua's stale-dependency check requires it. The
        # workload itself is exercised by the ordinary suite.
        write(joinpath(dst, "src", "precompile.jl"), "using PrecompileTools: @setup_workload\n")
        isnothing(m) || apply_mutation(dst, m)
        logfile = joinpath(dir, "log.txt")
        cmd = if GPU_SUITE
            env = joinpath(dst, "test", "gpu", "cpu")
            addenv(
                `$(Base.julia_cmd()) --check-bounds=yes --project=$env -e "using Pkg; Pkg.instantiate(); include(ARGS[1])" $(joinpath(dst, "test", "gpu", "runtests.jl"))`,
                "PUREMLIP_GPU_BACKEND" => "CPU",
            )
        else
            `$(Base.julia_cmd()) --project=$dst -e "using Pkg; Pkg.test()"`
        end
        limit = isnothing(m) ? Inf : timeout_s()
        ok, timed_out = open(logfile, "w") do io
            run_limited(cmd, io, limit)
        end
        log = read(logfile, String)
        isempty(logdir) || write(joinpath(logdir, name * ".log"), log)
        return ok, log, timed_out
    finally
        rm(dir; recursive = true, force = true)
    end
end

function main(args)
    muts = get(TOML.parsefile(joinpath(@__DIR__, GPU_SUITE ? "mutations_gpu.toml" : "mutations.toml")), "mutation", Dict{String, Any}[])
    names = [m["name"] for m in muts]
    allunique(names) || error("mutation names are not unique")
    filters = Regex.(args)
    selected = isempty(filters) ? muts : filter(m -> any(f -> occursin(f, m["name"]), filters), muts)
    isempty(selected) && error("no mutation matches $(args)")
    # Validate every selected `find` before starting the slow runs.
    for m in selected
        n = count(m["find"], read(joinpath(PKG_DIR, m["file"]), String))
        n == 1 || error("mutation $(m["name"]): `find` occurs $n times in $(m["file"]), expected exactly 1")
    end
    logdir = get(ENV, "MUTATION_LOG_DIR", "")
    isempty(logdir) || mkpath(logdir)

    if get(ENV, "MUTATION_NO_CONTROL", "") == "1"
        println("control (unmutated): skipped")
    else
        ok, _, _ = run_suite(nothing, logdir, "control")
        println("control (unmutated): ", ok ? "passed" : "FAILED")
        flush(stdout)
        ok || return 1
    end

    bad = String[]
    for m in selected
        t = @elapsed ((ok, log, timed_out) = run_suite(m, logdir, m["name"]))
        verdict, items = timed_out ? (:killed_timeout, String[]) : classify(ok, log, m["expect"])
        println(rpad(m["name"], 34), " ", rpad(verdict, 17), " (", round(Int, t), " s)")
        if verdict in (:killed, :killed_elsewhere)
            println("    failed items: ", join(items, " | "))
        elseif verdict === :killed_timeout
            println("    no result within ", timeout_s(), " s; process tree killed")
        end
        verdict in (:invalid, :survived, :killed_elsewhere) && push!(bad, string(m["name"], " (", verdict, ")"))
        flush(stdout)
    end
    if isempty(bad)
        println("all ", length(selected), " selected mutations killed")
        return 0
    end
    println("PROBLEMS: ", join(bad, ", "))
    return 1
end

abspath(PROGRAM_FILE) == (@__FILE__) && exit(main(ARGS))
