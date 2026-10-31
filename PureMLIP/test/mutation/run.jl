# Mutation smoke check. Each entry of mutations.toml names a source file, an exact substring
# (`find`, which must occur exactly once), a deliberate bug (`replace`) and a regex (`expect`)
# matched against the names of the test items expected to fail. The script first runs the
# unmutated package as a control and stops if the suite fails. For each selected entry it then
# copies PureMLIP to a temporary directory, applies the one edit, runs the full test suite in a
# fresh Julia process and classifies the run:
#
#   invalid           the package did not precompile or load, or the suite never reported
#   killed            a test item failed, and a failing item matches `expect`
#   killed_elsewhere  test items failed, but none matches `expect`
#   survived          the suite passed
#
# The exit status is nonzero when the control fails or any mutation is invalid or survives.
#
# Usage: julia run.jl [filter ...]
# A filter is a regular expression matched against mutation names; with none, all run.
# Set PUREMLIP_CHECKPOINT_DIR to the checkpoint cache the suite reads, and
# MUTATION_LOG_DIR to keep each run's log as <name>.log.

using TOML

const PKG_DIR = dirname(dirname(@__DIR__))
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

# Run the suite on a copy of the package, after applying `m` when it is given.
function run_suite(m, logdir, name)
    dir = mktempdir()
    try
        dst = joinpath(dir, "PureMLIP")
        copy_package(PKG_DIR, dst)
        isnothing(m) || apply_mutation(dst, m)
        logfile = joinpath(dir, "log.txt")
        cmd = `$(Base.julia_cmd()) --project=$dst -e "using Pkg; Pkg.test()"`
        ok = open(logfile, "w") do io
            success(pipeline(cmd; stdout = io, stderr = io))
        end
        log = read(logfile, String)
        isempty(logdir) || write(joinpath(logdir, name * ".log"), log)
        return ok, log
    finally
        rm(dir; recursive = true, force = true)
    end
end

function main(args)
    muts = TOML.parsefile(joinpath(@__DIR__, "mutations.toml"))["mutation"]
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

    ok, _ = run_suite(nothing, logdir, "control")
    println("control (unmutated): ", ok ? "passed" : "FAILED")
    flush(stdout)
    ok || return 1

    bad = String[]
    for m in selected
        t = @elapsed ((ok, log) = run_suite(m, logdir, m["name"]))
        verdict, items = classify(ok, log, m["expect"])
        println(rpad(m["name"], 34), " ", rpad(verdict, 17), " (", round(Int, t), " s)")
        if verdict in (:killed, :killed_elsewhere)
            println("    failed items: ", join(items, " | "))
        end
        verdict in (:invalid, :survived) && push!(bad, string(m["name"], " (", verdict, ")"))
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
