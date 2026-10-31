# Mutation smoke check. Each entry of mutations.toml names a source file, an exact substring
# (`find`, which must occur exactly once), a deliberate bug (`replace`) and a regex (`expect`)
# for the test item expected to fail. For each selected entry the script copies PureMLIP to a
# temporary directory, applies the one edit, runs the full test suite in a fresh Julia process
# and reports "killed" when the suite fails and "survived" when it passes. The exit status is
# nonzero when any mutation survives.
#
# Usage: julia run.jl [filter ...]
# A filter is a regular expression matched against mutation names; with none, all run.
# Set PUREMLIP_CHECKPOINT_DIR to the checkpoint cache the suite reads.

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

function run_mutation(m)
    dir = mktempdir()
    try
        dst = joinpath(dir, "PureMLIP")
        copy_package(PKG_DIR, dst)
        apply_mutation(dst, m)
        log = joinpath(dir, "log.txt")
        cmd = `$(Base.julia_cmd()) --project=$dst -e "using Pkg; Pkg.test()"`
        ok = open(log, "w") do io
            success(pipeline(cmd; stdout = io, stderr = io))
        end
        text = read(log, String)
        matched = occursin(Regex(m["expect"]), text)
        return ok ? :survived : :killed, matched, text
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
    survivors = String[]
    for m in selected
        t = @elapsed (verdict, matched, text) = run_mutation(m)
        println(rpad(m["name"], 34), " ", rpad(verdict, 9), " expected-item-in-log=", matched, "  (", round(Int, t), " s)")
        if verdict === :killed
            ls = filter(l -> occursin(r"Test Failed|Error During Test|^ERROR:|Failed to precompile", l), split(text, '\n'))
            println("    first failure: ", isempty(ls) ? "(none found)" : first(ls))
        end
        flush(stdout)
        if verdict === :survived
            push!(survivors, m["name"])
        end
    end
    if isempty(survivors)
        println("all ", length(selected), " selected mutations killed")
        return 0
    end
    println("SURVIVORS: ", join(survivors, ", "))
    return 1
end

exit(main(ARGS))
