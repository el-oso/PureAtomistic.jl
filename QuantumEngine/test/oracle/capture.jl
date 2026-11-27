# Capture the oracle's outputs for the parity harness:
#
#     QUANTUM_ENGINE_ORACLE=<binary> QUANTUM_ENGINE_ORACLE_BASIS=<basis_sets dir> \
#     QUANTUM_ENGINE_ORACLE_COMMIT=<full hash> \
#     julia --project=QuantumEngine/test QuantumEngine/test/oracle/capture.jl [--fixtures] <case>...
#
# Each <case> names a [case] table of references.toml. The script runs the oracle command the case
# names (`run`, or `aimext` for the critical-point report) on its input and writes
# captured/<case>.stdout, captured/<case>.log when the command writes a log, and the files the case
# exports, with the lines that name local paths rewritten and the timing lines removed. It stores in
# references.toml the extracted quantities (the compared ones and the known oracle defects), the
# SHA-256 of each captured file and the oracle commit. `--fixtures` also runs the vendored fixture
# files unchanged and stores the settings the oracle echoes for each in a [fixture] table, with its
# standard output in captured/fixtures/.
# A capture with a commit other than the stored one must name every case and `--fixtures`, so
# that references.toml never mixes two oracle commits.

using TOML

# Load the @testmodule of harness.jl as a plain module.
const OracleHarness = let
    ex = Meta.parseall(read(joinpath(@__DIR__, "harness.jl"), String))
    i = findfirst(a -> Meta.isexpr(a, :macrocall) && a.args[1] == Symbol("@testmodule"), ex.args)
    i === nothing && error("capture: harness.jl holds no @testmodule")
    m = ex.args[i]
    Core.eval(Main, Expr(:module, true, m.args[3], m.args[4]))
end

function main(args)
    fixtures = "--fixtures" in args
    cases = String[a for a in args if a != "--fixtures"]
    isempty(cases) && !fixtures && error("capture: name at least one case, or --fixtures")
    commit = get(ENV, "QUANTUM_ENGINE_ORACLE_COMMIT", "")
    occursin(r"^[0-9a-f]{40}$", commit) ||
        error("capture: QUANTUM_ENGINE_ORACLE_COMMIT must be the oracle's full 40-digit commit hash")
    OracleHarness.oracle_binary() === nothing && error("capture: QUANTUM_ENGINE_ORACLE (the oracle binary) is not set")
    OracleHarness.oracle_basis_dir()

    refs = OracleHarness.references()
    known = sort!(collect(keys(refs["case"])))
    unknown = setdiff(cases, known)
    isempty(unknown) || error("capture: no case $(join(unknown, ", ")) in references.toml")
    stored = get(refs["meta"], "oracle_commit", "")
    if !isempty(stored) && stored != commit && !(fixtures && sort(unique(cases)) == known)
        error("capture: references.toml holds commit $stored; a capture at $commit must name every case and --fixtures")
    end

    mkpath(OracleHarness.captured("fixtures"))
    for case in unique(cases)
        capture_case!(refs, case)
    end
    fixtures && capture_fixtures!(refs)
    refs["meta"]["oracle_commit"] = commit
    open(io -> OracleHarness.write_references(io, refs), OracleHarness.REFERENCES, "w")
    return nothing
end

function capture_case!(refs, case)
    t = refs["case"][case]
    dir = mktempdir()
    r = OracleHarness.run_oracle(case; dir)
    r.exitcode == 0 || error("capture: the oracle run of $case exited with code $(r.exitcode)")
    log = OracleHarness.scrub(r.log, case * ".inp", dir)
    out = OracleHarness.scrub(r.stdout, case * ".inp", dir)
    hashes = Dict{String, Any}()
    outputs = OracleHarness.writes_log(OracleHarness.case_command(case)) ? (("log", log), ("stdout", out)) : (("stdout", out),)
    for (ext, text) in outputs
        path = OracleHarness.captured(case * "." * ext)
        write(path, text)
        hashes[ext] = OracleHarness.file_sha256(path)
    end
    for ext in t["exports"]
        src = joinpath(dir, case * "." * ext)
        isfile(src) || error("capture: the oracle run of $case wrote no $case.$ext")
        dst = OracleHarness.captured(case * "." * ext)
        write(dst, OracleHarness.scrub(read(src, String), case * ".inp", dir))
        hashes[ext] = OracleHarness.file_sha256(dst)
    end
    for key in OracleHarness.captured_quantities(case)
        OracleHarness.quantity_extractor(key) in t["extract"] ||
            error("capture: case $case stores $key but does not list its extractor")
        t[key] = OracleHarness.quantity(case, key, log, out, OracleHarness.CAPTURED)
    end
    t["sha256"] = hashes
    println("captured ", case)
    return nothing
end

function capture_fixtures!(refs)
    tables = Dict{String, Any}()
    for (rel, _) in OracleHarness.FIXTURE_FILES
        dir = mktempdir()
        name = basename(rel)
        text = read(joinpath(OracleHarness.ORACLE_DIR, "inputs", rel), String)
        r = OracleHarness.run_oracle_file(name, text; dir)
        out = OracleHarness.scrub(r.stdout, name, dir)
        path = OracleHarness.fixture_stdout(rel)
        write(path, out)
        d = OracleHarness.echoed_settings(out)
        d["exitcode"] = r.exitcode
        d["sha256"] = OracleHarness.file_sha256(path)
        tables[rel] = d
        println("captured fixture ", rel, " (exit code ", r.exitcode, ")")
    end
    refs["fixture"] = tables
    return nothing
end

main(ARGS)
