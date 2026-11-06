# Per-file line-coverage gate. Run after the suite with coverage enabled:
#   julia --project=PureAtomistic/test/coverage PureAtomistic/test/coverage/check.jl
# or, for the extension files measured by the :gpu items on the KernelAbstractions CPU backend,
#   julia --project=PureAtomistic/test/coverage PureAtomistic/test/coverage/check.jl --root ext/PureAtomisticGPUExt --thresholds thresholds_gpu.toml
# Compares each src file's covered/coverable line fraction with thresholds.toml
# and exits non-zero on any file below its threshold or without a row.
using CoverageTools, Logging, TOML

const PKG = normpath(joinpath(@__DIR__, "..", ".."))
# Options: --root <dir> (default src) names the folder of .cov files to judge, --thresholds <file>
# (default thresholds.toml) the table of floors, next to this script.
function option(name, default)
    i = findfirst(==(name), ARGS)
    isnothing(i) && return default
    i < length(ARGS) || error("$name needs a value")
    return ARGS[i + 1]
end
const SRC = joinpath(PKG, option("--root", "src"))
const THRESHOLDS = option("--thresholds", "thresholds.toml")

thresholds = TOML.parsefile(joinpath(@__DIR__, THRESHOLDS))
# process_folder merges the .cov files of every process; a source file without
# a .cov file reports all its coverable lines as uncovered.
measured = Dict{String, Float64}()
quiet = ConsoleLogger(stderr, Logging.Warn)
for fc in with_logger(() -> CoverageTools.process_folder(SRC), quiet)
    covered, total = CoverageTools.get_summary(fc)
    measured[relpath(fc.filename, PKG)] = total == 0 ? 1.0 : covered / total
end
isempty(measured) && error("no source files found under $SRC")

failures = String[]
for file in sort!(collect(keys(measured)))
    frac = measured[file]
    if !haskey(thresholds, file)
        push!(failures, "$file: no row in $THRESHOLDS (measured $(round(frac; digits = 4)))")
    elseif frac < thresholds[file]
        push!(failures, "$file: coverage $(round(frac; digits = 4)) is below threshold $(thresholds[file])")
    end
    println(rpad(file, 40), round(frac; digits = 4))
end
for file in sort!(collect(keys(thresholds)))
    haskey(measured, file) || push!(failures, "$file: row in $THRESHOLDS has no source file")
end

if !isempty(failures)
    println(stderr, "\nCoverage gate failed:")
    foreach(f -> println(stderr, "  ", f), failures)
    exit(1)
end
println("\nCoverage gate passed for $(length(measured)) files.")
