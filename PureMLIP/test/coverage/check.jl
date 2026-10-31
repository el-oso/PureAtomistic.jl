# Per-file line-coverage gate. Run after the suite with coverage enabled:
#   julia --project=PureMLIP/test/coverage PureMLIP/test/coverage/check.jl
# Compares each src file's covered/coverable line fraction with thresholds.toml
# and exits non-zero on any file below its threshold or without a row.
using CoverageTools, Logging, TOML

const PKG = normpath(joinpath(@__DIR__, "..", ".."))
const SRC = joinpath(PKG, "src")

thresholds = TOML.parsefile(joinpath(@__DIR__, "thresholds.toml"))
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
        push!(failures, "$file: no row in thresholds.toml (measured $(round(frac; digits = 4)))")
    elseif frac < thresholds[file]
        push!(failures, "$file: coverage $(round(frac; digits = 4)) is below threshold $(thresholds[file])")
    end
    println(rpad(file, 40), round(frac; digits = 4))
end
for file in sort!(collect(keys(thresholds)))
    haskey(measured, file) || push!(failures, "$file: row in thresholds.toml has no source file")
end

if !isempty(failures)
    println(stderr, "\nCoverage gate failed:")
    foreach(f -> println(stderr, "  ", f), failures)
    exit(1)
end
println("\nCoverage gate passed for $(length(measured)) files.")
