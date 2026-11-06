# Adds the autocorrelation-based errors to `results/md_thermostat_check.json`, computed from the
# stored kinetic-temperature series; the raw data and the block estimates are left unchanged.
# Usage: julia --project=bench bench/md_thermostat_errors.jl
using JSON, Statistics
include(joinpath(@__DIR__, "autocorrelation.jl"))

const FILE = joinpath(@__DIR__, "results", "md_thermostat_check.json")
d = JSON.parsefile(FILE)
t, T = Float64.(d["t_fs"]), Float64.(d["temperature_K"])
x = T[t .> d["settings"]["analysis_window_fs"][1]]
th, ob = d["theory"], d["observed"]
ob["autocorrelation"] = autocorrelation_record(x, th["mean_K"], ob["skewness"], th["skewness"])
open(FILE, "w") do io
    JSON.print(io, d, 1)
    println(io)
end
