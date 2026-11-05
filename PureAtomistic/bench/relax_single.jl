# One relaxer on one cell with MACE-MPA-0, in a process of its own, recording its peak memory.
#
#   PUREMLIP_CHECKPOINT_DIR=<dir> julia -t 1 --project=PureAtomistic/bench PureAtomistic/bench/relax_single.jl \
#       <cell> <method> [rounds]
#
# <cell> is `vacancy` or `strained` (256 sites) or `large` (3,999 atoms: a 10 × 10 × 10 fcc cell with
# one vacancy); <method> is a name of relax_methods.jl or `baseline`, which evaluates the potential
# once and relaxes nothing. The process compiles the method on a 4-atom cell, then relaxes the cell
# `rounds` times (default 1) from the same start. The peak resident set of the process
# (`Sys.maxrss`) is recorded after each round; the method's own peak memory is its figure after
# the first round minus that of `baseline` on the same cell after one round, which holds the
# model, the cell and one potential with its workspace. One thread, BLAS on one thread.
#
# Appends to PureAtomistic/bench/results/relax_single_mac.json (or PUREATOMISTIC_SINGLE_OUT).
using JSON, Statistics
include(joinpath(@__DIR__, "cells.jl"))
include(joinpath(@__DIR__, "relax_methods.jl"))

BLAS.set_num_threads(1)
const CELL = ARGS[1]
const METHOD = ARGS[2]
const ROUNDS = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1
const BUILDS = (vacancy = vacancy_cell, strained = strained_cell, large = large_vacancy_cell)
build = BUILDS[Symbol(CELL)]

function baseline(sys, p)
    prepare!(p, sys.positions, sys.cell)
    energy_forces!(p, sys.positions, sys.cell)
    return 0, 0, Dict{String, Any}()
end
f = METHOD == "baseline" ? baseline : method(METHOD)

model = MACEModel(load_mace_checkpoint(mpa0_path()))
run_one(model, tiny, f)
runs = Any[]
maxrss_rounds = Int[]
for r in 1:ROUNDS
    res = run_one(model, build, f)
    push!(runs, res)
    push!(maxrss_rounds, Int(Sys.maxrss()))
    @printf(
        "%-8s %-18s round %d  evals %4d  %8.1f s  E %.7f eV  fmax %.2e  maxrss %.0f MiB  %s\n",
        CELL, METHOD, r, res.evals, res.seconds, res.energy, res.fmax, Sys.maxrss() / 2^20, string(res.extra)
    )
    flush(stdout)
end
entry = Dict{String, Any}(
    "cell" => CELL, "method" => METHOD, "rounds" => ROUNDS, "natoms" => natoms(build()),
    "seconds" => [x.seconds for x in runs], "median_seconds" => median(x.seconds for x in runs),
    "evals" => [x.evals for x in runs], "iterations" => [x.iterations for x in runs],
    "energy" => [x.energy for x in runs], "fmax" => [x.fmax for x in runs],
    "state_bytes" => first(runs).state_bytes, "extra" => [x.extra for x in runs],
    "maxrss_bytes" => Sys.maxrss(), "maxrss_bytes_after_round" => maxrss_rounds,
    "julia" => string(VERSION), "cpu" => host_cpu(),
    "threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
)
path = get(ENV, "PUREATOMISTIC_SINGLE_OUT", joinpath(@__DIR__, "results", "relax_single_mac.json"))
out = isfile(path) ? JSON.parsefile(path; dicttype = Dict{String, Any}) : Dict{String, Any}(
        "description" => "One relaxer per process on MPA-0 Cu cells, fmax = 1e-3 eV/Å, with the process's peak resident set",
        "fmax" => FMAX, "runs" => Any[],
    )
push!(out["runs"], entry)
open(io -> (JSON.json(io, out; pretty = true); println(io)), path, "w")
println("wrote ", path)
