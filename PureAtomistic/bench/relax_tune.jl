# Parameters of the relaxers on the two Cu cells with MACE-MPA-0: force evaluations to
# fmax = 1e-3 eV/Å for each setting.
#
#   PUREMLIP_CHECKPOINT_DIR=<dir> PUREATOMISTIC_TUNE_METHOD=FIRE \
#       julia -t 1 --project=PureAtomistic/bench PureAtomistic/bench/relax_tune.jl
#
# PUREATOMISTIC_TUNE_METHOD is FIRE (the default), FIRE2 or TrustRegionLBFGS. Without
# PUREATOMISTIC_TUNE_CONFIGS, FIRE and FIRE2 run their defaults and `dt`, `dtmax`, `maxstep` and
# `nmin` (FIRE2: `ndelay`) varied one at a time, and TrustRegionLBFGS runs `memory` = 1, 3, 5, 10,
# 20, 40. PUREATOMISTIC_TUNE_CONFIGS replaces that list by its own: one setting per `;`, each a
# comma-separated list of `name=value` (for example `dt=0.2,dtmax=2.0;maxstep=0.2`).
#
# Every run is deterministic, so its evaluation count is reproduced exactly by every round; the
# count is the measure, and wall time (recorded alongside) is proportional to it because one MPA-0
# evaluation of a 256-site cell costs the same at every setting. Every run starts from the same
# configuration and stops when the largest atomic force is below fmax or after MAXSTEPS steps.
# One thread, BLAS on one thread.
#
# Appends to PureAtomistic/bench/results/relax_tune_mac.json (or PUREATOMISTIC_TUNE_OUT).
using JSON, Printf
include(joinpath(@__DIR__, "cells.jl"))
using PureAtomistic.Integrators: FIRE, FIRE2, TrustRegionLBFGS

BLAS.set_num_threads(1)
const FMAX = 1.0e-3
const MAXSTEPS = 2000
const METHOD = get(ENV, "PUREATOMISTIC_TUNE_METHOD", "FIRE")
METHOD in ("FIRE", "FIRE2", "TrustRegionLBFGS") || error("unknown PUREATOMISTIC_TUNE_METHOD $METHOD")

const DEFAULTS = Dict{String, Any}(
    "FIRE" => (dt = 0.1, dtmax = 1.0, maxstep = 0.1, nmin = 5),
    "FIRE2" => (dt = 0.1, dtmax = 1.0, maxstep = 0.1, ndelay = 20),
    "TrustRegionLBFGS" => (memory = 10,),
)[METHOD]
const INTEGER_KEYS = ("nmin", "ndelay", "memory")

function default_configs()
    METHOD == "TrustRegionLBFGS" && return [Dict("memory" => Float64(m)) for m in (1, 3, 5, 10, 20, 40)]
    configs = [Dict{String, Float64}()]
    delay = METHOD == "FIRE" ? "nmin" => (2.0, 10.0) : "ndelay" => (5.0, 10.0)
    for (k, vals) in ("dt" => (0.05, 0.2, 0.3), "dtmax" => (0.5, 2.0, 3.0), "maxstep" => (0.05, 0.2, 0.3), delay)
        for v in vals
            push!(configs, Dict(k => v))
        end
    end
    return configs
end

function parse_configs(s)
    return [
        Dict(String(strip(first(split(kv, "=")))) => parse(Float64, last(split(kv, "="))) for kv in split(c, ",") if !isempty(strip(kv)))
            for c in split(s, ";")
    ]
end

const CONFIGS = haskey(ENV, "PUREATOMISTIC_TUNE_CONFIGS") ? parse_configs(ENV["PUREATOMISTIC_TUNE_CONFIGS"]) : default_configs()

function settings(c)
    d = Dict{Symbol, Any}(pairs(DEFAULTS))
    for (k, v) in c
        d[Symbol(k)] = k in INTEGER_KEYS ? Int(v) : v
    end
    # dtmax must not fall below dt.
    haskey(d, :dtmax) && (d[:dtmax] = max(d[:dtmax], d[:dt]))
    return (; d...)
end

function state(sys, s)
    METHOD == "FIRE" && return FIRE(sys; s...)
    METHOD == "FIRE2" && return FIRE2(sys; s...)
    return TrustRegionLBFGS(sys; s...)
end

function run_one(model, build, s)
    sys = build()
    p = Counting(mace(model, sys))
    opt = state(sys, s)
    t0 = time_ns()
    r = relax!(sys, p, opt; fmax = FMAX, maxsteps = MAXSTEPS)
    t = (time_ns() - t0) / 1.0e9
    fm, E = fresh_fmax(model, sys)
    boxed = opt isa TrustRegionLBFGS ? opt.boxed : -1
    return (; seconds = t, evals = p.evals, steps = r.steps, converged = r.converged, energy = E, fmax = fm, boxed)
end

model = MACEModel(load_mace_checkpoint(mpa0_path()))
let sys = fcc_cell(A0, 1, :Cu)
    sys.positions .+= 0.05 .* randn(Xoshiro(2), size(sys.positions))
    relax!(sys, mace(model, sys), state(sys, settings(Dict{String, Float64}())); fmax = FMAX, maxsteps = 20)
end
path = get(ENV, "PUREATOMISTIC_TUNE_OUT", joinpath(@__DIR__, "results", "relax_tune_mac.json"))
out = isfile(path) ? JSON.parsefile(path; dicttype = Dict{String, Any}) : Dict{String, Any}(
        "description" => "Relaxer settings on the MPA-0 Cu cells: force evaluations to fmax = 1e-3 eV/Å",
        "fmax" => FMAX, "maxsteps" => MAXSTEPS, "runs" => Any[],
    )
out["julia"] = string(VERSION)
out["cpu"] = host_cpu()
out["threads"] = Threads.nthreads()
for c in CONFIGS, (name, build) in pairs(CELLS)
    s = settings(c)
    res = run_one(model, build, s)
    @printf(
        "%-16s %-8s %-60s evals %4d  %6.1f s  converged %s  E %.7f  fmax %.2e  boxed %d\n",
        METHOD, name, string(s), res.evals, res.seconds, res.converged, res.energy, res.fmax, res.boxed
    )
    flush(stdout)
    push!(
        out["runs"], Dict{String, Any}(
            "method" => METHOD, "cell" => String(name),
            "settings" => Dict(String(k) => v for (k, v) in pairs(s)), "evals" => res.evals,
            "steps" => res.steps, "seconds" => res.seconds, "converged" => res.converged,
            "energy" => res.energy, "fmax" => res.fmax, "boxed" => res.boxed,
        )
    )
    open(io -> (JSON.json(io, out; pretty = true); println(io)), path, "w")
end
println("wrote ", path)
