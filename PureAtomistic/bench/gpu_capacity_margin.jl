# How much spare edge capacity an MD run on the device needs. A `DeviceMACEPotential` rebuilds its
# neighbor list and workspace when a step holds more edges than the list has room for, and every
# spare slot costs evaluation time (the per-edge kernels launch over every slot). This records the
# edge count of every step of a constant-energy run and replays it for several spare fractions:
# the number of rebuilds each fraction would take, against the cost of a rebuild.
#
#   GPU_BACKEND=Metal PUREMLIP_CHECKPOINT_DIR=<dir> julia --project=PureAtomistic/bench/gpu/metal PureAtomistic/bench/gpu_capacity_margin.jl
#
#   GPU_BACKEND              Metal (default), CUDA or AMDGPU; the environment is bench/gpu/<backend in lower case>
#   PUREMLIP_CHECKPOINT_DIR  directory with mace-mpa-0-medium.model
#   MARGIN_SIZE              fcc cells per edge of the Cu crystal (default 5: 500 atoms)
#   MARGIN_STEPS             steps of 2 fs (default 2000)
#   MARGIN_TEMPERATURE       initial temperature in K (default 2600; a crystal started on its
#                            lattice sites at T0 equilibrates near T0/2)
#   PUREMLIP_BENCH_OUT       JSON output path (default results/gpu_capacity_margin_<backend>.json)
#
# On macOS, export HWLOC_COMPONENTS=-darwin first (see the KernelAbstractions rule in AGENTS.md).
#
# The run uses MACE-MPA-0 in Float32 with skin 0 and room for as many edges again as the first
# configuration holds, so that it never rebuilds; it reads the edge count back after every step,
# which a production run does not. A rebuild is timed as a new list and workspace for the largest
# edge count of the run, synchronized, as the median of five.
using PureAtomistic, PureAtomistic.Systems, PureMLIP, KernelAbstractions, Random, Statistics, JSON

const BACKEND_NAME = get(ENV, "GPU_BACKEND", "Metal")
BACKEND_NAME in ("Metal", "CUDA", "AMDGPU") || error("GPU_BACKEND must be Metal, CUDA or AMDGPU, got ", BACKEND_NAME)
@eval using $(Symbol(BACKEND_NAME))
const BACKEND = BACKEND_NAME == "Metal" ? MetalBackend() : BACKEND_NAME == "CUDA" ? CUDABackend() : ROCBackend()

const SIZE = parse(Int, get(ENV, "MARGIN_SIZE", "5"))
const STEPS = parse(Int, get(ENV, "MARGIN_STEPS", "2000"))
const TEMPERATURE = parse(Float64, get(ENV, "MARGIN_TEMPERATURE", "2600"))
const FRACTIONS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
const OUT = get(ENV, "PUREMLIP_BENCH_OUT", joinpath(@__DIR__, "results", "gpu_capacity_margin_$(lowercase(BACKEND_NAME)).json"))

# The rebuilds that a list with spare fraction `f` takes over the edge counts `counts`: it starts
# with room for counts[1] (1 + f) edges and, when a step holds more, is rebuilt with room for that
# count (1 + f), as `DeviceMACEPotential` does.
function rebuilds(counts, f)
    capacity = counts[1] + ceil(Int, f * counts[1])
    n = 0
    for m in counts
        if m > capacity
            n += 1
            capacity = m + ceil(Int, f * m)
        end
    end
    return n
end

function main()
    T = Float32
    path = joinpath(ENV["PUREMLIP_CHECKPOINT_DIR"], "mace-mpa-0-medium.model")
    model = MACEModel{T}(load_mace_checkpoint(path))
    host = fcc_cell(T(3.615), SIZE, :Cu)
    maxwell_boltzmann!(host, TEMPERATURE, Xoshiro(1))
    sys = PureAtomistic.to_device(host, BACKEND)
    p = PureAtomistic.device_potential(model, sys; skin = 0, extra_capacity_fraction = 1.0)
    sim = Simulation(sys, p, VelocityVerlet(2.0); rng = Xoshiro(0))
    counts = Int[Array(p.nl.n_edges)[1]]
    temps = Float64[]
    t0 = time()
    for k in 1:STEPS
        step!(sim)
        push!(counts, Array(p.nl.n_edges)[1])
        k % 10 == 0 && push!(temps, Float64(temperature(sim)))
        if k % 200 == 0
            println("step $k: $(counts[end]) edges, $(round(temps[end]; digits = 1)) K, $(round(time() - t0; digits = 1)) s")
            flush(stdout)
        end
    end
    p.n_regrows == 0 || error("the run rebuilt its list; raise extra_capacity_fraction")
    m = maximum(counts)
    times = Float64[]
    for _ in 1:5
        KernelAbstractions.synchronize(BACKEND)
        t = time_ns()
        nl = PureMLIP.device_neighbor_list(sys.positions, sys.cell, model.r_max; extra_capacity = ceil(Int, 0.1 * m))
        PureMLIP.device_workspace(p.model, natoms(host), nl.capacity)
        KernelAbstractions.synchronize(BACKEND)
        push!(times, (time_ns() - t) / 1.0e6)
        GC.gc()
    end
    second = temps[(length(temps) ÷ 2 + 1):end]
    result = Dict(
        "backend" => BACKEND_NAME, "atoms" => natoms(host), "steps" => STEPS, "dt_fs" => 2.0,
        "initial_temperature_K" => TEMPERATURE, "mean_temperature_second_half_K" => mean(second),
        "edges_first" => counts[1], "edges_min" => minimum(counts), "edges_max" => m,
        "edges_max_over_first" => m / counts[1], "edges_min_over_first" => minimum(counts) / counts[1],
        "edges" => counts, "rebuild_ms" => times, "rebuild_median_ms" => median(times),
        "fractions" => FRACTIONS, "rebuilds" => [rebuilds(counts, f) for f in FRACTIONS],
    )
    mkpath(dirname(OUT))
    open(io -> JSON.print(io, result, 2), OUT, "w")
    println("edges $(counts[1]) first, $(minimum(counts))–$m; mean T (second half) $(round(mean(second); digits = 1)) K")
    println("rebuilds for fractions $FRACTIONS: $(result["rebuilds"]); rebuild $(round(median(times); digits = 1)) ms")
    return nothing
end

main()
