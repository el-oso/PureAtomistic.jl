# How often a `DeviceMACEPotential` rebuilds its neighbor list under the Berendsen barostat. The
# barostat scales the cell every step, so the edge count follows the volume: a shrinking cell packs
# more edges into the same cutoff sphere, and a list with too little spare room rebuilds. This runs
# NPT on the device (Bussi thermostat and Berendsen barostat) at the default spare fraction and
# records each step's edge count, volume and the potential's regrowths, then replays the edge
# counts for several spare fractions, as `gpu_capacity_margin.jl` does at constant volume.
#
#   GPU_BACKEND=Metal PUREMLIP_CHECKPOINT_DIR=<dir> julia --project=PureAtomistic/bench/gpu/metal PureAtomistic/bench/gpu_barostat_regrow.jl
#
#   GPU_BACKEND              Metal (default), CUDA or AMDGPU; the environment is bench/gpu/<backend in lower case>
#   PUREMLIP_CHECKPOINT_DIR  directory with mace-mpa-0-medium.model
#   REGROW_SIZE              fcc cells per edge of the Cu crystal (default 5: 500 atoms)
#   REGROW_STEPS             steps of 2 fs per target pressure (default 2000)
#   REGROW_P0                comma-separated target pressures in GPa (default "0,10")
#   PUREMLIP_BENCH_OUT       JSON output path (default results/gpu_barostat_regrow_<backend>.json)
#
# On macOS, export HWLOC_COMPONENTS=-darwin first (see the KernelAbstractions rule in AGENTS.md).
#
# Each run starts from the 300 K lattice constant 3.615 Å with velocities at 2600 K on the lattice
# sites (the crystal equilibrates near 1300 K), MACE-MPA-0 in Float32, skin 0, the default spare
# fraction 0.05, the Bussi thermostat at 1300 K (τ = 100 fs) and the Berendsen barostat at P0 with
# τ_P = 500 fs and β_T = 1/140 GPa⁻¹ (about the compressibility of copper). It reads the edge count
# and the cell back after every step, which a production run does not.
using PureAtomistic, PureAtomistic.Systems, PureMLIP, KernelAbstractions, Random, Statistics, JSON

const BACKEND_NAME = get(ENV, "GPU_BACKEND", "Metal")
BACKEND_NAME in ("Metal", "CUDA", "AMDGPU") || error("GPU_BACKEND must be Metal, CUDA or AMDGPU, got ", BACKEND_NAME)
@eval using $(Symbol(BACKEND_NAME))
const BACKEND = BACKEND_NAME == "Metal" ? MetalBackend() : BACKEND_NAME == "CUDA" ? CUDABackend() : ROCBackend()

const SIZE = parse(Int, get(ENV, "REGROW_SIZE", "5"))
const STEPS = parse(Int, get(ENV, "REGROW_STEPS", "2000"))
const PRESSURES = parse.(Float64, split(get(ENV, "REGROW_P0", "0,10"), ","))
const FRACTION = 0.05
const FRACTIONS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
const OUT = get(ENV, "PUREMLIP_BENCH_OUT", joinpath(@__DIR__, "results", "gpu_barostat_regrow_$(lowercase(BACKEND_NAME)).json"))

# The rebuilds that a list with spare fraction `f` takes over the edge counts `counts`, as in
# gpu_capacity_margin.jl.
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

function run_one(model, P0)
    T = Float32
    host = fcc_cell(T(3.615), SIZE, :Cu)
    maxwell_boltzmann!(host, 2600.0, Xoshiro(1))
    sys = PureAtomistic.to_device(host, BACKEND)
    p = PureAtomistic.device_potential(model, sys; skin = 0, extra_capacity_fraction = FRACTION)
    sim = Simulation(
        sys, p, VelocityVerlet(2.0); thermostat = BussiThermostat(1300.0, 100.0),
        barostat = BerendsenBarostat(P0, 500.0, 1 / 140), rng = Xoshiro(0)
    )
    counts = Int[Array(p.nl.n_edges)[1]]
    volumes = Float64[cell_volume(Array(sys.cell))]
    regrow_steps = Int[]
    t0 = time()
    for k in 1:STEPS
        r = p.n_regrows
        step!(sim)
        p.n_regrows > r && push!(regrow_steps, k)
        push!(counts, Array(p.nl.n_edges)[1])
        push!(volumes, cell_volume(Array(sys.cell)))
        if k % 200 == 0
            println("P0 $P0 GPa, step $k: $(counts[end]) edges, V/V0 $(round(volumes[end] / volumes[1]; digits = 4)), $(p.n_regrows) regrowths, $(round(time() - t0; digits = 1)) s")
            flush(stdout)
        end
    end
    return Dict(
        "P0_GPa" => P0, "regrowths" => p.n_regrows, "regrow_steps" => regrow_steps,
        "readbacks" => p.readbacks, "edges_first" => counts[1], "edges_min" => minimum(counts),
        "edges_max" => maximum(counts), "edges_max_over_first" => maximum(counts) / counts[1],
        "volume_min_over_first" => minimum(volumes) / volumes[1], "volume_max_over_first" => maximum(volumes) / volumes[1],
        "volume_last_over_first" => volumes[end] / volumes[1], "edges" => counts, "volumes_A3" => volumes,
        "fractions" => FRACTIONS, "replayed_rebuilds" => [rebuilds(counts, f) for f in FRACTIONS],
        "wall_s" => time() - t0,
    )
end

function main()
    path = joinpath(ENV["PUREMLIP_CHECKPOINT_DIR"], "mace-mpa-0-medium.model")
    model = MACEModel{Float32}(load_mace_checkpoint(path))
    runs = [run_one(model, P0) for P0 in PRESSURES]
    result = Dict(
        "backend" => BACKEND_NAME, "atoms" => 4 * SIZE^3, "steps" => STEPS, "dt_fs" => 2.0,
        "extra_capacity_fraction" => FRACTION, "thermostat" => "Bussi 1300 K, tau 100 fs",
        "barostat" => "Berendsen tau_P 500 fs, beta_T 1/140 per GPa", "runs" => runs,
    )
    mkpath(dirname(OUT))
    open(io -> JSON.print(io, result, 2), OUT, "w")
    for r in runs
        println("P0 $(r["P0_GPa"]) GPa: $(r["regrowths"]) regrowths in $STEPS steps at steps $(r["regrow_steps"]); edges $(r["edges_first"]) first, $(r["edges_min"])–$(r["edges_max"]); V/V0 $(round(r["volume_min_over_first"]; digits = 4))–$(round(r["volume_max_over_first"]; digits = 4)); replayed rebuilds for $FRACTIONS: $(r["replayed_rebuilds"])")
    end
    return nothing
end

main()
