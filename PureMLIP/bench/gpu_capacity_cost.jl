# The cost of spare capacity in a device neighbor list: the per-edge kernels of an evaluation
# launch over every edge slot of the list, so spare slots cost time even though they hold no edge.
# Float32, MACE-MPA-0, `mace_energy_forces!` with the list of `r_max` and extra capacity of 0, 10 %
# and 50 % of the edges.
#
#   GPU_BACKEND=Metal PUREMLIP_CHECKPOINT_DIR=<dir> julia --project=PureMLIP/bench/gpu/metal PureMLIP/bench/gpu_capacity_cost.jl
#
#   GPU_BACKEND              Metal (default), CUDA or AMDGPU; the environment is bench/gpu/<backend in lower case>
#   PUREMLIP_CHECKPOINT_DIR  directory with mace-mpa-0-medium.model
#   CAPACITY_CASES           cu_bench sizes (default "5,14")
#   PUREMLIP_BENCH_OUT       JSON output path (default results/gpu_capacity_cost_<backend>.json)
#
# On macOS, export HWLOC_COMPONENTS=-darwin first (see the KernelAbstractions rule in AGENTS.md).
#
# Each evaluation is synchronized and timed as the median of 10 after two warm-up calls, in five
# rounds that visit the capacities in turn; each figure is the median over the rounds.
using PureMLIP, KernelAbstractions, Statistics, JSON

const BACKEND_NAME = get(ENV, "GPU_BACKEND", "Metal")
BACKEND_NAME in ("Metal", "CUDA", "AMDGPU") || error("GPU_BACKEND must be Metal, CUDA or AMDGPU, got ", BACKEND_NAME)
@eval using $(Symbol(BACKEND_NAME))
const VENDOR = getfield(Main, Symbol(BACKEND_NAME))
VENDOR.functional() || error(BACKEND_NAME, " is not functional on this machine")
const backend = BACKEND_NAME == "Metal" ? VENDOR.MetalBackend() : BACKEND_NAME == "CUDA" ? VENDOR.CUDABackend() : VENDOR.ROCBackend()

include(joinpath(pkgdir(PureMLIP), "test", "reference", "structures.jl"))

const CASES = parse.(Int, split(get(ENV, "CAPACITY_CASES", "5,14"), ","))
const FRACTIONS = (0.0, 0.1, 0.5)
const ROUNDS = 5

function todevice(A::AbstractArray)
    D = KernelAbstractions.allocate(backend, eltype(A), size(A))
    copyto!(D, A)
    return D
end

function median_ms(f; runs = 10)
    f()
    f()
    KernelAbstractions.synchronize(backend)
    t = Float64[]
    for _ in 1:runs
        t0 = time_ns()
        f()
        KernelAbstractions.synchronize(backend)
        push!(t, (time_ns() - t0) / 1.0e6)
    end
    return median(t)
end

function main()
    ck = load_mace_checkpoint(joinpath(ENV["PUREMLIP_CHECKPOINT_DIR"], "mace-mpa-0-medium.model"))
    m = MACEModel{Float32}(ck)
    dm = PureMLIP.device_model(m, backend; elements = [29])
    record = Dict{String, Any}("backend" => BACKEND_NAME, "precision" => "Float32", "model" => "mace-mpa-0-medium", "julia" => string(VERSION), "cases" => Any[])
    for N in CASES
        cu = cu_bench(N)
        P, C = todevice(Float32.(permutedims(cu.positions))), todevice(Float32.(cu.cell))
        Z = todevice(Int32.(cu.atomic_numbers))
        n = size(P, 2)
        E = PureMLIP.device_neighbor_list(P, C, Float64(m.r_max)).capacity
        arms = map(FRACTIONS) do f
            nl = PureMLIP.device_neighbor_list(P, C, Float64(m.r_max); extra_capacity = round(Int, f * E))
            ws = PureMLIP.device_workspace(dm, n, nl.capacity)
            res = PureMLIP.device_result(dm, n)
            () -> mace_energy_forces!(res, ws, dm, P, C, Z, nl)
        end
        times = [Float64[] for _ in FRACTIONS]
        for _ in 1:ROUNDS, (k, run) in enumerate(arms)
            push!(times[k], median_ms(run))
        end
        med = median.(times)
        for (f, t) in zip(FRACTIONS, med)
            println("cu_bench($N): extra capacity $(round(Int, 100f)) %: $(round(t; digits = 2)) ms, $(round(t / med[1]; digits = 4)) of no spare capacity")
            flush(stdout)
        end
        push!(
            record["cases"], Dict{String, Any}(
                "cu_bench" => N, "atoms" => n, "edges" => E, "extra_fraction" => collect(FRACTIONS), "evaluation_ms" => med,
                "rounds_ms" => times, "time_ratio_to_no_spare" => med ./ med[1],
            )
        )
    end
    out = get(ENV, "PUREMLIP_BENCH_OUT", joinpath(@__DIR__, "results", "gpu_capacity_cost_$(lowercase(BACKEND_NAME)).json"))
    write(out, JSON.json(record; pretty = 2))
    println("wrote $out")
    return nothing
end

main()
