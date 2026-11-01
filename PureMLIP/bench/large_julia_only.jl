# Julia-only time and memory of `mace_energy_forces!` on cu_bench(14) (10,976 atoms), and the
# per-field memory of `MACEWorkspace` in Float64. LAMMPS does not fit in memory at this size.
#
#   julia -t 1 --project=PureMLIP/bench PureMLIP/bench/large_julia_only.jl
#
#   PUREMLIP_CHECKPOINT_DIR  directory with mace-mpa-0-medium.model
#   PUREMLIP_BENCH_OUT       JSON output path (default results/mace_cu14_julia.json)
using PureMLIP, Chairmarks, JSON, LinearAlgebra, Statistics

BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("run with julia -t 1")
include(joinpath(@__DIR__, "..", "test", "reference", "structures.jl"))
const OUT = get(ENV, "PUREMLIP_BENCH_OUT", joinpath(@__DIR__, "results", "mace_cu14_julia.json"))

swap_used_mb() = parse(Float64, match(r"used = ([0-9.]+)M", read(`sysctl -n vm.swapusage`, String))[1])

# Bytes of each workspace field, and which extent it scales with.
function field_breakdown(ws, n_atoms, n_edges)
    rows = []
    for f in fieldnames(typeof(ws))
        v = getfield(ws, f)
        arrays = v isa AbstractArray ? (v,) : v isa Tuple && all(x -> x isa AbstractArray, v) ? v : continue
        sz = size(first(arrays))
        scale = n_edges in sz ? "per-edge" : n_atoms in sz ? "per-atom" : "fixed/chunk"
        push!(rows, (; field = string(f), type = string(typeof(v)), size = collect(sz), count = length(arrays), bytes = sum(Base.summarysize, arrays), scale))
    end
    sort!(rows; by = r -> -r.bytes)
    return rows
end

ck = load_mace_checkpoint(joinpath(ENV["PUREMLIP_CHECKPOINT_DIR"], "mace-mpa-0-medium.model"))
cu = cu_bench(14)
z = cu.atomic_numbers
n = length(z)
cases = []
breakdown = nothing
for T in (Float64, Float32)
    model = MACEModel{T}(ck)
    positions = Matrix{T}(cu.positions'); cell = Matrix{T}(cu.cell')
    nl = neighbor_list(positions, cell, model.r_max; extra_capacity = 64)
    ws = MACEWorkspace{T}(model, n, nl.n_edges)
    res = MACEResult{T}(n)
    sw0 = swap_used_mb()
    mace_energy_forces!(res, ws, model, positions, cell, z, nl)  # warm-up, touches every page
    t = [b.time for b in (@be mace_energy_forces!(res, ws, model, positions, cell, z, nl) evals = 1 samples = 3 seconds = 600).samples]
    tn = [b.time for b in (@be neighbor_list!(nl, positions, cell) evals = 1 samples = 3 seconds = 600).samples]
    sw1 = swap_used_mb()
    push!(
        cases, (;
            precision = string(T), n_atoms = n, n_edges = nl.n_edges,
            workspace_bytes = Base.summarysize(ws), neighbor_list_bytes = Base.summarysize(nl),
            julia_peak_rss_bytes_so_far = Sys.maxrss(),
            julia_samples_s = t, julia_median_s = median(t), neighbor_list_samples_s = tn, neighbor_list_median_s = median(tn),
            swap_start_mb = sw0, swap_end_mb = sw1,
        )
    )
    T === Float64 && (global breakdown = field_breakdown(ws, n, nl.n_edges))
    println(cases[end])
    ws = nothing; res = nothing; GC.gc()
end
doc = (;
    schema = "puremlip-mace-cu14-julia-v1", julia_version = string(VERSION),
    machine = strip(read(`sysctl -n machdep.cpu.brand_string`, String)),
    timed = "mace_energy_forces! with a fixed neighbor list, 3 samples, one thread",
    cases, workspace_float64_fields = breakdown,
)
open(io -> (JSON.json(io, doc; pretty = true); println(io)), OUT, "w")
