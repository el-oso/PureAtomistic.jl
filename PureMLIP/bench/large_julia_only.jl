# Memory of `MACEWorkspace` and `MACEEnergyWorkspace` on cu_bench(14) (10,976 atoms), in Float64
# and Float32, with the per-buffer breakdown of each. Workspaces are constructed, not evaluated:
# LAMMPS does not fit in memory at this size, and the sizes follow from the constructors.
#
#   julia -t 1 --project=PureMLIP/bench PureMLIP/bench/large_julia_only.jl
#
# macOS only: it reads `sysctl` (swap usage, CPU brand).
#
#   PUREMLIP_CHECKPOINT_DIR  directory with mace-mpa-0-medium.model
#   PUREMLIP_BENCH_OUT       JSON output path (default results/mace_cu14_julia.json)
using PureMLIP, JSON, LinearAlgebra

BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("run with julia -t 1")
include(joinpath(@__DIR__, "..", "test", "reference", "structures.jl"))
const OUT = get(ENV, "PUREMLIP_BENCH_OUT", joinpath(@__DIR__, "results", "mace_cu14_julia.json"))

swap_used_mb() = parse(Float64, match(r"used = ([0-9.]+)M", read(`sysctl -n vm.swapusage`, String))[1])

# The commit of this checkout, with "-dirty" when tracked files differ from it.
function head_commit()
    sha = readchomp(`git -C $(@__DIR__) rev-parse --short=12 HEAD`)
    clean = isempty(readchomp(`git -C $(@__DIR__) status --porcelain --untracked-files=no`))
    return clean ? sha : sha * "-dirty"
end

# Bytes of the workspace's array fields, one row per distinct underlying `Memory`: several
# arrays that share a block (a pool, or a reshape of an array) are listed under the one row that
# counts the block once. A tuple field holds arrays of different sizes; each member is named
# `field[i]`. `scale` is the extent the block's members scale with.
function field_breakdown(ws, n_atoms, n_edges)
    members = Pair{String, AbstractArray}[]
    for f in fieldnames(typeof(ws))
        v = getfield(ws, f)
        if v isa AbstractArray
            push!(members, string(f) => v)
        elseif v isa Tuple && all(x -> x isa AbstractArray, v)
            for (i, a) in pairs(v)
                push!(members, "$f[$i]" => a)
            end
        end
    end
    order = Memory[]
    groups = IdDict{Memory, Vector{Pair{String, AbstractArray}}}()
    for m in members
        mem = m.second.ref.mem
        haskey(groups, mem) || (push!(order, mem); groups[mem] = [])
        push!(groups[mem], m)
    end
    rows = map(order) do mem
        ms = groups[mem]
        arrays = last.(ms)
        scale = any(a -> n_edges in size(a), arrays) ? "per-edge" : any(a -> n_atoms in size(a), arrays) ? "per-atom" : "fixed/chunk"
        (; members = first.(ms), sizes = [collect(size(a)) for a in arrays], shared = length(ms) > 1, bytes = sizeof(mem), scale)
    end
    sort!(rows; by = r -> -r.bytes)
    return rows
end

ck = load_mace_checkpoint(joinpath(ENV["PUREMLIP_CHECKPOINT_DIR"], "mace-mpa-0-medium.model"))
cu = cu_bench(14)
z = cu.atomic_numbers
n = length(z)
const HEAD_COMMIT = head_commit()
cases = []
for T in (Float64, Float32), W in (MACEWorkspace, MACEEnergyWorkspace)
    model = MACEModel{T}(ck)
    positions = Matrix{T}(cu.positions'); cell = Matrix{T}(cu.cell')
    nl = neighbor_list(positions, cell, model.r_max; extra_capacity = 64)
    sw0 = swap_used_mb()
    ws = W{T}(model, n, nl.n_edges)
    sw1 = swap_used_mb()
    rows = field_breakdown(ws, n, nl.n_edges)
    push!(
        cases, (;
            workspace = string(nameof(W)), precision = string(T), n_atoms = n, n_edges = nl.n_edges,
            workspace_bytes = Base.summarysize(ws), workspace_bytes_per_atom = Base.summarysize(ws) / n,
            memory_blocks_bytes = sum(r -> r.bytes, rows), neighbor_list_bytes = Base.summarysize(nl),
            julia_peak_rss_bytes_so_far = Sys.maxrss(), swap_start_mb = sw0, swap_end_mb = sw1, blocks = rows,
        )
    )
    println(cases[end].workspace, " ", T, " ", cases[end].workspace_bytes, " B  ", round(cases[end].workspace_bytes_per_atom; digits = 1), " B/atom")
    ws = nothing; GC.gc()
end
doc = (;
    schema = "puremlip-mace-cu14-julia-v2", head_commit = HEAD_COMMIT, julia_version = string(VERSION),
    machine = strip(read(`sysctl -n machdep.cpu.brand_string`, String)),
    blas_libraries = [basename(l.libname) for l in BLAS.get_config().loaded_libs],
    measured = "workspace construction on cu_bench(14); no evaluation",
    cases,
)
open(io -> (JSON.json(io, doc; pretty = true); println(io)), OUT, "w")
