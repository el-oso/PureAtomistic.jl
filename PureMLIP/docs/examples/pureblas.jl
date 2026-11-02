# # Speed with PureBLAS
#
# The matrix products of the model go through `LinearAlgebra.mul!`, so loading PureBLAS and
# activating it routes them to PureBLAS without any change to PureMLIP:
#
# ```julia
# using PureBLAS
# PureBLAS.activate()
# ```
#
# PureBLAS is not a dependency of PureMLIP, and this page does not load it or time anything. It
# reads two committed records of interleaved A/B runs from `PureMLIP/bench/results/`:
# `ab_plan_total_openblas_mac.json` (OpenBLAS on both sides) and
# `ab_plan_total_pureblas_mac.json` (PureBLAS active on the measured side). In both, the
# reference is the same PureMLIP commit with OpenBLAS, and each pair of child processes reports
# the ratio of measured to reference time. The speed-up shown is the reference time divided by
# the measured time, so a value above 1 is faster than the reference.
#
# The measured side is a later PureMLIP commit than the reference, so both bars include the
# code changes between the two commits, and in `mace_energy!` the measured side uses a
# `MACEEnergyWorkspace` where the reference uses a `MACEWorkspace`. The PureBLAS bars differ
# from the OpenBLAS bars only in the BLAS library.

using PureMLIP, Printf, JSON
using GLMakie
include(pkgdir(PureMLIP, "docs", "examples", "setup.jl")); #hide

results = pkgdir(PureMLIP, "bench", "results")
runs = [
    "OpenBLAS" => JSON.parsefile(joinpath(results, "ab_plan_total_openblas_mac.json")),
    "PureBLAS active" => JSON.parsefile(joinpath(results, "ab_plan_total_pureblas_mac.json")),
]
for (name, d) in runs
    @printf(
        "%-16s reference: PureMLIP %s with OpenBLAS; measured: PureMLIP %s; %s, Julia %s, %d thread\n",
        name, d["base_commit"], d["head_commit"], d["machine"], d["julia_version"], d["threads"]
    )
    @printf("%-16s mace_energy! workspace: reference %s, measured %s\n", name, d["energy_workspace"]["base"], d["energy_workspace"]["head"])
end

# One row per structure, precision and entry point: the speed-up from the median pair ratio,
# and in parentheses the range over the kept pairs.

rows = []
for (name, d) in runs, c in d["cases"], e in c["entries"]
    push!(rows, (; name, system = c["system"], precision = c["precision"], entry = e["entry_point"], s = 1 / e["median"], lo = 1 / e["max"], hi = 1 / e["min"], pairs = length(e["pair_ratios"])))
end
for r in rows
    @printf("%-16s %-12s %-8s %-20s speed-up %.2f (%.2f–%.2f, %d pairs)\n", r.name, r.system, r.precision, r.entry, r.s, r.lo, r.hi, r.pairs)
end

# The ratio of the two speed-ups is the gain from PureBLAS alone. The two records come from
# separate runs against the same reference, so this ratio is not itself an interleaved
# measurement.

for r in filter(r -> r.name == "PureBLAS active", rows)
    o = only(filter(q -> q.name == "OpenBLAS" && (q.system, q.precision, q.entry) == (r.system, r.precision, r.entry), rows))
    @printf("%-8s %-20s PureBLAS / OpenBLAS: %.2f\n", r.precision, r.entry, r.s / o.s)
end

groups = unique([(r.precision, r.entry) for r in rows])
x = [findfirst(==((r.precision, r.entry)), groups) for r in rows]
dodge = [r.name == "OpenBLAS" ? 1 : 2 for r in rows]
fig = Figure(; size = (720, 420), fontsize = 14)
ax = Axis(
    fig[1, 1]; xticks = (eachindex(groups), ["$e\n$p" for (p, e) in groups]), ylabel = "speed-up (above 1 is faster)",
    title = "cu_bench(5), MACE-MPA-0, against PureMLIP $(runs[1][2]["base_commit"]) with OpenBLAS"
)
colors = [:steelblue, :darkorange]
barplot!(ax, x, [r.s for r in rows]; dodge, color = colors[dodge], bar_labels = :y, label_formatter = v -> @sprintf("%.2f", v), label_offset = 12)
xs = x .+ (dodge .- 1.5) .* 0.4
rangebars!(ax, xs, [r.lo for r in rows], [r.hi for r in rows]; color = :black, whiskerwidth = 6)
hlines!(ax, [1.0]; color = :gray, linestyle = :dash)
ylims!(ax, 0, 1.15 * maximum(r.hi for r in rows))
Legend(fig[1, 2], [PolyElement(; color = c) for c in colors], first.(runs))
save(media("pureblas.png"), fig)
nothing #hide

# ![](pureblas.png)
