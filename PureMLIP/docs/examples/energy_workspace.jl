# # Energies of a large cell
#
# When only energies are needed, a `MACEEnergyWorkspace` takes the place of the
# `MACEWorkspace` in `mace_energy!`. It holds no derivatives and nothing for a backward pass,
# so it is several times smaller. This example evaluates a 4,000-atom Cu cell with it.

using PureMLIP, LinearAlgebra, Printf, JSON
using GLMakie
include(pkgdir(PureMLIP, "docs", "examples", "setup.jl")); #hide

model = MACEModel(load_mace_checkpoint(checkpoint_path(:mpa0)));

# Ten by ten by ten conventional fcc cells at a = 3.63 Å, with each coordinate displaced by up
# to 0.05 Å.

positions, cell = fcc(3.63, 10)
positions .+= 0.05 .* sin.(1.3 .* reshape(1:length(positions), size(positions)))
n = size(positions, 2)
Z = fill(29, n)
nl = neighbor_list(positions, cell, model.r_max)
@printf("%d atoms, %d directed edges (%.1f per atom)\n", n, length(nl.senders), length(nl.senders) / n)

ws_e = MACEEnergyWorkspace{Float64}(model, n, length(nl.senders))
res = MACEResult{Float64}(n)
mace_energy!(res, ws_e, model, positions, cell, Z, nl)
@printf("energy: %.6f eV (%.6f eV per atom)\n", res.energy, res.energy / n)

# The same energy from the full evaluation with a `MACEWorkspace`, which also gives forces and
# stress:

ws = MACEWorkspace{Float64}(model, n, length(nl.senders))
res_full = MACEResult{Float64}(n)
mace_energy_forces!(res_full, ws, model, positions, cell, Z, nl)
@printf("energy: %.6f eV, difference %.1e eV\n", res_full.energy, res_full.energy - res.energy)

# ## Memory per atom
#
# `Base.summarysize` counts every object a workspace refers to once, including buffers that
# share memory.

@printf("MACEWorkspace{Float64}:       %8.0f bytes per atom\n", Base.summarysize(ws) / n)
@printf("MACEEnergyWorkspace{Float64}: %8.0f bytes per atom\n", Base.summarysize(ws_e) / n)

# The committed record `PureMLIP/bench/results/mace_cu14_julia.json` holds the same measure
# for `cu_bench(14)` (10,976 atoms, 78 edges per atom) in both precisions:

record14 = JSON.parsefile(pkgdir(PureMLIP, "bench", "results", "mace_cu14_julia.json"))
cases = [(c["workspace"], c["precision"], c["workspace_bytes_per_atom"]) for c in record14["cases"]]
for (w, p, b) in cases
    @printf("%-20s %-8s %8.0f bytes per atom\n", w, p, b)
end

# The figures per atom are larger for the 4,000-atom cell because part of each workspace has a
# size independent of the number of atoms and edges; the record marks those buffers
# `fixed/chunk`. Adding that part back at 4,000 atoms reproduces the figures measured above:

for c in record14["cases"]
    fixed = sum(b["bytes"] for b in c["blocks"] if b["scale"] == "fixed/chunk")
    growing = (c["workspace_bytes"] - fixed) / c["n_atoms"]
    @printf(
        "%-20s %-8s fixed %5.1f MB + %6.0f bytes per atom; at %d atoms: %6.0f bytes per atom\n",
        c["workspace"], c["precision"], fixed / 1.0e6, growing, n, growing + fixed / n
    )
end

labels = ["$w\n$p" for (w, p, _) in cases]
kb = [b / 1000 for (_, _, b) in cases]
fig = Figure(; size = (820, 420), fontsize = 14)
ax = Axis(
    fig[1, 1]; xticks = (eachindex(labels), labels), ylabel = "kB per atom",
    title = "Workspace memory, cu_bench(14), MACE-MPA-0"
)
barplot!(ax, eachindex(kb), kb; color = [w == "MACEWorkspace" ? :steelblue : :darkorange for (w, _, _) in cases], bar_labels = :y, label_formatter = x -> @sprintf("%.1f", x))
ylims!(ax, 0, 1.15 * maximum(kb))
save(media("energy_workspace.png"), fig)
nothing #hide

# ![](energy_workspace.png)
