# # Energy conservation in constant-energy MD
#
# 500 Cu atoms (a 5 × 5 × 5 block of fcc cells at the model's lattice constant) with MACE-MPA-0,
# velocity Verlet, 2 ps. The velocities are drawn from a Maxwell-Boltzmann distribution at 300 K
# (seeded) with the center-of-mass momentum removed, so the crystal starts at the lattice sites with
# a kinetic temperature of 300 K and settles near 150 K as the energy is shared with the potential.
# The same initial state is used for every time step.
#
# This page does not run the dynamics. It reads the record `bench/results/md_nve_conservation.json`
# written by `bench/md_validation.jl`, which also holds the commit, the date and the settings
# of the runs, and plots it.
#
# **Criterion for a stable time step**, fixed before the runs: the run completes with finite
# energies, the drift (the slope of the least-squares line through the total energy per atom,
# sampled every 8 fs) is at most 1 meV/atom/ps in magnitude, and the RMS of the residual about that
# line is at most 0.5 meV/atom.

using CairoMakie, JSON, Printf, PureAtomistic
const RESULTS = joinpath(pkgdir(PureAtomistic), "bench", "results")
d = JSON.parsefile(joinpath(RESULTS, "md_nve_conservation.json"))
s = d["settings"]
@printf(
    "commit %s, %s, Julia %s, BLAS %s on %d thread, %d atoms, seed %d\n",
    s["commit"], s["date"], s["julia"], s["blas"], s["blas_threads"], s["natoms"], s["seed"]
)
runs = d["runs"]
nothing #hide

# ## Drift and fluctuation
#
# The drift is the slope of the line fitted to the samples; the error is the standard error of the
# slope assuming independent residuals, which the samples are not, so it is a lower bound on the
# uncertainty. Energies are in meV per atom.

@printf("%6s  %10s  %22s  %12s  %8s\n", "dt/fs", "status", "drift (meV/atom/ps)", "RMS (meV)", "stable")
for r in runs
    if haskey(r, "drift_eV_per_atom_per_ps")
        @printf(
            "%6g  %10s  %12.4f ± %-8.4f  %12.4f  %8s\n", r["dt_fs"], r["status"], 1.0e3 * r["drift_eV_per_atom_per_ps"],
            1.0e3 * r["drift_stderr"], 1.0e3 * r["rms_residual_eV_per_atom"], r["stable"]
        )
    else
        @printf("%6g  %10s  %s\n", r["dt_fs"], r["status"], r["message"])
    end
end
stable = [r["dt_fs"] for r in runs if r["stable"]]
@printf("largest stable time step among those run: %g fs\n", maximum(stable))

# The left panel shows the energy of the runs with dt up to 8 fs; at larger steps the deviation is
# too large for the same axis (see the table). The fluctuation of the total energy of velocity Verlet is a property of the shadow Hamiltonian
# and scales as `dt²`; the dashed line is that scaling through the smallest time step.

fig = Figure(; size = (900, 380), fontsize = 13)
ax1 = Axis(fig[1, 1]; xlabel = "time (ps)", ylabel = "E(t) − E(0) (meV/atom)", title = "total energy")
for r in filter(r -> r["dt_fs"] <= 8, runs)
    r["status"] == "completed" || continue
    lines!(ax1, r["t_fs"] ./ 1000, 1.0e3 .* (r["total_eV_per_atom"] .- r["total_eV_per_atom"][1]); label = "dt = $(r["dt_fs"]) fs")
end
axislegend(ax1; position = :lb)
ax2 = Axis(
    fig[1, 2]; xlabel = "dt (fs)", ylabel = "RMS residual (meV/atom)", xscale = log2, yscale = log10, title = "fluctuation against time step"
)
ok = [r for r in runs if haskey(r, "rms_residual_eV_per_atom")]
dts = [r["dt_fs"] for r in ok]
rms = [1.0e3 * r["rms_residual_eV_per_atom"] for r in ok]
scatter!(ax2, dts, rms; markersize = 10)
lines!(ax2, dts, rms[1] .* (dts ./ dts[1]) .^ 2; color = :gray, linestyle = :dash, label = "∝ dt²")
hlines!(ax2, [1.0e3 * d["settings"]["stable_rms_eV_per_atom"]]; color = :crimson, linestyle = :dot, label = "criterion")
axislegend(ax2; position = :lt)
save("nve_energy.png", fig)
nothing #hide

# ![](nve_energy.png)

# ## Animation
#
# The atoms of the dt = 2 fs run (colored by their distance from the starting lattice site, in Å) and
# the total-energy record of all the runs up to the time shown. The animation is rendered
# separately by `docs/render_animations.jl` from the trajectory of that run, which is not stored in
# the repository, and saved to `docs/src/assets/`.
#
# ```@raw html
# <video src="../assets/nve_conservation.mp4" controls loop muted playsinline width="720"></video>
# ```
