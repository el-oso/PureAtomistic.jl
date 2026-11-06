# # A vacancy in hot Cu
#
# One vacancy in a 4 × 4 × 4 block of fcc cells (255 atoms) of Cu with MACE-MPA-0 at 1300 K, run
# at constant volume with the Bussi thermostat, and followed by a `VacancyTracker`.
#
# **This is an illustration of the tracker on a molecular-dynamics run, not a measurement of a
# diffusion coefficient or a migration barrier.** One cell, one temperature and one trajectory of
# a few picoseconds give a handful of hops at most; their count carries a large statistical error
# (a Poisson count of `n` has a relative error of `1/√n`), the cell is held at the lattice constant
# of the zero-temperature crystal rather than at its thermal expansion at 1300 K, and the
# tracker's window and dwell are chosen by hand. The campaign over a range of temperatures that
# would give an Arrhenius fit is not part of this page.
#
# This page does not run the dynamics. It reads `bench/results/md_vacancy_hop.json`, which holds
# the tracker's record together with the commit, the date and the settings, and plots it.

using CairoMakie, JSON, Printf, PureAtomistic
const RESULTS = joinpath(pkgdir(PureAtomistic), "bench", "results")
d = JSON.parsefile(joinpath(RESULTS, "md_vacancy_hop.json"))
s = d["settings"]
@printf(
    "commit %s, %s, Julia %s, BLAS %s on %d thread, %d atoms, %g K, dt = %g fs, seed %d\n",
    s["commit"], s["date"], s["julia"], s["blas"], s["blas_threads"], s["natoms"], s["temperature_K"], s["dt_fs"], s["seed"]
)
@printf(
    "%s: %d steps = %.2f ps in %.0f s of wall time (budget %.0f s); mean kinetic temperature %.0f K\n",
    d["status"], d["steps"], d["duration_ps"], s["wall_seconds"], s["wall_budget_seconds"], d["mean_temperature_K"]
)
@printf(
    "tracker: frame every %g fs, positions averaged over %d frames, a new site must hold for %d frames\n",
    s["frame_interval_fs"], s["tracker_window_frames"], s["tracker_dwell_frames"]
)
@printf("confirmed hops: %d, at t = %s fs\n", d["hops"], join(map(t -> @sprintf("%.0f", t), d["hop_times_fs"]), ", "))
isempty(d["message"]) || println("run stopped by: ", d["message"])

# ## The track
#
# The unwrapped position of the vacancy and its squared displacement from the starting site, with a
# dashed line at each confirmed hop. A hop of the vacancy to a nearest-neighbor site is a
# displacement of `a/√2 ≈ 2.57 Å`, so its squared displacement is `a²/2 ≈ 6.6 Å²`. The
# squared displacement of one trajectory is not a mean-square displacement and is not fitted.

t = Float64.(d["t_fs"]) ./ 1000
R = reduce(hcat, [Float64.(r) for r in d["vacancy_R"]])
hops = Float64.(d["hop_times_fs"]) ./ 1000
fig = Figure(; size = (900, 520), fontsize = 13)
ax1 = Axis(fig[1, 1]; ylabel = "vacancy position (Å)")
for (c, name) in enumerate(("x", "y", "z"))
    lines!(ax1, t, R[c, :]; label = name)
end
isempty(hops) || vlines!(ax1, hops; color = (:gray, 0.7), linestyle = :dash)
axislegend(ax1; position = :lt)
ax2 = Axis(fig[2, 1]; xlabel = "time (ps)", ylabel = "squared displacement (Å²)")
lines!(ax2, t, [sum(abs2, R[:, k] - R[:, 1]) for k in axes(R, 2)])
isempty(hops) || vlines!(ax2, hops; color = (:gray, 0.7), linestyle = :dash)
linkxaxes!(ax1, ax2)
save("vacancy_track.png", fig)
nothing #hide

# ![](vacancy_track.png)

# ## Animation
#
# The atoms of the cell (small, colored by height) and the vacancy site (large red sphere), with the
# unwrapped vacancy position up to the time shown. The animation is rendered separately by
# `docs/render_animations.jl` from the trajectory of the run, which is not stored in the repository,
# and saved to `docs/src/assets/`.
#
# ```@raw html
# <video src="../assets/vacancy_hop_md.mp4" controls loop muted playsinline width="720"></video>
# ```
