# # Canonical sampling with the Bussi thermostat
#
# 500 Cu atoms with MACE-MPA-0 and velocity Verlet, coupled to the stochastic velocity-rescaling
# thermostat of Bussi, Donadio and Parrinello at 1200 K for 2 ps. In the canonical ensemble the
# kinetic energy `K` of the `N_f = 3N − 3` degrees of freedom is Gamma-distributed with shape
# `N_f/2` and scale `k_B T`, whatever the potential, so the kinetic temperature `2K/(N_f k_B)` has
# mean `T` and variance `2T²/N_f`. The thermostat is meant to reproduce that distribution; this run
# checks it on the real potential.
#
# This page does not run the dynamics. It reads `bench/results/md_thermostat_check.json`, which
# holds the kinetic temperature of every step together with the commit, the date and the settings,
# and plots it. The first half of the run is discarded and the second half (1 to 2 ps) is analyzed.
# The crystal starts at the lattice sites with 1200 K velocities, so it equilibrates while the
# thermostat holds the kinetic temperature.

using CairoMakie, JSON, Printf, PureAtomistic, Statistics
const RESULTS = joinpath(pkgdir(PureAtomistic), "bench", "results")
d = JSON.parsefile(joinpath(RESULTS, "md_thermostat_check.json"))
s, th, ob = d["settings"], d["theory"], d["observed"]
@printf(
    "commit %s, %s, Julia %s, BLAS %s on %d thread, %d atoms, dt = %g fs, tau = %g fs, seed %d\n",
    s["commit"], s["date"], s["julia"], s["blas"], s["blas_threads"], s["natoms"], s["dt_fs"], s["tau_fs"], s["seed"]
)
t, T = Float64.(d["t_fs"]), Float64.(d["temperature_K"])
half = t .> 1000
x = T[half]
nothing #hide

# ## Mean and variance against theory
#
# Consecutive samples (2 fs apart) are strongly correlated, so they are not independent. The error
# of the mean comes from the integrated autocorrelation time `τ` of the analyzed series, with
# Sokal's automatic window (`M ≥ 6τ`): `SEM = σ √(2τ/N)`, with `N_eff = N/(2τ)` independent
# samples. The error of the skewness is `√(6/N_eff)`, the large-sample value for independent
# Gaussian samples. `z` is the deviation from theory in units of that error. The variance error is a
# delete-one-block jackknife over 5 blocks of 100 samples (200 fs each); the blocking table below
# shows that this block length is too short for the mean, and the variance error was not rechecked.

ac = ob["autocorrelation"]
@printf("τ = %.1f samples (window %d), N_eff = %.0f of %d samples\n", ac["tau_samples"], ac["window_samples"], ac["n_eff"], ac["n_samples"])
@printf("%-10s %14s %18s %10s\n", "", "observed", "theory", "z")
@printf("%-10s %9.2f ± %-5.2f %18.2f %10.2f\n", "mean (K)", ob["mean_K"], ac["mean_error_K"], th["mean_K"], ac["mean_z"])
@printf("%-10s %9.0f ± %-5.0f %18.0f %10.2f\n", "var (K²)", ob["variance_K2"], ob["variance_error_K2"], th["variance_K2"], ob["variance_z"])
@printf("%-10s %9.2f %24.2f\n", "std (K)", ob["std_K"], th["std_K"])
@printf("%-10s %9.3f ± %-5.3f %16.3f %10.2f\n", "skewness", ob["skewness"], ac["skewness_error"], th["skewness"], ac["skewness_z"])

# The mean agrees with 1200 K within about one standard error. The window constant `c` of the
# automatic window changes the error: `c = 4, 5, 6, 8` give the standard error and `z` below.

for r in ac["window_sensitivity"]
    @printf("c = %d: τ = %.1f samples, SEM %.2f K, z = %.2f\n", r["c"], r["tau_samples"], r["mean_error_K"], r["mean_z"])
end

# Blocking analysis of the mean: the standard error of the mean against block length. A plateau means
# the blocks are longer than the correlation time of the temperature. There is none here: the error
# is still rising at 50 samples (10 blocks), so the 5 blocks of 100 samples used for the first
# estimate underestimate the error. The autocorrelation estimate above is the one reported; the
# 5-block value is printed for comparison.

@printf("5-block estimate: SEM %.2f K, z = %.2f\n", ob["mean_error_K"], ob["mean_z"])

for b in ob["blocking_sem_K"]
    @printf("block of %3d samples (%4d fs): SEM %.3f K\n", b["block_samples"], b["block_samples"] * s["dt_fs"], b["sem_K"])
end

# The histogram is of the analyzed samples; the curve is the Gamma density of the kinetic
# temperature (normalized numerically on the plotted range).

dof = s["dof"]
Tset = th["mean_K"]
grid = range(Tset - 5 * th["std_K"], Tset + 5 * th["std_K"]; length = 400)
logpdf = @. (dof / 2 - 1) * log(grid) - grid * dof / (2 * Tset)
pdf = exp.(logpdf .- maximum(logpdf))
pdf ./= sum(pdf) * step(grid)
fig = Figure(; size = (900, 380), fontsize = 13)
ax1 = Axis(fig[1, 1]; xlabel = "time (ps)", ylabel = "kinetic temperature (K)")
lines!(ax1, t ./ 1000, T; linewidth = 1)
hlines!(ax1, [Tset]; color = :crimson)
vlines!(ax1, [1.0]; color = :gray, linestyle = :dash)
ax2 = Axis(fig[1, 2]; xlabel = "kinetic temperature (K)", ylabel = "density")
hist!(ax2, x; normalization = :pdf, bins = 20, color = (:steelblue, 0.6), label = "$(length(x)) samples, 1 to 2 ps")
lines!(ax2, grid, pdf; color = :crimson, linewidth = 2, label = "canonical")
axislegend(ax2; position = :lt)
save("thermostat_distribution.png", fig)
nothing #hide

# ![](thermostat_distribution.png)

# ## Animation
#
# The atoms of the run colored by their distance from the starting lattice site (in Å), and the kinetic temperature up to the time shown, with
# the target and one canonical standard deviation. The animation is rendered separately by
# `docs/render_animations.jl` from the trajectory of the run, which is not stored in the repository,
# and saved to `docs/src/assets/`.
#
# ```@raw html
# <video src="../assets/thermostat_check.mp4" controls loop muted playsinline width="720"></video>
# ```
