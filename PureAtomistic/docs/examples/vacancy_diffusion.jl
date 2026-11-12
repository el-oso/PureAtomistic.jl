# # Vacancy diffusion in Cu
#
# The migration energy `E_m` and prefactor `Γ₀` of one vacancy in Cu with MACE-MPA-0, from the hop
# rates at 900, 980, 1080 and 1200 K, and the diffusion coefficients at each temperature. The
# method was fixed before any run (`docs/superpowers/specs/2026-11-08-vacancy-campaign-proposal.md`).
#
# Each run holds one vacancy in 4 × 4 × 4 conventional fcc cells (255 atoms), in Float32 on the
# GPU (Metal) with a 2 fs time step. It starts with 20 ps of Bussi thermostat and Berendsen
# barostat at zero pressure; the cell is then fixed at the mean volume of the last 10 ps, 2 ps of
# constant-volume dynamics are discarded, and the production run follows at constant volume with
# the Bussi thermostat (τ 100 fs). A `VacancyTracker` reads a frame every 20 fs (window 5,
# dwell 3) and counts the confirmed hops; a trajectory frame is stored every 100 fs. There are two
# independent runs per temperature, run as two processes at once, and one 499-atom run
# (5 × 5 × 5 cells) at 1200 K checks the cell size.
#
# This page runs no dynamics. It reads `bench/results/vacancy_campaign_analysis.json`, which
# `bench/vacancy_analysis.jl` computes from the run records in `bench/results/vacancy_campaign/`,
# the NEB record `bench/results/neb_vacancy.json` and the per-run MSD records
# `bench/results/trajectory_diffusion_<run>.json` (computed by `bench/trajectory_diffusion.jl`
# from the stored trajectories, which are not in the repository).

using CairoMakie, JSON, Markdown, Printf, PureAtomistic
const RESULTS = joinpath(pkgdir(PureAtomistic), "bench", "results")
d = JSON.parsefile(joinpath(RESULTS, "vacancy_campaign_analysis.json"))
Ts = d["temperatures"]
fit = d["fit"]
neb = d["neb"]
@printf("%.1f GPU hours on Metal, completed %s, Julia %s\n", d["campaign_gpu_hours"], d["campaign_completed"], d["julia"])

# ## Hop rates
#
# Per temperature the two runs are pooled: `Γ = n/t` with the Poisson error `√n/t`. `T` is the mean
# kinetic temperature of the production runs and `a` the lattice constant set by the barostat
# stage. "Transients" counts the episodes in which the tracker saw no single vacancy for at most
# 500 fs (a jump in progress) and skipped those frames.

row(cols) = "| " * join(cols, " | ") * " |"
lines = [
    row(["T (K)", "mean T (K)", "a (Å)", "hops", "time (ps)", "Γ (1/ps)", "binomial p", "segment χ² / dof (p)", "transients"]),
    row(fill("---:", 9)),
]
for t in Ts
    push!(
        lines, row(
            [
                @sprintf("%.0f", t["nominal_K"]), @sprintf("%.1f", t["mean_temperature_K"]), @sprintf("%.4f", t["lattice_constant_A"]),
                string(t["hops"]), @sprintf("%.0f", t["duration_ps"]), @sprintf("%.4f ± %.4f", t["rate_per_ps"], t["rate_se_per_ps"]),
                @sprintf("%.2f", t["hop_test"]["p"]), @sprintf("%.1f / %d (%.3f)", t["segment_chi2"], t["segment_dof"], t["segment_dispersion_p"]),
                string(t["transient_episodes"]),
            ]
        )
    )
end
Markdown.parse(join(lines, "\n"))

# "binomial p" is the exact conditional binomial test of the two runs' counts given their total
# (`p ≥ 0.01` was the stated criterion; every temperature passes). The segment `χ²` tests the counts of the
# 100 ps segments of both runs against equal rates; its p-value is the fraction of multinomial
# draws of the same total with a `χ²` at least as large. The runs one by one:

lines = [
    row(["run", "hops", "time (ps)", "Γ (1/ps)", "segment hops", "χ² / dof (p)", "longest gap (ps)", "p of that gap", "transients"]),
    row(fill("---:", 9)),
]
for t in Ts, r in t["runs"]
    push!(
        lines, row(
            [
                r["run"], string(r["hops"]), @sprintf("%.0f", r["duration_ps"]), @sprintf("%.4f ± %.4f", r["rate_per_ps"], r["rate_se_per_ps"]),
                join(r["segment_hops"], " "), @sprintf("%.1f / %d (%.4f)", r["segment_chi2"], r["segment_dof"], r["segment_dispersion_p"]),
                @sprintf("%.1f (%.1f–%.1f)", r["longest_gap_ps"], r["longest_gap_from_ps"], r["longest_gap_to_ps"]),
                @sprintf("%.2g", r["longest_gap_p"]), string(r["transient_episodes"]),
            ]
        )
    )
end
Markdown.parse(join(lines, "\n"))

# The "longest gap" is the longest stretch without a confirmed hop, counting from the start and to
# the end of production; its p-value is the exact probability that `n` hop times spread uniformly
# over the run leave a gap at least that long.
#
# One run stands out. `T1080_dt2_s06` has no hop in its first 143 ps, against a mean waiting time
# of 12 ps at 1080 K; its segment counts give `p = 0.002` and the gap `p = 1.7 × 10⁻⁴`. Over the
# eight runs, the chance that some run shows a gap that unlikely is about eight times larger,
# `1.3 × 10⁻³`. In the stored trajectory frames (every 100 fs), assigning each atom to its nearest
# site, without the tracker's window, leaves the same one site empty from the start to 143.3 ps,
# the time of the tracker's first hop, with no excursion in any frame. The dwell is real; its cause
# is not known. The run is kept as it is; with the 1080 K rate taken from `T1080_dt2_s05` alone the fit
# below changes by 0.001 eV.

# ## Arrhenius fit
#
# `arrhenius_fit` fits `ln Γ = ln Γ₀ − E_m/(k_B T)` with weights `n` (the inverse variances of
# `ln Γ` for Poisson counts) and does not rescale the errors by `χ²`.

@printf("E_m = %.3f ± %.3f eV (target error %.3f eV)\n", fit["E_m_eV"], fit["σ_E_m_eV"], fit["target_σ_E_m_eV"])
@printf("Γ₀ = %.0f /ps, σ(ln Γ₀) = %.2f (a factor of %.2f)\n", fit["prefactor_per_ps"], fit["σ_ln_prefactor"], fit["prefactor_factor_1σ"])
@printf("cov(E_m, ln Γ₀) = %.4f eV\n", fit["cov_E_ln_prefactor_eV"])
@printf("χ² = %.3f on %d degrees of freedom (p = %.2f)\n", fit["chi2"], fit["dof"], fit["chi2_p"])
@printf("weighted center %.0f K, Γ there known to %.1f %%\n", fit["weighted_center_K"], 100 * fit["rate_at_center_relative_se"])
@printf("at the nominal temperatures: E_m = %.4f ± %.4f eV\n", fit["nominal_temperatures"]["E_m_eV"], fit["nominal_temperatures"]["σ_E_m_eV"])

# The error of `E_m` meets the proposal's expected 0.036 eV. The `χ²` is small for two degrees of
# freedom (`p = 0.95`), so the four rates lie closer to a line than their Poisson errors require;
# with four points that is not evidence against the errors.
#
# ## Comparison with the NEB barrier
#
# The climbing-image NEB (`bench/results/neb_vacancy.json`, 255 atoms, Float64, fixed cell) gives
# 0.704 eV at the 0 K lattice constant 3.6404 Å and 0.621 eV at 3.7263 Å, the zero-pressure
# lattice constant at 1200 K: the barrier falls as the lattice expands. The fit runs at zero
# pressure, so its lattice constant grows with `T`, from 3.7006 Å at 900 K to 3.7277 Å at 1200 K.
# The NEB barrier is taken as linear in `a` through its two points (the linearity between them is
# assumed, not measured):

@printf("dE_m/da = %.3f eV/Å, da/dT = %.2e Å/K over the campaign\n", neb["dE_m_da_eV_per_A"], neb["da_dT_A_per_K"])
for t in Ts
    k = string(Int(t["nominal_K"]))
    @printf("%5s K: a = %.4f Å, NEB E_m(a) = %.3f eV\n", k, t["lattice_constant_A"], neb["E_m_at_campaign_a_eV"][k])
end

# The barrier at the lattice constant of each temperature is about 0.63 eV, 0.11 eV (3.2 σ) below
# the fit. That comparison leaves out the expansion. With `x = 1/(k_B T)`, a rate
# `Γ = ν exp(−x E_m(a(x)))` has the Arrhenius slope
#
# ```
# E_eff = −d ln Γ / dx = E_m(a) + x (dE_m/da)(da/dx) = E_m(a) − T (dE_m/da)(da/dT).
# ```
#
# With `dE_m/da < 0` and `da/dT > 0`, the second term is positive: the barrier is lower at the
# temperatures where the rate is higher, so the rate rises faster with `T` than the barrier at
# any one lattice constant implies. For `a` linear in `T`, `E_eff` equals the barrier at the lattice
# constant extrapolated along `a(T)` to `T = 0`. The same term moves the intercept: the fitted
# `Γ₀` is `ν exp(T (dE_m/da)(da/dT)/(k_B T))`. The script fits the rates that the NEB line and
# the measured `a(T)` predict with a constant `ν`, with the weights of the data, and finds:

@printf("E_eff from NEB and expansion = %.3f eV (a extrapolated to 0 K: %.4f Å)\n", neb["E_eff_expected_eV"], neb["a_extrapolated_to_0K_A"])
@printf("fit − E_eff = %.3f eV = %.2f σ\n", fit["E_m_eV"] - neb["E_eff_expected_eV"], neb["z_fit_minus_E_eff"])
@printf("ln Γ₀ shift = %.2f, so ν = %.0f /ps, %.2f THz per jump direction (12 neighbors)\n", neb["ln_prefactor_shift"], neb["attempt_rate_implied_per_ps"], neb["attempt_frequency_per_jump_THz"])

# The fit agrees with the expansion-corrected NEB barrier within 0.5 σ. This comparison sets the
# barrier's own temperature dependence at fixed volume (vibrational entropy, anharmonicity) and the
# Float32 to Float64 difference to zero; at the error of the fit, 0.034 eV, the data do not
# resolve them. The 7.6 THz per jump direction is of the size of a phonon frequency of Cu; it is
# derived from the fit, not measured on its own.
#
# The 1300 K runs at the 0 K lattice constant (`bench/results/md_vacancy_hop_f32{,_metal}.json`,
# about 9 GPa) are not part of the fit. Their rate against the fit's rate at 1300 K, with the same
# prefactor, gives the barrier increase from compressing the lattice from its zero-pressure value
# at 1300 K (3.7361 Å) to 3.6404 Å; the NEB line gives the same quantity directly:

v = d["fixed_volume_1300K"]
@printf("1300 K at %.4f Å: %d hops in %.0f ps, Γ = %.4f /ps; the fit gives %.4f /ps\n", v["lattice_constant_A"], v["hops"], v["duration_ps"], v["rate_per_ps"], v["rate_from_fit_per_ps"])
@printf("barrier increase from the rates %.3f ± %.3f eV, from the NEB line %.3f eV\n", v["barrier_increase_from_rates_eV"], v["barrier_increase_from_rates_se_eV"], v["barrier_increase_from_neb_line_eV"])

# They agree within the error, which leaves out the NEB line's own uncertainty and any pressure
# dependence of the prefactor.

# ## Cell size
#
# The 499-atom run at 1200 K against the two 255-atom runs: the exact conditional binomial test of
# the counts, the exact (Clopper-Pearson) 95 % interval of the rate ratio, and the shift of the
# barrier it would mean, `δ = −k_B T ln ρ`.

s = d["size_check"]
@printf("499 atoms: %d hops in %.0f ps, Γ = %.4f ± %.4f /ps, a = %.4f Å, %d transients\n", s["hops"], s["duration_ps"], s["rate_per_ps"], s["rate_se_per_ps"], s["lattice_constant_A"], s["transient_episodes"])
@printf("255 atoms: %d hops in %.0f ps\n", s["baseline_hops"], s["baseline_duration_ps"])
@printf("ratio %.3f, 95 %% interval [%.3f, %.3f], binomial p = %.2f\n", s["rate_ratio"], s["rate_ratio_ci95"]..., s["binomial_p"])
@printf("δ = %.3f eV, 95 %% interval [%.3f, %.3f] eV\n", s["barrier_shift_eV"], s["barrier_shift_ci95_eV"]...)
@printf("δ expected from the 0.0017 Å larger lattice constant alone: %.4f eV\n", s["barrier_shift_from_lattice_constant_eV"])

# The two sizes agree. A barrier shift beyond about ±0.03 eV from the cell size is excluded at
# 95 %, which is within the fit's error of `E_m`.

# ## Diffusion coefficients
#
# Two estimates of each diffusion coefficient. For the atoms, the hop rate predicts
# `D = f Γ λ²/(6N) = f Γ a²/(12N)` with the jump length `λ = a/√2` and the tracer correlation
# factor `f` of the periodic cell, measured by a lattice walk of one vacancy in the same cell
# (see [Trajectory analysis](../analysis.md)); the MSD of all atoms over the stored trajectory gives
# `D` directly (one block per 100 ps segment, lags up to 50 ps, fit over 10 to 50 ps). For the
# vacancy, `D_v = Γ a²/12` against the MSD of the tracked vacancy position. The MSD values are the
# mean of the two runs.

lines = [
    row(["T (K)", "f", "D atoms, rate (Å²/ps)", "D atoms, MSD (Å²/ps)", "z", "D_v, rate (Å²/ps)", "D_v, MSD (Å²/ps)", "z"]),
    row(fill("---:", 8)),
]
for x in d["diffusion"]
    push!(
        lines, row(
            [
                @sprintf("%.0f", x["nominal_K"]), @sprintf("%.4f", x["correlation_factor"]),
                @sprintf("%.2e ± %.1e", x["D_atoms_from_rate_A2_per_ps"], x["D_atoms_from_rate_se_A2_per_ps"]),
                @sprintf("%.2e ± %.1e", x["D_atoms_msd_A2_per_ps"], x["D_atoms_msd_se_A2_per_ps"]), @sprintf("%.2f", x["z_atoms"]),
                @sprintf("%.4f ± %.4f", x["D_vacancy_from_rate_A2_per_ps"], x["D_vacancy_from_rate_se_A2_per_ps"]),
                @sprintf("%.4f ± %.4f", x["D_vacancy_msd_A2_per_ps"], x["D_vacancy_msd_se_A2_per_ps"]), @sprintf("%.2f", x["z_vacancy"]),
            ]
        )
    )
end
Markdown.parse(join(lines, "\n"))

# Every pair agrees within one combined error, so the tracker's hop count accounts for the
# diffusion of the atoms and of the vacancy, with no sign of missed or doubled hops at the level of
# the errors (10 to 20 %). The MSD errors come from the spread of the block estimates and are
# themselves uncertain. 1 Å²/ps is 10⁻⁴ cm²/s. These are diffusion coefficients at one vacancy in
# 255 sites; the self-diffusion coefficient of Cu also needs the equilibrium vacancy
# concentration, which the campaign does not measure.

# ## Figure
#
# Left: `ln Γ` against `1/(k_B T)` with the fit, the slope expected from the NEB barrier with the
# expansion (drawn through the fit at its weighted center), the slope of the NEB barrier at the
# lattice constant of that center, and the two points not in the fit. Right: the atomic `D` from
# the MSD against the prediction from the hop rate.

kB = 8.617333262e-5
x = [1 / (kB * t["mean_temperature_K"]) for t in Ts]
y = [log(t["rate_per_ps"]) for t in Ts]
e = [t["rate_se_per_ps"] / t["rate_per_ps"] for t in Ts]
xc = 1 / (kB * fit["weighted_center_K"])
yc = log(fit["prefactor_per_ps"]) - fit["E_m_eV"] * xc
xr = [minimum(x) - 0.3, maximum(x) + 0.3]
fig = Figure(; size = (1000, 430), fontsize = 13)
ax = Axis(
    fig[1, 1]; xlabel = "1/(k_B T) (1/eV)", ylabel = "ln(Γ / ps⁻¹)",
    title = @sprintf("E_m = %.3f ± %.3f eV, Γ₀ = %.0f /ps", fit["E_m_eV"], fit["σ_E_m_eV"], fit["prefactor_per_ps"])
)
lines!(ax, xr, log(fit["prefactor_per_ps"]) .- fit["E_m_eV"] .* xr; color = :black, label = "fit")
lines!(ax, xr, yc .- neb["E_eff_expected_eV"] .* (xr .- xc); color = :seagreen, linestyle = :dash, label = @sprintf("NEB with expansion, %.3f eV", neb["E_eff_expected_eV"]))
lines!(ax, xr, yc .- neb["E_m_at_weighted_center_eV"] .* (xr .- xc); color = :orange, linestyle = :dot, label = @sprintf("NEB at a(%.0f K), %.3f eV", fit["weighted_center_K"], neb["E_m_at_weighted_center_eV"]))
errorbars!(ax, x, y, e; color = :black, whiskerwidth = 6)
scatter!(ax, x, y; color = :black, label = "255 atoms, zero pressure")
xs = 1 / (kB * s["mean_temperature_K"])
errorbars!(ax, [xs], [log(s["rate_per_ps"])], [1 / sqrt(s["hops"])]; color = :royalblue, whiskerwidth = 6)
scatter!(ax, [xs], [log(s["rate_per_ps"])]; color = :royalblue, marker = :diamond, label = "499 atoms (not fitted)")
xv = 1 / (kB * v["mean_temperature_K"])
errorbars!(ax, [xv], [log(v["rate_per_ps"])], [1 / sqrt(v["hops"])]; color = :firebrick, whiskerwidth = 6)
scatter!(ax, [xv], [log(v["rate_per_ps"])]; color = :firebrick, marker = :utriangle, label = "1300 K at a(0 K) (not fitted)")
axislegend(ax; position = :lb, framevisible = false)
ax2 = Axis(fig[1, 2]; xlabel = "1/(k_B T) (1/eV)", ylabel = "atomic D (Å²/ps)", yscale = log10, title = "D of the atoms, 255 atoms")
Dr = [z["D_atoms_from_rate_A2_per_ps"] for z in d["diffusion"]]
Drs = [z["D_atoms_from_rate_se_A2_per_ps"] for z in d["diffusion"]]
Dm = [z["D_atoms_msd_A2_per_ps"] for z in d["diffusion"]]
Dms = [z["D_atoms_msd_se_A2_per_ps"] for z in d["diffusion"]]
errorbars!(ax2, x .- 0.02, Dr, Drs; color = :black, whiskerwidth = 6)
scatter!(ax2, x .- 0.02, Dr; color = :black, label = "f Γ a²/(12N)")
errorbars!(ax2, x .+ 0.02, Dm, Dms; color = :seagreen, whiskerwidth = 6)
scatter!(ax2, x .+ 0.02, Dm; color = :seagreen, marker = :rect, label = "MSD of the atoms")
axislegend(ax2; position = :lb, framevisible = false)
save("vacancy_arrhenius.png", fig)
nothing #hide

# ![](vacancy_arrhenius.png)

# ## Published values
#
# Two experimental values were checked against the abstracts of their papers:
#
# - Simmons and Balluffi, Phys. Rev. 129, 1533 (1963): equilibrium vacancy concentrations from
#   length and lattice-parameter expansion; the results imply a monovacancy migration energy of
#   0.88 ± 0.13 eV, with an assumed formation entropy.
# - Berger, Ockers and Siegel, J. Phys. F 9, 1023 (1979): the formation enthalpy 1.30 ± 0.05 eV from
#   resistometry of quenched samples, which with the self-diffusion activation enthalpy gives a
#   migration enthalpy of 0.77 eV; the activation enthalpy of post-quench annealing in the same
#   work is 0.76 ± 0.04 eV.
#
# The compilation value 0.70 ± 0.02 eV often quoted from Ehrhart (Landolt-Börnstein III/25) has
# not been checked against the source. The fit's 0.742 ± 0.034 eV lies within the range of the
# checked values and within one error of 0.76 ± 0.04 eV. The experimental values are enthalpies at
# zero pressure from lower temperatures (quenching, annealing) or combine several measurements; the
# fit is the zero-pressure Arrhenius slope over 900 to 1200 K of one interatomic potential.
#
# ## Limits
#
# - Float32 on the GPU throughout; the NEB barriers are Float64. The Float32 and Float64 hop counts
#   have been compared on 5 hops only.
# - The barrier's own temperature dependence at fixed volume is not separated from the expansion
#   term; at the fit's error it is consistent with zero.
# - The tracker counts confirmed hops. A jump undone within its dwell time counts as none; the
#   agreement of the diffusion coefficients bounds such misses only at the 10 to 20 % level.
# - The 4 fs run at 1200 K (`T1200_dt4_s02`) is kept as data and not used: 263 hops in 1296 ps,
#   rate ratio to 2 fs 1.015 (95 % interval 0.82 to 1.25), stopped by a 520 fs saddle transient
#   of a concerted two-atom jump.
