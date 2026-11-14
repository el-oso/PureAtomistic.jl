# Trajectory analysis

`PureAtomistic.Analysis` computes structural and transport quantities from a run: the radial
distribution function `g(r)` and its running integral, coordination numbers, the mean-square
displacement and the diffusion coefficient, and the root-mean-square deviation from a reference.

Each function reads `frames`, any collection with `eachindex` and `getindex` whose elements have
`positions` (3 × n, Å), `cell` (3 × 3, lattice vectors in columns) and `species`: a vector of
`System`s, a vector of `IO.Frame`s, or a `TrajectoryReader`, which reads one frame per index.
Cells may be triclinic. Positions, cells and species are read through their own axes, so views
and `OffsetArray`s are accepted and give the same results as plain arrays. The arrays are read
element by element on the host.

```julia
using PureAtomistic, PureAtomistic.Analysis, PureAtomistic.IO
traj = TrajectoryReader("run.patraj")
(; r, g, n) = rdf(traj, 6.0, 120)                     # g(r) to 6 Å in 0.05 Å bins
z = coordination_numbers(traj[1], 3.0)                # neighbors within 3 Å, per atom
(; lag_fs, msd_A2) = msd(traj; max_lag = 500, nblocks = 5)
(; D, σ_D) = diffusion_coefficient(lag_fs, msd_A2; fit_window = (10_000, 50_000))  # Å²/fs
```

## Periodic images

A pair search to a distance `r_max` uses the minimum image when `r_max` is at most half the
shortest cell height: every pair closer than that is found once, at its true distance, in any
cell shape. A longer `r_max` visits every periodic image within it, including an atom's own
images, as in the infinite periodic crystal. `rdf` normalizes the pair count of each shell by
the count of an ideal gas of the same composition. Within the minimum-image range an atom never
pairs with itself, so the ideal count per unit volume is `(N_A N_B - N_AB) / V`, with `N_AB` the
atoms selected by both members of the pair, and an ideal gas gives `g = 1` on average for every
selection, `nothing` included. Beyond it the count is `N_A N_B / V`, because the periodic images
of the `B` atoms, an atom's own images included, fill space at the density `N_B / V`; `g` then
tends to 1 at large `r` in any cell, and a species with a single atom works.

The mean-square displacement unwraps the positions frame to frame: the displacement between
consecutive frames is taken under the minimum image. That is right only if every atom moves less
than half the shortest cell height between frames, and choosing a frame interval short enough is
the caller's responsibility. `unwrap!` throws for a displacement in the ambiguous range, but a
longer move maps onto a shorter image and passes unnoticed: in a cubic cell of side `L`, a move
of `0.7 L` along an axis reads as `-0.3 L`. The frames are split into
blocks, and `msd` returns one column per block; `diffusion_coefficient` fits the Einstein
relation `⟨|Δr|²⟩ = 6 D τ + c` to each column over the fit window and returns the mean and its
standard error over the blocks. With few blocks that error is itself uncertain, by a fraction of
about `1/√(2(nblocks - 1))`.

`rmsd` takes positions as given, with no periodic images, and with `align = true` removes the
best translation and proper rotation (Kabsch's method) before measuring the deviation.

`rdf!`, `msd!`, `rmsd!` and `reference_sites!` are the in-place forms: they write into arrays the
caller provides and allocate nothing for frames already in memory. `rdf`, `msd`, `rmsd` and
`reference_sites` allocate those arrays and call them.

## Example: atomic diffusion by a vacancy at 1200 K

The first 1200 K production run of the vacancy-diffusion campaign: one vacancy in 4 × 4 × 4
conventional fcc cells (255 Cu atoms) with MACE-MPA-0 in Float32, 700 ps of constant-volume
dynamics with the Bussi thermostat, a trajectory frame every 100 fs. Its `VacancyTracker` counted
140 hops. The MSD of all atoms has one block per 100-ps segment (7 blocks), lags up to 50 ps,
and the fit runs over lags of 10 to 50 ps.

The hop count predicts the atomic diffusion coefficient. Each vacancy jump moves one atom by a
nearest-neighbor vector of length `λ = a/√2`, so after `n = Γt` jumps the squared displacements
of the `N` atoms sum to `nλ²` plus the cross terms of successive jumps of the same atom. Those
are negative, because an atom that has just exchanged with the vacancy is likely to exchange
back; their long-time effect is the tracer correlation factor `f`. With
`Σᵢ |Δrᵢ|² = 6 N D t`,

```
D = f Γ λ² / (6 N) = f Γ a² / (12 N).
```

For the infinite fcc lattice `f = 0.78145` (Compaan and Haven, 1956; the value is not taken from
the source). The record measures `f` for this periodic cell by a lattice random walk of one
vacancy among 255 atoms in the same 4 × 4 × 4 lattice, with the center-of-mass displacement
removed as in the MD run, whose total momentum is zero; it reproduces the infinite-lattice value
within its error.

A known limit: `Γ` counts the hops that the vacancy tracker confirms, and a jump that is undone
within the tracker's dwell time counts as no hop. Such a pair moves no atom and leaves `D`
unchanged, but the lattice walk's `f` includes it, so the prediction is low by the fraction of
hops missed this way, which is not measured.

This page does not read the trajectory. `bench/trajectory_diffusion.jl` computes the record
`bench/results/trajectory_diffusion_T1200_dt2_s01.json` from the run's trajectory and its
campaign record, and the code below reads it.

```@example diffusion
using CairoMakie, JSON, Printf, PureAtomistic
d = JSON.parsefile(joinpath(pkgdir(PureAtomistic), "bench", "results", "trajectory_diffusion_T1200_dt2_s01.json"))
@printf("%s: %d atoms, %.0f K, %d frames %g fs apart, %d hops\n", d["run"], d["natoms"], d["temperature_K"], d["frames"], d["frame_interval_fs"], d["hops"])
@printf("hop rate Γ = %.4f ± %.4f /ps, a = %.4f Å, λ = %.4f Å\n", d["rate_per_ps"], d["rate_se_per_ps"], d["lattice_constant_A"], d["jump_length_A"])
@printf("f in this cell = %.4f ± %.4f (infinite lattice %.5f)\n", d["correlation_factor_cell"], d["correlation_factor_cell_se"], d["correlation_factor_infinite"])
@printf("D atoms, from the MSD:      %.3e ± %.3e Å²/ps\n", d["D_atoms_A2_per_ps"], d["D_atoms_se_A2_per_ps"])
@printf("D atoms, from the hop rate: %.3e ± %.3e Å²/ps\n", d["D_atoms_predicted_A2_per_ps"], d["D_atoms_predicted_se_A2_per_ps"])
@printf("difference / combined error: %.2f\n", d["z_measured_minus_predicted"])
@printf("largest center-of-mass shift over the run: %.3f Å\n", d["com_max_shift_A"])
```

The two estimates agree within their errors. The MSD error comes from the spread of seven block
estimates, so it is itself uncertain by about 30 % (the standard deviation of 7 values); the
prediction's error is the Poisson error of 140 hops. Units: 1 Å²/ps is 10⁻⁴ cm²/s.

```@example diffusion
lag = Float64.(d["msd_lag_ps"])
m = Float64.(d["msd_mean_A2"])
w = (d["fit_window_fs"][1] / 1000) .<= lag .<= (d["fit_window_fs"][2] / 1000)
slope = 6 * d["D_atoms_A2_per_ps"]
c = sum(m[w] .- slope .* lag[w]) / count(w)
pred = 6 * d["D_atoms_predicted_A2_per_ps"]
cp = sum(m[w] .- pred .* lag[w]) / count(w)
fig = Figure(; size = (760, 420), fontsize = 13)
ax = Axis(fig[1, 1]; xlabel = "lag (ps)", ylabel = "mean-square displacement (Å²)", title = "255 Cu atoms with one vacancy, 1200 K")
lines!(ax, lag, m; label = "MSD, mean of 7 blocks")
lines!(ax, lag[w], c .+ slope .* lag[w]; linestyle = :dash, label = "6Dτ + c, D from the MSD fit")
lines!(ax, lag[w], cp .+ pred .* lag[w]; linestyle = :dot, label = "6Dτ + c, D from the hop rate")
axislegend(ax; position = :rb)
fig
```

The rise at the shortest lags is the vibrational amplitude of the atoms, which the intercept `c`
absorbs; the slope beyond it is the diffusion carried by the vacancy.

## Reference

```@docs
PureAtomistic.Analysis.rdf
PureAtomistic.Analysis.rdf!
PureAtomistic.Analysis.rdf_histogram!
PureAtomistic.Analysis.coordination_numbers
PureAtomistic.Analysis.coordination_numbers!
PureAtomistic.Analysis.msd
PureAtomistic.Analysis.msd!
PureAtomistic.Analysis.unwrap!
PureAtomistic.Analysis.diffusion_coefficient
PureAtomistic.Analysis.rmsd
PureAtomistic.Analysis.rmsd!
```
