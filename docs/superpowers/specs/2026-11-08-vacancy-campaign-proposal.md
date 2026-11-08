# Vacancy-diffusion campaign: proposal and cost

Date: 2026-11-08. Status: proposal; nothing in it runs before the user approves it.

Goal: the migration energy `E_m` and prefactor `Γ₀` of a vacancy in Cu with MACE-MPA-0, from an
Arrhenius fit of hop rates at four temperatures, and `D_v(T)` at each, compared with published
values (design spec 2026-10-30, section 7, items 4 and 5).

## Inputs (measured)

| quantity | value | source |
|---|---|---|
| hop rate, 255 atoms, 1300 K, fixed volume at the 0 K lattice constant 3.6404 Å | 56 hops / 440 ps (Metal), 64 / 440 ps (CPU), both Float32; pooled 120 / 880 ps = 0.136 ± 0.012 ps⁻¹ | `md_vacancy_hop_f32{,_metal}.json` |
| Poisson check of those hops (11 bins of 40 ps) | χ² = 8.0 (Metal) and 8.5 (CPU) on 10 degrees of freedom | hop times in the same records |
| zero-pressure volume at 1300 K (Berendsen, 500 atoms) | 13.05 Å³/atom, a = 3.738 Å (2.7 % above 3.6404 Å); the 10 GPa run gives a = 3.627 Å, so the fixed-volume runs sit near 9 GPa (linear interpolation in a) | `gpu_barostat_regrow_metal.json` |
| Metal step, 255 atoms, Bussi, one process | 45.9–46.4 ms (NVT), 45.4–46.0 ms (NVT + Berendsen); 47.4 ms over the whole 440 ps run, frames written | this proposal (below); vacancy record |
| Metal step, 255 atoms, two processes at once | 66.8–68.6 ms each, so 33.4–34.3 ms per step of throughput (1.36× one process) | this proposal (below) |
| Metal step, 500 / 10,976 atoms | 70.0 ms / 1,354 ms | `gpu_compare_md_metal.json` |
| threaded CPU step, 255 atoms (6 threads, PureBLAS) | 61 ms (mean of a 440 ps run) | `md_vacancy_hop_f32.json` |
| barrier from our own MPA-0 data | no relaxed barrier exists; the unrelaxed straight-line path peaks at 1.13 eV, an upper bound | PureMLIP vacancy example page |

New measurement (CPU lock held, 8.3 min in all, swap unchanged): the 255-atom vacancy cell on
Metal, Float32, skin 0, state copied to the host and the tracker updated every 10 steps, three
500-step blocks per mode after 200 warm-up steps, in two fresh processes; then two processes
running NVT at once, four blocks each. The script is outside the repository.

**Barrier estimate.** Our data bound `E_m` from above (1.13 eV) only; a nudged-elastic-band
calculation would be a separate small task. The rates below use published Cu values: 0.71 eV
(simulation, Sugio, Sato and Shimomura 1996), 0.77 eV (formation enthalpy combined with
self-diffusion, J. Phys. F 1979), 0.88 ± 0.13 eV (equilibrium vacancy concentrations). The
estimate uses 0.70 eV with the range 0.6 to 0.9 eV. The commonly quoted compilation value
0.70 ± 0.02 eV (Ehrhart, Landolt-Börnstein III/25) has not been checked against the source. The
analysis page cites the checked values.

## Physics

- **Cell.** 4 × 4 × 4 fcc cells, 255 atoms, one vacancy. The cost per hop is proportional to the
  step time (the hop rate of one vacancy does not depend on N), and 255 atoms is the cheapest
  admissible size: 3 × 3 × 3 has an edge of about 11.2 Å, below MPA-0's receptive field of
  2 × 6 Å = 12 Å (two interactions, `r_max` 6 Å, read from the checkpoint). At 4 × 4 × 4 the edge
  is 14.6 to 15.0 Å over 900 to 1200 K, so the hopping atom is at least 12 Å from the image
  vacancy. That bounds the direct interaction only, not the elastic one. **Finite-size check**:
  5 × 5 × 5 (499 atoms) at 1200 K, 100 hops. At 1200 K a barrier shift δ changes the rate by
  e^(−δ/kT), with kT = 0.103 eV. The rate ratio has σ ≈ 12 %, so |δ| ≳ 0.02 eV is detected at 2σ.
- **Precision and device.** Float32 on Metal. In the validation runs on Metal, the hop count
  matched the Float32 CPU run (p = 0.52) and NVE and NVT matched too.
- **Temperatures.** 900, 980, 1080 and 1200 K, equally spaced in 1/T. Widening the range is
  cheaper than adding hops: 1000–1200 K with 200 hops per T gives σ(E_m) = 0.049 eV for 104 h,
  and this set gives 0.036 eV for about the same cost. 1200 K is the top: the experimental
  melting point of Cu is 1358 K, and MPA-0's is not known.
- **Per run.** Maxwell-Boltzmann velocities at T. Then 20 ps of Bussi (τ 100 fs) plus Berendsen
  (P₀ 0, τ_P 500 fs, β_T 1/140 GPa⁻¹). The volume settles within about 3 ps in the records, and
  the mean over the last 10 ps sets the cell. Then 2 ps of NVT to equilibrate (discarded), then
  production NVT with Bussi (τ 100 fs) and dt 2 fs (the validated settings), tracker frames
  every 20 fs (window 5, dwell 3). A residual pressure of 0.1 GPa moves the rate by about
  PΔV/kT ≈ 1 % for a migration volume of 2 Å³, which is negligible. The fit gives the
  zero-pressure migration enthalpy, which is what experiments report.
- **Runs.** Two independent runs per T (different seeds), run at the same time on the GPU. The
  pair gives the between-run consistency test (exact conditional binomial, `hop_test`) and the
  1.36× throughput.

## Statistics

| T (K) | expected Γ (ps⁻¹) at 0.70 eV | target hops (2 runs) | ns per run | steps per run | wall time, Metal pair (h) |
|---|---|---|---|---|---|
| 1200 | 0.081 | 200 | 1.23 | 628,000 | 11.8 [11.7–12.0] |
| 1080 | 0.038 | 100 | 1.31 | 666,000 | 12.5 [12.4–12.7] |
| 980 | 0.0177 | 100 | 2.82 | 1,421,000 | 26.7 [26.4–27.1] |
| 900 | 0.0085 | 100 | 5.89 | 2,958,000 | 55.5 [54.9–56.4] |

Γ(T) = 0.136 ps⁻¹ · exp(−E_m/k · (1/T − 1/1300 K)). Steps include 11,000 steps of equilibration.
The wall time is the steps of one run times the per-process step time of the pair, 67.5
[66.8–68.6] ms.

- **Rates.** Pooled per T: Γ = n/t with σ = √n/t (`hop_rate`). The stopping rule does not
  change the Poisson likelihood, so runs are extended in whole segments until the target is
  reached. The bias of n/t is of order 1/n.
- **Checks.** Between the two runs: `hop_test` p ≥ 0.01. Overdispersion: χ² of the per-segment
  counts. The vacancy diffusion coefficient D_v = Γa(T)²/12 (jump length a/√2) is compared with
  `msd_diffusion`; a disagreement points at multiple jumps or recrossings.
- **Fit.** `arrhenius_fit(T, Γ, σ)` (weights n): σ(E_m) = 1/√(Σ nᵢ(xᵢ − x̄)²) with x = 1/kT. The
  fit also gives σ(ln Γ₀) = √(1/Σn + x̄²/Sxx) and cov(E_m, ln Γ₀) = x̄/Sxx; x̄ = 10.96 eV⁻¹. It
  reports χ² on 2 degrees of freedom and does not rescale the errors. The temperature error
  (Bussi mean, SE ≈ 1 K) is negligible.
- **Expected:** σ(E_m) = **0.036 eV** (5 % of 0.70); σ(ln Γ₀) = 0.39, that is Γ₀ within a factor
  of 1.48. The rate at the weighted center (about 1060 K) is known to 4.5 %. With the hop targets
  reached, σ(E_m) does not depend on E_m; only the cost does.

## Cost (hours of computing, steps × step time)

| E_m | total steps | Metal, 2 processes | Metal, 1 process | CPU, 6 threads | RTX 4070 / R9700 (estimate) |
|---|---|---|---|---|---|
| 0.6 eV | 8.4 M | 79 [78–80] | 107 [106–111] | 142 | 54–161 |
| **0.70 eV** | **11.3 M** | **106 [105–108]** | 145 [143–149] | 192 | 72–217 |
| 0.9 eV | 21.5 M | 201 [199–205] | 275 [271–283] | 364 | 137–412 |

Optional finite-size check: 617,000 steps at 70 ms, 12 h. The CPU figure is one mean, not a
range; the x86 CPU has not been measured. **The RTX 4070 and R9700 figures are not measured.**
They assume a 255-atom step between 0.5× and 1.5× the Metal step. At this size the step is
dominated by fixed per-step costs (the linear fit through 255 and 500 atoms leaves about 21 ms
of 46), which the hardware specifications do not predict. A 3-minute measurement like the one
above replaces the bracket on each machine. Running on the x86 machines needs the user's
approval each time.

**Segments.** Each run is cut into segments of 100 ps (50,000 steps, about 56 min for a pair).
At the end of each segment the positions, velocities, generator, tracker and series are stored
(the mechanism of the 440 ps runs, which reproduces a continuous run bit for bit). The CPU lock
is released between segments, and a run resumes from its last segment. That makes about 114
paired segments, 4.4 days of GPU time at 0.70 eV, plus waits for the lock. Order: 1200 → 1080 →
980 → 900 K. After 1200 K the measured zero-pressure rate replaces the fixed-volume estimate, and
the remaining cost is reported again. The rate at zero pressure differs by an unknown
exp(−PΔV_m/kT) at about 9 GPa.

## Risks

- **Few hops at low T.** If E_m = 0.9 eV, the planned lengths give 45 hops at 900 K. Extending
  to 100 roughly doubles that temperature, and the total rises from 106 to 201 h. Fallback: raise
  the low end to 925 K, where σ(E_m) = 0.040 eV and the cost falls by 16 %.
- **Divacancy, interstitial, melting.** With one vacancy, a divacancy needs a Frenkel pair, which
  costs several eV and does not form at these temperatures. Any frame whose occupancy is not one
  vacancy stops the run (`"pureatomistic: occupancy is not one vacancy at frame …"`), and the
  segment checkpoint is kept for inspection. Melting shows as that error and as a jump in the
  NPT volume. Mitigation: drop the top temperature to 1150 K.
- **Thermal expansion.** The volume is set per T from NPT. The 1300 K reference rate was taken at
  the 0 K volume (about 9 GPa), so the cost estimate carries that offset. The 1200 K run
  re-anchors it.
- **Float32.** Float32 against Float64 hop counts rest on 5 hops (p = 0.375). The Float32 Metal
  and CPU runs agree, but that does not test precision.

## Outputs

- **In the repository** (small JSON, `PureAtomistic/bench/results/vacancy_campaign/`): one record
  per run with the settings and provenance stamp, the NPT volume (mean, SE) and the chosen a(T),
  hop times and sites, mean T and pressure with SEs, per-segment wall times, regrowths, Γ,
  D_v and the MSD D. A summary record holds the per-T pooled rates, the test results, the fit
  (E_m, σ, Γ₀, σ(ln Γ₀), covariance, χ²) and the step times.
- **Outside the repository:** trajectories (every 100 fs, about 0.7 GB in all), per-frame
  vacancy tracks and segment checkpoints.
- **Analysis page** `PureAtomistic/docs/src/examples/vacancy_diffusion.md`. It reads the records
  and does not run MD. It has a D_v(T) table, `plot_arrhenius`, E_m and Γ₀ with errors, the
  checked published values, and the finite-size result. The 1300 K fixed-volume runs are kept
  apart as different conditions.
- **Code** (after approval): a campaign script, `PureAtomistic/bench/vacancy_campaign.jl`, built
  from the vacancy mode of `md_validation.jl`. It adds the temperature, the seed, the NPT stage
  and the paired runs.
