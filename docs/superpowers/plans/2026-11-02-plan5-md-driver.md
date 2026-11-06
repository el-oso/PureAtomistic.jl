# PureAtomistic MD driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the second package of the monorepo, `PureAtomistic/`, with an MD driver (velocity Verlet, Langevin BAOAB, Bussi thermostat, Berendsen barostat) on any potential that implements its contract, MPA-0 through PureMLIP as the main one, extxyz and binary trajectory I/O, vacancy tracking and diffusion analysis, a Makie extension, and a trimmed `md_run` binary.

**Architecture:** A small trimmable core (`[deps]`: PureMLIP, StrictMode, TypeContracts, PrecompileTools, stdlibs) with submodules Potentials, Systems, Integrators, Thermostats, Barostats, IO, Analysis; Makie, Optim and AtomsBase/AtomsCalculators only as package extensions. The neighbor list and workspace are PureMLIP's, rebuilt each step (1.5 ms against ~0.7–1.1 s per evaluation at 500 atoms). Every physics piece is validated on cheap potentials (harmonic, Lennard-Jones) and on short MPA-0 runs; the long vacancy-diffusion campaign is deferred to GPU support.

**Tech Stack:** Julia 1.12/1.13, PureMLIP (path source), TypeContracts 0.14, StrictMode 0.5.8 / StrictModeTest 0.5.7, TestItems, TrimCheck, JuliaC, Makie (extension; CairoMakie in tests, GLMakie in docs), Optim 2 (extension).

**Spec:** `docs/superpowers/specs/2026-10-30-pureatomistic-design.md` (sections 2, 4, 6.2, 6.3, 7, 11, 13, 14). Measurements and literature behind this plan: `plan5-recon.md` (outside the repo).

## Global Constraints

- PureAtomistic `[deps]`: PureMLIP (`[sources] PureMLIP = {path = "../PureMLIP"}`), StrictMode 0.5.8, TypeContracts 0.14, PrecompileTools, PureDAQP, LinearAlgebra, Random. No `_jll`/artifact in the core closure. Makie, AtomsBase, AtomsCalculators only in `[weakdeps]`/`[extensions]`. Every `[compat]` bounded; `julia = "1.12"`.
- AGENTS.md rules apply to the new package unchanged (generic indexing, `@inbounds` only with measured gain and proof, `lazy"…"` messages with a `pureatomistic:` prefix, concrete fields, fail fast, `@test_throws "message"`, present-tense comments, tolerance policy, coverage row and mutation entry for every new src file).
- Units: Å, eV, fs, amu, K; `k_B` and the amu→eV·fs²/Å² factor are named constants with their CODATA values in a comment.
- Every integrator, thermostat and barostat step is allocation-free after warm-up (`@test_noalloc`, Float64 and Float32) and passes TrimCheck `@validate`; the MD loop sees concrete return types (TypeContracts `@verify` with `trim_compat = true`).
- Integrator and thermostat equations come from the cited papers (Verlet 1967; Swope et al. 1982; Leimkuhler & Matthews 2013; Bussi, Donadio & Parrinello 2007; Berendsen et al. 1984; Marsaglia & Tsang 2000); each task re-derives its update against the paper and cites it in the docstring.
- No Python; no GPL code read; no Distributions.jl (OpenSpecFun_jll via SpecialFunctions).
- Speed is reported as speed-up against a named reference; timing claims come from interleaved runs.
- CI: the existing matrix gains a `package: [PureMLIP, PureAtomistic]` axis on Julia 1.12 and 1.13 with `check_bounds: 'yes'`; every push to main is followed by watching every CI job.

## Review Focus

1. **Neighbor-list capacity overflow mid-run** (a hot step packs more edges than the list holds): the potential regrows the list and the workspace once, logs it, and the step result equals a fresh evaluation. Test in task 2.
2. **Atoms leaving the cell** (unwrapped positions after many steps, triclinic cells): forces and energies equal those of the wrapped configuration. Test in task 3.
3. **Float32 runs**: every integrator and thermostat works in Float32 with the Float32 model; temperatures and energies are accumulated so that a long run does not drift from rounding alone (accumulate the kinetic energy in the working type, state the measured drift). Test in task 4.
4. **Zero or one atom, and a cell with no neighbors** (dilute gas): integrators and thermostats run; temperature with zero degrees of freedom fails with a clear message instead of dividing by zero. Test in tasks 3 and 5.
5. **Reproducibility**: a run with a given RNG seed reproduces bit for bit (thermostat noise drawn in a fixed order); restarting from a written trajectory frame continues identically. Test in tasks 5 and 7.

---

### Task 1: Scaffold `PureAtomistic/`

**Files:** Create `PureAtomistic/{Project.toml, src/PureAtomistic.jl, test/{Project.toml, runtests.jl, coverage/, mutation/}, docs/{Project.toml, make.jl, src/index.md}}`; modify `.github/workflows/{CI.yml, docs.yml, mutation.yml}` (package matrix axis; `PUREMLIP_CHECKPOINT_DIR` for PureAtomistic jobs), `AGENTS.md` (the new package's rules and layer graph).

**Interfaces:** `module PureAtomistic` with empty submodules `Potentials, Systems, Integrators, Thermostats, Barostats, IO, Analysis`; layer graph `Systems → Potentials → {Integrators, Thermostats, Barostats} → Analysis`, `IO` depends on `Systems` only. Docs: a second site deployed under a `PureAtomistic/` subfolder of the same gh-pages branch (PureMLIP stays at the root); the deployment is verified by the first docs run after the push.

- [ ] **Step 1:** Write `test_layers`, Aqua and ExplicitImports items for the empty package; run: FAIL (package missing).
- [ ] **Step 2:** Create the scaffold (generate a fresh UUID), wire CI and docs.
- [ ] **Step 3:** Local suite on 1.12 and 1.13 green; docs build locally.
- [ ] **Step 4:** Commit `Add the PureAtomistic package scaffold`.

### Task 2: Potential contract and the MACE adapter

**Files:** Create `src/potentials/{contracts.jl, mace.jl, harmonic.jl, lennard_jones.jl}`, `test/potentials_tests.jl`.

**Interfaces:**
- `abstract type AbstractPotential end` with a TypeContracts `@contract`: `n_atoms(p)::Int`, `cutoff(p)`, `prepare!(p, positions, cell)::Nothing` (refresh neighbors, regrow on capacity overflow), `energy_forces!(p, positions, cell)::Nothing` (fills `p.result`), optional `stress(p)` (eV/Å³, tension positive), `node_energies(p)`.
- `AbstractStressPotential <: AbstractPotential` requires `stress`.
- `MACEPotential{T}(model, positions, cell, atomic_numbers; extra_capacity_fraction = 0.1)` holds `NeighborList{T}`, `MACEWorkspace{T}`, `MACEResult{T}`.
- `HarmonicPotential` (springs to sites) and `LennardJones` (cut, shifted) for integrator tests.

- [ ] **Step 1:** Tests: `@verify` each concrete potential (`trim_compat = true`); forces equal minus the central finite-difference gradient (measured-margin tolerance); rotation and translation invariance; MACE energy and forces equal `mace_energy_forces` exactly; capacity-regrow item (Review Focus 1); `@test_noalloc` of `prepare!` + `energy_forces!` after warm-up without regrow.
- [ ] **Step 2:** Run: FAIL. **Step 3:** Implement. **Step 4:** Suite, coverage, TrimCheck: PASS.
- [ ] **Step 5:** Commit `Add the potential contract and the MACE adapter`.

### Task 3: System state and initial conditions

**Files:** Create `src/systems/{system.jl, units.jl, init.jl}`, `test/systems_tests.jl`.

**Interfaces:** `System{T}` (positions 3×n, velocities 3×n, masses, species, cell 3×3 columns = lattice vectors); `wrap!(sys)` for triclinic cells; `kinetic_energy`, `temperature(sys; dof)`, `pressure(sys, stress)`; `maxwell_boltzmann!(sys, T, rng; remove_com = true)` that rescales to the exact target temperature; `fcc_cell(a, nrep, element)`, `remove_atom(sys, i)`.

- [ ] **Step 1:** Tests: exact target temperature after sampling; zero center-of-mass momentum; wrap invariance of a MACE evaluation (Review Focus 2); zero-dof temperature throws `"pureatomistic: temperature needs at least one degree of freedom"` (Review Focus 4); OffsetArray and view inputs; noalloc of the observables.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `Add the system state and initial conditions`.

### Task 4: Velocity Verlet and the NVE driver

**Files:** Create `src/integrators/verlet.jl`, `src/driver.jl`, `test/verlet_tests.jl`.

**Interfaces:** `abstract type AbstractIntegrator end`; `VelocityVerlet(dt)`; `Simulation(system, potential, integrator; thermostat = nothing, barostat = nothing, rng)`; `step!(sim)::Nothing`; `run!(sim, nsteps; every = k, observer = f)` calling `observer(sim)` every `k` steps.

- [ ] **Step 1:** Tests on `HarmonicPotential` and `LennardJones` (Float64): energy-fluctuation scaling with dt (log-log slope 2 ± 0.2 over dt = 1, 2, 4, 8 fs); time reversibility (n steps, flip v, n steps) to a stated rounding bound; momentum conservation; Float32 run drift measured and bounded (Review Focus 3). MPA-0 on the tiny checkpoint: 50 NVE steps conserve energy within a measured margin. `@test_noalloc step!`; TrimCheck.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `Add velocity Verlet and the NVE driver`.

### Task 5: Thermostats — Langevin BAOAB and Bussi

**Files:** Create `src/thermostats/{langevin.jl, bussi.jl, gamma_sampler.jl}`, `test/thermostats_tests.jl`.

**Interfaces:** `abstract type AbstractThermostat end`; `LangevinBAOAB(T, γ)` (its own splitting, one force evaluation per step); `BussiThermostat(T, τ)` applied as half-steps around velocity Verlet; `randgamma(rng, shape)` (Marsaglia–Tsang, shape < 1 by the boost).

- [ ] **Step 1:** Tests: Marsaglia–Tsang moments for shapes 0.3, 1, 15; Bussi ideal-gas kinetic energy follows Gamma(N_f/2, kT) (mean, variance, 4 quantiles by bootstrap); Bussi with τ → ∞ equals NVE bit for bit; BAOAB harmonic oscillator ⟨x²⟩ = kT/k and ⟨v²⟩ = kT/m within the block-averaging error; seeded runs reproduce bit for bit (Review Focus 5); zero-dof system fails clearly; noalloc; TrimCheck.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `Add the Langevin and Bussi thermostats`.

### Task 6: Berendsen barostat

**Files:** Create `src/barostats/berendsen.jl`, `test/barostat_tests.jl`.

**Interfaces:** `abstract type AbstractBarostat end`; `BerendsenBarostat(P0, τ_P, β_T)` (isotropic); requires an `AbstractStressPotential`; scales positions and cell each step; pressure from PureMLIP stress (eV/Å³ → GPa via 160.2177) plus the kinetic term.

- [ ] **Step 1:** Tests: pressure relaxes exponentially toward P0 on a Lennard-Jones solid (log-linear decay fit); MPA-0 Cu at P0 = 0 relaxes to the lattice constant of the zero-stress point of E(V) from the gallery's finite-difference method, within a measured margin; noalloc; TrimCheck.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `Add the Berendsen barostat`.

### Task 7: extxyz and the binary trajectory

**Files:** Create `src/io/{extxyz.jl, trajectory.jl}`, `test/io_tests.jl`, `test/fixtures/extxyz/*.extxyz` (hand-written in the format of the ASE documentation).

**Interfaces:** `write_extxyz(io, frame)`, `read_extxyz(path)::Vector{Frame{Float64}}` (known keys into struct fields: Lattice, Properties, pbc, energy, stress, virial, Time; others kept as `Vector{Pair{String,String}}`); `TrajectoryWriter(path, n_atoms, species)` / `TrajectoryReader(path)` (fixed header + Float32 frames, memory-mappable); restart from a frame.

- [ ] **Step 1:** Tests: bitwise round trip; fixtures parse; malformed-header fuzz with `@test_throws "extxyz: …"`; restart from a frame continues bit for bit (Review Focus 5); TrimCheck for the writer.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `Add extxyz and binary trajectory I/O`.

### Task 8: Vacancy tracking and diffusion analysis

**Files:** Create `src/analysis/{vacancy.jl, diffusion.jl, arrhenius.jl}`, `test/analysis_tests.jl`.

**Interfaces:** `reference_sites(cell, a, nrep)`, `wigner_seitz_occupancy!(occ, positions, sites, cell)`, `VacancyTracker(sites; window, dwell)` with `update!(tracker, positions, t)`, `msd_diffusion(track)`, `hop_rate(track)`, `arrhenius_fit(T, rates, errors)` → `(E_m, σ, prefactor)`.

- [ ] **Step 1:** Tests with an analysis-only oracle: a lattice random walk with a known hop rate fed as synthetic frames recovers Γ and D_v within the Poisson/MSD error; a synthetic Arrhenius set recovers a chosen E_m; a Frenkel pair fails with `"pureatomistic: occupancy is not one vacancy at frame …"`; the dwell filter suppresses recrossings in a constructed trajectory; TrimCheck.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `Add vacancy tracking and diffusion analysis`.

### Task 9: Makie extension

**Files:** Create `src/viz.jl` (stubs with docstrings), `ext/PureAtomisticMakieExt.jl`, `test/makie_tests.jl` (CairoMakie).

**Interfaces:** `animate_trajectory(path; kwargs...)`, `plot_vacancy_track(tracker)`, `plot_arrhenius(fit)`, `plot_energy_conservation(log)`; written against `Makie`, so GLMakie and CairoMakie both work.

- [ ] **Step 1:** Tests: each function renders with CairoMakie (figure exists, expected plot objects, nonzero PNG/mp4); Aqua stale-deps confirms the core has no Makie.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `Add the Makie extension`.

### Task 10: Geometry optimization

**Files:** Create `src/optimize/fire.jl` (own FIRE in the core, trimmable) and `ext/PureAtomisticOptimExt.jl` (Optim's L-BFGS for users who load Optim), `test/optimize_tests.jl`.

**Interfaces:** `relax!(sys, potential; fmax = 1e-3, maxsteps)` (FIRE, Bitzek et al. 2006) returning convergence info; the extension adds `relax!(sys, potential, Optim.LBFGS(); …)`.

- [ ] **Step 1:** Tests: FIRE reaches `fmax` on Lennard-Jones and MPA-0 Cu with a vacancy; the relaxed vacancy formation energy reproduces the gallery's value (1.068 eV, 256 atoms) within a measured margin; Optim extension agrees with FIRE; FIRE noalloc and TrimCheck.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `Add geometry optimization`.

### Task 10b: Quasi-Newton relaxation with PureDAQP; remove Optim

- [ ] A trust-region BFGS relaxer: each step minimizes the quadratic model (gradient from the forces, BFGS curvature) inside a box of maximum atom displacement, solved by PureDAQP (`ActiveSet`); fixed atoms and frozen directions as constraint rows. Measure the four PureDAQP settings (`working_set` :rows/:gram × `scan` :all/:window) on the real step problems; check the conditioning before allowing `:gram`, and use `PureDAQP.faster_scan`. Check PureDAQP's dependency closure for `_jll` first. Benchmark FIRE, the PureDAQP relaxer and Optim (L-BFGS, BFGS, conjugate gradient; Optim only in a bench environment) on the Cu vacancy cell and a strained cell: force evaluations, wall time, final energy and fmax, speed-ups against a named reference. Remove the Optim extension and every Optim reference from the package. Amend spec section 6.2: geometry optimization is our FIRE plus PureDAQP.

### Task 10c: Relaxer study — FIRE tuning, FIRE 2.0, limited-memory BFGS

- [ ] FIRE: measure the parameter choices (`dt`, `dtmax`, `maxstep`, `nmin`) on the Cu vacancy and strained cells, and add FIRE 2.0 (Guénolé et al., Comput. Mater. Sci. 175, 109584 (2020)) as a separate optimizer; FIRE's defaults change only on measured evidence. A limited-memory BFGS relaxer in pure Julia (two-loop recursion, memory `m` measured) with the same trust region, acceptance rules and fixed-atom/frozen-direction handling as `TrustRegionBFGS`; PureDAQP solves the step only when the box binds. Benchmark FIRE, FIRE 2.0, `TrustRegionBFGS`, the limited-memory relaxer and Optim L-BFGS (bench environment only) on the two 256-site cells and a cell of at least 4,000 atoms: force evaluations, wall time, peak memory, final energy and fmax, speed-ups against FIRE. The default relaxer is chosen from these measurements at both sizes. Includes the open Task 10b review items (noise-branch radius on rejection, docstring benchmark figures moved to the JSON, citation wording, the KKT model-value tolerance).

### Task 11: Trimmed `md_run` binary

**Files:** Create `PureAtomistic/juliac/{build.jl, entry.jl}`; the `:juliac` test item.

**Interfaces:** `md_run <model.model> <config.toml> <out.traj>`: reads a `.model`, builds the system from the config (lattice, size, vacancy, temperature, thermostat, steps, dt), runs, writes the binary trajectory and an extxyz final frame.

- [ ] **Step 1:** Test: the `:juliac` item builds with `--trim=safe` and runs 100 steps on the tiny checkpoint; output equals the untrimmed run bit for bit.
- [ ] **Step 2–4:** FAIL → implement → PASS. **Step 5:** Commit `Build md_run with juliac --trim`.

### Task 12: Validation runs, gallery pages, results

**Files:** Create `PureAtomistic/docs/examples/{nve_conservation.jl, thermostat_check.jl, vacancy_hop_md.jl}`, `PureAtomistic/bench/results/*.json`.

- [ ] **Step 1:** Short MPA-0 runs only (each well under an hour on one core): NVE energy conservation of 500 Cu atoms over 2 ps at dt = 1, 2, 4 fs (find the largest stable dt); Bussi NVT at 1200 K for 2 ps (temperature distribution); one vacancy cell at 1300 K for as long as fits in an hour, recording any hops — an illustration, not a diffusion measurement. GLMakie animations of each.
- [ ] **Step 2:** Docs build with the pages; CI green on every job after the push.
- [ ] **Step 3:** Commit `Add MD validation runs and gallery pages`; final review; push to main.

**Deferred (user ruling 2026-11-02):** the vacancy-diffusion campaign over a temperature range (spec section 7, items 4–5) waits for GPU support; plan 5 builds and validates every tool it needs.
