# QuantumEngine step 03 (molecular integrals) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the module `QuantumEngine.Integrals`: the Boys function, one-electron integrals (overlap, kinetic, nuclear attraction with point charges, dipole, angular momentum, ESP), McMurchie–Davidson shell-quartet ERIs with Schwarz and density screening, integral-direct and in-core J/K (serial and on a persistent worker pool), density fitting (RI-J, RI-K, the 3-center tensor), MO transforms, and derivative integrals, matching the Rust oracle `quantum_engine` (commit 4a78a1d) where the oracle is right and an independent reference where it is not.

**Architecture:**
- One new module, `src/integrals/Integrals.jl`, above `Basis` (step 02) in the layer graph: `Integrals → (Basis, System, Errors)` (the exact set is the one the code uses: Task 1 registers the row with the modules its code uses, and Tasks 3, 5 and 12a each add the module their code first uses).
- One numerical core, shared by every integral: Hermite expansion coefficients `E` per primitive pair (`hermite.jl`), the Hermite Coulomb table `R^m_tuv` in packed `t+u+v ≤ L` storage (`hermite_r.jl`), and the Boys function (`boys.jl`, a 16-point-per-unit Taylor table stored as hex literals). Overlap and kinetic use `E` only; nuclear attraction, ESP, ERIs and the RI integrals use `E` and `R`.
- ERIs are computed per shell quartet into a `ncart(la)×ncart(lb)×ncart(lc)×ncart(ld)` block from shell-pair data precomputed once per geometry (`ShellPairData`), not per Cartesian component as the oracle does. This is a change of cost, not of numerics.
- J/K loops over the oracle's unique-quartet order with fixed-order accumulation. The threaded path runs one accumulator pair per worker on a persistent worker pool owned by a `JKWorkspace` (the PureMLIP `ChunkWorkers` pattern), deals the outer shell pairs to workers by a fixed rule and reduces the accumulators in worker order, so J/K are bitwise reproducible for a fixed worker count (Q-DETERMINISM, decided).

**Tech Stack:**
- Julia 1.13; LinearAlgebra (stdlib), StaticArrays 1.9, StrictMode 0.5.21 (already in `[deps]`); TypeContracts 0.14.7 enters `[deps]` with the `AbstractIntegralEngine` contract (Task 8, step-01 D14). PureTaylor enters through `[sources]` in Task 15 under the Q-DERIVINT default. Printf (stdlib) is the only dependency of the generator project `gen/Project.toml` (Task 1); it is not in the package `[deps]`.
- Tests: TestItems/TestItemRunner, StrictModeTest, TrimCheck 0.1.5, Aqua, ExplicitImports, OffsetArrays, TypeContracts (all already in `test/Project.toml`). No new test dependency unless D3 picks the libcint option.

**Spec:** `03-integrals.md` of the QuantumEngine rewrite specs, with `00-overview.md`, `PLAN.md`, `02-basis.md` (the `Basis` interface this plan consumes) and `99-open-questions.md` (its Decisions section). They live outside the repository (step-01 plan D19), so the executing coordinator hands them to each implementer and reviewer with this plan. The oracle checkout is named by `QUANTUM_ENGINE_ORACLE_SRC`; it is read-only: never modified, never built, never run by an agent (step-01 plan D17).

**Precondition.** Step 02 (`QuantumEngine.Basis`) is merged into the working branch before Task 1 starts. This plan uses the names of spec 02's public interface (listed under each task's Interfaces). They were checked against the draft step-02 plan `2026-11-25-qe-02-basis.md` (lines 213–231: `ncart`, `cartesian_component`, `primitive_norm`, `contraction_renorm`, `Shell{T}(center, atom, l, exponents, coefficients, ao_offset)`, `BasisSet{T}(shells, nao, name)`, `ao_range`, `shell_pairs`, `build_basis(mol, name[, T])`) and agree; that plan's `Shell` constructor accepts any `l ≥ 0` (its line 226) and `build_basis` throws only when it places an I shell (its D2), which the l = 6 items of Tasks 3, 5 and 7 rely on. If the step-02 code changes any of them, the coordinator updates this plan before Task 1.

**Evidence.** The research behind this plan is in a session scratch directory, which is not durable. Every figure the tasks rely on is in the appendix "Evidence" with its command or source line. A task that needs a figure not in the appendix measures it again and records it.

---

## Decisions needed

These rows change a literal requirement of spec 03, of PLAN.md or of the user's standing rules, or are open questions of `99-open-questions.md` that PLAN.md places before step 03. The user decides them; the plan does not. The default is what the plan implements if the user confirms it. **No task a row blocks starts until the user has answered that row.** A row marked "Decided by the user (2026-11-26)" is answered; its text says how the tasks apply it, and it blocks nothing unless it carries a follow-up question.

**User decisions of 2026-11-26** (recorded in `99-open-questions.md`, "Decided 2026-11-26 (user)"): Q-DETERMINISM, Q-BLAS, Q-AUXDEFAULT and Q-LMAX below. The `libxc_*` passthroughs are implemented in PureXC later and do not touch step 03. Step 02 extends the step-01 `Units` element table past Z = 36 to every element the shipped basis files use (a step-02 task, step-02 plan D6); step 03 relies on nothing above Kr.

| # | Question | Default offered | Blocks |
|---|---|---|---|
| Q-DETERMINISM | **Decided by the user (2026-11-26):** "fixed-order reductions; J/K bitwise reproducible for a fixed thread count." | Applied (option (a) of 99): `JKWorkspace(eng; nworkers = 1)`. With `nworkers = 1`, `jk!` accumulates straight into `J` and `K`. With `nworkers = w > 1` there is one chunk per worker (as `ChunkWorkers` runs them, D6), the outer shell pairs `(a, b)` go to chunk `mod(pair_index(a, b) − 1, w) + 1`, each chunk accumulates into its own J/K pair, and the pairs are summed into `J`, `K` in chunk order 1:w. Results are `==` across calls with the same `w`; across different `w` they differ by summation order only (a class-(b) bound, Task 11). Accumulator memory is `w × 2N × nao² × sizeof(T)` for N densities (w = 4, N = 2, nao = 1000: 128 MB, arithmetic). **Follow-up question:** the previous default was option (b) (bitwise across thread counts: `nchunks = 16` fixed, per-chunk accumulators even in serial runs, a chunk-to-worker mapping), which also meets the decision but costs `nchunks × 2N × nao² × 8` bytes (512 MB at nao = 1000, N = 2). Keep (a) as applied, or (b)? | 11 (follow-up only) |
| Q-BLAS | **Decided by the user (2026-11-26):** "write against LinearAlgebra only; pick the default (OpenBLAS or PureBLAS) after interleaved SCF benchmarks." | Applied: code calls `LinearAlgebra` only. The backend choice is step 04's (its SCF benchmarks); Task 18 only records the backend each benchmark ran on (`LinearAlgebra.BLAS.get_config()`) and runs no second BLAS arm. | none (decided) |
| Q-LMAX | **Decided by the user (2026-11-26):** "l ≤ 5 now, with a BasisError naming the I shell; l = 6 later." | Applied: step 02 throws the `BasisError` when `build_basis` places an I shell (step-02 D1, D2). Here `BOYS_M_MAX = 22` (4·5 for an ERI plus 2 derivative orders), the table holds m = 0:29 for i = 0:576 (577 × 30 entries), and the packed R table holds at most `binomial(22 + 4, 4) = 14,950` entries per primitive quartet. A basis set built directly from `Shell`s with l = 6 is rejected by the `Integrals` constructors (Global Constraints, "Angular-momentum limit"). | none (decided) |
| Q-AUXDEFAULT | **Decided by the user (2026-11-26):** "an omitted aux_basis … is an alias for the shipped def2-universal-jkfit, with a logged note." | Applied in step 02 (`build_basis` resolves the alias and logs the note, step-02 D3) and step 04 (what an omitted `aux_basis` means). `RIContext(bs, aux)` takes an explicit auxiliary `BasisSet`; step 03 has nothing to apply. | none (decided) |
| Q-SCFGATE | Where the oracle speed gate lives (99, "03 specifics"). PLAN.md says "03 and 04 carry speed gates against the Rust oracle"; the 74 s benzene figure is not recorded anywhere (E11), the oracle prints no J/K timings, and agents may not run the oracle (step-01 D17). A release binary exists at `target/release/quantum_engine`, not run (E11). Under (a) step 03 has no pass/fail speed check: a regression in Tasks 7–11 (for example per-component instead of shell-quartet blocks) would merge unnoticed, and the only baselines (E4, E5) were measured under load average 4.3–5.5. | **(a):** 03 delivers a reported benchmark (Task 18); the pass/fail oracle comparison moves to 04, with oracle wall times captured by the user through `capture.jl`. PLAN.md's "03 and 04 carry speed gates" then reads "04 carries the speed gate; 03 reports". **Offered addition (a′):** a Julia-only gate in Task 18: on benzene/6-31G at 1 thread, `jk!` time ≤ c × the integral-only sweep (the same quartets through `eri_shell_quartet!`, no scatter) re-measured interleaved in the same run, with c set by the user (a ratio within one process pair is robust to this machine's drift). If the user instead names the binary and asks, Task 18 Step 5 times whole oracle runs (water/6-31G and benzene/6-31G RHF, 1 and 4 threads) as reported figures. | 18 |
| Q-BENCHDENSITY | The benchmark density. Spec 03: "`jk!` wall time per call … with a stored converged density". No converged RHF density exists before 04. Step 01 commits an oracle-converged water density, `test/oracle/captured/water_pbe_631g.molden` (step-01 plan, Task 13), which the Molden reader loads; there is none for benzene. Density screening depends on the density. | Water/6-31G uses the captured PBE density (`P = 2 C_occ C_occᵀ` from the Molden orbitals; a converged PBE density screens like the RHF one, which is not checked). Benzene/6-31G uses the core-Hamiltonian guess density, recorded as such in the JSON, and 04 reruns benzene with its converged density. | 18 |
| Q-BOYSPARITY | HF parity tolerance and the oracle's Boys defect (99). Measured (scratch `qe-plans/p03_oracleboys_sens.jl`, run): in the oracle's upward branch (t < 15) the error depends on the exact bits of F_0. Shifting F_0 by −2…+2 ulp around the correctly rounded value gives relative errors of F_4(1e-4) of 115, 49.6, 16.0, 81.5, 147; of F_4(1e-3) of 0.013 … 1.05e-4 … 0.013; of F_6(0.01) of 0.48 … 8.3e-3 … 0.47. With a correctly rounded `erf` but the oracle's Float64 expression `0.5·sqrt(π/t)·erf(sqrt t)` the errors are 0.013 at (1e-3, 4), 0.228 at (0.01, 6) and 3.4e-3 at (0.1, 8), against spec 03's 1.0e-4, 8.3e-3 and 1.1e-4. So spec 03's six figures hold only for a correctly rounded F_0; the oracle's real defect depends on how `libm::erf` rounds (not measured) and is likely larger. No variant built on this package's `erf` reproduces either. | **(2):** Task 16's variant keeps this package's `erf` (Task 2) and the oracle's algorithm and expression order (`F_0 = ½√(π/t)·erf(√t)`, upward `((2m−1)F − e^{−t})·(0.5/t)`). Its test checks the algorithm, not the spec figures: against a `BigFloat` evaluation of the same recurrence started from the same Float64 `F_0` and `e^{−t}` (class (b), the bound propagated step by step in the test). Its own defects against `boys_big` at the six points are reported, not asserted. It does not claim to reproduce the oracle, and the Task 16 integral and J/K deltas are estimates of the oracle's defect. The energy-level measurement and the parity row ("measured Boys defect + 1e-9 Ha") belong to 04. **Alternative (1):** bit-identical F_0 by porting the `erf` of the Rust `libm` crate (an FreeBSD msun derivative; its license is not checked) with the oracle's operation order; this borders on the no-translation rule for the Apache-2.0 oracle's dependency code, so it is the user's call. Either way, spec 03 Facts and its Q-BOYSPARITY entry should say that the six figures are lower bounds for a correctly rounded F_0 (a spec correction for the coordinator). | 16 |
| Q-DERIVINT | How derivative integrals are computed (99). PLAN.md: "derivatives are computed by automatic differentiation wherever possible". Spec 03 option (c): "shift relations for gradients, AD for the Hessian", and "the second-derivative entries 08 consumes join the public interface". PLAN.md ("Code the specs call placeholder") puts an analytic Hessian in scope; the oracle's analytic Hessian takes its two-electron term from finite differences of the analytic 2e gradient (spec 08), which PLAN.md says becomes AD; spec 08 says the oracle's analytic Hessian "is not ported". | **(c):** first derivatives by angular-momentum shift relations (Task 14, which states in its docstrings why AD is not used there: the gradient is on every optimization step's hot path); second derivatives of the one-electron integrals (the oracle has them for S, T and V) by PureTaylor through the generic one-electron kernels (Task 15), which loosens those kernels' bound from `T<:AbstractFloat` to `T<:Real`. **Two-electron second derivatives:** the contraction itself is assigned to 08; 03 exposes what 08 needs for it: Task 15 also loosens `eri_shell_quartet!` (and `hermite_E!`, `hermite_R!`, `ShellPairData`) to `T<:Real` with a Taylor method of the Boys function, and tests a second-order Taylor ERI block against Task 14's shift-relation first derivatives and a central difference of them. Task 15 benchmarks option (b) (first-order PureTaylor through that committed kernel) against Task 14 and reports it, so the user can revisit. The user also reconciles spec 08's "not ported" with the analytic Hessian that PLAN.md keeps in scope. | 14, 15 |
| Q-ORACLEDEAD | Dead and unreachable oracle code (99). | **(a):** a "Disposition of oracle code" table in `docs/src/integrals.md` (Task 17), one row per oracle file: `nuclear.rs`, `eri_shell.rs`, `eri_hrr.rs` (no call sites) and the lattice-image S/T sums and `integrals/ewald.rs` (unreachable from the CLI) are not ported; ZORA (`zora.rs`, `zora_so.rs`) has no entry point in 03: the run-mode driver of 04 throws `NotImplemented` naming the later step when an input asks for it (step-01 D21), and the row says so; `magnetic.rs` is empty; `erf`/`erfc` are kept (Task 2) for 15's `pw/ewald.rs` and for Task 16. | 2, 17 |
| Q-ORACLESRC | The disposition check reads the oracle checkout. The copied Global Constraint allows exactly one test that reads `QUANTUM_ENGINE_ORACLE_SRC` (step-01 Task 1b's vendored-files check). | The check is a manual step of Task 17 (the list of oracle files and the table compared once, the result recorded in the commit body); no second `:oracle_src` item. **Alternative:** allow a second `:oracle_src` item and amend the constraint. | 17 |
| Q-ERFACC | `erf`/`erfc` accuracy and placement. Spec 03: `erfc` "Float64 accuracy ≤ 1 ulp class" in `Integrals.Special`. | Task 2 asserts the fixed bound `ulp_error ≤ 1` for both functions on the stated grids (spec). If the Cody implementation measures above 1 ulp anywhere, the task stops and asks; the tolerance is not set from the measurement. The functions live in a submodule `Integrals.Special` (`src/integrals/special.jl`), as the spec names it. | 2 |
| Q-RIFIT | The RI Coulomb fitting-error test. Spec 03 compares a class-(c) bound against "the measured fitting error of the oracle's `use_ri_j true` run against its exact run". In 03 that compares two different quantities (our ΔE_J at a guess density against the oracle's total-energy difference at convergence); their ratio is not measured, so a 20× bound may fail or never bind, and it cannot be calibrated before 04 produces converged densities. | Move the oracle comparison to 04 (E_RI − E_exact at convergence on both sides). 03 keeps only `"ri: close to exact Coulomb"` (class (c), self-measured), and Task 12b is not run. **Alternative:** keep it in 03 as Task 12b (the captures, then the item). | 12b |
| Q-RIF32 | RI conditioning. `ri_coulomb!` applies `V⁺` with `1/λ` on eigenvalues `> RI_METRIC_CUTOFF = 1e-10`. In Float32 the eigenvalue noise is about `eps(Float32)·λ_max ≈ 1e-7·λ_max`, far above 1e-10, so noisy eigenvalues are kept and their inverses amplify the error; the in-core-vs-direct Float64 comparison passes summation-order differences through `V⁺` too. `cond(V)` for def2-universal-jkfit on water is not measured. | Task 12a Step 3 measures the smallest eigenvalue and `cond_kept` of the water/def2-universal-jkfit metric in Float64 and Float32 and records them. The in-core-vs-direct and Float32 bounds carry a `cond_kept` factor (class (b)), keeping the spec's absolute cutoff for both types. **Alternative:** for Float32 a cutoff relative to `eps(T)·λ_max` (a change of the spec's cutoff). If the Float32 `cond_kept` makes the bound vacuous (above `1/eps(Float32)`), the task stops and asks. | 12a |
| D1 | `K[0,0] = 9.697012` (spec 03, unasserted PySCF literal of `test_jk_verify.rs`). Three independent codes give 9.990743 (E1); the literal equals the value with the three O 1s–2p exchange terms subtracted (9.697014, E2). That an older oracle build produced it is not checked. | Drop the literal as a reference. The test asserts `K[0,0]` against 9.990743 (libcint 6.1.2 and GaussianBasis.jl's own integrals, agreeing to 8 printed digits, E1) within `5e-7 + 8 eps · N_q · S_abs`, class (b): half a unit in the sixth printed decimal of the reference plus the rounding bound of our sum; the comment records both values and E2's arithmetic. Spec 03's acceptance list is the user's to correct. | 10 |
| D2 | `J[0,0] = 18.502906` fails the spec's ±1e-6: the same three codes give 18.502908 (E1). | Assert 18.502908 within the D1 bound (same provenance comment as D1); record the literal and the 2e-6 gap. The alternative is to keep the PySCF literal at ±1e-6 and investigate, which the spec asks for on failure ("never answered by a wider tolerance"). | 10 |
| D3 | Independent references above l = 1. The oracle's asserted literals cover s and p only; spec 03 checks d, f, g only through symmetry, normalization and finite differences, which a consistent convention error would pass. | A test-only, pure-Julia `BigFloat` reference by a different algorithm (Obara–Saika vertical recurrence plus Head-Gordon–Pople horizontal transfer, in `test/reference/os_reference.jl`, written from Obara & Saika 1986 and Head-Gordon & Pople 1988, never from the oracle or from the package's MD code) for S, T, V and ERIs of every class up to (dd|dd) and spot quartets of (ff|ff), (gg|ss) and (hs|hs). The alternative is a separate test environment `QuantumEngine/test/libcint/` with GaussianBasis.jl (libcint_jll 6.1.2, OpenSpecFun_jll; E7), as PureXC keeps Libxc in `PureXC/test/libxc/`; libcint's license was not checked. | 10 |
| D4 | Screening classes and cuts (E9). The oracle applies the `> 45` exponent cut and the `< 1e-18` prefactor cut to its McMurchie–Davidson classes only, not to (ss\|ss) and (ps\|ss); skips individual E terms `< 1e-15`; and for every class skips a contraction weight below 1e-15 at each nesting level (`c_a`, `c_a c_b`, `c_a c_b c_c`, the full weight; `eri_contracted.rs:25-44`), each on `coef_phys`, which includes the per-component norm. | MD for every class, with the exponent and prefactor cuts applied to every class from `IntegralThresholds`; no per-E-term skip (the shell-quartet products are dense); **no contraction-weight skip**: under A4 the per-primitive weight carries the `(l,0,0)` norm and not the per-component one, so the oracle's test cannot be stated on the same number. `thresholds.primitive` is not used inside `eri_shell_quartet!`; it gates only the integral skip of `jk!` (`|I| < primitive` is not scattered) and the entries `PackedERI` stores. Task 7 measures the largest difference on water/6-31G and benzene/6-31G against a scratch variant with the oracle's per-class rule and its weight skip (on `c_a c_b c_c c_d · primitive_norm` per component), and records it; the difference is expected at the `exp(−45) ≈ 2.9e-20` relative level (estimate, not measured). | 7 |
| D5 | Spec 04 asks for `memory_limit_elements` of `RIContext` to become `memory_limit_bytes` ("03 spec change, noted for the coordinator"). | `RIContext(bs, aux; memory_limit_bytes = 512_000_000, …)`, in-core when `npairs · naux · sizeof(T) ≤ memory_limit_bytes` (512,000,000 bytes is the oracle's 64,000,000 Float64 elements). | 12a |
| D6 | The worker pool code. QuantumEngine does not depend on PureMLIP; `ChunkWorkers` (`PureMLIP/src/model/workers.jl`, 172 lines) passes PureMLIP's trim and threaded juliac items (read, not rerun, E8). It runs chunks `1:n−1` on workers `1:n−1` and chunk `n` on the calling task, one chunk per worker (`workers.jl:3-5, 115-146`, read). | Copy it into `QuantumEngine/src/integrals/workers.jl`, with a comment naming `PureMLIP/src/model/workers.jl` as the origin within this repository, and change the error prefix to `integrals:`. Q-DETERMINISM as applied uses one chunk per worker, which is the pool's own model. The alternative is to factor it into a shared package, which changes PureMLIP (a shared file, and its owner's session). | 11 |

---

## Decisions this plan assumes

The spec leaves these open. The plan implements the stated default. **No task a row blocks starts until the user has confirmed or changed that row**; every row is confirmed at plan review.

| # | Question | Default this plan implements | Blocks |
|---|---|---|---|
| A1 | `boys!` for number types other than Float32 and Float64. The table has Float64 accuracy, so a `BigFloat` or double-double caller would get Float64 accuracy silently. | `boys!` and `boys` are defined for `T ∈ (Float32, Float64)` (Float32 rounds the Float64 result). A second method `boys!(F::AbstractVector, nmax::Int, t::Real)` (and `boys(m::Int, t::Real)`) throws `ArgumentError("integrals: boys! supports Float32 and Float64; the table has Float64 accuracy")` for any other type. The kernels stay `T<:AbstractFloat` (spec), so a `BigFloat` ERI fails fast at the Boys call. Task 15 adds the Taylor method (Q-DERIVINT). | 1 |
| A2 | Boys table source form. | Hex float literals (`0x1.5555555555555p-2`), exact by construction, in `src/integrals/boys_table.jl`, written by the generator `QuantumEngine/gen/boys_table.jl` (1024-bit series) in its own project `QuantumEngine/gen/Project.toml` (Printf only); the generator is committed and its output is checked byte-for-byte by a `:slow` item that runs it in a child process. Task 1 measures the load-time cost of the file and reports it. | 1 |
| A3 | One-electron scheme. Spec 03 fixes MD for ERIs; it leaves S, T and V open (the oracle uses Obara–Saika for them). | MD for all: S and T from the 1D Hermite `E` (kinetic by the `l ± 2` overlap relation per axis), V from `E` and `R`. Numerics agree with Obara–Saika to rounding; the oracle literal `S_AB` and D3's reference check them. | 5 |
| A4 | Per-component normalization (spec 02: the consumer applies `primitive_norm(α, lx, ly, lz)`). | Each primitive pair weight holds `c_a c_b · primitive_norm(α_a, l_a, 0, 0) · primitive_norm(α_b, l_b, 0, 0)`; the α-independent ratio `primitive_norm(α, lx, ly, lz) / primitive_norm(α, l, 0, 0)` is applied once per component after contraction (E9: ≤ 1.1e-15 relative from the per-component form in the prototype). A test checks that the ratio is α-independent. | 3 |
| A5 | `MDEngine.workspaces` (spec: "one per worker of the J/K pool") versus `JKWorkspace(eng; nworkers)`, which also sets the worker count. | `MDEngine` holds one `ERIWorkspace` (`workspaces[1]`, for serial callers and `schwarz_bounds`); `JKWorkspace` owns one `ERIWorkspace`, one block buffer and (for `nworkers > 1`) one J/K accumulator pair per worker. The engine is never mutated by `jk!`. | 7, 11 |
| A6 | `eri_gradient_contract!(G, eng, P, W; exx)`: spec 03 does not define `W`; spec 06 calls `two_electron_gradient!(G, eng, P; exx)` and puts the energy-weighted density in `pulay_gradient!`. | `eri_gradient_contract!(G, gws::ERIGradientWorkspace, eng, densities::NTuple{N,AbstractMatrix}; exx)` with N = 1 (total P, RHF) or 2 (P_α, P_β), adding `∂E₂/∂R` to `G` (natom × 3), and an exempt entry-point form without `gws`. No `W` argument. | 14 |
| A7 | RI gradient entries (spec: names only). | `ri_three_center_gradient!(G, ctx, Γ::AbstractMatrix)` adds `Σ Γ[pq, A] ∂(μν\|A)/∂R` (Γ packed `[npairs, naux]`) and `ri_two_center_gradient!(G, ctx, Γ2::AbstractMatrix)` adds `Σ Γ2[A, B] ∂(A\|B)/∂R`; 06 forms Γ for RI-J and RI-K. Their scratch is the derivative scratch the `RIContext` owns when built with `deriv_order = 1` (Task 14). | 14 |
| A8 | RI-K scratch. `ri_exchange!` must not allocate, but its scratch depends on the occupied count. | `RIContext(...; max_occupied = nao)` sizes the scratch; `ri_exchange!` throws `ArgumentError` naming both counts when `size(Cocc, 2) > max_occupied`. | 12a |
| A9 | Asymmetric densities. Spec 03: "`densities` is a tuple of symmetric nao×nao matrices"; the 8-fold loop would symmetrize an asymmetric one silently. An exact `==` check could reject a density built by `mul!(P, C, C')`, whose two triangles are not guaranteed to agree bitwise (not checked). | `jk!`, `coulomb!` and the `PackedERI` `jk!` read only the lower triangle (`i ≥ j`) of each density, as `Symmetric(P, :L)` does, and say so in their docstrings; a test shows that changing the strict upper triangle changes nothing. The alternative is an exact symmetry check that throws. | 8 |
| A10 | `core_hamiltonian` and ECPs (E9: the oracle adds `ecp_shell_shell` at `scf/utils.rs:103`; spec 02 sends ECP integrals to 14). | `core_hamiltonian(bs, mol) = T + V` (nuclei and point charges). Its docstring says that ECP terms are absent and are added by step 14; `use_ecp true` already throws `NotImplemented("14-embedding §ECP", …)` in 02. | 5 |
| A11 | Fixtures. | `test/fixtures.jl` gains `water_eri()` (the `test_eri_compare.rs:8-42` geometry verbatim: O at the origin, H at `(±r sin θ, 0, r cos θ)`, r = 1.809 bohr, θ = 104.52°/2) in Task 3, its first user, and `benzene()` (planar D6h, C–C 1.39 Å, C–H 1.09 Å, converted with `Units.ANGSTROM_TO_BOHR`) and `h2o_sto3g_jk_density()` (`diag(2, 2, 2, 2, 2, 0, 0)`) in Task 5. H2 at R = 1.4 bohr is the existing `h2()`. | 3, 5, 7 |
| A12 | J/K conventions. | As `src/scf/jk.rs:6-60`: `J_μν = Σ (μν\|λσ) P_λσ` and `K_μν = Σ (μλ\|νσ) P_λσ`, no factor ½; 04 applies the factors. Neither J nor K is symmetrized afterwards beyond what the 8-fold scatter writes (E9: `jk_symmetric.rs` does not symmetrize). | 8 |
| A13 | Scratch for the `!` functions. Spec 03 gives `overlap!(S, bs)` etc.; "functions ending in `!` must not allocate", and their scratch (E tables, the R table, Boys values) depends on the basis set's l_max. | Each one-electron `!` function has two methods: `overlap!(S, bs, ws::OneElectronWorkspace{T})` allocates nothing (the `@test_noalloc` target), and `overlap!(S, bs)` builds the workspace and calls it (exempted by name as an entry point). The same for `kinetic!`, `nuclear_attraction!`, the derivative `!` functions of Task 14 (one-electron with `OneElectronWorkspace`, ERI with `ERIGradientWorkspace`) and the hessian `!` functions of Task 15. `three_center!`, `ri_coulomb!`, `ri_exchange!` and the RI gradients use the scratch the engine or `RIContext` owns and have no workspace-free form. | 5, 12a, 14, 15 |
| A14 | Which Boys function the integrals use. 04 needs energies with the oracle's Boys algorithm under Q-BOYSPARITY. The oracle uses the same `boys.rs` for nuclear attraction, ESP and the RI 2c/3c integrals (`nuclear_attraction.rs`, `esp.rs`, `eri_2c2e.rs`, `eri_3c2e.rs`, read by `grep`). | `hermite_R!` takes the Boys function as an argument (default `boys!`). `MDEngine{T,F}` carries `boys::F` (a trailing type parameter, default `boys!`); `OneElectronWorkspace{T,F}` carries it too (`OneElectronWorkspace(bs; boys = boys!)`), and `esp_matrix`, `two_center` and `RIContext` take a `boys` keyword (RI passes it to its engine). A test can then build every integral on Task 16's variant without a second kernel. | 4, 5, 6, 7, 12a |
| A15 | `IntegralThresholds` construction. The spec's `Base.@kwdef struct IntegralThresholds{T}` with `exponent_cut::T = 45` makes `IntegralThresholds(density = 0.0)` a `MethodError` (measured, scratch `qe-plans/kwdef3.jl`: `no method matching IntegralThresholds(::Float64, ::Float64, ::Float64, ::Float64, ::Int64)`), and a bare `@kwdef` does not validate. | No `@kwdef`. The cascade: an outer keyword constructor `IntegralThresholds(; kw...)` calls `IntegralThresholds{Float64}(; …)`; `IntegralThresholds{T}(; schwarz = 1e-12, density = 1e-12, primitive = 1e-15, prefactor = 1e-18, exponent_cut = 45) where {T}` calls the inner constructor `IntegralThresholds{T}(schwarz, density, primitive, prefactor, exponent_cut)`, whose arguments are unconstrained, which converts with `new` and throws the validation error. Both `IntegralThresholds(density = 0.0)` and `IntegralThresholds{Float32}()` work. | 7 |

## Global Constraints

Copied verbatim from the step-01 plan (`2026-11-25-qe-01-foundations.md`), which copies them from `AGENTS.md` and from the rewrite decisions. Every task's requirements include them. The step-03 additions follow the copied block.


- **Placement (Q-REPO).** "QuantumEngine is a sub-package of the PureAtomistic monorepo with a path dependency on PureXC." It is self-contained: "its own `Project.toml`, `src/`, `test/`, `docs/`". The PureXC dependency enters at step 05 (D14).
- **Julia floor.** `[compat] julia = "1.13"`: "Julia 1.12 (LLVM 18) miscompiles long straight-line Float32 kernels for AVX-512 targets at `-O2` and above".
- **Closure.**
  - "No `_jll` packages, no artifacts", outside the standard libraries. The item "core closure is binary-free" checks it.
  - "No Python in any form: no interpreter, no Python-backed package, no Python-installed stack."
  - KernelAbstractions only as a `[weakdeps]` extension, never `[deps]` (Q-GPU); none in step 01.
  - "Every dependency has a `[compat]` bound. Stdlibs are excepted in the test project."
  - JSON is JSON.jl, never JSON3 (D2 fixes the bound). FFTs use FFTL, which step 01 does not use.
  - Libxc is never in `QuantumEngine/Project.toml` or `QuantumEngine/test/Project.toml` (as for PureXC in AGENTS.md); the item "Libxc is not a dependency" checks both.
  - "Every new dependency: its closure is resolved and checked `_jll`-free before it is added" (PLAN.md). The task that adds a dependency resolves it first in a throwaway environment outside the repository, inspects the Manifest for non-stdlib `_jll` packages and artifacts, and records the package list and versions in its commit body and in `QuantumEngine/docs/src/dependencies.md`.
- **Code rules (AGENTS.md).**
  - "Generic indexing: iterate `eachindex`/`axes`, or call `Base.require_one_based_indexing` to declare a 1-based assumption. Tests run entry points on `OffsetArray` and `view` inputs."
  - "No `@inbounds` without a benchmark showing the bounds check matters end to end, a local proof of the index range, and a test run under `--check-bounds=yes`." Step 01 has none.
  - "Return types and struct fields are concrete; no `Any` containers."
  - "Functions ending in `!` must not allocate."
  - "Fail fast: errors are `ArgumentError` or carry a module prefix … No silent fallbacks." The prefixes are D13's.
  - "Use `lazy\"...\"` for interpolated messages with four or more pieces."
  - "Comments describe the code as it is: invariants, non-obvious reasons, constraints. No history, plans, or task references." Code ported from MIT-licensed work carries an attribution comment. The oracle is Apache-2.0: its code is not copied; its input files are vendored as data (D16).
  - No fast-math anywhere.
- **Name trap.** Inside `QuantumEngine` and its `IO` module the name `IO` is the submodule, so stream arguments are annotated `Base.IO`. Tests never write `using QuantumEngine.IO` as a bare statement: after it, `IO` is ambiguous between Base and the submodule (`UndefVarError: IO not defined … also exported by Base`, measured on Julia 1.13.4). They write `using QuantumEngine.IO: name1, name2` or qualify as `QuantumEngine.IO.name`.
- **Logging.** `IO` and every module above it log through `IO.Logger` (spec 01); `Errors`, `Units` and `System` sit below `IO` and do not print. The quality item "no bare printing to stdout" scans `QuantumEngine/src` outside `src/io/logger.jl` for the regex `\b(println|print|printstyled)\((\"|lazy\"|stdout)` and for `@show`, and fails on any match.
- **Messages.** Error and warning messages are stored as `LazyString` (field type `msg::LazyString`), so that an interpolated `lazy"..."` message is not converted to a `String` when the error is built. Constructors take any `AbstractString` and convert with `LazyString(msg)`. A message is printed only by walking `msg.parts` in one helper, `Errors._print_message(io::Base.IO, m::LazyString)`, which writes each part with a call on its concrete type (`String`, nested `LazyString`, `SubString{String}`, `Symbol`, `Char`, `Bool`, `Int`, `Float64`) and throws for any other part type. `print(io, ::LazyString)` is never called: measured during Task 1, it fails the typestable and noboxing proofs. The `showerror` methods and `InputWarning`'s `show` print through this helper, and the juliac entry prints an error with `showerror`, so through it too.
- **StrictMode.**
  - `@strict_policy (:typestable, :noalloc)` is the last statement of every module, preceded by the comment `# Last statement: the policy checks the methods defined above it.`
  - A module imports from StrictMode only the macros it uses: `using StrictMode: @strict_policy`, plus `@strict_exempt` in a module that has an exemption (ExplicitImports' stale-import check fails otherwise).
  - `@strict_exempt` is allowed only for entry points and allocating constructors, narrowed to `:noalloc`, with a present-tense reason naming its kind. The one exception is the message rendering below, which is also exempt from `:noboxing` (user decision 2026-11-25: "keep it fast and trim-compatible"). Each exemption names its function, as `PureAtomistic/src/io/IO.jl` does: `@strict_exempt :noalloc read_xyz reason = "entry point: xyz reader"`. An allocating helper that only an entry point calls is exempted by name with a reason naming that entry point. Every allocating constructor is exempted in the task that writes it: `@strict_exempt :noalloc <Type> reason = "entry point: <kind> constructor"`, plus `@strict_exempt :noalloc :_ reason = "entry point: keyword body of the <Type> constructor"` where a constructor has keyword arguments.
  - `Base.showerror` methods and `InputWarning`'s `Base.show` are swept with the module that defines them (measured: `test_compiled(Errors)` and `audit(Errors; sweep = true)` judge `showerror(::IOBuffer, ::InputError)`). Each is written from single-argument `write`/`print` calls, not one varargs `print(io, a, b, c, …)`, prints the message through `Errors._print_message` (Messages), and is exempted `@strict_exempt :noalloc reason = "entry point: renders the error message"`. Measured: with the `LazyString` field, `print(io, e.msg)` fails typestable and noboxing; the parts walk passes typestable and TrimCheck, but reading `parts::Tuple` at a runtime index still boxes (`tuple.jl:33`). So `_print_message`, every `showerror` and `InputWarning`'s `show` are exempted by name from `:noalloc` and `:noboxing`, reason "entry point: renders a LazyString message, whose parts are an abstract Tuple". They run only on the error path. While every function in `Errors` is an exempted entry point, the noalloc scan judges nothing; its count assertion is `@test_broken` until Task 2 sweeps `Units`, which turns it back into `@test`.
  - The methods `@enum` writes into the defining module (`Base.Enums.namemap`, `Base.Enums._enum_hash`, `instances`, `typemin`, `typemax`; measured on Julia 1.13.4) are skipped by `other_function` in `test/strict/check.jl`, with a comment saying that `@enum` writes them.
  - The items "StrictMode policy holds in QuantumEngine" and "StrictMode gates reject the failing control" run `test/strict/check.jl` in a fresh process after `test/strict/workload.jl`.
  - Each task adds its entry points to the workload, so that every exempted function has a compiled specialization.
  - If a sweep that a gate requires to be non-empty comes back empty, stop and report; never drop the assertion.
- **Gates.**
  - Every test is a `@testitem`, with shared setup in `@testmodule`s.
  - Value items compare with oracles; `OffsetArray`/`view` items check generic indexing; error items use `@test_throws "message"`.
  - Aqua (all sub-checks), ExplicitImports, the binary-free closure, `test_layers` plus the public-read check, TrimCheck `@validate` over the exported functions (Float64 and Float32 where generic), and `@test_noalloc` on the allocation-free kernels.
  - The juliac item (tag `:juliac`, `QUANTUMENGINE_JULIAC_BUILD=1`), per-file coverage rows (the ratchet rule), and the mutation runner.
- **Tolerance policy.** "Every numeric comparison in a test states its oracle and a tolerance justified by exactly one of: (a) the oracle's own stated tolerance …; (b) a rounding bound `c · eps(T) · N` …; (c) a measured margin … at most 100 times that." Use `==` where the result is exact by construction. A tolerance is never raised to make a test pass.
- **Parity tolerances (spec 01).**
  - HF and MP2: 1e-9 Ha.
  - UHF: 1e-7.
  - DFT closed shell: the measured angular-table defect + 1e-8. DFT open shell: the measured defect + 2e-6. Both DFT rows are added by step 05.
  - Geometries: 1e-4 Å. Frequencies: 1 cm⁻¹.
- **Run command.**
  - Full suite: `julia --check-bounds=yes --project=QuantumEngine -e 'using Pkg; Pkg.test()'`.
  - Selected items: prefix the full-suite command with `QUANTUMENGINE_ONLY=<stem>`, where `<stem>` is a test file's base name without `.jl` (Task 1 adds this filter; it matches `basename(ti.filename)` exactly, so `inp_tests` does not select `inp_fixture_tests`).
- **Oracle input files.** Tests read the vendored oracle inputs (Task 1b) through the Fixtures helper `oracle_input(relpath)::String`, which reads `joinpath(pkgdir(QuantumEngine), "test", "oracle", "inputs", relpath)`. No test reads QUANTUM_ENGINE_ORACLE_SRC except the vendored-files check of Task 1b, tagged `:oracle_src`, which runs only when that variable is non-empty.
- **Comparing parsed structs.** `XYZFrame`, `InputFile` and the other structs with `Vector` fields have no `==` method (the default falls back to `===`). Tests compare them with the Fixtures helper `fields_equal(a, b) = all(f -> getfield(a, f) == getfield(b, f), fieldnames(typeof(a)))`, applied recursively where a field is itself such a struct.
- **Test item names.** Every `@testitem` a mutation `expect` or a `QUANTUMENGINE_ONLY` run refers to has the exact name the task gives.
- **Formatting and commits.**
  - Run `runic -i QuantumEngine` (runic 1.13, `~/.julia/bin/runic`) before every commit.
  - Use the repository's git identity; never change it.
  - Subject ≤ 72 characters (aim for 50). The body is for a reader with only the repository; `Refs #6.` is the last line of its last body paragraph.
  - Then a blank line, and the trailers alone in the final paragraph: `Assisted-by: Claude <model> <noreply@anthropic.com>` and `Claude-Session: <the executing session's URL>`.
  - "No host machine names or local absolute paths in tracked files or commit messages." The oracle is reached only through the environment variables `QUANTUM_ENGINE_ORACLE`, `QUANTUM_ENGINE_ORACLE_BASIS`, `QUANTUM_ENGINE_ORACLE_SRC` and `QUANTUM_ENGINE_ORACLE_COMMIT` (the full hash `capture.jl` records, Task 13).
  - Never push (D20).
- **Scratch and runs.** Julia runs in the foreground with at most 4 threads. Package probes go in throwaway environments outside the repository.


**Step-03 additions.**

- **Oracle code.** The oracle is Apache-2.0 and is read only to learn behavior: no code is copied or translated line by line. Kernels are written from the literature: McMurchie & Davidson, J. Comput. Phys. 26, 218 (1978); Helgaker, Jørgensen & Olsen, *Molecular Electronic-Structure Theory* (2000), §9.2–9.9 (Hermite E and R, one-electron integrals, Boys function); Obara & Saika, J. Chem. Phys. 84, 3963 (1986) and Head-Gordon & Pople, J. Chem. Phys. 89, 5777 (1988) for the D3 test reference only. Each kernel file names its source in one comment line.
- **Error prefix.** Every `ArgumentError` and `DimensionMismatch` message of the module starts with `integrals:`. Task 1 adds `integrals:` to the AGENTS.md line that lists the `ArgumentError` prefixes (a shared file, named in the D20 notice).
- **Interpolated values.** A message interpolates only the part types `Errors._print_message` prints (`String`, `LazyString`, `SubString{String}`, `Symbol`, `Char`, `Bool`, `Int`, `Float64`), so a `Float32` (or any other) value is converted to `Float64`, `Int` or `String` first, as in the step-02 plan's "Basis error messages". Every error item that can be reached with a Float32 argument has a Float32 case (`"boys: errors"`, `"eri: thresholds"`).
- **Angular-momentum limit.** l > 5 is rejected by the `Integrals` constructors, as their first statement and before any call into `Basis` (whose `cartesian_component` throws its own `basis:` message): `ShellPairData`, `MDEngine` (through `ShellPairData`), `OneElectronWorkspace` and `RIContext` (for the auxiliary set) throw `ArgumentError("integrals: l = 6 is above the supported l ≤ 5")` with the actual `l`. The kernels take their shells from these objects, so they need no check of their own.
- **Run before every commit.** Each task's "Run" step means: the task's selected items (`QUANTUMENGINE_ONLY=<stem>`), then the full suite under `--check-bounds=yes` (which includes the StrictMode policy and exemption items, TrimCheck, Aqua/ExplicitImports and `test_layers`), both passing before the commit. The selected items alone never justify a commit.
- **Scratch probes.** A variant built only to measure something (the D4 per-class engine, any timing harness that is not a `bench/` script) lives in the session scratch directory, outside the repository, and is not committed; only its figures go into the commit body and the Evidence appendix.
- **Number types.** `T<:AbstractFloat` on every kernel and entry point (spec), except where Q-DERIVINT loosens it (Task 15) and where A1 narrows `boys!`. Float64 is the default; Float32 flows through the one-electron integrals, the Boys function and RI (the GPU density-fitting path of step 16). TrimCheck validates each generic entry point in Float64 and Float32.
- **Buffers.** Internal kernels that index a preallocated buffer arithmetically (`hermite_E!`, `hermite_R!`, `eri_shell_quartet!`, the packed stores) call `Base.require_one_based_indexing` on it. Public functions that take user matrices (`overlap!`, `jk!`, `ri_coulomb!`, …) honor their axes (`axes(S, 1)`, `firstindex`), and tests run them on `OffsetArray` and `view` inputs and compare with `==` against the `Array` results.
- **No `@inbounds`** is added by any task of this plan. Task 18 measures the end-to-end effect of bounds checks on `jk!` and reports it; an annotation, if the user wants one, is a separate change that meets the AGENTS.md conditions.
- **Thresholds.** `IntegralThresholds` defaults (spec): `schwarz = 1e-12`, `density = 1e-12`, `primitive = 1e-15`, `prefactor = 1e-18`, `exponent_cut = 45`. `RI_METRIC_CUTOFF = 1e-10`. They are named constants or fields, never literals in kernel code.
- **Benchmarks.** Single-thread and 4-thread runs, `-t 4,0 --gcthreads=1,0`, arms interleaved, at least 7 rounds, min and median reported; speed-ups are reported as baseline/new (> 1 means faster) with the baseline named. Results are JSON files under `QuantumEngine/bench/results/` with the Julia version, CPU model string (`Sys.cpu_info()[1].model`), thread counts, BLAS backend and commit; no host name.
- **Slow items.** Items over about 30 s (benzene, the full Boys table check) are tagged `:slow` (run with `QUANTUMENGINE_SLOW=1`); the task that writes one states its measured time in the item's comment.

## Review Focus

These are the inputs most likely to bite a user that no spec requirement exercises. Each has its pinning test in the task named.

1. **h shells (l = 5, the oracle's limit) in every integral type**, and an l = 6 shell. Expected: (hh|hh), (hs|hs) and the one-electron h blocks are finite, satisfy the 8-fold symmetry, and (hs|hs) matches the D3 reference; an l = 6 shell is rejected by `build_basis` in 02, and a `BasisSet` built directly from an l = 6 `Shell` is rejected by `ShellPairData`, `MDEngine` and `OneElectronWorkspace` with `"integrals: l = 6 is above the supported l ≤ 5"`. Owner: Tasks 3, 5 and 7 (and Task 10 for the reference).
2. **Extreme exponent ranges** (an O 1s exponent of 1.0e5 with a diffuse 0.01 on another atom; quartets whose Boys argument is exactly 0, 35.99 and 250, built as Task 7 states). Expected: finite, no `NaN`/`Inf`, 8-fold symmetric, and screened quartets return exact zeros. Owner: Task 7.
3. **A point charge or ghost atom at a basis-function center** (Boys argument t = 0 in the nuclear attraction), and a point charge with q = 0. Expected: finite; the ghost contributes nothing (`==` to the molecule without it); q = 0 gives `==` to no charge. Owner: Task 5.
4. **A `JKWorkspace` reused with another engine** (after a geometry step, as 07 will do), and a `PackedERI` or `RIContext` used with matrices of another basis size. Expected: `ArgumentError("integrals: the JKWorkspace belongs to another engine")`, and `DimensionMismatch` naming both sizes; never stale results. Owner: Tasks 8, 9, 12a.
5. **Densities and outputs of the wrong size, and `view`/`OffsetArray` wrappers.** Expected: `DimensionMismatch("integrals: density 1 is 12×12, the basis has 13 functions")`; wrapped inputs give results `==` to `Array` inputs, with output axes those of the output argument. Owner: Task 8.

---

## File structure

```
QuantumEngine/
  src/integrals/Integrals.jl        module file: exports, constants, @strict_policy last
  src/integrals/boys.jl             boys!, boys, BOYS_T_LARGE, BOYS_M_MAX
  src/integrals/boys_table.jl       the generated F_m(i/16) table (hex literals)
  src/integrals/special.jl          submodule Integrals.Special: erf, erfc (Cody rational approximations)
  src/integrals/hermite.jl          hermite_E! (1D Hermite expansion coefficients)
  src/integrals/hermite_r.jl        hermite_index, hermite_R_length, hermite_R!
  src/integrals/pairs.jl            ShellPairData, pair_index
  src/integrals/one_electron.jl     OneElectronWorkspace, overlap, kinetic, nuclear_attraction, core_hamiltonian
  src/integrals/properties.jl       dipole, angular_momentum, esp_matrix
  src/integrals/eri.jl              IntegralThresholds, ERIWorkspace, MDEngine, eri_shell_quartet!, schwarz_bounds
  src/integrals/jk.jl               AbstractIntegralEngine contract, JKWorkspace, jk!, coulomb!
  src/integrals/workers.jl          the chunk worker pool (D6)
  src/integrals/packed.jl           PackedERI, jk!(J, K, store, densities)
  src/integrals/ri.jl               two_center, three_center!, RIContext*, ri_coulomb!, ri_exchange!
  src/integrals/mo.jl               mo_eri_ovov, mo_eri_oovv, ri_mo_tensor
  src/integrals/derivatives.jl      first-derivative integrals, ERIGradientWorkspace (Task 14)
  src/integrals/second_derivatives.jl   Taylor-number second derivatives, HessianWorkspace (Task 15)
  gen/boys_table.jl                 generator of src/integrals/boys_table.jl
  gen/Project.toml                  the generator's project (Printf)
  bench/jk_bench.jl  bench/quartet_bench.jl  bench/results/*.json
  test/{boys,special,hermite,one_electron,properties,eri,jk,packed,references,jk_threads,ri,mo,derivatives,second_derivatives,oracle_boys}_tests.jl   (disposition_tests.jl only under Q-ORACLESRC's alternative)
  test/reference/{boys_bigfloat.jl,os_reference.jl,oracle_boys.jl}   test-only references (@testmodule)
  docs/src/integrals.md             conventions, thresholds, parity notes, the disposition table (created by Task 1)
```

Modified: `src/QuantumEngine.jl` (include, re-export, `public Integrals`), `test/layers_tests.jl` (Tasks 1, 3, 5, 12a), `test/strict/check.jl` (the sweep tuple gains `Integrals`), `test/strict/workload.jl`, `test/trim_tests.jl`, `test/fixtures.jl`, `test/coverage/thresholds.toml`, `test/mutation/mutations.toml`, `docs/src/api.md`, `docs/make.jl`, `docs/src/dependencies.md`, `Project.toml` (TypeContracts in Task 8; PureTaylor in Task 15 under the Q-DERIVINT default), `AGENTS.md` (module list and the `integrals:` prefix; shared).

---

### Task 1: The `Integrals` module and the Boys function

**Files:**
- Create: `src/integrals/Integrals.jl`, `src/integrals/boys.jl`, `src/integrals/boys_table.jl`, `gen/boys_table.jl`, `gen/Project.toml` (`[deps] Printf`, `[compat] Printf = "1.13"`, `julia = "1.13"`), `test/boys_tests.jl`, `test/reference/boys_bigfloat.jl`, `docs/src/integrals.md` (the Boys conventions; later tasks add sections).
- Modify: `docs/make.jl` (the new page), `src/QuantumEngine.jl`, `test/layers_tests.jl` (`QE.Integrals => (…)` with the modules Task 1's code uses, the public-read and `test_calls` loops), `test/strict/check.jl`, `test/strict/workload.jl`, `test/trim_tests.jl`, coverage, mutations, `AGENTS.md` (line 14 module list; the `integrals:` prefix).

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (exported from `Integrals`, re-exported by `QuantumEngine`):
  - `const BOYS_T_LARGE = 36`, `const BOYS_M_MAX = 22` (Q-LMAX, decided).
  - `boys!(F::AbstractVector{T}, nmax::Int, t::T) where {T<:Union{Float32,Float64}} -> F`, with `F[begin + m] = F_m(t)` for `m = 0:nmax`.
  - `boys(m::Int, t::T)::T`.
  - Errors: `nmax > BOYS_M_MAX` → `ArgumentError("integrals: Boys order 23 is above BOYS_M_MAX = 22")`; `t < 0` or not finite → `ArgumentError("integrals: Boys argument must be finite and ≥ 0, got -1.0")`; `length(F) < nmax + 1` → `DimensionMismatch("integrals: Boys scratch has length 4, order 8 needs 9")`; other `T` → A1's message, from the fallback methods `boys!(F::AbstractVector, nmax::Int, t::Real)` and `boys(m::Int, t::Real)`. A Float32 `t` is converted to `Float64` before it is interpolated (Global Constraints, "Interpolated values").
- Test module `BoysReference` (`test/reference/boys_bigfloat.jl`): `boys_big(m::Int, t::BigFloat)::BigFloat`, the series `Σ_k (−t)^k / (k!(2m+2k+1))`, summed until a term is below `2^-1100` of the partial sum, at the precision `max(1024, ceil(Int, 1.5 · t · log2(e)) + 1200)` bits (the series cancels about `t·log2(e)` bits; measured with a fixed 4096 bits, scratch `qe-plans/p03_boysbig_hi.jl`: relative error 0 at t = 1000, 9e-153 at 2500, 2.8e65 at 3000, Inf at 4000 and 20000). It throws `ArgumentError` for `t > 5e4`, so a reference outside the range it was checked on fails loudly; the D3 Obara–Saika reference (Task 10) calls it with every geometry's arguments. `ulp_error(x::Float64, ref::BigFloat)::Float64 = abs(x − ref) / eps(Float64(ref))`.

- [ ] **Step 1: Write the generator** `gen/boys_table.jl` (run with `julia --project=QuantumEngine/gen QuantumEngine/gen/boys_table.jl [outfile]`; the project holds Printf only, so `@sprintf` does not enter the package closure): for i = 0:576 and m = 0:(BOYS_M_MAX + 7), `F_m(i/16)` by the 1024-bit series (`F_m(0) = 1/(2m+1)` at i = 0), rounded to nearest Float64, written as a `const BOYS_TABLE = (…)::NTuple{577, NTuple{30, Float64}}` of hex literals (`repr` of a Float64 in hex: `@sprintf("%a", x)`) with a header comment naming the generator and the formula. Write the output to `src/integrals/boys_table.jl` (committed in Step 7).
- [ ] **Step 2: Write the failing tests** in `test/boys_tests.jl`. Items:
  - `"boys: values against BigFloat"`: for `nmax ∈ (0, 4, 8, 16, BOYS_M_MAX)` and `t ∈ (0, 1e-12, 1e-10, 1/32, 0.1, 1, 5, 14.9, 15.1, 30, 35.97, 36, 36.03, 59.9, 60.1, 200)` plus 200 values from `rand(MersenneTwister(3), 200) .* 40`, every `m ≤ nmax` satisfies `ulp_error(F[begin+m], boys_big(m, big(t))) ≤ 2 + s`, with `s = nmax − m` for t < 36 and `s = m` for t ≥ 36. Comment: class (b), "2 ulp for the table value and the Taylor sum, plus at most 1 ulp per neutrally stable step" (spec); measured worst 6.6 ulp (s = 16) and 6.7 ulp (s = 13) (E3).
  - `"boys: every table entry"` (tag `:slow` if Step 4 measures it above 30 s): each `BOYS_TABLE[i+1][m+1]` is within ½ ulp of `boys_big(m, big(i)/16)`, and for every stored n ≤ BOYS_M_MAX the truncation bound `(1/32)^8/8! · F_{n+8}(t₀)/F_n(t₀) ≤ eps(Float64)/4` holds at every node (in BigFloat).
  - `"boys: the generator reproduces the table"` (tag `:slow`): runs the generator in a child process (`julia --project=<pkgdir>/gen <pkgdir>/gen/boys_table.jl <tmpfile>`) and compares bytes with `src/integrals/boys_table.jl` (`==`).
  - `"boys: Float32"`: `boys!(F32, 16, 2.5f0)` equals `Float32.(boys!(F64, 16, 2.5))` (`==`, A1: Float32 rounds the Float64 result).
  - `"boys: errors"`: the four messages of Interfaces, with `@test_throws "message"`; the argument message also for a Float32 `t = -1f0` (it prints `-1.0`), and A1's message for `boys!(zeros(BigFloat, 3), 2, big(1.0))`.
  - `"boys: references are accurate at large t"`: `boys_big(0, big(t))` against `½√(π/t)` at t = 3000 and 20000 (relative error below `2^-1000`, class (b): `erfc(√t)` is below `2^-4000` there and the working precision leaves at least 1,200 bits).
  - `"boys: generic indexing"`: `boys!(OffsetArray(zeros(17), -8:8), 16, 3.0)` gives `parent(...) == boys!(zeros(17), 16, 3.0)`.
  - `"boys: allocation-free"`: `@test_noalloc boys!(F, 16, 3.0)` in Float64 and Float32.
- [ ] **Step 3: Run** `QUANTUMENGINE_ONLY=boys_tests julia --check-bounds=yes --project=QuantumEngine -e 'using Pkg; Pkg.test()'`. Expected: FAIL (`boys!` not defined).
- [ ] **Step 4: Implement** `boys.jl` and the module file. t < 36: nearest node `i = round(Int, 16t)`, `δ = t − i/16`, `F_nmax = Σ_{k=0}^{7} (−δ)^k/k! · TABLE[i+1][nmax+k+1]` by Horner, then downward `F_m = (2t F_{m+1} + e^{−t})/(2m+1)`. t ≥ 36: `F_0 = ½√(π/t)`, upward `F_m = ((2m−1)F_{m−1} − e^{−t})/(2t)`. Module file: `using StrictMode: @strict_policy` (plus `@strict_exempt` when the first exemption appears), `include`s, `export`s, and the policy last after the comment line of the Global Constraints. Register the module in `QuantumEngine.jl` (`include("integrals/Integrals.jl")`, re-export, `public Integrals`).
- [ ] **Step 5: Run** the boys items, then the full suite (Global Constraints, "Run before every commit"). Expected: PASS. Record the time of `"boys: every table entry"` in its comment and the change in `@elapsed using QuantumEngine` (fresh process, 5 runs, min, before and after the table file) in the commit body.
  - Workload: `boys!` in Float64 and Float32. TrimCheck: `boys!(::Vector{Float64}, ::Int, ::Float64)` and the Float32 form. Coverage row for `boys.jl` (the table file is data; its row is whatever the run measures).
- [ ] **Step 6: Mutations:** `(−δ)^k` → `δ^k` (`expect = "boys: values against BigFloat"`); `2t * F` → `t * F` in the downward step (same expect); `(2m − 1)` → `(2m + 1)` in the upward step (same expect).
- [ ] **Step 7: Commit** `Add the Integrals module and the Boys function`.

### Task 2: `erf` and `erfc` in pure Julia

**Blocked by:** Q-ORACLEDEAD, Q-ERFACC.

**Files:** Create `src/integrals/special.jl` (the submodule `Integrals.Special`, with its own `@strict_policy` last), `test/special_tests.jl`. Modify `test/strict/check.jl` (the sweep tuple gains `Integrals.Special`), `test/layers_tests.jl` (the public-read and `test_calls` loops gain it), workload, trim, coverage, mutations.

**Interfaces:**
- Produces: `Special.erf(x::T)::T` and `Special.erfc(x::T)::T` for `T ∈ (Float32, Float64)`, public in `Special` (`public erf, erfc`, not exported from `Integrals` or `QuantumEngine`: the names clash with SpecialFunctions in user code). Float32 evaluates in Float64 and rounds.
- Consumers: Task 16 (`erf` for the oracle-Boys variant); step 15's `pw/ewald` (`erfc`).

- [ ] **Step 1: Write the failing tests.** `"erf and erfc against BigFloat"`: on a grid of 2,001 points over [−6, 6] plus 500 points over [6, 26.5] for `erfc`, compare with a reference written in the test: the Maclaurin series of `erf` at 4096 bits, and `erfc = 1 − erf` in the same precision (at x = 26.5 the series cancels about 1,013 bits and `erfc ≈ 1e-307` needs about 1,020 more, so 4096 bits leave about 2,000). Assert `ulp_error ≤ 1` for both functions (Q-ERFACC: the spec's "≤ 1 ulp class", a fixed bound, not one set from the measurement). `"erf: limits"`: `erf(0.0) == 0.0`, `erf(-x) == -erf(x)` on the grid, `erfc(Inf) == 0.0`, `erf(NaN)` is NaN. `"erf: allocation-free"`.
- [ ] **Step 2: Run** `QUANTUMENGINE_ONLY=special_tests`. Expected: FAIL.
- [ ] **Step 3: Implement** Cody's rational Chebyshev approximations (W. J. Cody, Math. Comp. 23, 631 (1969); the coefficient sets of the public-domain netlib SPECFUN routine `CALERF`, attributed in a comment) over |x| < 0.5, 0.5 ≤ |x| < 4 and |x| ≥ 4, with the `exp(−x²)` split for accuracy.
- [ ] **Step 4: Run** the items, then the full suite; record the measured worst ulp error for each function in the test comments. Expected: PASS. If either function exceeds 1 ulp anywhere on the grids, stop and ask the user (Q-ERFACC); the bound is not raised. Workload, TrimCheck (Float64, Float32), coverage.
- [ ] **Step 5: Mutation:** one coefficient sign in the middle interval of `erfc` (`expect = "erf and erfc against BigFloat"`).
- [ ] **Step 6: Commit** `Add erf and erfc`.

### Task 3: Hermite expansion coefficients and shell-pair data

**Files:** Create `src/integrals/hermite.jl`, `src/integrals/pairs.jl`, `test/hermite_tests.jl`. Modify `test/fixtures.jl` (A11's `water_eri()`), `test/layers_tests.jl` (the `Integrals` row gains `Basis`), workload, trim, coverage, mutations.

**Interfaces:**
- Consumes (step 02, `QuantumEngine.Basis`): `BasisSet{T}` (`shells`, `nao`), `Shell{T}` (`center`, `l`, `exponents`, `coefficients`), `ncart(l)`, `cartesian_component(l, i)`, `primitive_norm(α, lx, ly, lz)`, `ao_range(shell)`, `shell_pairs(bs)`.
- Produces:
  - `hermite_E!(E::AbstractArray{T,3}, la::Int, lb::Int, a::T, b::T, XAB::T, E00::T = exp(−a*b/(a+b) * XAB^2)) -> E`: one Cartesian axis; `E[i+1, j+1, t+1] = E_t^{ij}` for `0 ≤ i ≤ la`, `0 ≤ j ≤ lb`, `0 ≤ t ≤ i+j`, others zero; the recursion starts from `E_0^{00} = E00`, and every entry is linear in it, so the default gives the full Gaussian-product coefficients and `E00 = one(T)` gives them divided by `exp(−ab/(a+b) · XAB²)`; `XAB = A_x − B_x`. Requires one-based indexing and `size(E) ⊇ (la+1, lb+1, la+lb+1)`; otherwise `DimensionMismatch("integrals: hermite_E! needs a 4×3×6 array, got 3×3×6")` (sizes as `Int`s).
  - **Prefactor convention (one place only).** The stored E tables of `ShellPairData` are built with `E00 = one(T)` on every axis, so their product over the three axes carries no exponential. The Gaussian-product exponential `exp(−exparg)` lives only in `Kab`. Every formula below (overlap, kinetic, nuclear attraction, ESP, ERIs, RI) uses the stored tables with `Kab`, never the default-`E00` form. The default form exists for the product-identity test and for callers outside `ShellPairData`.
  - `pair_index(i::Int, j::Int)::Int = i*(i−1)÷2 + j` for `i ≥ j` (the order of `shell_pairs`).
  - `ShellPairData{T}` with the spec's fields `zeta`, `P`, `Kab`, `E`, `E_offset`, `prim_range`, `Q`, plus `exparg::Vector{T}` (`ab/ζ · R_AB²` per primitive pair), `comp_scale::Vector{T}` (per AO, A4's α-independent ratio) and `deriv_order::Int`. `Kab` holds `exp(−exparg) · c_a c_b · primitive_norm(α_a, l_a, 0, 0) · primitive_norm(α_b, l_b, 0, 0)` (A4). The stored E tables run to `la + deriv_order` and `lb + deriv_order` on each axis, `(la+d+1)(lb+d+1)(la+lb+2d+1)` entries per axis with `d = deriv_order`, so Task 14's shift relations read `l + 1` from the same tables. `prim_range[pair_index(i, j)]` indexes the primitive pairs of shell pair (i, j). `Q` is allocated `zeros(T, nshell, nshell)` here and filled by `MDEngine` (Task 7).
  - `ShellPairData(bs::BasisSet{T}; deriv_order::Int = 0) where {T}`, `deriv_order ∈ 0:1` (else `ArgumentError("integrals: deriv_order must be 0 or 1, got 2")`); its first statement is the angular-momentum check (Global Constraints). Exempted `@strict_exempt :noalloc ShellPairData reason = "entry point: shell-pair data constructor"` and `@strict_exempt :noalloc :_ reason = "entry point: keyword body of the ShellPairData constructor"`.

- [ ] **Step 1: Write the failing tests.**
  - `"hermite E: product identity"`: in BigFloat (`setprecision(256)`), for `la, lb ∈ 0:5`, exponents (0.3, 1.7) and XAB = 0.9: at 7 points x, `(x−A)^i (x−B)^j exp(−a(x−A)² − b(x−B)²) = Σ_t E_t^{ij} Λ_t(x)` with `Λ_t(x) = ∂^t/∂P^t exp(−p(x−P)²)` evaluated by the Hermite-polynomial formula `Λ_t = (√p)^t H_t(√p (x−P)) exp(−p(x−P)²)` written in the test. Assert `abs(lhs − rhs) ≤ 2.0^-200 · max(1, abs(lhs))` (class (b): 256-bit arithmetic, at most ~50 terms of magnitude ≤ 1e6; the comment states the count).
  - `"hermite E: Float64 rounding"`: Float64 `hermite_E!` against the BigFloat call of the same function; per entry `|ΔE| ≤ 4 · eps · (la + lb + 1) · S_abs`, with `S_abs` the BigFloat recursion run on absolute values (written in the test), class (b).
  - `"pair data: component scale is exponent-independent"`: for l = 0:5 and each component, `primitive_norm(0.3, lx, ly, lz)/primitive_norm(0.3, l, 0, 0)` and the same at α = 41.0 agree within `4 eps` relative (class (b): two norms of ≤ 4 roundings each).
  - `"pair data: layout"`: on water/STO-3G (`build_basis(Fixtures.water_eri(), "sto-3g")`), the number of primitive pairs is `Σ_{i≥j} K_i K_j` (`==`), `prim_range` covers `1:npp` contiguously in `shell_pairs` order (`==`), and `E_offset` lengths equal `3 · (la+1)(lb+1)(la+lb+1)` per primitive pair (`==`), and `3 · (la+2)(lb+2)(la+lb+3)` with `deriv_order = 1` (`==`).
  - `"pair data: the exponential is in Kab only"` (the prefactor convention): for every primitive pair of water/STO-3G, the stored `E_0^{00}` of each axis is `== 1`, and the product `Kab · E^x_0 E^y_0 E^z_0 · (π/ζ)^{3/2}` for two s primitives on different centers (O 1s with H 1s) equals the closed-form `S_ab` of Task 5's s-s item within `16 eps` relative (class (b)).
  - `"hermite E: errors"`: an `E` of size `3×3×6` for `la = 3, lb = 2` throws `"integrals: hermite_E! needs a 4×3×6 array, got 3×3×6"`.
  - `"pair data: errors"`: `ShellPairData` on a `BasisSet` built from a `Shell` with l = 6 (the `Shell` constructor accepts it, step 02) throws `"integrals: l = 6 is above the supported l ≤ 5"`; `deriv_order = 2` throws its message.
  - `"hermite E: allocation-free"`: `@test_noalloc hermite_E!(E, 3, 2, 0.5, 1.2, 0.7)` and the form with `E00 = 1.0`.
- [ ] **Step 2: Run** `QUANTUMENGINE_ONLY=hermite_tests`. Expected: FAIL.
- [ ] **Step 3: Implement** `hermite_E!` by the three-term recursions (Helgaker et al. eqs. 9.5.6–9.5.7): `E_t^{i+1,j} = E_{t−1}^{ij}/(2p) − (b/p) XAB E_t^{ij} + (t+1) E_{t+1}^{ij}` and its `j+1` partner with `+(a/p) XAB`. `ShellPairData` loops over `shell_pairs(bs)` and stores the three axis tables per primitive pair contiguously.
- [ ] **Step 4: Run.** Expected: PASS. Workload (`ShellPairData`, `hermite_E!`), TrimCheck (`hermite_E!` Float64 and Float32; `ShellPairData(::BasisSet{Float64})` and `ShellPairData(::BasisSet{Float32})`), coverage rows for both files.
- [ ] **Step 5: Mutations:** `/(2p)` → `/p` (`expect = "hermite E: product identity"`); the sign of `(b/p) XAB` (same expect).
- [ ] **Step 6: Commit** `Add Hermite expansion coefficients and shell-pair data`.

### Task 4: The Hermite Coulomb table

**Files:** Create `src/integrals/hermite_r.jl`, test items in `test/hermite_tests.jl`. Modify workload, trim, coverage, mutations.

**Interfaces:**
- Consumes: `boys!` (Task 1).
- Produces:
  - `hermite_index(t::Int, u::Int, v::Int)::Int`: packed slot of `(t, u, v)`, ordered by `n = t+u+v`, then `t` descending, then `u` descending: `binomial(n + 2, 3) + (n − t)(n − t + 1) ÷ 2 + (n − t − u) + 1`.
  - `hermite_R_length(L::Int)::Int = binomial(L + 4, 4)`: the scratch for all orders m = 0:L (`binomial(L+3, 3)` slots hold the m = 0 result).
  - `hermite_R!(R::AbstractVector{T}, F::AbstractVector{T}, L::Int, p::T, PC::SVector{3,T}, boys::B = boys!) -> R`: after the call `R[hermite_index(t,u,v)] = R^0_{tuv}(p, PC)` for `t+u+v ≤ L`; `F` is Boys scratch of length ≥ L+1, filled by `boys(F, L, p * sum(abs2, PC))` (A14: the engine passes its own Boys function). Errors: `length(R) < hermite_R_length(L)` → `DimensionMismatch("integrals: R scratch has length 100, L = 8 needs 495")`; `length(F) < L + 1` → `DimensionMismatch("integrals: Boys scratch has length 4, L = 8 needs 9")`; `L > BOYS_M_MAX` → `ArgumentError("integrals: Hermite order L = 23 is above BOYS_M_MAX = 22")`.

- [ ] **Step 1: Write the failing tests.**
  - `"hermite R: closed form"`: in a test helper, `R_big(t,u,v,p,PC)` = `Σ_{a,b,c} t!/(a!(t−2a)!) · u!/(b!(u−2b)!) · v!/(c!(v−2c)!) · (2X)^{t−2a} (2Y)^{u−2b} (2Z)^{v−2c} · (−p)^{n} F_n(p|PC|²)` with `n = t+u+v−a−b−c` and `F_n` from `boys_big` (the chain rule for a function of `|PC|²`, independent of the recursion). For L = 8, p = 0.8, PC = (0.3, −0.5, 1.1) and PC = 0: every entry of Float64 `hermite_R!` satisfies `|R − R_big| ≤ 8 · eps · (L + 1) · R_abs`, with `R_abs` the same sum of absolute terms (class (b): L+1 recursion levels of 2-term updates plus the Boys error ≤ 9 ulp).
  - `"hermite R: index is a bijection"`: for L = 0:22, `sort([hermite_index(t,u,v) for t+u+v ≤ L]) == 1:binomial(L+3, 3)`.
  - `"hermite R: errors"`: the three messages of Interfaces, with `@test_throws "message"`.
  - `"hermite R: the Boys argument is used"`: with a test-local Boys function that writes `F_m = 2 · boys_m` the m = 0 block is `==` twice the default result (the scaling by a power of two is exact).
  - `"hermite R: allocation-free"`: `@test_noalloc` for L = 8 and L = 20.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** the recursion `R^m_{t+1,u,v} = t R^{m+1}_{t−1,u,v} + X R^{m+1}_{tuv}` (and the u, v forms), starting from `R^m_{000} = (−2p)^m F_m(p|PC|²)`, sweeping m from L down to 0 in the packed buffer (order-m results at offset `binomial(L−m+4,4) − binomial(L−m+3,3)` or any layout that keeps the m = 0 block at the front; the test only reads the m = 0 block).

  Note on the closed form: Helgaker's `R^n_{000} = (−2p)^n F_n`, and the chain-rule form above uses `(−p)^n` with factors `2X`; both give the same `R^0_{tuv}` (the test is the check).
- [ ] **Step 4: Run.** Expected: PASS. Workload, TrimCheck, coverage.
- [ ] **Step 5: Mutation:** `t * R[...]` → `(t + 1) * R[...]` (`expect = "hermite R: closed form"`).
- [ ] **Step 6: Commit** `Add the Hermite Coulomb table`.

### Task 5: Overlap, kinetic, nuclear attraction and the core Hamiltonian

**Files:** Create `src/integrals/one_electron.jl`, `test/one_electron_tests.jl`. Modify `test/fixtures.jl` (A11's `benzene()`, `h2o_sto3g_jk_density()`), `test/layers_tests.jl` (the `Integrals` row gains `System`), workload, trim, coverage, mutations.

**Interfaces:**
- Consumes: Tasks 1, 3, 4; step 01 `Molecule{T}` (`atoms`, `point_charges`), `Atom` (`Z`, `position`, `ghost`), `PointCharge` (`charge`, `position`); step 02 `build_basis(mol, name)`, `ao_range`.
- Produces:
  - `OneElectronWorkspace(bs::BasisSet{T}; boys = boys!) -> OneElectronWorkspace{T,typeof(boys)}` (E tables for `l_max + 2`, R table and Boys scratch for `L = 2 l_max + 2`; A14's Boys function); its first statement is the angular-momentum check. Exemptions: `@strict_exempt :noalloc OneElectronWorkspace reason = "entry point: one-electron workspace constructor"` and `@strict_exempt :noalloc :_ reason = "entry point: keyword body of the OneElectronWorkspace constructor"`.
  - `overlap(bs)`, `overlap!(S, bs[, ws])`; `kinetic(bs)`, `kinetic!(T, bs[, ws])`; `nuclear_attraction(bs, mol)`, `nuclear_attraction!(V, bs, mol[, ws])` (nuclei with `Z > 0`, ghosts skipped, plus every point charge with charge q); `core_hamiltonian(bs, mol)` → `T + V`. The non-`!` forms and the two-argument `!` forms are exempted by name as entry points (A13). Output matrices are dense `nao × nao`, filled for `i ≥ j` and mirrored, and indexed through their own axes.
  - Docstring of `core_hamiltonian`: ECP terms are absent (step 14) (A10).

- [ ] **Step 1: Write the failing tests.**
  - `"one-electron: H2 overlap literal"`: `overlap(build_basis(Fixtures.h2(), "sto-3g"))[1, 2]` within 1e-6 of 0.65931812. Class (a), the oracle's ±1e-6 (`test_gradient_h2.rs:223-234`). Prototype value 0.65931821 (E1).
  - `"one-electron: normalization"`: `diag(S)` for `build_basis(<a single O atom>, "cc-pvqz")` (s to g shells, step 02 ships them) and for a hand-built h shell (l = 5, three primitives, renormalized with `contraction_renorm`): `|S_μμ − 1| ≤ 4(l+1) · eps · K²` per shell (spec, class (b)). The h shell also gives finite, symmetric T and V blocks (Review Focus 1).
  - `"one-electron: s-s closed forms"`: for two normalized s primitives (α = 0.8, β = 1.3, R = 1.1), `S = (2√(αβ)/(α+β))^{3/2} exp(−μR²)` and `T = μ(3 − 2μR²) S` with `μ = αβ/(α+β)`; the nuclear attraction of a unit charge at C: `−2π/p · N_a N_b exp(−μR²) F_0(p|PC|²)` with `F_0` from `boys_big`. Each within `16 eps` relative (class (b): ≤ 16 roundings, written in the comment).
  - `"one-electron: symmetry"`: `S == S'`, `T == T'`, `V == V'` (mirrored by construction).
  - `"one-electron: ghosts and point charges"` (Review Focus 3): adding a ghost O at an H position leaves `V` unchanged on the shared AO block (`==`); a point charge with q = 0 gives `==` to none; a point charge of q = +1 at the position of H1 gives `V(mol + charge)` within `8 eps · nprim² · max|V|` (class (b)) of `2 V(mol) − V(mol with H1 a ghost)`, all on the basis of `mol` (a ghost carries the basis functions of the element it stands for, `System.basis_element`, so `nao` is the same in all three); all entries finite.
  - `"one-electron: Float32"`: `overlap(build_basis(…, Float32))` is a `Matrix{Float32}` within `8 eps(Float32) · K²` of the Float64 matrix.
  - `"one-electron: generic indexing"`: `overlap!(OffsetArray(zeros(n, n), 0:n-1, 0:n-1), bs)` has `parent == overlap(bs)`; the same with a `view` of a larger matrix; `nuclear_attraction` with `Molecule(OffsetArray(atoms, -1:1))` `==` the plain molecule (step 01's `Molecule{T,V}` takes any `AbstractVector{Atom{T}}`, `System.jl:127-140`).
  - `"one-electron: allocation-free"`: `@test_noalloc overlap!(S, bs, ws)`, `kinetic!`, `nuclear_attraction!` in Float64 and Float32.
  - `"one-electron: errors"`: a 12×12 `S` for a 13-function basis throws `"integrals: S is 12×12, the basis has 13 functions"`; `OneElectronWorkspace` on a basis with an l = 6 shell throws `"integrals: l = 6 is above the supported l ≤ 5"`.
- [ ] **Step 2: Run** `QUANTUMENGINE_ONLY=one_electron_tests`. Expected: FAIL.
- [ ] **Step 3: Implement** (A3), with the stored tables of the prefactor convention (Task 3: `E_0^{00} = 1`, the exponential in `Kab`): `S_ab = Kab · E^x_0 E^y_0 E^z_0 · (π/p)^{3/2}` per primitive pair; kinetic by `T_ij = b(2j+1) S_{i,j} − 2b² S_{i,j+2} − ½ j(j−1) S_{i,j−2}` per axis, combined across axes; nuclear attraction `−q · Kab · 2π/p · Σ_{tuv} E^x_t E^y_u E^z_v R^0_{tuv}(p, P − C)` per center C, with `R` from `hermite_R!` called with the workspace's Boys function. Component scale (A4) applied once per AO pair.
- [ ] **Step 4: Run.** Expected: PASS. Workload, TrimCheck (Float64 and Float32 for the `ws` forms, the allocating forms and `OneElectronWorkspace(::BasisSet)`), coverage.
- [ ] **Step 5: Mutations:** the `2b²` term sign in the kinetic relation (`expect = "one-electron: s-s closed forms"`); skip the point-charge loop (`expect = "one-electron: ghosts and point charges"`).
- [ ] **Step 6: Commit** `Add overlap, kinetic and nuclear attraction integrals`.

### Task 6: Dipole, angular momentum and electrostatic potential

**Files:** Create `src/integrals/properties.jl`, `test/properties_tests.jl`. Modify workload, trim, coverage, mutations.

**Interfaces:**
- Consumes: Tasks 3–5.
- Produces: `dipole(bs::BasisSet{T}; origin = zero(SVector{3,T})) -> NTuple{3,Matrix{T}}` (symmetrized ½(D + Dᵀ), as `dipole.rs`), `angular_momentum(bs::BasisSet{T}; origin::SVector{3,T}) -> NTuple{3,Matrix{T}}`, `esp_matrix(bs, r::SVector{3}; boys = boys!) -> Matrix{T}` (A14). All three are entry points (exempt by name, plus the `:_` keyword-body exemption for each). They take no user-indexed array (a `BasisSet` and an `SVector` origin or point), so Task 6 has no generic-indexing item: an `OffsetArray` converted to an `SVector` loses its axes before the call (scratch `qe-plans/probe03`), so such an item would test StaticArrays only.

- [ ] **Step 0: Read** `$QUANTUM_ENGINE_ORACLE_SRC/src/integrals/angular_momentum.rs` and `esp.rs` and record, in the docstrings, the sign convention of each: which real matrix `angular_momentum` returns (`(r − O) × ∇` or its negative), and whether `esp_matrix` is `+⟨μ|1/|r_e − r||ν⟩` or its negative. The tests below are written for the convention found; if it is not one of these, stop and report.
- [ ] **Step 1: Write the failing tests.**
  - `"dipole: origin shift"`: `D_k(o′) = D_k(o) − (o′_k − o_k) S` on water/6-31G, `|Δ| ≤ 8 eps · (|D_k| + |o′_k − o_k| |S|)` per entry (class (b)).
  - `"dipole: s-s closed form"`: for two s primitives, `⟨a|x − o_x|b⟩ = (P_x − o_x) S_ab`, within `8 eps` relative.
  - `"angular momentum: one-center p functions"`: for normalized p functions on one center at the origin, `⟨p_x|(r × ∇)_z|p_y⟩ = 1` and `⟨p_y|(r × ∇)_z|p_x⟩ = −1` (the identity follows from integration by parts; `4 eps` per entry), with the sign of the Step 0 convention; `L_k == −L_kᵀ` on water/6-31G within `8 eps · |L|`.
  - `"esp: equals a unit point charge"`: `esp_matrix(bs, r)` equals the attraction matrix of a lone point charge q = 1 at r (the molecule's nuclei removed) up to the Step 0 sign, within `8 eps · nprim²` (class (b)).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** with E tables of `l_b + 1` (dipole: `x_C = x_B + (B − C)`) and the `l ± 1` derivative relation for `∇`.
- [ ] **Step 4: Run.** Expected: PASS. Workload, TrimCheck (Float64, Float32), coverage.
- [ ] **Step 5: Mutation:** drop the `(B − C)` shift term in the dipole (`expect = "dipole: origin shift"`).
- [ ] **Step 6: Commit** `Add dipole, angular momentum and ESP integrals`.

### Task 7: Shell-quartet ERIs and Schwarz bounds

**Blocked by:** D4, A15.

**Files:** Create `src/integrals/eri.jl`, `test/eri_tests.jl`. Modify workload, trim, coverage, mutations.

**Interfaces:**
- Consumes: Tasks 1, 3, 4; `Fixtures.water_eri()`.
- Produces:
  - `IntegralThresholds{T}` with the spec's five fields and default values, built by A15's cascade (no `@kwdef`): `IntegralThresholds(; …)` → `IntegralThresholds{Float64}(; …)`; `IntegralThresholds{T}(; schwarz = 1e-12, density = 1e-12, primitive = 1e-15, prefactor = 1e-18, exponent_cut = 45) where {T}` → the inner `IntegralThresholds{T}(schwarz, density, primitive, prefactor, exponent_cut) where {T}` with unconstrained arguments, which converts through `new` and validates: a negative or non-finite threshold throws `ArgumentError("integrals: threshold schwarz must be finite and ≥ 0, got -1.0")` (the value converted to `Float64` before it is interpolated). Exemptions: `@strict_exempt :noalloc :_ reason = "entry point: keyword body of the IntegralThresholds constructor"` if the sweep reports the keyword body (it allocates nothing, so the exemption is added only if the exemption item needs it).
  - `ERIWorkspace(L::Int, T)`: Boys scratch `L+1`, a packed R buffer `hermite_R_length(L)`, the bra and ket Hermite product blocks and the primitive accumulator; `MDEngine` builds it with `L = 4 l_max + deriv_order` (spec: "L = 4 l_max plus the derivative order"). `@strict_exempt :noalloc ERIWorkspace reason = "entry point: ERI scratch constructor"`.
  - `MDEngine{T,F} <: AbstractIntegralEngine` with fields `pairs::ShellPairData{T}`, `thresholds::IntegralThresholds{T}`, `workspaces::Vector{ERIWorkspace{T}}` (one entry, A5), `boys::F` (A14), and the basis set `basis::BasisSet{T}`. Constructor `MDEngine(bs::BasisSet{T}; thresholds = IntegralThresholds{T}(), boys = boys!, deriv_order::Int = 0)`, which builds `ShellPairData(bs; deriv_order)` first (so an l > 5 shell throws the angular-momentum message before anything else) and fills `pairs.Q`. Exemptions: `@strict_exempt :noalloc MDEngine reason = "entry point: integral engine constructor"` and `@strict_exempt :noalloc :_ reason = "entry point: keyword body of the MDEngine constructor"`.
  - `eri_shell_quartet!(buf::AbstractArray{T,4}, ws::ERIWorkspace{T}, eng::MDEngine{T}, a::Int, b::Int, c::Int, d::Int) -> buf`: `buf[i, j, k, l] = (ab|cd)` over the Cartesian components of shells a, b, c, d in `cartesian_component` order, any shell order (not only a ≥ b). Requires one-based `buf` of size `(ncart(la), ncart(lb), ncart(lc), ncart(ld))` (`DimensionMismatch("integrals: ERI block is 3×3×3×1, the quartet needs 3×3×3×3")`). A primitive quartet is skipped when `exparg_ab + exparg_cd > exponent_cut` or `|pref| < prefactor` (D4); a quartet whose primitive quartets are all skipped gives exact zeros. `thresholds.primitive` is not used here (D4). It uses `eng.boys` through `hermite_R!`.
  - `schwarz_bounds(eng::MDEngine)::Matrix{T}` returns `eng.pairs.Q` (no copy), with `Q[a, b] = Q[b, a] = sqrt(max_{μ∈a, ν∈b} |(μν|μν)|)`.
  - `abstract type AbstractIntegralEngine end` (exported; its contract is written in Task 8 with `jk!`).

- [ ] **Step 1: Write the failing tests.**
  - `"eri: oracle literals"`: on `build_basis(Fixtures.water_eri(), "sto-3g")` with shells 1 = O 1s, 2 = O 2s, 3 = O 2p, 4 = H1 1s, 5 = H2 1s: `(O1s O1s|O1s O1s)` within 1e-6 of 4.78506540, `(O1s O1s|O2s O1s)` of 0.74138035, `(O1s O1s|H1s O1s)` and `(O1s O1s|H2s O1s)` of 0.17176253, all class (a) (`test_eri_compare.rs:80-82`); `(O1s O1s|O2p_x O2p_x)` within 1e-6 of 1.11581381 with the comment "unasserted PySCF literal, printed by `test_eri_simple.rs:6-27`; same value from libcint 6.1.2 and GaussianBasis.jl (E1)" (Review Focus 1's one-center case, t = 0).
  - `"eri: 8-fold symmetry"`: on water/6-31G, for 200 random shell quartets (`MersenneTwister(7)`) and all eight index permutations, `|Δ| ≤ 8 eps · N · S_abs`, with N the number of primitive-quartet products and `S_abs` their absolute sum, both accumulated by a test-local copy of the contraction loop (spec, class (b)).
  - `"eri: Schwarz bounds"`: `schwarz_bounds(eng)[a, b] == sqrt(maximum(abs, diag of (ab|ab) block))` computed in the test from `eri_shell_quartet!` (`==`: same arithmetic), and `Q == Q'`.
  - `"eri: screened quartets are exact zeros"`: two s shells A, B 12 bohr apart, one primitive each with exponent 5.0. The quartet is `(AB|AB)`: the test asserts `exparg_AB + exparg_AB = 2 · (5·5/10) · 144 = 720 > exponent_cut` from `eng.pairs` and then `all(iszero, buf)`. (`(AA|BB)` has `exparg = 0` and is not cut; the item asserts that it is nonzero, so the test exercises the cut and not a zero by accident.)
  - `"eri: h shells and extreme exponents"` (Review Focus 1, 2): a hand-built basis with an h shell (l = 5) on O and an s shell on H: (hh|hh) and (hs|hs) are finite and satisfy the 8-fold check above. A basis with an O s exponent 1.0e5 and an H s exponent 0.01: all quartets finite and symmetric. Boys arguments: single-primitive s shells of exponent 1.0 on centers A and C, quartet `(AA|CC)`, so `p = q = 2`, `α = pq/(p+q) = 1` and `t = |AC|²`; t = 0 with C = A, t = 35.99 with `|AC| = sqrt(35.99)`, t = 250 with `|AC| = sqrt(250)`. The test asserts `abs(α · |PQ|² − t) ≤ 4 eps · t` from the pair data before checking that the block is finite and matches `2π^{5/2}/(pq√(p+q)) · Kab Kcd · F_0(t)` within `16 eps` relative (class (b); `F_0` from `boys_big`).
  - `"eri: l = 6 is rejected"`: `MDEngine` on a `BasisSet` with an l = 6 `Shell` throws `"integrals: l = 6 is above the supported l ≤ 5"`.
  - `"eri: allocation-free"`: `@test_noalloc eri_shell_quartet!(buf, ws, eng, a, b, c, d)` for (pp|pp) on water/6-31G and for (dd|dd) and (ff|ff) on `build_basis(Fixtures.water_eri(), "cc-pvtz")` (O carries d and f shells), in Float64 and Float32.
  - `"eri: thresholds"`: the `ArgumentError` of Interfaces for `IntegralThresholds(schwarz = -1.0)` and `IntegralThresholds{Float32}(schwarz = -1f0)` (both print `-1.0`); `IntegralThresholds(density = 0.0) isa IntegralThresholds{Float64}` and `IntegralThresholds{Float32}().exponent_cut === 45f0`.
- [ ] **Step 2: Run** `QUANTUMENGINE_ONLY=eri_tests`. Expected: FAIL.
- [ ] **Step 3: Implement** the kernel, with the stored E tables of Task 3's prefactor convention (`E_0^{00} = 1`, the exponential in `Kab`): for each bra primitive pair (ab), form `E_ab[h, ij]` (Hermite index h over `t+u+v ≤ la+lb`, ij over component pairs) from the three axis tables; for each ket primitive pair (cd) with the sign `(−1)^{τ+υ+φ}` folded into `E_cd`, compute `R^0` at `α = pq/(p+q)`, `PQ`, `L = la+lb+lc+ld` with `eng.boys`, and accumulate `T[h, kl] += pref · Σ_{h′} R[h + h′] E_cd[h′, kl]` with `pref = 2π^{5/2}/(pq√(p+q)) · Kab Kcd` (the only place the exponentials enter); then `block += E_abᵀ T`. Apply `comp_scale` per component once at the end. `MDEngine` computes Q from the diagonal quartets.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Measure D4.** Build a scratch engine variant (Global Constraints, "Scratch probes": outside the repository, not committed) that applies the exponent and prefactor cuts only to quartets with `la+lb+lc+ld ≥ 2` (the oracle's MD classes) and the oracle's contraction-weight skip (D4), compute every quartet of water/6-31G and benzene/6-31G with both, and record the largest absolute and relative difference in the commit body and in Evidence. If any difference exceeds 1e-14 absolute (about five orders above the estimate `exp(−45) ≈ 2.9e-20` times the largest water STO-3G integral 4.8, i.e. 1.4e-19), stop and report to the user before continuing.
- [ ] **Step 6: Gates.** Workload (`MDEngine`, `eri_shell_quartet!` for (pp|pp) and (dd|dd), Float64 and Float32), TrimCheck (`eri_shell_quartet!(::Array{T,4}, ::ERIWorkspace{T}, ::MDEngine{T,typeof(boys!)}, ::Int, ::Int, ::Int, ::Int)` for T = Float64 and Float32; `MDEngine(::BasisSet{Float64})`, `MDEngine(::BasisSet{Float32})`, `IntegralThresholds{Float64}()`), coverage row.
- [ ] **Step 7: Mutations:** the phase `(−1)^{τ+υ+φ}` removed (`expect = "eri: 8-fold symmetry"`; E2 shows that this fault is what the K literal records); `> exponent_cut` → `< exponent_cut` (`expect = "eri: oracle literals"`); `2π^{5/2}` → `2π^{3/2}` (`expect = "eri: oracle literals"`).
- [ ] **Step 8: Commit** `Add shell-quartet ERIs and Schwarz bounds`.

### Task 8: Integral-direct J and K (serial) and the engine contract

**Files:** Create `src/integrals/jk.jl`, `test/jk_tests.jl`. Modify `Project.toml` (TypeContracts 0.14.7 in `[deps]` and `[compat]`), `docs/src/dependencies.md`, workload, trim, coverage, mutations.

**Interfaces:**
- Consumes: Task 7.
- Produces:
  - `JKWorkspace(eng::MDEngine{T}) -> JKWorkspace{T}`. It keeps `eng` (identity check), one `ERIWorkspace`, the shell-pair density maxima scratch, the quartet block buffer and two counters, `nschwarz::Int` and `ndensity::Int`, the numbers of unique quartets the last `jk!` call skipped by each screen (public fields, documented as diagnostics). Serial `jk!` accumulates straight into `J` and `K`. Task 11 adds the `nworkers` keyword and the per-worker accumulators. `@strict_exempt :noalloc JKWorkspace reason = "entry point: J/K workspace constructor"`.
  - `jk!(J::NTuple{N,AbstractMatrix}, K::NTuple{N,AbstractMatrix}, ws::JKWorkspace, eng::MDEngine, densities::NTuple{N,AbstractMatrix}; want_K::Bool = true) -> nothing`, N ∈ 1:2: overwrites `J[n]` and, when `want_K`, `K[n]` (A12 conventions). Reads only the lower triangle of each density (A9). Quartet order: `a` ascending, `b ≤ a`, `c ≤ a`, `d ≤ (c == a ? b : c)` (E9, `jk_symmetric.rs:68-71`). Screening: `Q_ab Q_cd < schwarz` skip; `Q_ab Q_cd P_max < density` skip, `P_max` the largest of the six shell-pair density maxima (`P_ab, P_cd, P_ac, P_ad, P_bc, P_bd`) over all densities; integrals with `|I| < primitive` not scattered (D4).
  - `coulomb!(J, ws, eng, P)` = `jk!((J,), (K_unused,), …; want_K = false)` without needing a K argument.
  - In `jk.jl`: `@contract AbstractIntegralEngine "Two-electron integrals over a basis set." begin … end` requiring `jk!(::NTuple, ::NTuple, ::JKWorkspace, ::Self, ::NTuple)::Nothing`, `eri_shell_quartet!(::AbstractArray, ::ERIWorkspace, ::Self, ::Int, ::Int, ::Int, ::Int)::AbstractArray` and `schwarz_bounds(::Self)::AbstractMatrix`, and `@verify MDEngine{Float64,typeof(boys!)} trim_compat = true` (the `PureAtomistic/src/potentials` pattern; import only `@contract, @verify, Self`). The contract is McMurchie–Davidson-shaped for now: `ERIWorkspace` and `JKWorkspace` are MD scratch, and the docstring says so. A second engine (Rys or HGP, the spec's mitigation for large l) changes the contract to `workspace(eng)` and `jk_workspace(eng; …)` with loosely typed workspace arguments, in the task that adds it.

- [ ] **Step 0: Resolve the closure** of TypeContracts 0.14.7 in a throwaway environment outside the repository (as step-01 Task 3 Step 0) and record every package with its version in the commit body and `docs/src/dependencies.md`. Stop only on a non-stdlib `_jll` or an `Artifacts.toml`.
- [ ] **Step 1: Write the failing tests.**
  - `"jk: matches the explicit sum"`: on water/6-31G with a random symmetric density (`MersenneTwister(11)`) and `IntegralThresholds{Float64}(schwarz = 0.0, density = 0.0, primitive = 0.0)` (every screen of `jk!` off; the exponent and prefactor cuts act inside `eri_shell_quartet!`, which the explicit sum calls too), J and K against the explicit four-index sums built in the test from `eri_shell_quartet!` over every ordered quartet; per entry `|ΔJ_μν| ≤ 8 eps · N_q · Σ_λσ |(μν|λσ) P_λσ|` (spec, class (b)), N_q accumulated by the test; the same for K.
  - `"jk: density screening bound"`: a structured density on water/6-31G: `P` is nonzero (random, `MersenneTwister(12)`) only between the O shells and the H shells, and zero within the O block and within the H block, so for a quartet `(O O|H H)` the Coulomb blocks `P_ab`, `P_cd` vanish while `P_ac` does not. With the default thresholds, `|J − J_0|` and `|K − K_0|` per entry (`J_0`, `K_0` with the screens off as above) are at most the sum of `Q_ab Q_cd P_max` over the quartets the screens dropped, enumerated in the test from `schwarz_bounds` and the six block maxima, plus the class-(b) bound above (spec). The test also asserts `ws.nschwarz + ws.ndensity ==` the number of quartets it enumerated as dropped.
  - `"jk: two densities"`: `jk!((Ja, Jb), (Ka, Kb), …, (Pa, Pb))` equals two single-density calls `==` (same order of accumulation per density).
  - `"jk: want_K = false"`: K is untouched (`==` its previous contents) and J `==` the `want_K = true` J.
  - `"jk: lower triangle only"` (A9): changing `P[1, 2]` (strict upper) leaves J and K `==`.
  - `"jk: generic indexing"` (Review Focus 5): `OffsetArray` and `view` densities and outputs (N = 1 and N = 2, `jk!` and `coulomb!`) give `parent(J) == J_ref`, and the outputs keep their own axes.
  - `"jk: errors"` (Review Focus 4, 5): a 12×12 density for 13 functions throws `"integrals: density 1 is 12×12, the basis has 13 functions"`; a 12×12 `J` throws `"integrals: J 1 is 12×12, the basis has 13 functions"`; a workspace built for another engine throws `"integrals: the JKWorkspace belongs to another engine"`.
  - `"jk: allocation-free"`: `@test_noalloc jk!(…)` and `coulomb!(…)` (serial, N = 1 and N = 2).
  - `"engine contract"`: `TypeContracts` reports `MDEngine` as satisfying `AbstractIntegralEngine`, and a test-local `struct NoEngine <: AbstractIntegralEngine end` fails the check (the `PureAtomistic/test/verlet_tests.jl:368` pattern).
- [ ] **Step 2: Run** `QUANTUMENGINE_ONLY=jk_tests`. Expected: FAIL.
- [ ] **Step 3: Implement** `jk.jl`. Degeneracy: weight each unique quartet by its permutation multiplicity (½ for a = b, ½ for c = d, ½ for (ab) = (cd)) and scatter the eight J and K terms.
- [ ] **Step 4: Run.** Expected: PASS. Workload, TrimCheck (`jk!` and `coulomb!` serial, Float64; `JKWorkspace(::MDEngine{Float64,typeof(boys!)})`), coverage, `test_layers` unchanged.
- [ ] **Step 5: Mutations:** drop the ½ for (ab) = (cd) (`expect = "jk: matches the explicit sum"`); `P_max` taken over `P_ab` and `P_cd` only (`expect = "jk: density screening bound"`: the structured density makes the mutant drop `(O O|H H)` quartets whose exchange contribution is large, and the count assertion fails as well).
- [ ] **Step 6: Commit** `Add integral-direct J and K`.

### Task 9: In-core packed ERIs

**Files:** Create `src/integrals/packed.jl`, `test/packed_tests.jl`. Modify workload, trim, coverage, mutations.

**Interfaces:**
- Consumes: Tasks 7, 8.
- Produces: `PackedERI(eng::MDEngine{T}; memory_limit_bytes::Int) -> PackedERI{T}`: 8-fold packed `(μν|λσ)` over `npair = nao(nao+1)/2`, `npair(npair+1)/2` entries, holding exactly the quartets that pass the Schwarz screen and whose integrals satisfy `|I| ≥ primitive`; others zero (the kernel's exponent and prefactor cuts act inside `eri_shell_quartet!`, D4). Throws `ArgumentError("integrals: PackedERI needs 123456 bytes, above memory_limit_bytes = 1000")` when the store does not fit. Exemptions: `@strict_exempt :noalloc PackedERI reason = "entry point: packed ERI store constructor"` and `@strict_exempt :noalloc :_ reason = "entry point: keyword body of the PackedERI constructor"`. `jk!(J, K, store::PackedERI, densities; want_K = true)`, same conventions and lower-triangle rule as Task 8 (A9), no density screening.

- [ ] **Step 1: Write the failing tests.**
  - `"packed: matches direct J/K"`: water/6-31G, direct with `IntegralThresholds{Float64}(density = 0.0)` (the Schwarz and `primitive` cuts as the store applies them) vs packed, class (b) bound of the spec (`|ΔJ_μν| ≤ 8 eps · N_q · Σ|(μν|λσ) P_λσ|`).
  - `"packed: lower triangle only"` (A9): changing `P[1, 2]` leaves J and K `==`.
  - `"packed: two densities"`: N = 2 equals two single-density calls `==`.
  - `"packed: generic indexing"`: `OffsetArray` and `view` densities and outputs give `parent(J) == J_ref`.
  - `"packed: memory limit"`: the message.
  - `"packed: size mismatch"`: `DimensionMismatch` naming both sizes (Review Focus 4).
  - `"packed: allocation-free"`: `@test_noalloc jk!(J, K, store, densities)`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS. Workload, TrimCheck (`jk!` on `PackedERI{Float64}`, `PackedERI(::MDEngine{Float64,typeof(boys!)})`), coverage.
- [ ] **Step 5: Mutation:** the packed index `ij*(ij−1)÷2 + kl` → `ij*(ij+1)÷2 + kl` (`expect = "packed: matches direct J/K"`).
- [ ] **Step 6: Commit** `Add the in-core packed ERI store`.

### Task 10: Independent references

**Blocked by:** D1, D2, D3.

**Files:** Create `test/reference/os_reference.jl` (`@testmodule OSReference`), `test/references_tests.jl`. (With D3's libcint option instead: `test/libcint/{Project.toml,runtests.jl}` and a README, outside the main suite, run by hand as PureXC's `test/libxc/`.)

**Interfaces:**
- Consumes: Tasks 5, 7, 8; `Fixtures.h2o_sto3g_jk_density()`.
- Produces (test only): `OSReference.overlap_ref`, `kinetic_ref`, `nuclear_ref`, `eri_ref(shells, a, b, c, d)::Array{BigFloat,4}` at `setprecision(256)`, with Boys values from `BoysReference.boys_big` (whose precision follows t, Task 1) and component normalization from `primitive_norm` in BigFloat. Each also returns `S_abs`, the sum of the absolute values of its terms, for the error measure below.

- [ ] **Step 1: Write the reference** from Obara & Saika 1986 (VRR) and Head-Gordon & Pople 1988 (HRR), without reading `eri.jl`, `one_electron.jl` or the oracle (Global Constraints, Oracle code).
- [ ] **Step 2: Write the tests.** The error measure of the two Obara–Saika items is normalized, not per-entry relative (an entry that is small by cancellation, or zero by symmetry in planar water, has no meaningful relative error): `err = |Δ| / S_abs` per entry, with `S_abs` the reference's absolute term sum for that entry (for an entry that is zero by symmetry the reference returns `S_abs > 0` from its nonzero terms, so the measure stays defined).
  - `"references: J and K of the oracle fixture"`: on `water_eri()`/STO-3G with `P = diag(2, 2, 2, 2, 2, 0, 0)`, `jk!` with `IntegralThresholds{Float64}(schwarz = 0.0, density = 0.0, primitive = 0.0)`: `J[1,1]` against 18.502908 and `K[1,1]` against 9.990743 (D1, D2 defaults), each within `5e-7 + 8 eps · N_q · S_abs` (class (b): half a unit in the sixth printed decimal of the reference, plus the rounding bound of our sum with N_q and S_abs accumulated by the test), the comment giving the PySCF literals 18.502906 and 9.697012, the three codes that agree (E1) and E2's arithmetic.
  - `"references: one-electron against Obara–Saika"`: S, T, V on water/6-31G* and on `build_basis(<a single O atom>, "cc-pvqz")` (g shells); `err ≤ tol`, class (c): the worst `err` measured at Step 3, in the comment, × 20.
  - `"references: ERIs against Obara–Saika"`: every class up to (dd|dd) on water/6-31G*, and one quartet each of (ff|ff), (gg|ss) and (hs|hs) (Review Focus 1); class (c) as above.
- [ ] **Step 3: Run** `QUANTUMENGINE_ONLY=references_tests`; write the measured errors and each item's run time into the comments; tag an item `:slow` if it takes more than 30 s (Global Constraints, "Slow items"). Expected: PASS. If an `err` exceeds 1e-12 on any class, stop and report: that is a convention or algorithm fault, not a tolerance question. Then the full suite.
- [ ] **Step 4: Commit** `Check integrals against an independent reference`.

### Task 11: Threaded J/K on a persistent worker pool

**Blocked by:** D6, Q-DETERMINISM's follow-up question.

**Files:** Create `src/integrals/workers.jl`, `test/jk_threads_tests.jl`. Modify `src/integrals/jk.jl`, `test/jk_tests.jl` (the `nworkers` errors), workload, trim, coverage, mutations, `AGENTS.md` (the "Functions ending in `!`" exception gains QuantumEngine's first threaded `jk!`; shared).

**Interfaces:**
- Consumes: Task 8; `PureMLIP/src/model/workers.jl` (copied, D6: `_chunk!`, `_WorkerPool`, `ChunkWorkers`, `_run_chunks!`).
- Produces (Q-DETERMINISM as applied):
  - `JKWorkspace(eng::MDEngine{T}; nworkers::Int = 1, memory_limit_bytes::Int = 2^31) -> JKWorkspace{T}`. With `nworkers = w > 1` it holds w `ERIWorkspace`s and block buffers and w J/K accumulator pairs for `N ≤ 2` densities, `4 w nao² sizeof(T)` bytes in all; the docstring gives the formula (w = 4, nao = 1000, Float64: 128 MB). Errors: `nworkers < 1` → `ArgumentError("integrals: nworkers must be ≥ 1, got 0")`; `nworkers > Threads.nthreads(:default)` → `ArgumentError("integrals: nworkers = 8 is above the 4 threads of the default pool")`; accumulators above the limit → `ArgumentError("integrals: JKWorkspace needs 3200000000 bytes for 4 workers, above memory_limit_bytes = 2147483648")`. Exemptions: the constructor's `:_` keyword body, `@strict_exempt :noalloc :_ reason = "entry point: keyword body of the JKWorkspace constructor"`.
  - With `w > 1`, `jk!` calls `_run_chunks!(workers, job, phase, w)`: chunk `c ∈ 1:w` (workers 1:w−1 and the calling task, as `ChunkWorkers` runs them) handles the outer shell pairs `(a, b)` with `mod(pair_index(a, b) − 1, w) + 1 == c` in the serial quartet order, into accumulator pair `c`; then `J`, `K` are the sums of the accumulators in chunk order 1:w. Results are `==` across calls with the same w and differ from serial only by summation order. The pool starts on the first threaded call.
  - `Base.close(ws::JKWorkspace)` stops the workers. A second concurrent `jk!` on one workspace throws `"integrals: the JKWorkspace is in use by another call; a workspace serves one call at a time"`.
  - Exemptions copied with `workers.jl`: `grep strict_exempt` over `PureMLIP/src` finds none naming the worker functions (read); any allocating worker function the QuantumEngine sweep reports (the pool start of the first call, `close`) is exempted by name with the reason `"entry point: starts or stops the J/K worker tasks of a threaded jk!"`, and the task lists each in its commit body.

- [ ] **Step 1: Write the failing tests.**
  - `"jk threads: matches serial"`: in a child process with `JULIA_NUM_THREADS=4`, `jk!` on water/6-31G with `nworkers = 4` against the serial result (computed in the child), within the class-(b) bound of `"jk: matches the explicit sum"` (`8 eps · N_q · Σ|terms|`, N_q accumulated by the test).
  - `"jk threads: repeatable"`: in the same child, two threaded calls give `==` results, and a second `JKWorkspace` with `nworkers = 4` gives `==` results too (Q-DETERMINISM: bitwise for a fixed thread count).
  - `"jk threads: allocations"`: the first threaded call's bytes are below an upper bound measured at Step 4 (written in the comment, ×2), and the fewest bytes of five later calls is 0 (spec).
  - `"jk threads: busy workspace"`: the message above (a call started on another task while one runs, as PureMLIP tests it).
  - `"jk: nworkers errors"` (in `test/jk_tests.jl`): the three messages of Interfaces.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement**: a job type `JKJob` with `_chunk!(job::JKJob, phase, c)` running chunk c's (a, b) pairs into accumulator pair c with its own `ERIWorkspace` and block buffer.
- [ ] **Step 4: Run.** Expected: PASS. Workload (a threaded call), TrimCheck `@validate` of threaded `jk!` (research E8: plain `@spawn` fails TrimCheck, so this item is the proof that the copied pool trims here) and of `JKWorkspace(::MDEngine{Float64,typeof(boys!)}; nworkers = 4)`, coverage, `"every exemption is used and reasoned"`.
- [ ] **Step 5: Mutation:** the reduction sums accumulator pairs `1:w−1` only, dropping the calling task's chunk (`expect = "jk threads: matches serial"`).
- [ ] **Step 6: Commit** `Run J/K on a persistent worker pool`.

### Task 12a: Density fitting

**Blocked by:** D5, Q-RIF32.

**Files:** Create `src/integrals/ri.jl`, `test/ri_tests.jl`. Modify `test/layers_tests.jl` (the `Integrals` row gains `Errors`, for `NotImplemented`), `docs/src/integrals.md` (an RI section), workload, trim, coverage, mutations.

**Interfaces:**
- Consumes: Tasks 3, 4, 7, 8; step 02 `build_basis(mol, "def2-universal-jkfit")`.
- Produces:
  - `const RI_METRIC_CUTOFF = 1e-10`.
  - `two_center(aux::BasisSet{T}; boys = boys!)::Matrix{T}` (`(A|B)`; exempt by name and its `:_` keyword body); `three_center!(B::AbstractMatrix{T}, ws::ERIWorkspace{T}, eng::MDEngine{T}, aux::BasisSet{T}) -> B`, packed `[npairs, naux]` with pair index `μ ≥ ν` in `pair_index` order over AOs; `ws` must be sized for `L = 2 l_max(AO) + l_max(aux) + deriv_order` (`DimensionMismatch("integrals: ERIWorkspace is sized for L = 8, the three-center integrals need L = 10")`).
  - `AbstractRIContext{T}`, `RIContextInCore{T}` and `RIContextDirect{T}` with the spec's fields plus exchange scratch (A8), the three-center `ERIWorkspace` and the engine it builds; `RIContext(bs, aux; memory_limit_bytes = 512_000_000, max_occupied = bs.nao, array_type = Array, boys = boys!, deriv_order = 0)` chooses in-core when `npairs · naux · sizeof(T) ≤ memory_limit_bytes` (D5); `array_type ≠ Array` throws `NotImplemented("16-gpu", …)`; an auxiliary shell with l > 5 throws the angular-momentum message. Exemptions: `@strict_exempt :noalloc RIContext reason = "entry point: density-fitting context constructor"`, `@strict_exempt :noalloc :_ reason = "entry point: keyword body of the RIContext constructor"`, and the same pair for `two_center`.
  - `ri_coulomb!(J, ctx, P)`: `γ_A = Σ (μν|A) P_μν`, `d = V⁺ γ` with `1/λ` on eigenvalues `> RI_METRIC_CUTOFF` (pseudo-inverse), `J_μν = Σ_A (μν|A) d_A`. Reads the lower triangle of `P` (A9).
  - `ri_exchange!(K, ctx, Cocc)`: with `B̃ = (μν|A) V^{-1/2}` (same cutoff), `K_μν = Σ_i Σ_A B̃_{μi,A} B̃_{νi,A}`; throws per A8.
  - The direct context computes `(μν|A)` per call in aux-shell blocks into preallocated scratch.
- [ ] **Step 1: Write the failing tests.**
  - `"ri: metric"`: `two_center` is symmetric (`==` after mirroring) and its kept eigenvalues are all `> RI_METRIC_CUTOFF`; `Vinvhalf * V * Vinvhalf` equals the projector onto the kept space within `8 eps · naux · cond_kept` (class (b), `cond_kept` the ratio of the largest to the smallest kept eigenvalue, printed in the comment). The item also runs on an auxiliary set that has an eigenvalue below the cutoff by construction (Step 3 decides which; see there) and asserts that the dropped count is `≥ 1`.
  - `"ri: in-core equals direct"`: both contexts on water/6-31G give J and K within the class-(b) bound `8 eps · naux · cond_kept · Σ|terms|` (same integrals, different summation order; the summation-order difference in γ passes through `V⁺`, whose gain is bounded by `cond_kept`, Q-RIF32).
  - `"ri: exchange against its formula"`: `ri_exchange!` equals the explicit `Σ` built in the test from `three_center!` and `Vinvhalf`, class (b).
  - `"ri: close to exact Coulomb"`: on water/6-31G with def2-universal-jkfit at the core-Hamiltonian guess density (`2 C_occ C_occᵀ` from `eigen(Symmetric(H), Symmetric(S))`), `|E_J^{RI} − E_J|` with `E_J = ½ tr(P J)` is at most 20 × the value measured at Step 3 (class (c), the measurement in the comment).
  - `"ri: Float32"`: `ri_coulomb!` in Float32 within `8 eps(Float32) · naux · cond_kept(Float32) · Σ|terms|` of Float64 (Q-RIF32; `cond_kept(Float32)` from the Float32 metric's own eigenvalues).
  - `"ri: generic indexing"`: `ri_coulomb!` and `ri_exchange!` with `OffsetArray` and `view` `J`, `K`, `P`, `Cocc`, and `three_center!` into an `OffsetArray` and a `view` `B`: `parent(...) ==` the `Array` results.
  - `"ri: errors"`: A8's message, `NotImplemented` for `array_type`, `DimensionMismatch` for a wrong-size `P` (Review Focus 4), the three-center workspace message, and an l = 6 auxiliary shell.
  - `"ri: allocation-free"`: `@test_noalloc` on `ri_coulomb!`, `ri_exchange!` (both contexts) and `three_center!`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement**, then measure and record in the item comments and in `docs/src/integrals.md`: the measured `|E_J^{RI} − E_J|`; the smallest metric eigenvalue and `cond_kept` of water/6-31G with def2-universal-jkfit in Float64 and Float32 (Q-RIF32; if `cond_kept(Float32) · eps(Float32) ≥ 1`, stop and ask). If the smallest Float64 eigenvalue is above `RI_METRIC_CUTOFF`, the second set of `"ri: metric"` is def2-universal-jkfit on water plus a copy of one O s shell with its exponents scaled by `1 + 1e-6`, whose overlap with the original puts an eigenvalue near `1e-12` (estimate; the item asserts the dropped count, so an estimate that is wrong fails loudly). The known deviation of the oracle's direct RI-K path (`ri_k.rs:57-68`, cutoff 1e-12; this package uses `RI_METRIC_CUTOFF` everywhere) goes into the RI section.
- [ ] **Step 4: Run.** Expected: PASS. Workload, TrimCheck (`ri_coulomb!`, `ri_exchange!`, `three_center!` in Float64 and Float32; `RIContext(::BasisSet{Float64}, ::BasisSet{Float64})` and the Float32 form; `two_center`), coverage.
- [ ] **Step 5: Mutations:** `> RI_METRIC_CUTOFF` → `≥ 0` (`expect = "ri: metric"`: the second set has an eigenvalue below the cutoff, so the mutant keeps it); `V⁺` → `V^{-1/2}` in `ri_coulomb!` (`expect = "ri: close to exact Coulomb"`).
- [ ] **Step 6: Commit** `Add density fitting for J and K`.

### Task 12b: The oracle's RI fitting error (only if Q-RIFIT keeps it in 03)

**Blocked by:** Q-RIFIT, the user's captures.

**Files:** Modify `test/ri_tests.jl`, `test/oracle/references.toml` (two RI cases), `docs/src/integrals.md`.

- [ ] **Step 1: USER STEP.** Add two cases to `test/oracle/references.toml`: water/6-31G RHF with `use_ri_j true` and `aux_basis def2-universal-jkfit` (02: the capture must pass the aux basis explicitly), and the same without RI. Ask the user to run `capture.jl` for them. Nothing in Step 2 proceeds until the files exist; no captured value is created by an agent. Task 12a and the tasks after it do not wait for this.
- [ ] **Step 2: Write the item** `"ri: Coulomb fitting error against the oracle"`: the `|E_J^{RI} − E_J|` of `"ri: close to exact Coulomb"` is at most 20 × the oracle's measured fitting error (spec, class (c)), taken as the difference between the oracle's `use_ri_j true` and exact total energies from the two captures. The comment says that the two sides are different quantities of the same fitting error (a Coulomb energy at a guess density against a total-energy difference at convergence), which is how the spec states the test.
- [ ] **Step 3: Run.** Expected: PASS. Record the oracle's fitting error in the item comment and in `docs/src/integrals.md`.
- [ ] **Step 4: Commit** `Compare the RI fitting error with the oracle`.

### Task 13: MO integral transforms

**Files:** Create `src/integrals/mo.jl`, `test/mo_tests.jl`. Modify workload, trim, coverage, mutations.

**Interfaces:**
- Consumes: Tasks 7, 12a.
- Produces (entry points, exempt by name): `mo_eri_ovov(eng, Cocc, Cvir)::Array{T,4}` with `out[i, a, j, b] = (ia|jb)`; `mo_eri_oovv(eng, Cocc, Cvir)::Array{T,4}` with `out[i, j, a, b] = (ij|ab)`; `ri_mo_tensor(ctx, C1, C2)::Array{T,3}` with `out[p, q, A] = Σ (μν|A) C1_μp C2_νq` contracted with `V^{-1/2}` (spec). N⁵: the first index is transformed shell pair by shell pair into a half-transformed store, the rest by `mul!`. The outputs are new 1-based `Array`s indexed by orbital number, so the functions declare the assumption with `Base.require_one_based_indexing(Cocc, Cvir)` (and `C1`, `C2`) rather than carry offset axes into an orbital-numbered result.
- [ ] **Step 1: Write the failing tests.**
  - `"mo: ovov and oovv against the AO sum"` on water/STO-3G with random orthonormal C (o = 5, v = 2): the naive `Σ_{μνλσ} C C C C (μν|λσ)` in the test, class (b) `8 eps · nao⁴ · Σ|terms|`.
  - `"mo: ri tensor"` against its definition, class (b).
  - `"mo: generic indexing"`: `view` coefficient matrices (columns of a larger matrix) give `==` results; `OffsetArray` coefficient matrices throw the `ArgumentError` of `require_one_based_indexing`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS. Workload, TrimCheck, coverage.
- [ ] **Step 5: Mutation:** in `mo_eri_ovov`'s final store, write the half-transformed entry with the two virtual indices exchanged, `out[i, a, j, b] = (ib|ja)` (`expect = "mo: ovov and oovv against the AO sum"`; the shapes agree, the values do not).
- [ ] **Step 6: Commit** `Add MO integral transforms`.

### Task 14: First-derivative integrals

**Blocked by:** Q-DERIVINT, A6, A7.

**Files:** Create `src/integrals/derivatives.jl`, `test/derivatives_tests.jl`. Modify workload, trim, coverage, mutations.

**Interfaces:**
- Consumes: Tasks 3–8, 12a (`MDEngine(bs; deriv_order = 1)`, `ShellPairData` tables to `l + 1`, `RIContext(…; deriv_order = 1)`).
- Produces (default (c)):
  - `overlap_gradient!(dS::NTuple{3,AbstractMatrix}, bs, atom::Int[, ws])`, `kinetic_gradient!(dT, bs, atom[, ws])`, `nuclear_attraction_gradient!(dV, bs, mol, atom[, ws]; include_operator_derivative = true)`: `∂X/∂R_{atom,k}` for k = 1:3. `ws::OneElectronWorkspace` (its E tables already run to `l_max + 2`).
  - `ERIGradientWorkspace(eng::MDEngine)` (exempt constructor): the quartet derivative blocks and the shell-pair density maxima scratch; it throws `ArgumentError("integrals: ERI gradients need an MDEngine built with deriv_order = 1")` when `eng.pairs.deriv_order == 0`.
  - `eri_gradient_contract!(G::AbstractMatrix, gws::ERIGradientWorkspace, eng, densities::NTuple{N,AbstractMatrix}; exx::Real)` (A6) and the entry-point form without `gws`: adds to `G[atom, k]` the derivative of `E₂ = ½ Σ P_μν P_λσ (μν|λσ) − exx · ¼ Σ P_μλ P_νσ (μν|λσ)` (N = 1, total P) or `½ Σ P P (μν|λσ) − exx · ½ Σ_s P^s_μλ P^s_νσ (μν|λσ)` (N = 2), with Schwarz screening as `jk!`.
  - `ri_three_center_gradient!(G, ctx, Γ)`, `ri_two_center_gradient!(G, ctx, Γ2)` (A7), using the derivative scratch of a `RIContext` built with `deriv_order = 1` (else the same kind of `ArgumentError`).
  - Exemptions (A13): `@strict_exempt :noalloc` by name for each workspace-free form (`overlap_gradient!`, `kinetic_gradient!`, `nuclear_attraction_gradient!`, `eri_gradient_contract!`; reason `"entry point: builds its workspace and calls the workspace form"`), for `ERIGradientWorkspace` (constructor), and `:_` for the keyword bodies of `nuclear_attraction_gradient!` and `eri_gradient_contract!` if the sweep reports them.
  - Derivative of a primitive: `∂/∂A_k φ(l) = −l_k φ(l − 1_k) + 2α φ(l + 1_k)`, evaluated on the unnormalized primitive with the original primitive's weight (E9: both oracle forms give the derivative of the normalized function). Each function's docstring states why the shift relation and not AD is used (the Q-DERIVINT answer).
- [ ] **Step 1: Write the failing tests.** On water/6-31G*:
  - `"derivatives: one-electron against finite differences"`: analytic vs central difference of `overlap`, `kinetic`, `nuclear_attraction` at h = 1e-4 per atom and axis; per element `|D − D_h| ≤ (4/3)|D_h − D_{h/2}| + 4 eps · max|f| / h` (spec, class (b)).
  - `"derivatives: ERI contraction against finite differences"`: `eri_gradient_contract!` against the central difference of `E₂` computed with `jk!` at displaced geometries, with `IntegralThresholds{Float64}(schwarz = 0.0, density = 0.0, primitive = 0.0)` on both sides so that no screen makes `E₂` jump with the geometry (the kernel's exponent and prefactor cuts remain; the terms they drop are below `exp(−45)` relative, about seven orders under the `4 eps · max|f| / h` term); same bound; N = 1 and N = 2, `exx ∈ (0, 0.25, 1)`.
  - `"derivatives: RI against finite differences"`: the two RI entries with random Γ against the difference of `Σ Γ (μν|A)` and `Σ Γ2 (A|B)`.
  - `"derivatives: translational invariance"`: `|Σ_atoms ∂X/∂R| ≤ 8 eps · N · S_abs` (spec, class (b), N and S_abs accumulated by the test) for S, T and V (with the operator derivative) and for the `eri_gradient_contract!` result (spec: "translational invariance of gradients", including the two-electron part; it guards against an implementation that computes all four centers inconsistently).
  - `"derivatives: generic indexing"`: `OffsetArray` and `view` matrices in `dS`, `dT`, `dV`, `G` and the densities give `parent(...) ==` the `Array` results.
  - `"derivatives: allocation-free"`: `@test_noalloc` on the `ws` and `gws` forms, `ri_three_center_gradient!` and `ri_two_center_gradient!`.
  - `"derivatives: errors"`: the two `deriv_order` messages.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS. Workload (every exempted form), TrimCheck (the `ws`/`gws` forms in Float64 and Float32), coverage.
- [ ] **Step 5: Mutations:** `−l_k` → `+l_k` in the shift relation (`expect = "derivatives: one-electron against finite differences"`); skip the operator-derivative term when `include_operator_derivative` is true (`expect = "derivatives: translational invariance"`).
- [ ] **Step 6: Commit** `Add first-derivative integrals`.

### Task 15: Second-derivative integrals by Taylor numbers

**Blocked by:** Q-DERIVINT.

**Files:** Create `src/integrals/second_derivatives.jl`, `test/second_derivatives_tests.jl`. Modify `Project.toml` (PureTaylor through `[sources] PureTaylor = {path = "../PureTaylor"}`, `[deps]`, `[compat]`), `docs/src/dependencies.md`, the type bound of the kernels the Taylor evaluation passes through (`T<:Real`, Q-DERIVINT: the one-electron kernels, `hermite_E!`, `hermite_R!`, `ShellPairData` and `eri_shell_quartet!`), workload, trim, coverage, mutations.

**Interfaces:**
- Consumes: Tasks 5, 7, 14; PureTaylor's `Taylor{N,K,M,T,L} <: Real` (`PureTaylor/src/type.jl:10`).
- Produces (default (c)):
  - `overlap_hessian!(d2S, bs, atom1, atom2[, ws])`, `kinetic_hessian!`, `nuclear_attraction_hessian!(…, mol, …[, ws]; include_operator_derivative = true)`, each filling the 3×3 blocks `∂²X/∂R_{atom1,k}∂R_{atom2,l}` as a `NTuple{9,AbstractMatrix}`, by evaluating the one-electron kernels on second-order Taylor numbers in the six coordinates of the two atoms. `ws` is a `HessianWorkspace(bs)` holding the Taylor-typed `OneElectronWorkspace` scratch (exempt constructor); the workspace-free forms are exempt entry points (A13).
  - `eri_shell_quartet!` accepts Taylor-typed `ShellPairData`/`MDEngine` (built from a `BasisSet` whose centers are Taylor numbers; whether step 02's `Shell{T}` and `BasisSet{T}` accept a `T` that is not an `AbstractFloat` is not checked, and if they do not, Task 15 stops and reports before loosening anything), so step 08 can form the two-electron second-derivative contraction by AD (Q-DERIVINT: the contraction itself is 08's).
  - A `boys!` method for Taylor arguments in this file: `boys!` stays Float32/Float64 (A1) on scalars, and the Taylor method evaluates `F_m` at the scalar part and propagates derivatives by `dF_m/dt = −F_{m+1}`.
- [ ] **Step 0: Resolve the closure** of PureTaylor (path) as Task 8 Step 0 and record it.
- [ ] **Step 1: Write the failing tests.**
  - `"second derivatives against differences of first derivatives"`: central differences of Task 14's analytic first derivatives at h = 1e-4, with the spec's bound.
  - `"second derivatives: symmetry"`: `∂²/∂R_a∂R_b == (∂²/∂R_b∂R_a)ᵀ` per block within `8 eps · |X|`.
  - `"second derivatives: translational invariance"` as in Task 14.
  - `"second derivatives: Taylor ERIs"`: on water/6-31G with a random density, `E₂` contracted in the test from Taylor-typed `eri_shell_quartet!` blocks: its first-order part equals `eri_gradient_contract!` within `8 eps · N · S_abs` (class (b)), and its second-order part equals the central difference of `eri_gradient_contract!` at h = 1e-4 within the spec's finite-difference bound.
  - `"second derivatives: generic indexing"`: `OffsetArray` and `view` blocks in `d2S` give `parent(...) ==` the `Array` results.
  - `"second derivatives: allocation-free"`: `@test_noalloc` on the `ws` forms.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Loosen only the kernels the Taylor evaluation passes through.
- [ ] **Step 4: Run.** Expected: PASS, including the full suite: the strict, trim and allocation items of Tasks 3–7 and 14 run again on the loosened kernels.
- [ ] **Step 5: Benchmark option (b).** On water/6-31G*, time `eri_gradient_contract!` (Task 14) against the same contraction evaluated by first-order Taylor numbers through the committed `eri_shell_quartet!`, 1 thread, interleaved, and report speed-up = AD/shift (> 1 means the shift relations are faster) in the commit body and Evidence for the user's Q-DERIVINT review. The timing harness is a scratch probe (Global Constraints).
- [ ] **Step 6: Gates.** Workload (the hessian `ws` forms and a Taylor-typed `eri_shell_quartet!` call, so the sweep judges the Taylor specializations), TrimCheck (Float64 hessian `ws` forms and the Taylor-typed `eri_shell_quartet!`), coverage, mutation: drop the `−F_{m+1}` chain term (`expect = "second derivatives against differences of first derivatives"`).
- [ ] **Step 7: Commit** `Add second-derivative integrals by Taylor numbers`.

### Task 16: The oracle's Boys algorithm as a test variant

**Blocked by:** Q-BOYSPARITY.

**Files:** Create `test/reference/oracle_boys.jl` (`@testmodule OracleBoys`), `test/oracle_boys_tests.jl`. Modify `docs/src/integrals.md` (parity notes).

**Interfaces:**
- Consumes: Tasks 1, 2 (`Special.erf`), 5 and 7 (A14's `boys` argument on `OneElectronWorkspace` and `MDEngine`), 8.
- Produces (test only, Q-BOYSPARITY default (2)): `OracleBoys.oracle_boys!(F, nmax, t)`: t < 1e-10 → `1/(2m+1)`; t > 60 → the leading asymptotic term; t < 15 → `F_0 = 0.5 * sqrt(π / t) * Special.erf(sqrt(t))` and the upward recurrence `F_m = ((2m − 1) F_{m−1} − e^{−t}) * (0.5 / t)`, in that operation order; else downward from `n_start = max(n + ⌊t⌋ + 45, 50)` (spec Facts, `boys.rs:25-63`, read for behavior only). It is written from that description, not from the Rust source text. Its `F_0` differs from the oracle's wherever this package's `erf` and `libm::erf` round differently, and the upward branch amplifies that difference (Q-BOYSPARITY), so the variant is the oracle's algorithm, not the oracle's numbers.
- [ ] **Step 1: Write the tests.**
  - `"oracle boys: implements the algorithm"`: on the upward branch at `t ∈ (1e-4, 1e-3, 0.01, 0.1, 1, 14.9)` and `m ≤ 8`, `oracle_boys!` against a `BigFloat` evaluation of the same recurrence started from the same Float64 `F_0`, `e^{−t}` and `0.5/t` (the values the variant computed), within a bound the test propagates step by step: `δ_m = (2m−1)·(0.5/t)·δ_{m−1} + 3u·((2m−1)|F_{m−1}| + e^{−t})·(0.5/t)`, `δ_0 = 0`, `u = eps(Float64)/2`, times 2 (class (b): three roundings per step, propagated through the recurrence's own gain). The same check for the downward branch at `t ∈ (15, 30, 59.9)` from the same start value, and `==` for the two closed-form branches.
  - `"oracle boys: its own defects"` (reported): the relative errors against `boys_big` at the six (t, m) points of spec 03 Facts, written to the test log through `@info` and into `docs/src/integrals.md` next to spec 03's figures and the F_0 sensitivity of Q-BOYSPARITY; the item asserts only that each is finite.
  - `"oracle boys: effect on integrals"` (reported): with `MDEngine(bs; boys = oracle_boys!)` and `OneElectronWorkspace(bs; boys = oracle_boys!)` on water/6-31G and 6-31G*, the largest absolute differences in V, in the ERIs, and in `J`, `K` at a fixed random density; the item asserts only that the values are finite and writes them to the test log through `@info`. These are estimates of the oracle's defect (Q-BOYSPARITY).
- [ ] **Step 2: Run** and record the reported differences in `docs/src/integrals.md` and Evidence, labelled as estimates. Expected: PASS.
- [ ] **Step 3: Commit** `Add the oracle-algorithm Boys variant for parity runs`.

### Task 17: Disposition of oracle code

**Blocked by:** Q-ORACLEDEAD, Q-ORACLESRC.

**Files:** Modify `docs/src/integrals.md`.

- [ ] **Step 1: Write the table** in `docs/src/integrals.md`: one row per file under the oracle's `src/integrals/` and `src/integrals/eri/` (E9's map). Columns: file (the first column, the file's path relative to `src/integrals/` in backticks, e.g. `` `eri/eri_ssss.rs` ``), call sites, reachable from the CLI, action (ported in Task N / not ported, with the reason / `NotImplemented` from the run-mode driver of a later step, naming that step). Include `ecp.rs` (step 14, A10), `zora.rs`, `zora_so.rs` (no 03 entry point; 04's run-mode driver throws `NotImplemented`, step-01 D21), `ewald.rs` and the lattice-image sums (unreachable, `src/engine/mod.rs:94-96`), `magnetic.rs` (empty), `nuclear.rs`, `eri_shell.rs`, `eri_hrr.rs` (no call sites).
- [ ] **Step 2: Check the table** (Q-ORACLESRC default: a manual step, no test item): list the `*.rs` files under those two directories of `$QUANTUM_ENGINE_ORACLE_SRC` (`find … -name '*.rs'`, paths relative to `src/integrals/`), extract the first-column names from the table with the regex `` ^\| `([^`]+\.rs)` \| ``, and compare the two sorted lists. Record the command and "the lists are equal (N files)" in the commit body. If they differ, fix the table. (If the user picks the alternative, this becomes the item `"disposition: every oracle integral file has a row"`, tag `:oracle_src`, in `test/disposition_tests.jl`, with the same regex, and the Global Constraint "Oracle input files" is amended to allow it.)
- [ ] **Step 3: Commit** `Record the disposition of the oracle's integral code`.

### Task 18: Benchmarks

**Blocked by:** Q-SCFGATE, Q-BENCHDENSITY.

**Files:** Create `bench/quartet_bench.jl`, `bench/jk_bench.jl`, `bench/results/`. Modify `docs/src/integrals.md` (a results section).

- [ ] **Step 1: Quartet throughput.** `quartet_bench.jl`: ns per shell quartet and per primitive quartet for (ss|ss), (pp|pp), (dd|dd), (ff|ff) at K = 1 and 3 on four distinct centers (the E4 cases), 1 thread, min of 7 rounds. Write `bench/results/quartet_<date>.json`.
- [ ] **Step 2: J/K per call.** `jk_bench.jl`: `jk!` wall time per call on water/6-31G and benzene/6-31G (`Fixtures.benzene()`) with the Q-BENCHDENSITY densities (water: the captured `water_pbe_631g.molden` density; benzene: the core-Hamiltonian guess density `2 C_occ C_occᵀ` from `eigen(Symmetric(H), Symmetric(S))`; each recorded in the JSON as what it is), `nworkers ∈ (1, 4)`, arms interleaved in child processes, 7 rounds, min and median. The BLAS backend each run used is recorded (Q-BLAS: the choice of default is 04's).
- [ ] **Step 3: Bounds checks end to end.** Run Step 2's benzene arm under `--check-bounds=no` and default, interleaved, and report the speed-up. No `@inbounds` is added (Global Constraints).
- [ ] **Step 4: Report** in `docs/src/integrals.md`: the tables, the E5 prototype figure (1.9 s for the benzene integral sweep) beside the measured `jk!` time, and, under Q-SCFGATE (a), the sentence that the oracle comparison is 04's gate. Under (a′), the gate item (`"bench: jk! against the integral sweep"`, tag `:slow`) asserts the ratio the user set.
- [ ] **Step 5: Only if the user named the oracle binary and asked** (Q-SCFGATE row): time whole oracle runs of water/6-31G and benzene/6-31G RHF at 1 and 4 threads, interleaved with nothing (the Julia arm has no SCF yet), and record them as reported figures with the oracle commit.
- [ ] **Step 6: Commit** `Add integral benchmarks and their results`.

### Task 19: Step completion gate

**Files:** Modify `test/coverage/thresholds.toml` (final rows), `docs/src/api.md` (every exported and public name of `Integrals` in an `@docs` block), `docs/src/integrals.md`.

- [ ] **Step 0: Preconditions.** Every row of both decision tables has the user's answer, recorded in this plan; Task 12b's captures exist if Q-RIFIT keeps that task in 03.
- [ ] **Step 1: Run the full suite** under `--check-bounds=yes`. Expected: every item passes, none skipped outside the tagged groups.
- [ ] **Step 2: Run the slow items** with `QUANTUMENGINE_SLOW=1`. Expected: PASS.
- [ ] **Step 3: Run the juliac item** with `QUANTUMENGINE_JULIAC_BUILD=1`. Expected: PASS (the trimmed entry is unchanged; the `Integrals` entry points are proved by TrimCheck items).
- [ ] **Step 4: Coverage** (`Pkg.test(coverage = true)`, then `check.jl`). Expected: PASS, each new row at its measured floor.
- [ ] **Step 5: Mutations** (`julia QuantumEngine/test/mutation/run.jl`). Expected: the control passes and every entry is killed by its `expect` item.
- [ ] **Step 6: Docs** build locally without deploying; no missing-docstring warning. `runic --check QuantumEngine` clean.
- [ ] **Step 7: Commit** `Complete QuantumEngine step 03` (docs and coverage rows only, if changed).
- [ ] **Step 8: Review.** One review of the branch against spec 03, PLAN.md's "Gates between steps" and this plan. Findings are fixed on the branch, Steps 1–6 rerun, and the review is requested again for the changes.
- [ ] **Step 9: Report** to the coordinator: the review outcome; the D4 measurement; the Task 16 defects; the Task 15 option-(b) benchmark; the Task 18 tables; and the shared files changed (AGENTS.md) for the D20 notice.

---

## Self-review

1. **Spec coverage.** Boys (Task 1), `erfc` in `Integrals.Special` (2), shell-pair data and E (3), R (4), S/T/V with point charges and ghosts and `core_hamiltonian` (5), dipole/angular momentum/ESP (6), ERIs, thresholds, Schwarz, `@test_noalloc` (pp|pp)–(ff|ff) (7), direct J/K and the contract (8), `PackedERI` (9), oracle literals and independent references (10), threading and determinism (11), RI with both cutoffs and the in-core switch (12a; the oracle fitting-error comparison is 12b or 04, Q-RIFIT), MO transforms (13), derivative integrals (14, 15; the two-electron second-derivative contraction is 08's, Q-DERIVINT), Q-BOYSPARITY (16), Q-ORACLEDEAD and ZORA (17), performance report (18), generic indexing (1, 5, 8, 9, 12a, 13, 14, 15; Task 6 takes no user-indexed arrays), the GPU path (out of scope: `array_type` throws, step 16). The spec's "translational invariance of gradients" is tested for the one-electron parts and the ERI contraction (14). The spec's mention of an RI-J "1e-12 test" in 04 has no counterpart in spec 03; Task 12a's in-core-vs-direct class-(b) item is the test 04 refers to.
2. **Not covered here.** PCM (11) and ESP fitting (13) need the attraction integrals of many arbitrary charges without nuclei and the density-contracted potential at many points. Neither is in spec 03's interface; steps 11 and 13 add `point_charge_attraction!(V, bs, charges, ws)` and an ESP-at-points contraction on top of Task 5's kernel.
3. **Steps.** Each code step names its signature and file; bodies are given only for the kernel recursions, whose form the tests do not fix.
4. **Types.** `MDEngine{T,F}` (Tasks 7, 8, 11, 12a, 14, 16), `JKWorkspace` (8, 11), `ShellPairData` (3, 7, 14), `ERIWorkspace` (7, 11, 12a), `ERIGradientWorkspace` (14), `pair_index` (3, 8, 12a), `hermite_index` (4, 7), `OneElectronWorkspace{T,F}` (5, 14, 16), `HessianWorkspace` (15) are used with the same names and arguments throughout.
5. **Review Focus.** Each of the five lines has an owning item (Tasks 3, 5, 7, 8, 9, 10, 12a).
6. **Proportion.** The plan has no kernel bodies beyond the recursion formulas; its length is in the decisions and test assertions, which the spec's open questions require.

---

## Appendix: Evidence

"Read" means read in the oracle checkout at 4a78a1d or a package source; "run" means a command run during the research for this plan (Julia 1.13.4, macOS, Apple M6, single-thread timings under load average 4.3–5.5, arms interleaved); "not checked" means neither. The scripts and raw outputs were in a scratch directory that is not durable.

**E1. Reference values from three codes** (run). Water STO-3G at the `test_eri_compare.rs:8-42` geometry with the oracle's `sto-3g.gbs:13-18, 75-84` data; H2 STO-3G at 1.4 bohr. Prototype (pure Julia MD, this plan's design), libcint 6.1.2 (through GaussianBasis 0.6.0 in a scratch environment, values divided by `sqrt(S_ii)` per index) and GaussianBasis.jl's own Julia integrals (Acsint):

| quantity | PySCF literal | prototype | libcint | Acsint |
|---|---|---|---|---|
| (O1s O1s\|O1s O1s) | 4.78506540 | 4.78506575 | 4.78506575 | 4.78506575 |
| (O1s O1s\|O2s O1s) | 0.74138035 | 0.74138032 | 0.74138032 | 0.74138032 |
| (O1s O1s\|H1 1s O1s) | 0.17176253 | 0.17176253 | 0.17176253 | 0.17176253 |
| (O1s O1s\|O2p_x O2p_x) | 1.11581381 (unasserted) | 1.11581382 | 1.11581382 | 1.11581382 |
| H2 S_AB | 0.65931812 | 0.65931821 | — | — |
| J[0,0] | 18.502906 | 18.502908 | 18.502908 | 18.502908 |
| K[0,0] | 9.697012 | 9.990743 | 9.990743 | 9.990743 |

Water STO-3G 8-fold symmetry violation in the prototype: 4.44e-16. The source of the last-digit differences between the PySCF literals and the three codes is not checked.

**E2. The K literal** (run). Exchange terms `(0l|0l)`, l = 1..5: 4.785065752, 0.136873367, 0.024477411, 0.024477411, 0.024477411. `2[(00|00) + (01|01) − (02|02) − (03|03) − (04|04)] = 9.697014` against the literal 9.697012. The oracle has the comment "PHASE FACTOR MISSING FROM MCMURCHIE-DAVIDSON" at `eri_general_md.rs:174` (read). That an earlier oracle build printed the literal is a hypothesis, not checked.

**E3. Boys accuracy** (run, spec 03 and research). This plan's design against a 1024-bit series, m ≤ 24: Taylor plus downward recurrence ≤ 6.6 ulp (worst t ∈ [30, 36)), ≤ 1.5 ulp for t < 2; large-t branch ≤ 6.7 ulp (t = 36, m = 13). The prototype's Boys at nmax = 16 over 115 t values: worst 6.66 ulp at t = 36, m = 13. The oracle algorithm's defects with a correctly rounded F_0 are spec 03 Facts' six figures; they depend on the bits of F_0 (E12), and `libm::erf` was not measured. Cost (run): `boys!` 9.9, 14.3, 12.2, 25.3 ns at nmax 0, 4, 8, 16 over random t ∈ [0, 40); `exp + sqrt + div` 5.7 ns.

**E4. Per-quartet timings** (run, prototype vs a Julia emulation of the oracle's per-component structure, which is not the oracle). Four distinct centers, ns per shell quartet, min of 7 rounds × 3 interleaved repetitions, default bounds checking:

| case | prims⁴ | prototype ns | emulation ns | emulation/prototype |
|---|---|---|---|---|
| (ss\|ss) K=3 | 81 | 1,566 | 7,770 | 5.0 |
| (pp\|pp) K=3 | 81 | 64,873 | 1,310,295 | 20.2 |
| (pp\|pp) K=1 | 1 | 1,154 | 16,215 | 14.0 |
| (dd\|dd) K=1 | 1 | 22,474 | 631,398 | 28.1 |
| (dd\|dd) K=3 | 81 | 762,278 | 49,876,167 | 65.4 |
| (ff\|ff) K=1 | 1 | 432,396 | 10,821,750 | 25.0 |

A second run agreed within 15 % ((dd|dd) K=1 23,781 ns; (ff|ff) 440,061 ns). Prototype vs emulation agreement: ≤ 1.12e-15 relative ((dd|dd) K=1), 1.50e-16 (ss|ss), 2.63e-16 (pp|pp), 3.75e-16 (dd|dd) K=3. With `--check-bounds=no` the prototype was 1.3–2.5× faster ((pp|pp) K=3 29.8 µs, (dd|dd) K=1 13.2 µs, (ff|ff) 217 µs); that run's emulation column moved by up to 3×, so no ratio is taken from it.

**E5. Benzene/6-31G integral sweep** (run). Planar benzene, C–C 1.39 Å, C–H 1.09 Å, the oracle's `6-31g.gbs:13-19, 74-87`: 42 shells, 66 AOs, 408,156 unique quartets, 386,937 after Schwarz 1e-12, 14,838,921 primitive quartets. Prototype, all surviving quartets, 1 thread: 1.907 s (integrals only, no J/K scatter, no density screening). Emulation of the oracle's structure on a 2,000-quartet sample: 5.0× slower, scaled to 9.8 s (estimate).

**E6. Roofline** (run). Single-core FP64 `dgemm` (OpenBLAS 0.3.30, 1 thread, n = 256 and 1024): 49.5–52.6 GFLOP/s; the prototype reaches 8–10 GFLOP/s on (dd|dd) and (ff|ff) and 0.1–3 GFLOP/s on (ss|ss) and (pp|pp). A Boys-only floor for benzene/6-31G: 14.8 M × ≈ 10 ns ≈ 0.15 s (estimate from two measurements).

**E7. Package closures** (run, each resolved alone in a fresh environment). SpecialFunctions 2.9.2: OpenSpecFun_jll 0.5.6+0 (excluded). GaussianBasis 0.6.0: libcint_jll 6.1.2+1 and OpenSpecFun_jll (48 packages). OhMyThreads 0.8.9, ChunkSplitters 3.2.0, TensorOperations 5.8.4, LoopVectorization 0.12.176: no non-stdlib `_jll`, no artifacts. Bessels 0.2.8 has no erf/erfc. Fermi 0.5.0 and ElemCo 0.16.0 depend on libcint_jll, JuliaChem 0.1.0 and Lints 0.4.1 on libint_jll (registry `Deps.toml`, read). No registered pure-Julia ERI or Boys package was found. libcint's license: not checked.

**E8. TrimCheck 0.1.5** (run, throwaway package). Pass: `boys!`, `hermite_R!` (dense prototype layout), the prototype shell-quartet kernel, `eigen(Symmetric(V))` with a cutoff and V^{-1/2}. Fail: plain `Threads.@sync`/`@spawn`, ChunkSplitters with `@spawn`, OhMyThreads `tforeach`; in all three the verifier itself raised `UndefVarError(:mapssavaluetypes, …)` while printing its finding, so what it found is unknown. PureMLIP's `ChunkWorkers` is validated by `PureMLIP/test/trim_tests.jl:53-85` and `test/juliac_build_test.jl:32-80` (read, not rerun).

**E9. Oracle facts the plan relies on** (read). MD early exits `eri_general_md.rs:46` (`> 45.0`), `:58` (`< 1e-18`), E skips `< 1e-15` at 154-172; (ss|ss) `eri_ssss.rs:15-64` and (ps|ss) `eri_vrr.rs:63-111` apply neither cut; the contraction-weight skip `< 1e-15` applies to all (`eri_contracted.rs:25-44, 157-171`). J/K loop `jk_symmetric.rs:68-71`, density screening `:82-87`, integral skip `< 1e-15` `:143`, rayon fold/reduce `:59-169`; `jk.rs:73-85` symmetrizes, `jk_symmetric.rs` does not. RI cutoffs `ri_j.rs:75` (`> 1e-10`), `ri_k.rs:60` (`> 1e-12`), `MAX_INCORE_3C_ELEMENTS = 64_000_000` (`ri_j.rs:15`). The oracle is Cartesian, l ≤ 5 (`basis/shell.rs:71-82`), each component normalized by `norm_for_ang` (`primitive.rs:49-60`); the prototype's after-contraction component scale agrees with the per-component form to ≤ 1.1e-15 relative. `ecp_shell_shell` is called at `scf/utils.rs:103`. Integral files: 4,470 lines under `src/integrals/` (`wc -l`).

**E10. R table size** (computed). Packed storage for all orders m = 0:L holds `binomial(L+4, 4)` entries: 10,626 for L = 20, 14,950 for L = 22; the m = 0 block holds `binomial(L+3, 3)` = 1,771 (L = 20). The prototype's dense layout needs `(L+1)^4` per order, 194,481 for L = 20.

**E11. Oracle timings** (read). No recorded timings in the oracle tree (`grep -i "time\|elapsed\|seconds"` over `benchmarks/` and `docs/`); no "74 s" figure in README, MANUAL or ARCHITECTURE. A release binary `target/release/quantum_engine` (4,505,808 bytes, Nov 25 02:21) exists and was not run (step-01 plan D17). The 74 s benzene PBE/6-31G figure is not checked.

**E12. Sensitivity of the oracle's upward Boys branch to F_0** (run, scratch `qe-plans/p03_oracleboys_sens.jl`, 1024-bit reference). Relative error of F_m after the upward recurrence `((2m−1)F − e^{−t})·(0.5/t)` with F_0 at −2, −1, 0, +1, +2 ulp from the correctly rounded value, and (last column) with `F_0 = 0.5·sqrt(π/t)·erf(sqrt t)` in Float64 with a correctly rounded `erf`:

| (t, m) | −2 | −1 | 0 | +1 | +2 | oracle expression |
|---|---|---|---|---|---|---|
| (1e-4, 2) | 7.31e-8 | 3.15e-8 | 1.01e-8 | 5.18e-8 | 9.34e-8 | 1.01e-8 |
| (1e-4, 4) | 115 | 49.6 | 16.0 | 81.5 | 147 | 16.0 |
| (1e-3, 4) | 0.013 | 6.46e-3 | 1.05e-4 | 6.67e-3 | 0.0132 | 0.013 |
| (0.01, 6) | 0.481 | 0.245 | 8.26e-3 | 0.228 | 0.465 | 0.228 |
| (0.1, 8) | 3.16e-3 | 1.52e-3 | 1.11e-4 | 1.74e-3 | 3.38e-3 | 3.38e-3 |
| (1, 8) | 8.83e-11 | 6.89e-11 | 2.05e-11 | 2.81e-11 | 4.76e-11 | 2.05e-11 |

**E13. `@kwdef` thresholds** (run, scratch `qe-plans/kwdef3.jl`, Julia 1.13). With the spec's `Base.@kwdef struct IntegralThresholds{T}` and `exponent_cut::T = 45`: `IntegralThresholds{Float64}()` and `IntegralThresholds{Float32}()` work (45.0, 45.0f0); `IntegralThresholds(density = 0.0)` throws `MethodError: no method matching IntegralThresholds(::Float64, ::Float64, ::Float64, ::Float64, ::Int64)`; `IntegralThresholds{Float64}(density = 0.0)` works.

**E14. Fixed-precision Boys reference** (run, scratch `qe-plans/p03_boysbig_hi.jl`). The series for F_0 at a fixed 4096 bits against `½√(π/t)`: relative error 0 at t = 1000, 8.9e-153 at 2500, 2.8e65 at 3000, Inf at 4000 and 20000.

**E15. The worker pool's chunk model** (read). `PureMLIP/src/model/workers.jl:3-5, 115-146`: `_run_chunks!(w, job, phase, n)` starts `n − 1` workers (`_start_workers!(w, n − 1)`), runs chunks `1:n−1` on them and chunk `n` on the calling task, one chunk per worker. `grep strict_exempt` over `PureMLIP/src` finds no exemption naming a worker function.

**E16. Step-01 and step-02 facts** (read). `Molecule{T,V}` takes any `AbstractVector{Atom{T}}` (`QuantumEngine/src/system/System.jl:127-140`); a ghost atom carries the basis of the element it stands for (`System.basis_element`). `QuantumEngine/Project.toml` `[deps]`: LinearAlgebra, StaticArrays, StrictMode (no Printf). The step-02 plan: `Shell{T}` accepts any `l ≥ 0` (line 226), `build_basis` throws only when it places an I shell (D2), `cartesian_component` throws `basis: l = 6 is above the supported l ≤ 5` (line 214), and the Float32-to-Float64 conversion rule for messages (line 127).
