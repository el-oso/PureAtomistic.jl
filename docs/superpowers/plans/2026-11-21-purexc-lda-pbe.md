# PureXC (LDA and PBE, CPU) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure-Julia package `PureXC/` that evaluates `lda_x`, `lda_c_pw`, `lda_c_pw_mod`, `gga_x_pbe` and `gga_c_pbe`, spin-restricted and spin-polarized, with derivatives of any order. It agrees with Libxc 7.0.0 to at least 6 significant digits, and it is faster than Libxc at orders 0–2.

**Architecture:** Each functional is an isbits parameter struct with one generic scalar energy-density function. Every derivative comes from that function evaluated on `PureTaylor.Taylor` numbers. Thresholds follow each spec's "PureXC rule". A checked point layer validates the inputs; the array layer validates the whole input first, then loops over the unchecked kernels so that LLVM can vectorize the loop. Libxc runs only in the test and bench environments.

**Tech Stack:**
- Julia 1.12 and 1.13.
- PureTaylor through `[sources]`, and StrictMode.
- Tests: TestItemRunner, StrictModeTest, TrimCheck, Aqua, ExplicitImports, OffsetArrays and ForwardDiff (as an oracle).
- Libxc.jl over Libxc 7.0.0, in `test/libxc/` and `bench/` only.

**Spec:** `docs/superpowers/specs/2026-11-20-xc-functionals-design.md` (approved), together with the functional specs `PureXC/specs/lda/{lda_x,lda_c_pw,lda_c_pw_mod}.md` and `PureXC/specs/gga/{gga_x_pbe,gga_c_pbe}.md`.

**Out of this plan** (later plans, each with its own approval):
- the KernelAbstractions extension (device array layer);
- the DftFunctionals wrapper;
- functional steps 3–6 of the design: PBEsol, revPBE, BLYP, SCAN, r2SCAN and hybrids. Each needs a room-1 spec first.

## Global Constraints

- **Two rooms.**
  - No implementer or reviewer of this plan opens Libxc's C or Maple source, or any file under the package directories of `Libxc` or `Libxc_jll`. Libxc is used only by calling Libxc.jl's Julia functions (`Libxc.Functional`, `Libxc.evaluate`, `Libxc.evaluate!`) and reading their outputs.
  - The inputs are the files in `PureXC/specs/` and the repository.
  - Every dispatch uses `PureXC/specs/prompts/implementer.md` or `reviewer.md` (Task 1).
- **Dependencies.**
  - No Python in any form. No `_jll` and no `Artifacts.toml` in the core closure.
  - `[deps]`: PureTaylor (`[sources] PureTaylor = {path = "../PureTaylor"}`) and StrictMode (`[compat] StrictMode = "0.5.10"`). `[compat] julia = "1.12"`.
  - Libxc.jl only in `PureXC/test/libxc/` and `PureXC/bench/`.
- **StrictMode policy.** Every module ends with `@strict_policy (:typestable, :noalloc)`. `@strict_exempt` is allowed only for entry points that allocate a workspace, narrowed to `:noalloc`, with a present-tense reason. PureXC is expected to have none.
- **Caller errors throw** `ArgumentError` with the message prefix `"xc: "`:
  - ρ < 0, σ < 0;
  - a NaN or infinite input;
  - σαβ² > σαα·σββ·(1 + 16·eps(T)), a rounding bound of kind (b) for σ's formed as 3-term dot products: each σ carries at most 3 eps, a product of two σ's at most 6 eps, and a factor of 2 is margin;
  - mismatched axes (`DimensionMismatch` for axes).
  - Nothing clamps a caller error. ρ = −0.0 is not negative and is treated as 0.
- **Thresholds.** Each functional follows its spec's "PureXC rule" exactly. Thresholds are fields of the functional, typed in its number type.
  - Float64 defaults are the spec values.
  - Float32 defaults are the Float64 values converted, with ε_ζ = eps(Float32), and any floor below floatmin(Float32) raised to floatmin(Float32).
  - Where a rule raises a value to a floor, only the value is raised. The series part stays, so the derivatives are those at the floor.
  - A factor replaced by the constant ε_ζ^p has no series part.
- **Accuracy (decided by the user).**
  - In Float64, relative error ≤ 1e-6 against Libxc at every grid point above the thresholds, for every output at orders 0–2.
  - Order 3 and above: the same floor against the same Julia code in 256-bit BigFloat.
  - Float32 has its own conditioning-based bound (Task 6).
  - Every numeric comparison states its oracle and its bound kind ((a) the oracle's stated tolerance, (b) a rounding bound c·eps·N, (c) a measured margin ≤ 100× the measured value) in a comment. Use `==` where the result is exact by construction.
- **Speed (decided by the user).**
  - Every functional, in both spin cases, at each order 0, 1 and 2, is faster in Float64 than Libxc.
  - The measure is `xc_evaluate!`, input validation included, against `Libxc.evaluate!`, by the interleaved A/B method of AGENTS.md.
  - The gate is the minimum speed-up over the rounds, which must be > 1.
  - Code that is slower is not accepted.
- **Float32 stays Float32.** A functional built for Float32 and given Float32 inputs never widens.
- **Indexing.**
  - Generic: `eachindex(ρ, σ)`, with outputs allocated or checked against the input axes.
  - No `@inbounds`.
  - Raw pointers and `@simd ivdep` only in the dense fast path of Task 8, with the conditions stated there.
- **Comments and records.**
  - Comments describe the code as it is (AGENTS.md): no history, no plan or task references.
  - Commits: subject ≤ 72 characters; trailers `Assisted-by: Claude <model> <noreply@anthropic.com>` and `Claude-Session: <url>`.
  - Run `runic -i PureXC` before every commit.
  - No machine names or local paths in tracked files.
  - Push to `main` only after review, and watch every CI job.

## Conventions fixed by this plan

- **Inputs.**
  - Restricted: `ρ::Real`, plus `σ::Real` for a GGA.
  - Polarized: `ρ::NTuple{2}` = (ρα, ρβ), plus `σ::NTuple{3}` = (σαα, σαβ, σββ).
  - The result type is `promote_type` of the functional's number type and the input types.
- **Outputs** of `xc_derivatives(f, Val(k), ρ[, σ])`: a `NamedTuple` with every order from 0 to k.
  - **Order 0** is the field `e`, the energy per volume (e = ρ·ε).
  - **Order j ≥ 1:** one field for each split (a, b) with a + b = j, a descending.
    - The name is `"v" * (j > 1 ? "$j" : "")`, then `"rho" * (a > 1 ? "$a" : "")` if a > 0, then `"sigma" * (b > 1 ? "$b" : "")` if b > 0.
    - Examples: `vrho`, `vsigma`, `v2rho2`, `v2rhosigma`, `v2sigma2`, `v3rho2sigma`.
  - **Restricted:** each field is a scalar.
  - **Polarized:** each field is a tuple.
    - The ρ part lists all multisets of size a over (α, β) in lexicographic order. The σ part lists all multisets of size b over (αα, αβ, ββ) in lexicographic order. The ρ part is the outer loop.
    - Example: `v2rhosigma` = (α·αα, α·αβ, α·ββ, β·αα, β·αβ, β·ββ).
    - The order-1 and order-2 fields are checked against Libxc's outputs in Task 6, so a layout error fails there.
- **Seeds.** Every input is a PureTaylor variable with scale s = max(|x|, r, floatmin(T)^(1/K)), where K is the derivative order and r is the input's natural size at that point: the total density n for a density input, and 4(3π²)^(2/3)·n^(8/3) (the σ at which the reduced gradient is 1) for a σ component. The last term keeps every s^k, k ≤ K, a normal number. The scale changes only the representation; the derivatives are read with `derivatives(a, 1 ./ s)`.
- **Polarized correlation** is computed in (ρ, ζ, σ) and converted to the density variables with `PureTaylor.substitute`. Polarized exchange uses the exact spin scaling, channel by channel.

## Review Focus

1. **A fully polarized point** (ρβ = 0, σββ = σαβ = 0) gives finite values for every output at every order, equal to the rule's floor behavior. Today's naive formulas give NaN there.
2. **ρ = 0 and ρ = −0.0** (restricted and polarized) give exact zeros at every order, never NaN.
3. **σ = 0 exactly** (s = 0): every derivative is finite, and `vsigma` equals its limit. The seed scale uses the floor, never 0.
4. **Float32 inputs with a Float64 functional** promote to Float64. `pbe_x(Float32)` with Float32 inputs stays Float32, and its order-3 values are infinite only where the BigFloat value exceeds floatmax(Float32).
5. **The σαβ bound:** σαβ = −√(σαα σββ) exactly is accepted; one rounding bound beyond it throws, with the `"xc: "` message.

Each line has a pinning test in the task that owns the code (Tasks 3, 4 and 5).

---

### Task 1: Package skeleton and room-2 prompts

**Files:**
- Create: `PureXC/Project.toml`, `PureXC/src/PureXC.jl`, `PureXC/test/Project.toml`, `PureXC/test/runtests.jl`, `PureXC/test/quality_tests.jl`, `PureXC/specs/prompts/implementer.md`, `PureXC/specs/prompts/reviewer.md`.
- Modify: `.github/workflows/CI.yml` (test axis only).

**Interfaces:**
- Produces: module `PureXC`; abstract types `AbstractFunctional{T}`, with subtypes `AbstractLDA{T}` and `AbstractGGA{T}`; `family(f)` returning `:lda` or `:gga`; `kind(f)` returning `:exchange` or `:correlation`.

- [ ] **Step 1: Prompts.**
  - `implementer.md` and `reviewer.md` state the two-room rule of Global Constraints in the reader's terms:
    - which files may be read;
    - that Libxc is reached only through the three Libxc.jl calls;
    - what to do when a spec leaves something open: stop and report it, never look it up in Libxc.
  - They point to the spec format (design, "Two-room spec process").
  - They have placeholders `<task-brief>`, `<report-file>` and `<specs>`.
  - They must not contain any text from Libxc.
- [ ] **Step 2: Failing tests** in `quality_tests.jl`:
  - `Aqua.test_all(PureXC)`;
  - ExplicitImports checks as in PureTaylor;
  - an item "core closure is binary-free", copied from PureMLIP's item, run against `PureXC`;
  - an item asserting that `Libxc` is in neither `PureXC/Project.toml` nor `PureXC/test/Project.toml`.
- [ ] **Step 3: Implement** the skeleton.
  - The test environment mirrors `PureTaylor/test/Project.toml`, plus OffsetArrays and PureTaylor by `[sources]` path.
  - The module ends with `@strict_policy (:typestable, :noalloc)`.
  - Add `PureXC` to CI.yml's `package` test axis.
- [ ] **Step 4: Run** `julia --project=PureXC -e 'using Pkg; Pkg.test()'` on 1.13 and 1.12. Expect all tests to pass.
- [ ] **Step 5: Commit** `Add the PureXC package skeleton and room-2 prompts`.

### Task 2: Validation, thresholds and the derivative driver

**Files:**
- Create: `PureXC/src/validate.jl`, `PureXC/src/thresholds.jl`, `PureXC/src/driver.jl`, `PureXC/test/driver_tests.jl`.

**Interfaces:**
- Produces:
  - `Thresholds{T}(density, sigma, zeta)`. `sigma` is the reported σ threshold, and the floor is `sigma^2`.
  - `default_thresholds(::Type{T}, density, sigma, zeta)` applies the Float32 rule of Global Constraints.
  - `_raise(x, floor)` raises the value to the floor and keeps the series part, for a `Real` or a `Taylor`.
  - `_check_inputs(f, ρ[, σ])` throws the caller errors.
  - `xc_derivatives(f, Val(k), ρ[, σ])` is the checked point entry. It calls `_check_inputs`, seeds the variables, calls the functional's `_energy_density(f, ρ[, σ])` on Taylor numbers, and builds the output `NamedTuple` per Conventions.
  - `energy_density(f, ρ[, σ]) = xc_derivatives(f, Val(0), ρ[, σ]).e`.
  - `_xc_derivatives_unchecked` is the same without the check, for the array layer.
- Consumes: `PureTaylor.variable`, `derivatives`, `value`, `substitute`.

- [ ] **Step 1: Failing tests.**
  - **Error messages** (`@test_throws "xc: "`): ρ = −1e-300, σ = −1e-300, NaN and ±Inf in every input position; σαβ just beyond the bound.
  - **Accepted:** σαβ = −√(σαα σββ) computed in Float64 (Review Focus 5); ρ = −0.0.
  - **Field names** at k = 0…3, restricted and polarized, for a test-only functional `_TestGGA` with e = ρ²σ³ + ρ·σ (defined in the test file, subtype of `AbstractGGA`). The values are compared with `==` against the hand-differentiated polynomial at ρ = 2, σ = 3, since the arithmetic is exact in Float64 there.
  - **Polarized layout** against a polynomial whose every mixed partial has a distinct value.
  - **`_raise`** keeps the series: `value(r) == 1e-15` and `PureTaylor.coefficient(r, (1,)) == 1e-15` for `r = _raise(variable(Val(1), Val(2), Val(1), 1e-20, 1e-15), 1e-15)`: the value moves to the floor and the first-order coefficient (the seed scale) is unchanged.
- [ ] **Step 2: Implement.** Generate the name and layout tables at generation time from `Val(k)` and `Val(N)`; no runtime strings. Read the Taylor coefficients with `PureTaylor.multi_indices` order and map them into the fields per Conventions.
- [ ] **Step 3: Run** the suite. Expect it to pass, and `@test_noalloc` on `xc_derivatives(_TestGGA(), Val(2), (1.0, 2.0), (0.1, 0.0, 0.2))`.
- [ ] **Step 4: Commit** `Validate XC inputs and map Taylor results to outputs`.

### Task 3: LDA functionals

**Files:**
- Create: `PureXC/src/lda/lda_x.jl`, `PureXC/src/lda/pw92.jl`, `PureXC/test/lda_tests.jl`.

**Interfaces:**
- Produces:
  - `LDAExchange{T}(thr)` with `lda_x(T = Float64)`.
  - `PW92Correlation{T}(params, fz20, thr)` with `lda_c_pw(T = Float64)` and `lda_c_pw_mod(T = Float64)`.
  - `_energy_density` methods for both, restricted and polarized.
  - `_pw92_eps(pw, rs, ζ)`: ε_c of the PW92 spec (E4), exported to Task 4 as an internal function.
- Consumes: Task 2.

- [ ] **Step 1: Failing tests**, from the specs alone.
  - **Reference values:** every row of each spec's "Reference values" table, at orders 0–1. Tolerance (a): 1e-6 relative, the user's floor. The comment states it, and states that the spec's measured difference is ≤ 9e-15.
  - **BigFloat agreement:** each functional evaluated on `BigFloat` inputs with a `BigFloat` functional matches the Float64 result, orders 0–3, at the specs' grid (n log-spaced over [1e-6, 1e3], ζ ∈ {0, 0.25, −0.6, 0.95}). Bound (b): c·eps·N, with c and N stated.
  - **Rule items:** every case in each spec's section 7, "Libxc behavior":
    - the cut-offs at n = 0.999e-15 and n = 1.0e-15;
    - the identical outputs at (0.01, 0), (0.01, 0.5e-15) and (0.01, 1e-15);
    - for `lda_x`, the per-channel zero and the restricted zero for n ≤ 2 n_thr.
  - **Review Focus items 1 and 2** for these functionals.
  - **PW92 constants:** `lda_c_pw` uses f''(0) = 1.709921. `lda_c_pw_mod` uses `4/(9(cbrt(big(2)) − 1))` rounded to T, and the comment states the value 1.7099209341613657.
- [ ] **Step 2: Implement** from the specs' equations, citing their equation labels (E1)… in comments. Use the `fast_*` functions of PureTaylor for cbrt, x^(−1/3), log and log1p.
- [ ] **Step 3: Run.** Expect all to pass, and `@test_noalloc` on `xc_derivatives` for each functional at `Val(2)`, in Float32 and Float64.
- [ ] **Step 4: Commit** `Add LDA exchange and PW92 correlation`.

### Task 4: PBE exchange and correlation

**Files:**
- Create: `PureXC/src/gga/pbe_x.jl`, `PureXC/src/gga/pbe_c.jl`, `PureXC/test/pbe_tests.jl`.

**Interfaces:**
- Produces:
  - `PBEExchange{T}(κ, μ, thr)` with `pbe_x(T = Float64)`.
  - `PBECorrelation{T}(β, γ, B, pw::PW92Correlation{T}, thr)` with `pbe_c(T = Float64)`, built on `lda_c_pw_mod(T)`.
  - Their `_energy_density` methods.
- Consumes: Tasks 2 and 3 (`_pw92_eps`).

- [ ] **Step 1: Failing tests**, as in Task 3, from `gga_x_pbe.md` and `gga_c_pbe.md`:
  - the reference-value rows;
  - BigFloat agreement at orders 0–3, with s ∈ {0} ∪ log-spaced [1e-3, 10];
  - every section-7 rule case, including the σ floor of σ_thr² and the correlation's single cut-off at n < n_thr;
  - Review Focus items 1–3.
  - **Parameter gradient:** `pbe_x` built with `κ` as a `Taylor` variable gives ∂e/∂κ equal to a central difference in BigFloat (step 1e-20, bound (c)).
  - **Review Focus 4:** `pbe_x()` with Float32 inputs returns Float64. `pbe_x(Float32)` with Float32 inputs returns Float32, with `@inferred`.
- [ ] **Step 2: Implement** from the specs. Polarized correlation goes through (ρ, ζ, σ) and `substitute` (Conventions).
- [ ] **Step 3: Run.** Expect all to pass, plus `@test_noalloc` as in Task 3.
- [ ] **Step 4: Commit** `Add PBE exchange and correlation`.

### Task 5: Array layer and composites

**Files:**
- Create: `PureXC/src/array.jl`, `PureXC/src/composite.jl`, `PureXC/test/array_tests.jl`.

**Interfaces:**
- Produces:
  - `xc_evaluate!(out::NamedTuple, f, Val(k), ρ[, σ])`.
    - Restricted inputs are `AbstractVector`s.
    - Polarized inputs are a 2×n `AbstractMatrix` for ρ and a 3×n `AbstractMatrix` for σ. Spin is dimension 1, as in Libxc.jl.
    - `out` has exactly the fields of `xc_derivatives` at order k. Each field is an array whose trailing axis matches the inputs, with first-dimension length equal to the number of components.
    - It validates every point first (throwing as `_check_inputs` does), then loops over `eachindex` and calls `_xc_derivatives_unchecked`.
  - `xc_outputs(f, Val(k), ρ[, σ])` allocates `out` with `similar`.
  - `XC(terms::Tuple{Vararg{Tuple{Real,AbstractFunctional}}})`, with `energy_density`, `xc_derivatives` and `xc_evaluate!` summing the terms.
  - `pbe(T = Float64) = XC(((1, pbe_x(T)), (1, pbe_c(T))))`.
  - `zk(e, ρ)`, a helper that returns e/ρ, or 0 where ρ = 0.
- Consumes: Tasks 2–4.

- [ ] **Step 1: Failing tests.**
  - **Point agreement:** `xc_evaluate!` equals `xc_derivatives` point by point (`==`) for every functional at k = 0…3.
  - **Generic arrays:** `OffsetArray` and `view` inputs give the same values, and the output axes follow the input axes.
  - **Validation:** a single negative ρ in the middle of 1000 points throws before any output is written. The outputs keep a sentinel value.
  - **Axes:** mismatched axes throw `DimensionMismatch`.
  - **Composite:** `pbe()` equals the sum of its terms.
  - **Allocation:** `@test_noalloc` on `xc_evaluate!` with dense `Vector`/`Matrix` inputs.
- [ ] **Step 2: Implement.** The loop body is the unchecked point call only, so that LLVM can vectorize it. Check `<2 x double>` in the LLVM IR for `pbe_x` at k = 1 in a test item ("array loop vectorizes").
- [ ] **Step 3: Run**, then **commit** `Add the array layer and composite functionals`.

### Task 6: Oracle comparison against Libxc

**Files:**
- Create: `PureXC/test/libxc/Project.toml` (Libxc.jl, with Libxc 7.0.0 pinned through compat), `PureXC/test/libxc/runtests.jl`, `PureXC/test/libxc/grid.jl`.
- Modify: `.github/workflows/CI.yml` (a `purexc-libxc` job, Julia 1.13).

**Interfaces:**
- Consumes: the public API; `Libxc.Functional(id; n_spin)`; `Libxc.evaluate(f; rho, sigma, derivatives = 0:2)`.

- [ ] **Step 1: Grid** (design, "Validation").
  - ρ log-spaced over [1e-10, 1e3] (131 points).
  - s ∈ {0} ∪ log-spaced [1e-3, 10].
  - ζ in a set including ±(1 − ε_ζ), ±1, 0, 0.25, −0.6 and 0.95.
  - Each point is labeled "above" when n ≥ n_thr, every channel n_σ > n_thr, every σ_σσ > σ_thr² and 1 ± ζ > ε_ζ; otherwise "below". The two populations are reported separately.
- [ ] **Step 2: Float64 items**, for each functional and spin case, at orders 0–2.
  - **(A)** At every "above" point: |jl − libxc| ≤ 1e-6·|libxc| for every component (the user's floor, kind (a)). Where the BigFloat value of a component is exactly 0, the requirement is |jl| ≤ 1e-6·m instead, where m is the largest |BigFloat value| among the components of that field at that point.
  - **(B)** |jl − big| ≤ c·max(|libxc − big|, 4·eps·|big|), with c measured and stated (kind (c)).
  - **"Below" points:** printed, not gated. Their behavior is gated by the rule items of Tasks 3–4.
  - The comparison covers the layout of every order-1 and order-2 field.
- [ ] **Step 3: Order 3, and order 4 on a 10-point subgrid.** |jl − big|/|big| ≤ 1e-6 in Float64 (the floor), with the same zero rule. Add a loose check against Richardson differences of Libxc's order 2, with tolerance 1e-3 (kind (a): their stated accuracy is 1e-6 to 1e-4).
- [ ] **Step 4: Float32 items.**
  - The point tolerance is c·eps(Float32)·max(1, κ̂), where κ̂ = |jl64 − big| / (eps(Float64)·|big|) is the Float64 error in ulps at that point (the formula's measured conditioning). c is measured and stated (kind (c)).
  - Values that are infinite are accepted only where |big| > floatmax(Float32).
  - Every Float32 result on the grid must be finite or meet that condition. If the converted thresholds of Global Constraints break this, stop and report the measured case.
- [ ] **Step 5: Run** `julia --project=PureXC/test/libxc PureXC/test/libxc/runtests.jl`. Every item passes, except Float32 polarized correlation at order 3, which goes to Task 7 as `@test_broken`, with its measured maximum stated.
- [ ] **Step 6: Commit** `Compare PureXC with Libxc on the oracle grid`.

### Task 7: Float32 polarized correlation at order 3

**Files:** whatever the cause needs. The report goes to the coordinator first.

- [ ] **Step 1: Find the cause** (decided by the user: find the cause first).
  - Locate the worst points of Task 6's broken item.
  - Evaluate each intermediate of `pbe_c` in Float32 and in BigFloat on those points.
  - Name the first intermediate whose relative error exceeds 100·eps(Float32), and measure the conditioning of that step.
  - Report the measurements and **stop** for the coordinator's ruling before changing any code or bound.
- [ ] **Step 2: Apply the ruling.**
  - If the step is ill-conditioned by the formula, document it as the Float32 bound, with the measured κ.
  - Otherwise, fix the evaluation (for example a reordered or factored form, checked against BigFloat) and turn the `@test_broken` into `@test`.
- [ ] **Step 3: Commit** with a message that states the cause.

### Task 8: Speed gate

**Files:**
- Create: `PureXC/bench/Project.toml`, `PureXC/bench/libxc_speed.jl`, `PureXC/bench/results/`.

- [ ] **Step 1: Bench.** Use the method of `PureMLIP/bench/ab_compare.jl`:
  - each arm runs in its own child process, with one thread;
  - 7 interleaved rounds, with the arm order alternating;
  - 10^6 points from the oracle grid's "above" population;
  - a calibration loop discards disturbed pairs.

  The arms are PureXC `xc_evaluate!` (validation included) and `Libxc.evaluate!`. Both use preallocated outputs and orders 0…k for k ∈ {0, 1, 2}. The bench runs both spin cases, for all five functionals and `pbe()`. Float32 is reported against Libxc's Float64 time, and is not gated. The record is a JSON file in `bench/results/` naming the hardware class and the Julia, PureXC and Libxc versions.
- [ ] **Step 2: Gate.** Every Float64 cell needs a minimum speed-up over the rounds of > 1.
  - **A failing cell:** profile it and optimize the code, by measured steps.
  - **The allowed dense fast path:** a method for `Array` arguments that reads and writes through raw pointers under `@simd ivdep`, after checking that no output shares memory with another argument (`Base.mightalias`). It is kept only if an interleaved benchmark against the generic loop shows the gain, and the test suite runs under `--check-bounds=yes`.
  - **When to stop:** if a cell still fails, stop and report it with its measurements. No cell is accepted below 1.
- [ ] **Step 3: Commit** `Add the Libxc speed comparison and its record`. Quote the table (speed-up = Libxc time / PureXC time, median [min, max]) in the commit body.

### Task 9: StrictMode gates, TrimCheck, coverage and mutations

**Files:** as in PureTaylor: `PureXC/test/strict_tests.jl`, `PureXC/test/strict/{check.jl,workload.jl}`, `PureXC/test/coverage/`, `PureXC/test/mutation/`, CI.yml (coverage) and mutation.yml.

- [ ] **Step 1:** Add the StrictMode gates exactly as PureTaylor has them:
  - `test_compiled` for `:typestable`;
  - an `audit` sweep plus the AllocCheck proof for `:noalloc` and `:noboxing`;
  - `test_signatures` for the methods on foreign functions;
  - a failing control for each gate, with the guarantee asserted from the finding itself;
  - a failing control for the unlisted-function check.

  If the scan reports a finding that the proof does not, stop and report it, with no exemption.
- [ ] **Step 2:** Add TrimCheck items for `energy_density`, `xc_derivatives` and `xc_evaluate!` of every functional, in Float32 and Float64.
- [ ] **Step 2b:** A juliac item (tag `:juliac`, run only with `PUREXC_JULIAC_BUILD=1`, as PureAtomistic's) builds a small program with `--trim=safe` that evaluates `xc_evaluate!` for `pbe()` at k = 2, and requires bit-identical output to the untrimmed run.
- [ ] **Step 3: Coverage rows** for every `src` file, at 1.00 unless a line cannot run at run time and the row says why.
- [ ] **Step 4: Mutations** for:
  - every constant of every spec's section 5;
  - every threshold comparison (`<` against `≤`, and the floor's square);
  - the σαβ bound;
  - the output layout map;
  - the ζ-variable change.

  Each one must be killed by the item named in its `expect`, in the main suite or the Libxc suite. The mutation runner can run the Libxc suite as a second command.
- [ ] **Step 5: Run** every gate on 1.13 and 1.12 with `--check-bounds=yes`. Then **commit** `Gate PureXC with StrictMode, TrimCheck and mutations`.

### Task 10: Documentation and repository rules

**Files:** `PureXC/docs/` (DocumenterVitepress, deployed under `PureXC/`), `.github/workflows/docs.yml`, `AGENTS.md` and `README.md`.

- [ ] **Step 1: Docs.** The docs cover:
  - the three layers;
  - the output names and the polarized layout;
  - the thresholds rule of each functional;
  - the accuracy tables measured in Task 6 and the speed table of Task 8, with their sources;
  - the two-room process, pointing to `PureXC/specs/`;
  - an API page.

  Every exported name has a docstring.
- [ ] **Step 2: AGENTS.md.**
  - Four packages.
  - PureXC's `[deps]` rule, and its Libxc-only test and bench environments.
  - The two-room rule as a repository rule, with its reason (MIT code; Libxc is MPL-2.0).
  - Its gates.
  - The output-name and layout conventions.
- [ ] **Step 3:** Build the docs locally. Then **commit** `Document PureXC and add its repository rules`.
