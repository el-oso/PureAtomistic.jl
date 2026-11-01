# MACE speed and memory on CPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `mace_energy_forces!` faster and its workspace about 3× smaller, add a small energy-only workspace, and test and measure PureMLIP with `PureBLAS.activate()`.

**Architecture:** Vectorize the two hot kernel families (tensor-product pullback, symmetric contraction) where the bounds check blocks SIMD; then shrink `MACEWorkspace` in three steps (alias buffers by lifetime, recompute per-edge data per chunk, run the per-atom linear layers per block of receivers). `PureBLAS.activate()` redirects `mul!`, so PureMLIP uses PureBLAS with no code change; a separate Julia 1.13 test environment checks it. Every step is gated by an interleaved A/B ratio against the previous main, by bytes per atom, and by the unchanged LAMMPS tolerances.

**Tech Stack:** Julia 1.12/1.13, LinearAlgebra, PureBLAS (via `activate()`, Julia 1.13, test environment only), Chairmarks, StrictMode/StrictModeTest 0.5.6, TrimCheck, TestItems.

**Spec:** `docs/superpowers/specs/2026-10-30-pureatomistic-design.md` (sections 4, 6.1, 6.3, 9, 13). Measurements behind every number below: `plan4-recon.md` (kept outside the repo; facts restated here where a task needs them).

## Global Constraints

- PureMLIP `[deps]` stay `Inflate, LinearAlgebra, PrecompileTools`; PureMLIP has no dependency on PureBLAS. Users switch with `using PureBLAS; PureBLAS.activate()`.
- PureBLAS requires Julia 1.13; PureMLIP supports 1.12. PureBLAS must not enter `test/Project.toml`; it lives in `PureMLIP/test/pureblas/Project.toml` (`[sources]` by URL + commit), run only in a 1.13 CI job.
- Speed is gated only by **interleaved A/B ratios on one run** (task 1's harness), never absolute times: 1 thread, `BLAS.set_num_threads(1)`, swap must not grow during a pair, median over at least 5 pairs.
- Accuracy: the LAMMPS reference items pass with unchanged tolerances (Float64 1e-6 eV target; Float32 margins as documented in `model_tests.jl`/`forces_tests.jl`).
- Allocation-free after warm-up (`@test_noalloc`) for every evaluation entry point in Float64 and Float32; TrimCheck `@validate`, `test_signatures(:typestable)` with explicit `Base.return_types` checks, `conversion_report` and `resource_leak_report` gates cover every new public type and function.
- `@inbounds` only where measured: each one wraps the minimal inner loop, carries a one-line comment with its local proof (the index comes from `eachindex` over exactly the arrays indexed), and the commit body cites the ratio with and without it. The suite runs under `--check-bounds=yes` (CI already does).
- Public API unchanged except additions: `MACEWorkspace{T}(model, n_atoms, n_edges; edge_chunk = 4096)` keeps working.
- New src files get a coverage row and a mutation entry. Comments and commit messages state present facts only.

## Review Focus

1. **Edge order within a receiver block:** block fusion (task 6) relies on edges sorted by receiver; a `NeighborList` built by hand or by another tool may not be sorted — expect a clear `model:` error, not wrong energies. Test in task 6.
2. **Block size larger than the atom count or not dividing it** (e.g. 7 atoms, block 256; 500 atoms, block 64): results must equal the unblocked ones within rounding. Test in task 6.
3. **PureBLAS active while the Dual-number tests run** (the composed-pullback tests push `ForwardDiff.Dual` through `mul!`): generic `mul!` on non-BLAS element types does not reach PureBLAS, so they must still pass. Test in task 8 by running the composed-pullback item in the PureBLAS environment.
4. **Edge count growing within capacity after a `neighbor_list!` rebuild** (the documented repeated-evaluation sequence) must still work with the pooled and blocked buffers. Test in task 6 (extend the existing docs-sequence test).
5. **Single atom / zero edges** through every new path (pooled, recomputed, blocked, energy-only): energy = E0, zero forces, no error. Test in tasks 6 and 7.

---

### Task 1: Interleaved A/B timing harness

**Files:**
- Create: `PureMLIP/bench/ab_compare.jl`
- Create: `PureMLIP/bench/results/ab_compare_mac.json` (written by the run in Step 3)

**Interfaces:**
- Produces: `PUREMLIP_AB_BASE=<git ref> PUREMLIP_BENCH_CASES="5:64,5:32" [PUREMLIP_AB_LOAD=PureBLAS] julia --project=PureMLIP/bench PureMLIP/bench/ab_compare.jl` → JSON with per-pair `head/base` ratios for `mace_energy_forces!` and `mace_energy!`, their median and min–max, swap before/after each pair, both commit SHAs, CPU string, Julia version. Every later task's speed gate is a run of this script with `PUREMLIP_AB_BASE` = the commit before the task.

- [ ] **Step 1:** Implement: `git worktree add` the base ref under a temporary directory (removed at exit); one child-process environment per side that `[sources]`-points PureMLIP at that side's tree; alternate child processes base, head, base, head, … (each warms up, then reports the median of 3 evaluations); discard a pair if `vm.swapusage` grew; no host names or paths in the JSON.
- [ ] **Step 2:** A/A check: `PUREMLIP_AB_BASE=HEAD` — Expected: median ratio within 1.00 ± 0.02 for both cases and both entry points.
- [ ] **Step 3:** Commit `Add an interleaved A/B timing harness` with the A/A table in the body.

### Task 2: Vectorize the tensor-product kernels

**Files:**
- Modify: `PureMLIP/src/model/tensor_product.jl:61-100` (`tp_edge_add!`, `tp_edge_pullback_add!`), `PureMLIP/src/model/evaluate.jl` (`_messages_pullback!`)
- Test: `PureMLIP/test/layers_model_tests.jl`

**Interfaces:**
- Produces: `tp_edge_pullback_add!(…; need_dx::Bool = true)`; `_messages_pullback!` passes `need_dx = false` for interaction 1, whose `dx` is never read.

- [ ] **Step 1:** Add a test item `"TP pullback without dx matches the full pullback"`: same `dY` and `dw` with `need_dx = false` as with `true` (`==`), and `dx` untouched (fill with NaN before; still NaN after).
- [ ] **Step 2:** Run it: FAIL (`need_dx` keyword unknown).
- [ ] **Step 3:** Rewrite the inner loops: hoist column views (`dMk`, `xi`, `dxi` per entry; `wp`, `dwp` per path); inner loop `@inbounds @simd for u in eachindex(dMk, wp, xi, dxi, dwp)` with a scalar `acc` accumulated then `dY[j] += v * acc`. Same column-view shape plus `@inbounds` in `tp_edge_add!`. Add `need_dx`.
- [ ] **Step 4:** Run the layer items, the full suite with coverage, and check `@code_llvm` of `tp_edge_pullback_add!` shows `<2 x double>` (Float64) and `<4 x float>` (Float32) multiplies in the inner loop. Expected: PASS; vector types present.
- [ ] **Step 5:** A/B vs the commit before the task. Gate: Float64 ≤ 0.80, Float32 ≤ 0.71 (recon measured 0.777 / 0.686). Also record the ratio with the `@inbounds` removed (column views only; recon: 0.924 / 0.933).
- [ ] **Step 6:** Commit `Vectorize the tensor-product kernels` with both ratios in the body.

### Task 3: Vectorize the symmetric contraction; keep σ in the radial MLP

**Files:**
- Modify: `PureMLIP/src/model/symmetric_contraction.jl:145-155, 218-228`, `PureMLIP/src/model/radial_mlp.jl:70-84, 108-118`
- Test: `PureMLIP/test/symmetric_contraction_tests.jl`, `PureMLIP/test/layers_model_tests.jl`

**Interfaces:**
- Produces: after `radial_mlp!`, `pre` holds σ(pre-activation) (the logistic), not the pre-activation; `radial_mlp_pullback!` computes the slope as `cst * σ + act * (1 - σ)` from it (`act = cst * z * σ`). Docstrings of both state this.

- [ ] **Step 1:** Add `@test` that after `radial_mlp!` the `pre` buffer equals `1 ./ (1 .+ exp.(-z))` for the hidden pre-activations `z` (computed independently from the weights) to `4eps`.
- [ ] **Step 2:** Run it: FAIL.
- [ ] **Step 3:** Implement: same column-view + `eachindex` + `@inbounds` shape over `c` for the U3 loops (lines 145–147, 226–228) and the still-hot lines 153–155, 218–224; `_mlp_layer!` stores σ.
- [ ] **Step 4:** Covering items + full suite: PASS (existing ForwardDiff/finite-difference pullback tests unchanged).
- [ ] **Step 5:** A/B vs main at plan start (596b5b2). Gate: Float64 ≤ 0.75, Float32 ≤ 0.58 (recon 0.728 / 0.558 for tasks 2+3 together).
- [ ] **Step 6:** Commit `Vectorize the symmetric contraction; keep the logistic in the radial MLP`.

### Task 4: Alias workspace buffers by lifetime (option A)

**Files:**
- Modify: `PureMLIP/src/model/evaluate.jl` (`MACEWorkspace`, its constructor, the forward and backward bodies)
- Test: `PureMLIP/test/forces_tests.jl`, `PureMLIP/test/strictmode_tests.jl`

**Interfaces:**
- Produces: `MACEWorkspace` fields regrouped into pools — one 40-component pool shared by `M[1]`, `M[2]`, `dM[2]`, `dM[1]`; two 16-component pools for `L0`/`dX2`/`dX1` and `dL0` (forward-only small buffers `h0`, `h1`, `B`, `sc`, `h2` alias into the second); persistent `X1`, `L2`, `x1`, `x2`; no `dx[1]`. Constructor signature unchanged. `ddensity₀ = -⟨dX1, X1⟩ / q` (valid because `X1ᵢ = W_{sᵢ} L0ᵢ` is a per-atom linear map).
- Produces (test helper): `workspace_bytes_per_atom(model, T, n_atoms, n_edges; kw...)` in `test/testutils.jl`, building a workspace without evaluating and returning `Base.summarysize(ws) / n_atoms`.

- [ ] **Step 1:** Add a test item `"workspace memory per atom"`: at cu_bench(14) sizes (10,976 atoms, 856,128 edges, `edge_chunk = 4096`), `workspace_bytes_per_atom(m64, Float64, …) <= 180_000` (recon estimate 174.3 kB; main 295.2 kB).
- [ ] **Step 2:** Run: FAIL (295.2 kB).
- [ ] **Step 3:** Implement the pools and the ddensity₀ identity.
- [ ] **Step 4:** Full suite with coverage: PASS. LAMMPS items unchanged; `@test_noalloc` both precisions; `resource_leak_report` gate unchanged.
- [ ] **Step 5:** A/B vs the commit before the task. Gate: ratio ≤ 1.01.
- [ ] **Step 6:** Commit `Share workspace buffers by lifetime`.

### Task 5: Recompute per-edge data per chunk (option B)

**Files:**
- Modify: `PureMLIP/src/model/evaluate.jl` (`_edges!`, `_messages!`, `_messages_pullback!`, the force loop)

**Interfaces:**
- Produces: per edge only `r` (3), `d` (1) and `g` (3, the edge-vector gradient) persist; `Y`, `feats`, `dfeats` are recomputed per chunk into chunk buffers; `dY` and `dd` are folded into `g` through `dYdr` inside each pullback.

- [ ] **Step 1:** Tighten the memory item to `<= 150_000` (recon estimate 143.2 kB).
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Full suite with coverage: PASS.
- [ ] **Step 5:** A/B vs the commit before the task. Gate: ratio ≤ 1.02.
- [ ] **Step 6:** Commit `Recompute per-edge features per chunk`.

### Task 6: Receiver-block fusion (option C)

**Files:**
- Create: `PureMLIP/src/model/blocks.jl` (receiver-block ranges and their iteration)
- Modify: `PureMLIP/src/model/evaluate.jl`, `PureMLIP/src/model/Model.jl` (include), `test/coverage/thresholds.toml`, `test/mutation/mutations.toml`
- Test: `PureMLIP/test/forces_tests.jl`, `PureMLIP/test/tiny_model_tests.jl`

**Interfaces:**
- Produces: `MACEWorkspace{T}(model, n_atoms, n_edges; edge_chunk = 4096, block = 256)`. `M`, `dM`, `L0`, `dL0` live in a block buffer of `block` atoms; `_messages!` + `linear!` + `/q` (forward) and `linear_pullback!` + `_messages_pullback!` (backward) run per block of receivers; edge chunks end on receiver boundaries.
- Fails fast: `model: neighbor list edges must be sorted by receiver` when they are not.

- [ ] **Step 1:** Tests:
  - memory item `<= 100_000` (Float64) and `<= 50_000` (Float32) (recon estimate 87.2 kB);
  - energies, forces, virial for cu_bench(5) with `block` ∈ (1, 7, 64, 256, 10_000) equal the `block = n_atoms` result within `10eps(T)·1280` relative;
  - an unsorted hand-built `NeighborList` throws the message above (`@test_throws "model: neighbor list edges must be sorted by receiver"`);
  - single atom / zero edges: energy == E0, forces all zero;
  - the documented repeated-evaluation sequence (edge count grows within capacity) passes with `block = 64`.
- [ ] **Step 2:** Run: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Full suite with coverage, coverage gate, `@test_noalloc` both precisions: PASS. Add the mutation (e.g. block end off by one) and show it is killed.
- [ ] **Step 5:** A/B vs the commit before the task. Gate: ratio ≤ 1.02.
- [ ] **Step 6:** Commit `Run the per-atom layers per block of receivers`.

### Task 7: Energy-only workspace

**Files:**
- Modify: `PureMLIP/src/model/evaluate.jl`, `PureMLIP/src/model/Model.jl` and `PureMLIP/src/PureMLIP.jl` (exports), `PureMLIP/docs/src/model.md`, `test/trim_tests.jl`, `test/strictmode_tests.jl`
- Test: `PureMLIP/test/model_tests.jl`

**Interfaces:**
- Produces: `MACEEnergyWorkspace{T}(model::MACEModel, n_atoms, n_edges; edge_chunk = 4096, block = 256)` (exported) and `mace_energy!(res::MACEResult, ws::MACEEnergyWorkspace, model, positions, cell, atomic_numbers, nl)` running the block-fused forward with no derivative or backward-persistent buffers. Forces/virial fields of `res` are left untouched.

- [ ] **Step 1:** Tests: energies and per-atom energies equal `mace_energy_forces!` within `10eps(T)·1280` relative for all reference structures, both precisions; memory `<= 30_000` bytes/atom Float64 (recon estimate 21.0 kB); zero edges gives E0.
- [ ] **Step 2:** Run: FAIL (`MACEEnergyWorkspace` undefined).
- [ ] **Step 3:** Implement; add to TrimCheck, `test_signatures` + `Base.return_types`, `resource_leak_report` (registered scoped), `@test_noalloc` (both precisions); document in `model.md` with an `@docs` entry.
- [ ] **Step 4:** Full suite with coverage, coverage gate, juliac item: PASS.
- [ ] **Step 5:** A/B (`mace_energy!` with the energy workspace vs main's `mace_energy!`). Gate: ratio ≤ 1.00.
- [ ] **Step 6:** Commit `Add an energy-only workspace`.

### Task 8: Run and test PureMLIP with `PureBLAS.activate()`

PureMLIP's matrix products go through `mul!`, which `PureBLAS.activate()` redirects for the whole process. PureMLIP needs no code change and no dependency on PureBLAS (not even a weak one); this task adds the test environment, the CI job, the docs and the measurement.

**Files:**
- Create: `PureMLIP/test/pureblas/Project.toml`, `PureMLIP/test/pureblas/runtests.jl`
- Modify: `PureMLIP/docs/src/model.md` (a "Using PureBLAS" section: `using PureBLAS; PureBLAS.activate()`, Julia 1.13, the effect is process-wide), `.github/workflows/CI.yml` (a 1.13 job running `test/pureblas`), `PureMLIP/bench/blas_compare.jl` header (it is the periodic check against OpenBLAS and Accelerate), `AGENTS.md`

**Interfaces:**
- `test/pureblas/Project.toml`: `[sources] PureBLAS = {url = "https://github.com/el-oso/PureBLAS.jl", rev = "<PureBLAS master commit at task start>"}`, `PureMLIP = {path = "../.."}`.
- `test/pureblas/runtests.jl` calls `PureBLAS.activate()` once, asserts `PureBLAS.is_active()`, then runs its items.

- [ ] **Step 1:** Write the items: all reference structures within the LAMMPS tolerances of the main suite, both precisions; results agree with the main suite's OpenBLAS values within `10eps(T)·1280` relative (OpenBLAS values computed in a child process without PureBLAS, or stored by the main suite); `@test_noalloc` for both entry points with PureBLAS active.
- [ ] **Step 2:** Run in the 1.13 environment: PASS (or report any disagreement as a PureBLAS bug, with the shape).
- [ ] **Step 3:** A/B with `PUREMLIP_AB_LOAD=PureBLAS` (the head side runs `PureBLAS.activate()` before warm-up) vs main at plan start (596b5b2, OpenBLAS). Gate: Float64 ≤ 0.50 (recon 0.453 with tasks 2–3); Float32 reported only, until PureBLAS's SME Float32 route lands.
- [ ] **Step 4:** Commit `Test and document PureMLIP with PureBLAS active`.

### Task 9: Record the results; push

**Files:**
- Modify: `PureMLIP/bench/results/` (re-run `mace_cpu.jl` at cu_bench(5) and `large_julia_only.jl` at cu_bench(14) with the new workspace; add the BLAS vendor and whether PureBLAS was loaded to the JSON), `AGENTS.md` (memory and speed figures stated as present measurements)

- [ ] **Step 1:** Run both benches (OpenBLAS, and with PureBLAS loaded in the PureBLAS environment), LAMMPS only at sizes that keep its resident set under 12 GB and swap flat.
- [ ] **Step 2:** Full suite, coverage gate, juliac item, mutation check on the new entries.
- [ ] **Step 3:** Commit `Record CPU speed and memory after the workspace changes` with the tables in the body; whole-branch review; push to main.
