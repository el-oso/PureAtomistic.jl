# Plan 2b: hardening — boundaries, call checks, coverage, oracles, latency

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make machine-checked confidence in agent-written code a property of PureMLIP's CI before the model code arrives: written repository rules, declared module boundaries, whole-package call checks, a per-file line-coverage gate, fuzzed parsers, a mutation check that proves the tests can fail, and a measured latency baseline.

**Architecture:** PureMLIP's flat module splits into two submodules, `PureMLIP.Checkpoint` and `PureMLIP.Equivariant`, each declaring its interface with `public`/`export`; `PureMLIP` re-exports their public names, so callers see no change. StrictModeTest gates (`test_layers`, `test_calls`) and Aqua run as test items. Coverage is measured per file in CI against a committed threshold table that may only rise. Two new test kinds — parser fuzzing and a mutation smoke check — test the tests.

**Tech Stack:** StrictMode 0.5.3 and StrictModeTest 0.5.2 (latest registered on 2026-10-31; StrictModeTest 0.5.2 accepts StrictMode `0.5.2 - 0.5`), Aqua.jl, Coverage tooling in CI only, PrecompileTools.jl (pure Julia, small; package dependency for the precompile workload).

**Spec:** `docs/superpowers/specs/2026-10-30-pureatomistic-design.md`, sections 4 and 6.3.

## Global Constraints

- PureMLIP `[deps]` after this plan: Inflate, LinearAlgebra, PrecompileTools. Nothing else. Test/CI-only tools go in `PureMLIP/test/Project.toml` or the CI workflow.
- No public behavior change: every existing test passes unchanged except for module-qualified names inside tests (`PureMLIP.x` keeps working because `PureMLIP` re-exports or forwards every name tests use).
- Thresholds are measured, never guessed: each gate's number comes from a recorded baseline run on this branch, stated in the commit body.
- Coverage thresholds only rise: the table's values may be raised by later commits, never lowered without the user's approval.
- Rules from plans 1-2 still apply (no Python; no host names or local paths in tracked files or commits; comments state present facts; `lazy"..."` for 4+-piece interpolations; no `@inbounds`; commit trailers `Assisted-by`/`Claude-Session`; push to `main` only after the final review).

## Review Focus

1. **A layer violation hidden behind a re-export** (Equivariant code calling an unexported Checkpoint helper through `PureMLIP.`): `test_layers` must catch it. Positive control in Task 2.
2. **A gate that cannot fail**: each new gate gets a positive control that makes it fail (Tasks 2-5).
3. **Fuzzed input that hangs** (a pickle loop or a huge declared size): every fuzz case finishes within a time bound. Task 4.
4. **Coverage table drift**: a file added later without a row must fail the coverage check, not pass silently. Task 3.

---

### Task 1: StrictMode 0.5.3 and repository rules

**Files:** Modify `PureMLIP/test/Project.toml` (`StrictMode = "0.5.3"` lower bound), create `AGENTS.md` at the repo root, create `CLAUDE.md` containing one line that points to `AGENTS.md`.

- [ ] **Step 1:** Bump the StrictMode compat in the test project to `"0.5.3"`; resolve; run the full suite (`PUREMLIP_CHECKPOINT_DIR=/tmp/pm_ck julia --project=PureMLIP -e 'using Pkg; Pkg.test()'`). Expected: all pass. Record the resolved StrictMode and StrictModeTest versions.
- [ ] **Step 2:** Write `AGENTS.md` (plain, present-tense, ≤ 150 lines) with these sections, each stated as rules with the reason in one line: *Scope and layout* (monorepo, packages, submodules, where plans/specs live); *Dependencies* (PureMLIP deps list; pure Julia, no artifacts, LinearAlgebra/SparseArrays allowed, GPU backends allowed; test-only deps exempt; no Python); *Conventions* (textbook SH/CG, e3nn only through the conversion layer, index layout, even-sum restriction); *Code rules* (generic indexing, no `@inbounds`, concrete types, `!` allocation-free, fail-fast errors with module prefixes, `lazy"..."`, comments describe present code); *Gates* (every test item kind and what it proves; how to run the suite, the juliac item, coverage; thresholds only rise); *Oracles and tolerances* (the tolerance policy of Task 5); *Commits and publishing* (author identity, trailers, push to `main`, no PRs, no host names or local paths).
- [ ] **Step 3:** Commit: `Bump StrictMode to 0.5.3 and add repository rules`.

### Task 2: Submodules, public interfaces, layer and call checks

**Files:** Modify `PureMLIP/src/PureMLIP.jl`; wrap `src/checkpoint/*.jl` in `module Checkpoint` (new file `src/checkpoint/Checkpoint.jl` that includes the existing files) and `src/equivariant/*.jl` in `module Equivariant` (`src/equivariant/Equivariant.jl`); create `PureMLIP/test/layers_tests.jl`; add Aqua to the test project.

**Interfaces:**
- `PureMLIP.Checkpoint` exports `load_mace_checkpoint, MACECheckpoint, MACEHyperparameters, CheckpointTensor, torch_array` and declares `public` the names tests and later plans use (`open_torch_archive, entry_data, embedded_archives, unpickle, read_state_dict, numel, StateDict, IrrepMul, InteractionSpec, ReadoutSpec, PickleTree, PickleNode, TorchArchive, ZipEntry` and the `PK_*` constants).
- `PureMLIP.Equivariant` exports the equivariant API of plan 2 and keeps `_mode_product`, `_gauss_legendre`, `_real_sh!` internal.
- `PureMLIP` does `using .Checkpoint, .Equivariant`, re-exports their exports, and declares `public` the forwarded public names, so `PureMLIP.entry_data` etc. keep working.

- [ ] **Step 1: Write the failing tests** in `layers_tests.jl`:

```julia
@testitem "layer graph holds" tags = [:checks] begin
    using StrictModeTest
    test_layers(PureMLIP => (PureMLIP.Checkpoint, PureMLIP.Equivariant),
                PureMLIP.Checkpoint => (),
                PureMLIP.Equivariant => ())
end

@testitem "every compiled signature resolves (test_calls)" tags = [:checks] begin
    using StrictModeTest
    test_calls(PureMLIP)
end

@testitem "Aqua" tags = [:checks] begin
    using Aqua
    Aqua.test_all(PureMLIP)
end
```

- [ ] **Step 2:** Run; expected FAIL (submodules do not exist).
- [ ] **Step 3:** Implement the split. Fix every `test_calls` and Aqua finding at its source (ambiguities, unbound type parameters, undefined exports). A finding that cannot be fixed is listed with its reason in the commit body, and exempted by name in the test (`exempt = (...)`) — never globally.
- [ ] **Step 4: Positive control.** On a scratch edit (not committed), make an Equivariant function call an unexported Checkpoint helper through `PureMLIP.Checkpoint._something`; confirm "layer graph holds" fails; revert. Record the failing output in the report.
- [ ] **Step 5:** Full suite + juliac item green. Commit: `Split PureMLIP into Checkpoint and Equivariant submodules with layer checks`.

### Task 3: Per-file line coverage gate

**Files:** Create `PureMLIP/test/coverage/thresholds.toml` and `PureMLIP/test/coverage/check.jl`; modify `.github/workflows/CI.yml` (one job on Julia 1.13 runs the suite with coverage and then `check.jl`).

**Interfaces:** `check.jl` reads the `.cov` files Julia writes under `PureMLIP/src`, computes per-file covered/coverable lines (use CoverageTools.jl in the CI step's environment, not in PureMLIP's deps), compares to `thresholds.toml` (`"src/checkpoint/pickle.jl" = 0.93` style, fraction), and exits non-zero naming every file below its threshold **and every src file without a row**.

- [ ] **Step 1:** Run the suite locally with `Pkg.test(coverage = true)`, compute the baseline per file, and write `thresholds.toml` with each value = measured fraction rounded **down** to two decimals.
- [ ] **Step 2: Positive controls:** (a) delete one row → `check.jl` fails naming the file; (b) set one threshold to measured + 0.05 → fails naming it. Restore.
- [ ] **Step 3:** Add the CI job. In `AGENTS.md` (Gates) state the ratchet rule: raise a row when coverage rises; lowering needs the user's approval.
- [ ] **Step 4:** Commit: `Gate per-file line coverage against a ratcheting threshold table`. Body: the baseline table.

### Task 4: Parser fuzzing

**Files:** Create `PureMLIP/test/fuzz_tests.jl`.

- [ ] **Step 1: Write the tests** (seeded, deterministic):

```julia
@testitem "fuzzed checkpoints fail cleanly" setup = [ZipFixtures, Checkpoints] begin
    using Random
    rng = Xoshiro(20261031)
    good = read(Checkpoints.fetch(:mp0b3))
    allowed(e) = e isa ArgumentError || (e isa ErrorException && occursin(r"^(checkpoint|unpickle):", e.msg))
    for trial in 1:300
        bad = copy(good)
        kind = trial % 3
        if kind == 0                                  # flip bytes inside data.pkl or the central directory
            for _ in 1:rand(rng, 1:8); i = rand(rng, eachindex(bad)); bad[i] ⊻= rand(rng, UInt8(1):UInt8(255)); end
        elseif kind == 1                              # truncate
            resize!(bad, rand(rng, 0:length(bad) - 1))
        else                                          # splice random bytes into data.pkl's range
            # locate data.pkl through open_torch_archive on `good`, overwrite a random window
        end
        t = @elapsed r = try
            PureMLIP.load_mace_checkpoint(bad); :loaded
        catch e
            allowed(e) ? :clean_error : e
        end
        @test r === :loaded || r === :clean_error
        @test t < 5.0
    end
end
```

The third branch's window logic is written in the test. A byte flip that lands in a tensor's storage may still load (`:loaded`) — that is acceptable; anything other than a clean error or a load (a `BoundsError`, `InexactError`, `MethodError`, `StackOverflowError`, `OutOfMemoryError`, or a hang) is a failure.
- [ ] **Step 2:** Run. Every unclean failure is a bug: fix it at its source so it raises a prefixed error (`checkpoint:`/`unpickle:`/`ArgumentError`), and add the minimized input as a named regression test in the existing test file of that layer.
- [ ] **Step 3:** Same for the pickle layer alone: 1000 seeded random mutations of a small valid hand-built pickle (from `pickle_tests.jl`) through `unpickle`, with the same allowed-error rule and a 1 s bound per case.
- [ ] **Step 4:** Full suite green. Commit: `Fuzz the checkpoint and pickle readers`.

### Task 5: Mutation smoke check and tolerance policy

**Files:** Create `PureMLIP/test/mutation/mutations.toml`, `PureMLIP/test/mutation/run.jl`; add a manually triggered (`workflow_dispatch`) CI workflow `.github/workflows/mutation.yml`; add the tolerance policy to `AGENTS.md`.

**Interfaces:** `mutations.toml` lists entries `{ file, find, replace, expect }` — an exact source substring, its replacement (a deliberate bug), and the test item name regex expected to fail. `run.jl` copies `PureMLIP` to a temp dir, applies one mutation at a time, runs the full suite in a fresh process, and reports for each mutation whether the suite failed ("killed") or passed ("survived"); it exits non-zero if any mutation survives.

- [ ] **Step 1:** Write at least 12 mutations covering every source file of plans 1-2, for example: a sign flip in the e3nn sign rule; `sqrt(4π)` → `sqrt(2π)`; an off-by-one in `sh_index`; swapping two relabeled axes in `e3nn.jl`; dropping the CRC check; `(-1)^m` → `1` in `_complex_to_real`; Gauss-Legendre `l+1` → `l` nodes; skipping `None` values in the state-dict walk; reading BINFLOAT little-endian; a stride ignored in `gather`; `1e-13` → `1e-3` in the zeroing threshold; the projector in the SH gradient replaced by identity.
- [ ] **Step 2:** Run `run.jl`. Every survivor is a missing test: add a test that kills it, then rerun until all are killed. List each survivor and the added test in the report.
- [ ] **Step 3:** Tolerance policy in `AGENTS.md`: each comparison states its oracle and a tolerance justified by one of (a) the oracle's own stated tolerance (e3nn-core-rs 1e-10), (b) rounding: `c · eps(T) · (operation count or condition)` with `c` written down, (c) a measured margin (e.g. the 1e-13 zero threshold with its measured gap). Audit existing tests: any `atol` without one of these justifications gets a one-line comment citing it, or is tightened.
- [ ] **Step 4:** Commit: `Add a mutation smoke check and a tolerance policy`.

### Task 6: Precompile workload and latency baseline

**Files:** Create `PureMLIP/src/precompile.jl` (PrecompileTools `@setup_workload`/`@compile_workload`), included last from `PureMLIP.jl`; add `PrecompileTools` to `[deps]` with a `[compat]` bound; create `PureMLIP/test/latency_tests.jl`.

- [ ] **Step 1:** Measure the baseline first, before any workload: `test_latency(:PureMLIP; calls = ["PureMLIP.spherical_harmonics!(zeros(16), [0.1, 0.2, 0.9], 3)", "PureMLIP.e3nn_wigner_3j(1, 2, 3)"])` in measure-only mode (no budget), three fresh processes; record load time and first-call times.
- [ ] **Step 2:** Add the workload: native and e3nn spherical harmonics with gradients (Float64, lmax 3), `real_clebsch_gordan`/`e3nn_wigner_3j` for (1,1,2), `wigner_D(2, ·)`, and `load_mace_checkpoint` on a tiny synthetic checkpoint built in memory (no file, no network). The workload must not touch the network or the filesystem.
- [ ] **Step 3:** Measure again (three fresh processes); the first-call times must drop. Commit the workload with both measurements in the body.
- [ ] **Step 4:** `latency_tests.jl`: a `:latency`-tagged item calling `test_latency` in measure-only mode (it reports, never fails, because wall-clock budgets depend on the host). Document in `AGENTS.md` how to run it with a budget on a pinned host.
- [ ] **Step 5:** Full suite + juliac item green (the trimmed binary must still build with the workload present). Commit: `Add a precompile workload and a latency baseline`.
