# PureMLIP threading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate MACE energy, forces and stress on several threads, with results independent of the thread count up to rounding and bit-reproducible for a given chunk count, then compare multi-threaded PureMLIP against multi-threaded fastatomstruct.

**Architecture:** Static chunks of receiver blocks, one task per chunk; every mutable buffer that two chunks could touch is per chunk (indexed by chunk id, never `threadid`); the two races — the sender-indexed `dx` accumulation in the message pullback and the force/virial scatter — use per-chunk accumulators reduced in a fixed order. The serial path stays the default and unchanged.

**Tech Stack:** Julia 1.12/1.13 threads (`Threads.@spawn`), StrictMode 0.5.8 gates (`:concurrency_safe`, `:no_threadid_state`), the A/B harness, PureBLAS.

**Spec:** `docs/superpowers/specs/2026-10-30-pureatomistic-design.md` sections 2 and 14 (speed target: faster than fastatomstruct on each machine); plan 4's AGENTS.md rules; measurements in `plan5-recon.md` section 6 (outside the repo).

## Global Constraints

- The serial path (one chunk) gives bit-identical results to `main` at plan start, and its speed does not regress (A/B median ≤ 1.02 time ratio).
- Results for any thread count equal the serial result within `10·eps(T)·1280` relative; for a fixed chunk count they are bit-reproducible across runs.
- No `Threads.threadid()` indexing anywhere (`:no_threadid_state` gate); `:concurrency_safe` gate stays at 0 fails.
- Allocation: the serial path stays `@test_noalloc`. The threaded path may allocate only the per-call task objects; that allowance is measured, stated in bytes per evaluation, and gated as an exact bound.
- Trim: the threaded entry point passes TrimCheck and the juliac item (a minimal `Threads.@spawn` program is known to trim on 1.13.3).
- Threaded benchmarks follow the PureBLAS configuration: `julia -t <ncores>,0 --gcthreads=1,0`, a physical-core mask, BLAS threads 1 under Julia threading; speed-ups reported against named references.

## Review Focus

1. **Thread count larger than the number of receiver blocks** (small cells): extra chunks idle, results unchanged.
2. **One atom or zero edges** with many threads: energy = E0, zero forces.
3. **Unbalanced chunks** (a vacancy or surface makes some blocks lighter): correctness unaffected; load balance measured.
4. **PureBLAS active together with Julia threads**: no oversubscription (PureBLAS pool and Julia threads both stated), results agree.
5. **Float32**: same agreement bounds in the working precision.

---

### Task 1: Profile the stages

- [ ] Profile `mace_energy_forces!` at 500 and 10,976 atoms (Float64, Float32; OpenBLAS and PureBLAS) by stage: edges, radial features, radial MLP, tensor product, symmetric contraction, linear layers, pullbacks, force scatter. Commit the share table to `bench/results/stage_profile_*.json`. It decides the chunk granularity and which stages need their own parallel loop.

### Task 2: Per-chunk workspace layout

- [ ] `MACEWorkspace{T}(…; nchunks = 1)` owns `nchunks` chunk scratches (radial buffers, block buffers, `dx`, force and virial accumulators); `nchunks = 1` keeps today's layout and memory. Tests: memory per atom stated for `nchunks = 1, 4, 8`; serial results bit-identical to main; noalloc.

### Task 2b: Symmetric contraction as matrix products

- [ ] Product 1 (forward and pullback) and the product-2 pullback become per-channel GEMMs over 64-atom chunks (channel-batched strided layout, measured best in the spike report `gemm-spike-report.md`, outside the repo); the product-2 forward and the tensor product stay loops. Dense per-element tensors (U·W combined, and the symmetrized pullback tensor) are built only for the elements present in the structure, cached per workspace, about 25 MB per element in Float64. Tests: agreement with the loop kernels within `10·eps(T)·1280` relative in both precisions, pullback against ForwardDiff and finite differences, LAMMPS references unchanged, noalloc after warm-up (tensors built at workspace construction), TrimCheck; memory stated per element. Gate: end-to-end A/B of `mace_energy_forces!` against the commit before the task with OpenBLAS and with PureBLAS active (speed-ups reported; the spike estimated about 1.3× end to end, unmeasured).

### Task 3: Threaded forward pass

- [ ] Chunks of receiver blocks run as tasks for the messages, per-atom layers, readouts and ZBL; energy and node energies reduced in fixed order. Tests: equality with serial within the bound, bit-reproducibility, Review Focus 1, 2, 5; `:concurrency_safe` and `:no_threadid_state` sweeps; A/B speed-up at 4 and 8 threads.
  Design note: each chunk after the first carries about 102 MB of edge buffers in Float64 at `edge_chunk = 4096` (1.8 MB per atom at 500 atoms with 8 chunks), so the threaded path needs a smaller default `edge_chunk`, or edge buffers sized to each chunk's edge count, with the choice measured against the cost of the radial-MLP matrix products.

### Task 4: Threaded backward pass and force scatter

- [ ] Per-chunk `dx` and force/virial accumulators with a fixed-order reduction. Tests as task 3 plus forces against finite differences on the threaded path and the LAMMPS references unchanged.

### Task 5: Allocation and trim

- [ ] Measure the per-call allocation of the threaded path; gate it as an exact bound; TrimCheck and the juliac item with threads.

### Task 6: Multi-threaded comparison with fastatomstruct

- [ ] On the Apple M6 (and an x86 machine with a locked clock, with the user's approval per machine): PureMLIP at 1, 4 and all performance cores against fastatomstruct's XLA CPU backend at its natural thread count and pinned to the same cores; 7 interleaved rounds; speed-ups with ranges and peak memory. Results outside the repository (GPL code), summarized in the design spec.
