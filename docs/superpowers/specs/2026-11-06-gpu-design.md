# GPU support for PureMLIP and PureAtomistic

## Intent

Run MACE molecular dynamics on GPUs, written once and portable across GPU vendors, so that:

1. every validation that passes on the CPU passes on each GPU;
2. on each GPU type, PureMLIP is faster than the reference for that hardware;
3. the vacancy-diffusion campaign (spec 2026-10-30, section 7, items 4–5), postponed because it
   would take about six days on the CPU, becomes affordable.

## Goals

- MACE energy, forces and stress on the GPU, from the same model and checkpoint as on the CPU.
- A whole MD step stays on the GPU: positions, velocities, cell, neighbor list, MACE evaluation,
  integrator, thermostats and barostat. The CPU only writes trajectory frames at the output
  interval and runs analysis.
- The CPU path is unchanged in results and speed.
- Backends in this order:

  | Step | GPU | Machine | Speed reference |
  |---|---|---|---|
  | 1 | Apple (Metal) | Mac mini (Apple M6) | PureMLIP threaded CPU, and fastatomstruct's best CPU result on the same machine |
  | 2 | NVIDIA (CUDA) | x86 machine with an attached eGPU | fastatomstruct's XLA CUDA backend, head to head |
  | 3 | AMD (ROCm) | AMD GPU machine | PureMLIP threaded CPU (fastatomstruct has no AMD backend) |

  fastatomstruct's MACE engine supports a native CPU backend and XLA on CPU and CUDA; it has no
  Metal or AMD backend.

## Non-goals

- Multi-GPU runs and mixed CPU/GPU evaluation.
- GPU training or fitting.
- Any change to the CPU numerics.

## Anti-goals (failure even if it works)

- Vendor-specific code where a portable kernel would do.
- A binary dependency in the core packages.
- Copying data between CPU and GPU on every MD step.
- A speed claim without interleaved measurement and a named reference.

## Constraints

- **Hardware-agnostic first.** Kernels are written with KernelAbstractions.jl. The backend comes
  from the arrays the caller passes (`get_backend`); the caller loads Metal.jl, CUDA.jl or
  AMDGPU.jl. Vendor-specific code exists only for a measured workaround or tuning that one GPU
  needs.
- **Core dependency rule** (AGENTS.md): no `_jll` or artifact in the core closure.
  KernelAbstractions 0.10 brings about fifteen `_jll` packages (LLVM, SPIR-V tools, pocl and
  others), so **KernelAbstractions is a weak dependency: all GPU code lives in a package
  extension** (`KernelAbstractions` → `PureMLIPGPUExt`, and the matching PureAtomistic
  extension) that loads when the user loads KernelAbstractions or a vendor package. The core
  stays binary-free, and current vendor package versions stay usable. AGENTS.md's line that
  allows KernelAbstractions in the core is corrected to say this.
- **Trim:** the TrimCheck gates and the trimmed `md_run` binary cover the CPU path only. The GPU
  path runs from normal Julia and is exempt from the trim gates.
- **Precision:** Float32 on Metal (Apple GPUs have no Float64). Float32 and Float64 on CUDA and
  AMD where the hardware supports Float64.
- **Float32 baseline:** before the Metal work is judged, the plan 5 Task 12 validation runs are
  repeated on the CPU in Float32. GPU Float32 results are held to that CPU Float32 baseline. The
  baseline also records whether Float32 by itself meets the Float64 criteria; that is reported,
  not required.
- **Correctness oracles:** the five LAMMPS reference structures (plan 3), finite-difference
  forces, the stored CPU results in the same precision. GPU results agree with the CPU result
  within a derived bound in the working precision, stated per quantity.
- **Validation runs repeated on each GPU:** NVE energy conservation of 500 Cu over 2 ps, Bussi
  NVT at 1200 K (temperature distribution against theory), the vacancy run at 1300 K, with the
  plan 5 Task 12 criteria applied against the baseline of the same precision.
- **Per-step transfers** *(builder's call, set here)*: random numbers are drawn on the GPU. A
  few scalars read back per step (for example the kinetic energy for Bussi or the barostat) are
  allowed if the step-time cost is measured and stated; array copies per step are not. GPU
  thermostat trajectories must match the CPU statistically, not reproduce its random stream.
- **What "faster" means** *(builder's call, set here)*: the target is met at `cu_bench(14)`
  (about 11,000 atoms); `cu_bench(5)` is reported too. Comparisons use matching precision
  (GPU Float32 against CPU Float32, and on CUDA the precision fastatomstruct's XLA backend runs
  in, stated).
- **Speed and memory:** `cu_bench(5)` and `cu_bench(14)`, interleaved rounds, speed-up with
  [min–max] range against the named reference, peak GPU memory next to every speed figure.
- **Repository rules unchanged:** fail fast, generic indexing, TrimCheck, StrictMode gates,
  coverage rows, mutations for new math only, present-tense comments.
- **Machines:** timing on the x86 machine runs with the clock locked (`fleet_freqlock.sh`,
  verified under load before and after each window). fastatomstruct is built and run only outside
  the repositories, and its source is never read.
- **CI:** GitHub runners have no GPU. GPU tests are opt-in items (like the juliac items), run on
  each GPU machine before a push; the CPU suite stays the CI gate.

## Decided by the user

- KernelAbstractions.jl, hardware-agnostic, Metal first.
- NVIDIA second, because fastatomstruct supports CUDA and not Metal; AMD third.
- Goals: all validations on each GPU, faster than fastatomstruct where it runs, and the
  vacancy-diffusion campaign.
- KernelAbstractions as a weak dependency, with all GPU code in an extension.
- The GPU path is exempt from the trim gates.
- Float32 GPU runs are held to a CPU Float32 baseline.

## Left to the builder

- How the MACE stages map to kernels (which stages are fused, which use the vendor BLAS through
  `mul!` on GPU arrays), measured stage by stage as in plan 5b Task 1.
- The GPU neighbor list (cell list or another method) and how it is rebuilt.
- How the workspace and pools carry over to GPU arrays.
- Reductions (energy, virial) on the GPU and their reproducibility.
- The campaign's temperatures, cell size and run lengths, to be proposed with a cost estimate
  from the measured GPU speed before it runs.

## Measured results

Apple M6 with Metal, MACE-MPA-0, Float32. Each figure is the median over 7 interleaved rounds (one
process per arm per round, the arm order rotating); a speed-up is the reference time over the Metal
time, quoted as median [min, max] over the rounds. The records are
`PureMLIP/bench/results/gpu_compare_*metal*.json`, produced by
`PureMLIP/bench/gpu_compare.jl`. The unit is one energy + forces + stress evaluation, or one
velocity Verlet step of a `Simulation`; the device is synchronized before the clock stops.

| cell | unit | Metal | threaded CPU | speed-up | Metal device memory |
|---|---|---|---|---|---|
| `cu_bench(5)`, 500 atoms | evaluation | 0.0698 s | 0.0869 s | 1.25 [1.24, 1.30] | 0.35 GB |
| `cu_bench(14)`, 10,976 atoms | evaluation | 1.354 s | 2.113 s | 1.56 [1.53, 1.64] | 1.58 GB |
| `cu_bench(5)` | MD step | 0.0700 s | 0.0903 s | 1.29 [1.27, 1.45] | 0.35 GB |
| `cu_bench(14)` | MD step | 1.354 s | 2.232 s | 1.65 [1.60, 1.70] | 1.58 GB |

The CPU arm is threaded PureMLIP (6 threads under `-t 6,0 --gcthreads=1,0`, one BLAS thread,
PureBLAS, Float32, `ThreadedChunks()`); its maximum resident set is 1.7 GB at `cu_bench(5)` and
2.2 GB at `cu_bench(14)`. The target, a minimum speed-up above 1 at `cu_bench(14)`, is met (1.53).

| cell | fastatomstruct XLA CPU | Metal | speed-up | XLA maximum resident set |
|---|---|---|---|---|
| `cu_bench(5)` | 0.437 s | 0.0698 s | 6.25 [6.19, 6.31] | 6.2 GB |
| `cu_bench(8)`, 2,048 atoms (largest that fits) | 1.994 s | 0.250 s | 7.96 [7.88, 8.38] | 22.8 GB |
| `cu_bench(9)`, 2,916 atoms | does not fit | 0.352 s | not measured | 24.7 GB when stopped |

fastatomstruct's XLA CPU backend ran unpinned on all cores. It does not fit at `cu_bench(9)`: its
memory guard (free memory under 6 GiB, or swap use growing by more than 1 MB) stopped it at 24.7 GB,
and the 11.1 MB per atom measured at `cu_bench(8)` extrapolates to about 32.5 GB against 34.4 GB of
physical memory. The guard runs at `cu_bench(9)` grew swap use by about 161 MB in the first full
evaluation run and 73 MB in the second (about 234 MB in total; a probe before the runs, about
364 MB), on a host that already held about 16 GB of swap. They are the untimed pass that finds out
whether the program fits; no timed number came from them and no kept round saw swap growth. Single
probes of XLA alone at `cu_bench(6)` (0.74 s, 9.9 GB) and `cu_bench(7)` (1.21 s, 15.4 GB) have no
Metal counterpart, so no speed-up.

Rounds are discarded when swap use grows or the calibration loop disagrees: 10 % within a child and
10 % between children, looser than the 3 % and 1 % of `ab_compare.jl` because the single-thread child
showed 3.5 to 5 % spreads on this host. The multi-threaded CPU arm is left out of the pair
comparison and allowed 50 % within its own child, because its single-thread calibration directly
after the 6-thread burst reads 15 to 130 % off. The `cu_bench(5)`, 8 and 9 record came from an
earlier version of the script that held the CPU arm to 10 % and compared it in the pair check.

The status read of a device step costs a median 0.20 ms at `cu_bench(5)` (0.29 % of the step; the
mean of the per-round differences is 1.4 ms, because the step that reads is about 3 ms slower in 3
of 7 processes, and the reference step shows the same high mode, so it is process jitter) and is
not measurable at `cu_bench(14)` (median -0.05 % of the step, within +-0.14 %).
