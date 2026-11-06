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
