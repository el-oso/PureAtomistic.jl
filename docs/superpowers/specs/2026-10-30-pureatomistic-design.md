# PureAtomistic.jl and PureMLIP.jl: design

Date: 2026-10-30. Status: draft for approval.
Supporting material: `rust2julia-inventory.md` (source-project inventory, MACE feature list in section 9, decision log in section 10) and `mace-julia-ecosystem.md` (Julia package survey and reuse map).

## 1. Intent

Run molecular dynamics with MACE-class machine-learned interatomic potentials (MLIPs) in pure Julia, faster and more flexibly than mace-torch, and train and fine-tune those models in the same code. The work also serves the user's interest in materials discovery of the CuspAI kind (porous sorbents, carbon capture), where PureAdsorb.jl already covers classical adsorption.

Why Julia: generic code that specializes to fast native code for each element type and device, one code base for CPU (including Apple SME through PureBLAS) and GPU (through KernelAbstractions), and `juliac --trim` binaries, without Python.

## 2. Goals

1. **PureMLIP.jl** evaluates ScaleShiftMACE models: energy, forces, stress (virial), per-atom energies. It loads published mace-torch `.model` checkpoints (MPA-0 first, also the MP-0 family) through a pure-Julia parser and reproduces mace-torch results.
2. **PureAtomistic.jl** runs MD with any potential that implements its potential interface, with PureMLIP as the main one.
3. Training and fine-tuning of PureMLIP models (milestone 2), including fine-tuning from a foundation checkpoint.
4. Speed: **2x faster than mace-torch on the same machine** for MD steps per second; **1.05x is accepted for the first attempt**. Measured on each target machine (section 5).

## 3. Non-goals for now

- Long-range and charge models (MACELES, PolarMACE, OMOL charge/spin, dipole and dielectric models). Later, written from the papers.
- Other MLIP architectures (NequIP, Allegro, PET, ...). The package name leaves room; the first version does not include them.
- Other material classes as first targets (MOFs, batteries, catalysts, liquids, amorphous, phonons). All are wanted later; none drive milestone 1.
- Distributed multi-node training.
- Running mace-torch or any Python inside either package.

## 4. Hard constraints

- **No Python in any project, in any capacity.** One exception was granted for a single run of MACE's `create_lammps_model.py`, in a throwaway folder outside all projects, to produce the LAMMPS reference model file. That folder is deleted afterward; only the exported file is kept as reference data.
- **Reuse the Julia ecosystem; write from scratch only what does not exist.** Ranking for dependencies: (1) pure Julia with no `_jll`/artifacts anywhere in the dependency closure, (2) `juliac --trim` compatible, (3) features.
- **Exception:** KernelAbstractions and GPU backends (Metal.jl, AMDGPU.jl, CUDA.jl) are accepted with all their dependencies as they are.
- **Exception:** the LinearAlgebra and SparseArrays standard libraries may be used. Their OpenBLAS and SuiteSparse backends are to be replaced by PureBLAS and PureSparse through libblastrampoline, so they do not count as artifact dependencies.
- **Makie** (visualization) is outside the core packages: a package extension or a separate small package. Its binary dependencies are acceptable there.
- **License:** MIT or Apache-2.0 code only. fastatomstruct is GPL-3: do not read it for implementation, do not port it. MACE and e3nn are MIT, llmlip.rs is Apache-2.0, e3nn-core-rs is MIT/Apache.
- The user's Julia rules apply (generic indexing over `AbstractArray` with axes, no `@inbounds` without a benchmark and test, concrete return types, `lazy"..."` in trimmed paths, StrictMode/StrictModeTest gates, Chairmarks for timing).

## 5. Hardware targets

| Machine | CPU path | GPU path |
|---|---|---|
| Apple silicon Mac | PureBLAS (SME route) | Metal.jl through KernelAbstractions, Float32 only |
| x86 machine with an AMD GPU | PureBLAS (x86 kernels) | AMDGPU.jl through KernelAbstractions |
| Zen 5 machine | PureBLAS (x86 kernels) | CUDA.jl through KernelAbstractions (NVIDIA eGPU) |

Float64 is required for the energy-conservation (NVE) tests, so those run on CPU, AMD and NVIDIA, not Metal. Running anything on the x86 machines needs the user's approval each time.

## 6. Architecture

One monorepo, **PureAtomistic.jl**, with two registered packages in subfolders (the multi-package layout Makie.jl uses):

```
PureAtomistic.jl/
  PureMLIP/        models, checkpoint parser, equivariant math
  PureAtomistic/   MD driver, analysis, extra potentials; depends on PureMLIP
  docs/superpowers/specs/   this document
```

PureMLIP can later move to its own repository without changing its users.

### 6.1 PureMLIP.jl

| Part | Decision |
|---|---|
| Spherical harmonics | Our own generic-`T` real spherical harmonics in the **textbook convention**: z polar axis, m = -l..l, orthonormal, with analytic gradients. Port SpheriCart's recurrence (MIT/Apache). SpheriCart.jl itself fails the no-artifact rule (LLVMExtra_jll, OpenSpecFun_jll per the survey). |
| Clebsch-Gordan | WignerSymbols.jl or CGcoefficient.jl at setup time (complex, Condon-Shortley), converted once to real coefficients in our basis. |
| Irreps, tensor product, equivariant linear, symmetric contraction, gates, readouts, scale-shift | Written ourselves. No pure-Julia MACE or e3nn tensor product exists (survey). |
| Conventions | Textbook where a textbook convention exists. Where none exists, e3nn's: "component" feature scale, activation scaling to second moment 1, `1/sqrt(fan_in)` path normalization, "mul_ir" layout (unless a benchmark favors another), the symmetric-contraction coupling basis. |
| e3nn compatibility | A conversion layer used only when loading mace-torch checkpoints: input axis permutation and per-l scale for spherical harmonics, real-basis sign/phase change for Clebsch-Gordan, U matrices read from the checkpoint buffers and rotated into our basis, activation constants reproduced as e3nn computes them (1e6 seeded Gaussian samples) or shown equal to the exact integral within 1e-6. |
| Radial part | Bessel basis, polynomial cutoff, Agnesi transform, ZBL written ourselves (small). The survey flags EmpiricalPotentials' ZBL as possibly wrong, so do not reuse it without a numeric check. |
| `.model` parser | Pure-Julia reader for mace-torch `.model` files: zip archive (torch writes entries uncompressed, ZIP64) plus pickle. Extracts weights, buffers (including U matrices), and hyperparameters (irreps, cutoff, `avg_num_neighbors`, E0s, scale, shift, interaction class). Pickle.jl reads torch zips but brings Libiconv_jll and ZipFile/Zlib_jll. **The parser must be `juliac --trim` compatible** (it is inside the trimmed binary, section 14), so it is our own minimal STORED ZIP64 reader plus a restricted unpickler that maps only the opcodes and globals mace-torch checkpoints use into concrete typed Julia structs, failing fast with a clear error on anything else. Pickle.jl (dynamic `Defer` trees, jlls) is not used in the package; it may serve as a test-time cross-check. |
| Forces and stress | Hand-written analytic pullbacks (the ACEsuit pattern), the only route the survey found compatible with both KernelAbstractions GPU kernels and trim. Stress from the strain derivative. Enzyme on CPU as a test-time cross-check only. |
| Second order (training) | ForwardDiff dual numbers pushed through the hand-written pullbacks (forward-over-reverse, as ACEpotentials does). Enzyme nested reverse on CPU as a cross-check. |
| Layer interface | LuxCore (pure): parameters and state kept outside the model struct. |
| Tensor contractions | **Open: decided by the contraction spike** (OMEinsum vs. TensorOperations vs. plain generic loops/KernelAbstractions kernels vs. PureBLAS gemm on reshaped contractions). See section 10. |
| Atoms interface | AtomsBase and AtomsCalculators (pure) at the outer boundary. Our own interface for training, because AtomsCalculators' generated methods ignore parameters and state (survey). |

### 6.2 PureAtomistic.jl

| Part | Decision |
|---|---|
| MD driver | Our own, small, in the core: velocity Verlet (NVE), Langevin and one deterministic thermostat (Nosé-Hoover or Bussi; builder's choice), one barostat (Bussi or Berendsen), periodic triclinic cells. Written from the published papers. |
| Neighbor list | CellListMap.jl on CPU (pure; we add periodic shift vectors for stress). GPU: our own KernelAbstractions cell list, or NeighbourLists.jl if it passes the dependency rules. The list is shared between the driver and the model, built once per rebuild. |
| GPU residency | Positions, velocities, forces and the neighbor list stay on the device between MD steps; no per-step host copy. |
| Trajectory output | A simple writer that the Makie package reads. extxyz reader and writer written ourselves (no pure-Julia reader exists). |
| Geometry optimization | Optim.jl. |
| Molly.jl | Optional connection only (extension), because it brings Chemfiles, FFTW/MKL and Rmath binaries. |
| Dispersion D3 | Later. Needs a license decision on the C6 parameter table (GPL-1+, LGPL-3 or Apache labels, depending on source). |
| Visualization | Makie, in an extension or separate package: atom animation, vacancy track, Arrhenius plot, energy-conservation plot. |

### 6.3 Engineering stack (both packages)

Follow the setup PureBLAS.jl uses (its `Project.toml`, `test/Project.toml`, `docs/Project.toml`, `src/contracts.jl` are the reference).

| Tool | Role |
|---|---|
| **TypeContracts.jl** (el-oso) | `@contract` on every abstract interface, with return types, checked by `@verify` at precompile time (`trim_compat=true` where the path is trimmed). Interfaces to contract at least: the potential interface between PureMLIP and PureAtomistic (energy, forces, stress, per-atom energies), integrators, thermostats, barostats, neighbor lists, radial bases, and the model-layer interface. `@invariants`/`test_behavior` for behavioral properties (for example rotation invariance of energy, forces equal minus the energy gradient). BaseTypeContracts.jl where Base interfaces (AbstractArray) are implemented. |
| **StrictMode.jl** in `src`, **StrictModeTest.jl** in `test` | `src` declares, tests prove. Allocation-free gates (`@assert_noalloc` / `@test_noalloc`) on the MD step, model evaluation, force/stress pullbacks and neighbor-list update after warm-up; trim gates (`test_compiled` / `test_signatures` with `:trim_compatible`) on the entry points meant for `juliac --trim`; `test_coverage` gate; `@assert_noescape` for scratch buffers. Bounds from PureBLAS (StrictMode 0.5 / StrictModeTest 0.5 at the time of writing; use the current pair, which move together). |
| **TestItems** (`@testitem`, run by **TestItemRunner.jl**) | All tests as test items. Includes the generic-indexing tests (entry points on `OffsetArray`- and `view`-wrapped inputs, results and output axes compared) and TrimCheck `@validate` items. |
| **DocumenterVitepress.jl** | Documentation site for both packages (one site for the monorepo, or one per package; builder's choice). |

## 7. Milestone 1: vacancy diffusion with MPA-0

1. Load MPA-0 from its `.model` file with the pure-Julia parser.
2. Evaluate energy, forces and stress on CPU (Float64, Float32) and GPU.
3. MD of a vacancy in an fcc metal (for example Cu or Ni; builder's choice), about 500 atoms, then 10,000 to 100,000 atoms for speed. Then the same in a multi-element alloy (for example CrCoNi).
4. NVT at several temperatures (for example 600 to 1200 K) after a barostat relaxation. NVE runs for energy conservation.
5. Outputs: vacancy animation in Makie, diffusion coefficient per temperature, activation energy from an Arrhenius fit, compared with published values; in the alloy, which elements the vacancy prefers as neighbors.

## 8. Milestone 2: training and fine-tuning

The mace-torch training features listed in `rust2julia-inventory.md` section 9 are in scope; the first cut is:
- losses: weighted energy, forces, stress (mace-torch defaults: energy 1, forces 100);
- Optimisers.jl (AMSGrad plus weight decay composed), plus what it lacks and we write: global gradient-norm clipping, weight EMA (0.99), ReduceLROnPlateau;
- E0 handling (average from data, or given);
- fine-tuning from a foundation checkpoint;
- data: extxyz reader (ours); MACE HDF5 later (JLD2 may read it; unverified).

## 9. Correctness and speed references

References come from existing files (option B) and LAMMPS (option C). Nothing runs mace-torch on our side.
- **Part by part:** e3nn-core-rs fixtures (MIT/Apache; e3nn 0.6.0 spherical harmonics up to l=4 in component normalization, a fully connected tensor product, MACE 0.3.16 radial functions) checked through the conversion layer to about 1e-12.
- **Model structure:** llmlip.rs (Apache-2.0) inspection file for MACE-MP-0b3 (irreps per block, `avg_num_neighbors` = 61.964672446250916, interaction class) checked against our parsed MP-0b3 checkpoint.
- **End to end:** LAMMPS with MACE's libtorch pair style (C++ at run time) on the same machine, with the model file from the one-time export. Energies and forces must match to about 1e-6 eV (llmlip.rs reports 5e-6 eV energy error on MP-0b3, so this level is reachable).
- **Physics checks:** energy invariant under rotation and translation; forces equal finite-difference gradients; NVE energy conservation.
- **Speed:** MD steps per second against the LAMMPS-MACE run on the same machine, at each system size, on each target machine. Timing with Chairmarks; samples saved as JSON.

## 10. Tensor contractions: decided by measurement

A throwaway spike (2026-10-30; Apple M6, clock not locked; Julia 1.13.3; 1 thread unless noted; code and raw Chairmarks samples in the session scratchpad `spike-contract/results/`) measured the real MPA-0 contractions. Hyperparameters were read from `mace-mpa-0-medium.model` without Python: 128x0e+128x1o hidden, max_ell 3, correlation 3, 89 elements, r_max 6.0, RealAgnosticDensity(Residual) interaction blocks, U3 shapes (16,16,16,23) and (3,16,16,16,51). System: fcc Cu, 78 neighbors per atom, 500 and 10,976 atoms. U values were random and dense (the real U is sparse).

| Variant | sym. contraction step 3, F64 0e / 1o (500 atoms) | conv TP + scatter F64, 500 / 10,976 atoms | allocation per call | TrimCheck |
|---|---|---|---|---|
| PureBLAS.gemm! on the GEMM-lowered form (SME engaged) | **42.5 / 187 ms** | 424 ms / 9.17 s | 160 B | pass |
| OpenBLAS `mul!` | 199 / 1280 ms | 742 ms / 15.8 s | 160 B | pass |
| OMEinsum (optimized order) | 258 / 1355 ms | 312 ms / 6.16 s | 409 MB / 41 GB | fail |
| TensorOperations | 205 / 1292 ms | 742 ms | up to 375 KB | fail; rejects batch indices |
| plain loops, reordered (U·W per element precomputed; Y·CG first) | 47.3 / 158 ms | **92 ms / 2.05 s** | 0 B | pass |
| KernelAbstractions CPU backend | 3228 / 40830 ms | 570 ms | ~850 B | fail |
| KernelAbstractions on Metal, F32 | 123 / 909 ms (naive kernel) | **15 ms / 329 ms** | - | not checked |

**Decision: no OMEinsum, no TensorOperations, no new tenferro-style package.** MACE has six fixed einsum strings and one tensor product. They are lowered by hand: on CPU, plain generic loops plus GEMM through PureBLAS; on GPU, KernelAbstractions kernels, plus the vendor matrix multiply for the symmetric-contraction GEMM (on Metal, `mul!` on MtlArray ran step 3 in 5.7 ms against 123 ms for the naive KA kernel). Contraction order mattered more than the library.

Findings that constrain the build:
- PureBLAS SME engages only for Float64 (`_sme_eligible` requires `T === Float64`); Float32 on CPU runs at OpenBLAS speed.
- The KernelAbstractions CPU kernel launch fails TrimCheck (`KernelAbstractions.__run`) and is 3-6x slower than the same plain loop. This supports the plain-loops-on-CPU call in section 13.
- KernelAbstractions 0.10.4 hung for over 20 minutes during precompile on this Mac, inside PoCL's hardware-topology code (root cause unverified). The spike used KA 0.9.45 with Metal.jl 1.11.1.
- Not measured: real sparse U matrices; whether a trimmed binary can use Metal; loops, KA and TensorOperations for the full contraction at 10,976 atoms.

## 11. Left to the builder

- Which thermostat (Nosé-Hoover or Bussi) and barostat (Bussi or Berendsen) to write first.
- Which metal and alloy for milestone 1, as long as published vacancy diffusion values exist for comparison.
- Internal module layout, type hierarchy, and the exact potential interface between the packages, subject to the GPU-residency and shared-neighbor-list requirements.
- Inflate support: needed only if a published `.model` file turns out to contain deflated entries (torch writes them uncompressed; unverified for the released files).

## 12. Risks

- A pure-Julia MACE core (tensor product, symmetric contraction, all pullbacks, second order on GPU) is the largest piece of new code. The Molly.jl Allegro pull request (#283, KernelAbstractions, e3nn conventions, analytic tensor-product pullbacks) is the closest existing work: a source to learn from, a possible collaboration, and a possible duplicate.
- Enzyme with KernelAbstractions on GPU has open bugs; the design does not depend on it.
- Several packages in the survey cap KernelAbstractions at 0.9 while 0.10 is current; mixing them may not resolve.
- The 2x target is measured, not assumed. If the first attempt misses 1.05x, profile before redesigning.

## 13. Calls made by the design author (not user decisions; check these)

- **CPU and GPU code paths.** The CPU path is plain generic Julia plus PureBLAS, so it can be trimmed. KernelAbstractions is used for the GPU paths only, following the current KA release as the user accepted. Reason: the KA 0.10 CPU backend compiles at run time (PoCL/SPIR-V), which the survey judges very unlikely to trim.
- **Test-only dependencies** (Enzyme, JET, AllocCheck, TrimCheck) may carry jlls, as PureBLAS's test environment already does. The no-artifact rule applies to the packages' own dependency closures.
- **ForwardDiff** (pulls OpenSpecFun_jll through SpecialFunctions) is used only for training. Training code lives in a package extension or subpackage, so the inference and MD closure stays artifact-free.
- **Minimum Julia version: 1.12**, because `juliac --trim` requires it. No 1.10 support.
- **Molly.jl PR #283** is reference reading only. Contacting its authors happens only with the user's approval of the exact text (the user's GitHub rule).
- **Reference data storage.** Checkpoints are downloaded at test time by URL with a SHA-256 check (stdlib Downloads, not Artifacts). LAMMPS reference outputs are committed as small JSON files with the machine, LAMMPS version and model hash. CI never runs LAMMPS.

## 14. User decisions after review

- **KernelAbstractions 0.10 with the current GPU packages** (Metal.jl 1.12+, AMDGPU.jl 2.9+, current CUDA.jl), not a 0.9 pin. The 0.10.4 precompile hang on the Apple M6 Mac (inside PoCL's hardware-topology code) is a bug to reproduce and fix as an early plan task. Any upstream issue text needs the user's approval before it is posted. A temporary fallback to 0.9 is only taken if the user decides it then.
- **Speed baseline (agreed):** same device and same precision on both sides; LAMMPS-MACE with cuEquivariance on for NVIDIA; Float32; measured at about 10,000 atoms. Milestone 1 is signed off only at 1.05x or better on each machine that has a same-device baseline. On the Mac, Metal is compared against LAMMPS on CPU and reported, not gated, unless LAMMPS-MACE turns out to support the Mac GPU.
- **Trimmed binary (option C):** everything builds with `juliac --trim`: the MD driver, the model, the neighbor list, the trajectory writer, and the `.model` reader. A trim gate (TrimCheck test items plus a real `juliac` build) covers the whole path from reading a `.model` file to writing a trajectory. GPU paths in the trimmed binary are not required for milestone 1 (unverified whether they can be trimmed).
- **Python session (agreed):** one session exports three files: MPA-0 Float64, MPA-0 Float32, MP-0b3 Float64. No later additions.
- **Milestones (agreed):** milestone 1 supports MPA-0 only (MP-0b3 is used only as a parser test against llmlip.rs's inspection file). Milestone 2 is the first training set in section 8, on CPU first. Everything else in the mace-torch training list (LoRA, multi-head replay, the other losses, other interaction blocks, GPU training) is backlog without a date.
