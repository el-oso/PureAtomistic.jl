# Plan 3: the MACE model on CPU (energy, forces, stress)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PureMLIP evaluates MACE-MPA-0 (and MACE-MP-0b3 as a second check) on CPU in Float64 and Float32: total and per-atom energy, forces and stress, matching LAMMPS with MACE's libtorch pair style to about 1e-6 eV, allocation-free in a preallocated workspace, trim-compatible.

**Architecture:** A third submodule, `PureMLIP.Model` (`src/model/`), built only on the public API of `PureMLIP.Checkpoint` and `PureMLIP.Equivariant`. `Checkpoint` additionally extracts the pickled constants the forward pass needs (`normalize2mom.cst`, e3nn instruction lists with path weights, flags). `Model` holds a CellListMap-backed periodic neighbor list, the layers of the ScaleShiftMACE forward pass written as plain generic loops plus `mul!`, a hand-written pullback per layer, and the assembled `mace_energy!` / `mace_energy_forces!` entry points. Reference data comes from one LAMMPS-MACE run on this Mac, committed as JSON.

**Tech Stack:** Julia 1.12+ (CI 1.12 and 1.13). No new package dependency (the neighbor list is an own cell list). New test-side data: e3nn-core-rs `mace_radial_reference_f64.json`, LAMMPS-MACE reference JSON. Benchmark-only: Chairmarks.jl in `PureMLIP/bench/Project.toml`. Reference tooling outside the repository: LAMMPS (ACEsuit `mace` branch) with libtorch 2.8.0, Homebrew `cmake` and `open-mpi`.

**Spec:** `docs/superpowers/specs/2026-10-30-pureatomistic-design.md`, sections 4 (constraints), 6.1 (PureMLIP parts), 6.3 (engineering stack), 7 (milestone 1), 9 (references), 13 (design calls), 14 (user decisions). Repository rules: `AGENTS.md`.

## Global Constraints

- **No Python in any capacity in anything an agent runs.** One Python session is allowed, once, run by the user by hand, in a throwaway folder outside all projects, only to export three LAMMPS model files with MACE's `create_lammps_model.py`: MPA-0 Float64, MPA-0 Float32, MP-0b3 Float64. The Float64 exports run with `torch.set_default_dtype(torch.float64)` set before the script runs. Nothing Python, and no exported model file, enters the repository. The folder is deleted afterward.
- **References:** e3nn-core-rs fixtures (already vendored, plus `mace_radial_reference_f64.json`), the llmlip.rs MP-0b3 inspection, and LAMMPS with MACE's libtorch pair style on this Mac. Reference outputs are committed JSON carrying the software versions and the SHA-256 of every model file. CI never runs LAMMPS.
- **Correctness target (Float64):** energy within 1e-6 eV absolute per structure and every force component within 1e-6 eV/Å of LAMMPS-MACE; per-atom energies within 1e-6 eV; virial components within 1e-6 eV. Physics checks: rotation and translation invariance, forces equal minus the finite-difference energy gradient, stress equals the strain derivative. Float32 tolerances are measured margins (AGENTS.md policy (c)).
- **Forces and stress come from hand-written pullbacks, one per layer**, each checked against ForwardDiff and finite differences. ForwardDiff and any Enzyme use are test-only.
- **`normalize2mom` constants are read from each checkpoint**, never hard-coded or recomputed (the stored value is 1.6e-3 relative above the exact integral, and MPA-0 and MP-0b3 differ by 1 ulp).
- **PureMLIP `[deps]` after this plan:** Inflate, LinearAlgebra, PrecompileTools (unchanged; the neighbor list is an own cell list, not CellListMap). Pure Julia; no `_jll`, no artifacts. Every dependency has a `[compat]` bound.
- **Code rules (AGENTS.md):** no `@inbounds`; generic indexing or a declared `Base.require_one_based_indexing`; concrete struct fields and return types; `!` functions allocation-free after warm-up; fail-fast errors with a module prefix (`"model: ..."`, `"neighbor list: ..."`, `"checkpoint: ..."`) or `ArgumentError`/`DimensionMismatch`; `lazy"..."` for interpolations of four or more pieces; comments state present facts only.
- **Layer graph:** `PureMLIP => (Checkpoint, Equivariant, Model)`, `Model => (Checkpoint, Equivariant)`, `Checkpoint => ()`, `Equivariant => ()`. `Model` reads only exported or `public` names of `Checkpoint` and `Equivariant`.
- **Gates for every new exported function:** TrimCheck `@validate` and `test_signatures(...; guarantees = (:typestable,))`; `@test_noalloc` for every exported `!` function; a row in `test/coverage/thresholds.toml` for every new `src` file; at least one entry in `test/mutation/mutations.toml` per new `src` file except declaration-only files (`src/model/Model.jl`).
- **Precompile:** the workload gains `load_mace_checkpoint` plus one model evaluation on a tiny synthetic checkpoint built in memory.
- **Threading:** not part of this plan; every kernel is single-threaded.
- **Supported architecture:** exactly the ScaleShiftMACE shape of MPA-0 and MP-0b3 (section "Forward pass" below). Channel counts, element count, Bessel count, radial-MLP widths and U-matrix path counts are read from the checkpoint, not assumed. Anything else throws `model: unsupported checkpoint: <reason>`.
- **Speed:** measured, recorded, not gated (gating belongs to milestone 1).
- **Commits:** repository git identity; subject ≤ 72 characters (aim for 50); body for a reader with only the repository; trailers `Assisted-by: Claude <model> <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01RQSsNnmff93o3JgocxDLMn`. After the final review, push straight to `main`; no pull requests. No host names or local absolute paths in any tracked file or commit message (use `$LAMMPS_SRC`-style placeholders).
- **Commands.** Full suite: `PUREMLIP_CHECKPOINT_DIR=<dir> julia --project=PureMLIP -e 'using Pkg; Pkg.test()'`. One item (written below as "Run item `<pattern>`"): `julia --project=PureMLIP -e 'using TestEnv; TestEnv.activate(); using TestItemRunner; TestItemRunner.run_tests("PureMLIP"; filter = ti -> occursin(ARGS[1], ti.name))' '<pattern>'` from the repository root (or the same call in a warm MCP session).

### Verified facts

Checked 2026-10-31 by reading source at the pins below, by Julia scripts against the vendored and downloaded files, and by registry queries. No Python was run. **UNVERIFIED** marks what was not read or measured.

**Pins.** mace-torch `github.com/ACEsuit/mace` @ `4d2da09413ac1407f37cdbb6b81fa28e4c15655e` (0.3.16; the loader all citations refer to, `file:line` at that commit); e3nn 0.4.4; reference environment of the llmlip.rs fixture: e3nn 0.4.4, mace 0.3.16, torch 2.8.0. LAMMPS: `github.com/ACEsuit/lammps` branch `mace` @ `4d222cb3ee2a6b14083c778968497bf9e0efc4b4` (2025-09-18, `LAMMPS_VERSION "10 Sep 2025"`). libtorch: `https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.8.0.zip` (redirects to `download-r2.pytorch.org`, 71,787,762 bytes). e3nn-core-rs @ `9da369cf1d854da6af1a46928bc953c7966e9382`, file `crates/e3nn-core/fixtures/mace/v0_3_16_4d2da09/mace_radial_reference_f64.json` (5,350 bytes, SHA-256 `0f81adb38387bb16a91b71915ef5963f1f808554d9c12c5a659b45efc11e2e7b`; generated by mace 0.3.16 @ 4d2da09, torch 2.8.0, float64).

**CellListMap dependency closure** (General registry, latest 0.10.10, weak dependencies excluded, recursive): ChunkSplitters, Compat, ConstructionBase, MacroTools, PrecompileTools, Preferences, ProgressMeter, Setfield, StaticArrays, StaticArraysCore, Statistics, TestItems, TOML, plus stdlibs. **No `_jll`.** Not checked: TrimCheck on its calls, allocation-free updates, cutoffs larger than half the cell width, its `unitcell` column convention (all **UNVERIFIED**; Task 2 Step 1 checks them).

**Radial fixture** (`mace_radial_reference_f64.json`): graph of 3 nodes with `species_atomic_numbers = [1, 6, 8]`, `edge_index = [[0,1,1,2,0],[1,0,2,1,2]]` (row 0 sender, row 1 receiver), `distances = [0.43, 0.61, 0.74, 0.92, 0.57]`. `bessel`: `r_max = 5.0`, `num_basis = 4`, `cutoff_p = 6`, Bessel on the raw distance; `agnesi`: `a = 1.0805, p = 4.5791, q = 0.9183`; `zbl`: `c = [0.1818, 0.5099, 0.2802, 0.02817]`, `p = 6`, `a_exp = 0.3`, `a_prefactor = 0.4543`. Each block carries cotangents and `vjp_distances`. The formulas in "Forward pass" with covalent radii H 0.31, C 0.76, O 0.66 reproduce `agnesi.output` exactly, `bessel.basis`/`embedding` to 2.2e-16, `bessel.cutoff` exactly, `zbl.output_node_energy` to 8.9e-16 (measured).

**Tensor-product fixtures** (`fully_connected_tensor_product{,_lmax4}_smoke_f64.json`, vendored): per sample `x`, `y`, `w` (one weight per instruction), `out`, `grad_out`, `grad_x`, `grad_y`, `grad_w`; per instruction segments, `instruction_path_weight`, `instruction_weight_offset`. `out[c3+k] = Σ_instr pw · w[instr] · Σ C[i,j,k] x[c1+i] y[c2+j]` reproduces `out` to 6.9e-14 and the three VJPs to 1.3e-11 (measured), within the fixtures' stated 1e-10.

**Pickled constants** (read with the plan-1 reader, both files; module path → value):

| module path | MPA-0 | MP-0b3 |
|---|---|---|
| `interactions.{0,1}.conv_tp_weights.layer{0,1,2}.act`, `readouts.1.non_linearity.acts.0` (class `e3nn.math._normalize_activation.normalize2mom`) | `cst = 1.679176792398942`, `f = torch.nn.functional.silu` | `cst = 1.6791767923989418`, same `f` |
| root attribute `pair_repulsion` | `true` | `true` |
| `products.0.use_sc`, `products.1.use_sc` | `false`, `true` | same |
| `node_embedding.linear` path weights | `[0.105999788000636]` | same |
| `interactions.0.linear_up` | `[0.08838834764831845]` | same |
| `interactions.0.conv_tp` (TensorProduct) | `[1.0, 1.7320508075688772, 2.23606797749979, 2.6457513110645907]` | same |
| `interactions.0.linear` | `0.08838834764831845` × 4 | same |
| `interactions.0.skip_tp` (FullyConnectedTensorProduct) | `[0.009369146112448268, 0.01622783709029683, 0.020950047598562217, 0.024788430610565716]` | same |
| `interactions.1.linear_up` | `0.08838834764831845` × 2 | same |
| `interactions.1.conv_tp` | `[1.0, 1.0, 1.7320508075688772 ×3, 2.23606797749979 ×3, 2.6457513110645907 ×2]` | same |
| `interactions.1.linear` | `[0.0625, 0.05103103630798288, 0.05103103630798288, 0.0625]` (4 instructions over `256x0e+384x1o+384x2e+256x3o`) | `[0.0625, 0.0625, 0.05103103630798288 ×6, 0.0625, 0.0625]` (10 instructions over 10 slots of 128) |
| `interactions.1.skip_tp` | `[0.009369146112448268]` | same |
| `products.0.linear`, `products.1.linear` | `0.08838834764831845` × 2, × 1 | same |
| `readouts.0.linear`, `readouts.1.linear_1`, `readouts.1.linear_2` | `[0.08838834764831845]`, `[0.08838834764831845]`, `[0.25]` | same |

Other globals in the MPA-0 pickle: `e3nn.o3._linear.Linear`/`Instruction`, `e3nn.o3._tensor_product._tensor_product.TensorProduct`, `..._sub.FullyConnectedTensorProduct`, `..._instruction.Instruction`, `e3nn.nn._fc.FullyConnectedNet`/`_Layer`, `e3nn.nn._activation.Activation`, `mace.modules.radial.{AgnesiTransform, BesselBasis, PolynomialCutoff, ZBLBasis}`. Exact integral for comparison: `E[silu(Z)²] = 0.35577551981735223`, `cst_exact = 1.6765324703310911`.

**Folded constants.** `pw_l * (1 / sqrt(2l+1))` with the `skip_tp` path weights equals the embedded-code literals `0.0093691461124482678, 0.0093691461124482695, 0.0093691461124482678, 0.009369146112448266` bit-exactly (division `pw_l / sqrt(2l+1)` misses l = 3 by 1 ulp). `Float64(n)^-0.5` equals the stored Linear path weights for n = 89, 128, 256, 384, 16 bit-exactly; `1/sqrt(128)` and `1/sqrt(384)` do not. `1.0 * (1/sqrt(3)) == 0.57735026918962584`.

**State tensors** (MPA-0; MP-0b3 identical names and shapes): `node_embedding.linear.weight [11392]`; `radial_embedding.bessel_fn.{bessel_weights [10], r_max, prefactor}`; `radial_embedding.distance_transform.{q, p, a, covalent_radii [119]}`; `radial_embedding.cutoff_fn.{p, r_max}`; `pair_repulsion_fn.{c [4], p, covalent_radii [119], a_exp, a_prefactor}` (its `r_max` and `cutoff.*` are unused); `atomic_energies_fn.atomic_energies [89]`; `interactions.0.{linear_up.weight [16384], conv_tp_weights.layer0..3.weight [10,64] [64,64] [64,64] [64,512], linear.weight [65536], skip_tp.weight [5832704], density_fn.layer0.weight [10,1]}`; `interactions.1.{linear_up.weight [32768], conv_tp_weights.layer3.weight [64,1280], linear.weight [163840], skip_tp.weight [1458176], density_fn.layer0.weight [10,1]}`; `products.0.symmetric_contractions.contractions.{0,1}.{weights_max, U_matrix_1..3, weights.0, weights.1}`, `products.0.linear.weight [32768]`; `products.1.symmetric_contractions.contractions.0.*`, `products.1.linear.weight [16384]`; `readouts.0.linear.weight [128]`, `readouts.1.linear_1.weight [2048]`, `readouts.1.linear_2.weight [16]`; `scale_shift.{scale, shift}`. Bias tensors have length 0.

**LAMMPS pair style** (`src/ML-MACE/pair_mace.cpp` at the LAMMPS pin): `pair_style mace no_domain_decomposition`; `pair_coeff * * <model.pt> <element symbols per type>`; requires `newton on` (error otherwise) and uses `atom->map` (so `atom_modify map yes`); full neighbor list with `rsq < r_max²`; edge `sender = i` (local), `receiver = j_local`, `shift = x[ghost] − x[j_local]`; the model dtype is taken from `model.node_embedding.linear.weight`; the code never sets libtorch's default dtype. Energy is `total_energy_local` (sum of the model's `node_energy`), per-atom energy is `node_energy`; `virial[0..5] += vir_xx, vir_yy, vir_zz, sym xy, sym xz, sym yz` with `vir = −∂E/∂ε` from `LAMMPS_MACE.forward` (`mace/calculators/lammps_mace.py`). Metal units: `nktv2p = 1.6021765e6` (`src/update.cpp`). `src/STUBS/mpi.h` declares no `MPI_Comm_split_type`, which `pair_mace.cpp` calls, so the build needs a real MPI. `LAMMPS_MACE` returns `node_energy` from the model; whether it includes E0 is **UNVERIFIED** (the `isolated_cu` structure decides it).

**Export script** (`mace/cli/create_lammps_model.py` at the mace pin): arguments `model_path`, `--dtype` (default `float64`), `--format` (default `libtorch`), `--head`; one head → no prompt; writes `<model_path>-lammps.pt`; does not set torch's default dtype. Its `cuequivariance` and `lammps.mliap` imports are guarded by `try/except ImportError`.

**Reference structures** (generated by the formulas in Task 0; directed edge counts with `d < 6.0` by a brute-force image sum with image ranges ±5, identical with ±3):

| name | atoms | composition | cell (rows, Å) | edges | min distance (Å) | max image index |
|---|---|---|---|---|---|---|
| `cu_vacancy` | 31 | Cu 31 | 7.23·I | 2340 | 2.4507 | 1 |
| `crconi_alloy` | 32 | Cr 10, Co 11, Ni 11 | 7.12·I | 2496 | 2.4258 | 1 |
| `triclinic_ni3al` | 4 | Ni 3, Al 1 | (4.2,0,0),(1.3,4.0,0),(0.9,1.1,4.4) | 176 | 2.4703 | 2 |
| `multi_element` | 8 | H C N O Fe Cu Zr W | 10·I | 44 | 1.0 | 0 |
| `isolated_cu` | 1 | Cu | 20·I | 0 | – | – |

### Forward pass (ScaleShiftMACE, mace 0.3.16 loader; e3nn 0.4.4 embedded TorchScript)

Notation: `n` atoms, `E` directed edges, `s_i` the 1-based index of `Z_i` in `atomic_numbers` (the z table, stored order; an element outside it is an error). Torch flat weights are row-major: `(a,b)` stores `W[p,q]` at `p*b+q`, `(a,b,c)` stores `W[p,q,r]` at `(p*b+q)*c+r` (0-based), so `torch_array` gives Julia arrays with the same index order. e3nn feature layout: irreps blocks in order, each block multiplicity-major `x[u*(2l+1)+m]`. Both checkpoints: `r_max = 6.0`, two interactions, 89 elements (`[1..83, 89..94]`), one head, hidden `128x0e+128x1o`, `max_ell = 3`, 10 Bessel functions, cutoff `p = 5`, radial MLP `[64,64,64]`, correlation 3, `avg_num_neighbors = 61.964672446250916` stored and **not used**. `scale` MP-0b3 `0.8140724313304148`, MPA-0 `0.7735790334431056`; both `shift = 0.0`. Atomic energies, U matrices, Bessel weights and covalent radii are bit-identical between the files.

1. **Edges** (`data/neighborhood.py:7-66`, `modules/utils.py:261-276`): `r_e = pos[receiver] − pos[sender] + shift_e`, `shift_e = cell · n_e` (integer `n_e`), `d_e = ‖r_e‖`. Self edges with zero shift are dropped; self edges through an image are kept. Messages flow sender → receiver. An edge with `d_e ≥ 6` contributes exactly zero.
2. **E0 and embedding** (`blocks.py:42-58, 359-377`): `E0_i = atomic_energies[s_i]` (not scaled). `h0_i[w] = 0.105999788000636 · W_emb[s_i, w]`, `W_emb = reshape(weight, 89, 128)` row-major.
3. **Radial embedding** (`radial.py:17-57, 112-145, 224-281`, `blocks.py:390-431`): `f_cut(d) = (1 − 21t⁵ + 35t⁶ − 15t⁷)·[d < r_max]`, `t = d/r_max` (general `p`: coefficients `(p+1)(p+2)/2, p(p+2), p(p+1)/2`, integer powers). Agnesi: `r0 = 0.5(R_cov[Z_u] + R_cov[Z_v])` (`covalent_radii` indexed by atomic number, entry 0 a dummy), `x = d/r0`, `d̃ = 1/(1 + a·x^q/(1 + x^(q−p)))`, `q = 0.9183, p = 4.5791, a = 1.0805`. Bessel on `d̃`: `B_n = prefactor · sin(w_n d̃)/d̃`, `w_n = π/6·n` bit-exact as `pi / 6.0 * n`, `prefactor = 0.5773502691896257`. `edge_feats = B · f_cut(d)` (raw `d` in the cutoff), `[E, 10]`. The cutoff is folded into `edge_feats`; nothing else is multiplied by it.
4. **ZBL** (`radial.py:187-218`, `models.py:507-514`): `a_e = 0.4543·0.529/(Z_u^0.3 + Z_v^0.3)`, `ρ = d/a_e`, `φ = 0.1818e^{−3.2ρ} + 0.5099e^{−0.9423ρ} + 0.2802e^{−0.4028ρ} + 0.02817e^{−0.2016ρ}`, `v_e = 14.3996·Z_u Z_v/d·φ`, `R_e = R_cov[Z_u] + R_cov[Z_v]`, `env_e = (1 − 21τ⁵ + 35τ⁶ − 15τ⁷)·[d < R_e]`, `τ = d/R_e`, powers as `pow(τ, p)` with the Float64 buffer `p = 5.0`; `V_i = Σ_{receiver(e)=i} 0.5·v_e·env_e`. **UNVERIFIED:** under a float32 torch default dtype `14.3996·Z_u·Z_v` is float32-rounded; the Float64 exports set the default dtype to float64; LAMMPS's run-time default dtype is not set by the pair style.
5. **Spherical harmonics:** `edge_attrs = e3nn_spherical_harmonics!(Y, r_e, 3)` (16 values, `Y_0 = 1`), gradients from `e3nn_spherical_harmonics_and_gradient!` (measured 3.3e-15 against e3nn 0.4.4's polynomials).
6. **Radial MLP** (e3nn `nn/_fc.py:38-48, 69-87`): `a1 = cst·silu(feats·(L0/√10))`, `a2 = cst·silu(a1·(L1/8))`, `a3 = cst·silu(a2·(L2/8))`, `tp_w = a3·(L3/8)`; weights `[h_in, h_out]`, divided before the product; `W = 512` (interaction 0) or 1280. **Density** (`blocks.py:899-905, 1022-1028`): `ρ_e = tanh((feats_e · L_d[:,1]/√10)²)`, `density_i = Σ_{receiver=i} ρ_e`.
7. **Interaction 0** (`RealAgnosticDensityInteractionBlock`, `blocks.py:876-925`): `x = 0.08838834764831845·h0·W_up`; `m_e,l[u,m] = tp_w[e, 128l+u]·x_sender[u]·Y_l[m]` (path weights fold to exactly 1); `M_i = Σ_{receiver=i} m_e`; `L_i,l = 0.08838834764831845·M_i,l·W_l/(density_i + 1)` (4 consecutive `[128,128]` blocks); `S_i,l[w,m] = c_l Σ_u W_l[u, s_i, w]·L_i,l[u,m]` with `c_l` the folded constants above and `skip_tp.weight` as 4 consecutive `[128, 89, 128]` blocks; output `X1[u, l²+m] = S_l[u,m]`, `[n, 128, 16]`.
8. **Interaction 1** (`RealAgnosticDensityResidualInteractionBlock`, `blocks.py:998-1049`): first `sc_i[w] = 0.0093691461124482678·Σ_u W[u, s_i, w]·a_i[u]` on the block input's `0e` part; `linear_up` two `[128,128]` blocks; `tp_w [E, 1280]`; conv_tp with 10 output slots `0e,0e,1o,1o,1o,2e,2e,2e,3o,3o` (5120 per edge), slot `k` weight `tp_w[e, 128k+u]`, instructions `(i1,i2,iout)` = (0,0,0) (1,1,1) (0,1,2) (1,0,3) (1,2,4) (0,2,5) (1,1,6) (1,3,7) (0,3,8) (1,2,9) with the path weights above; each slot equals `pw · Σ_{i,j} C_{l1 l2 l3}[i,j,k]·x_sender[u,i]·Y_l2[j]·w` with `C = e3nn_wigner_3j(l1,l2,l3)` (the specialized rows agree because `e3nn_wigner_3j(l,l,0)`, `(0,l,l)`, `(l,0,l)` equal `I/sqrt(2l+1)` to ≤ 8.3e-16); `M_i = Σ_{receiver=i} m_e`; `linear` → 2048: `out_0e = 0.0625·(slots 0,1)`, `out_1o = 0.05103103630798288·(slots 2,3,4)`, `out_2e = 0.05103103630798288·(slots 5,6,7)`, `out_3o = 0.0625·(slots 8,9)`, each slot its own consecutive `[128,128]` block of `interactions.1.linear.weight` (the MPA-0 `[256,128]`/`[384,128]` blocks are the same flat layout), divided by `(density_i + 1)`; reshape to `X2 [n,128,16]`. Returns `(X2, sc)`.
9. **Symmetric contraction** (`symmetric_contraction.py:85-87, 128-216, 239-261`): products.0 → `[0e, 1o]`, products.1 → `[0e]`. U shapes and nonzero counts: l=0 `U1 [16,1]` 1, `U2 [16,16,4]` 16, `U3 [16,16,16,23]` 353; l=1 `U1 [3,16,1]` 3, `U2 [3,16,16,6]` 70, `U3 [3,16,16,16,51]` 2846. `weights_max [89,K3,128]` pairs with U3, `weights.0` with U2, `weights.1` with U1. `U3` is not symmetric in its ell indices; use it as stored. With `X[c,i]` (`i = l²+m`, 0..15) and species `s`:
   `A3[c,w,x] = Σ_{i,k} U3[w,x,i,k]·Wmax[s,k,c]·X[c,i]`; `C2 = Σ_k U2[w,x,k]·W0[s,k,c] + A3`; `A2[c,w] = Σ_x C2[c,w,x]·X[c,x]` (contracts the last ell index); `C1 = Σ_k U1[w,k]·W1[s,k,c] + A2`; `B0[c] = Σ_w C1[c,w]·X[c,w]`. For l=1 every U and C has a leading index `m ∈ 0..2`, giving `B1[c,m]`. Output `[B0 (128), vec(B1) (128×3, c-major)]`.
   products.0: `h1 = linear(B)` (two blocks, `use_sc = false`); products.1: `h2 = linear(B) + sc` (`use_sc = true`).
10. **Readouts** (`blocks.py:62-115`): `e1_i = 0.08838834764831845·Σ_u h1_i[u]·w_r0[u]` (the `1o` part unused); `g = 0.08838834764831845·h2·W1 [128,16]`, `g = cst·silu(g)`, `e2_i = 0.25·Σ_j g[j]·w_r1[j]`.
11. **Energy** (`models.py:540-581`): `node_energy_i = E0_i + scale·(V_i + e1_i + e2_i) + shift`; `E = Σ_i node_energy_i`. The ZBL energy is inside the scaled sum.
12. **Forces and stress** (`models.py:583-595`, `utils.py:22-109`): with `g_e = ∂E/∂r_e`, `F_i = Σ_{sender=i} g_e − Σ_{receiver=i} g_e`; `∂E/∂ε = sym(Σ_e r_e ⊗ g_e)`, `virial = −∂E/∂ε`, `stress = ∂E/∂ε / |det cell|` (positive tensile). Each unordered pair appears as two directed edges.

**UNVERIFIED (from the reconnaissance):** the exact mace commit that wrote MPA-0 (evidence: v0.3.8-era) and the e3nn version of its embedded code; the float32 rounding of the ZBL prefactor under a float32 default dtype; the energy error from using the exact `cst` (estimated at meV per atom); PureMLIP's spherical harmonics at a zero vector (they throw `ArgumentError`; such edges are rejected earlier by the neighbor list); the `normalize` eps `1e-12` (from torch docs).

## Review Focus

1. **An element outside the model's 89-element table** (for example Po, Z = 84, which MPA-0 lacks): expected `ArgumentError` naming the element, not a wrong index. Test in Task 6.
2. **A MACE checkpoint of a different architecture** (another interaction class, correlation, readout or a missing ZBL): expected `model: unsupported checkpoint: <reason>` at model construction, never a silently different energy. Test in Task 6.
3. **Unwrapped positions** (atoms outside the cell, as after MD): energy, forces and stress identical to the wrapped configuration within rounding. Test in Tasks 2 and 7.
4. **Small periodic cells** (cutoff larger than half a cell width, several images of the same pair, self-images): every image within `r_max` counted. Test in Task 2 (`triclinic_ni3al`, 176 edges) and Task 6 against LAMMPS.
5. **A neighbor list or workspace too small for the configuration** (MD compresses the system): expected an `ArgumentError` naming the needed and available sizes, never an out-of-range access. Test in Tasks 2 and 7.

---

### Task 0: Reference data from LAMMPS-MACE

**Files:**
- Create: `PureMLIP/test/reference/structures.jl` (deterministic structure generator, no dependencies beyond LinearAlgebra), `PureMLIP/test/reference/run_lammps.jl` (writes LAMMPS data files, runs LAMMPS as a subprocess, writes the JSON), `PureMLIP/test/reference/in.mace` (LAMMPS input template)
- Create: `PureMLIP/test/fixtures/lammps-mace/{mpa0_f64,mpa0_f32,mp0b3_f64}.json`, `PureMLIP/test/fixtures/lammps-mace/NOTICE`
- Create: `PureMLIP/test/fixtures/e3nn-core-rs/mace_radial_reference_f64.json` (verbatim from the e3nn-core-rs pin); modify that folder's `NOTICE` to list it
- Modify: `PureMLIP/test/testutils.jl` (new `@testmodule LammpsReference`); Create: `PureMLIP/test/reference_tests.jl`

**Interfaces:**
- Produces `structures.jl`: `reference_structures()::Vector{@NamedTuple{name::String, cell::Matrix{Float64}, positions::Matrix{Float64}, atomic_numbers::Vector{Int}}}` with `cell` rows = lattice vectors (LAMMPS restricted triclinic form) and `positions` one row per atom, in the order of the table above, generated as:
  - fcc sites: `for i in 0:N-1, j in 0:N-1, k in 0:N-1, b in 1:4` (i slowest) → `a .* ([i,j,k] .+ F[b,:])`, `F = [0 0 0; 0.5 0.5 0; 0.5 0 0.5; 0 0.5 0.5]`.
  - `cu_vacancy`: `a = 3.615`, `N = 2`, drop site 1; atom `n` (1..31) displaced by `0.05·sin(1.3·(3n+d))` Å in component `d`.
  - `crconi_alloy`: `a = 3.56`, `N = 2`; `Z[s] = (24, 27, 28)[mod(5s + s ÷ 3, 3) + 1]`; displacement `0.04·sin(0.7·(3s+d) + 0.3)`.
  - `triclinic_ni3al`: rows `[4.2 0 0; 1.3 4.0 0; 0.9 1.1 4.4]`; fractional `F` rows times the cell rows; displacement `0.03·sin(2.1·(3k+d))`; `Z = [28, 28, 13, 28]`.
  - `multi_element`: cell `10·I`; Z `[6, 1, 8, 7, 26, 29, 40, 74]` at `(5,5,5), (6,5,5), (5,6.25,5), (3.8,4.4,5.6), (7.5,7.5,2.5), (2.5,7.5,7.5), (7.6,2.4,7.4), (2.4,2.6,2.4)`.
  - `isolated_cu`: cell `20·I`, one Cu at `(10,10,10)`.
  - Benchmark structures (not in the JSON): `cu_bench(N)` = perfect fcc Cu, `a = 3.615`, displacement `0.05·sin(1.3·(3n+d))`, `N = 5` (500 atoms) and `N = 14` (10,976 atoms).
- Produces JSON schema `puremlip-lammps-mace-v1`: top level `schema`, `model` (`name`, `source_url`, `source_sha256`, `export_sha256`, `dtype`, `export_default_dtype`), `software` (`lammps_version`, `lammps_repository`, `lammps_commit`, `libtorch`, `torch`, `mace`, `mace_commit`, `e3nn`), `machine` (CPU model string only), `nktv2p` (`1.6021765e6`), `structures`: per structure `name`, `cell` (rows), `positions` (rows), `atomic_numbers`, `energy` (eV), `node_energies` (eV), `forces` (rows, eV/Å), `pressure_virial_bar` (LAMMPS `compute pressure NULL virial`: `[pxx, pyy, pzz, pxy, pxz, pyz]`), `volume` (Å³). Every float written with 17 significant digits.
- Produces `LammpsReference.load(name::Symbol)` (`:mpa0_f64`, `:mpa0_f32`, `:mp0b3_f64`) returning the parsed JSON; `LammpsReference.structure(ref, name::String)` returning `(; name, positions::Matrix{Float64} (3×n), cell::Matrix{Float64} (columns = lattice vectors), atomic_numbers::Vector{Int}, energy::Float64, node_energies::Vector{Float64}, forces::Matrix{Float64} (3×n), virial::Matrix{Float64})` with `virial[a,b] = p_ab · volume / nktv2p` (symmetric 3×3, eV); `LammpsReference.structures(ref)` returning all of them in file order.

- [ ] **Step 1: Write the failing test** in `reference_tests.jl`:

```julia
@testitem "LAMMPS reference: structures, metadata, sanity" setup = [LammpsReference, Checkpoints] begin
    using SHA
    include(pkgdir(PureMLIP, "test", "reference", "structures.jl"))
    gen = reference_structures()
    for name in (:mpa0_f64, :mpa0_f32, :mp0b3_f64)
        ref = LammpsReference.load(name)
        ϵ = name == :mpa0_f32 ? Float64(eps(Float32)) : eps(Float64)
        @test ref["schema"] == "puremlip-lammps-mace-v1"
        @test [s["name"] for s in ref["structures"]] == [g.name for g in gen]
        for g in gen
            s = LammpsReference.structure(ref, g.name)
            # Written with 17 significant digits: the round trip is exact.
            @test s.positions == permutedims(g.positions) && s.cell == permutedims(g.cell)
            @test s.atomic_numbers == g.atomic_numbers
            # Newton's third law: rounding bound 64·ϵ·Σ|F| in the export's precision (n ≤ 32 atoms,
            # ≤ 100 edge terms per atom); isolated_cu has no edges and its forces are exactly zero.
            @test maximum(abs, sum(s.forces; dims = 2)) <= 64 * ϵ * sum(abs, s.forces)
        end
        src = name == :mp0b3_f64 ? :mp0b3 : :mpa0
        @test ref["model"]["source_sha256"] == bytes2hex(open(sha256, Checkpoints.fetch(src)))
    end
    # isolated_cu has no edges: its LAMMPS energy is E0(Cu) exactly when node_energy includes E0.
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0))
    e0cu = ck.state.f64["atomic_energies_fn.atomic_energies"].data[findfirst(==(29), ck.hyperparameters.atomic_numbers)]
    @test LammpsReference.structure(LammpsReference.load(:mpa0_f64), "isolated_cu").energy == e0cu
end
```

- [ ] **Step 2:** Run item `LAMMPS reference`; expected FAIL (no JSON).
- [ ] **Step 3: Build LAMMPS-MACE** outside the repository (agent): `brew install cmake open-mpi`; download and unzip libtorch 2.8.0 (URL above) to `$LIBTORCH`; `git clone --branch mace https://github.com/ACEsuit/lammps $LAMMPS_SRC && git -C $LAMMPS_SRC checkout 4d222cb3ee2a6b14083c778968497bf9e0efc4b4`; `cmake -S $LAMMPS_SRC/cmake -B $LAMMPS_BUILD -D CMAKE_BUILD_TYPE=Release -D CMAKE_PREFIX_PATH=$LIBTORCH -D CMAKE_CXX_STANDARD=17 -D PKG_ML-MACE=ON -D BUILD_MPI=ON -D BUILD_OMP=OFF -D PKG_PYTHON=OFF && cmake --build $LAMMPS_BUILD -j 6`. Verify: `$LAMMPS_BUILD/lmp -h` lists pair style `mace`. If the libtorch CMake configuration needs Python, stop and report to the user.
- [ ] **Step 4: Python export (user, by hand, once).** In a new empty folder outside every project, with the two checkpoints copied in (SHA-256 as in `Checkpoints.SOURCES`):

```bash
python3 -m venv venv
venv/bin/pip install torch==2.8.0 e3nn==0.4.4 "mace-torch @ git+https://github.com/ACEsuit/mace.git@4d2da09413ac1407f37cdbb6b81fa28e4c15655e"
X='import sys, runpy, torch; torch.set_default_dtype(getattr(torch, sys.argv[1])); sys.argv = ["create_lammps_model.py", sys.argv[2], "--dtype", sys.argv[3], "--format", "libtorch"]; runpy.run_module("mace.cli.create_lammps_model", run_name="__main__")'
venv/bin/python -c "$X" float64 mace-mpa-0-medium.model float64 && mv mace-mpa-0-medium.model-lammps.pt mpa0_f64-lammps.pt
venv/bin/python -c "$X" float32 mace-mpa-0-medium.model float32 && mv mace-mpa-0-medium.model-lammps.pt mpa0_f32-lammps.pt
venv/bin/python -c "$X" float64 mace-mp-0b3-medium.model float64 && mv mace-mp-0b3-medium.model-lammps.pt mp0b3_f64-lammps.pt
venv/bin/pip freeze > versions.txt; shasum -a 256 *.pt
```

The user moves the three `.pt` files and `versions.txt` to a cache directory outside the repository (`$PUREMLIP_LAMMPS_MODEL_DIR`), then deletes the folder. The Float32 export uses torch's float32 default.
- [ ] **Step 5: Write `in.mace` and `run_lammps.jl`, then generate the JSON.** Input: `units metal`, `atom_style atomic`, `atom_modify map yes`, `newton on`, `boundary p p p`, `read_data`, `mass * 1.0`, `pair_style mace no_domain_decomposition`, `pair_coeff * * ${model} <symbols>`, `compute pe_atom all pe/atom`, `compute pv all pressure NULL virial`, dump of `id fx fy fz c_pe_atom` sorted by id with `%.17g`, energy and `c_pv[1..6]` and `vol` printed with `%.17g`, `run 0`. `run_lammps.jl` takes `$PUREMLIP_LAMMPS` (the `lmp` binary) and `$PUREMLIP_LAMMPS_MODEL_DIR`, runs with `OMP_NUM_THREADS=1`, writes the three JSON files with the metadata above (versions from `versions.txt`, `lmp -h` and the pins; `machine` from `sysctl -n machdep.cpu.brand_string`). Run: `julia --project=PureMLIP/test PureMLIP/test/reference/run_lammps.jl`.
- [ ] **Step 6:** Vendor `mace_radial_reference_f64.json` (verify SHA-256 `0f81adb3...e2e7b`); write the `lammps-mace/NOTICE` (how the values were produced, the pins, that model files are not included). Run item `LAMMPS reference`; expected PASS. If `isolated_cu` fails, LAMMPS's energy excludes E0: stop and report (Tasks 6-7 compare against it).
- [ ] **Step 7: Commit** `Add LAMMPS-MACE reference data and structures` (body: pins, versions, how to regenerate; no paths).

### Task 1: Pickled constants in the checkpoint reader

**Files:**
- Create: `PureMLIP/src/checkpoint/constants.jl`
- Modify: `PureMLIP/src/checkpoint/Checkpoint.jl`, `load.jl` (`MACECheckpoint` gains `constants`), `PureMLIP/src/PureMLIP.jl` (re-export/public), `PureMLIP/test/testutils.jl` (`tiny_mace_zip` gains `use_sc` on its product), `PureMLIP/test/checkpoint_tests.jl`

**Interfaces:**
- Consumes: `PickleTree`, `module_class`, `module_state`, `dict_get`, `dict_pairs`, `irreps_value` (Checkpoint internals).
- Produces (exported: `MACEConstants`; public: the rest):

```julia
struct Normalize2Mom; cst::Float64; f::String; end                    # f: the pickled global, e.g. "torch.nn.functional.silu"
struct LinearInstruction; i_in::Int; i_out::Int; path_shape::Vector{Int}; path_weight::Float64; end   # 0-based indices as pickled
struct LinearSpec; irreps_in::Vector{IrrepMul}; irreps_out::Vector{IrrepMul}; instructions::Vector{LinearInstruction}; end
struct TPInstruction; i_in1::Int; i_in2::Int; i_out::Int; connection_mode::String; has_weight::Bool; path_weight::Float64; path_shape::Vector{Int}; end
struct TPSpec; irreps_in1::Vector{IrrepMul}; irreps_in2::Vector{IrrepMul}; irreps_out::Vector{IrrepMul}; instructions::Vector{TPInstruction}; end
struct MACEConstants
    normalize2mom::Dict{String, Normalize2Mom}   # module path => constant, every normalize2mom module in the tree
    linear::Dict{String, LinearSpec}             # every e3nn Linear
    tensor_product::Dict{String, TPSpec}         # every TensorProduct and FullyConnectedTensorProduct
    pair_repulsion::Bool                         # root attribute; false when absent (MACE's own hasattr rule)
    distance_transform::String                   # class of radial_embedding.distance_transform; "" when absent
    product_use_sc::Vector{Bool}                 # one per product block; missing attribute is an error
end
MACECheckpoint(hyperparameters, state, constants)      # new third field
read_constants(t::PickleTree)::MACEConstants          # internal; called by load_mace_checkpoint
```

Module paths use `.` separators without a trailing dot (`"interactions.1.conv_tp"`). A module reached twice through the pickle memo (layers 1 and 2 share layer 0's `act`) is recorded under each path.

- [ ] **Step 1: Write the failing tests:**

```julia
@testitem "MPA-0 and MP-0b3 pickled constants" setup = [Checkpoints] begin
    s3, s5, s7 = 1.7320508075688772, 2.23606797749979, 2.6457513110645907
    a, b = 0.0625, 0.05103103630798288
    for (name, cst, lin1) in ((:mpa0, 1.679176792398942, [a, b, b, a]), (:mp0b3, 1.6791767923989418, [a, a, b, b, b, b, b, b, a, a]))
        k = load_mace_checkpoint(Checkpoints.fetch(name)).constants
        acts = [["interactions.$i.conv_tp_weights.layer$j.act" for i in 0:1 for j in 0:2]; "readouts.1.non_linearity.acts.0"]
        @test sort(collect(keys(k.normalize2mom))) == sort(acts)
        # Pickled floats pass through unchanged: exact comparison.
        @test all(k.normalize2mom[p].cst == cst && k.normalize2mom[p].f == "torch.nn.functional.silu" for p in acts)
        @test cst != 1.6765324703310911                     # the exact second-moment integral
        @test k.pair_repulsion && k.distance_transform == "mace.modules.radial.AgnesiTransform"
        @test k.product_use_sc == [false, true]
        lw(p) = [i.path_weight for i in k.linear[p].instructions]
        tw(p) = [i.path_weight for i in k.tensor_product[p].instructions]
        @test lw("node_embedding.linear") == [0.105999788000636]
        @test lw("interactions.0.linear") == fill(0.08838834764831845, 4)
        @test lw("interactions.1.linear_up") == fill(0.08838834764831845, 2)
        @test lw("interactions.1.linear") == lin1
        @test lw("readouts.1.linear_2") == [0.25]
        @test tw("interactions.0.conv_tp") == [1.0, s3, s5, s7]
        @test tw("interactions.1.conv_tp") == [1.0, 1.0, s3, s3, s3, s5, s5, s5, s7, s7]
        @test tw("interactions.0.skip_tp") == [0.009369146112448268, 0.01622783709029683, 0.020950047598562217, 0.024788430610565716]
        @test [(i.i_in1, i.i_in2, i.i_out) for i in k.tensor_product["interactions.1.conv_tp"].instructions] ==
            [(0, 0, 0), (1, 1, 1), (0, 1, 2), (1, 0, 3), (1, 2, 4), (0, 2, 5), (1, 1, 6), (1, 3, 7), (0, 3, 8), (1, 2, 9)]
        @test all(i -> i.connection_mode == "uvu", k.tensor_product["interactions.1.conv_tp"].instructions)
    end
    P = PureMLIP.Checkpoint.parse_irreps
    @test load_mace_checkpoint(Checkpoints.fetch(:mpa0)).constants.linear["interactions.1.linear"].irreps_in == P("256x0e+384x1o+384x2e+256x3o")
end

@testitem "constants of a skeleton checkpoint" setup = [ZipFixtures] begin
    k = load_mace_checkpoint(ZipFixtures.tiny_mace_zip()).constants
    @test isempty(k.normalize2mom) && isempty(k.linear) && !k.pair_repulsion && k.distance_transform == ""
    @test k.product_use_sc == [false]
end
```

- [ ] **Step 2:** Run item `constants`; expected FAIL.
- [ ] **Step 3:** Implement `read_constants` as one walk of the module tree from the root (the `_modules` dicts), dispatching on `module_class`; instruction tuples are `PK_CALL`s of the two `Instruction` globals (Linear: `(i_in, i_out, path_shape, path_weight)`; TP: `(i_in1, i_in2, i_out, connection_mode, has_weight, path_weight, path_shape)`). Malformed entries throw `checkpoint: <path>: malformed <what>`. Add `"use_sc" => pk_bool(false)` to the product in `ZipFixtures.tiny_mace_zip`.
- [ ] **Step 4:** Run the checkpoint items and the fuzz items (`fail cleanly`); expected PASS (fuzzed inputs reaching the new reader raise prefixed errors only).
- [ ] **Step 5:** Add a mutation to `mutations.toml` (e.g. record `path_weight` from the wrong tuple position; `expect = "pickled constants"`). Commit `Read normalize2mom constants and e3nn instructions`.

### Task 2: `PureMLIP.Model` and the periodic neighbor list

**Files:**
- Modify: none to `PureMLIP/Project.toml` (the neighbor list is an own cell list; CellListMap failed its gate)
- Create: `PureMLIP/src/model/Model.jl` (module, `using ..Checkpoint, ..Equivariant, LinearAlgebra`, includes, exports), `PureMLIP/src/model/neighbors.jl`
- Modify: `PureMLIP/src/PureMLIP.jl` (include, `using .Model`, re-export), `PureMLIP/test/layers_tests.jl` (graph, ExplicitImports loop, `test_calls` sweep of `Model`), `trim_tests.jl`, `strictmode_tests.jl`, `testutils.jl` (`@testmodule NeighborOracle`); Create: `PureMLIP/test/neighbor_tests.jl`

**Interfaces:**
- Produces (exported):

```julia
mutable struct NeighborList{T}   # fields concrete
    r_cut::Float64
    n_atoms::Int
    n_edges::Int
    senders::Vector{Int}            # length = capacity; first n_edges entries valid
    receivers::Vector{Int}
    shifts::Matrix{Int}             # 3 × capacity, integer image n_e: r_e = pos[:, receiver] - pos[:, sender] + cell * n_e
    ...                             # cell-grid scratch
end
neighbor_list(positions::AbstractMatrix{T}, cell::AbstractMatrix{T}, r_cut::Real; extra_capacity::Integer = 0) -> NeighborList{T}
neighbor_list!(nl::NeighborList{T}, positions::AbstractMatrix{T}, cell::AbstractMatrix{T}) -> nl
```

  `positions` is `3 × n` (one column per atom, Å), `cell` is `3 × 3` with **columns** the lattice vectors; all three directions periodic. Edges are every `(sender, receiver, n_e)` with `0 < ‖r_e‖ < r_cut`, sorted by receiver (counting sort into preallocated buffers; order within a receiver as CellListMap visits pairs, single-threaded, so deterministic). Shifts are computed relative to the positions as given, so unwrapped positions work. Capacity = edges found at construction + `extra_capacity`. `neighbor_list!` counts every edge before writing any, so the capacity error reports the full count. Errors: `ArgumentError(lazy"neighbor list: $m edges exceed the capacity $c")`; `ArgumentError(lazy"neighbor list: atoms $i and $j coincide")` for a zero-length edge; `DimensionMismatch` when `size(positions, 2) != nl.n_atoms`; `ArgumentError("neighbor list: positions and cell must be finite")`; `ArgumentError("neighbor list: the cell is singular")`. Entry points declare `Base.require_one_based_indexing(positions, cell)`.
- Test oracle `NeighborOracle.brute_force(positions, cell, r_cut)::Set{Tuple{Int,Int,NTuple{3,Int}}}`: every image `n` with `|n_k| ≤ ceil(r_cut / h_k)`, `h_k` the cell height along `k` (volume over the area of the other two vectors).

- [ ] **Step 1: CellListMap gate (stop point).** In a scratch session, before writing the module: (a) TrimCheck `@validate` on a minimal wrapper that updates a CellListMap periodic system and visits all pairs for `Matrix{Float64}` positions and a triclinic cell; (b) `@test_noalloc` on the update-and-visit call after warm-up; (c) its pair visits on `triclinic_ni3al` (cutoff 6.0 > half the cell width) reproduce the 176 brute-force edges. If any of the three fails, stop and report the output to the user; do not substitute another neighbor list without their decision.
- [ ] **Step 2: Write the failing tests:**

```julia
@testitem "neighbor list matches brute force on the reference structures" setup = [LammpsReference, NeighborOracle] begin
    ref = LammpsReference.load(:mpa0_f64)
    for (name, count) in (("cu_vacancy", 2340), ("crconi_alloy", 2496), ("triclinic_ni3al", 176), ("multi_element", 44), ("isolated_cu", 0))
        s = LammpsReference.structure(ref, name)
        nl = neighbor_list(s.positions, s.cell, 6.0)
        got = Set((nl.senders[e], nl.receivers[e], Tuple(nl.shifts[:, e])) for e in 1:nl.n_edges)
        @test nl.n_edges == count && length(got) == count
        @test got == NeighborOracle.brute_force(s.positions, s.cell, 6.0)
        @test issorted(view(nl.receivers, 1:nl.n_edges))
    end
end

@testitem "neighbor list: unwrapped positions, refill, errors" setup = [LammpsReference] begin
    s = LammpsReference.structure(LammpsReference.load(:mpa0_f64), "triclinic_ni3al")
    vecs(nl, P) = sort([P[:, nl.receivers[e]] - P[:, nl.senders[e]] + s.cell * nl.shifts[:, e] for e in 1:nl.n_edges])
    P2 = copy(s.positions); P2[:, 3] .+= s.cell[:, 1] .- 2 .* s.cell[:, 2]
    # Rounding bound: |r| ≤ 6 Å built from coordinates ≤ 20 Å, 8 operations: 8·eps·20.
    @test all(isapprox.(vecs(neighbor_list(P2, s.cell, 6.0), P2), vecs(neighbor_list(s.positions, s.cell, 6.0), s.positions); atol = 160 * eps()))
    nl = neighbor_list(s.positions, 1.2 .* s.cell, 6.0)
    @test_throws "neighbor list: 176 edges exceed the capacity $(nl.n_edges)" neighbor_list!(nl, s.positions, s.cell)
    @test_throws "neighbor list: atoms 1 and 2 coincide" neighbor_list([1.0 1.0; 2.0 2.0; 3.0 3.0], 10.0 * [1.0 0 0; 0 1 0; 0 0 1], 6.0)
    @test_throws DimensionMismatch neighbor_list!(neighbor_list(s.positions, s.cell, 6.0), s.positions[:, 1:3], s.cell)
    using OffsetArrays
    @test_throws ArgumentError neighbor_list(OffsetArray(s.positions, 0, -1), s.cell, 6.0)
    v = neighbor_list(view(hcat(s.positions, zeros(3)), :, 1:4), s.cell, 6.0)
    @test v.n_edges == 176
end
```

  Extend the gates: `test_layers(PureMLIP => (Checkpoint, Equivariant, Model), Model => (Checkpoint, Equivariant), Checkpoint => (), Equivariant => ())`; add `PureMLIP.Model` to the ExplicitImports loop and to the `test_calls` loop (after calling `neighbor_list`/`neighbor_list!`); TrimCheck and `test_signatures` gain `neighbor_list(Matrix{Float64}, Matrix{Float64}, Float64)` and `neighbor_list!(NL, Matrix{Float64}, Matrix{Float64})` with `NL = typeof(neighbor_list(...))` computed in the `init` block / item; `@test_noalloc neighbor_list!` through a `@noinline` barrier after warm-up.
- [ ] **Step 3:** Run items `neighbor list` and `layer graph`; expected FAIL.
- [ ] **Step 4:** Implement.
- [ ] **Step 5:** Run items `neighbor list`, `layer graph`, `submodules read only`, `test_calls`, `Aqua`, `TrimCheck`, `StrictMode`; expected PASS.
- [ ] **Step 6:** Mutation for `neighbors.jl` (e.g. drop self-image edges; `expect = "brute force"`). Commit `Add PureMLIP.Model with a periodic neighbor list`.

### Task 3: Radial embedding and ZBL with their derivatives

**Files:**
- Create: `PureMLIP/src/model/radial.jl`, `PureMLIP/test/radial_tests.jl`
- Modify: `PureMLIP/test/testutils.jl` (`E3nnFixtures.radial()` loads the radial fixture)

**Interfaces** (internal to `Model`; scalar functions generic in `S <: Real` for the argument and `T` for parameters; each returns the value and its derivative with respect to `d`, which is the layer's pullback for a scalar input):

```julia
polynomial_cutoff(d::S, r_max, p::Int) -> (f::S, df::S)
agnesi_transform(d::S, r0, a, p, q) -> (dt::S, ddt::S)
bessel_basis!(B::AbstractVector{S}, dB::AbstractVector{S}, x::S, weights::AbstractVector, prefactor) -> B
struct RadialEmbedding{T}; bessel_weights::Vector{T}; prefactor::T; r_max::T; cutoff_p::Int
                           agnesi_a::T; agnesi_p::T; agnesi_q::T; covalent_radii::Vector{T}; end   # radii indexed by Z + 1
radial_embedding!(feats::AbstractVector{S}, dfeats::AbstractVector{S}, emb::RadialEmbedding, d::S, Zu::Int, Zv::Int) -> feats
struct ZBLParams{T}; c::NTuple{4,T}; p::T; a_exp::T; a_prefactor::T; covalent_radii::Vector{T}; end
zbl_edge_energy(d::S, Zu::Int, Zv::Int, zbl::ZBLParams) -> (e::S, de::S)    # e = 0.5 · v_e · env_e, credited to the receiver
```

`dfeats[n] = d feats[n] / dd` (through Agnesi and the cutoff). The literals `0.529, 3.2, 0.9423, 0.4028, 0.2016, 14.3996, 0.5` live in `zbl_edge_energy` with a citation comment (MIT, mace `radial.py:203-216`).

- [ ] **Step 1: Write the failing tests:**

```julia
@testitem "radial functions match the MACE radial fixture" setup = [E3nnFixtures] begin
    M = PureMLIP.Model
    f = E3nnFixtures.radial()                       # parsed mace_radial_reference_f64.json
    Z = [1, 6, 8]; R = Dict(1 => 0.31, 6 => 0.76, 8 => 0.66)   # MACE covalent radii (Cordero 2008)
    snd = f["graph"]["edge_index"][1] .+ 1; rcv = f["graph"]["edge_index"][2] .+ 1
    d = [x[1] for x in f["graph"]["distances"]]
    bp = f["bessel"]["parameters"]; nb = bp["num_basis"]
    for e in eachindex(d)
        B = zeros(nb); dB = zeros(nb)
        M.bessel_basis!(B, dB, d[e], Float64.(bp["bessel_weights"]), bp["prefactor"])
        fc, dfc = M.polynomial_cutoff(d[e], bp["r_max"], bp["cutoff_p"])
        # Oracle tolerance: e3nn-core-rs fixtures are stated to 1e-10.
        @test B ≈ Float64.(f["bessel"]["basis"][e]) atol = 1.0e-10
        @test fc ≈ f["bessel"]["cutoff"][e][1] atol = 1.0e-10
        @test B .* fc ≈ Float64.(f["bessel"]["embedding"][e]) atol = 1.0e-10
        @test sum(Float64.(f["bessel"]["cotangent_embedding"][e]) .* (dB .* fc .+ B .* dfc)) ≈ f["bessel"]["vjp_distances"][e][1] atol = 1.0e-10
        ap = f["agnesi"]["parameters"]
        t, dt = M.agnesi_transform(d[e], 0.5 * (R[Z[snd[e]]] + R[Z[rcv[e]]]), ap["a"], ap["p"], ap["q"])
        @test t ≈ f["agnesi"]["output"][e][1] atol = 1.0e-10
        @test f["agnesi"]["cotangent"][e][1] * dt ≈ f["agnesi"]["vjp_distances"][e][1] atol = 1.0e-10
    end
    zp = f["zbl"]["parameters"]
    radii = zeros(119); for (z, r) in R; radii[z + 1] = r; end
    zbl = M.ZBLParams{Float64}(Tuple(Float64.(zp["c"])), Float64(zp["p"]), zp["a_exp"], zp["a_prefactor"], radii)
    V = zeros(3); vjp = zeros(length(d))
    for e in eachindex(d)
        v, dv = M.zbl_edge_energy(d[e], Z[snd[e]], Z[rcv[e]], zbl)
        V[rcv[e]] += v; vjp[e] = f["zbl"]["cotangent_node_energy"][rcv[e]][1] * dv
    end
    @test V ≈ Float64.(f["zbl"]["output_node_energy"]) atol = 1.0e-10
    @test vjp ≈ [x[1] for x in f["zbl"]["vjp_distances"]] atol = 1.0e-10
end

@testitem "radial derivatives match ForwardDiff and finite differences" begin
    using ForwardDiff
    M = PureMLIP.Model
    radii = fill(0.2, 119); radii[30] = 1.32; radii[14] = 1.21       # Cu, Al
    emb = M.RadialEmbedding{Float64}(pi / 6.0 .* (1:10), sqrt(2 / 6), 6.0, 5, 1.0805, 4.5791, 0.9183, radii)
    for d in 0.6:0.37:5.95
        F = zeros(10); dF = zeros(10)
        M.radial_embedding!(F, dF, emb, d, 29, 13)
        J = ForwardDiff.derivative(x -> M.radial_embedding!(zeros(typeof(x), 10), zeros(typeof(x), 10), emb, x, 29, 13), d)
        # Rounding bound: 64·eps·max|F'| (chain of three transcendental factors).
        @test dF ≈ J atol = 64 * eps() * max(1.0, maximum(abs, J))
    end
    F = zeros(10); dF = zeros(10)
    @test M.radial_embedding!(F, dF, emb, 6.0, 29, 29) == zeros(10) && dF == zeros(10)      # at r_max: exactly zero
end
```

  Add one more item with a central finite difference (`h = 1e-6`) on `zbl_edge_energy` for Cu–Cu at 2.5 Å and H–C at 1.0 Å, tolerance a measured margin (written in the comment).
- [ ] **Step 2:** Run item `radial`; expected FAIL.
- [ ] **Step 3:** Implement. Vendor no new fixture code beyond `E3nnFixtures.radial() = load("mace_radial_reference_f64")`.
- [ ] **Step 4:** Run item `radial`; expected PASS.
- [ ] **Step 5:** Mutations for `radial.jl` (Agnesi exponent `q − p` → `p − q`; ZBL `0.5` → `1.0`). Commit `Add MACE radial embedding and ZBL with derivatives`.

### Task 4: e3nn layers with pullbacks

**Files:**
- Create: `PureMLIP/src/model/linear.jl` (`E3nnLinear`, `SpeciesLinear`), `PureMLIP/src/model/tensor_product.jl` (`ChannelwiseTP`), `PureMLIP/src/model/radial_mlp.jl` (`RadialMLP`, density)
- Create: `PureMLIP/test/layers_model_tests.jl`

**Interfaces** (internal; activations of any `S <: Real`, weights `T`). Node features are `Array{S,3}` of size `(C, K, n)`: channel fastest, then e3nn component `k` (block `l` occupies `l² + m + 1`, `m = 0..2l`, for the 16-component `0..3` layout; in general consecutive blocks in slot order), then node. Pullbacks named `*_pullback!` overwrite their first argument; functions named `*_add!` accumulate.

```julia
struct E3nnLinear{T}   # one weight block per instruction
E3nnLinear{T}(weight::AbstractVector{<:Real}, in_ls::Vector{Int}, out_ls::Vector{Int},
              instructions::Vector{Tuple{Int,Int,Float64}}, C_in::Int, C_out::Int)
    # instructions: (input slot, output slot, path weight), 1-based, in weight order; block p is the
    # torch [C_in, C_out] row-major segment p of `weight`; input slots without an instruction are ignored
linear!(y::AbstractArray{S,3}, lin::E3nnLinear, x::AbstractArray{S,3}) -> y
linear_pullback!(dx::AbstractArray{S,3}, lin::E3nnLinear, dy::AbstractArray{S,3}) -> dx

struct SpeciesLinear{T}   # FullyConnectedTensorProduct with one-hot species: skip_tp
SpeciesLinear{T}(weight::AbstractVector{<:Real}, ls::Vector{Int}, instructions::Vector{Tuple{Int,Int,Float64}},
                 C_in::Int, n_species::Int, C_out::Int)
    # block p is the torch [C_in, n_species, C_out] row-major segment p; field c::Vector{T}, one per
    # instruction, c[p] = pw * (1 / sqrt(2l+1)) (bit-exact with the embedded TorchScript literals)
species_linear!(y, sl::SpeciesLinear, x, species::AbstractVector{Int}) -> y
species_linear_pullback!(dx, sl::SpeciesLinear, dy, species) -> dx

struct ChannelwiseTP{T}   # uvu tensor product, one weight per channel per instruction
ChannelwiseTP{T}(in1_ls::Vector{Int}, in2_ls::Vector{Int}, out_ls::Vector{Int},
                 instructions::Vector{Tuple{Int,Int,Int,Float64}})   # (in1 slot, in2 slot, out slot, path weight), 1-based
    # coupling tensor per instruction: path weight · e3nn_wigner_3j(l1, l2, l3)
tp_edge_add!(M::AbstractMatrix{S}, tp::ChannelwiseTP, x::AbstractMatrix{S}, Y::AbstractVector{S}, w::AbstractMatrix{S}) -> M
    # M (C, K_out) += message; x (C, K_in1) sender features; Y (K_in2); w (C, n_instructions)
tp_edge_pullback_add!(dx::AbstractMatrix{S}, dY::AbstractVector{S}, dw::AbstractMatrix{S},
                      tp::ChannelwiseTP, dM::AbstractMatrix{S}, x, Y, w) -> nothing   # accumulates all three

struct RadialMLP{T}; W::NTuple{4, Matrix{T}}; cst::T; end   # W[k] = torch layer k / sqrt(h_in), as Julia [h_in, h_out]
RadialMLP{T}(layers::NTuple{4, AbstractMatrix{<:Real}}, cst::Real)
radial_mlp!(out::AbstractMatrix{S}, pre::AbstractArray{S,3}, act::AbstractArray{S,3}, mlp::RadialMLP, feats::AbstractMatrix{S}) -> out
    # feats (n_basis, nb) → out (W_out, nb); pre/act (hidden, nb, 3) keep the hidden pre-activations and activations
radial_mlp_pullback!(dfeats::AbstractMatrix{S}, g::AbstractArray{S,3}, mlp::RadialMLP, dout::AbstractMatrix{S}, pre, act) -> dfeats
    # g (hidden, nb, 2) scratch
density_edge(feats::AbstractVector{S}, w::AbstractVector) -> (ρ::S, c::S)   # ρ = tanh(z²), z = feats·w/√n_basis; ∂ρ/∂feats = c .* w
normalized_silu(z::S, cst) -> (a::S, da::S)   # a = cst·z·sigmoid(z), da = ∂a/∂z; used by RadialMLP and the nonlinear readout
```

- [ ] **Step 1: Write the failing tests:**

```julia
@testitem "channelwise TP matches the e3nn-core-rs tensor-product fixtures" setup = [E3nnFixtures] begin
    M = PureMLIP.Model
    for name in ("fully_connected_tensor_product_smoke_f64", "fully_connected_tensor_product_lmax4_smoke_f64")
        d = E3nnFixtures.load(name)
        l1, l2, l3 = (E3nnFixtures.segment_degrees(d[k]) for k in ("irreps_in1", "irreps_in2", "irreps_out"))
        ins = [(d["instruction_in1_segment"][p] + 1, d["instruction_in2_segment"][p] + 1, d["instruction_out_segment"][p] + 1,
                Float64(d["instruction_path_weight"][p])) for p in eachindex(d["instruction_path_weight"])]
        tp = M.ChannelwiseTP{Float64}(l1, l2, l3, ins)
        for s in eachindex(d["x"])
            x = reshape(Float64.(d["x"][s]), 1, :); y = Float64.(d["y"][s]); w = reshape(Float64.(d["w"][s]), 1, :)
            out = zeros(1, length(d["out"][s]))
            M.tp_edge_add!(out, tp, x, y, w)
            # Oracle tolerance: e3nn-core-rs fixtures are stated to 1e-10.
            @test vec(out) ≈ d["out"][s] atol = 1.0e-10
            dx = zero(x); dy = zero(y); dw = zero(w)
            M.tp_edge_pullback_add!(dx, dy, dw, tp, reshape(Float64.(d["grad_out"][s]), 1, :), x, y, w)
            @test vec(dx) ≈ d["grad_x"][s] atol = 1.0e-10
            @test dy ≈ d["grad_y"][s] atol = 1.0e-10
            @test vec(dw) ≈ d["grad_w"][s] atol = 1.0e-10
        end
    end
end

@testitem "MACE layers match the forward-pass formulas" setup = [Checkpoints] begin
    M = PureMLIP.Model
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); st = ck.state.f64; k = ck.constants
    # Folded skip_tp constants, bit-exact against the embedded TorchScript literals.
    sk = M.SpeciesLinear{Float64}(st["interactions.0.skip_tp.weight"].data, [0, 1, 2, 3],
        [(p, p, k.tensor_product["interactions.0.skip_tp"].instructions[p].path_weight) for p in 1:4], 128, 89, 128)
    @test sk.c == [0.0093691461124482678, 0.0093691461124482695, 0.0093691461124482678, 0.009369146112448266]
    using Random
    x = randn(Xoshiro(3), 128, 16, 3)
    up = M.E3nnLinear{Float64}(st["interactions.0.linear_up.weight"].data, [0], [0], [(1, 1, 0.08838834764831845)], 128, 128)
    y = zeros(128, 1, 3); M.linear!(y, up, view(x, :, 1:1, :))
    W = st["interactions.0.linear_up.weight"].data
    naive = [0.08838834764831845 * sum(x[u, 1, i] * W[(u - 1) * 128 + w] for u in 1:128) for w in 1:128, i in 1:3]
    # Rounding bound: 4·eps·N·max|terms|, N = 128 terms per output.
    @test y[:, 1, :] ≈ naive atol = 4 * eps() * 128 * maximum(abs, x) * maximum(abs, W)
end
```

  The same item continues with one naive-loop assertion per remaining layer, each written from the forward-pass formulas with torch flat indices: `interactions.0.linear` (section 7, four blocks), `interactions.0.skip_tp` and `interactions.1.skip_tp` (sections 7-8, `[u, s, w]` at `((u-1)·89 + s-1)·128 + w`), `interactions.0.conv_tp_weights` (section 6, `cst = 1.679176792398942`), `interactions.0.density_fn` (section 6), `readouts.1.linear_1` (section 10); same tolerance form with `N` the contraction length. Add an item **"layer pullbacks match ForwardDiff"**: for each layer `f`, random `x`, random cotangent `dy`, `@test pullback(dy) ≈ ForwardDiff.gradient(x -> sum(dy .* f(x)), x)` with a rounding-bound tolerance, plus one central finite difference per layer (measured margin). Add an item **"layers accept views"** running `linear!` and `species_linear!` on `view`s of larger arrays.
- [ ] **Step 2:** Run items `channelwise TP`, `MACE layers`, `layer pullbacks`; expected FAIL.
- [ ] **Step 3:** Implement. `linear!` and the radial MLP use `mul!` on the `(C, k, :)` slices (unit stride in the first dimension); the TP kernel is a plain loop over instructions, nonzero coupling entries and channels (channel innermost).
- [ ] **Step 4:** Run the items; expected PASS.
- [ ] **Step 5:** Mutations for the three files (e.g. `1 / sqrt(2l+1)` → `1 / sqrt(2l)` in `SpeciesLinear`; swap `l1`/`l2` in the coupling lookup; drop `cst` on the last hidden layer). Commit `Add e3nn linear, tensor-product and radial-MLP layers`.

### Task 5: Symmetric contraction

**Files:**
- Create: `PureMLIP/src/model/symmetric_contraction.jl`, `PureMLIP/test/symmetric_contraction_tests.jl`

**Interfaces** (internal):

```julia
struct Contraction{T}   # one output irrep: l = 0 (1 component) or l = 1 (3 components)
Contraction{T}(l::Int, U1::AbstractArray{<:Real}, U2::AbstractArray{<:Real}, U3::AbstractArray{<:Real},
               Wmax::AbstractArray{<:Real,3}, W0::AbstractArray{<:Real,3}, W1::AbstractArray{<:Real,3})
    # U as torch_array of the stored buffers (leading m index when l = 1); W as torch_array [n_species, K, C];
    # U is stored as coordinate lists of its nonzero entries
struct SymmetricContraction{T}; contractions::Vector{Contraction{T}}; end
symmetric_contraction!(B::AbstractArray{S,3}, sc::SymmetricContraction, X::AbstractArray{S,3},
                       species::AbstractVector{Int}, scratch::AbstractArray{S}) -> B
    # X (C, 16, n); B (C, K_out, n) with K_out = 1 + 3 for [0e, 1o], 1 for [0e]
symmetric_contraction_pullback!(dX::AbstractArray{S,3}, sc::SymmetricContraction, dB::AbstractArray{S,3},
                                X::AbstractArray{S,3}, species::AbstractVector{Int}, scratch::AbstractArray{S}) -> dX
symmetric_contraction_scratch_size(sc::SymmetricContraction) -> Int   # elements of `scratch` (per-node C2/C1 buffers)
symmetric_contraction_from_state(st::Dict{String, CheckpointTensor{Float64}}, prefix::String, ::Type{T}) -> SymmetricContraction{T}
    # contractions of `prefix.symmetric_contractions.contractions.*` in order, pairing weights_max↔U3, weights.0↔U2, weights.1↔U1
```

Algorithm: per node, per contraction, for each nonzero `(m, w, x, i, k, u)` of U3 and each channel `c`, accumulate `C2[c, m, w, x]`; then the U2, `A2`, U1 and `B` steps of section 9, channel innermost. The pullback recomputes `C2` and `C1` per node.

- [ ] **Step 1: Write the failing tests:**

```julia
@testitem "symmetric contraction matches the dense formula" setup = [Checkpoints] begin
    using Random
    M = PureMLIP.Model
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); st = ck.state.f64
    sc = M.symmetric_contraction_from_state(st, "products.0", Float64)
    @test [[length(c.U1), length(c.U2), length(c.U3)] for c in sc.contractions] == [[1, 16, 353], [3, 70, 2846]]
    X = 0.3 .* randn(Xoshiro(5), 128, 16, 3); species = [29, 1, 63]                    # z-table indices
    B = zeros(128, 4, 3); M.symmetric_contraction!(B, sc, X, species, zeros(M.symmetric_contraction_scratch_size(sc)))
    p = "products.0.symmetric_contractions.contractions.0."
    U1, U2, U3 = torch_array(st[p * "U_matrix_1"], Val(2)), torch_array(st[p * "U_matrix_2"], Val(3)), torch_array(st[p * "U_matrix_3"], Val(4))
    Wm, W0, W1 = (torch_array(st[p * k], Val(3)) for k in ("weights_max", "weights.0", "weights.1"))
    for a in 1:3, c in (1, 64, 128)
        s = species[a]; x = X[c, :, a]
        C2 = [sum(U2[w, y, k] * W0[s, k, c] for k in axes(U2, 3)) +
              sum(U3[w, y, i, k] * Wm[s, k, c] * x[i] for i in 1:16, k in axes(U3, 4)) for w in 1:16, y in 1:16]
        C1 = [sum(U1[w, k] * W1[s, k, c] for k in axes(U1, 2)) + sum(C2[w, y] * x[y] for y in 1:16) for w in 1:16]
        # Rounding bound: 4·eps·N relative, N = 16·16·23 terms per output.
        @test B[c, 1, a] ≈ sum(C1[w] * x[w] for w in 1:16) rtol = 4 * eps() * 16 * 16 * 23
    end
    # The same loops with the leading m index for contraction 1 check B[c, 2:4, a].
end

@testitem "symmetric contraction is equivariant" setup = [Checkpoints] begin
    # Rotate each l-block of X with D_e3nn(l, R) = e3nn_basis_change(l) * wigner_D(l, R) * e3nn_basis_change(l)'
    # (R = rotation by 0.7 rad about z then 1.1 rad about x). Assertions: B0(RX) ≈ B0(X) and
    # B1(RX)[c, :] ≈ D_e3nn(1, R) * B1(X)[c, :] for every channel; tolerance a measured margin.
end

@testitem "symmetric contraction pullback matches ForwardDiff and finite differences" setup = [Checkpoints] begin
    # dX from symmetric_contraction_pullback! ≈ ForwardDiff.gradient(X -> sum(dB .* B(X)), X) for 2 nodes, C = 128
    # (rounding bound); one central difference (h = 1e-6) on X[5, 7, 1] (measured margin).
end
```
  (`torch_array` gives 1-based arrays in torch index order, so the section-9 formulas apply with every index shifted by one.)
- [ ] **Step 2:** Run item `symmetric contraction`; expected FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run; expected PASS.
- [ ] **Step 5:** Mutation (contract the first ell index instead of the last in `A2`; `expect = "symmetric contraction"`). Commit `Add the symmetric contraction with its pullback`.

### Task 6: The model and its energy

**Files:**
- Create: `PureMLIP/src/model/model.jl` (`MACEModel` from a checkpoint, architecture validation), `PureMLIP/src/model/evaluate.jl` (`MACEWorkspace`, `MACEResult`, `mace_energy!`)
- Create: `PureMLIP/test/model_tests.jl`; Modify: `trim_tests.jl`, `strictmode_tests.jl`, `layers_tests.jl` (`test_calls` calls)

**Interfaces:**
- Consumes: Tasks 1-5; `load_mace_checkpoint`, `torch_array`, `MACEConstants`, `e3nn_spherical_harmonics!`, `e3nn_wigner_3j`.
- Produces (exported):

```julia
struct MACEModel{T}   # concrete fields; public fields: z_table::Vector{Int}, r_max::T, e0::Vector{T}, scale::T, shift::T
MACEModel{T}(ck::MACECheckpoint) where {T <: AbstractFloat}
MACEModel(ck::MACECheckpoint) = MACEModel{Float64}(ck)
mutable struct MACEResult{T}; energy::T; node_energies::Vector{T}; forces::Matrix{T}; virial::Matrix{T}; stress::Matrix{T}; end
MACEResult{T}(n_atoms::Integer)
struct MACEWorkspace{T}   # every per-node and per-edge buffer of the forward and backward pass
MACEWorkspace{T}(model::MACEModel{T}, n_atoms::Integer, n_edges::Integer; edge_chunk::Integer = 4096)
mace_energy!(res::MACEResult{T}, ws::MACEWorkspace{T}, model::MACEModel{T}, positions::AbstractMatrix{T},
             cell::AbstractMatrix{T}, atomic_numbers::AbstractVector{<:Integer}, nl::NeighborList) -> res
    # fills res.energy and res.node_energies (forward pass, sections 1-11)
```

  Validation in `MACEModel{T}(ck)`, each failure `ArgumentError(lazy"model: unsupported checkpoint: $what $found, expected $expected")` (for example `model: unsupported checkpoint: correlations [2, 3], expected [3, 3]`): two interactions of classes `RealAgnosticDensityInteractionBlock`, `RealAgnosticDensityResidualInteractionBlock`; correlations `[3, 3]`; readouts `LinearReadoutBlock`, `NonLinearReadoutBlock`; one head; hidden irreps `Cx0e + Cx1o`; edge attributes `1x0e+1x1o+1x2e+1x3o`; `pair_repulsion`; distance transform `AgnesiTransform`; every `normalize2mom.f == "torch.nn.functional.silu"` and all `cst` equal within the model; `product_use_sc == [false, true]`; `interactions.1.linear` either 10 instructions of `(C, C)` or the 4-instruction simplified form, expanded to 10 slots with each instruction's path weight. Errors in `mace_energy!`: `ArgumentError(lazy"model: element $z is not in the model's element table")`; `ArgumentError(lazy"model: the neighbor list cutoff $rc is below r_max $rm")`; `ArgumentError(lazy"model: the workspace holds $cap edges, the neighbor list has $m")` and the same form for atoms; `DimensionMismatch` for size mismatches; `Base.require_one_based_indexing(positions, cell, atomic_numbers)`.

- [ ] **Step 1: Write the failing tests:**

```julia
@testitem "energies match LAMMPS-MACE" setup = [LammpsReference, Checkpoints] begin
    for (ref, src) in ((:mpa0_f64, :mpa0), (:mp0b3_f64, :mp0b3))
        model = MACEModel{Float64}(load_mace_checkpoint(Checkpoints.fetch(src)))
        for s in LammpsReference.structures(LammpsReference.load(ref))
            nl = neighbor_list(s.positions, s.cell, model.r_max)
            ws = MACEWorkspace{Float64}(model, size(s.positions, 2), nl.n_edges)
            res = mace_energy!(MACEResult{Float64}(size(s.positions, 2)), ws, model, s.positions, s.cell, s.atomic_numbers, nl)
            # Target from the design (section 9): 1e-6 eV per structure against LAMMPS-MACE.
            @test abs(res.energy - s.energy) <= 1.0e-6
            @test maximum(abs, res.node_energies - s.node_energies) <= 1.0e-6
        end
    end
end

@testitem "MPA-0 model contents and Float32" setup = [LammpsReference, Checkpoints] begin
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); m = MACEModel(ck)
    @test m.scale == 0.7735790334431056 && m.shift == 0.0 && m.r_max == 6.0 && length(m.z_table) == 89
    @test MACEModel(load_mace_checkpoint(Checkpoints.fetch(:mp0b3))).scale == 0.8140724313304148
    s = LammpsReference.structure(LammpsReference.load(:mpa0_f64), "isolated_cu")
    nl = neighbor_list(s.positions, s.cell, 6.0)
    r = mace_energy!(MACEResult{Float64}(1), MACEWorkspace{Float64}(m, 1, 0), m, s.positions, s.cell, s.atomic_numbers, nl)
    @test r.energy == m.e0[findfirst(==(29), m.z_table)]   # no edges: E0 exactly
    m32 = MACEModel{Float32}(ck)
    for s in LammpsReference.structures(LammpsReference.load(:mpa0_f32))
        P, C = Float32.(s.positions), Float32.(s.cell)
        nl = neighbor_list(P, C, 6.0)
        r = mace_energy!(MACEResult{Float32}(size(P, 2)), MACEWorkspace{Float32}(m32, size(P, 2), nl.n_edges), m32, P, C, s.atomic_numbers, nl)
        # Measured margin against the Float32 export: tol32 = 10-20 × the largest difference, recorded here.
        @test abs(r.energy - s.energy) <= tol32
    end
end

@testitem "model errors" setup = [Checkpoints] begin
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); m = MACEModel(ck); h = ck.hyperparameters
    bad = MACECheckpoint(MACEHyperparameters(h.model_class, h.r_max, h.num_interactions, h.atomic_numbers, h.heads,
        h.interactions, [2, 3], h.readouts), ck.state, ck.constants)
    @test_throws "model: unsupported checkpoint: correlations [2, 3], expected [3, 3]" MACEModel(bad)
    P = [0.0 2.5; 0.0 0.0; 0.0 0.0]; cell = [10.0 0 0; 0 10.0 0; 0 0 10.0]
    nl = neighbor_list(P, cell, 6.0); ws = MACEWorkspace{Float64}(m, 2, nl.n_edges)
    @test_throws "model: element 84 is not in the model's element table" mace_energy!(MACEResult{Float64}(2), ws, m, P, cell, [29, 84], nl)
    @test_throws "model: the workspace holds 1 edges" mace_energy!(MACEResult{Float64}(2), MACEWorkspace{Float64}(m, 2, 1), m, P, cell, [29, 29], nl)
end
```

  Also: an item running `mace_energy!` on `view` inputs (same energy, `==`) and asserting an `OffsetArray` position matrix throws `ArgumentError`; an item comparing the model's `interactions.1.linear` for MPA-0 with the section-8 formula (`0.0625` / `0.05103103630798288` literals, rounding bound). TrimCheck and `test_signatures` gain `MACEModel{Float64}(MACECheckpoint)`, `MACEModel{Float32}(MACECheckpoint)`, `MACEResult{Float64}(Int)`, `MACEWorkspace{Float64}(MACEModel{Float64}, Int, Int)`, `mace_energy!(MACEResult{Float64}, MACEWorkspace{Float64}, MACEModel{Float64}, Matrix{Float64}, Matrix{Float64}, Vector{Int}, NL)`; `@test_noalloc mace_energy!` after warm-up.
- [ ] **Step 2:** Run item `energies match`; expected FAIL.
- [ ] **Step 3:** Implement. Edges are processed in chunks of `edge_chunk` (radial MLP and TP weights live only for the chunk); per-edge arrays held for all edges: `r_e`, `d_e`, `Y` (16), `edge_feats` (10).
- [ ] **Step 4:** Run items `energies match`, `model`; expected PASS. An energy outside 1e-6 eV is investigated layer by layer (never by raising the tolerance); report the largest difference per structure in the commit body.
- [ ] **Step 5:** Mutations: `model.jl` — `scale` applied to E0 as well (`expect = "energies match"`); `evaluate.jl` — the `+ 1` of the density normalization dropped (`expect = "energies match"`). Commit `Evaluate MACE energies on CPU`.

### Task 7: Forces and stress

**Files:**
- Modify: `PureMLIP/src/model/evaluate.jl`; Create: `PureMLIP/test/forces_tests.jl`; Modify: `trim_tests.jl`, `strictmode_tests.jl`

**Interfaces:**
- Produces (exported):

```julia
mace_energy_forces!(res::MACEResult{T}, ws::MACEWorkspace{T}, model::MACEModel{T}, positions::AbstractMatrix{T},
                    cell::AbstractMatrix{T}, atomic_numbers::AbstractVector{<:Integer}, nl::NeighborList) -> res
    # forward pass, then the reverse sweep through every layer pullback; fills energy, node_energies,
    # forces (3 × n, eV/Å), virial (−∂E/∂ε, eV), stress (∂E/∂ε / |det cell|, eV/Å³, positive tensile)
mace_energy_forces(model::MACEModel{T}, positions::AbstractMatrix{T}, cell::AbstractMatrix{T},
                   atomic_numbers::AbstractVector{<:Integer}) -> MACEResult{T}      # allocates list, workspace, result
```

  Reverse sweep: seeds `∂E/∂e1_i = ∂E/∂e2_i = ∂E/∂V_i = scale`; node adjoints back through readouts, products, interaction 1, interaction 0; per edge chunk the MLP is recomputed, `tp_edge_pullback_add!` gives `∂E/∂tp_w`, `∂E/∂x_sender`, `∂E/∂Y`; `g_e = (∂E/∂d)·r_e/d_e + dYᵀ·∂E/∂Y` with `∂E/∂d` from `radial_embedding!`'s `dfeats` (through the MLP and density pullbacks) and `zbl_edge_energy`'s `de`; forces and virial by section 12.

- [ ] **Step 1: Write the failing tests:**

```julia
@testitem "forces and virial match LAMMPS-MACE" setup = [LammpsReference, Checkpoints] begin
    using LinearAlgebra
    for (ref, src) in ((:mpa0_f64, :mpa0), (:mp0b3_f64, :mp0b3))
        model = MACEModel(load_mace_checkpoint(Checkpoints.fetch(src)))
        for s in LammpsReference.structures(LammpsReference.load(ref))
            r = mace_energy_forces(model, s.positions, s.cell, s.atomic_numbers)
            # Targets from the design (section 9): 1e-6 eV and 1e-6 eV/Å against LAMMPS-MACE.
            @test abs(r.energy - s.energy) <= 1.0e-6
            @test maximum(abs, r.forces - s.forces; init = 0.0) <= 1.0e-6
            @test maximum(abs, r.virial - s.virial) <= 1.0e-6
            @test r.stress ≈ -r.virial ./ abs(det(s.cell)) rtol = 4 * eps()
        end
    end
end

@testitem "forces are minus the energy gradient; stress is the strain derivative" setup = [LammpsReference, Checkpoints] begin
    using LinearAlgebra
    m = MACEModel(load_mace_checkpoint(Checkpoints.fetch(:mpa0))); ref = LammpsReference.load(:mpa0_f64)
    E(P, C, Z) = mace_energy_forces(m, P, C, Z).energy
    for (name, atoms) in (("cu_vacancy", (1, 7)), ("multi_element", (2,)))      # atom 2 of multi_element: H, ZBL active
        s = LammpsReference.structure(ref, name); Z = s.atomic_numbers
        F = mace_energy_forces(m, s.positions, s.cell, Z).forces
        h = 1.0e-4
        for a in atoms, d in 1:3
            Pp = copy(s.positions); Pp[d, a] += h; Pm = copy(s.positions); Pm[d, a] -= h
            # Measured margin: tol_fd = 10-20 × the largest difference over these cases, recorded here.
            @test F[d, a] ≈ -(E(Pp, s.cell, Z) - E(Pm, s.cell, Z)) / 2h atol = tol_fd
        end
    end
    for name in ("triclinic_ni3al", "crconi_alloy")
        s = LammpsReference.structure(ref, name); Z = s.atomic_numbers
        W = mace_energy_forces(m, s.positions, s.cell, Z).virial
        h = 1.0e-5
        for (a, b) in ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))
            S = zeros(3, 3); S[a, b] += 0.5; S[b, a] += 0.5
            Ep = E((I + h * S) * s.positions, (I + h * S) * s.cell, Z); Em = E((I - h * S) * s.positions, (I - h * S) * s.cell, Z)
            # ∂E/∂ε_ab = −virial_ab. Measured margin tol_strain, recorded here.
            @test -W[a, b] ≈ (Ep - Em) / 2h atol = tol_strain
        end
    end
end

@testitem "rotation, translation and unwrapping" setup = [LammpsReference, Checkpoints] begin
    using LinearAlgebra
    m = MACEModel(load_mace_checkpoint(Checkpoints.fetch(:mpa0)))
    s = LammpsReference.structure(LammpsReference.load(:mpa0_f64), "triclinic_ni3al"); Z = s.atomic_numbers
    r0 = mace_energy_forces(m, s.positions, s.cell, Z)
    b = 64 * eps() * sum(abs, r0.node_energies)          # rounding bound on the summed energy
    R = [1 0 0; 0 cos(1.1) -sin(1.1); 0 sin(1.1) cos(1.1)] * [cos(0.7) -sin(0.7) 0; sin(0.7) cos(0.7) 0; 0 0 1]
    r1 = mace_energy_forces(m, R * s.positions, R * s.cell, Z)
    @test abs(r1.energy - r0.energy) <= b
    @test r1.forces ≈ R * r0.forces atol = tol_rot       # measured margin, recorded here
    @test r1.virial ≈ R * r0.virial * R' atol = tol_rot
    P2 = s.positions .+ [0.37, -1.1, 2.3]; P2[:, 3] .+= s.cell[:, 1] .- s.cell[:, 2]
    r2 = mace_energy_forces(m, P2, s.cell, Z)
    @test abs(r2.energy - r0.energy) <= b
    @test r2.forces ≈ r0.forces atol = tol_rot
    @test r2.virial ≈ r0.virial atol = tol_rot
end
```

  `tol_fd`, `tol_strain` and `tol_rot` are set in Step 4 from the measured largest differences (AGENTS.md policy (c)), with the measurement in the comment.

  Also: `@test_noalloc mace_energy_forces!` after warm-up; `mace_energy_forces` equals the `!` form (`==`); the zero-edge structure gives zero forces and zero virial exactly; Float32 forces against `:mpa0_f32` (measured margin); a workspace sized for `cu_vacancy` used on `crconi_alloy` throws the capacity message. TrimCheck and `test_signatures` gain `mace_energy_forces!` (same argument types as `mace_energy!`) and `mace_energy_forces(MACEModel{Float64}, Matrix{Float64}, Matrix{Float64}, Vector{Int})`.
- [ ] **Step 2:** Run items `forces`, `rotation`; expected FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run once with the tolerances unset to measure the largest differences; set `tol_fd`, `tol_strain`, `tol_rot` and the Float32 margins from them; run again; expected PASS. Record the largest LAMMPS differences (energy, force, virial) per model in the commit body.
- [ ] **Step 5:** Mutation for the pullback (drop the angular term `dYᵀ·∂E/∂Y`; `expect = "forces"`). Commit `Add MACE forces and stress through layer pullbacks`.

### Task 8: Precompile workload, remaining gates, documentation

**Files:**
- Create: `PureMLIP/src/tiny_checkpoint.jl` (internal to `PureMLIP`, included before `precompile.jl`), `PureMLIP/test/tiny_model_tests.jl`, `PureMLIP/docs/src/model.md`
- Modify: `PureMLIP/src/precompile.jl`, `PureMLIP/src/PureMLIP.jl`, `PureMLIP/docs/make.jl` (pages), `PureMLIP/test/coverage/thresholds.toml`, `PureMLIP/test/mutation/mutations.toml`, `PureMLIP/test/latency_tests.jl`, `AGENTS.md`

**Interfaces:**
- Produces `tiny_mace_checkpoint()::Vector{UInt8}`: an in-memory torch zip (STORED, ZIP64) holding a pickled ScaleShiftMACE of the supported architecture with `C = 2` channels, elements `[1, 29]`, radial MLP `[4, 4, 4]`, 10 Bessel functions, one coupling path per U matrix (one nonzero entry each), readout MLP `2x0e`, every module and attribute that `load_mace_checkpoint` and `MACEModel` read, and weights `0.1·sin(k)` for the k-th stored element. It uses no test code and no file or network access.

- [ ] **Step 1: Write the failing test:**

```julia
@testitem "tiny in-memory checkpoint loads and evaluates" begin
    ck = load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint())
    m = MACEModel(ck)
    P = [0.0 1.4 0.2; 0.0 0.1 1.3; 0.0 0.3 0.1]; cell = [6.5 0 0; 0.4 6.2 0; 0.3 0.2 6.8]
    r = mace_energy_forces(m, P, cell, [29, 1, 29])
    @test isfinite(r.energy) && all(isfinite, r.forces)
    # Forces against a central difference on every coordinate (h = 1e-5): measured margin.
end
```

- [ ] **Step 2:** Run item `tiny`; expected FAIL.
- [ ] **Step 3:** Implement `tiny_checkpoint.jl`; extend `precompile.jl`'s workload with `ck = load_mace_checkpoint(tiny_mace_checkpoint())`, `MACEModel{Float64}(ck)` and one `mace_energy_forces` call on the three-atom system above (keep the existing calls). Add `"PureMLIP.mace_energy_forces(PureMLIP.MACEModel(PureMLIP.load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint())), [0.0 1.4; 0.0 0.1; 0.0 0.3], [6.5 0 0; 0 6.2 0; 0 0 6.8], [29, 1])"` to the latency item's `calls`.
- [ ] **Step 4:** StrictMode borrow report: in `strictmode_tests.jl`, `register_scoped!(PureMLIP.MACEWorkspace{Float64})` and `@assert_noescape mace_energy_forces!(res, ws, model, P, cell, Z, nl)` on the `cu_vacancy` system. It reports and does not gate; record its outcome in the commit body for the StrictMode maintainers.
- [ ] **Step 5:** Documentation: `model.md` (loading a model, `positions` 3 × n in Å, `cell` columns, all-periodic requirement, units, stress sign and Voigt mapping, the neighbor list's capacity and `r_cut ≥ r_max`, the supported architecture, Float32). `AGENTS.md`: three submodules and the new layer graph; `[deps]` list with CellListMap; `@test_noalloc` list adds `neighbor_list!`, `mace_energy!`, `mace_energy_forces!`; the precompile paragraph (the workload now loads a tiny checkpoint written in memory by `src/tiny_checkpoint.jl`); oracles add the LAMMPS-MACE JSON (1e-6 eV target) and the e3nn-core-rs radial fixture; CI never runs LAMMPS.
- [ ] **Step 6:** Run the full suite with coverage (`Pkg.test(coverage = true)`), then `julia --project=PureMLIP/test/coverage PureMLIP/test/coverage/check.jl`; add a row for each new file at its measured value rounded down to two decimals (`src/model/Model.jl` = 1.00). Add a mutation for `tiny_checkpoint.jl` (a wrong CRC for one entry; `expect = "tiny"`). Run the mutation check (`PureMLIP/test/mutation/run.jl`) on all entries; every new mutation is killed. Run the juliac item (`PUREMLIP_JULIAC_BUILD=1`); `mace_inspect` still builds and runs.
- [ ] **Step 7:** Commit `Precompile model evaluation; document the MACE model`.

### Task 9: Speed measurement against LAMMPS-MACE, final review, push

**Files:**
- Create: `PureMLIP/bench/Project.toml` (Chairmarks, JSON, PureMLIP by path), `PureMLIP/bench/mace_cpu.jl`, `PureMLIP/bench/in.bench` (LAMMPS), `PureMLIP/bench/results/mace_cpu_mac.json`

**Interfaces:** `mace_cpu.jl` takes `$PUREMLIP_LAMMPS` and `$PUREMLIP_LAMMPS_MODEL_DIR`; for `cu_bench(5)` (500 atoms) and `cu_bench(14)` (10,976 atoms), Float64 and Float32, records: Julia median seconds of `mace_energy_forces!` (Chairmarks `@be`, fixed neighbor list, after warm-up), median of `neighbor_list!`, workspace bytes, and LAMMPS seconds per step (`run 0` then `run 20`, NVE, `timestep 0.001`, zero velocities, `OMP_NUM_THREADS=1`, "Loop time" / 20) with the MPA-0 export of the same precision.

- [ ] **Step 1:** Run `julia -t 1 --project=PureMLIP/bench PureMLIP/bench/mace_cpu.jl` with `BLAS.set_num_threads(1)` set in the script; write the JSON (CPU model string, Julia version, LAMMPS version, all medians, ratio LAMMPS/Julia).
- [ ] **Step 2:** Commit `Record CPU speed of MACE evaluation against LAMMPS` with the table in the body. No gate.
- [ ] **Step 3:** Full suite and juliac item green; scan every new commit and tracked file for host names and local paths; request the whole-branch review; after approval, `git push origin main`.
