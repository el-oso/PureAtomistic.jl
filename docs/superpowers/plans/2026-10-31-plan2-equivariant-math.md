# Plan 2: equivariant math (spherical harmonics, Clebsch-Gordan, e3nn conversion)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give PureMLIP textbook-convention real spherical harmonics (with gradients), real Clebsch-Gordan coefficients and real Wigner D matrices, plus an e3nn conversion layer that reproduces e3nn's spherical harmonics and `wigner_3j` exactly, verified against e3nn-core-rs fixtures and against the Wigner-3j tables stored in the MACE-MP-0b3 checkpoint.

**Architecture:** New folder `PureMLIP/src/equivariant/`. Native objects use the textbook convention: real spherical harmonics with the z axis as polar axis, `m = -l..l`, orthonormal on the sphere; real Clebsch-Gordan from Condon-Shortley complex coefficients. A separate file maps native objects to e3nn's convention (input-axis relabeling, `sqrt(4π)` scale, a sign per coupling). The checkpoint reader gains DEFLATE support so the TorchScript archives embedded in a checkpoint can be opened. Everything is generic in the element type where the math allows it, allocation-free in `!` functions, and trim-compatible.

**Tech Stack:** Julia 1.12+. New package dependency: Inflate.jl (pure Julia, MIT, no dependencies; passes TrimCheck `@validate` on `inflate(::Vector{UInt8})`, checked 2026-10-31). Test-only additions: WignerSymbols.jl (reference Clebsch-Gordan), ForwardDiff.jl (generic-type check), OffsetArrays.jl (generic-indexing check).

**Spec:** `docs/superpowers/specs/2026-10-30-pureatomistic-design.md`, sections 4 (constraints), 6.1 (spherical harmonics, Clebsch-Gordan, conventions and e3nn compatibility rows), 6.3 (engineering stack), 9 (part-by-part references).

## Global Constraints

- PureMLIP `[deps]`: Inflate only (plus Base/stdlib). No `_jll`, no Artifacts. WignerSymbols is **not** a package dependency: it fails TrimCheck `@validate` on `clebschgordan(Type{Float64}, Int, Int, Int, Int, Int, Int)` (checked 2026-10-31), so it is used in tests only.
- `julia = "1.12"` in `[compat]`; every added package gets a `[compat]` bound at its resolved `major.minor`.
- Native convention: real spherical harmonics, z polar axis, `m = -l..l`, orthonormal ("integral" normalization): `∫ Y_lm Y_l'm' dΩ = δ δ`. Layout of all `l ≤ lmax` values in one vector: index `l^2 + l + m + 1` (1-based).
- e3nn convention (only in `e3nn.jl`): component normalization, `Y_e3nn(v) = sqrt(4π) · Y_tb(v3, v1, v2)` for unit `v`, l-major, e3nn index `0..2l` inside each `l`.
- Clebsch-Gordan and e3nn coupling are supported only for `l1 + l2 + l3` even (the parity rule makes every MACE path even); an odd sum or a triangle violation throws `ArgumentError`.
- `!` functions allocate nothing after warm-up (StrictModeTest `@test_noalloc`, measured through a `@noinline` function barrier).
- Every exported function passes TrimCheck `@validate` and `test_signatures(...; guarantees = (:typestable,))`.
- Functions taking `AbstractVector`/`AbstractMatrix` honor arbitrary axes (`eachindex`, `axes`) or call `Base.require_one_based_indexing`; tests run entry points on `OffsetArray` and `view` inputs.
- No `@inbounds`. Errors in reachable code interpolating four or more pieces use `lazy"..."`.
- Comments state present facts only; no plan or task references.
- Commits: subject ≤ 72 chars; body for a reader with only the repo; trailers `Assisted-by: Claude <model> <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01RQSsNnmff93o3JgocxDLMn`. Author `el-oso <ursus.est@gmail.com>` (repo-local git config already set). Push straight to `main` at the end of the plan; no pull request. No host names or local absolute paths in any tracked file or commit.

Verified facts this plan relies on (2026-10-31; numerical checks done in throwaway Julia scripts):

- **e3nn-core-rs fixtures**, repo `github.com/mizukami-group/e3nn-core-rs` at commit `9da369cf1d854da6af1a46928bc953c7966e9382`, dual MIT OR Apache-2.0 (`LICENSE-MIT`: "Copyright (c) 2026 Mizukami Group and e3nn-core-rs contributors"). Generated with e3nn 0.6.0 and torch 2.8.0, float64. Stated tolerance 1e-10 absolute. Files under `crates/e3nn-core/fixtures/e3nn_torch/v0_6_0_2aa7f584/`:
  - `spherical_harmonics_lmax{2,3,4}_component_normalized_f64.json`: keys `vectors` (5×3, not unit: the three unit axes, `(1,2,3)`, `(-0.25,0.5,1.25)`), `y` (5 × (L+1)², e3nn `normalize=True`, `normalization="component"`), `grad_y` (cotangent, same shape as `y`), `grad_vectors` (5×3, vector-Jacobian product of `y` with respect to the raw `vectors`).
  - `fully_connected_tensor_product_smoke_f64.json` and `..._lmax4_smoke_f64.json`: keys include `irreps_in1/in2/out`, and per instruction the flattened sparse coefficients `cg_i_m, cg_j_m, cg_k_m, cg_value` (0-based m inside each irrep copy) with `instruction_cg_offset`, `instruction_cg_len`, and the segment lists. The stored coefficients equal e3nn `wigner_3j` for those paths (max diff 5e-16).
  - Checked: fixture `y` = `sqrt(4π) · Y_tb(v3, v1, v2)` (v normalized) to 2.2e-15 for lmax 2, 3, 4.
- **Map between conventions.** For unit `v`: `Y_e3nn,l(v) = sqrt(4π) · Y_tb,l(R v)` with `R v = (v3, v1, v2)`; equivalently `Y_e3nn,l(v) = sqrt(4π) · M'_l · Y_tb,l(v)` with `M'_l = D^l_tb(R)` orthogonal, det +1. Rows index e3nn components `0..2l`, columns textbook `m = -l..l`. `M'_0 = [1]`, `M'_1 = [0 0 1; 1 0 0; 0 1 0]`, `M'_2 = [0 0 0 1 0; 1 0 0 0 0; 0 0 -1/2 0 -√3/2; 0 1 0 0 0; 0 0 √3/2 0 -1/2]`. Verified to l = 11 for the input-relabeling form, to l = 6 for the matrix form.
- **Textbook real CG.** `CG_real[a,b,c] = Σ conj(U1[a,m1]) conj(U2[b,m2]) U3[c,m3] ⟨l1 m1 l2 m2 | l3 m3⟩`, with `⟨⋯⟩` the Condon-Shortley complex coefficient and `U_l` the complex-to-real unitary `Y_real = U · Y_complex` of the Wikipedia real form (`m<0`: `(i/√2)(Y^m − (−1)^m Y^{−m})`; `m=0`: `Y^0`; `m>0`: `(1/√2)(Y^{−m} + (−1)^m Y^m)`). It is real for even `l1+l2+l3` and `‖CG_real‖_F² = 2 l3 + 1`. Unit-norm form `C_tb = CG_real / sqrt(2 l3 + 1)`. Equivariance under real textbook D holds to 4.4e-16.
- **Relation to e3nn (even sums).** `C_e3nn[a,b,c] = s · Σ M'_l1[a,a'] M'_l2[b,b'] M'_l3[c,c'] C_tb[a',b',c']` with `s = (−1)^((l1+l2−l3)/2)`. Verified for all 69 even-sum triples with every `l ≤ 5`, worst residual 1.8e-15. Signs: (1,1,2) +1, (1,2,1) −1, (1,2,3) +1, (1,3,2) −1, (1,1,0) −1, (2,2,2) −1, (3,3,3) −1. e3nn 0.4.4's table and e3nn ≥ 0.5's algorithm agree on every even-sum triple tested (they differ in sign on some odd-sum triples, which this plan does not support).
- **Wigner-3j tables in MP-0b3** (URL and SHA-256 in plan 1). The outer `data.pkl` holds 18 TorchScript archives as `PK_BYTES`; exactly one contains `archive/data/` entries. Its tensors (shape, strides in elements, all offset 0, Float64, little-endian):

  | name | entry | shape | strides |
  |---|---|---|---|
  | `_w3j_1_2_1` | `data/0` | (3,5,3) | (5,1,15) |
  | `_w3j_1_1_2` | `data/1` | (3,3,5) | (15,5,1) |
  | `_w3j_1_3_2` | `data/2` | (3,7,5) | (35,1,7) |
  | `_w3j_1_2_3` | `data/3` | (3,5,7) | (35,7,1) |

  They equal e3nn 0.4.4 `wigner_3j` (three bit-identical, one within 7.2e-16), Frobenius norm 1. That archive's `data.pkl` and `data/` entries are STORED; its four `code/` entries are DEFLATED. Its own pickle uses `torch.fx` globals outside PureMLIP's allowlist, so the table above was read from it by hand and is restated in the test.
- **SpheriCart** (`github.com/lab-cosmo/sphericart`, MIT): z-polar, Condon-Shortley phase cancelled, orthonormal; matches the textbook definition above to 2.1e-15 for l ≤ 6. Its real-SH recurrence: `julia/src/ka_kernels.jl` lines 117-187 and `julia/src/generated_kernels.jl` lines 42-125. Port the recurrence (MIT, keep attribution in a comment); do not depend on the package (its closure carries `_jll`s).

## Review Focus

1. **Zero-length direction vector** (two atoms at the same position, or a bug upstream): expected `ArgumentError("spherical harmonics of a zero vector are undefined")`, not NaNs. Test in Task 2.
2. **Non-unit input vectors** (MACE passes raw edge vectors): the functions normalize internally, and the gradient is with respect to the raw vector. Test in Task 2 and against the fixture `grad_vectors` in Task 4.
3. **Odd `l1+l2+l3` or a triangle violation passed to the CG or e3nn coupling functions**: expected an `ArgumentError` naming the triple. Test in Task 3.
4. **High `l`** (a user asks for lmax 8 or more): values stay orthonormal to 1e-12 and no overflow occurs in the factorials of the CG formula for every `l ≤ 8`. Test in Tasks 2 and 3.
5. **A checkpoint whose embedded archive uses a compression method other than STORED or DEFLATE**: expected `compressed entries are not supported: <name> (method <n>)`. Test in Task 1.

---

### Task 1: DEFLATE entries and embedded archives

**Files:**
- Modify: `PureMLIP/Project.toml` (add Inflate with `[compat]`), `PureMLIP/src/checkpoint/torchzip.jl`
- Create: `PureMLIP/src/checkpoint/embedded.jl`
- Modify: `PureMLIP/test/zip_tests.jl`, `PureMLIP/test/testutils.jl`; Create: `PureMLIP/test/embedded_tests.jl`

**Interfaces:**
- Consumes: `TorchArchive`, `ZipEntry`, `open_torch_archive`, `entry_data` (torchzip.jl); `PickleTree`, `PK_BYTES` (pickle.jl).
- Produces:
  - `ZipEntry` gains `method::UInt16` and `compressed_size::Int`; method 0 (STORED) and 8 (DEFLATE) are accepted at open time, any other method throws `compressed entries are not supported: <name> (method <n>)` (unchanged text).
  - `entry_data(a::TorchArchive, name::AbstractString)` keeps one concrete return type, `SubArray{UInt8, 1, Vector{UInt8}, Tuple{UnitRange{Int}}, true}`: for STORED entries a view of the archive bytes (as now); for DEFLATE entries a full-range view of the decoded bytes (`Inflate.inflate` on the raw stream), which are cached in a new concrete field `inflated::Dict{String,Vector{UInt8}}` so a second access does not decode again. Decoded bytes are CRC-checked against the uncompressed CRC and size-checked against the recorded uncompressed size (`inflated size mismatch in <name>`).
  - `embedded_archives(t::PickleTree)::Vector{TorchArchive}`: one archive for each `PK_BYTES` node whose bytes start with `PK\x03\x04`, in node order; each opened with `open_torch_archive` on a `Vector{UInt8}` copy.
- Test helper: `ZipFixtures.stored_zip` gains a keyword `deflate::Vector{String} = String[]` naming entries to store with method 8 (compressed with a minimal valid DEFLATE stream: a single "stored block" — header byte `0x01`, then `LEN`, `NLEN`, data — which `inflate` decodes).

- [ ] **Step 1: Write the failing tests:**

```julia
@testitem "deflated entry is decoded and CRC-checked" setup = [ZipFixtures] begin
    z = ZipFixtures.stored_zip(["a/data.pkl" => UInt8[0x80, 0x02, 0x2e], "a/code.py" => collect(codeunits("x = 1\n"))]; deflate = ["a/code.py"])
    a = PureMLIP.open_torch_archive(z)
    @test collect(PureMLIP.entry_data(a, "code.py")) == collect(codeunits("x = 1\n"))
end

@testitem "unsupported method still rejected" setup = [ZipFixtures] begin
    z = ZipFixtures.stored_zip(["a/data.pkl" => UInt8[1, 2, 3]]; method = 0x000c)
    @test_throws "compressed entries are not supported: a/data.pkl (method 12)" PureMLIP.open_torch_archive(z)
end

@testitem "MP-0b3 embeds one archive with Wigner tables" setup = [Checkpoints] begin
    a = PureMLIP.open_torch_archive(read(Checkpoints.fetch(:mp0b3)))
    archives = PureMLIP.embedded_archives(PureMLIP.unpickle(PureMLIP.entry_data(a, "data.pkl")))
    @test length(archives) == 18
    with_data = filter(x -> haskey(x.entries, x.prefix * "data/0"), archives)
    @test length(with_data) == 1
    @test [with_data[1].entries[with_data[1].prefix * "data/$k"].size for k in 0:3] == [360, 360, 840, 840]
end
```

Delete the old plan-1 test that expected method 8 to be rejected; the method-12 test replaces it.
- [ ] **Step 2:** Run `julia --project=PureMLIP -e 'using Pkg; Pkg.test()'`; expected FAIL.
- [ ] **Step 3:** Implement. Add Inflate with `Pkg.add` in the PureMLIP project and a `[compat]` bound.
- [ ] **Step 4:** Run tests; expected PASS (including all plan-1 tests).
- [ ] **Step 5:** Commit: `Decode DEFLATE zip entries and open embedded archives`.

### Task 2: Textbook real spherical harmonics with gradients

**Files:**
- Create: `PureMLIP/src/equivariant/spherical_harmonics.jl`
- Create: `PureMLIP/test/sh_tests.jl`

**Interfaces:**
- Produces (all exported):
  - `sh_length(lmax::Integer)::Int` = `(lmax+1)^2`.
  - `sh_index(l::Integer, m::Integer)::Int` = `l^2 + l + m + 1`.
  - `spherical_harmonics!(Y::AbstractVector{T}, v::AbstractVector{T}, lmax::Integer)::typeof(Y)`: fills `Y` (length `sh_length(lmax)`, any axes) with `Y_lm(v/‖v‖)`, textbook convention. `v` has length 3 (any axes; components read in `eachindex` order as x, y, z).
  - `spherical_harmonics_and_gradient!(Y::AbstractVector{T}, dY::AbstractMatrix{T}, v::AbstractVector{T}, lmax::Integer)`: also fills `dY[k, i] = ∂Y_k/∂v_i` with respect to the raw, unnormalized `v` (`size(dY) == (sh_length(lmax), 3)`, any axes).
  - Both throw `ArgumentError("spherical harmonics of a zero vector are undefined")` when `‖v‖ == 0`, and `DimensionMismatch` with a message naming the expected lengths when sizes disagree.
- Algorithm: port SpheriCart's recurrence (see Verified facts) generic in `T`; no table of precomputed coefficients that fixes `T` to Float64.

- [ ] **Step 1: Write the failing tests:**

```julia
@testitem "closed forms for l ≤ 2" begin
    v = [0.3, -0.5, 0.8]; u = v / sqrt(sum(abs2, v)); x, y, z = u
    Y = zeros(9); PureMLIP.spherical_harmonics!(Y, v, 2)
    c0 = 1 / (2sqrt(pi)); c1 = sqrt(3 / (4pi)); c2 = sqrt(15 / (4pi)); c20 = sqrt(5 / (16pi))
    @test Y ≈ [c0, c1 * y, c1 * z, c1 * x, c2 * x * y, c2 * y * z, c20 * (3z^2 - 1), c2 * x * z, c2 / 2 * (x^2 - y^2)] atol = 1e-14
end

@testitem "orthonormal up to lmax 8" begin
    # Gauss-Legendre in cos θ (n = 20 nodes) × uniform φ (n = 40): exact for degree ≤ 16 polynomials
    G = PureMLIP_test_quadrature(20, 40)          # helper in the test file: returns (weights, unit vectors)
    L = 8; n = PureMLIP.sh_length(L)
    S = zeros(n, n); Y = zeros(n)
    for (w, v) in G
        PureMLIP.spherical_harmonics!(Y, v, L); S .+= w .* (Y * Y')
    end
    @test maximum(abs, S - I) < 1e-12
end

@testitem "gradient matches finite differences and ForwardDiff" begin
    using ForwardDiff
    v = [1.0, 2.0, 3.0]; L = 4; n = PureMLIP.sh_length(L)
    Y = zeros(n); dY = zeros(n, 3); PureMLIP.spherical_harmonics_and_gradient!(Y, dY, v, L)
    J = ForwardDiff.jacobian(w -> PureMLIP.spherical_harmonics!(zeros(eltype(w), n), w, L), v)
    @test dY ≈ J atol = 1e-12
end

@testitem "zero vector and size errors" begin
    @test_throws "spherical harmonics of a zero vector are undefined" PureMLIP.spherical_harmonics!(zeros(4), zeros(3), 1)
    @test_throws DimensionMismatch PureMLIP.spherical_harmonics!(zeros(3), [1.0, 0, 0], 1)
end

@testitem "OffsetArray and view inputs" begin
    using OffsetArrays
    v = [0.2, 0.4, -0.9]; Y = zeros(16); PureMLIP.spherical_harmonics!(Y, v, 3)
    Yo = OffsetVector(zeros(16), -3); PureMLIP.spherical_harmonics!(Yo, OffsetVector(v, 5), 3)
    @test parent(Yo) == Y
    buf = zeros(20); PureMLIP.spherical_harmonics!(view(buf, 3:18), view([9.0, v...], 2:4), 3)
    @test buf[3:18] == Y
end
```

(`I` from LinearAlgebra in the test; the quadrature helper is written in the test file.)
- [ ] **Step 2:** Run tests; expected FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run tests; expected PASS.
- [ ] **Step 5:** Commit: `Add textbook real spherical harmonics with gradients`.

### Task 3: Real Clebsch-Gordan and real Wigner D

**Files:**
- Create: `PureMLIP/src/equivariant/clebsch_gordan.jl`, `PureMLIP/src/equivariant/wigner_d.jl`
- Create: `PureMLIP/test/cg_tests.jl`

**Interfaces:**
- Consumes: `spherical_harmonics!`, `sh_index` (Task 2).
- Produces (exported):
  - `clebsch_gordan_complex(l1, m1, l2, m2, l3, m3)::Float64`: Condon-Shortley ⟨l1 m1 l2 m2 | l3 m3⟩ by the Racah formula, integer `l`, factorials in Float64 (exact for every argument with `l1+l2+l3 ≤ 21`).
  - `real_clebsch_gordan(l1::Integer, l2::Integer, l3::Integer)::Array{Float64,3}`: `C_tb` of Verified facts, shape `(2l1+1, 2l2+1, 2l3+1)`, Frobenius norm 1. Throws `ArgumentError(lazy"real Clebsch-Gordan needs |l1-l2| ≤ l3 ≤ l1+l2 and even l1+l2+l3, got ($l1, $l2, $l3)")`.
  - `wigner_D(l::Integer, R::AbstractMatrix{<:Real})::Matrix{Float64}`: the real textbook matrix with `Y_l(R v) = D · Y_l(v)` for every `v`, `R` a 3×3 rotation (`R'R ≈ I` and `det R ≈ 1` to 1e-12, else `ArgumentError`). Computed by evaluating `spherical_harmonics!` at a fixed set of `2l+1` or more well-spread unit vectors and solving the linear system.

- [ ] **Step 1: Write the failing tests:**

```julia
@testitem "complex CG matches WignerSymbols" begin
    using WignerSymbols
    for l1 in 0:6, l2 in 0:6, l3 in abs(l1 - l2):min(l1 + l2, 6), m1 in -l1:l1, m2 in -l2:l2
        m3 = m1 + m2; abs(m3) <= l3 || continue
        @test PureMLIP.clebsch_gordan_complex(l1, m1, l2, m2, l3, m3) ≈ WignerSymbols.clebschgordan(Float64, l1, m1, l2, m2, l3, m3) atol = 1e-14
    end
end

@testitem "real CG: norm, equivariance, errors" begin
    using LinearAlgebra
    R = [cos(0.7) -sin(0.7) 0; sin(0.7) cos(0.7) 0; 0 0 1] * [1 0 0; 0 cos(1.1) -sin(1.1); 0 sin(1.1) cos(1.1)]
    for (l1, l2, l3) in ((1, 1, 0), (1, 1, 2), (1, 2, 1), (1, 2, 3), (2, 2, 2), (3, 3, 4), (4, 4, 8))
        C = PureMLIP.real_clebsch_gordan(l1, l2, l3)
        @test norm(C) ≈ 1 atol = 1e-14
        D1, D2, D3 = (PureMLIP.wigner_D(l, R) for l in (l1, l2, l3))
        C2 = [sum(C[a, b, c] * D1[a, i] * D2[b, j] * D3[c, k] for a in axes(C, 1), b in axes(C, 2), c in axes(C, 3)) for i in axes(C, 1), j in axes(C, 2), k in axes(C, 3)]
        @test C2 ≈ C atol = 1e-13
    end
    @test_throws "even l1+l2+l3" PureMLIP.real_clebsch_gordan(1, 1, 1)
    @test_throws "|l1-l2| ≤ l3" PureMLIP.real_clebsch_gordan(1, 1, 3)
end

@testitem "wigner_D is a representation" begin
    using LinearAlgebra
    Ra = [cos(0.3) 0 sin(0.3); 0 1 0; -sin(0.3) 0 cos(0.3)]; Rb = [1 0 0; 0 cos(1.2) -sin(1.2); 0 sin(1.2) cos(1.2)]
    for l in 0:6
        Da, Db = PureMLIP.wigner_D(l, Ra), PureMLIP.wigner_D(l, Rb)
        @test PureMLIP.wigner_D(l, Ra * Rb) ≈ Da * Db atol = 1e-12
        @test Da' * Da ≈ I atol = 1e-12
    end
    @test_throws "rotation" PureMLIP.wigner_D(1, [2.0 0 0; 0 1 0; 0 0 1])
end
```

(The equivariance statement in the second item — contracting each index with `D` returns `C` — is the invariance of the coupling; if the implementer finds the transposed form is the correct one for this definition, use `D'` consistently and state it in the docstring. The reference check above is the arbiter.)
- [ ] **Step 2:** Run tests; expected FAIL.
- [ ] **Step 3:** Implement. Add WignerSymbols to `test/Project.toml` only, with a `[compat]` bound.
- [ ] **Step 4:** Run tests; expected PASS.
- [ ] **Step 5:** Commit: `Add real Clebsch-Gordan coefficients and real Wigner D`.

### Task 4: e3nn conversion layer

**Files:**
- Create: `PureMLIP/src/equivariant/e3nn.jl`
- Create: `PureMLIP/test/fixtures/e3nn-core-rs/` with the five JSON fixtures named in Verified facts (verbatim, from the pinned commit), `LICENSE-MIT` and `LICENSE-APACHE` from that commit, and a `NOTICE` (source repo, path, commit, "redistributed unmodified", MIT OR Apache-2.0).
- Create: `PureMLIP/test/e3nn_tests.jl`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces (exported):
  - `e3nn_spherical_harmonics!(Y::AbstractVector{T}, v::AbstractVector{T}, lmax::Integer)`: e3nn `normalize=True, normalization="component"` values, l-major, e3nn order, computed as `sqrt(4π) · Y_tb(v3, v1, v2)`.
  - `e3nn_spherical_harmonics_and_gradient!(Y, dY, v, lmax)`: as Task 2, gradient with respect to the raw `v` in the original `(v1, v2, v3)` order.
  - `e3nn_basis_change(l::Integer)::Matrix{Float64}`: `M'_l` (orthogonal; `Y_e3nn = sqrt(4π) · M'_l · Y_tb`), computed as `wigner_D(l, R)` with `R` the matrix of `v ↦ (v3, v1, v2)`.
  - `e3nn_wigner_3j(l1::Integer, l2::Integer, l3::Integer)::Array{Float64,3}`: `s · (M'_l1 ⊗ M'_l2 ⊗ M'_l3) · C_tb` with `s = (−1)^((l1+l2−l3)/2)` (Verified facts); same errors as `real_clebsch_gordan`.

- [ ] **Step 1: Write the failing tests:**

```julia
@testitem "e3nn SH and gradient match e3nn-core-rs" setup = [E3nnFixtures] begin
    for L in (2, 3, 4)
        f = E3nnFixtures.sh(L)                                   # parsed JSON: vectors, y, grad_y, grad_vectors as matrices
        n = PureMLIP.sh_length(L)
        for i in axes(f.vectors, 1)
            Y = zeros(n); dY = zeros(n, 3)
            PureMLIP.e3nn_spherical_harmonics_and_gradient!(Y, dY, f.vectors[i, :], L)
            @test Y ≈ f.y[i, :] atol = 1e-10
            @test dY' * f.grad_y[i, :] ≈ f.grad_vectors[i, :] atol = 1e-10
        end
    end
end

@testitem "e3nn_wigner_3j matches the tensor-product fixtures" setup = [E3nnFixtures] begin
    for name in ("fully_connected_tensor_product_smoke_f64", "fully_connected_tensor_product_lmax4_smoke_f64")
        for (l1, l2, l3, entries) in E3nnFixtures.cg_instructions(name)   # entries: (i, j, k, value), 0-based m
            C = PureMLIP.e3nn_wigner_3j(l1, l2, l3)
            @test all(isapprox(C[i + 1, j + 1, k + 1], v; atol = 1e-12) for (i, j, k, v) in entries)
            @test count(!iszero, C) == length(entries)
        end
    end
end

@testitem "e3nn_wigner_3j matches the tables stored in MP-0b3" setup = [Checkpoints] begin
    a = PureMLIP.open_torch_archive(read(Checkpoints.fetch(:mp0b3)))
    arch = only(filter(x -> haskey(x.entries, x.prefix * "data/0"), PureMLIP.embedded_archives(PureMLIP.unpickle(PureMLIP.entry_data(a, "data.pkl")))))
    # name, entry, shape, strides (elements): read from that archive's own data.pkl, which uses torch.fx globals the reader does not allowlist
    for (l, entry, shape, strides) in (((1, 2, 1), "data/0", (3, 5, 3), (5, 1, 15)), ((1, 1, 2), "data/1", (3, 3, 5), (15, 5, 1)),
                                       ((1, 3, 2), "data/2", (3, 7, 5), (35, 1, 7)), ((1, 2, 3), "data/3", (3, 5, 7), (35, 7, 1)))
        raw = reinterpret(Float64, collect(PureMLIP.entry_data(arch, entry)))
        T = [raw[1 + (i - 1) * strides[1] + (j - 1) * strides[2] + (k - 1) * strides[3]] for i in 1:shape[1], j in 1:shape[2], k in 1:shape[3]]
        @test PureMLIP.e3nn_wigner_3j(l...) ≈ T atol = 1e-14
    end
end

@testitem "e3nn_basis_change for l = 1, 2" begin
    @test PureMLIP.e3nn_basis_change(1) ≈ [0 0 1; 1 0 0; 0 1 0] atol = 1e-14
    @test PureMLIP.e3nn_basis_change(2) ≈ [0 0 0 1 0; 1 0 0 0 0; 0 0 -1/2 0 -sqrt(3)/2; 0 1 0 0 0; 0 0 sqrt(3)/2 0 -1/2] atol = 1e-14
end
```

`E3nnFixtures` is a new `@testmodule` in `testutils.jl` that reads the vendored JSON with the test-only JSON package.
- [ ] **Step 2:** Run tests; expected FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run tests; expected PASS.
- [ ] **Step 5:** Commit: `Add e3nn conversion for spherical harmonics and Wigner-3j`.

### Task 5: Gates, docs, push

**Files:**
- Modify: `PureMLIP/test/trim_tests.jl`, `PureMLIP/test/strictmode_tests.jl`, `PureMLIP/docs/src/index.md` (or a new page `docs/src/equivariant.md` added to `make.jl` pages), `PureMLIP/src/PureMLIP.jl` (exports)

- [ ] **Step 1: Write the failing tests** (extend the existing items):

```julia
# trim_tests.jl, inside the existing @validate list
PureMLIP.spherical_harmonics!(Vector{Float64}, Vector{Float64}, Int),
PureMLIP.spherical_harmonics_and_gradient!(Vector{Float64}, Matrix{Float64}, Vector{Float64}, Int),
PureMLIP.real_clebsch_gordan(Int, Int, Int),
PureMLIP.wigner_D(Int, Matrix{Float64}),
PureMLIP.e3nn_spherical_harmonics_and_gradient!(Vector{Float64}, Matrix{Float64}, Vector{Float64}, Int),
PureMLIP.e3nn_wigner_3j(Int, Int, Int),
PureMLIP.entry_data(PureMLIP.TorchArchive, String),
```

```julia
@testitem "StrictMode: spherical harmonics are allocation-free" tags = [:checks] begin
    using StrictModeTest, StrictMode
    Y = zeros(25); dY = zeros(25, 3); v = [0.1, 0.2, 0.9]
    @noinline f!(Y, v) = PureMLIP.spherical_harmonics!(Y, v, 4)
    @noinline g!(Y, dY, v) = PureMLIP.e3nn_spherical_harmonics_and_gradient!(Y, dY, v, 4)
    f!(Y, v); g!(Y, dY, v)
    @test_noalloc f!(Y, v)
    @test_noalloc g!(Y, dY, v)
end
```

Also add the new exported functions to the existing `test_signatures` list.
- [ ] **Step 2:** Run `Pkg.test()`; fix every finding at its source.
- [ ] **Step 3:** Docs: one page explaining the native convention, the index layout, the e3nn mapping (the two formulas and the sign rule), and the even-sum restriction.
- [ ] **Step 4:** Run the full suite once (`Pkg.test()`), and the juliac build item (`PUREMLIP_JULIAC_BUILD=1`), both green.
- [ ] **Step 5:** Commit: `Gate equivariant math with TrimCheck and StrictMode`.
- [ ] **Step 6:** Scan every new commit for host names and local paths; fast-forward `main`; `git push origin main`.
