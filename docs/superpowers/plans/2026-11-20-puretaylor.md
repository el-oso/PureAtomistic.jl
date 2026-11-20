# PureTaylor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A dependency-free package `PureTaylor/` in this monorepo providing a truncated multivariate Taylor number type and branch-free elementary functions, fast enough that PureXC can beat Libxc with it.

**Architecture:** One module, `PureTaylor`. `Taylor{N,K,M,T,L} <: Real` stores the normalized coefficients of the monomials in the variables of the bitmask `M` up to total degree `K`, once each, in an `NTuple`. All arithmetic is `@generated` straight-line `muladd` code, inlined. Univariate functions compute their K+1 derivatives once and compose them by Horner's rule. A separate family of `fast_*` functions gives branch-free scalar kernels for Float32/Float64 so that a caller's loop over points vectorizes.

**Tech Stack:** Julia 1.12 and 1.13; `[deps]` StrictMode only; tests with TestItemRunner, StrictModeTest, TrimCheck, Aqua, ExplicitImports, ForwardDiff (oracle only).

**Spec:** `docs/superpowers/specs/2026-11-20-xc-functionals-design.md` (section "Derivatives" and "Decided by the user"). Read it and `AGENTS.md` before the first task. The working spike that measured this design is not in the repository; this plan carries what was learned from it.

## Global Constraints

- No Python in any form. No `_jll` and no package with an `Artifacts.toml` in the package closure.
- `[deps]`: StrictMode only (`[compat] StrictMode = "0.5.10"`). `[compat] julia = "1.12"`.
- Every module ends with `@strict_policy (:typestable, :noalloc)`; `@strict_exempt` only for entry points that allocate a workspace (PureTaylor has none, so none is expected).
- Float32 stays Float32: no operation may widen a Float32 Taylor or scalar to Float64.
- No `@inbounds`. No raw pointers.
- Fail fast: Base-named methods (`log`, `sqrt`, `cbrt`, `^`, ...) on a `Taylor` keep Base's domain behavior for the value (they throw `DomainError` where Base does). Only the `fast_*` functions skip domain checks, and their domains are documented.
- Comments describe the code as it is; no history, no plan or task references (AGENTS.md).
- Tolerance policy of AGENTS.md: every numeric comparison states its oracle and a bound of kind (a), (b) or (c), in a comment.
- Commits: subject ≤ 72 characters; trailers `Assisted-by: Claude <model> <noreply@anthropic.com>` and `Claude-Session: <url>`; push straight to `main` after review; watch every CI job after a push.
- The implementer never opens Libxc source or any file outside this repository that is not named in this plan.

## Conventions fixed by this plan

- **Multi-index order** (`multi_indices(N, K)`): graded by total degree, degree 0 first; within one degree, descending lexicographic order of the exponent tuple. N = 2, K = 2: `(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)`. N = 3, degree 2: `(2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)`.
- **Stored coefficients** of `Taylor{N,K,M,T,L}`: the coefficients `c_α = (∂^α f)/α!` for every α in `multi_indices(N, K)` whose nonzero exponents all lie in the variables of `M` (bit `i-1` of `M` set means variable `i`), in that order. `L == binomial(count_ones(M) + K, K)`. `c[1]` is the value.
- **Scaled seeds:** `variable(Val(N), Val(K), Val(i), x, s)` is the number `x + s·u_i`; the series is in `u`. `derivatives(a, invscale)` returns `∂^α f = α! · c_α · Π invscale_i^α_i`, with `invscale_i = 1/s_i`. Seeding with `s = x` keeps every coefficient of order 1 in size; without it, the third σ-derivative of PBE exchange at ρ = 1e-6 overflows Float32 in an intermediate although the result fits.

## Review Focus

1. **Tuple lengths above 32** (N = 5, K = 3 gives L = 56): `map`/`ntuple` without `Val` over long tuples goes dynamic and allocates. Expect: zero allocations and type stability at N = 5, K = 3. Pinned in Task 6's workload and Task 2's allocation item.
2. **Mixed Float32/Float64 operands:** `Taylor{…,Float32}` with a `Float64` scalar must stay Float32 (the scalar is converted); two Taylors of different `T` must raise a `MethodError`, never widen. Pinned in Task 1.
3. **Operands with different masks** (`M1 ≠ M2`, including disjoint ones and `M = 0` constants): the result has mask `M1 | M2` and the correct coefficients. Pinned in Task 1 and Task 2.
4. **Base domain errors through a Taylor:** `log` of a Taylor with a negative value must throw `DomainError` as `log(-1.0)` does, not return NaN. Pinned in Task 3.
5. **K = 0:** a Taylor of order 0 behaves as its scalar value in every operation. Pinned in Task 2 and Task 3.

---

### Task 1: Package skeleton, the type, seeds, read-out, addition

**Files:**
- Create: `PureTaylor/Project.toml`, `PureTaylor/src/PureTaylor.jl`, `PureTaylor/src/indices.jl`, `PureTaylor/src/type.jl`, `PureTaylor/src/arith.jl`
- Create: `PureTaylor/test/Project.toml`, `PureTaylor/test/runtests.jl`, `PureTaylor/test/type_tests.jl`, `PureTaylor/test/quality_tests.jl`
- Modify: `.github/workflows/CI.yml` (add `PureTaylor` to every `package:` matrix axis; it needs no `PUREMLIP_CHECKPOINT_DIR`)

**Interfaces:**
- Produces:
  - `struct Taylor{N,K,M,T,L} <: Real; c::NTuple{L,T}; end` (exported). Inner constructor `Taylor{N,K,M,T,L}(c)` checks `L == binomial(count_ones(M) + K, K)` and `0 ≤ M < 2^N` (an `ArgumentError` with prefix `"PureTaylor: "`).
  - `multi_indices(N::Int, K::Int)::Vector{NTuple{N,Int}}` (internal; used at generation time only).
  - `variable(::Val{N}, ::Val{K}, ::Val{i}, x::T, s::T = one(T)) where {N,K,i,T}::Taylor{N,K,1<<(i-1),T,K+1}` (exported).
  - `constant(::Val{N}, ::Val{K}, x::T)::Taylor{N,K,0,T,1}` (exported).
  - `value(a::Taylor) = a.c[1]`, `value(x::Real) = x` (exported).
  - `coefficient(a::Taylor{N}, α::NTuple{N,Int})` → `c_α`, or `zero(T)` when α is not stored (public).
  - `derivatives(a::Taylor{N,K,M,T}, invscale::NTuple{N,T})::NTuple{binomial(N+K,K),T}` in `multi_indices(N, K)` order, zero where α is not stored (exported).
  - `Base.:+`, `Base.:-` (binary and unary) for Taylor–Taylor with equal `N`, `K`, `T` (mask `M1 | M2`) and Taylor–`Real` (the scalar is `convert`ed to `T`); `zero`/`one` on instances and types; `isfinite`, `isnan` (over all coefficients).

- [ ] **Step 1: Write the failing tests** in `test/type_tests.jl`:

```julia
@testitem "multi-index order" begin
    using PureTaylor: multi_indices
    @test multi_indices(2, 2) == [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]
    @test multi_indices(3, 2)[5:10] == [(2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)]
    @test length(multi_indices(5, 3)) == 56
end

@testitem "seeds and read-out" begin
    a = variable(Val(2), Val(2), Val(2), 3.0, 0.5)     # 3 + 0.5 u₂
    @test a isa Taylor{2,2,0b10,Float64,3}
    @test value(a) == 3.0 && coefficient(a, (0, 1)) == 0.5 && coefficient(a, (1, 0)) == 0.0
    # ∂/∂x₂ of x₂ is 1 after unscaling by 1/s
    @test derivatives(a, (1.0, 2.0)) == (3.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    @test constant(Val(2), Val(2), 4.0f0) isa Taylor{2,2,0,Float32,1}
end

@testitem "addition over different masks" begin
    x = variable(Val(2), Val(2), Val(1), 1.0); y = variable(Val(2), Val(2), Val(2), 2.0)
    z = x - y + 0.5
    @test z isa Taylor{2,2,0b11,Float64,6}
    @test z.c == (-0.5, 1.0, -1.0, 0.0, 0.0, 0.0)            # exact: integer-valued arithmetic
    @test (x + constant(Val(2), Val(2), 1.0)) isa Taylor{2,2,0b01,Float64,3}
end

@testitem "Float32 stays Float32; mixed T is an error" begin
    x = variable(Val(1), Val(2), Val(1), 1.0f0)
    @test (x + 0.1) isa Taylor{1,2,1,Float32,3}
    @test_throws MethodError x + variable(Val(1), Val(2), Val(1), 1.0)
end

@testitem "invalid parameters" begin
    @test_throws "PureTaylor: " Taylor{2,2,0b100,Float64,1}((1.0,))
    @test_throws "PureTaylor: " Taylor{2,2,0b01,Float64,2}((1.0, 2.0))
end
```

`test/quality_tests.jl`: `Aqua.test_all(PureTaylor)` with every sub-check, and the ExplicitImports checks used for PureAtomistic (AGENTS.md "PureAtomistic's gates"), and the item "core closure is binary-free" copied from PureMLIP's test with the package name changed.

- [ ] **Step 2: Run, expect failure:** `julia --project=PureTaylor -e 'using Pkg; Pkg.test()'` → fails (package does not exist / `UndefVarError`).
- [ ] **Step 3: Implement** the files above. `Project.toml`: name `PureTaylor`, a fresh UUID, version `0.1.0`, `authors = ["el_oso"]`, `[deps] StrictMode`, `[compat] StrictMode = "0.5.10"`, `julia = "1.12"`. Test `Project.toml` as in PureAtomistic for Aqua, ExplicitImports, StrictMode, StrictModeTest, Test, TestItemRunner, TrimCheck, plus `ForwardDiff = "1"` (oracle), `[preferences.StrictMode] checks_enabled = true`. Add-sub is one `@generated` function over the union mask with an inline marker on the generated body (`Expr(:meta, :inline)`); out-of-line generated bodies passed tuples through memory in the spike.
- [ ] **Step 4: Run, expect pass** (same command), and `Pkg.resolve()` on Julia 1.12 and 1.13 so both `Manifest-v1.12.toml` and `Manifest-v1.13.toml` exist in `PureTaylor/` and `PureTaylor/test/`.
- [ ] **Step 5: Commit** `Add PureTaylor with its Taylor type and addition`.

### Task 2: Product, powers, inverse, division

**Files:** Modify `PureTaylor/src/arith.jl`; Create `PureTaylor/src/univariate.jl` (`compose` only); Test `PureTaylor/test/arith_tests.jl`

**Interfaces:**
- Consumes: Task 1.
- Produces: `Base.:*` (Taylor–Taylor, Taylor–Real), `Base.inv`, `Base.:/` (all four operand pairs; `a/b = a*inv(b)`), `Base.literal_pow(::typeof(^), a::Taylor, ::Val{n})` for every integer `n` (negative through `inv`), and `Base.:^(a::Taylor, n::Integer)`. Internal: `mul_truncated(a, b, ::Val{skip0}, ::Val{D})` — the product restricted to degrees `≤ D`, skipping `a`'s constant term when `skip0`.

- [ ] **Step 1: Failing tests.** Oracle for products: a naive dense product written in the test (loop over all pairs of `multi_indices`, keep `|γ| ≤ K`), on `Rational{BigInt}` coefficients, compared with `==` (exact arithmetic). Oracle for `inv`: `inv(a) * a` equals `one` exactly on `Rational{BigInt}`; on Float64, `derivatives(inv(x*y + 2))` against nested `ForwardDiff` on `BigFloat` with a rounding bound `c·eps(Float64)·N` (state `c` and `N` in the comment). Allocation item: `@test_noalloc` (StrictModeTest) on `a * b` and `inv(a)` for `N = 5, K = 3`, Float32 and Float64 (L = 56; Review Focus 1). `K = 0` item: `variable(Val(1), Val(0), Val(1), 2.0)^3 / 4` has value `2.0` and no other coefficient.
- [ ] **Step 2: Run, expect failure.**
- [ ] **Step 3: Implement.** The product is one `@generated` straight-line expression per output coefficient, a chain of `muladd`, inlined. `inv` composes the univariate series of `1/x` with `compose` (signature in Task 3, implemented here in `univariate.jl`; Task 3 tests it directly). Long tuples are built with `ntuple(f, Val(L))`, never `map` over a tuple longer than 32.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** `Add products, powers and division of Taylor numbers`.

### Task 3: Univariate composition and the Base elementary functions

**Files:** Modify `PureTaylor/src/univariate.jl`; Test `PureTaylor/test/univariate_tests.jl`

**Interfaces:**
- Consumes: Task 2 (`mul_truncated`, `compose`).
- Produces: (`compose` from Task 2, restated here) internal `compose(a::Taylor{N,K,M,T}, d::NTuple{K+1,T}, s::T)` = `Σ_k d_k (s·h)^k` with `h = a - value(a)`, by Horner's rule with degree-limited products. Base methods on `Taylor`: `sqrt`, `cbrt`, `exp`, `expm1`, `log`, `log1p`, `^(a, p::Real)`. Each computes the value with the Base function of the same name (so domain errors are Base's), then the K normalized derivatives in closed form (for `x^p`: `binomial(p, k)·x^(p-k)`; for `log`: `(-1)^(k-1)/(k·x^k)`; `log1p` as `log` at `1+x`; `exp`/`expm1`: `exp(x)/k!` for `k ≥ 1`).

- [ ] **Step 1: Failing tests.** Oracle 1: closed-form k-th derivatives in BigFloat (precision 256) for each function on `Taylor{1,4,1,BigFloat}`, tolerance `1e-70` relative (rounding bound, 256-bit). Oracle 2: a composite `f(x, y) = log1p(exp(-x)·y^(4/3)) / sqrt(x + cbrt(y))` at (0.7, 1.3), all derivatives to order 3 of `Taylor{2,3}` on Float64 against nested `ForwardDiff` on BigFloat, rounding bound in the comment. Error items (Review Focus 4): `@test_throws DomainError log(variable(Val(1), Val(2), Val(1), -1.0))`, same for `sqrt` and for `^(a, 0.5)` with a negative value. K = 0 item: `exp(constant(Val(1), Val(0), 1.0))` has value `exp(1.0)`.
- [ ] **Step 2–4:** run (fail), implement, run (pass).
- [ ] **Step 5: Commit** `Add elementary functions of Taylor numbers`.

### Task 4: Branch-free scalar kernels

**Files:** Create `PureTaylor/src/fast.jl`; Test `PureTaylor/test/fast_tests.jl`

**Interfaces:**
- Produces (public, exported): `fast_sqrt`, `fast_cbrt`, `fast_invcbrt` (`x^(-1/3)`), `fast_log`, `fast_log1p`, `fast_exp`, `fast_expm1`, with methods for `Float32` and `Float64` (branch-free kernels), and a fallback `fast_f(x::Real) = f(x)` (Base) for every other `Real` (BigFloat, ForwardDiff duals). Documented domains, each in the docstring: `fast_cbrt`, `fast_invcbrt`, `fast_log`: positive normal `x`; `fast_log1p`: `x > -1` with `1 + x` normal; `fast_exp`, `fast_expm1`: `|x| ≤ 700` (Float64), `|x| ≤ 85` (Float32); `fast_sqrt`: `x ≥ 0`. Outside the domain the result is unspecified; the caller validates.

- [ ] **Step 1: Failing tests.**
  - Accuracy, Float32: exhaustive over every Float32 in the documented domain, error in ulps against the Base function evaluated in Float64 and rounded. Accuracy, Float64: 10^7 log-uniform samples over the domain plus the domain ends, powers of two and their neighbors, and (for `log`, `log1p`, `expm1`) the region near the point where the function is zero; reference the Base function in BigFloat. The bound is a measured margin (policy (c)): write the largest error found in the comment and assert it. If any kernel exceeds 4 ulp, stop and report instead of loosening it.
  - If the exhaustive Float32 item takes more than 60 s on one thread, tag it `:exhaustive`, filter it from the main suite, and add a CI job that runs only that tag on Julia 1.13.
  - Branch-free item: for each kernel and each of Float32, Float64, the LLVM IR (`code_llvm` to a string, optimized, no debug info) contains no conditional branch (`br i1`) and no `call` other than to `@llvm.` intrinsics. This is the property the kernels exist for.
  - Float32 stays Float32: `fast_f(1.5f0) isa Float32` for each kernel.
  - Fallback: `fast_cbrt(big"2.0") == cbrt(big"2.0")`.
- [ ] **Step 2: Run, expect failure.**
- [ ] **Step 3: Implement.** Algorithms, as measured to work (≤ 2.0 ulp on random samples, vectorizing):
  - `fast_invcbrt`: initial guess from the bits, `bits(y) = M − bits(x)/3` with the division done in floating point (no 64-bit vector multiply-high is needed), `M = round((4/3)·(bias − 0.0450465)·2^(mantissa bits))` with bias 1023/127 and 52/23 mantissa bits; then 3 (Float64) or 2 (Float32) steps of `y ← y·(1 + e/3 + 2e²/9)`, `e = 1 − x·y³`.
  - `fast_cbrt(x) = c` with `c = x·y²`, `y = fast_invcbrt(x)`, plus one Newton correction `c += (x − c³)·y²/3`.
  - `fast_log`: `x = 2^k·m`, `m ∈ [√½, √2)`, `f = m − 1`, `s = f/(2 + f)`, `log m = f − s·(f − R)`, `R = Σ_{j≥1} 2 s^(2j)/(2j+1)` (9 terms Float64, 5 terms Float32), `log x = k·ln2_hi + (log m + k·ln2_lo)` with `ln2_hi` having trailing zero bits so that `k·ln2_hi` is exact.
  - `fast_log1p(x) = log(u) + (x − (u − 1))/u`, `u = 1 + x` rounded.
  - `fast_exp`, `fast_expm1`: range reduction by `k = round(x/ln2)` with a two-part `ln2`, polynomial on the remainder, scaling by `2^k` through the exponent bits; `expm1` keeps relative accuracy near 0 (choose the polynomial and its degree, and verify by the tests).
  - `fast_sqrt(x) = Base.sqrt_llvm(x)`.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** `Add branch-free scalar elementary kernels`.

### Task 5: Fast functions on Taylor numbers, and change of variables

**Files:** Modify `PureTaylor/src/fast.jl`, `PureTaylor/src/univariate.jl`; Create `PureTaylor/src/substitute.jl`; Test `PureTaylor/test/fast_taylor_tests.jl`, `PureTaylor/test/substitute_tests.jl`

**Interfaces:**
- Consumes: Tasks 3 and 4.
- Produces: `fast_*` methods on `Taylor` (value from the `fast_*` scalar kernel, derivatives as in Task 3, no domain check). `substitute(E::Taylor{N,K,M,T}, u::NTuple{N,<:Union{Taylor{N2,K,<:Any,T},T}})` (exported): the series `E`, whose variables are the increments of N quantities, evaluated with each increment replaced by the nilpotent part of `u[i]` (`u[i] − value(u[i])`); the result is a `Taylor{N2,K}` over the variables of the `u`'s. Used to compute in convenient variables (for example density and spin polarization) and convert to the caller's variables.

- [ ] **Step 1: Failing tests.** `fast_f(t)` agrees with `f(t)` on Taylor numbers inside the domain within the Task 4 bounds on the value and a rounding bound on the rest (oracle: Task 3's Base methods, which use Base's value functions). Pass-through: a function written with `fast_cbrt` and `fast_log1p` differentiated by nested `ForwardDiff` gives the same derivatives as with `cbrt` and `log1p` (fallback path; equality within a rounding bound). `substitute` oracle: for `g(p, q)` with `p = a + b`, `q = (a − b)/(a + b)`, the derivatives in (a, b) of `substitute(E_g, (p, q))` equal the derivatives of `g` computed directly in (a, b) by the Taylor type, to order 3, within a rounding bound.
- [ ] **Step 2–4:** run (fail), implement, run (pass).
- [ ] **Step 5: Commit** `Add fast functions on Taylor numbers and substitution`.

### Task 6: StrictMode policy, its gates and their failing controls; TrimCheck

**Files:** Modify `PureTaylor/src/PureTaylor.jl` (last statement `@strict_policy (:typestable, :noalloc)`); Create `PureTaylor/test/strict/workload.jl`, `PureTaylor/test/strict/check.jl`, `PureTaylor/test/strict_tests.jl`, `PureTaylor/test/trim_tests.jl`

**Interfaces:** Consumes everything above. Produces the gates; no new API.

- [ ] **Step 1: Write the items.**
  - Workload (`test/strict/workload.jl`): every exported and public function and every Base method the package defines, for `(N, K)` in `(1,0), (1,3), (2,2), (2,3), (5,2), (5,3)` and `T` in Float32, Float64, with masks that are full, single-variable and zero.
  - Gate item, in a fresh process after the workload, as PureAtomistic's "StrictMode policy holds in every module": `test_compiled(PureTaylor; guarantees = (:typestable,))` and an `audit` sweep for `(:noalloc, :noboxing)`, each required to be non-empty and to find nothing.
  - Methods on functions PureTaylor does not own (`+`, `-`, `*`, `/`, `inv`, `^`, `literal_pow`, `sqrt`, `cbrt`, `exp`, `expm1`, `log`, `log1p`, `zero`, `one`, `isfinite`, `isnan`): the released sweeps do not cover them, so `test/strict/check.jl` lists each with its guarantees `(:typestable, :noalloc, :noboxing)` and `test_signatures` (JET and AllocCheck) proves every compiled specialization from the workload. A Base method defined in `src/` and missing from the list fails the item.
  - Failing controls, one per gate, each required to report at least one finding: the same three checks run on `Taylor{2,2,0b11,Real,6}` arguments (abstract `T`; measured by the StrictMode project to fail `:typestable`, `:noalloc` and `:noboxing`). Do not use a small isbits `Union` as a control: `:typestable` accepts it by design.
  - TrimCheck `@validate` over every exported function in Float32 and Float64, tagged `:checks`.
- [ ] **Step 2: Run** the suite; fix every finding in `src/` (never by exemption). Expect pass, controls failing as required.
- [ ] **Step 3: Coverage and mutations.** Add `PureTaylor/test/coverage/` (`check.jl`, `thresholds.toml` with a row per `src` file, as PureMLIP's) and `PureTaylor/test/mutation/` with `mutations.toml` entries for the new math: the product's index arithmetic, the Horner composition, each closed-form derivative, each `fast_*` kernel's constants, and the scale handling in `derivatives`. Each mutation must be killed by the item named in its `expect`. Add `PureTaylor` to the coverage job and `mutation.yml` matrix.
- [ ] **Step 4: Run** the full suite on Julia 1.12 and 1.13 with `--check-bounds=yes` (CI's setting), the coverage check, and the mutation run. Expect all pass and every mutation killed.
- [ ] **Step 5: Commit** `Gate PureTaylor with StrictMode, TrimCheck and mutations`.

### Task 7: Documentation and repository rules

**Files:** Create `PureTaylor/docs/` (DocumenterVitepress site as PureAtomistic's, deployed under `PureTaylor/`); Modify `.github/workflows/docs.yml`, `AGENTS.md`, `README.md` if it lists the packages.

- [ ] **Step 1:** Docs: one page with the type's layout and conventions (multi-index order, normalized coefficients, masks, scaled seeds and why), the `fast_*` domains and measured accuracy, and the API reference from docstrings. Every exported name has a docstring.
- [ ] **Step 2:** AGENTS.md: the monorepo has three packages; PureTaylor's `[deps]` rule; its gates (the Task 6 items); the multi-index order and coefficient convention as a "Conventions" rule with its reason (PureBLAS builds coefficient-plane products on it).
- [ ] **Step 3:** Build the docs locally (`julia --project=PureTaylor/docs PureTaylor/docs/make.jl`), expect success.
- [ ] **Step 4: Commit** `Document PureTaylor and add its repository rules`; push; watch every CI job until green.
