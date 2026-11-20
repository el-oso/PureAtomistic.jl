# Pure-Julia exchange-correlation functionals

Status: approved by the user on 2026-11-20. Issue #7, first step of #6.

Every decision below is marked **decided by the user** (stated in issue #7 or in the list
below) or **proposed** (in this design, approved with it). Claims about speed,
accuracy and dependencies come from the two spikes described under "Measured so far"; anything
not measured says "not checked".

## Decided by the user on 2026-11-20

The work starts now, on the CPU, without waiting for GPU phases B and C of plan 6.

- Derivatives come from an own small Taylor/dual type with no dependencies.
  ForwardDiff is a test oracle only.
- The library is `PureXC/`, a third package in this monorepo.
- Thresholds are applied the same way as Libxc applies them, written down as PureXC's own
  documented rule in each spec (see "Thresholds and edge cases"). Agreement with
  Libxc must be at least 6 significant digits at every grid point above the thresholds, for
  every output the test compares (relative error ≤ 1e-6 per point), in Float64. This is a
  floor; the tolerance policy's conditioning-based bounds are tighter where they apply.
  Float32 has its own documented conditioning-based bound. Order 3 and above, which Libxc does
  not provide, is checked to the same floor against the Julia code in BigFloat.
- ρ < 0 and σ < 0 always throw; no clamp for small negatives.
- The Taylor type lives in its own small dependency-free package in this monorepo, so PureBLAS
  can support it without depending on PureXC.
- Float32 spin-polarized correlation at order 3 (relative error up to 0.62 in both spikes): the
  cause is found during implementation, before its Float32 bound is fixed. If the formula's
  conditioning is the cause, it is documented as the bound; otherwise it is fixed.
- Speed gate: code slower than Libxc is not accepted. Each functional must
  be faster than Libxc at every order Libxc provides (0–2), by the interleaved A/B method of
  AGENTS.md. Order 3 and above has no Libxc reference.
- StrictMode is a core dependency, with StrictModeTest in the test environment, as in
  PureAtomistic. Every module ends with `@strict_policy (:typestable, :noalloc)`. The only
  `@strict_exempt` entries are entry points that allocate a workspace, each narrowed to
  `:noalloc` with a present-tense reason. The gate is an item that sweeps the compiled
  specializations (`test_compiled` for `:typestable`, an `audit` sweep for `:noalloc`), as in
  PureAtomistic.

## Intent

An MIT-licensed exchange-correlation (XC) functional library in pure Julia, usable inside a
caller's grid loop and inside a GPU kernel, with derivatives of any order, so that the pure-Julia
DFT reference calculations of #6 need no binary functional library. Libxc is a test oracle
only. *(decided by the user)*

## Goals

- LDA, GGA and meta-GGA energy densities and their derivatives with respect to the density
  variables (ρ, σ, τ, and Δρ where a functional needs it), spin-restricted and spin-polarized.
  *(decided by the user: families, spin cases)*
- Generic over the number type: Float32, Float64, wider types, dual numbers. *(decided by the
  user)*
- Derivatives of any order, and gradients with respect to functional parameters, by automatic
  differentiation. *(decided by the user)*
- One code path for CPU and GPU, with KernelAbstractions as a weak dependency. *(decided by the
  user)*
- Values match Libxc to near machine precision on a dense grid, including low-density and
  spin-polarized edge cases. *(decided by the user)* "Near machine precision" is defined per
  functional as the larger of a rounding bound and the measured conditioning of the formula
  (see Validation). *(proposed)*
- Order: PBE family and LDA, then PBEsol, BLYP, revPBE, then SCAN and r2SCAN, then hybrid and
  range-separated parameters. *(decided by the user)*

## Non-goals

- Exact exchange, range-separated integrals, or VV10 nonlocal correlation integrals. The library
  reports the mixing coefficients (exact-exchange fraction, ω, α, β) and the semilocal part
  only. *(proposed)*
- Grid generation, density evaluation on grids, or the XC integration itself. The caller owns
  the grid loop. *(proposed)*
- All 711 Libxc functionals. Coverage grows by the order above, one spec per functional.
  *(proposed)*
- Kinetic-energy functionals. *(proposed)*

## Anti-goals (failure even if it works)

- Any Libxc text (C or Maple) reaching the implementer's context.
- A `_jll` or artifact in the core closure.
- A second code path for the GPU.
- Silent clamping of caller errors (negative density, NaN input).
- Agreement claimed with a tolerance that has no stated source.

## Constraints

- **License and two rooms** *(decided by the user)*: spec writer and implementer are separate
  agents; the implementer's context never holds Libxc text; specs and prompts are kept in the
  repository; Libxc runs only in a test environment.
- **Dependencies** *(decided by the user for KernelAbstractions; rest proposed)*: no `_jll`
  outside the standard libraries and no package with an `Artifacts.toml` in the core closure;
  KernelAbstractions only as a weak dependency. ForwardDiff fails this rule (see "Measured so
  far"), so it cannot be a core dependency.
- **Trim**: the CPU path passes TrimCheck and `juliac --trim=safe`. *(decided by the user:
  trim-compatible)*
- **Repository rules** (AGENTS.md): fail fast, generic indexing, no `@inbounds` without a
  benchmark, concrete return types and fields, tolerance policy, mutations for new math.
- Julia 1.12 and 1.13, matching the other packages. *(proposed)*

## API *(proposed)*

Three layers. The point layer is the contract; the other two are thin loops over it.

**1. Functional values.** Each functional is an isbits struct of its parameters, parametric in
their number type, e.g. `PBEExchange{T}(κ, μ)`, `PBECorrelation{T}(β, γ, pw::PW92{T})`.
Named constructors build the published variants (`pbe_x()`, `pbesol_x()`, `revpbe_x()`). A
composite `XC(terms...)` holds a tuple of `(coefficient, functional)` pairs and the
exact-exchange and range-separation parameters. Because a functional is isbits, it is passed by
value into a kernel and its parameters can be duals (parameter gradients).

**2. Point layer** (scalar, `@inline`, allocation-free, no throw on a valid input):

```julia
e = energy_density(f, ρ, σ)                     # GGA, spin-restricted: e = ρ ε, per volume
e = energy_density(f, (ρa, ρb), (σaa, σab, σbb)) # spin-polarized
r = xc_derivatives(f, Val(k), ρ, σ)              # NamedTuple: e, vρ, vσ, v2ρ2, v2ρσ, ...
```

LDA drops σ; meta-GGA adds τ (and Δρ where needed). Inputs are plain numbers or tuples, so the
caller's loop or kernel holds them in registers. Output names and the order of spin components
follow Libxc's documented layout (`vrho`, `vsigma`, `v2rho2`, `v2rhosigma`, ... with σ ordered
αα, αβ, ββ), so the oracle comparison and users moving from Libxc need no remapping. The energy
is per volume (e = ρε), as in DftFunctionals; `zk = e / ρ` is a helper.

**3. Array layer:**

```julia
xc_evaluate!(out, f, Val(k), ρ, σ)   # out: NamedTuple of arrays, axes matching ρ
```

It validates inputs first (see Thresholds), then loops over `eachindex(ρ, σ)` calling the point
layer. The loop is written so that LLVM vectorizes it across points (see Derivatives); the
spike measured this as the largest single speed gain. A fast method for dense, contiguous
arrays may read and write through raw pointers under `@simd ivdep`, after checking that no
output array shares memory with another argument; it is kept only with an interleaved
benchmark against the generic `eachindex` loop that shows the gain, and with the test suite
run under `--check-bounds=yes`. Every other array type (views, `OffsetArray`) takes the
generic loop, and items check that both give the same results. The KernelAbstractions extension adds a method for device arrays that launches one
work-item per point running the same point function, and reports caller errors through a status
word read once after the launch (the pattern used in PureMLIPGPUExt).

A caller who fuses XC into its own loop (density, XC, quadrature weight in one pass) calls the
point layer directly; nothing in the array layer is required.

## Derivatives *(proposed)*

Each functional implements one generic scalar function, its energy density. All derivatives
come from it by forward-mode AD with an own truncated multivariate Taylor type *(decided by
the user: own type, no dependencies)*. Any order follows from the same code. Parameter
gradients come from the caller passing a functional whose parameters are Taylor numbers.

The type *(proposed; each point below was measured in the second spike)*:

- `Taylor{N,K,T,L} <: Real`: N variables, order K, the `L = binomial(N+K, K)` normalized
  coefficients ∂^α f/α! stored once as `NTuple{L,T}`. Order 2 in n variables carries
  n(n+1)/2 second-order coefficients, not the n² of nested duals. It is isbits, and Float32
  stays Float32.
- Products are `@generated` straight-line `muladd` code and must be inlined
  (`@inline` on the generated body): out of line, tuples pass through memory and the type lost
  to nested ForwardDiff.
- Each value records in its type which inputs it depends on, so a function of ρ alone stays a
  one-variable series. Spin-polarized correlation is computed in (ρ, ζ, σ) and converted.
- Univariate functions (`inv`, `sqrt`, `cbrt`, `x^(-1/3)`, `^`, `exp`, `expm1`, `log`, `log1p`)
  compute their derivatives once and apply them to the nilpotent part by Horner's rule.
- Seeds are scaled: the expansion is in u = x/x0 and is divided by x0 at the end. Without this,
  the third σ-derivative at ρ = 1e-6 overflows Float32 (an intermediate of about 3.6e41) though
  the result fits.
- The elementary functions are own branch-free pure-Julia kernels (cbrt, x^(-1/3), log,
  log1p, expm1, exp): Base's versions branch and do not inline, which stopped the point loop
  from vectorizing. Measured error: at most 2.0 ulp on random samples in both precisions; each
  kernel's input domain is stated and tested (exhaustively in Float32).
- Exchange uses the exact spin scaling and the split ρ^(4/3)·F(s) wherever a functional
  factorizes.

ForwardDiff duals still pass through the generic point functions without a dependency;
ForwardDiff remains a test oracle. Nested duals and TaylorDiff were measured and rejected
(slower; see "Measured so far").

The Taylor type's public interface is sent to the PureBLAS project, which plans a matching
coefficient-plane gemm.

## Thresholds and edge cases *(proposed)*

Two kinds, kept apart:

- **Caller errors throw** (`ArgumentError`, prefix `"xc: "`): ρ < 0, σ < 0, τ < 0, a NaN or
  infinite input, a spin-polarized σαβ with σαβ² > σαα σββ beyond a rounding bound, mismatched
  axes. The array layer checks before computing; the device path sets the status word and the
  host throws after the launch. Libxc returns 0 for ρ = −1 and a finite value for σ = −1
  (measured), so this is a deliberate difference, and the oracle tests feed it only valid inputs.
- **Physical limits are documented behavior, per functional spec,** applied the same way as
  Libxc applies them *(decided by the user)*, so the oracle comparison holds up to the
  thresholds: total density below the density threshold gives zero energy and zero
  derivatives; otherwise each spin channel is raised to the density threshold; σ is raised to
  the square of the reported σ threshold; a factor (1 ± ζ)^p with 1 ± ζ ≤ ε_ζ becomes ε_ζ^p;
  derivatives at a raised value are those at the floor; for meta-GGAs τ is raised to the von
  Weizsäcker bound. Each spec's "PureXC rule" section gives the exact rule for its functional.
  Thresholds are parameters of the functional, typed in its number type, with defaults taken
  from Libxc's public API values (density 1e-15 for PBE exchange and 1e-12 for PBE
  correlation; σ 1e-20 and 1e-16; ζ 2.2e-16 = eps(Float64), measured) and scaled for Float32.
  Libxc's clamp of σαβ is not copied: PureXC throws for σαβ² > σαα σββ beyond a rounding bound.

Measured reason this matters: at a fully polarized point (ρβ = 0) the spike's naive formulas
give NaN for every derivative and for the exchange energy; Libxc returns finite values. At ρ = 0
the spike gives NaN; Libxc gives 0.

## Validation against Libxc *(proposed)*

- Test environment only: `<pkg>/test/libxc/Project.toml` holds Libxc.jl; the core and main test
  environments never load it. CI runs it as its own job.
- Grid per functional: ρ log-spaced over [1e-10, 1e3] (131 points), reduced gradient s in
  {0} ∪ log-spaced [1e-3, 10], τ over [τ_W, 10 τ_W] for meta-GGAs, ζ in a set including ±(1 − ε_ζ)
  and the fully polarized points. Each point is labeled above or below the thresholds, and the
  two populations are reported separately.
- Orders 0–2 against Libxc's outputs. Order 3 and higher: Libxc_jll 7.0.0 is built without
  third derivatives (`supported derivatives [0, 1, 2]`, measured), so order 3 is checked
  against the same Julia code in 256-bit BigFloat, to the 6-digit floor *(decided by the
  user)*, and loosely against Richardson central differences of Libxc's order 2 (themselves
  accurate only to about 1e-6 to 1e-4). No Libxc build with third
  derivatives is made.
- Tolerances follow the repository's tolerance policy, with the BigFloat evaluation as the
  measure of conditioning: a point's tolerance is a stated multiple of the larger of the Julia
  and Libxc errors against BigFloat. Measured: PBE correlation agrees with Libxc only to 4e-12
  in Float64 because both codes are 2–3e-12 from the BigFloat value at s = 10 (ε_c + H is 1e4
  times smaller than ε_c), so a flat 1e-14 would fail on correct code.
- Generic-indexing items (OffsetArray, view inputs), Float32 items (against Float64 results
  with a conditioning-based bound), TrimCheck, allocation and type-stability gates as in the
  other packages.

## Two-room spec process *(proposed beyond the issue's four points)*

- Location: `<pkg>/specs/<family>/<libxc_id>.md` (for example `specs/gga/gga_c_pbe.md`), and
  the agent prompts in `<pkg>/specs/prompts/` (`spec_writer.md`, `implementer.md`,
  `reviewer.md`).
- Spec format, one file per functional:
  1. identifier, family, kind, spin cases;
  2. references (DOI, equation numbers);
  3. variables and units;
  4. equations, each with its paper and equation number;
  5. constants, each with its value, source, and one of "paper" or "checked against Libxc";
  6. building blocks used (other specs, e.g. `lda_c_pw_mod`);
  7. domain, thresholds and limits;
  8. known misprints or ambiguities, with how they are resolved;
  9. Libxc oracle identifier and the parameter values its public API reports.
- The spec writer reads papers first and Libxc's public documentation, API and outputs; it opens
  Libxc's source only for what those leave open, and writes facts only (equations and numbers),
  never code. The implementer reads specs and papers only. A reviewer agent with the same
  restriction as the implementer checks code against the spec. *(proposed)*
- Measured why "checked against Libxc" is needed even for PBE: with the constants printed in
  the PBE and PW92 papers (β = 0.066725, μ = 0.21951, PW92 Table I) the energy differs from Libxc
  by up to 3.3e-6 (exchange) and 3.6e-5 (correlation) relative. Libxc's API reports β =
  0.06672455060314922 and μ = βπ²/3, and its PBE correlation matches its `lda_c_pw_mod`
  (A = 0.0310907, 0.01554535, 0.0168869; f''(0) = 4/(9(2^(1/3) − 1))) to 1e-16, not
  `lda_c_pw` (paper values). These came from Libxc's public ext-params API and outputs only.

## First functional set *(order decided by the user; split proposed)*

1. `lda_x`, `lda_c_pw`, `lda_c_pw_mod` (the variant PBE uses).
2. `gga_x_pbe`, `gga_c_pbe`, spin-restricted and spin-polarized.
3. PBE variants by parameters: `gga_x_pbe_sol`, `gga_c_pbe_sol`, `gga_x_pbe_r` (revPBE).
4. BLYP: `gga_x_b88`, `gga_c_lyp`.
5. `mgga_x_scan`, `mgga_c_scan`, `mgga_x_r2scan`, `mgga_c_r2scan`.
6. Hybrids: PBE0, B3LYP (with its VWN variant), HSE06 parameters.

Each step: spec, implementation, oracle grid, speed check against Libxc at orders 0–2.

## Extend DftFunctionals.jl or build a separate package *(proposed: separate package)*

DftFunctionals.jl 0.3.3 (MIT, JuliaMolSim) was read. It covers LDA exchange, PW correlation and
PBE exchange and correlation and their parameter variants; spin-polarized GGAs are not
implemented ("Multiple spins not yet implemented for fallback functionals"); its array API
allocates per point (views, `ComponentArray` parameters); on the GPU it copies to the CPU and
back; its thresholds are Float64 constants marked TODO.

Reasons for a separate package:

1. Dependency rule: DftFunctionals depends on ForwardDiff, whose closure holds `OpenSpecFun_jll`
   (measured), and on ComponentArrays. Upstream would have to accept dropping both.
2. The point layer, isbits functionals and the device path are a different core design, not an
   extension; upstreaming means rewriting most of its 700 lines against its users' (DFTK's)
   compatibility.
3. The two-room record (specs and prompts in the repository) is a process upstream has not
   agreed to.

What is kept from DftFunctionals: its names where they fit (`energy_density`, `family`, `kind`,
parameters as values) and an optional weak-dependency extension that wraps our functionals as
`DftFunctionals.Functional`s, so DFTK users can try them. Contributing back fixes or offering the
package to JuliaMolSim later stays possible. Code ported from DftFunctionals carries an
attribution comment.

## Package location and dependency closure *(proposed)*

- A third package in this monorepo, `PureXC/` *(decided by the user)*, with its own
  `Project.toml`, `src/`, `test/`, `docs/`, registrable separately. It depends on nothing from
  PureMLIP or PureAtomistic, and AGENTS.md's layout section gains its rules. The Taylor type
  lives in its own dependency-free package in this monorepo, `PureTaylor/`, which PureXC
  depends on *(decided by the user: own package)*.
- `[deps]`: StrictMode *(decided by the user)*; nothing else outside the standard libraries
  (own Taylor type; constants in Julia source). `[weakdeps]`: KernelAbstractions (device array layer, `[compat]` 0.10, with the
  macOS `HWLOC_COMPONENTS=-darwin` note), DftFunctionals (wrapper).
- Test environments: main (TestItemRunner, OffsetArrays, ForwardDiff as an oracle, TrimCheck,
  StrictModeTest, Aqua); `test/libxc/` (Libxc.jl); `test/gpu/<backend>/`.
- An item "core closure is binary-free" as in PureMLIP.

## Measured so far

### Second spike: own Taylor type, vectorized across points

A throwaway spike outside the repository (same machine and versions as the first spike below;
one thread; 10^6 points; 7 interleaved rounds with alternating arm order; speed-up = Libxc
Float64 time / our time, median [min, max]; the Float32 column also divides by Libxc's Float64
time):

| case | order | Libxc ns/pt | F64 | F32 |
|---|---|---|---|---|
| restricted x | 0 | 5.89 | 5.00 [4.96, 5.23] | 12.99 [12.85, 13.42] |
| | 1 | 8.56 | 2.86 [2.83, 2.90] | 6.94 [6.65, 6.97] |
| | 2 | 12.56 | 2.43 [2.38, 2.44] | 5.42 [5.31, 5.46] |
| restricted c | 0 | 37.8 | 3.63 [3.60, 3.64] | 11.00 [10.93, 11.07] |
| | 1 | 48.6 | 2.91 [2.89, 2.92] | 7.11 [7.08, 7.14] |
| | 2 | 70.2 | 3.08 [3.06, 3.09] | 7.70 [7.67, 7.73] |
| polarized x | 0 | 13.05 | 5.02 [4.98, 5.10] | 12.63 [12.56, 13.01] |
| | 1 | 20.54 | 3.30 [3.28, 3.32] | 7.84 [7.56, 7.91] |
| | 2 | 42.96 | 2.00 [2.00, 2.01] | 3.91 [3.90, 3.91] |
| polarized c | 0 | 48.0 | 2.71 [2.65, 2.73] | 7.12 [7.06, 7.16] |
| | 1 | 69.2 | 2.42 [2.42, 2.43] | 5.76 [5.75, 5.78] |
| | 2 | 162.9 | 3.21 [3.20, 3.21] | 7.51 [7.46, 7.60] |

Order 3, against the best Taylor arm without vectorization, F64: restricted x 4.62, restricted
c 5.31, polarized x 1.53, polarized c 16.5.

Float64 max relative error against BigFloat, orders 0/1/2/3: restricted x 8.5e-16, 1.8e-15,
1.1e-13, 5.6e-14; restricted c 4.0e-12, 1.4e-12, 3.6e-12, 9.0e-11; polarized x 8.2e-16,
2.3e-15, 4.5e-14, 3.9e-14; polarized c 7.9e-12, 1.9e-12, 3.0e-12, 1.5e-9. Float32: exchange at
most 6.6e-5; correlation at most 7.4e-3 at orders 0–2; polarized correlation at order 3 reaches
0.62 (cause not examined). 14 Float32 order-3 values are infinite, all in polarized exchange,
all with an exact value beyond `floatmax(Float32)`.

Gains, each measured separately: the point loop vectorized (`<2 x double>` in the IR) once the
kernel was inlined, closures no longer held bounds-check throws, and the branch-free elementary
functions replaced Base's (restricted x 2.77 / 7.09 / 11.80 → 1.16 / 2.88 / 5.15 ns at orders
0 / 1 / 2; own cbrt 0.87 ns against Base 3.84); `@inline` on the generated products (restricted
c order 2, 64 → 23 ns); `@simd ivdep` (polarized x order 2, 43 → 21 ns); raw pointers taken
before the loop at order 3, where 56 outputs made `map` build a `Vector{Any}` (137 → 67 ns);
dependency tracking in the type (polarized c order 2, 141 → 45 ns; not interleaved).

Not checked in the second spike, and each can lower the speed-ups above: input validation and
thresholds (the spike applies none); the share of Libxc's time spent in the Libxc.jl wrapper;
the different output (ρ·zk rather than zk; one vector per derivative rather than per-point
matrices); `--check-bounds=yes`; Julia 1.12; other machines; native code (only LLVM IR was
read). The speed gate is therefore judged again on the package, with validation and thresholds
in place.

### First spike: nested ForwardDiff

A throwaway spike outside the repository (Julia 1.13.4, ForwardDiff 1.5.0, Libxc.jl 0.3.27 over
Libxc 7.0.0, Apple M6, one thread) implemented PBE exchange and correlation from the papers,
spin-restricted and spin-polarized, with derivatives by nested ForwardDiff duals.

Accuracy, Float64, max relative error against Libxc, using Libxc's API constants, points above
Libxc's thresholds with s > 0 (order 3 against Richardson differences of Libxc's order 2, whose
own accuracy is about 1e-6 to 1e-4):

| | order 0 | order 1 | order 2 | order 3 |
|---|---|---|---|---|
| PBE x | 6.6e-16 | 1.7e-15 | 1.3e-13 | 7.5e-6 |
| PBE c | 4.3e-12 | 2.4e-12 | 5.5e-11 | 1.3e-4 |
| PBE x, Julia vs BigFloat | 5.8e-16 | 8.6e-16 | 9.0e-14 | 3.4e-14 |
| PBE c, Julia vs BigFloat | 2.5e-12 | 1.9e-12 | 2.5e-12 | 1.2e-11 |

Spin-polarized orders 0–2: exchange ≤ 4.8e-15, correlation ≤ 6.2e-12 (v2rho2 1.8e-10).
Float32 (ρ ≥ 1e-6): exchange 2.5e-7 to 5.0e-5; correlation up to 2.3e-3, from the s = 10
cancellation (energy 7.2e-7 for s ≤ 1).

Speed, ns per point, 10^6 points, 7 interleaved rounds (arm order alternating), speed-up =
Libxc time / Julia time, median [min, max]; Libxc runs in Float64 only:

| functional | order | Libxc | Julia F64 | Julia F32 | speed-up F64 | speed-up F32 |
|---|---|---|---|---|---|---|
| PBE x | 0 | 5.8 | 4.5 | 3.4 | 1.28 [1.28, 1.29] | 1.68 [1.67, 1.69] |
| PBE x | 1 | 8.4 | 7.9 | 6.6 | 1.07 [1.07, 1.08] | 1.28 [1.27, 1.28] |
| PBE x | 2 | 12.4 | 27.3 | 24.0 | 0.45 [0.45, 0.46] | 0.51 [0.51, 0.53] |
| PBE x | 3 | n/a | 103.1 | 88.6 | n/a | n/a |
| PBE c | 0 | 37.2 | 33.8 | 26.6 | 1.10 [1.10, 1.10] | 1.40 [1.39, 1.40] |
| PBE c | 1 | 48.0 | 43.8 | 39.0 | 1.09 [1.09, 1.10] | 1.23 [1.23, 1.24] |
| PBE c | 2 | 69.5 | 84.2 | 78.5 | 0.83 [0.81, 0.83] | 0.89 [0.87, 0.89] |
| PBE c | 3 | n/a | 250.7 | 246.5 | n/a | n/a |

TaylorDiff (directional series, polarization), 2·10^5 points: order 2 / 3 took 50.2 / 100.8
ns (x) and 222.9 / 372.0 ns (c) against nested 24.9 / 97.6 and 83.0 / 245.0, with errors
against BigFloat of 1e-11 to 4e-10.

Dependency closures, fresh environments: ForwardDiff 1.5.0 (27 packages), DftFunctionals 0.3.3
(43) and TaylorDiff 0.3.6 (45) each contain `OpenSpecFun_jll`, which is not a standard library
and has an `Artifacts.toml`.

Not checked: nested duals inside a KernelAbstractions kernel on Metal (Float32) and the CPU
backend; TrimCheck on the nested-dual path; the cost of a home-made dual type against
ForwardDiff's.

## Open questions

None. The last two were answered on 2026-11-20 and are in "Decided by the user".
