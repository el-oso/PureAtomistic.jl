# Repository rules

Plain rules for anyone (human or agent) changing this repository. Each rule carries its reason.

## Scope and layout

- The repository is a monorepo, `PureAtomistic.jl`. `PureMLIP/` is the only package now; `PureAtomistic/` (the simulation layer on top of PureMLIP) is added later.
- Each package is self-contained: its own `Project.toml`, `src/`, `test/`, `docs/`. This keeps packages registrable and testable separately.
- `PureMLIP/src/checkpoint` reads mace-torch `.model` checkpoints; `PureMLIP/src/equivariant` holds spherical harmonics, Clebsch-Gordan coefficients, Wigner D matrices and the e3nn conversion layer.
- Designs live in `docs/superpowers/specs/`, implementation plans in `docs/superpowers/plans/`.

## Dependencies

- PureMLIP's dependency closure is pure Julia: no `_jll` packages, no artifacts. Reason: binaries block `juliac --trim` builds and portable executables.
- LinearAlgebra and SparseArrays are allowed. Their BLAS and SuiteSparse backends are replaced by PureBLAS and PureSparse through libblastrampoline.
- KernelAbstractions and GPU backends are allowed (later plans).
- Test-only dependencies live in `PureMLIP/test/Project.toml` and are exempt. WignerSymbols is test-only because it fails TrimCheck.
- Every dependency has a `[compat]` bound (stdlibs excepted). Julia 1.12 is the minimum.
- No Python in any form: no interpreter, no Python-backed package, no Python-installed stack. When a capability seems to need Python, use a native library through `ccall` or a command-line tool.

## Conventions

- Real spherical harmonics are the textbook ones: z is the polar axis, `m = -l..l`, orthonormal on the sphere, stored at index `l^2 + l + m + 1`.
- Real Clebsch-Gordan coefficients derive from the complex ones with the Condon-Shortley phase.
- e3nn is reached only through the conversion layer: `Y_e3nn(v) = sqrt(4π)·Y_tb(v3, v1, v2)` and `C_e3nn = s·(M'⊗M'⊗M')·C_tb` with `s = (−1)^((l1+l2−l3)/2)`. Reason: one internal convention, one place where the other is translated.
- Clebsch-Gordan and e3nn coupling are defined only for even `l1+l2+l3`; odd sums throw.

## Code rules

- Generic indexing: iterate `eachindex`/`axes`, or call `Base.require_one_based_indexing` to declare a 1-based assumption. Tests run entry points on `OffsetArray` and `view` inputs.
- No `@inbounds` without a benchmark showing the bounds check matters end to end, a local proof of the index range, and a test run under `--check-bounds=yes`. Reason: an unchecked out-of-range access is silent undefined behavior.
- Return types and struct fields are concrete; no `Any` containers. Reason: `juliac --trim=safe` and TrimCheck reject them.
- Functions ending in `!` allocate nothing; `StrictModeTest` `@test_noalloc` proves it.
- Fail fast: errors are `ArgumentError` or carry a module prefix (`"checkpoint: ..."`, `"unpickle: ..."`). No silent fallbacks.
- Use `lazy"..."` for interpolated messages with four or more pieces. Reason: it keeps error construction trim-compatible.
- Comments describe the code as it is: invariants, non-obvious reasons, constraints. No history, plans, or task references. Code ported from MIT-licensed work carries an attribution comment.

## Gates

- Test items are `@testitem`s run by TestItemRunner.
- Value items check results against oracles (see below); `OffsetArray`/`view` items check generic indexing; error items check each fail-fast message with `@test_throws "message"`.
- `TrimCheck` `@validate` proves each entry point infers fully, as `juliac --trim` requires. `StrictModeTest` `test_signatures(:typestable)` proves every public function is type-stable. `@test_noalloc` proves `!` functions allocate nothing.
- The juliac item (tag `:juliac`) builds `PureMLIP/juliac/build/bin/mace_inspect` with `--trim=safe` and runs it on MP-0b3. It runs only with `PUREMLIP_JULIAC_BUILD=1`.
- Run the suite: `PUREMLIP_CHECKPOINT_DIR=/tmp/pm_ck julia --project=PureMLIP -e 'using Pkg; Pkg.test()'`. Tests download checkpoints by URL and verify SHA-256; `PUREMLIP_CHECKPOINT_DIR` caches them.
- CI runs Julia 1.12 and 1.13 with `check_bounds: 'yes'`.
- Coverage thresholds, once a gate exists, only rise.

## Oracles and tolerances

- Oracles: e3nn-core-rs fixtures (MIT/Apache, tolerance 1e-10); llmlip.rs inspection of MP-0b3 (Apache-2.0); the Wigner-3j tables stored in MP-0b3; WignerSymbols; ForwardDiff; finite differences; quadrature orthonormality.
- A test compares against an independent oracle, not against the code under test.

### Tolerance policy

See below

## Commits and publishing

- Commits are authored by `el-oso <ursus.est@gmail.com>` (repository config); do not change it.
- Subject at most 72 characters (aim for 50); the body explains the change and its invariant for a reader with only the repository.
- Trailers: `Assisted-by: Claude <model> <noreply@anthropic.com>` and `Claude-Session: <url>`.
- Reviewed work is pushed straight to `main`; no pull requests.
- No host machine names or local absolute paths in tracked files or commit messages.
