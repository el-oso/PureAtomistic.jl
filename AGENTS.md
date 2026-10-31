# Repository rules

Plain rules for anyone (human or agent) changing this repository. Each rule carries its reason.

## Scope and layout

- The repository is a monorepo, `PureAtomistic.jl`. `PureMLIP/` is the only package now; `PureAtomistic/` (the simulation layer on top of PureMLIP) is added later.
- Each package is self-contained: its own `Project.toml`, `src/`, `test/`, `docs/`. This keeps packages registrable and testable separately.
- `PureMLIP/src/checkpoint` reads mace-torch `.model` checkpoints; `PureMLIP/src/equivariant` holds spherical harmonics, Clebsch-Gordan coefficients, Wigner D matrices and the e3nn conversion layer.
- Designs live in `docs/superpowers/specs/`, implementation plans in `docs/superpowers/plans/`.

## Dependencies

- PureMLIP's `[deps]` are Inflate and LinearAlgebra; `[compat]` requires Julia 1.12. The package adds no binary libraries of its own: no `_jll` packages, no artifacts. LinearAlgebra's BLAS backend is OpenBLAS today; it is to be replaced by PureBLAS through libblastrampoline. Reason: binaries block `juliac --trim` builds and portable executables.
- LinearAlgebra and SparseArrays are allowed. Their BLAS and SuiteSparse backends are to be replaced by PureBLAS and PureSparse through libblastrampoline. Reason: the standard libraries stay usable while the pure-Julia backends take over.
- KernelAbstractions and GPU backends are allowed (later plans). Reason: GPU execution needs no binary artifact from this repository.
- Test-only dependencies live in `PureMLIP/test/Project.toml` and are exempt. Reason: they never ship in a built program. WignerSymbols is test-only because it fails TrimCheck.
- Every dependency has a `[compat]` bound (stdlibs excepted). Reason: resolution stays reproducible and Julia 1.12 stays supported.
- No Python in any form: no interpreter, no Python-backed package, no Python-installed stack. When a capability seems to need Python, use a native library through `ccall` or a command-line tool. Reason: the repository targets standalone compiled programs with no interpreter.

## Conventions

- Real spherical harmonics are the textbook ones (reason: one standard convention, with e3nn only at the boundary): z is the polar axis, `m = -l..l`, orthonormal on the sphere, stored at index `l^2 + l + m + 1`.
- Real Clebsch-Gordan coefficients derive from the complex ones with the Condon-Shortley phase. Reason: it is the standard phase, and the tables are checkable against independent references.
- e3nn is reached only through the conversion layer: `Y_e3nn(v) = sqrt(4π)·Y_tb(v3, v1, v2)` and `C_e3nn = s·(M'⊗M'⊗M')·C_tb` with `s = (−1)^((l1+l2−l3)/2)`. Reason: one internal convention, one place where the other is translated.
- Clebsch-Gordan and e3nn coupling are defined only for even `l1+l2+l3`; odd sums throw. Reason: odd sums have no equivariant coupling of this form, so a value would be meaningless.

## Code rules

- Generic indexing: iterate `eachindex`/`axes`, or call `Base.require_one_based_indexing` to declare a 1-based assumption. Tests run entry points on `OffsetArray` and `view` inputs. Reason: callers pass views and OffsetArrays, and a silent 1-based assumption gives wrong answers.
- No `@inbounds` without a benchmark showing the bounds check matters end to end, a local proof of the index range, and a test run under `--check-bounds=yes`. Reason: an unchecked out-of-range access is silent undefined behavior.
- Return types and struct fields are concrete; no `Any` containers. Reason: `juliac --trim=safe` and TrimCheck reject them.
- Functions ending in `!` must not allocate. Reason: they sit in hot loops and must not trigger garbage collection.
- Fail fast: errors are `ArgumentError` or carry a module prefix (`"checkpoint: ..."`, `"unpickle: ..."`). No silent fallbacks. Reason: unexpected input should surface where it occurs.
- Use `lazy"..."` for interpolated messages with four or more pieces. Reason: it keeps error construction trim-compatible.
- Comments describe the code as it is: invariants, non-obvious reasons, constraints. No history, plans, or task references. Reason: a reader has only the repository. Code ported from MIT-licensed work carries an attribution comment.

## Gates

- Tests are `@testitem`s run by TestItemRunner, with shared setup in `@testmodule`s. Reason: each item can run alone.
- Value items check results against oracles (see below); `OffsetArray`/`view` items check generic indexing; error items check each fail-fast message with `@test_throws "message"`. Reason: each kind of item proves a different property, and no one kind substitutes for another.
- `TrimCheck` `@validate` and `StrictModeTest` `test_signatures(:typestable)` each cover every exported function (the exported types are not checked) plus the internal entry points `embedded_archives` and `entry_data`; the type-stability list also covers `open_torch_archive` and `unpickle`. A new exported function is added to both lists. Reason: inference failures are what `juliac --trim` rejects.
- `@test_noalloc` covers `spherical_harmonics!`, `spherical_harmonics_and_gradient!` and `e3nn_spherical_harmonics_and_gradient!`. Other `!` functions are not gated yet.
- The juliac item (tag `:juliac`) builds `PureMLIP/juliac/build/bin/mace_inspect` with `--trim=safe` and runs it on MP-0b3. It runs only with `PUREMLIP_JULIAC_BUILD=1`. Reason: it proves the whole program trims, which per-function checks cannot, but it is slow.
- Run the suite: `PUREMLIP_CHECKPOINT_DIR=<dir> julia --project=PureMLIP -e 'using Pkg; Pkg.test()'`. Tests download checkpoints by URL and verify SHA-256; `PUREMLIP_CHECKPOINT_DIR` names a local cache directory of your choice.
- CI runs Julia 1.12 and 1.13 with `check_bounds: 'yes'`.
- Coverage thresholds, once a gate exists, only rise. Reason: coverage cannot erode silently.

## Oracles and tolerances

- Oracles (reason: each is independent of the code under test): e3nn-core-rs fixtures (MIT/Apache, tolerance 1e-10); llmlip.rs inspection of MP-0b3 (Apache-2.0); the Wigner-3j tables stored in MP-0b3; WignerSymbols; ForwardDiff; finite differences; quadrature orthonormality.
- A test compares against an independent oracle, not against the code under test. Reason: a check that shares code with what it checks cannot catch that code's bugs.

### Tolerance policy

Written in a later step of the hardening plan.

## Commits and publishing

- Commits use the repository's configured git identity (`el-oso`, set in the repository's local git config); do not change it, and no host-derived address may appear. Reason: one consistent public identity.
- Subject at most 72 characters (aim for 50; GitHub truncates long subjects); the body explains the change and its invariant for a reader with only the repository.
- Trailers: `Assisted-by: Claude <model> <noreply@anthropic.com>` and `Claude-Session: <url>`. Reason: they record agent assistance and the session.
- Reviewed work is pushed straight to `main`; no pull requests. Reason: review happens before the push.
- No host machine names or local absolute paths in tracked files or commit messages. Reason: they leak private information into public history.
