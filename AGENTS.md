# Repository rules

Plain rules for anyone (human or agent) changing this repository. Each rule carries its reason.

## Scope and layout

- The repository is a monorepo, `PureAtomistic.jl`. `PureMLIP/` is the only package now; `PureAtomistic/` (the simulation layer on top of PureMLIP) is added later.
- Each package is self-contained: its own `Project.toml`, `src/`, `test/`, `docs/`. This keeps packages registrable and testable separately.
- PureMLIP has two submodules. `PureMLIP.Checkpoint` (`src/checkpoint/`, entry file `Checkpoint.jl`) reads mace-torch `.model` checkpoints. `PureMLIP.Equivariant` (`src/equivariant/`, entry file `Equivariant.jl`) holds spherical harmonics, Clebsch-Gordan coefficients, Wigner D matrices and the e3nn conversion layer. Neither submodule uses the other. Reason: the reader and the math change independently, and code built on them depends only on their interfaces.
- A submodule's interface is what it `export`s or declares `public`; everything else is its implementation. `PureMLIP` re-exports both submodules' exports and re-declares their `public` names, so `PureMLIP.x` reaches every interface name. A name is `public` because other modules or later code use it, not because a test does. Package code outside a submodule uses only its interface. Reason: an implementation detail reached from outside cannot change without breaking the code that reaches it.
- Designs live in `docs/superpowers/specs/`, implementation plans in `docs/superpowers/plans/`.

## Dependencies

- PureMLIP's `[deps]` are Inflate, LinearAlgebra and PrecompileTools; `[compat]` requires Julia 1.12. The package adds no binary libraries of its own: no `_jll` packages, no artifacts. LinearAlgebra's BLAS backend is OpenBLAS today; it is to be replaced by PureBLAS through libblastrampoline. Reason: binaries block `juliac --trim` builds and portable executables.
- LinearAlgebra and SparseArrays are allowed. Their BLAS and SuiteSparse backends are to be replaced by PureBLAS and PureSparse through libblastrampoline. Reason: the standard libraries stay usable while the pure-Julia backends take over.
- KernelAbstractions and GPU backends are allowed (later plans). Reason: GPU execution needs no binary artifact from this repository.
- Test-only dependencies live in `PureMLIP/test/Project.toml` and are exempt. Reason: they never ship in a built program. WignerSymbols is test-only because it fails TrimCheck.
- Every dependency has a `[compat]` bound. Stdlibs are excepted in the test project; the package bounds LinearAlgebra at `1.12` because Aqua's compat check requires a bound for every package dependency. Reason: resolution stays reproducible and Julia 1.12 stays supported.
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
- `test_layers` (StrictModeTest) checks the layer graph `PureMLIP => (Checkpoint, Equivariant)`, `Checkpoint => ()`, `Equivariant => ()`: no module adds a method to a function owned by a module it does not declare. Its qualified-access rule also judges reads between modules of one package (it requires StrictModeTest >= 0.5.4, which passes `allow_internal_accesses = false` to ExplicitImports). A separate item is an independent second check of the same rule: it runs ExplicitImports' `check_all_qualified_accesses_are_public` on `PureMLIP`, `Checkpoint` and `Equivariant` with `allow_internal_accesses = false` and `from = (PureMLIP,)`: package source may read only `export`ed or `public` names of PureMLIP's modules. Reads of Base, Core and other packages are not judged, and neither are test files; tests may reach internals by full path (`PureMLIP.Checkpoint.crc32`). Reason: the submodule boundaries hold only while something checks them.
- `test_calls` (StrictModeTest) sweeps each submodule after the item has called its entry points, and requires the sweep to be non-empty: no compiled signature may carry a `MethodError` or `UndefVarError` report. An exemption names one function, with its reason in the commit body. Reason: a call that cannot resolve on an untested branch passes every other gate.
- `Aqua.test_all(PureMLIP)` runs with every sub-check enabled (ambiguities, unbound type parameters, undefined exports, project comparison, stale dependencies, compat bounds, piracy, persistent tasks). Reason: these defects are mechanical to find and costly to find late.
- `@test_noalloc` covers `spherical_harmonics!`, `spherical_harmonics_and_gradient!` and `e3nn_spherical_harmonics_and_gradient!`. Other `!` functions are not gated yet.
- The juliac item (tag `:juliac`) builds `PureMLIP/juliac/build/bin/mace_inspect` with `--trim=safe` and runs it on MP-0b3. It runs only with `PUREMLIP_JULIAC_BUILD=1`. Reason: it proves the whole program trims, which per-function checks cannot, but it is slow.
- Run the suite: `PUREMLIP_CHECKPOINT_DIR=<dir> julia --project=PureMLIP -e 'using Pkg; Pkg.test()'`. Tests download checkpoints by URL and verify SHA-256; `PUREMLIP_CHECKPOINT_DIR` names a local cache directory of your choice.
- `src/precompile.jl` (included last) runs a PrecompileTools workload over the spherical-harmonics, Clebsch-Gordan, Wigner and e3nn entry points, in memory only. It leaves out `load_mace_checkpoint`: a checkpoint needs a ZIP and pickle writer, which lives in the tests, and src code may not use test code. Reason: it moves first-call compile time into package precompilation.
- The `:latency` item runs `test_latency(:PureMLIP; calls = [...])` in measure-only mode: it records load and first-call time in fresh child processes and never fails on time. Reason: wall-clock budgets depend on the host. On a pinned host, add a budget: `test_latency(:PureMLIP; calls = [...], max_load_ms = 50, max_first_call_ms = 5)`; choose the numbers from that host's own baseline. Note: `test_latency`'s timing reports first calls far below a manual `time_ns` measurement on Julia 1.13 (0.1 ms against 600 ms for `e3nn_wigner_3j(1, 2, 3)` before the workload), so a first-call budget there proves little; check against a `time_ns` measurement.
- CI runs Julia 1.12 and 1.13 with `check_bounds: 'yes'`.
- Per-file line coverage is gated: run the suite with `Pkg.test(coverage = true)`, then `julia --project=PureMLIP/test/coverage PureMLIP/test/coverage/check.jl`. It compares each `src/**/*.jl` file with its row in `PureMLIP/test/coverage/thresholds.toml` and fails on any file below its row or without one, so a new source file needs a row. CI runs it in a Julia 1.13 job. Ratchet rule: raise a row when coverage rises; lowering a row needs the user's approval. A file with no coverable lines scores 1.00 and has row 1.00, so code added to it later must be covered. Reason: coverage cannot erode silently.

## Oracles and tolerances

- Oracles (reason: each is independent of the code under test): e3nn-core-rs fixtures (MIT/Apache, tolerance 1e-10); llmlip.rs inspection of MP-0b3 (Apache-2.0); the Wigner-3j tables stored in MP-0b3; WignerSymbols; ForwardDiff; finite differences; quadrature orthonormality.
- A test compares against an independent oracle, not against the code under test. Reason: a check that shares code with what it checks cannot catch that code's bugs.

### Tolerance policy

- Every numeric comparison in a test states its oracle and a tolerance justified by exactly one of: (a) the oracle's own stated tolerance (e3nn-core-rs fixtures: 1e-10); (b) a rounding bound `c · eps(T) · N`, with the constant `c` and the operation count or condition `N` written in a comment next to the comparison; (c) a measured margin: the largest error observed, written in the comment, with the tolerance at most 100 times that and normally 10 to 20 times. Reason: a tolerance with no stated source is either too loose to catch a wrong convention or too tight to survive another platform, and nobody can tell which.
- A tolerance is never raised to make a test pass. A failure above the stated bound is investigated; the comment and the number change together only when the measurement or the bound changes. Reason: a loosened tolerance hides exactly the error the test exists to find.
- A threshold that separates two populations (the `1e-13` zero threshold of `e3nn_wigner_3j`) is justified as a measured gap: the largest value on the low side and the smallest on the high side are both written down, and the threshold lies between them. The tests assert the gap at the measured extremes. Reason: a threshold with no gap measured is a guess that fails on the next input.
- Equality `==` is used where the computation is exact by construction (copies, integer arithmetic, a value passed through unchanged). Reason: any tolerance there would only hide a defect.
- The mutation smoke check (`PureMLIP/test/mutation/`, run by hand or by the `Mutation` workflow) applies one deliberate bug at a time from `mutations.toml` to a temporary copy of `PureMLIP/`, runs the full suite, and classifies each run: killed (an item matching `expect` fails), killed elsewhere (items fail, none matching `expect`), survived (the suite passes) or invalid (the package does not load or the suite never reports). It first runs the unmutated package as a control. A survivor or an invalid run fails the check; a survivor is a missing test and gets one. Each `find` string occurs exactly once in its file, so a stale entry fails the run instead of doing nothing. A new source file gets at least one mutation, except a file that holds no computed value: one that only changes what is compiled ahead of time (`src/precompile.jl`), or only declares modules, `using`, `include`, `export` and `public` (`src/PureMLIP.jl`, `Checkpoint.jl`, `Equivariant.jl`). Reason: coverage shows that a line ran, not that a test would notice it being wrong.

## Commits and publishing

- Commits use the repository's configured git identity (`el-oso`, set in the repository's local git config); do not change it, and no host-derived address may appear. Reason: one consistent public identity.
- Subject at most 72 characters (aim for 50; GitHub truncates long subjects); the body explains the change and its invariant for a reader with only the repository.
- Trailers: `Assisted-by: Claude <model> <noreply@anthropic.com>` and `Claude-Session: <url>`. Reason: they record agent assistance and the session.
- Reviewed work is pushed straight to `main`; no pull requests. Reason: review happens before the push.
- No host machine names or local absolute paths in tracked files or commit messages. Reason: they leak private information into public history.
