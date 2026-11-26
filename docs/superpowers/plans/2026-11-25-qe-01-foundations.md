# QuantumEngine step 01 (foundations) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the `QuantumEngine/` sub-package of the PureAtomistic monorepo with every repository gate in place from the first task. It holds the core types, one units and element table, the input grammar (`.inp`, JSON, any `AbstractDict`, YAML files through a package extension, `.xyz/.pdb/.cif`), molecule and Molden IO, the logger, the error types and the parity harness against the Rust oracle `quantum_engine` (commit 4a78a1d). Nothing from steps 02–17 is in scope.

**Architecture:**
- Modules, one directory each with entry file `<Name>.jl`: `Errors`, `Units`, `System`, `IO` and `Input`.
- Layer graph: `Errors → ()`, `Units → ()`, `System → (Units,)`, `IO → (System, Units, Errors)`, `Input → (IO, System, Units, Errors)`.
- The `.inp` parser is one line-driven pass with a `lenient` flag. Each place where the oracle silently accepts something is a single branch: strict mode throws an `InputError`, lenient mode records an `InputWarning` and takes the oracle's value.
- JSON text and any `AbstractDict` (a YAML file arrives as one, read by YAML.jl in `QuantumEngineYAMLExt`) are converted to one concrete tree type (`Input.Node`) at once. A single schema walker reads every structured format.
- The parity harness lives in `test/oracle/`. CI runs against stored references; the live `:oracle` items run only when a binary is named.

**Tech Stack:**
- Julia 1.13; StaticArrays 1.9 (SVector); JSON.jl 1.6; StrictMode 0.5.21; Printf (stdlib). YAML.jl is a `[weakdeps]` entry, loaded only by the extension `QuantumEngineYAMLExt` (D1).
- Tests: TestItems/TestItemRunner, StrictModeTest, TrimCheck 0.1.5, Aqua, ExplicitImports, OffsetArrays.
- Docs: Documenter and DocumenterVitepress.
- juliac: through PureMLIP's tool environment.

**Spec:** `01-foundations.md` of the QuantumEngine rewrite specs (with `00-overview.md`, `PLAN.md` and the Decisions section of `99-open-questions.md`). They live outside the repository today (decision D19), so the executing coordinator hands them to each implementer and reviewer with this plan. The oracle checkout is named by the environment variable `QUANTUM_ENGINE_ORACLE_SRC` in every task that reads it. It is read-only: never modified, never built.

**Evidence.** The research behind this plan exists only as probe scripts in a session scratch directory, which is not durable. Every figure and probe outcome the tasks rely on is therefore written out in the appendix "Evidence" at the end of this plan, with the command or source line it came from. A task that needs a figure not in the appendix measures it again and records it.

---

## Decisions needed

These rows change a literal requirement of the specs or of the user's standing rules, so the user decides them; the plan does not. Each names the task it blocks. **No blocked task starts until the user has answered.** The default stated is what the plan implements if the user confirms it.

| # | Question | Default offered | Blocks |
|---|---|---|---|
| D1 | Q-YAML. **Decided by the user (2026-11-25):** "don't worry about YAML.jl. It technically reads the .yaml file and converts it to a Dict, other packages do the same. Just worry what to do with a read dict." | The core reads structured input from an `AbstractDict` (`node(::AbstractDict)` into Task 10's `Node`, then `parse_structured`). Reading a `.yaml` file into a Dict is done by YAML.jl in a package extension (`QuantumEngineYAMLExt`, YAML.jl a `[weakdeps]` entry), so its closure stays out of the core. Task 11 implements this. | none |
| D2 | JSON compat and route. The user's rule is JSON.jl "pinned to `"1.6"`" (spec 01), i.e. compat `JSON = "1.6"`. That admits 1.6.1 … 1.13.4. `JSON.lazy`, `JSON.lazyfile` and `JSON.LazyValue` are declared `public` in JSON 1.6.1 (`src/JSON.jl:14-23`); `JSON.applyobject`/`applyarray` are not, and their key conversion differs between 1.6.1 and 1.13.4 (appendix E4). | Compat `"1.6"` and a public route only: either `JSON.parse(text; dicttype = T)` with an `AbstractDict` type `T` whose `setindex!` throws on a repeated key, or a walk over `LazyValue` through public calls. Task 10 Step 0 measures both routes (duplicate detection, the 1.6.1 and 1.13.4 resolutions, TrimCheck) and takes the first that passes. If neither passes, Task 10 stops and asks; the alternative it then offers is compat `"~1.6"` plus the internal `applyobject` walk. | None while a public route passes (the default follows the user's rule literally); 10 from Step 1 on if neither does |
| D28 | Spec 01's acceptance test says the `631-g` typo in `examples/h2o_opt.inp` must fail with a specific message. The oracle accepts `631-g` as a 6-31G alias (`basis/reader.rs:62`: `"631-g" \| "631g" \| "6-31-g" \| "6-31g" => "6-31g"`). | **Decided by the user (2026-11-25): accept as alias.** It parses (`basis == "631-g"`), and alias resolution happens in step 02. The spec text is the user's to correct, as D18 does for spec 02. | none |
| D29 | Default method. Spec 01 says "same keyword set and defaults as the oracle" and names `method rhf` as a default; the oracle's default is `Method::RHF` (`types.rs:236`), which it resolves to RHF or UHF by multiplicity. | **Decided by the user (2026-11-25): the oracle's automatic choice.** `MethodSpec(HF, AutoReference, (0,0), 0.0)`, so an omitted `method` behaves as the oracle does. The alternative, `Restricted`, makes every open-shell input without `method` an error under the "rhf with multiplicity ≠ 1" rule. | none |
| D30 | Spec 01 says the six fixture files are checked "against a table extracted once from the Rust parser". Agents may not build the oracle, so the table is read from `parser.rs` by hand, with the line cited per field; Task 13 adds the six fixture files to the user's capture, and the echoed settings the oracle prints (`Title:`, `Method:`, `Basis Set:`, `Multiplicity:`, `Charge:`, `Convergence:`, `Atoms:`) are cross-checked against the table. | **Decided by the user (2026-11-25): hand-read plus cross-check.** The hand-read table plus the run-derived cross-check of the echoed fields. | none |

---

## Decisions this plan assumes

The research for this plan raised questions that the spec and the 2026-11-25 decisions leave open. In each row below, the plan implements the stated default. **No task that a row blocks starts until the user has confirmed or changed that row.** Every other row is confirmed at plan review. A changed row changes the named task before it is executed.

| # | Question | Default this plan implements | Blocks |
|---|---|---|---|
| D1 | Q-YAML: decided by the user; see "Decisions needed" and Task 11. YAML.jl (any version) is a weak dependency read in `QuantumEngineYAMLExt`, outside the core closure. | Task 11 | none |
| D2 | JSON compat and route. | **Under "Decisions needed".** | 10 |
| D3 | JSON duplicate keys. `JSON.parse` keeps the last value silently by default. The oracle reads `.json` through yaml-rust2 0.9, which rejects a repeated key with `"…: duplicated key in mapping"` (`yaml-rust2-0.9.0/src/yaml.rs:223-228`), so rejecting duplicates matches the oracle in both modes and needs no lenient case. | Detect duplicates through the D2 route and reject them in both modes. | 10 |
| D4 | JSON reading and `--trim`. `JSON.parse` fails TrimCheck (`repr(::Symbol)` at JSON `parse.jl:191`), and the trim-safety of the route chosen under D2 is not measured. | Task 10 measures the chosen route with TrimCheck. If it fails, the JSON front end stays outside the trimmed entry (`juliac/entry.jl` reads `.inp` only), and the choice for the CLI goes to step 17. | 10 |
| D5 | JSON/YAML integer given for a float key (`"scf_convergence": 1`). The oracle ignores it silently; JSON.jl converts it. | Strict accepts it and converts to Float64. Lenient takes the oracle's value (the key is ignored and the default kept) and warns. | 10 |
| D6 | The scope of `--lenient` under Q-LENIENT ("unknown keywords and bad values are errors; `--lenient` accepts them as the oracle does"). | **Class A** (unknown keywords, bad values, and the order-dependent results of the `:order_dependent` rows; the per-keyword table and the case table of Task 9 are the complete list) reproduces the oracle's value with a warning. **Oracle errors** (the keywords the oracle reads with `ok_or(...)?` or `return Err`, listed in Task 9) stay errors in both modes. **Class B** (the oracle misreads valid input) is read correctly in both modes. Class B is: the `cell`/`pbc` substring scans; the xyz blank-comment line; fragments block lines replacing each other; geometry-unit leakage between blocks, including a `charges` block, which takes an optional unit defaulting to Å while the oracle uses the last geometry block's unit (`parser.rs:331-336, 607-610`); YAML/JSON `unit: au` and `unit: ang`, which mean bohr and Å while the oracle tests only for `"bohr"` and reads every other value as Å (`parser.rs:1028`); lowercase `bq` ghosts; and quotes kept in `title`. Class B is not reproduced in either mode. | 8, 9, 10 |
| D7 | A geometry block without `end`. Five oracle test inputs rely on it. | Strict requires `end` (those five files parse only with `--lenient`). | 8 |
| D8 | Booleans. | Strict accepts exactly `true` and `false`, and a bare boolean keyword is an error. This includes the three values of the `pbc` block: strict rejects `1`, `0`, `T` and `F` there. Lenient takes each keyword's oracle fallback (Task 9 table); in `pbc` that is `true`, `1` or `T` → true and any other token → false (`parser.rs:649-653`). | 8 |
| D9 | Lexical extras. | In both modes, a token that starts with `#` ends the line (inline comment), and one pair of surrounding double quotes is removed from the `title` value. Strict rejects extra tokens. | 8 |
| D10 | Dead keywords: `diis_method` (copied into `ScfOptions`, never compared), `smearing`, `ts_target_mode`, `ts_trust_radius`, and `population`/`pop_analysis` (one field, `parser.rs:560`, with no consumer outside `src/input/` and the GUI). | Parse and store them with vocabulary checks (`diis_method ∈ {cdiis, ediis}`; each `population` entry ∈ {mulliken, lowdin}). Whether anything consumes them is decided by the later steps. | 7 |
| D11 | `ccsd(t)`/`uccsd(t)`. The oracle maps them to plain CCSD. | A distinct `CCSD_T` method kind, so the request is not relabeled silently. | 7 |
| D12 | `NotImplemented` clashes by name with TypeContracts 0.14.7's exported trait type. | Keep the spec's name, import TypeContracts only explicitly, and test that the two are different bindings. The alternative is to rename it `NotImplementedError`. | 1 |
| D13 | Error message prefixes (AGENTS.md: an `ArgumentError` or a module prefix). | `input:`, `basis:`, `scf:`, `optimization:`, `not implemented (<spec section>):`, `functional:`, `numerical:` are printed by `showerror` for the error types. `units:`, `system:`, `io:` and `molden:` (for the Molden data-type constructors) prefix `ArgumentError`s. The file readers throw `InputError`s whose `msg` starts with the format: `xyz: `, `pdb: `, `cif: ` or `molden: `, printed as `input: <source>:<line>: xyz: expected the atom count`. The line number lives only in the `line` field, never inside `msg`. A reader checks the symbol before it constructs the atom: it calls `System.element_number(symbol)`, which returns 0 for an unknown symbol, and then throws (or, in lenient mode, warns) with its own message `<fmt>: unknown element symbol "Xx"`, the text of `System`'s message with `system: ` replaced by the format, so line 3 of an xyz file prints `input: <source>:3: xyz: unknown element symbol "Xx"`. The `ArgumentError` that `element_number` throws for a symbol that names no element (a bare `Bq`) is caught and rethrown as `InputError(source, line, "<fmt>: " * <the message without its "system: " prefix>)` in both modes. | 1, 4, 6 |
| D14 | TypeContracts, `@verify_all` and the PureXC path dependency. Step 01 has no `@contract` and no XC call, and Aqua's stale-dependency check fails an unused `[deps]` entry. | Each enters with its first use: TypeContracts with the first `@contract` (step 03/04), PureXC at step 05. The contract test file is created then. | 1 |
| D15 | Default `Atom.mass`. The oracle's `.inp` table mixes isotopic (Z ≤ 10) and average masses; its YAML table uses average masses. | Mass of the most abundant isotope (the vibrational-frequency convention). Step 08 states the oracle difference in its parity table. | 3 |
| D16 | The oracle inputs in CI. The oracle declares Apache-2.0 in `Cargo.toml` and has no LICENSE file. | Vendor the input files verbatim into `QuantumEngine/test/oracle/inputs/` with the standard Apache-2.0 text and a README naming the source repository and commit (Task 1b, so that every later test reads them from the repository). | 1b |
| D17 | Who provides oracle outputs. | The user runs `test/oracle/capture.jl` with their binary. Agents never build the oracle; an agent runs an existing binary only when the user names it and asks. The captured logs are stored in `QuantumEngine/test/oracle/captured/` and committed, so CI parses them without the binary. | 13 |
| D18 | The Molden writer. | Full precision (shortest round-trip), `[Atoms] AU`, alpha and beta, and an error for l > 4. The spec 02 sentence "not the Molden order" is wrong for l = 2, 3, 4 (the oracle order equals the Molden document's); the spec text is the user's to correct. | 6 |
| D19 | Where the rewrite specs live. | They stay outside the repository; this plan names them. Alternative: commit them to `docs/superpowers/specs/`. | — |
| D20 | Branch. | Commits go on `qe-foundations`. Merging to `main` and pushing are for the coordinator, after the one-line notice to the PureAtomistic session (shared files: the three workflows, AGENTS.md, README.md, the root `.gitignore`, `docs/superpowers/`). This plan never pushes. | — |
| D21 | Keywords whose feature is not built yet. Spec 01 says they parse and then throw `NotImplemented`; Q-SCOPE/Q-STUBS say nothing is dropped. | Step 01 parses every keyword into `InputFile`. The run-mode driver of later steps throws `NotImplemented` for a feature that is not built yet. | 7 |
| D22 | Lattice and constraint units. | As the oracle: `cell`/`lattice` vectors are Ångström whatever the geometry unit, and `freeze_bond` targets are bohr. Both are stated in the docstrings. | 7, 8 |
| D23 | Mutation entries in step 01. AGENTS.md requires them for new math only. | Entries for the unit conversions, `nuclear_repulsion`, the electron-count and parity rules, and the Rust exponent formatter. None for the parsers. | 2, 3, 5 |
| D24 | Stale AGENTS.md lines 24 (StrictMode floor) and 68 (modules without a policy). | Not fixed here; they are named in the notice to their owner. | — |
| D25 | The PDB element columns. | Strict requires columns 77–78. Lenient takes the oracle's fallback (the first character of the atom name). | 4 |
| D26 | A file-type mismatch (YAML in `.inp`, `.inp` syntax in `.yaml`). | Strict names the mismatch. Lenient fails with the oracle's "no atoms" outcome. | 8, 11 |
| D27 | Duplicate scalar keywords (`charge` twice). | Strict error naming both lines; lenient keeps the last value and warns. The `freeze_*` keywords are repeatable. | 8 |
| D31 | `ecp <arg>`. The oracle reads a non-boolean argument as the ECP name (`use_ecp = true`, `ecp_name = arg`), and `ecp_name` also sets `use_ecp = true` (`parser.rs:489-504`). | Strict accepts `ecp true`, `ecp false` and `ecp <name>` (the name form sets both fields, as the oracle does); `ecp_name <name>` sets `use_ecp = true`. A bare `ecp` is `:bool_fallback` (lenient: true). | 7, 8 |
| D32 | An SMD solvent with a user `epsilon`. The oracle replaces the solvent by `Solvent::Custom` with the named solvent's properties and the given dielectric (`parser.rs:228-231`). | `SolvationSpec` keeps the named solvent in `solvent` and has a field `custom_epsilon::Bool`: `true` means `epsilon` was given and replaces that solvent's dielectric. Strict reads it order-independently. | 7, 8 |

## Global Constraints

Copied from `AGENTS.md` and from the rewrite decisions. Every task's requirements include them.

- **Placement (Q-REPO).** "QuantumEngine is a sub-package of the PureAtomistic monorepo with a path dependency on PureXC." It is self-contained: "its own `Project.toml`, `src/`, `test/`, `docs/`". The PureXC dependency enters at step 05 (D14).
- **Julia floor.** `[compat] julia = "1.13"`: "Julia 1.12 (LLVM 18) miscompiles long straight-line Float32 kernels for AVX-512 targets at `-O2` and above".
- **Closure.**
  - "No `_jll` packages, no artifacts", outside the standard libraries. The item "core closure is binary-free" checks it.
  - "No Python in any form: no interpreter, no Python-backed package, no Python-installed stack."
  - KernelAbstractions only as a `[weakdeps]` extension, never `[deps]` (Q-GPU); none in step 01.
  - "Every dependency has a `[compat]` bound. Stdlibs are excepted in the test project."
  - JSON is JSON.jl, never JSON3 (D2 fixes the bound). FFTs use FFTL, which step 01 does not use.
  - Libxc is never in `QuantumEngine/Project.toml` or `QuantumEngine/test/Project.toml` (as for PureXC in AGENTS.md); the item "Libxc is not a dependency" checks both.
  - "Every new dependency: its closure is resolved and checked `_jll`-free before it is added" (PLAN.md). The task that adds a dependency resolves it first in a throwaway environment outside the repository, inspects the Manifest for non-stdlib `_jll` packages and artifacts, and records the package list and versions in its commit body and in `QuantumEngine/docs/src/dependencies.md`.
- **Code rules (AGENTS.md).**
  - "Generic indexing: iterate `eachindex`/`axes`, or call `Base.require_one_based_indexing` to declare a 1-based assumption. Tests run entry points on `OffsetArray` and `view` inputs."
  - "No `@inbounds` without a benchmark showing the bounds check matters end to end, a local proof of the index range, and a test run under `--check-bounds=yes`." Step 01 has none.
  - "Return types and struct fields are concrete; no `Any` containers."
  - "Functions ending in `!` must not allocate."
  - "Fail fast: errors are `ArgumentError` or carry a module prefix … No silent fallbacks." The prefixes are D13's.
  - "Use `lazy\"...\"` for interpolated messages with four or more pieces."
  - "Comments describe the code as it is: invariants, non-obvious reasons, constraints. No history, plans, or task references." Code ported from MIT-licensed work carries an attribution comment. The oracle is Apache-2.0: its code is not copied; its input files are vendored as data (D16).
  - No fast-math anywhere.
- **Name trap.** Inside `QuantumEngine` and its `IO` module the name `IO` is the submodule, so stream arguments are annotated `Base.IO`. Tests never write `using QuantumEngine.IO` as a bare statement: after it, `IO` is ambiguous between Base and the submodule (`UndefVarError: IO not defined … also exported by Base`, measured on Julia 1.13.4). They write `using QuantumEngine.IO: name1, name2` or qualify as `QuantumEngine.IO.name`.
- **Logging.** `IO` and every module above it log through `IO.Logger` (spec 01); `Errors`, `Units` and `System` sit below `IO` and do not print. The quality item "no bare printing to stdout" scans `QuantumEngine/src` outside `src/io/logger.jl` for the regex `\b(println|print|printstyled)\((\"|lazy\"|stdout)` and for `@show`, and fails on any match.
- **Messages.** Error and warning messages are stored as `LazyString` (field type `msg::LazyString`), so that an interpolated `lazy"..."` message is not converted to a `String` when the error is built. Constructors take any `AbstractString` and convert with `LazyString(msg)`. A message is printed only by walking `msg.parts` in one helper, `Errors._print_message(io::Base.IO, m::LazyString)`, which writes each part with a call on its concrete type (`String`, nested `LazyString`, `SubString{String}`, `Symbol`, `Char`, `Bool`, `Int`, `Float64`) and throws for any other part type. `print(io, ::LazyString)` is never called: measured during Task 1, it fails the typestable and noboxing proofs. The `showerror` methods and `InputWarning`'s `show` print through this helper, and the juliac entry prints an error with `showerror`, so through it too.
- **StrictMode.**
  - `@strict_policy (:typestable, :noalloc)` is the last statement of every module, preceded by the comment `# Last statement: the policy checks the methods defined above it.`
  - A module imports from StrictMode only the macros it uses: `using StrictMode: @strict_policy`, plus `@strict_exempt` in a module that has an exemption (ExplicitImports' stale-import check fails otherwise).
  - `@strict_exempt` is allowed only for entry points and allocating constructors, narrowed to `:noalloc`, with a present-tense reason naming its kind. The one exception is the message rendering below, which is also exempt from `:noboxing` (user decision 2026-11-25: "keep it fast and trim-compatible"). Each exemption names its function, as `PureAtomistic/src/io/IO.jl` does: `@strict_exempt :noalloc read_xyz reason = "entry point: xyz reader"`. An allocating helper that only an entry point calls is exempted by name with a reason naming that entry point. Every allocating constructor is exempted in the task that writes it: `@strict_exempt :noalloc <Type> reason = "entry point: <kind> constructor"`, plus `@strict_exempt :noalloc :_ reason = "entry point: keyword body of the <Type> constructor"` where a constructor has keyword arguments.
  - `Base.showerror` methods and `InputWarning`'s `Base.show` are swept with the module that defines them (measured: `test_compiled(Errors)` and `audit(Errors; sweep = true)` judge `showerror(::IOBuffer, ::InputError)`). Each is written from single-argument `write`/`print` calls, not one varargs `print(io, a, b, c, …)`, prints the message through `Errors._print_message` (Messages), and is exempted `@strict_exempt :noalloc reason = "entry point: renders the error message"`. Measured: with the `LazyString` field, `print(io, e.msg)` fails typestable and noboxing; the parts walk passes typestable and TrimCheck, but reading `parts::Tuple` at a runtime index still boxes (`tuple.jl:33`). So `_print_message`, every `showerror` and `InputWarning`'s `show` are exempted by name from `:noalloc` and `:noboxing`, reason "entry point: renders a LazyString message, whose parts are an abstract Tuple". They run only on the error path. While every function in `Errors` is an exempted entry point, the noalloc scan judges nothing; its count assertion is `@test_broken` until Task 2 sweeps `Units`, which turns it back into `@test`.
  - The methods `@enum` writes into the defining module (`Base.Enums.namemap`, `Base.Enums._enum_hash`, `instances`, `typemin`, `typemax`; measured on Julia 1.13.4) are skipped by `other_function` in `test/strict/check.jl`, with a comment saying that `@enum` writes them.
  - The items "StrictMode policy holds in QuantumEngine" and "StrictMode gates reject the failing control" run `test/strict/check.jl` in a fresh process after `test/strict/workload.jl`.
  - Each task adds its entry points to the workload, so that every exempted function has a compiled specialization.
  - If a sweep that a gate requires to be non-empty comes back empty, stop and report; never drop the assertion.
- **Gates.**
  - Every test is a `@testitem`, with shared setup in `@testmodule`s.
  - Value items compare with oracles; `OffsetArray`/`view` items check generic indexing; error items use `@test_throws "message"`.
  - Aqua (all sub-checks), ExplicitImports, the binary-free closure, `test_layers` plus the public-read check, TrimCheck `@validate` over the exported functions (Float64 and Float32 where generic), and `@test_noalloc` on the allocation-free kernels.
  - The juliac item (tag `:juliac`, `QUANTUMENGINE_JULIAC_BUILD=1`), per-file coverage rows (the ratchet rule), and the mutation runner.
- **Tolerance policy.** "Every numeric comparison in a test states its oracle and a tolerance justified by exactly one of: (a) the oracle's own stated tolerance …; (b) a rounding bound `c · eps(T) · N` …; (c) a measured margin … at most 100 times that." Use `==` where the result is exact by construction. A tolerance is never raised to make a test pass.
- **Parity tolerances (spec 01).**
  - HF and MP2: 1e-9 Ha.
  - UHF: 1e-7.
  - DFT closed shell: the measured angular-table defect + 1e-8. DFT open shell: the measured defect + 2e-6. Both DFT rows are added by step 05.
  - Geometries: 1e-4 Å. Frequencies: 1 cm⁻¹.
- **Run command.**
  - Full suite: `julia --check-bounds=yes --project=QuantumEngine -e 'using Pkg; Pkg.test()'`.
  - Selected items: prefix the full-suite command with `QUANTUMENGINE_ONLY=<stem>`, where `<stem>` is a test file's base name without `.jl` (Task 1 adds this filter; it matches `basename(ti.filename)` exactly, so `inp_tests` does not select `inp_fixture_tests`).
- **Oracle input files.** Tests read the vendored oracle inputs (Task 1b) through the Fixtures helper `oracle_input(relpath)::String`, which reads `joinpath(pkgdir(QuantumEngine), "test", "oracle", "inputs", relpath)`. No test reads QUANTUM_ENGINE_ORACLE_SRC except the vendored-files check of Task 1b, tagged `:oracle_src`, which runs only when that variable is non-empty.
- **Comparing parsed structs.** `XYZFrame`, `InputFile` and the other structs with `Vector` fields have no `==` method (the default falls back to `===`). Tests compare them with the Fixtures helper `fields_equal(a, b) = all(f -> getfield(a, f) == getfield(b, f), fieldnames(typeof(a)))`, applied recursively where a field is itself such a struct.
- **Test item names.** Every `@testitem` a mutation `expect` or a `QUANTUMENGINE_ONLY` run refers to has the exact name the task gives.
- **Formatting and commits.**
  - Run `runic -i QuantumEngine` (runic 1.13, `~/.julia/bin/runic`) before every commit.
  - Use the repository's git identity; never change it.
  - Subject ≤ 72 characters (aim for 50). The body is for a reader with only the repository; `Refs #6.` is the last line of its last body paragraph.
  - Then a blank line, and the trailers alone in the final paragraph: `Assisted-by: Claude <model> <noreply@anthropic.com>` and `Claude-Session: <the executing session's URL>`.
  - "No host machine names or local absolute paths in tracked files or commit messages." The oracle is reached only through the environment variables `QUANTUM_ENGINE_ORACLE`, `QUANTUM_ENGINE_ORACLE_BASIS`, `QUANTUM_ENGINE_ORACLE_SRC` and `QUANTUM_ENGINE_ORACLE_COMMIT` (the full hash `capture.jl` records, Task 13).
  - Never push (D20).
- **Scratch and runs.** Julia runs in the foreground with at most 4 threads. Package probes go in throwaway environments outside the repository.

## Review Focus

These are the inputs most likely to bite a user that no requirement below would otherwise exercise. Each has its pinning test in the task named.

1. **Duplicate keywords or blocks** (`charge 0` on line 2 and `charge 1` on line 9; two `geometry` blocks). Strict throws "keyword `charge` given twice (lines 2 and 9)"; lenient keeps the oracle's result and warns (`:duplicate_keyword`, `:duplicate_block`). Owner: Task 8 (strict), Task 9 (lenient).
2. **CRLF line endings, a UTF-8 BOM, and tabs between tokens** in `.inp` and `.xyz`. These parse exactly as the LF/space version. Owner: Tasks 4 and 8.
3. **An empty file, or one with only comments or only keywords.** Strict throws "input: <file>: no atoms: the input has no geometry block". Owner: Task 8.
4. **Non-finite numbers** (`scf_convergence nan`, `damping inf`, a NaN coordinate). Strict throws "expected a finite number"; lenient keeps the oracle's parsed value and warns. Owner: Task 8.
5. **Coincident nuclei** (two real atoms at one position). `nuclear_repulsion` throws "system: atoms 1 and 2 coincide". A ghost atom sharing a real atom's position is allowed. Owner: Task 3.

---

## File structure

```
QuantumEngine/
  Project.toml                      name, uuid, deps, compat
  src/QuantumEngine.jl              includes the modules, re-exports their interfaces
  src/errors/Errors.jl              error types, InputWarning, showerror methods
  src/units/Units.jl                CODATA 2018 constants, Element, element()
  src/units/elements.jl             the H–Kr element table (data)
  src/system/System.jl              Atom, PointCharge, Molecule, electron counts, nuclear repulsion
  src/io/IO.jl                      module file
  src/io/lines.jl                   input_line and reject_or_warn, shared by the readers and parsers
  src/io/format.jl                  Rust-compatible fixed/exponent formatting
  src/io/logger.jl                  Logger, verbosity, fixed oracle output lines, log paths
  src/io/xyz.jl  pdb.jl  cif.jl     molecule readers (strict/lenient), xyz writer
  src/io/molden.jl                  Molden data types, writer, reader
  src/input/Input.jl                module file
  src/input/types.jl                enums, MethodSpec, InputFile, ParsedInput, _finalize_input
  src/input/vocabulary.jl           method, run-mode, solvent, kernel and other word tables
  src/input/inp.jl                  the .inp parser (strict and lenient)
  src/input/node.jl                 Node tree type
  src/input/structured.jl           schema walker Node → InputFile
  src/input/json.jl                 JSON.jl → Node
  src/input/dict.jl                 AbstractDict → Node, parse_dict, the parse_yaml stub
  src/input/dispatch.jl             parse_input(path), format by extension, geometry files
  ext/QuantumEngineYAMLExt.jl       YAML.jl → Dict → parse_dict (YAML.jl is a weak dependency)
  test/Project.toml  runtests.jl
  test/quality_tests.jl  layers_tests.jl  strict_tests.jl  trim_tests.jl  juliac_tests.jl
  test/strict/{check.jl,workload.jl}
  test/coverage/{Project.toml,check.jl,thresholds.toml}
  test/mutation/{run.jl,mutations.toml}
  test/fixtures.jl                  @testmodule Fixtures
  test/{errors,units,system,format,logger,xyz,pdb,cif,molden,input_types,inp,lenient,json,dict,yaml,inp_fixture}_tests.jl
  test/oracle/inputs/...            vendored oracle inputs + LICENSE + README.md (Task 1b) + EXPECTED.toml (Task 12b)
  test/oracle/{references.toml,tolerances.toml}
  test/oracle/captured/             logs, stdout and Molden files captured from the oracle
  test/oracle/{harness.jl,capture.jl,harness_tests.jl,inputs_tests.jl,live_tests.jl,vendored_tests.jl}
  docs/{Project.toml,make.jl,package.json}  docs/src/{index,api,dependencies,input,parity}.md
  juliac/{build.jl,entry.jl}
docs/superpowers/specs/2026-11-25-qe-checkpoint-formats.md   (Task 14)
```

Modified shared files (Task 1): `.github/workflows/{CI,docs,mutation}.yml`, `AGENTS.md`, `README.md`, `.gitignore`.

---

### Task 1: Package skeleton with every gate, and the error types

**Files:**
- Create:
  - `QuantumEngine/Project.toml`, `QuantumEngine/src/QuantumEngine.jl`, `QuantumEngine/src/errors/Errors.jl`.
  - Test project and items: `QuantumEngine/test/Project.toml`, `test/runtests.jl`, `test/quality_tests.jl`, `test/layers_tests.jl`, `test/strict_tests.jl`, `test/strict/check.jl`, `test/strict/workload.jl`, `test/trim_tests.jl`, `test/juliac_tests.jl`, `test/errors_tests.jl`.
  - Coverage and mutations: `test/coverage/{Project.toml,check.jl,thresholds.toml}`, `test/mutation/{run.jl,mutations.toml}`.
  - Docs and juliac: `docs/{Project.toml,make.jl,package.json}`, `docs/src/{index.md,api.md,dependencies.md}` (`dependencies.md` is the closure record of every dependency, started with StrictMode's), `juliac/{build.jl,entry.jl}`.
- Modify:
  - `.github/workflows/CI.yml`: line 19 and line 130 (append `QuantumEngine`); lines 154 and 161 (append `&& matrix.package != 'QuantumEngine'`); the comment on line 152 becomes `# PureTaylor, PureXC and QuantumEngine have no extension files.`
  - `.github/workflows/docs.yml`: line 25 (append); lines 38 and 51 (append `&& matrix.package != 'QuantumEngine'`); line 58 (append `|| matrix.package == 'QuantumEngine'`); the comment on lines 15–16 adds "and QuantumEngine's under QuantumEngine/", and the comment on lines 35–36 says "The PureTaylor, PureXC and QuantumEngine sites have no examples."
  - `.github/workflows/mutation.yml`: add `- {package: QuantumEngine, shard: ''}`; line 45 (append `&& matrix.package != 'QuantumEngine'`); the comment on line 43 becomes `# PureTaylor, PureXC and QuantumEngine have no extension files.`; the header comment (lines 2–6) gains a sentence that QuantumEngine runs unsharded.
  - `.gitignore`: add `QuantumEngine/docs/build` and `QuantumEngine/juliac/build/`.
  - `README.md`: the package list.
  - `AGENTS.md`:
    - line 7: "five packages", adding "`QuantumEngine/` (the quantum-chemistry program, a pure-Julia rewrite of the Rust `quantum_engine`)", and "apply to all five";
    - line 26: add `QuantumEngine/test/Project.toml` to the test-only projects;
    - line 81: add QuantumEngine to the packages run with `julia --check-bounds=yes --project=<package> -e 'using Pkg; Pkg.test()'`;
    - line 85: "CI runs all five packages" and the matrix list;
    - lines 85 and 86: add QuantumEngine wherever PureTaylor and PureXC are named as having no GPU steps or examples;
    - new bullets under "Scope and layout", "Dependencies" and "Gates" that state the QuantumEngine rules of this plan's Global Constraints.
  - All of these are shared files, named in the D20 notice.

**Interfaces:**
- Produces:
  - Module `QuantumEngine`.
  - Module `QuantumEngine.Errors`, which exports `QuantumEngineError` (abstract, `<: Exception`), `InputError(source::String, line::Int, msg::LazyString)`, `BasisError(msg::LazyString)`, `SCFNotConverged(msg::LazyString)`, `OptimizationFailed(msg::LazyString)`, `NotImplemented(section::String, msg::LazyString)`, `FunctionalUnavailable(msg::LazyString)`, `NumericalError(msg::LazyString)` and `InputWarning(source::String, line::Int, case::Symbol, msg::LazyString)`. Each constructor takes any `AbstractString` for `msg` and converts with `LazyString(msg)` (Global Constraints, Messages). `QuantumEngine` re-exports all of them.
  - `Base.showerror` methods giving the D13 prefixes. `InputError` prints `input: <source>:<line>: <msg>`, or `input: <source>: <msg>` when `line == 0`; `NotImplemented` prints `not implemented (<section>): <msg>`.
  - `Base.show(io::Base.IO, w::InputWarning)`, which prints the same text as `showerror` of the `InputError` with the same source, line and message. It is exempted `:noalloc` as an entry point.
  - `Errors._print_message(io::Base.IO, m::LazyString)`, the one helper that prints a message by walking `m.parts` (Global Constraints, Messages).
  - StrictMode: each error constructor is measured for allocation (Step 3), and each one that allocates is exempted by name.
  - In `test/runtests.jl`, the filter:
    - `:juliac` items run only with `QUANTUMENGINE_JULIAC_BUILD=1`;
    - `:oracle` items only when `QUANTUM_ENGINE_ORACLE` is non-empty;
    - `:oracle_src` items only when `QUANTUM_ENGINE_ORACLE_SRC` is non-empty (the filter gains this line in Task 1b, with the first such item);
    - `:slow` items only with `QUANTUMENGINE_SLOW=1`;
    - `:blocked_on_purexc` items only with `QUANTUMENGINE_PUREXC_BLOCKED=1` (they wait for a PureXC functional; none exist in step 01);
    - `:gpu` items never in this suite;
    - with `QUANTUMENGINE_ONLY=s`, only items with `splitext(basename(ti.filename))[1] == s`.

- [ ] **Step 1: Write the gate items** (they fail: there is no package).
  - **Copied from PureXC and adapted:**
    - `quality_tests.jl`: "ExplicitImports" (with `check_all_qualified_accesses_are_public(...; from = (QuantumEngine,), allow_internal_accesses = false)`), "Aqua" (`Aqua.test_all(QuantumEngine)`), "core closure is binary-free" and "Libxc is not a dependency" (checks `Project.toml` and `test/Project.toml`). New: "no bare printing to stdout" (Global Constraints, Logging); in Task 1 it scans `src/` and finds nothing, and it fails on a planted `println("x")` in a temporary copy (check once by hand, not committed).
    - `strict_tests.jl` and `strict/check.jl`, with these changes:
      - `const P = QuantumEngine`.
      - The module sweep runs over every submodule, `(Errors, Units, System, IO, Input)`, restricted to the modules that exist at each task. The top module `QuantumEngine` owns no functions (measured: `test_compiled(QuantumEngine)` checks 0 signatures and warns "green because it checked nothing"), so it is not swept.
      - The `CHECKED` count of each gate is summed over the swept modules, and each sum must be > 0. A module sweep is not required to be non-empty on its own.
      - `OTHER_FUNCTIONS = (Base.showerror, Base.show)`, held to `:typestable` only, so the `GATES` tuple copied from PureXC drops `("other", "noalloc")` and `("other", "noboxing")` while every listed function is typestable-only. (The module sweep already holds `showerror` and `show` to all three guarantees, with `:noalloc` exempt; the listing judges them a second time.)
      - `other_function` also returns `:skip` for the methods `@enum` writes (Global Constraints, StrictMode).
      - The control has two parts. The listing control is a method `Base.abs2(::_ControlReal)` (unlisted), defined only under `CONTROL`. The gate control is package-owned: under `CONTROL`, `@eval P.Errors` defines `struct _Control; x::Real; end` and `_control_value(c::_Control) = [c.x][1] * c.x` (abstract field: type-unstable, allocating and boxed), and the control workload calls it with `Base.invokelatest`, so every proof and scan gate reports a `FAIL` line for it.
    - `trim_tests.jl`: in Task 1, `@validate` holds `Base.showerror(::IOBuffer, ::T)` for each error type and `Base.show(::IOBuffer, ::InputWarning)`, since the juliac entry prints these. Task 2 adds `element`.
    - `juliac_tests.jl`: builds `qe_inspect` (Task 1 builds it from a trivial entry; Task 8 makes it read `.inp`). The criteria are PureXC's: the build log has no line with "unresolved call" or "Error #"; the binary's output equals the untrimmed `juliac/entry.jl` run byte for byte; nothing goes to stderr. In Task 1 the item runs the binary twice: with no argument (expect exit 1 and the line `usage: qe_inspect <file.inp>`) and with the argument `x.inp` (expect exit 0 and `ok\n`).
  - **`layers_tests.jl`:**
    - `test_layers(QE => (QE.Errors, QE.Units, QE.System, QE.IO, QE.Input), QE.Errors => (), QE.Units => (), QE.System => (QE.Units,), QE.IO => (QE.System, QE.Units, QE.Errors), QE.Input => (QE.IO, QE.System, QE.Units, QE.Errors))`. Each task registers its module as it creates it; until then the line for that module is absent.
    - The per-module public-read loop of `PureAtomistic/test/layers_tests.jl`.
    - "test_calls sweep finds no unresolved call": after the workload, `test_calls(M)` over each submodule (not the top module `QuantumEngine`, which owns no functions), requiring a non-empty sweep with no `MethodError`/`UndefVarError` report (the AGENTS.md rule for PureMLIP, which spec 01 extends to QuantumEngine). Measured: `test_calls(Errors)` returns 1 signature once `showerror` has run, so `Errors` is non-empty from Task 1.
    - "every exemption is used and reasoned": for each submodule, `@test isempty(test_exemptions(M))` (StrictModeTest 0.5.19 returns `:info` findings for an exemption that is no longer needed and for an exempted function with nothing compiled; it never throws), and `@test all(e -> !isempty(e.reason), StrictMode.exemptions())`. Exemptions name functions (Global Constraints), so an allocation-free helper is never exempted and never produces a finding.
  - **`errors_tests.jl`:**
    ```julia
    @testitem "error messages carry their prefix" begin
        using QuantumEngine
        @test_throws "input: a.inp:3: unknown keyword `foo`" throw(InputError("a.inp", 3, "unknown keyword `foo`"))
    @test_throws "input: a.inp:4: x = 2" throw(InputError("a.inp", 4, lazy"x = $(1 + 1)"))
    @test InputError("a.inp", 3, "m").msg isa LazyString
        @test_throws "input: a.json: scf.max_iter: expected an integer" throw(InputError("a.json", 0, "scf.max_iter: expected an integer"))
        @test_throws "basis: no shells for Xe" throw(BasisError("no shells for Xe"))
        @test_throws "scf: not converged after 50 iterations" throw(SCFNotConverged("not converged after 50 iterations"))
        @test_throws "optimization: " throw(OptimizationFailed("x"))
        @test_throws "not implemented (09-post-hf §CCSD): x" throw(NotImplemented("09-post-hf §CCSD", "x"))
        @test_throws "functional: " throw(FunctionalUnavailable("x"))
        @test_throws "numerical: " throw(NumericalError("x"))
        @test all(T -> T <: QuantumEngineError, (InputError, BasisError, SCFNotConverged, OptimizationFailed, NotImplemented, FunctionalUnavailable, NumericalError))
    end
    @testitem "InputWarning shows as the InputError of its source, line and message" begin
        using QuantumEngine
        @test sprint(show, InputWarning("a.inp", 3, :unknown_keyword, "unknown keyword `foo`")) == "input: a.inp:3: unknown keyword `foo`"
        @test sprint(show, InputWarning("a.json", 0, :ignored_key, lazy"x = $(1 + 1)")) == "input: a.json: x = 2"
        @test InputWarning("a.inp", 3, :c, "m").msg isa LazyString
    end
    @testitem "NotImplemented is not TypeContracts' trait" begin
        using QuantumEngine, TypeContracts
        @test QuantumEngine.NotImplemented !== TypeContracts.NotImplemented
    end
    ```
    The second item needs TypeContracts in the test project only (D12, D14).
- [ ] **Step 2: Run** the full suite. Expected: it fails because the package does not exist.
- [ ] **Step 2b: Resolve the closure** of StrictMode 0.5.21 in a throwaway environment outside the repository, as Task 3 Step 0 does for StaticArrays (stop only on a non-stdlib `_jll` or an `Artifacts.toml`). Record the full package list and versions in the commit body and in `docs/src/dependencies.md`, which this task creates.
- [ ] **Step 3: Implement** the package files.
  - **`Project.toml`:**
    - `name = "QuantumEngine"`, a fresh `uuid` (generated with `UUIDs.uuid4()`), `version = "0.1.0"`, `authors = ["el_oso"]`.
    - `[deps]` StrictMode only; `[compat]` `StrictMode = "0.5.21"`, `julia = "1.13"`.
  - **`test/Project.toml`:**
    - deps: Aqua, ExplicitImports, InteractiveUtils, OffsetArrays, Pkg, QuantumEngine (`[sources] QuantumEngine = {path = ".."}`), StrictMode, StrictModeTest, TOML, Test, TestItemRunner, TrimCheck, TypeContracts;
    - compat: Aqua "0.8", ExplicitImports "1.15", OffsetArrays "1", StrictMode "0.5.21", StrictModeTest "0.5.19", TestItemRunner "1.3", TrimCheck "0.1.5", TypeContracts "0.14.7", and "1" or "1.0" for the stdlibs, as in PureXC;
    - `[preferences.StrictMode] checks_enabled = true`.
  - **Module shape:** `QuantumEngine.jl` holds `using StrictMode: @strict_policy`, `include("errors/Errors.jl")`, `using .Errors: ...`, the `export` list, the comment line and `@strict_policy (:typestable, :noalloc)`. `Errors.jl` has the same shape with `using StrictMode: @strict_exempt, @strict_policy`, since its `showerror` methods are exempted.
  - **`showerror` and `show`:** each method writes its pieces with single-argument calls, e.g. `write(io, "input: "); write(io, e.source); write(io, ":"); print(io, e.line); write(io, ": "); _print_message(io, e.msg)`, and carries `@strict_exempt :noalloc reason = "entry point: renders the error message"`. `print(io, ::LazyString)` fails the typestable and noboxing proofs of `Errors` (measured), so the message is printed only by `_print_message`, which walks `msg.parts` (Global Constraints, Messages). `Base.show(io::Base.IO, w::InputWarning)` writes the same pieces as `showerror` of `InputError`, through the same location and message helpers.
  - **Constructor exemptions:** measure each error constructor (`InputError`, `BasisError`, `SCFNotConverged`, `OptimizationFailed`, `NotImplemented`, `FunctionalUnavailable`, `NumericalError`, `InputWarning`) with the StrictMode noalloc check, and exempt by name each that allocates: `@strict_exempt :noalloc <Type> reason = "entry point: error constructor"`. The commit body lists which constructors allocate.
  - **Coverage:** `test/coverage/check.jl` is PureXC's file with only its usage comment changed; `thresholds.toml` gets one row per `src` file (Step 4 says how).
  - **Mutation runner:** `test/mutation/run.jl` is PureXC's runner without the PureTaylor sibling copy, since step 01 has no path dependency; the copy returns with PureXC in step 05. `mutations.toml` holds the header comment only.
  - **Docs:** `docs/make.jl` follows PureXC's with `sitename = "QuantumEngine"`, `modules = [QuantumEngine]`, `deploy_url = "https://el-oso.github.io/PureAtomistic.jl/QuantumEngine"` and `dirname = "QuantumEngine"`. The pages are Home, API and Dependencies for now. `docs/Project.toml` has Documenter "1.19", DocumenterVitepress "0.3", `julia = "1.13"` and `[sources] QuantumEngine = {path = ".."}`.
  - **juliac:** `juliac/build.jl` follows PureXC's with the output name `qe_inspect`. The interim `juliac/entry.jl` prints the usage line and exits 1 without an argument; given a path, it prints `ok`.
- [ ] **Step 4: Run** the full suite under `--check-bounds=yes`.
  - Expected: every item passes, including "StrictMode gates reject the failing control" with a `FAIL` line for each gate.
  - Coverage, in this order: (1) run `julia --check-bounds=yes --project=QuantumEngine -e 'using Pkg; Pkg.test(coverage = true)'`; (2) run `julia --project=QuantumEngine/test/coverage QuantumEngine/test/coverage/check.jl` and read each file's figure from its report (it fails, since the rows are not written yet); (3) write each row at that figure rounded down to two decimals; (4) rerun `check.jl`. Expected: `PASS` for each file. Every later task that adds "the coverage row" follows the same four steps.
- [ ] **Step 5: Run the juliac item** with `QUANTUMENGINE_JULIAC_BUILD=1 QUANTUMENGINE_ONLY=juliac_tests`. Expected: PASS.
- [ ] **Step 6: Commit** with subject `Add the QuantumEngine package with its gates`. The body names the shared files touched and records StrictMode's closure (Step 2b), and the commit is not pushed (D20).

### Task 1b: Vendor the oracle input files

Every later task reads oracle inputs from the repository, so they are vendored before any test needs them (D16).

**Files:**
- Create:
  - `QuantumEngine/test/oracle/inputs/`: every `*.inp`, `*.in`, `*.yaml`, `*.yml`, `*.json` and `*.xyz` under the oracle's `examples/` (including `legacy_inputs/`), `tests/`, `molecules/`, `benchmarks/` and `src/input/`, at the same relative paths, copied byte for byte from `$QUANTUM_ENGINE_ORACLE_SRC` at 4a78a1d. There are 72 such files at that commit (appendix E1).
  - `test/oracle/inputs/LICENSE` (the standard Apache-2.0 text) and `test/oracle/inputs/README.md` (the source repository URL, the commit and "copied unchanged").
  - `test/fixtures.jl`: `@testmodule Fixtures` with `oracle_input_path(relpath)::String` (the vendored file's path), `oracle_input(relpath)::String` (its text) and `fields_equal(a, b)` (Global Constraints). Task 3 adds the molecule fixtures to it.
  - `test/oracle/vendored_tests.jl`: the item "vendored inputs: count", which checks that the files with the six extensions above number 72 (LICENSE, README.md and Task 12b's EXPECTED.toml are not counted); and the item "vendored inputs are the oracle's files", tagged `:oracle_src`, which checks that each file equals its source under `QUANTUM_ENGINE_ORACLE_SRC` byte for byte.
- Modify: `test/runtests.jl` (the `:oracle_src` filter line of Task 1: such items run only when `QUANTUM_ENGINE_ORACLE_SRC` is non-empty).

- [ ] **Step 1: Copy** the files with a shell loop over `find` (no editing), then write LICENSE and README.md.
- [ ] **Step 2: Check for local paths.** `grep -rnE '/Users/|/home/|<host name>' QuantumEngine/test/oracle/inputs`. Expected: no match (none at 4a78a1d, appendix E1).
- [ ] **Step 3: Run** the full suite. Expected: PASS. Add the coverage of `fixtures.jl` only if the coverage tool counts test files (it does not in PureXC's `check.jl`, which judges `src/`).
- [ ] **Step 4: Commit** `Vendor the oracle's input files for the tests`. The body names the Apache-2.0 source.

### Task 2: Units and the element table

**Files:**
- Create: `QuantumEngine/src/units/Units.jl`, `QuantumEngine/src/units/elements.jl`, `QuantumEngine/test/units_tests.jl`.
- Modify: `QuantumEngine.jl` (include, `public Units`), `layers_tests.jl`, `strict/workload.jl`, `trim_tests.jl`, `coverage/thresholds.toml`, `mutation/mutations.toml`.

**Interfaces:**
- Produces, in `QuantumEngine.Units` (all `public`; `element` and `Element` also exported by `QuantumEngine`):
  - **Constants (Float64, CODATA 2018 exact or recommended).**
    - Conversions: `BOHR_TO_ANGSTROM = 0.529177210903`, `ANGSTROM_TO_BOHR = 1 / BOHR_TO_ANGSTROM`, `HARTREE_TO_EV = 27.211386245988`, `HARTREE_TO_KCAL = 627.5094740631`, `HARTREE_TO_JOULE = 4.3597447222071e-18`, `HARTREE_TO_CM1 = 219474.6313632`.
    - SI values: `BOHR_TO_METER = 5.29177210903e-11`, `AMU_TO_KG = 1.66053906660e-27`, `ELEMENTARY_CHARGE = 1.602176634e-19`, `BOLTZMANN_K = 1.380649e-23`, `PLANCK_H = 6.62607015e-34`, `AVOGADRO_NA = 6.02214076e23`, `SPEED_OF_LIGHT = 299792458.0`.
    - Derived: `GAS_CONSTANT_R = AVOGADRO_NA * BOLTZMANN_K` and `AU_TO_DEBYE = ELEMENTARY_CHARGE * BOHR_TO_METER * SPEED_OF_LIGHT * 1e21`.
  - **`struct Element`**, with fields `Z::Int`, `symbol::String`, `mass_most_abundant::Float64`, `mass_average::Float64` (amu), `covalent_radius::Float64`, `bragg_slater_radius::Union{Nothing,Float64}`, `bondi_radius::Union{Nothing,Float64}` (Å). `nothing` means the source lists no value.
  - **Functions.**
    - `element(Z::Integer)::Element` and `element(symbol::AbstractString)::Element`. The symbol lookup is case-insensitive and allocation-free; the table covers Z = 1…36.
    - `covalent_radius(e)`, `bragg_slater_radius(e)` and `bondi_radius(e)`, each `::Float64`. They throw `ArgumentError("units: Bondi (1964) lists no van der Waals radius for Sc")` and `ArgumentError("units: Slater (1964) lists no Bragg–Slater radius for He")` when the source has none.
    - `nelements() = 36`.
  - The constants of the oracle's other modules (0.52917721067 in `xyz.rs`, 0.529177 in `molden.rs`, 1.8897261246 in `pw.rs`) are not reproduced.

- [ ] **Step 1: Write the failing tests** in `units_tests.jl`, as four items named exactly `"units: constants"`, `"units: element table spot checks"`, `"units: lookups and errors"` and `"units: allocation-free lookup"`.
  - **Constants** (`"units: constants"`, which also holds the `AU_TO_DEBYE` check). The oracle here is the CODATA 2018 tables (exact by definition or as published), so the comparisons use `==`. For example: `@test Units.BOHR_TO_ANGSTROM == 0.529177210903`, `@test Units.ANGSTROM_TO_BOHR == 1 / 0.529177210903`, `@test Units.BOHR_TO_METER == 5.29177210903e-11`, `@test Units.GAS_CONSTANT_R == 6.02214076e23 * 1.380649e-23`.
  - **`AU_TO_DEBYE`** within `rtol = 1e-9` of the CODATA-derived 2.541746473. That is a kind (a) bound: the published value has 10 significant digits.
  - **Element spot checks.** Every value is a transcription check against the cited source; a mismatch is reported, never edited to pass. They use `==` against the literal of the source.
    - `mass_average` (IUPAC conventional or standard atomic weights): H 1.008, C 12.011, O 15.999, Fe 55.845, Kr 83.798.
    - `mass_most_abundant` (AME2020): ¹H 1.007825031898, ¹²C 12.0, ¹⁶O 15.99491461926, ⁵⁶Fe 55.934935537, ⁸⁴Kr 83.91149772708 (the full digits of `mass_1.mas20.txt`).
    - Covalent radii (Cordero et al. 2008, C sp³, Mn/Fe/Co low spin): H 0.31, C 0.76, O 0.66, Fe 1.32, Kr 1.16.
    - Bragg–Slater (Slater 1964): H 0.25, C 0.70, O 0.60, Fe 1.40.
    - Bondi (1964): H 1.20, C 1.70, O 1.52, Kr 2.02.
  - **Lookups** (`"units: lookups and errors"`). `element("fe") === element("FE") === element(26)` (case-insensitive); `element("Xx")` throws `"units: unknown element symbol \"Xx\""`; `element(0)` throws `"units: no element with Z = 0 in the table (H–Kr)"` and `element(37)` throws `"units: no element with Z = 37 in the table (H–Kr)"`; `bondi_radius(element("Sc"))` throws `"units: Bondi (1964) lists no van der Waals radius for Sc"`; `bragg_slater_radius(element("He"))` throws `"units: Slater (1964) lists no Bragg–Slater radius for He"`.
  - **Table invariant** (in `"units: element table spot checks"`). `[element(z).Z for z in 1:36] == 1:36` and the symbols are unique.
  - **`@test_noalloc`** (`"units: allocation-free lookup"`) on `element("Fe")` and `element(26)`.
- [ ] **Step 2: Run** with `QUANTUMENGINE_ONLY=units_tests`. Expected: FAIL (`Units` is not defined).
- [ ] **Step 3: Implement** the module and transcribe `elements.jl`.
  - The table has one row per element and a header comment naming each source with its full citation:
    - masses: CIAAW/IUPAC standard and conventional atomic weights, and AME2020;
    - radii: Cordero, *Dalton Trans.* 2008, 2832; Slater, *J. Chem. Phys.* 41, 3199 (1964); Bondi, *J. Phys. Chem.* 68, 441 (1964).
  - `element(symbol)` scans the 36 rows with a case-insensitive, allocation-free comparison.
- [ ] **Step 4: Run** the units items, then the full suite. Expected: PASS. Add `element` calls to the workload, `element(::String)` and `element(::Int)` to the TrimCheck item, and coverage rows for both new files.
- [ ] **Step 5: Mutations.** Add one entry for `ANGSTROM_TO_BOHR` (replace `1 /` by `1 *`) with `expect = "units: constants"`, the exact name of the Step 1 item.
- [ ] **Step 6: Commit** `Add CODATA 2018 constants and the H–Kr element table`.

### Task 3: Atoms, molecules, electron counts and nuclear repulsion

**Files:**
- Create: `QuantumEngine/src/system/System.jl`, `QuantumEngine/test/system_tests.jl`.
- Modify: `QuantumEngine/Project.toml` (StaticArrays "1.9"), `QuantumEngine/test/Project.toml` (StaticArrays, compat "1.9": the tests build `SVector`s), `test/fixtures.jl` (the molecule fixtures), `QuantumEngine.jl`, `layers_tests.jl`, workload, trim, coverage, mutations, `docs/src/dependencies.md` (Task 1's closure record; StaticArrays' closure).

**Interfaces:**
- Consumes: `Units.element`, `Units.Element`.
- Produces, in `QuantumEngine.System` (exported and re-exported):
  - **`Atom{T<:Real}`**, with fields `Z::Int` (0 for a ghost), `symbol::String` (as written), `position::SVector{3,T}` (bohr), `ghost::Bool` and `mass::T` (amu; 0 for a ghost).
    - The inner constructor `Atom{T}(Z, symbol, position, ghost, mass) where {T}` takes unconstrained arguments.
    - The outer constructor is `Atom(symbol::AbstractString, position::AbstractVector; mass = nothing)`, with T = `float(eltype(position))`; position must have length 3.
    - The symbol rules, case-insensitive:
      - `El` is a real atom.
      - `El@`, `BqEl` and `Bq-El` are ghosts of El.
      - A bare `Bq` throws `"system: a Bq atom needs an element (BqO, O@); a point charge is \`Bq x y z q\`"`.
      - Anything else throws `"system: unknown element symbol \"Foo\""`.
    - Default mass: `mass_most_abundant` (D15).
  - **`PointCharge{T<:Real}(charge::T, position::SVector{3,T})`.**
  - **`Molecule{T<:Real, V<:AbstractVector{Atom{T}}}`**, with fields `atoms::V`, `charge::Int`, `multiplicity::Int` and `point_charges::Vector{PointCharge{T}}`.
    - The constructor is `Molecule(atoms::AbstractVector{Atom{T}}; charge = 0, multiplicity = 1, point_charges = PointCharge{T}[])`.
    - `multiplicity < 1` throws `"system: multiplicity must be at least 1, got 0"`.
  - **`basis_element(a::Atom)::Int`:** the element number of the atom, or of the element a ghost stands for, derived from its symbol. A ghost whose symbol names no element (built only by the lenient `:ghost_unchecked` branch of Task 9, through the inner constructor) throws `"system: ghost atom \"Foo@\" stands for no element"`; the test builds such an atom with the inner constructor.
  - **Electron counts.**
    - `nelectrons(mol)::Int` is the sum of `Z` minus `charge`. It throws `"system: charge 3 exceeds the nuclear charge 2"` when the count is negative.
    - `nalpha(mol)::Int` and `nbeta(mol)::Int` are `(n ± (mult − 1)) / 2`. They throw an `ArgumentError` with `"system: 2 electrons cannot have multiplicity 2"` when `n − (mult − 1)` is odd or negative.
  - **`nuclear_repulsion(mol)::T`** is Σ_{i<j} Z_i Z_j / r_ij over the atoms with Z > 0, iterated with `eachindex`. It throws `"system: atoms 1 and 2 coincide"` for r = 0 between two such atoms; the indices are positions in `eachindex(atoms)`. Point charges are not included: the charge–charge and nucleus–charge terms belong to step 14 (docstring).
- Produces, in `test/fixtures.jl`: `@testmodule Fixtures` with `h2()` (examples/h2.inp: H at z = ∓0.7 bohr), `water()` (examples/h2o_631g_sp.inp, Å converted with `Units.ANGSTROM_TO_BOHR`) and `o2_triplet()` (examples/o2_triplet.inp, multiplicity 3: O at (0, 0, 0) and (1.208, 0, 0) Å, converted with `Units.ANGSTROM_TO_BOHR` as in `water()`). Each returns a `Molecule{Float64}` with positions in bohr; the coordinates are copied verbatim from the files, and a file in Å is converted with `Units.ANGSTROM_TO_BOHR`.
- StrictMode: `Atom` (symbol parsing allocates the `String`), `Molecule` and its keyword body (the default `PointCharge{T}[]`) are exempted as allocating constructors (Global Constraints).

- [ ] **Step 0: Resolve the closure** of StaticArrays 1.9 in a throwaway environment outside the repository (`julia --project=<scratch>/sa -e 'using Pkg; Pkg.add(name = "StaticArrays", version = "1.9"); Pkg.status(mode = Pkg.PKGMODE_MANIFEST)'`). Inspect the Manifest for non-stdlib `_jll` packages and for `Artifacts.toml` in the package directories. Record the full closure found (every package with its version) in the commit body and in `docs/src/dependencies.md`. Non-stdlib packages without binaries (PrecompileTools, Preferences and the like) are recorded and accepted. Stop and report only on a non-stdlib `_jll` package or an `Artifacts.toml`.
- [ ] **Step 1: Write the failing tests.** The items are named `"electron counts"`, `"nuclear repulsion"`, `"Float32 flows through"`, `"ghost atoms"`, `"system errors"`, `"system: generic indexing"` and `"system: allocation-free counts"`.
  ```julia
  @testitem "electron counts" setup = [Fixtures] begin
      using QuantumEngine
      o2 = Fixtures.o2_triplet()
      @test (nelectrons(o2), nalpha(o2), nbeta(o2)) == (16, 9, 7)
      w = Fixtures.water()
      cation = Molecule(w.atoms; charge = 1, multiplicity = 2)
      @test (nalpha(cation), nbeta(cation)) == (5, 4)
      bad = Molecule(Fixtures.h2().atoms; multiplicity = 2)
      @test_throws "system: 2 electrons cannot have multiplicity 2" nalpha(bad)
      @test_throws "system: charge 3 exceeds the nuclear charge 2" nelectrons(Molecule(Fixtures.h2().atoms; charge = 3, multiplicity = 2))
      @test_throws "system: multiplicity must be at least 1, got 0" Molecule(Fixtures.h2().atoms; multiplicity = 0)
  end
  ```
  - **Value items.**
    - H2: `nuclear_repulsion(h2) ≈ 1 / 1.4` with `rtol = 4eps()`, a kind (b) bound: one subtraction, one square root and one division, each at most 1 ulp.
    - Water: within `rtol = 8eps()` of a BigFloat recomputation written in the test from the literal Å coordinates and `big"0.529177210903"`, which is independent of the package code. Kind (b): 3 pairs of ≤ 4 roundings plus 1 ulp from the conversion.
  - **Float32 flows through.** `Molecule{Float32}` built from Float32 positions: `nuclear_repulsion` returns a `Float32` within `rtol = 8eps(Float32)` of the Float64 value.
  - **Ghosts.** `Atom("O@", [0,0,0])` gives `Z == 0`, `ghost == true`, `basis_element == 8` and `mass == 0`. `Atom("bq-h", …)` is a ghost of H. A ghost at a real atom's position: `nuclear_repulsion` is unchanged and nothing throws (Review Focus 5).
  - **Errors:** "unknown element symbol", the Bq message, and "atoms 1 and 2 coincide" (Review Focus 5). `basis_element(Atom{Float64}(0, "Foo@", SVector(0.0, 0.0, 0.0), true, 0.0))` throws `"system: ghost atom \"Foo@\" stands for no element"` (the atom built with the inner constructor, as Task 9's `:ghost_unchecked` builds it).
  - **Generic indexing.** `Molecule(OffsetArray(atoms, -1:1))` and `Molecule(view(atoms, 1:3))` give the same `nuclear_repulsion` (`==`) and `nelectrons`. The coincidence message on the OffsetArray names the indices `-1 and 0`. `Atom("H", OffsetArray([0.0, 0.0, 1.0], 0:2))` has `position == SVector(0.0, 0.0, 1.0)`.
  - **`@test_noalloc`** on `nuclear_repulsion`, `nelectrons`, `nalpha` and `nbeta` for Float64 and Float32.
- [ ] **Step 2: Run** with `QUANTUMENGINE_ONLY=system_tests`. Expected: FAIL.
- [ ] **Step 3: Implement** `System.jl`. Follow the constructor cascade `Atom(symbol, position)` → `Atom{T}(...)` and `Molecule(atoms; ...)` → `Molecule{T,V}(...)`.
- [ ] **Step 4: Run** the system items, then the full suite. Expected: PASS.
  - Add to the workload: `nuclear_repulsion` (Float64, Float32), the electron counts and `Atom` construction.
  - Add to TrimCheck: the same entries in Float64 and Float32.
  - Add the coverage row.
- [ ] **Step 5: Mutations:**
  - the `/ 2` in `nalpha` → `/ 3`, `expect = "electron counts"`;
  - `Z_i * Z_j` → `Z_i + Z_j`, `expect = "nuclear repulsion"`;
  - the parity check `isodd` → `iseven`, `expect = "electron counts"`.
- [ ] **Step 6: Commit** `Add atoms, molecules and electron counting`.

### Task 4: Molecule files: xyz (multi-frame), pdb and cif

**Files:**
- Create: `QuantumEngine/src/io/IO.jl`, `src/io/lines.jl`, `src/io/xyz.jl`, `src/io/pdb.jl`, `src/io/cif.jl`, `test/xyz_tests.jl`, `test/pdb_tests.jl`, `test/cif_tests.jl`.
- Modify: `QuantumEngine.jl` (`public IO`; re-export the readers), layers, workload, trim, coverage.

**Interfaces:**
- Consumes: `System.Atom`, `Units.BOHR_TO_ANGSTROM`, `Units.ANGSTROM_TO_BOHR`, `Errors.InputError`, `Errors.InputWarning`.
- Produces, in `QuantumEngine.IO`:
  - **Frames:** `struct XYZFrame; atoms::Vector{Atom{Float64}}; comment::String; end`.
  - **`read_xyz(io::Base.IO; source = "<stream>", lenient = false, warnings = InputWarning[])::Vector{XYZFrame}`** reads every frame.
    - Line 2 of each frame is the comment, even when blank (class B, D6).
    - Positions are converted from Å to bohr.
  - **`write_xyz(io::Base.IO, atoms::AbstractVector{<:Atom}; comment::AbstractString = "")`** writes one frame: the count line, then the comment line, then `<symbol> <x> <y> <z>` per atom in Å, as shortest round-trip floats, separated by single spaces.
  - **`read_pdb(io::Base.IO; source, lenient, warnings)::Vector{Atom{Float64}}`** reads `ATOM  `/`HETATM` records.
    - Coordinates come from columns 31–38, 39–46 and 47–54.
    - Strict requires the element in columns 77–78 (D25).
  - **`read_cif(io::Base.IO; source, lenient, warnings)::Vector{Atom{Float64}}`** reads one `loop_` with `_atom_site_type_symbol` and `_atom_site_Cartn_x/y/z`.
    - Non-letter characters are stripped from the symbol (`Fe2+` → `Fe`).
    - A loop with fractional coordinates only throws `"cif: only Cartesian _atom_site coordinates are supported"`.
  - Errors are `InputError(source, line, msg)` with `line` set and `msg` starting `xyz: `, `pdb: ` or `cif: ` (D13); the line number is never repeated inside `msg`. The element is checked before the atom is constructed (D13): `System.element_number(symbol)` (public in `System`, not exported) returns 0 for an unknown symbol, which is `:unknown_element` with the reader's own message (the text of `System`'s, with `system: ` replaced by the format); the `ArgumentError` it throws for a bare `Bq` is caught and rethrown as `InputError(source, line, "<fmt>: " * <its message without the leading "system: ">)` in both modes.
  - **Shared helpers**, in `src/io/lines.jl`, declared `public` in `IO` because the `.inp` parser (Task 8, both helpers) and the structured walker (Task 10, `reject_or_warn`) in `Input` call them too (the public-read check of `layers_tests.jl`):
    - `input_line(raw::AbstractString, n::Int)::Tuple{SubString{String}, Vector{SubString{String}}}`: the line with a leading UTF-8 BOM removed when `n == 1`, a trailing `\r` removed and surrounding whitespace stripped, and its tokens split on spaces and tabs. Every reader and parser cleans and tokenizes its lines through it; none strips a BOM or `\r` on its own.
    - `reject_or_warn(source::AbstractString, line::Int, lenient::Bool, warnings::Vector{InputWarning}, case::Symbol, msg::AbstractString)::Nothing`: strict throws `InputError(source, line, msg)`; lenient pushes `InputWarning(source, line, case, msg)` and returns, and the caller then applies the oracle value. It is the only place that chooses between strict and lenient; the readers here, `parse_inp` and `parse_structured` all call it.
  - **Lenient cases L19a–L19h**, specified and tested here; each calls `reject_or_warn` and reproduces `io/molecule.rs:35-190` of the oracle (the implementer cites the line in each test). Task 9 refers to these rows and does not repeat them.

    | case | input | strict message | lenient value |
    |---|---|---|---|
    | L19a `:xyz_count` | count 3, 2 atom lines | "declares 3 atoms but has 2" | 2 atoms |
    | L19b `:unknown_element` | xyz `Xx 0 0 0` | "unknown element symbol" | built with the inner constructor `Atom{Float64}(1, symbol, position, false, element(1).mass_most_abundant)`: Z = 1, the symbol as written, not a ghost (verified: the oracle accepts the file and fails later at the basis lookup, appendix E2) |
    | L19c `:pdb_short_line` | 40-column ATOM line | "shorter than 54 columns" | skipped |
    | L19d `:pdb_element_columns` | no columns 77–78, atom name `CL` | "no element in columns 77–78" | C (a record shorter than 78 columns); a record 78 columns or wider, counted with its trailing blanks, whose columns 77–78 are blank gives the oracle's empty symbol: built with the inner constructor `Atom{Float64}(1, "", position, false, element(1).mass_most_abundant)`, Z = 1, one warning (`molecule.rs:95-106`) |
    | L19e `:cif_row` | 3 values, 4 tags | "row has 3 values for 4 tags" | row skipped |
    | L19f `:cif_coordinate` | coordinate `x` | "is not a number" | 0.0 |
    | L19g `:nonfinite` | an xyz, pdb or cif coordinate `nan`, `inf` or `1e400` | "expected a finite number, got \"nan\"" | the parsed value: NaN, ±Inf (Rust `f64` parses `nan` and `inf`, and a decimal past the range as ±Inf); a decimal below the subnormal range reads ±0.0 in both modes, a hex token is "not a number" |
    | L19h `:cif_columns` | an `_atom_site` coordinate loop without `_atom_site_type_symbol` or one of `_atom_site_Cartn_x/y/z` | "the _atom_site loop needs _atom_site_type_symbol and _atom_site_Cartn_x, _y and _z" | the loop's rows skipped (`molecule.rs:159-173`) |

    Recorded exception, not reproduced in lenient mode: an xyz frame with more atom lines than its count. The oracle reads the first frame and ignores the lines after it; `read_xyz` reads the next line as the next frame's count, an error in both modes when it is not one. The `read_xyz` docstring states this.
  - StrictMode: `read_xyz`, `write_xyz`, `read_pdb`, `read_cif` and each allocating helper they call are exempted by name (Global Constraints); `input_line` (it allocates the token vector) is exempted by name with the reason `"entry point helper: line reader of read_xyz, read_pdb, read_cif and parse_inp"`; `XYZFrame` is exempted as an allocating constructor if its constructor checks its arguments.

- [ ] **Step 1: Write the failing tests.** The items are named `"xyz: round trip"`, `"xyz: multi-frame"`, `"xyz: blank comment line"`, `"xyz: line endings"`, `"xyz: errors"`, `"xyz: oracle files"`, `"xyz: generic indexing"`, `"pdb: errors"`, `"cif: errors"`, plus one value item each for pdb and cif (`"pdb: values"`, `"cif: values"`, on small inline files), and the lenient items `"xyz: lenient"`, `"pdb: lenient"` and `"cif: lenient"`. In the messages below, `<src>` is the `source` the test passes.
  - **Lenient** (one `@testset` per L19 row of the table above): strict throws the row's message; lenient takes the row's value and records exactly one warning, whose `case` is the row's symbol.
  - **xyz round trip.** Water written and read back gives positions within `rtol = 4eps()` of the input. Kind (b): ×BOHR_TO_ANGSTROM, shortest-repr printing is exact, ×ANGSTROM_TO_BOHR, and the constants' product is within 1 ulp of 1. Symbols and the comment come back unchanged (`==`).
  - **Multi-frame.** Three frames written and read back give `length == 3`, each frame equal as above.
  - **Blank comment line.** A water frame with an empty line 2 reads 3 atoms, O first, in both modes (class B; the oracle reads 2 atoms, verified by probe).
  - **Line endings (Review Focus 2).** The same file with CRLF endings, a UTF-8 BOM and tab separators reads the same frames as the LF version: `fields_equal` on each frame and on each atom.
  - **Errors (strict):**
    - `"input: <src>:1: xyz: expected the atom count"` for a non-integer count;
    - `"input: <src>:1: xyz: the frame declares 3 atoms but has 2"`;
    - `"input: <src>:3: xyz: unknown element symbol \"Xx\""` (the text of `System`'s error with `system: ` replaced by `xyz: `, D13; the item also asserts that the message does not contain `"system: "`);
    - `"input: <src>:1: pdb: ATOM record shorter than 54 columns"`;
    - `"input: <src>:1: pdb: no element in columns 77–78"`;
    - `"input: <src>:7: cif: row has 3 values for 4 tags"`;
    - `"input: <src>:7: cif: coordinate \"x\" is not a number"`;
    - `"cif: only Cartesian _atom_site coordinates are supported"` (with its line).
  - **Values** (`"xyz: oracle files"`, read through `Fixtures.oracle_input`). `length(read_xyz(examples/water.xyz)[1].atoms) == 3`; `examples/legacy_inputs/adenina.xyz` has 32 atoms; `examples/legacy_inputs/neb_test.neb.xyz` has `length(frames) == 5` with 3 atoms in each (appendix E1).
  - **Generic indexing.** `write_xyz` with an `OffsetArray` and with a `view` of the atoms writes the same bytes as with the `Vector`.
- [ ] **Step 2: Run** with `QUANTUMENGINE_ONLY=xyz_tests` (then `pdb_tests`, `cif_tests`). Expected: FAIL.
- [ ] **Step 3: Implement** the three readers and the writer.
  - `lines.jl` first: `input_line` and `reject_or_warn`. The readers clean and tokenize every line through `input_line` and make every strict-or-lenient choice through `reject_or_warn`.
  - Exempt the entry points and their allocating helpers by name (Global Constraints).
- [ ] **Step 4: Run** the items, then the full suite. Expected: PASS. Add the entries to the workload and TrimCheck, and the coverage rows.
- [ ] **Step 5: Commit** `Read and write xyz, read pdb and cif molecule files`.

### Task 5: Output formatting and the logger

**Files:**
- Create: `QuantumEngine/src/io/format.jl`, `src/io/logger.jl`, `test/format_tests.jl`, `test/logger_tests.jl`.
- Modify: `QuantumEngine/Project.toml` (Printf, with a `[compat]` entry whose value the implementer verifies Pkg accepts for a stdlib, by `Pkg.resolve()` in the package and the test project, and records in the commit body; `"1.13"` is unverified), `IO.jl`, workload, trim, coverage, mutations.

**Interfaces:**
- Produces, in `QuantumEngine.IO`:
  - **Rust-compatible formatters** (the oracle prints with Rust's `format!`):
    - `fmt_fixed(x::Real, width::Int, prec::Int)::String`, as Rust `{:w.p}`, right-aligned;
    - `fmt_exp(x::Real, prec::Int; width::Int = 0)::String`, as Rust `{:w.pe}`: the exponent has no `+` and no leading zeros, as in `5.369e-1`, `0.000e0` and `1.260e0`;
    - `fmt_int(n::Integer, width::Int)::String`, right-aligned;
    - `fmt_str(s::AbstractString, width::Int)::String`, left-aligned as Rust pads strings.
  - **`@enum Verbosity Silent = 0 Low = 1 Normal = 2 High = 3`** and `verbosity_from_input(v::Integer)::Verbosity`: 0 → Low, 1 → Normal, ≥ 2 → High (`engine/mod.rs:46-52`); a negative value throws `"io: verbosity must be ≥ 0, got -1"`. This is the first `@enum`: the `other_function` skip of Task 1 must already cover its methods, and the policy item shows no `FAIL unlisted other` line for `Verbosity`.
  - **`struct Logger{A<:Base.IO, B<:Base.IO}; out::A; log::B; verbosity::Verbosity; end`.** `Logger(out, log; verbosity = Normal)`; `devnull` stands for no log file.
  - **Logger functions.** Each line goes to both streams when `lg.verbosity >= level`, and both are flushed after it.
    - `log_print(lg, level::Verbosity, msg::AbstractString)`.
    - `log_success(lg, msg)` (`"✓ "` prefix, Normal), `log_warning(lg, msg)` (`"⚠ Warning: "`, Normal) and `log_error(lg, msg)` (`"❌ Error: "`, Low).
    - `log_heading(lg, title)`: a blank line, `╔`, `═` repeated `ncodeunits(title) + 4` times, `╗`; then `║  TITLE  ║` in upper case; then the bottom line and a blank line. This is the oracle's `heading` (`io/writer.rs:68-73`). The width counts bytes, as Rust's `len()` does.
    - `log_section(lg, title)`: a blank line, the title, then `-` repeated `ncodeunits(title)` times, then a blank line.
    - `print_atom_table(lg, atoms::AbstractVector{<:Atom})`: the oracle's header line, `"  " * "─"^95`, and one row per atom `"  " * fmt_int(i,2) * " " * fmt_str(sym,4) * ("   " * fmt_fixed(c,12,8))` for the 3 bohr and the 3 Å coordinates. `i` counts from 1 in iteration order (`enumerate`); it is a counter, not the array index.
    - `print_total_energy(lg, e::Real)` writes `"  Total Energy:  " * fmt_fixed(e,20,12) * " Hartree"`.
    - `report_optimization_step(lg, step::Integer, e::Real, gnorm::Real)` writes `"[OPT] Step " * fmt_int(step,3) * ": E = " * fmt_fixed(e,17,10) * " |grad| = " * fmt_exp(gnorm,3)`, the oracle's `"[OPT] Step {:3}: E = {:17.10} |grad| = {:.3e}"` (`io/writer.rs:41`).
  - **Paths.** `log_path(input_path::AbstractString, output::Union{Nothing,String})::String` returns `output` verbatim, or the input path with its extension replaced by `.log` (next to the input). `output_base(log_path)::String` strips the extension. These are the oracle's rules (`engine/mod.rs:34-43`).
  - StrictMode: `Logger`'s constructor (keyword `verbosity`) is exempted as a constructor with a keyword body; the logger functions and formatters, which build `String`s, are exempted by name as entry points.

- [ ] **Step 1: Write the failing tests.** The oracle lines below are copied verbatim from `examples/h2o_pbe0_631g.log` in the oracle checkout. That file is an untracked oracle output of unknown date, so Task 13 re-pins the atom-table row against a capture the user provides; the other lines stay pinned to this file.
  ```julia
  @testitem "Rust exponent format" begin
      using QuantumEngine.IO: fmt_exp, fmt_fixed
      @test fmt_exp(0.0, 3; width = 8) == " 0.000e0"
      @test fmt_exp(1.260, 3; width = 8) == " 1.260e0"
      @test fmt_exp(0.5369, 3; width = 8) == "5.369e-1"
      @test fmt_exp(8.304e-9, 3; width = 8) == "8.304e-9"
      @test fmt_exp(-2.5e10, 3) == "-2.500e10"
      @test fmt_fixed(-76.301896228438, 20, 12) == "    -76.301896228438"
  end
  @testitem "oracle output lines" setup = [Fixtures] begin
      using QuantumEngine
      using QuantumEngine.IO: Logger, print_total_energy, print_atom_table, log_heading, report_optimization_step
      out = IOBuffer(); lg = Logger(out, devnull)
      print_total_energy(lg, -76.301896228438)
      @test String(take!(out)) == "  Total Energy:      -76.301896228438 Hartree\n"
      print_atom_table(lg, Fixtures.water().atoms)
      @test split(String(take!(out)), '\n')[3] ==
          "   1 O        0.00000000     0.00000000     0.22166487     0.00000000     0.00000000     0.11730000"
      log_heading(lg, "Final Results")
      @test String(take!(out)) == "\n╔═════════════════╗\n║  FINAL RESULTS  ║\n╚═════════════════╝\n\n"
      # io/writer.rs:41: "[OPT] Step {:3}: E = {:17.10} |grad| = {:.3e}"; -1.1167593075 is 13
      # characters, so {:17.10} pads it with 4 spaces after "E = ".
      report_optimization_step(lg, 3, -1.1167593075, 2.5e-4)
      @test String(take!(out)) == "[OPT] Step   3: E =     -1.1167593075 |grad| = 2.500e-4\n"
  end
  ```
  - **Generic indexing** (`"logger: generic indexing"`). `print_atom_table` on `OffsetArray(atoms, -1:1)` and on `view(atoms, 1:3)` writes the same bytes as on the `Vector`, with rows numbered 1 to 3.
  - **Levels** (`"logger: levels"`). A `Low` logger drops `Normal` lines and keeps `log_error`; a `Silent` logger writes nothing. Both streams get identical bytes.
  - **Verbosity mapping** (`"logger: verbosity mapping"`) 0, 1, 2 and 5, and the error for -1.
  - **Paths** (`"logger: paths"`). `log_path("dir/h2.inp", nothing) == "dir/h2.log"`, `log_path("h2.inp", "h2_test") == "h2_test"` and `output_base("dir/h2.log") == "dir/h2"`.
- [ ] **Step 2: Run** with `QUANTUMENGINE_ONLY=format_tests` and `QUANTUMENGINE_ONLY=logger_tests`. Expected: FAIL.
- [ ] **Step 3: Implement** the formatters with Printf (`%.*f` and `%.*e`) and rewrite the exponent for `fmt_exp`. Exact decimal ties may round differently from Rust (not measured). The parity harness compares parsed numbers, not text, so a tie difference is cosmetic; the `fmt_fixed` docstring states this.
- [ ] **Step 4: Run** the items, then the full suite. Expected: PASS.
  - Add to the workload: each formatter, `Logger` construction, each logger function on a `Logger{IOBuffer, Base.DevNull}`, `verbosity_from_input`, `log_path` and `output_base`.
  - Add the coverage rows for `format.jl` and `logger.jl`.
  - Run TrimCheck on `fmt_fixed`, `fmt_exp`, `print_total_energy`, `report_optimization_step` and `log_print` with `Logger{IOBuffer, Base.DevNull}`. If Printf fails TrimCheck, stop and report the finding chain, obtained with `TrimCheck.init_validation`/`validate_function` in process: TrimCheck 0.1.5 crashes while printing a failure on Julia 1.13.4 (`UndefVarError(:mapssavaluetypes)`).
- [ ] **Step 5: Mutations:** one entry in `fmt_exp`'s exponent rewrite (keep the leading zero), `expect = "Rust exponent format"`.
- [ ] **Step 6: Commit** `Add the logger and the oracle's fixed output formats`.

### Task 6: Molden writer and reader

**Files:**
- Create: `QuantumEngine/src/io/molden.jl`, `test/molden_tests.jl`.
- Modify: `IO.jl`, workload, trim, coverage, `docs/src/api.md`.

**Interfaces:**
- Produces, in `QuantumEngine.IO` (all `public`):
  - **Data types.**
    - `struct MoldenShell{T}; atom::Int; l::Int; exponents::Vector{T}; coefficients::Vector{T}; end`. Here `l ∈ 0:4`; the constructor throws `"molden: shell with l = 5 has no Molden label"` and `"molden: 3 exponents but 2 coefficients"`.
    - `struct MoldenOrbitals{T}; energies::Vector{T}; occupations::Vector{T}; coefficients::Matrix{T}; end`. `coefficients` is n_ao × n_mo; its constructor calls `Base.require_one_based_indexing` and checks that the sizes agree.
    - `struct MoldenData{T}; title::String; atoms::Vector{Atom{T}}; shells::Vector{MoldenShell{T}}; alpha::MoldenOrbitals{T}; beta::Union{Nothing,MoldenOrbitals{T}}; end`.
  - **`write_molden(io::Base.IO, d::MoldenData)`** writes these sections:
    - `[Molden Format]`, then `[Title]` and the title line.
    - `[Atoms] AU`, with one line per atom: `<symbol> <index> <Z> <x> <y> <z>` in bohr.
    - `[GTO]`, with one block per atom: `<atom> 0`, then for each shell `<s|p|d|f|g> <nprim> 1.00` followed by its `<exponent> <coefficient>` lines, then a blank line.
    - `[MO]`, with for each orbital `Sym= A`, `Ene= <e>`, `Spin= Alpha|Beta` and `Occup= <occ>`, followed by its `<ao> <coefficient>` lines.
    - Every float is printed with `string(::Float64)` (shortest round-trip, D18). There is no `[5D]` line, so the Cartesian default applies.
    - The docstring states the Cartesian AO order expected within a shell, which is the Molden document's order:
      - d: xx, yy, zz, xy, xz, yz;
      - f: xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz;
      - g: xxxx, yyyy, zzzz, xxxy, xxxz, yyyx, yyyz, zzzx, zzzy, xxyy, xxzz, yyzz, xxyz, yyxz, zzxy.
    - The coefficients are written as given; normalization is the caller's (step 02).
  - **`read_molden(io::Base.IO; source = "<stream>")::MoldenData{Float64}`.**
    - The `[Atoms]` unit may be `AU`, `(AU)`, `Bohr`, `Angs`, `(Angstrom)` or `Angstrom`, case-insensitive. Any other unit throws `InputError(source, 3, "molden: unknown [Atoms] unit \"nm\"")`, printed `input: <src>:3: molden: unknown [Atoms] unit "nm"` (D13).
    - `Spin=` is optional; an orbital without it is Alpha.
    - Fortran `D` exponents are accepted.
    - An unknown section throws `InputError(source, N, "molden: unknown section [X]")`.
  - The data-type constructors throw `ArgumentError`s with the `molden:` prefix (D13).
  - StrictMode: the `MoldenShell`, `MoldenOrbitals` and `MoldenData` constructors (they check their arguments) are exempted as allocating constructors; `write_molden`, `read_molden` and their allocating helpers by name as entry points.

- [ ] **Step 1: Write the failing tests.**
  - **Round trip.** A synthetic water `MoldenData`: O with s, s, p, d shells and H with s; 13 AOs; alpha and beta orbitals with random but seeded coefficients. Written and read back, it is `==` field by field, since shortest-repr printing is exact. The same holds with `beta = nothing`.
  - **Labels.** The l = 4 shell writes `g`, and l = 5 throws.
  - **Oracle file (added in Task 13).** `read_molden` reads `captured/water_pbe_631g.molden`, written by the oracle with Å coordinates and alpha only. The check is 3 atoms, the O position z = 0.22166487 bohr within 1e-6 (kind (a): the oracle prints Å with 6 decimals, ×1/0.529177 ≈ 1.9e-6, half-ulp of print 9.5e-7), and 13 AO coefficients per orbital.
  - **Errors:** each message above.
  - **Generic indexing.** `MoldenOrbitals` with an `OffsetArray` coefficient matrix throws `"offset arrays are not supported but got an array with index other than 1"` (the `ArgumentError` of `Base.require_one_based_indexing`, read in Julia 1.13.4 `base/abstractarray.jl:135`).
- [ ] **Step 2: Run** with `QUANTUMENGINE_ONLY=molden_tests`. Expected: FAIL.
- [ ] **Step 3: Implement** the types, writer and reader.
- [ ] **Step 4: Run** the items, then the full suite. Expected: PASS. Add `write_molden` and `read_molden` to the workload and TrimCheck, and the coverage row.
- [ ] **Step 5: Commit** `Write and read Molden files`.

### Task 7: Input types and vocabularies

**Files:**
- Create: `QuantumEngine/src/input/Input.jl`, `src/input/types.jl`, `src/input/vocabulary.jl`, `test/input_types_tests.jl`.
- Modify: `QuantumEngine.jl` (`public Input`; re-export `InputFile` and `ParsedInput`; `parse_inp` and `molecule` are exported in Task 8 and `parse_input` in Task 12a, where they are defined, since Aqua's undefined-exports check rejects an export without a binding), layers, workload, trim, coverage.

**Interfaces:**
- Consumes: `System.Atom`, `System.PointCharge`, `Errors.InputWarning`, `Units.ANGSTROM_TO_BOHR` (the `SolvationSpec.probe_radius` default).
- Produces, in `QuantumEngine.Input`:
  - **Enums:**
    - `RunMode`: `SinglePoint, Optimization, TransitionState, NEB, Frequency, TDDFT, Spectroscopy, OptFreq, BSSE, NBO, Bands, DOS, NMR, QTAIM, IRC`;
    - `Reference`: `AutoReference, Restricted, Unrestricted`;
    - `MethodKind`: `HF, MP2, MP2_F12, BWS2, DLPNO_MP2, CCSD, CCSD_T, CCSD_F12, CCSD_T_F12, LDA, PBE, BLYP, SCAN, TPSS, M06L, M062X, PBE0, B3LYP, RPA, B88_VDW, PBE_VDW, B97X_D, WB97M_V, B2PLYP, MPW2PLYP, DSD_PBEP86, EOM_CCSD, PWDFT, CASSCF, LIBXC_GGA, LIBXC_MGGA, LIBXC_HGGA, LIBXC_HMGGA`;
    - `ComputeMode`: `CPU, GPU, Hybrid`;
    - `DIISMethod`: `CDIIS, EDIIS`;
    - `SolvationModel`: `NoSolvation, PCM, SMD`;
    - `PCMSurface`: `VdwSurface, SASSurface, SMDSurface`;
    - `VdwKernel`: `D2, D3, D3BJ, B97X_D_DISP, VV10`;
    - `TDDFTMethod`: `TDA, RPA_TD, SpinFlip`;
    - `IRCDirection`: `Forward, Backward, Both`;
    - `PopulationKind`: `Mulliken, Lowdin`;
    - `ConstraintKind`: `FreezeAtom, FreezeBond, FreezeAngle, FreezeDihedral`;
    - `InputFormat`: `INP, JSON_INPUT, YAML_INPUT, XYZ, PDB, CIF`.
  - **Structs:**
    - `struct MethodSpec; kind::MethodKind; reference::Reference; ints::NTuple{2,Int}; exx::Float64; end`. The CASSCF (e, o) and libxc IDs go in `ints`, the hybrid fraction in `exx`, and unused slots are 0.
    - `struct Constraint; kind::ConstraintKind; atoms::NTuple{4,Int}; target::Union{Nothing,Float64}; end`. Atoms are 1-based and unused slots 0; the target is in bohr or radians.
    - `struct HubbardU; symbol::String; U::Float64; end`.
    - `struct ONIOMSpec; high::MethodSpec; low::MethodSpec; model_atoms::Vector{Int}; model_charge::Union{Nothing,Int}; model_multiplicity::Union{Nothing,Int}; end`.
    - `Base.@kwdef struct SolvationSpec; model::SolvationModel = NoSolvation; solvent::Symbol = :water; epsilon::Float64 = 78.39; custom_epsilon::Bool = false; radii_scaling::Float64 = 1.2; grid_order::Int = 110; surface::PCMSurface = VdwSurface; probe_radius::Float64 = 0.40 * ANGSTROM_TO_BOHR; end`. The defaults come from `solvation/mod.rs:29-37` and `smd.rs:319-326`. Under SMD, `custom_epsilon == true` means a user `epsilon` replaces the named solvent's dielectric, the oracle's `Solvent::Custom` (D32).
  - **`Base.@kwdef struct InputFile`**, with the fields and defaults of the table below. All atom indices are 1-based: the oracle's 0-based indices are converted at parse time.
  - **`struct ParsedInput; input::InputFile; warnings::Vector{InputWarning}; end`.**
  - StrictMode: the 13 enums add no unlisted method (Task 1's `@enum` skip). `InputFile`, `SolvationSpec` (each with its keyword body, `:_`), `ONIOMSpec`, `ParsedInput` and `HubbardU` are exempted as allocating constructors where their constructors allocate; the vocabulary functions and the `accepted_*` functions are allocation-free and are not exempted. Any helper that must allocate is exempted by name, with a reason naming the entry point that calls it.
  - **Vocabulary functions**, each returning `nothing` for an unknown word so that the parser decides between strict and lenient. Each compares its argument with the table case-insensitively without allocating: a byte-wise ASCII comparison, never `lowercase`. Each has a companion listing its accepted words for error messages: `accepted_run_modes`, `accepted_methods`, `accepted_solvation_models`, `accepted_solvents`, `accepted_vdw_kernels`, `accepted_tddft_methods`, `accepted_grids`, `accepted_irc_directions`, `accepted_compute_modes`, `accepted_diis_methods`, `accepted_populations` and `accepted_surfaces`, each `()::Vector{String}` returning a module-level `const` vector (no allocation per call).
    - `run_mode_word(w)::Union{Nothing,RunMode}`: the aliases of `parser.rs:64-81`.
    - `method_word(tokens::AbstractVector{<:AbstractString})::Union{Nothing,MethodSpec}`, with the reference rules below. It reads the tokens with `firstindex` and offsets from it, so a `view` of a token vector works.
    - `solvation_word(w)::Union{Nothing,SolvationModel}`: `pcm` and `true` → PCM; `smd` → SMD; `none` and `false` → NoSolvation (`parser.rs:202-211`).
    - `solvent_word(w)::Union{Nothing,Symbol}`: compared with the table while skipping the bytes `-`, `_` and space on both sides, without building a normalized string; the list of `smd.rs:281-305`.
    - `vdw_kernel_word(w)`, `tddft_method_word(w)`, `grid_word(w)`, `irc_direction_word(w)`, `compute_mode_word(w)`, `diis_word(w)`, `population_word(w)` and `surface_word(w)`.

**`InputFile` fields.** Columns: the `.inp` keyword and its aliases; the tokens after the keyword, with the `parser.rs` line that reads them; the field with its concrete type and default. The defaults are `types.rs:228-311`. Every field is written `field::Type = default`, so `Base.@kwdef` never makes a field `Any`. Token notation: `Int`, `Float`, `Bool` (`true`/`false`, D8), `word` (one token), `1+ words`, `[x]` optional. Block contents are in Task 8.

| keyword (aliases) | tokens (`parser.rs`) | field :: type = default |
|---|---|---|
| `geometry`, `product_geometry` blocks; `charges` block | `[unit]` (307, 319); none (331) | `atoms::Vector{Atom{Float64}} = Atom{Float64}[]`, `product_atoms::Vector{Atom{Float64}} = Atom{Float64}[]`, `point_charges::Vector{PointCharge{Float64}} = PointCharge{Float64}[]` |
| `title` | `1+ words`, joined by single spaces (84) | `title::String = ""` |
| `run`, `run_mode`; `optimize` | `word` (61); none (57) | `run_mode::RunMode = SinglePoint` |
| `charge` | `Int` (270) | `charge::Int = 0` |
| `multiplicity` | `Int` (275) | `multiplicity::Int = 1` |
| `basis` | `word` (377) | `basis::String = "sto-3g"` (lowercased; alias resolution in 02) |
| `method` | `word`; `libxc_gga x c` and `libxc_mgga x c`: `word Int Int`; `libxc_hgga id exx` and `libxc_hmgga id exx`: `word Int Float`; `casscf(e,o)` is one token (383, 1079-1167) | `method::MethodSpec = MethodSpec(HF, AutoReference, (0,0), 0.0)` (D29; the oracle's default is `Method::RHF`, `types.rs:236`) |
| `scf_convergence` | `Float` (539) | `scf_convergence::Float64 = 1e-6` |
| `max_iterations` | `Int` (544) | `max_iterations::Int = 50` |
| `diis_method` | `word` (549) | `diis_method::DIISMethod = CDIIS` |
| `conv_tol_density` | `Float` (179) | `conv_tol_density::Union{Nothing,Float64} = nothing` |
| `diis_reset`, `diis_reset_threshold` | `Float` (182) | `diis_reset_threshold::Float64 = 0.1` |
| `damping` | `Float` (161) | `damping::Float64 = 0.0` |
| `smearing`, `smear` | `Float` (165) | `smearing::Float64 = 0.0` |
| `grid`, `grid_level`, `xc_grid` | `word` or `Int` (168) | `grid_level::Int = 1` (coarse 0, standard/medium 1, fine 2, ultrafine 3, superfine 4, or 0–4) |
| `level_shift` | `Float` (191) | `level_shift::Float64 = 0.0` |
| `auto_stabilize`, `autostabilize` | `Bool` (185) | `auto_stabilize::Bool = true` |
| `robust_convergence`, `robust` | `Bool` (188) | `robust_convergence::Bool = false` |
| `field`, `electric_field` | `Float Float Float` (194) | `electric_field::Union{Nothing,SVector{3,Float64}} = nothing` |
| `density_fitting`, `ri_j` | `Bool` (113) | `use_ri_j::Bool = false` |
| `ri_k` | `Bool` (116) | `use_ri_k::Bool = false` |
| `aux_basis` | `word` (267) | `aux_basis::Union{Nothing,String} = nothing` (as written) |
| `gpu`; `compute_mode` | `Bool` (119); `word` (126) | `compute_mode::ComputeMode = CPU` |
| `checkpoint` | `word` (137) | `checkpoint::Union{Nothing,String} = nothing` |
| `load_checkpoint`, `restart` | `Bool` (140) | `load_checkpoint::Bool = false` |
| `ts_target_mode` | `Int` (89) | `ts_target_mode::Int = 0` |
| `ts_trust_radius` | `Float` (92) | `ts_trust_radius::Float64 = 0.1` |
| `neb_k`; `neb_images`; `climbing_image` | `Float` (95); `Int` (98); `Bool` (101) | `neb_k::Float64 = 0.1`, `neb_images::Int = 7`, `climbing_image::Bool = true` |
| `temperature`, `temp` | `Float` (104) | `temperature::Float64 = 298.15` |
| `pressure`, `press` | `Float` (107) | `pressure::Float64 = 1.0` |
| `symmetry_number`, `sigma` | `Float` (110) | `symmetry_number::Float64 = 1.0` |
| `dispersion` | `Bool` (143) | `dispersion::Bool = false` |
| `zora`, `zora_relativistic` | `Bool` (146) | `zora::Bool = false` |
| `f12`, `explicit_correlation`; `f12_gamma`, `gamma`, `gamma_f12`; `f12_model`, `f12_approximation` | `Bool` (149); `Float` (152); `word` (155) | `use_f12::Bool = false`, `f12_gamma::Float64 = 1.0`, `f12_model::String = "F12b"` |
| `nbo` | `Bool` (158) | `nbo::Bool = false` |
| `admm`; `admm_basis` | `Bool` (554); `word` (557) | `admm::Bool = false`, `admm_basis::Union{Nothing,String} = nothing` |
| `ecp`, `use_ecp`; `ecp_name` | `Bool` or `word` (489); `word` (501) | `use_ecp::Bool = false`, `ecp_name::Union{Nothing,String} = nothing` (D31: `ecp <name>` and `ecp_name <name>` set both) |
| `vdw_kernel` | `word` (286) | `vdw_kernel::Union{Nothing,VdwKernel} = nothing` |
| `solvation`; `solvent`; `epsilon`, `dielectric`; `pcm_radii`, `radii_scaling`; `pcm_grid`, `smd_grid` | `word` (202); `word` (212); `Float` (224); `Float` (238); `Int` (244) | `solvation::SolvationSpec = SolvationSpec()` (the defaults above) |
| `output` | `word` (252) | `output::Union{Nothing,String} = nothing` |
| `verbosity` | `Int` (255) | `verbosity::Int = 1` |
| `export_molden`, `export_cube`, `export_wfx` | `Bool` each (258-266) | `export_molden::Bool = false`, `export_cube::Bool = false`, `export_wfx::Bool = false` |
| `population`, `pop_analysis` | `1+ words` (560) | `population::Vector{PopulationKind} = [Mulliken, Lowdin]` |
| `pw_cutoff`, `cutoff` | `Float` (346) | `pw_cutoff::Union{Nothing,Float64} = nothing` |
| `kpoints`, `k_points`, `kgrid` | `Int Int Int` (349) | `kpoints::NTuple{3,Int} = (1,1,1)` |
| `dos_k_grid`, `dos_kgrid`; `dos_sigma` | `Int Int Int` (356); `Float` (363) | `dos_k_grid::NTuple{3,Int} = (10,10,10)`, `dos_sigma::Float64 = 0.1` (eV) |
| `kpath`, `k_path` | `1+ words` (366) | `kpath::Vector{String} = String[]` |
| `cell` block; `pbc` block | none; block lines (338, 622-657) | `lattice::Union{Nothing,SMatrix{3,3,Float64,9}} = nothing` (Å, rows are vectors, D22), `pbc::NTuple{3,Bool} = (true,true,true)` |
| `oniom` block (`high`, `low`: method tokens; `model_atoms`/`model_indices`: `1+ Int`; `model_charge`, `model_multiplicity`: `Int`); `oniom_high`, `oniom_low`: method tokens; `oniom_model`: `1+ Int`; `oniom_model_charge`, `oniom_model_multiplicity`: `Int` | (386-418, 475-488) | `oniom::Union{Nothing,ONIOMSpec} = nothing` |
| `freeze_atom`/`freeze`; `freeze_bond`; `freeze_angle`; `freeze_dihedral`/`freeze_torsion` (repeatable) | `1+ Int` (419); `Int Int [Float]` (426); `Int Int Int [Float]` (436); `Int Int Int Int [Float]` (446) | `constraints::Vector{Constraint} = Constraint[]` (angles in degrees → radians, bond targets bohr) |
| `irc_step_size`/`irc_step`; `irc_max_points`/`irc_points`; `irc_direction` | `Float` (456); `Int` (461); `word` (466) | `irc_step_size::Float64 = 0.10`, `irc_max_points::Int = 30`, `irc_direction::IRCDirection = Both` (forward/fwd, backward/bwd/back, both) |
| `tddft_states`, `num_tddft_states`, `nstates`; `tddft_method`; `compute_soc`/`soc`; `tddft_triplets`/`triplets` | `Int` (505); `word` (510); `Bool` (515); `Bool` (518) | `tddft_states::Int = 10`, `tddft_method::TDDFTMethod = TDA` (tda; rpa; spin_flip/spinflip/sf/sf-tddft), `compute_soc::Bool = false`, `tddft_triplets::Bool = false` |
| `dmft`/`use_dmft`; `dmft_temp`; `dmft_n_freq`/`dmft_matsubara`; `dmft_n_bath`/`dmft_bath` (YAML adds `u_map`, `j_map`) | `Bool` (521); `Float` (524); `Int` (529); `Int` (534) | `use_dmft::Bool = false`, `dmft_temperature::Float64 = 300.0`, `dmft_n_freq::Int = 1024`, `dmft_n_bath::Int = 4`, `dmft_u::Vector{HubbardU} = HubbardU[]`, `dmft_j::Vector{HubbardU} = HubbardU[]` |
| `hubbard` block | none (280) | `hubbard::Vector{HubbardU} = HubbardU[]` |
| `fragments` (inline or block) | `1+ words`, joined by single spaces, then split on `;` and `,` (289) | `fragments::Vector{Vector{Int}} = Vector{Int}[]` |

`use_fock_damping(inp) = inp.damping > 1e-10` is a function, as `parser.rs:161-164` derives it.

**Method words** (`parser.rs:1079-1167`; D11):
- **HF family:** `rhf` → (HF, Restricted); `uhf` → (HF, Unrestricted); `hf` → (HF, AutoReference).
- **Post-HF:**
  - `mp2` → MP2, Auto; `rmp2` → Restricted; `ump2` → Unrestricted.
  - `mp2-f12`, `mp2_f12`, `rmp2-f12` → MP2_F12.
  - `bws2`, `bw-s2`, `bwpt2` → BWS2; `dlpno`, `dlpno-mp2`, `dlpno_mp2` → DLPNO_MP2.
  - `ccsd` → CCSD, Auto; `rccsd` → Restricted; `uccsd` → Unrestricted.
  - `ccsd(t)` → CCSD_T, Auto; `uccsd(t)` → CCSD_T, Unrestricted.
  - `ccsd-f12`, `ccsd_f12`, `rccsd-f12` → CCSD_F12; `ccsd(t)-f12`, `ccsd(t)_f12`, `ccsd_t_f12` → CCSD_T_F12.
  - References follow this package's prefix rule, not the oracle, which drops the reference for every method word (`parser.rs:1126-1135`; `rmp2` and `ump2` both give MP2): a leading `r` gives Restricted and a leading `u` gives Unrestricted, so `rmp2-f12` and `rccsd-f12` give Restricted; every other F12 word gives AutoReference.
- **DFT and others:**
  - `lda`, `pbe`, `blyp`, `scan`, `tpss`, `pbe0` and `b3lyp`; `m06l`/`m06-l` and `m062x`/`m06-2x`.
  - `rpa`/`rirpa`, `b88-vdw`/`b88_vdw`, `pbe-vdw`/`pbe_vdw`, `b97x-d`/`b97xd`, `wb97m-v`/`wb97mv`, `b2plyp`, `mpw2plyp` and `dsd-pbep86`/`dsdpbep86`.
  - `eom-ccsd`/`eom_ccsd` and `pwdft`/`pw-dft`/`pw_dft`.
- **Parameterized:**
  - `casscf(e,o)` → CASSCF with `ints = (e,o)`.
  - `libxc_gga x c` and `libxc_mgga x c` take two `Int` IDs in `ints`. `libxc_hgga id exx` and `libxc_hmgga id exx` take one ID in `ints[1]` and the fraction in `exx`.

- [ ] **Step 1: Write the failing tests.**
  - `InputFile()` has every default of the table (one `@test` per field group).
  - **Method words.**
    - `method_word(["uhf"]).reference == Unrestricted`, `method_word(["ccsd(t)"]).kind == CCSD_T`, `method_word(["casscf(4,4)"]).ints == (4,4)`, `method_word(["libxc_hgga", "402", "0.2"]) == MethodSpec(LIBXC_HGGA, AutoReference, (402,0), 0.2)`.
    - `method_word(["dft"]) === nothing`; `method_word(["casscf(4)"]) === nothing`, since the malformed form is the parser's error.
    - `method_word(view(["x", "libxc_hgga", "402", "0.2"], 2:4))` equals the `Vector` result (`"input types: method words on a view"`).
  - **Other words.** `run_mode_word("counterpoise") == BSSE`, `solvation_word("true") == PCM`, `solvation_word("false") == NoSolvation`, `solvation_word("cosmo") === nothing`, `solvent_word("Di-chloro_methane") == :dcm` (normalization), `vdw_kernel_word("grimme-d2") == D2`, `vdw_kernel_word("pbe-vdw") === nothing`, `grid_word("ultrafine") == 3`.
  - **Coverage of the oracle lists.** Every alias listed in this task maps; the test holds the alias lists as literals.
  - **Allocation-free words** (`"input types: allocation-free words"`). `@test_noalloc` on `run_mode_word("Counterpoise")`, `solvent_word("Di-chloro_methane")` and `accepted_methods()`.
  - **Concrete fields** (`"input types: concrete fields"`). `all(isconcretetype, fieldtypes(InputFile))` fails, because some fields are `Union{Nothing,T}`; the item instead asserts that no field type is `Any` or abstract except a `Union{Nothing,T}` with concrete `T`: `all(T -> isconcretetype(T) || (T isa Union && all(isconcretetype, Base.uniontypes(T))), fieldtypes(S))` for `S` in `InputFile`, `SolvationSpec`, `ONIOMSpec`, `MethodSpec`, `Constraint` and `ParsedInput`.
- [ ] **Step 2: Run** with `QUANTUMENGINE_ONLY=input_types_tests`. Expected: FAIL.
- [ ] **Step 3: Implement** `types.jl` and `vocabulary.jl`. The word tables are tuples of `(word, value)` pairs, scanned linearly, so lookups are concrete and trimmable.
- [ ] **Step 4: Run** the items, then the full suite. Expected: PASS. Add the vocabulary functions to TrimCheck and the workload, and the coverage rows.
- [ ] **Step 5: Commit** `Add the typed input file and the oracle's word tables`.

### Task 8: The strict `.inp` parser

**Files:**
- Create: `QuantumEngine/src/input/inp.jl`, `test/inp_tests.jl`.
- Modify: `Input.jl`, `src/input/types.jl` (`_finalize_input`), `QuantumEngine.jl` (export `parse_inp` and `molecule`), `juliac/entry.jl` (now: read the `.inp` given in ARGS with `parse_inp`, build the molecule, print `title`, `nelectrons`, `nalpha`, `nbeta` and `fmt_fixed(nuclear_repulsion, 20, 12)`, one per line; an error is printed with `showerror`, so its message goes through `Errors._print_message`), `juliac_tests.jl` (the run reads `Fixtures.oracle_input("examples/h2.inp")`), workload, trim, coverage, `docs/src/input.md`, `docs/make.jl` (the Input page).
- StrictMode: `parse_inp`, `molecule`, `accepted_keywords` and their allocating helpers are exempted by name as entry points; the parser's state struct is exempted as an allocating constructor if its constructor allocates.

**Interfaces:**
- Consumes: Task 7's types and words; `System.Atom`, `Molecule`, `nalpha`; `Units.ANGSTROM_TO_BOHR`; `Errors.InputError`, `InputWarning`; `IO.input_line`, `IO.reject_or_warn` (Task 4).
- Produces:
  - **`parse_inp(text::AbstractString; source::AbstractString = "<string>", lenient::Bool = false)::ParsedInput`.**
  - **`_finalize_input(inp::InputFile; source::AbstractString, lenient::Bool, warnings::Vector{InputWarning}, method_line::Int = 0, multiplicity_line::Int = 0)::InputFile`**, in `types.jl`: the checks every input format shares, run once after the parse: no atoms, a restricted reference with multiplicity ≠ 1 (`:rhf_open_shell`), and the electron parity. `parse_inp` (here) and `parse_structured` (Task 10) both call it; neither repeats these checks. Its messages are those of "After the file" below; the structured path passes line 0.
  - **`molecule(inp::InputFile)::Molecule{Float64, Vector{Atom{Float64}}}`.**
  - **`accepted_keywords()::Vector{String}`:** sorted, with every alias.

**Strict grammar** (the lenient branches are Task 9's):
- **Lines.** In this order, for each line:
  1. `IO.input_line(raw, n)`: strip a leading BOM (first line only) and a trailing `\r`, then surrounding whitespace; it also returns the tokens used in 3.
  2. Title rule, before any comment handling: if the line starts with `#` and no title has been set yet, `title = strip(line[nextind(line, 1):end])` (the oracle's `line[1..].trim()`, `parser.rs:43-47`); the line is then skipped. Any other line starting with `#` is skipped. A `title` keyword overrides the comment title wherever it appears.
  3. Split tokens on whitespace (spaces and tabs). A token starting with `#` ends the line (D9). A line that is empty after this is skipped.
- **Keywords.** The keyword is token 1, lowercased.
  - Strict looks up keywords only outside blocks.
  - An unknown keyword throws `InputError(source, n, "unknown keyword `foo_bar`; accepted keywords: <accepted_keywords() joined by \", \">")`.
  - The tokens each keyword takes are the "tokens" column of Task 7's table. A keyword that gets a different count throws `"`charge` takes 1 value, got 2"`; for an optional token the message names the range (`"`freeze_bond` takes 2 or 3 values, got 1"`); a `1+` keyword with none throws `"`kpath` needs a value"`. A block opener that takes no value (`hubbard`, `cell`, `pbc`, `oniom`) and gets a token on its line throws exactly `"`hubbard` takes no value on its line"`, not the count template.
  - `title` takes the rest of the line, joined by single spaces, with one surrounding pair of `"` removed (D9).
  - `ecp` takes `true`, `false` or a name (D31).
- **Value types.**
  - Bool: exactly `true` or `false` (D8).
  - Integer: Julia `parse(Int, …)`.
  - Float: `parse(Float64, …)`, then `isfinite`, else `"expected a finite number, got \"nan\""` (Review Focus 4). A token that parses as a non-finite float (`nan`, `inf`) is the `:nonfinite` case for every keyword, checked before any per-keyword fallback (Task 9).
  - Word values go through Task 7's word functions. An unknown word throws `"unknown value `x` for `compute_mode`; accepted: cpu, gpu, hybrid"`.
- **Duplicates.** A scalar keyword given twice throws `"keyword `charge` given twice (lines 2 and 9)"`, aliases included. The `freeze_*` keywords are repeatable. A second `geometry` or `product_geometry` block throws `InputError(source, 9, "second geometry block (first at line 4)")`, which names both lines when printed: `input: <src>:9: second geometry block (first at line 4)` (Review Focus 1; the lenient case is `:duplicate_block`, Task 9).
- **Blocks.** `geometry [unit]`, `product_geometry [unit]`, `charges [unit]`, `hubbard`, `fragments`, `cell`, `pbc` and `oniom` open a block, and each closes with `end` (D7).
  - The unit is `angstrom`, `ang`, `bohr` or `au`, case-insensitive, and the default is angstrom per block; no unit carries over (class B). The oracle's `charges` takes no unit and uses the last geometry block's unit; here `charges` takes an optional unit with the Å default (class B, D6).
  - A line inside a block that is not that block's content throws `"`charge` inside the geometry block opened at line 6; close the block with `end`"`. An unclosed block at the end of the file throws `"the geometry block opened at line 6 is not closed by `end`"`.
- **Block contents.**
  - geometry: `Sym x y z` through `System.Atom`, or `Bq x y z q`, which is a point charge.
  - charges: `q x y z`.
  - hubbard: `Sym U`.
  - fragments: each block line, and the inline value of the `fragments` keyword, is its tokens joined by single spaces, then split on `;` into fragments and each fragment on `,` into indices, each index stripped (so `0,1,2 ; 3,4,5` of `tests/stress_dimer_bsse.in` reads two fragments). Indices are 0-based in the file and converted to 1-based; all block lines are appended (class B). A fragment that is empty or holds a non-integer is an error.
  - cell: exactly 3 lines of 3 floats in Å (D22).
  - pbc: one line of exactly 3 tokens, each `true` or `false` (D8; strict rejects the oracle's `1`, `0`, `T` and `F`, which are `:bool_fallback` in lenient mode).
  - oniom: the keywords `high`, `low`, `model_atoms`, `model_indices`, `model_charge` and `model_multiplicity`.
- **The `fragments` keyword** with ≥ 2 non-empty fragments inline needs no block.
- **After the file.** The first two checks and the parity check below are `_finalize_input`'s; the solvation and F12 checks stay in `parse_inp`.
  - No atoms throws `InputError(source, 0, "no atoms: the input has no geometry block")` (Review Focus 3).
  - `rhf`, `rmp2` or `rccsd` with multiplicity ≠ 1 throws at the method line: `"method `rhf` requests a restricted reference, but multiplicity is 3"`.
  - Solvation resolves order-independently, so the line order of the solvation keywords never changes the result:
    - the model is the `solvation` word if given (`solvation_word`); otherwise `solvent` gives SMD, and `epsilon` alone gives PCM;
    - `solvation pcm` (or `true`) together with `solvent` throws `"`solvent` needs `solvation smd`"`; `solvation none` (or `false`) with any other solvation keyword throws;
    - under PCM, `epsilon`, `pcm_radii` and `pcm_grid` apply; under SMD, `solvent`, `pcm_grid` and `epsilon` apply, and `epsilon` sets `custom_epsilon = true` (D32); `pcm_radii` under SMD throws;
    - `pcm_radii` or `pcm_grid` without a model throws `"`pcm_radii` needs `solvation pcm`"`.
  - `f12_gamma` or `f12_model` without `f12 true` or an F12 method throws.
  - The electron parity is checked through `nalpha(molecule(inp))`, with the `ArgumentError` rethrown as an `InputError` at the multiplicity line.
- **Type confusion (D26).** A first keyword token that ends in `:`, or a line starting with `- `, throws `"`basis:` looks like YAML; .inp files use `keyword value` lines, and YAML input needs a .yaml or .yml extension"`.

- [ ] **Step 1: Write the failing tests.** The items are named `"inp: h2.inp values"`, `"inp: units"`, `"inp: unknown keyword"`, `"inp: value counts"`, `"inp: value types"`, `"inp: duplicates"`, `"inp: blocks"`, `"inp: no atoms"`, `"inp: methods and references"`, `"inp: solvation"`, `"inp: f12 options"`, `"inp: type confusion"`, `"inp: line endings"`, `"inp: comments, title and quotes"`, `"inp: atom lines and blocks"`, `"inp: class B readings"` and `"inp: juliac entry"`. Each error item holds one message group of the grammar above.
  - **Values.** `Fixtures.oracle_input("examples/h2.inp")` parses to `title == "Hydrogen molecule"`, `charge 0`, `multiplicity 1`, `basis "sto-3g"`, `method.kind == HF`, `scf_convergence == 1e-6`, `max_iterations == 50`, and atoms H at (0, 0, ∓0.7) bohr (`==`: no conversion).
  - **Units.**
    - `geometry angstrom` with `O 0.0 0.0 0.1173` gives z == 0.1173 * ANGSTROM_TO_BOHR (`==`, one multiplication).
    - `geometry au` is bohr.
    - A `product_geometry` without a unit after `geometry bohr` is Å (class B: the oracle reads bohr). A `charges` block without a unit after `geometry bohr` is Å (class B).
  - **Errors** (`@test_throws`): each message in the grammar above. The unknown-keyword message also contains `"accepted keywords: "` and `"scf_convergence"`.
  - **Solvation.** Each ordering of the `:order_dependent` rows of Task 9 gives the same strict result in both line orders: `epsilon 4.0` and `solvation pcm` give PCM with epsilon 4.0; `solvation smd` and `solvent dcm` give SMD with `:dcm`; `solvation smd`, `solvent dcm` and `epsilon 9.0` give SMD, `:dcm`, epsilon 9.0, `custom_epsilon == true`.
  - **Review Focus 1–4:**
    - duplicates and second blocks: a second `geometry` block at line 9 after one at line 4 throws the full printed form `"input: <src>:9: second geometry block (first at line 4)"`, and likewise for `product_geometry`;
    - CRLF, BOM and tabs give the same `InputFile` as the LF version (`fields_equal`);
    - an empty file, a file of comments only, and a file of keywords only each throw "no atoms";
    - `scf_convergence nan` and `damping inf` throw "expected a finite number".
  - **Inline comments and quotes.** `charge 0 # neutral` parses; `title "H2O MP2-F12 Test"` gives `H2O MP2-F12 Test`; a `# first` line followed by `# second` gives title `first`; `# first` followed by `title second` gives `second`.
  - **Atom lines.** `Bq 0 0 1 0.5` gives a point charge at z = 1 Å with q = 0.5. The cell block gives a `lattice` with rows in Å. The fragments block of `Fixtures.oracle_input("tests/stress_dimer_bsse.in")` gives `[[1,2,3],[4,5,6]]`. `ecp def2` gives `use_ecp == true`, `ecp_name == "def2"`. `hubbard O:5.0` throws "`hubbard` takes no value on its line".
  - **Class B readings** (`"inp: class B readings"`, D6; each holds in both modes with no warning):
    - substring scans: text that contains `cell` or `pbc` without opening that block (for example `title supercell pbc test`; the implementer reads at `parser.rs:338, 622-657` which text the oracle's scans match, builds the inputs from it and cites the lines) leaves `lattice === nothing` and `pbc == (true, true, true)`;
    - multi-line `fragments` block: lines `0,1,2` and `3,4,5` give `fragments == [[1,2,3],[4,5,6]]` (the oracle keeps only the last line).
  - **juliac.** The untrimmed `juliac/entry.jl` on `examples/h2.inp` prints 5 lines: `Hydrogen molecule`, `2`, `1`, `1` and `fmt_fixed(1/1.4, 20, 12)`.
- [ ] **Step 2: Run** with `QUANTUMENGINE_ONLY=inp_tests`. Expected: FAIL.
- [ ] **Step 3: Implement** `inp.jl`.
  - One pass over the lines keeps a state struct: the open block, its start line, the first line of each keyword seen, the method line, and the solvation keywords seen (resolved after the pass).
  - Each line is cleaned and tokenized by `IO.input_line`. Each place where the oracle silently accepts something calls `IO.reject_or_warn(state.source, line, state.lenient, state.warnings, case::Symbol, msg)` (Task 4): in strict mode it throws `InputError(state.source, line, msg)`; in lenient mode it pushes `InputWarning` and returns, and the caller then applies the oracle value (Task 9).
  - After the pass, `parse_inp` resolves solvation and the F12 options, then calls `_finalize_input` with the method and multiplicity lines.
- [ ] **Step 4: Run** the items, then the full suite. Expected: PASS.
  - Add `parse_inp` to the workload and to TrimCheck.
  - Add the coverage row.
  - Run the juliac item (`QUANTUMENGINE_JULIAC_BUILD=1 QUANTUMENGINE_ONLY=juliac_tests`). Expected: PASS, with the binary output equal to the untrimmed run.
- [ ] **Step 5: Document** the grammar, the deviations from the oracle (class B, including the `charges` unit, and the strict rules above) and the keyword table in `docs/src/input.md`, and add the Input page to `docs/make.jl`.
- [ ] **Step 6: Commit** `Parse .inp input files strictly`.

### Task 9: `--lenient`: one test per case the oracle accepts silently

**Files:**
- Create: `QuantumEngine/test/lenient_tests.jl`.
- Modify: `src/input/inp.jl` (the lenient branches); `docs/src/input.md` (the case table, including Task 4's L19a–L19h, its recorded xyz exception, and Task 12a's `:xyz_frames`); `coverage/thresholds.toml` (the `inp.jl` row). The readers of Task 4 carry L19a–L19h and their tests.

**Interfaces:**
- Consumes: `parse_inp(text; lenient = true)`.
- Produces: each case below is a `Symbol` in `InputWarning.case`. That list is the public record of lenient behavior.

The oracle values are those of `src/input/parser.rs`, `io/molecule.rs`, `types.rs` and the consumers named at 4a78a1d; the implementer cites the line in each test. For the cases marked "verified", the value was confirmed by running the oracle binary on a probe (appendix E2).

**Assertions.** Unless a row says otherwise, each case gets one `@testitem "lenient: <case>"` with two assertions: strict throws the message substring; lenient gives the stated value, with exactly one warning of that case. The rows that do not fit that template state their own assertions in the table.

**The per-keyword table.** This is the complete record of what `--lenient` does with a bad or missing value of each `.inp` keyword (D6). "bad" is a token that does not parse as the keyword's type (Task 7's tokens column) or a word outside its vocabulary; "missing" is no token. The values are the oracle's. In strict mode every "bad" and "missing" entry is an error, except the rows whose class is "oracle error", which are errors in both modes. The item `"lenient: per-keyword table"` holds this table as a literal and loops over it: for each row and each of "bad" (input `<keyword> zzz`, or for a list keyword one entry `zzz`) and "missing" (input `<keyword>` alone), strict throws, and lenient gives the stated value with one warning of case `:bool_fallback`, `:number_fallback`, `:unknown_value`, `:missing_value`, `:too_few_values` or `:bad_list_entry` (as named in the row), or, for an "oracle error" row, throws in lenient mode too.
- A row whose bad column is `—` has no bad input; the loop runs only its "missing" half.
- A row whose bad token is valid in strict mode is skipped by the strict half of the loop for that token: `ecp zzz` is an ECP name in both modes (D31), and the item asserts instead that it parses with `ecp_name == "zzz"` and no warning.
- A token that parses as a non-finite float (`nan`, `inf`) goes to `:nonfinite` (case table) before any per-keyword fallback, so `damping inf` is `:nonfinite` with the parsed `Inf`, not `:number_fallback`. The loop's bad token `zzz` never parses as a float.

| keyword (aliases) | `parser.rs` | bad (lenient) | missing (lenient) | case |
|---|---|---|---|---|
| `run`, `run_mode` | 61-83 | oracle error `Unknown run mode` | unchanged | — / `:missing_value` |
| `title` | 84-88 | — | unchanged | `:missing_value` |
| `charge`, `multiplicity`, `scf_convergence`, `max_iterations`, `ts_target_mode`, `ts_trust_radius`, `neb_k`, `neb_images`, `temperature`/`temp`, `pressure`/`press`, `symmetry_number`/`sigma` | 89-110, 270-278, 539-547 | oracle error (`ok_or(...)?`) | oracle error | — |
| `basis` | 377-382 | — | oracle error `Missing basis set` | — |
| `method`, `high`, `low`, `oniom_high`, `oniom_low` | 383-409, 1079-1167 | oracle error `Unknown method` | oracle error `Missing method` | — |
| `geometry`, `product_geometry` unit | 307-330 | oracle error `Unknown geometry unit` | — | — |
| `climbing_image`, `density_fitting`/`ri_j`, `ri_k`, `gpu`, `load_checkpoint`/`restart`, `dispersion`, `zora`/`zora_relativistic`, `f12`/`explicit_correlation`, `nbo`, `auto_stabilize`/`autostabilize`, `robust_convergence`/`robust`, `dmft`/`use_dmft`, `admm` | 101, 113-122, 140-150, 158, 185-190, 521, 554 | true (`gpu`: GPU) | true | `:bool_fallback` |
| `export_molden`, `export_cube`, `export_wfx` | 258-266 | false | false | `:bool_fallback` |
| `compute_soc`/`soc`, `tddft_triplets`/`triplets` | 515-520 | `1` → true, any other token → false | true | `:bool_fallback` |
| `compute_mode` | 126-136 | unchanged | unchanged | `:unknown_value` / `:missing_value` |
| `checkpoint`, `aux_basis`, `admm_basis` | 137, 267, 557 | — | `nothing` | `:missing_value` |
| `f12_gamma`/`gamma`/`gamma_f12` | 152 | 1.0 | 1.0 | `:number_fallback` / `:missing_value` |
| `f12_model`/`f12_approximation` | 155 | — | `"F12b"` | `:missing_value` |
| `damping`, `smearing`/`smear`, `level_shift` | 161, 165, 191 | 0.0 | 0.0 | `:number_fallback` / `:missing_value` |
| `diis_reset`/`diis_reset_threshold`, `dos_sigma` | 182, 363 | 0.1 | 0.1 | `:number_fallback` / `:missing_value` |
| `verbosity` | 255 | 1 | 1 | `:number_fallback` / `:missing_value` |
| `grid`/`grid_level`/`xc_grid` | 168-178 | 1 | 1 | `:unknown_value` / `:missing_value` |
| `conv_tol_density`, `pw_cutoff`/`cutoff` | 179, 346 | `nothing` | `nothing` | `:number_fallback` / `:missing_value` |
| `field`/`electric_field` | 194-201 | entry → 0.0 | unchanged (`nothing`); fewer than 3 values → unchanged | `:bad_list_entry` / `:too_few_values` |
| `kpoints`/`k_points`/`kgrid` | 349-355 | entry → 1 | unchanged; fewer than 3 → unchanged | `:bad_list_entry` / `:too_few_values` |
| `dos_k_grid`/`dos_kgrid` | 356-362 | entry → 10 | unchanged; fewer than 3 → unchanged | `:bad_list_entry` / `:too_few_values` |
| `solvation` | 202-211 | unchanged | unchanged | `:unknown_value` / `:missing_value` |
| `solvent` | 212-223 | unchanged | `water` (and the model becomes SMD) | `:unknown_value` / `:missing_value` |
| `epsilon`/`dielectric` | 224-237 | 78.39, then the model rules of that line | 78.39, likewise | `:number_fallback` / `:missing_value` |
| `pcm_radii`/`radii_scaling` | 238-243 | 1.2 (applied only under PCM) | 1.2, likewise | `:number_fallback` / `:missing_value` |
| `pcm_grid`/`smd_grid` | 244-251 | 110 | 110 | `:number_fallback` / `:missing_value` |
| `output` | 252-254 | — | `""` (the oracle then writes no log) | `:missing_value` |
| `vdw_kernel` | 286-288 | `nothing` (the oracle stores the word and computes no dispersion) | `nothing` | `:unknown_value` / `:missing_value` |
| `kpath`/`k_path`, `population`/`pop_analysis` | 366, 560 | `population`: the default list | empty list; `population`: empty list | `:unknown_value` / `:missing_value` |
| `model_atoms`/`model_indices`, `oniom_model`, `freeze_atom`/`freeze` | 396-402, 411-425 | the entry is dropped | nothing added | `:bad_list_entry` / `:missing_value` |
| `freeze_bond`, `freeze_angle`, `freeze_dihedral`/`freeze_torsion` | 426-455 | a non-integer index drops the constraint; a bad target gives `target = nothing` | constraint dropped (also with too few indices) | `:bad_list_entry` / `:too_few_values` |
| `irc_step_size`/`irc_step`, `irc_max_points`/`irc_points`, `tddft_states`/`num_tddft_states`/`nstates`, `dmft_temp`, `dmft_n_freq`/`dmft_matsubara`, `dmft_n_bath`/`dmft_bath` | 456-465, 505-509, 524-538 | unchanged | unchanged | `:number_fallback` / `:missing_value` |
| `irc_direction` | 466-474 | Both | unchanged | `:unknown_value` / `:missing_value` |
| `tddft_method` | 510-514 | TDA (the oracle stores the word; `engine/electronic.rs:292-296` maps an unknown word to TDA) | unchanged | `:unknown_value` / `:missing_value` |
| `diis_method` | 549-553 | CDIIS (dead keyword, D10) | unchanged | `:unknown_value` / `:missing_value` |
| `model_charge`, `model_multiplicity`, `oniom_model_charge`, `oniom_model_multiplicity` | 475-488 | `nothing` | `nothing` | `:number_fallback` / `:missing_value` |
| `ecp`/`use_ecp` | 489-500 | (a non-boolean word is a name, accepted in both modes, D31) | `use_ecp = true` | `:bool_fallback` |
| `ecp_name` | 501-504 | — | `use_ecp = true`, `ecp_name = nothing` | `:missing_value` |

**The case table.** The cases that are not a bad or missing value of one keyword:

| case | input | strict message | lenient value |
|---|---|---|---|
| `:unknown_keyword` | `use_diis true` | "unknown keyword `use_diis`" | ignored (verified) |
| `:dropped_block_line` | `END_GEOMETRY` inside geometry | "`END_GEOMETRY` inside the geometry block" | dropped; atoms kept (neb_test.inp) |
| `:block_keyword` | `charge 1` inside an unclosed geometry block | "inside the geometry block opened at line" | `charge == 1`, atoms kept (keyword-first) |
| `:unclosed_block` | geometry with no `end` at EOF | "is not closed by `end`" | atoms kept |
| `:hubbard_inline_argument` | `hubbard O:5.0` then `use_diis true` | "`hubbard` takes no value on its line" | Both modes throw. Assertions: strict throws "`hubbard` takes no value on its line"; lenient throws `"Invalid U value: true"` (verified: the next 2-token line is read as a U line). The warning is checked on the variant `hubbard O:5.0` followed by `end`: lenient gives `hubbard == []` with one `:hubbard_inline_argument` warning |
| `:fragment_index` | `fragments 0,x;1,2` | "fragment index `x` is not an integer" | index dropped |
| `:nonfinite` | `scf_convergence nan`; `damping inf` | "expected a finite number" | the parsed value: NaN, Inf (Rust `f64` parses `nan` and `inf`); the molecule-file readers use the same case for coordinates (Task 4, L19g) |
| `:extra_values` | `charge 0 1`; `optimize now` | "`charge` takes 1 value, got 2" | 0; `run_mode == Optimization` |
| `:duplicate_keyword` | `charge 0` … `charge 1` | "given twice" | 1 |
| `:duplicate_block` | a second `geometry` block (line 9) after one at line 4; likewise `product_geometry` | `"input: <src>:9: second geometry block (first at line 4)"` (Task 8) | the oracle's result for a second block, which the implementer reads from `parser.rs:307-330`, cites in the test and states (atoms appended or replaced), with one warning |
| `:order_dependent` (1) | `pcm_radii 1.1` before `solvation pcm` | none | Assertions: strict gives `radii_scaling == 1.1` with no warning; lenient gives 1.2 (the oracle ignores it before a PCM model) with one warning |
| `:order_dependent` (2) | `epsilon 4.0` before `solvation pcm` | none | strict: epsilon 4.0, no warning; lenient: 78.39 (`solvation pcm` resets the options, `parser.rs:205`), one warning |
| `:order_dependent` (3) | `solvation smd` after `solvent dcm` | none | strict: SMD, `:dcm`, no warning; lenient: SMD, `:water` (`solvation smd` resets the solvent, `parser.rs:207`), one warning |
| `:order_dependent` (4) | `solvent dcm` after `solvation pcm` | "`solvent` needs `solvation smd`" | lenient: SMD with `:dcm` (`parser.rs:218-220` replaces the PCM model), one warning. Assertions: the template |
| `:order_dependent` (5) | `solvent dcm` before `solvation pcm` | "`solvent` needs `solvation smd`" | lenient: PCM with defaults (the later `solvation pcm` replaces the SMD model), one warning |
| `:option_without_model` | `pcm_radii 1.1` with no solvation | "`pcm_radii` needs `solvation pcm`" | ignored |
| `:casscf_shape` | `method casscf(4)` | "casscf needs (electrons,orbitals)" | (2,2) |
| `:ghost_unchecked` | `Foo@ 0 0 0`; `Bq 0 0 0` | "unknown element symbol"; the Bq message | built with the inner constructor `Atom{Float64}(0, symbol, position, true, 0.0)`: Z = 0, the symbol as written, ghost, mass 0; `basis_element` of such an atom throws `"system: ghost atom \"Foo@\" stands for no element"` |
| `:rhf_open_shell` | `method rhf`, `multiplicity 3` | "requests a restricted reference" | AutoReference |
| `:f12_option_alone` | `f12_gamma 0.9` alone | "needs `f12 true`" | gamma 0.9, `use_f12 == false` (`.inp` does not enable it) |
| `:yaml_in_inp` | the content of `molecules/h2_ccsd.inp` | "looks like YAML" | Both modes throw. Assertions: strict throws "looks like YAML"; lenient throws `"no atoms"` (the oracle's outcome, verified). The warning is not checked: the lenient run returns no `ParsedInput` |
| `:bool_fallback` (pbc) | `pbc` block line `1 T F` | "expected true or false" | `(true, true, false)` |

The reader cases L19a–L19h (`:xyz_count`, `:unknown_element`, `:pdb_short_line`, `:pdb_element_columns`, `:cif_row`, `:cif_coordinate`, `:nonfinite` for coordinates, `:cif_columns`) are specified and tested in Task 4, and `docs/src/input.md` lists them. Task 4 also records one input that lenient mode does not reproduce: an xyz frame with more atom lines than its count (the oracle ignores the extra lines; `read_xyz` reads the next line as a count). `:xyz_frames` belongs to `parse_input` and is specified and tested in Task 12a.

**Unclosed blocks.** A block keyword that ends an unclosed block emits `:unclosed_block` and, if that keyword is also out of place, `:block_keyword`. The implementer confirms this rule against `parser.rs:563-618` and reports if the source disagrees. Step 1 also derives, from the oracle source with cited lines, the expected warning list (and, for a lenient error, its message) of `tests/o2_hubbard.in` and `tests/stress_h2o_complex.in`, and pins each in an item `"lenient: <file name> cases"`. Tasks 12a and 12b use the same lists.

The inputs `dispersion False` (verified: lenient `dispersion == true`), `export_molden 1`, `compute_soc 1`, bare `nbo`, `damping abc`, `grid abc`, `irc_step abc`, `conv_tol_density abc`, `pw_cutoff abc`, `field 0.1 0.2`, `kpoints 2 2`, `freeze_bond 1`, `kpoints 2 2 x`, `model_atoms 0 x 2`, `compute_mode tpu`, `solvation cosmo`, `solvent foo`, `irc_direction sideways`, `tddft_method foo`, `vdw_kernel pbe-vdw`, `diis_method foo`, `population foo`, bare `title`, bare `run` and bare `output` are all instances of the per-keyword table; each is also kept as an explicit `@test` in `"lenient: per-keyword table"`, since `.inp` files in the oracle set use some of them.

`basis_element` of a Z = 0 atom: Task 3's `basis_element(a)` derives a ghost's element from its symbol (`O@`, `BqO`, `Bq-O` → 8). The lenient `:ghost_unchecked` atom's symbol names no element, so `basis_element` throws the message above for it; Task 3 writes that branch and its error test.

The structured-input cases (wrong types, unknown keys, methods, modes, solvents and surfaces, a solvation block without a model, integer for float, top-level `f12_gamma`, constraint indices, units and missing coordinates) are Task 10's.

- [ ] **Step 1: Write the failing tests**, one item per row as specified.
- [ ] **Step 2: Run** with `QUANTUMENGINE_ONLY=lenient_tests`. Expected: FAIL on each row whose lenient branch is not yet implemented; a row that already passes (its branch written in Task 8) is named in the task report.
- [ ] **Step 3: Implement** each lenient branch at its `IO.reject_or_warn` call. The keyword-first rule inside blocks applies only in lenient mode: a line whose first token is a keyword is processed as that keyword, as `parser.rs:563-618` does.
- [ ] **Step 4: Run** the items, then the full suite. Expected: PASS. Add one lenient `parse_inp` call to the workload, and update the `inp.jl` coverage row (the four steps of Task 1 Step 4).
- [ ] **Step 5: Commit** `Accept the oracle's lenient cases under --lenient`.

### Task 10: Structured input: `Node`, the schema walker, and JSON

**Files:**
- Create: `QuantumEngine/src/input/node.jl`, `src/input/structured.jl`, `src/input/json.jl`, `test/json_tests.jl`.
- Modify: `QuantumEngine/Project.toml` (JSON, compat per D2), `QuantumEngine/test/Project.toml` (JSON with the same compat, only if a test constructs or inspects JSON.jl objects directly; the planned tests do not), `Input.jl`, `src/QuantumEngine.jl` (export `parse_json`), workload, trim, coverage, `docs/src/input.md`, `docs/src/dependencies.md`.

**Interfaces:**
- Consumes: Task 7's types and words; `IO.reject_or_warn` (Task 4); `_finalize_input` (Task 8).
- Produces:
  - **The tree type.**
    - `@enum NodeKind NullNode BoolNode IntNode FloatNode StringNode ArrayNode MapNode` (covered by Task 1's `@enum` skip).
    - `struct Node; kind::NodeKind; b::Bool; i::Int; f::Float64; s::String; items::Vector{Node}; keys::Vector{String}; end`. A map holds `keys[k] => items[k]`, in order and without duplicates.
    - The constructors `Node(nothing)`, `Node(::Bool)`, `Node(::Int)`, `Node(::Float64)`, `Node(::String)`, `Node(items::Vector{Node})` and `Node(keys::Vector{String}, items::Vector{Node})`. The map constructor throws `ArgumentError("input: duplicate key \"a\"")`. Each constructor allocates (`Node[]`, `String[]`) and is exempted as an allocating constructor.
  - **`parse_structured(root::Node; source::AbstractString, lenient::Bool = false)::ParsedInput`:** the schema of `parse_yaml_string` (`parser.rs:677-1076`), below. Every strict-or-lenient choice goes through `IO.reject_or_warn` (Task 4), and the walk ends with `_finalize_input` (Task 8) at line 0, so the no-atoms, restricted-reference and parity checks and their messages are those of `.inp`.
  - **`json_node(text::AbstractString)::Node`:** built through the public JSON.jl route chosen in Step 0 (D2). Duplicate keys throw in both modes (D3: the oracle's yaml-rust2 rejects them too).
  - **`parse_json(text; source, lenient)::ParsedInput = parse_structured(json_node(text); source, lenient)`.**

**Schema** (the key paths of the oracle). Keys are case-sensitive. "fallback" means the top-level key is read when the `scf.` key is absent; strict throws when both are present.
- **Top level:**
  - `title`, `charge` (Int), `multiplicity` (Int), `basis` (String, lowercased), `method` (String; Task 7's `method_word`, with no `libxc_*` forms), `dispersion`, `zora`;
  - `mode`, else `task.mode`;
  - `hubbard` (map Sym → Float) and `vdw_kernel`;
  - the `geometry` or `atoms` list (both present → strict error) of `{symbol, position: [x,y,z]}` or `{symbol, x, y, z}`; in the second form strict requires all three of `x`, `y`, `z`;
  - `unit`: `angstrom`, `ang`, `bohr` or `au`, case-insensitive; default Å. `au` is bohr and `ang` is Å in both modes (class B, D6: the oracle tests only `== "bohr"`, `parser.rs:1028`); any other value is `:yaml_unknown_unit`;
  - `irc_step_size`/`irc_step`, `irc_max_points`/`irc_points`, `irc_direction`;
  - `constraints`, a list of `{freeze_atom: i}`, `{freeze_bond: [i,j], target}` and `{freeze_angle: [i,j,k], target}`, 0-based in the file;
  - `f12_model`, and `f12_gamma`/`gamma`.
- **`scf`:**
  - `max_iter` (fallback `max_iterations`), `conv_energy` (fallback `scf_convergence`);
  - `use_ri_j`, `use_ri_k`, `aux_basis`;
  - `gpu` (fallback top level), `compute_mode` (fallback);
  - `diis_method`, `verbosity`, `checkpoint`, `load_checkpoint`, `damping`, `smearing`, `grid` (Int only), `auto_stabilize`, `robust_convergence`, `level_shift`;
  - `field` (3 floats);
  - `pw_cutoff` (fallback), `lattice` (3 × 3, fallback), `pbc` (fallback);
  - `admm`, `admm_basis`, `population`;
  - `kpoints` (fallbacks `k_grid`, `kpoints`), `kpath` (fallback `k_path`);
  - `use_dmft` (fallback);
  - `dmft_options`, else top-level `dmft`, else `scf.dmft`: `{temperature, n_matsubara, n_bath, u_map, j_map}`.
- **`solvation`:** `model` (pcm/smd), else `pcm: bool`, else `smd: bool`.
  - SMD: `solvent`, `grid_order`, `probe_radius` (Å).
  - PCM: `epsilon`, `radii_scaling`, `grid_order`, `surface` (vdw/sas/smd).
- **`oniom`:** `high`, `low`, `model_atoms`, `model_charge`, `model_multiplicity`.
- **`tddft`:** `num_states`, `method`, `soc`, `triplets`.
- **`exports`:** `molden`, `cube`, `wfx`.
- **`f12`:** a bool, or a map of `gamma`/`f12_gamma` and `model`/`approximation`.

**Strict and lenient rules** (`:yaml_*` cases, which cover JSON too):

| case | strict | lenient (oracle) |
|---|---|---|
| `:yaml_unknown_key` | `"scf.foo: unknown key; accepted keys: …"` | ignored |
| `:yaml_wrong_type` | `"scf.max_iter: expected an integer, got a string"` | ignored, default kept (verified) |
| `:yaml_int_for_float` (D5) | accepted, converted | ignored, default kept (verified) |
| `:yaml_unknown_method` | "unknown value `dft` for `method`" | ignored, HF (verified) |
| `:yaml_unknown_mode` | "unknown value" for mode | ignored (verified) |
| `:yaml_unknown_solvent` | "unknown value" | water |
| `:yaml_unknown_surface` | "unknown value" | vdw |
| `:yaml_solvation_without_model` | "solvation: give model, pcm or smd" | ignored |
| `:yaml_f12_gamma_enables` | "f12_gamma needs f12" | `use_f12 = true` |
| `:yaml_constraint_index` | "constraints[1].freeze_bond: expected 2 indices" | 0, 1, 2 (0-based), stored as 1, 2, 3 |
| `:yaml_bad_number` | "geometry[2].position[3]: expected a number" | 0.0 |
| `:yaml_missing_coordinate` | "geometry[2]: missing key `z`" | 0.0 (`parser.rs:1020-1023`) |
| `:yaml_unknown_unit` | "unit: unknown value `nm`; accepted: angstrom, ang, bohr, au" | Å (`parser.rs:1028`) |
| `:yaml_lattice_row` | "lattice[2]: expected 3 numbers" | row of zeros |

Errors in both modes (as in the oracle): a missing atom `symbol`, fewer than 3 position values, an unknown ONIOM method, a top-level value that is not a map, an empty document, and a repeated key (D3).

- [ ] **Step 0: Resolve the closure and choose the route** (D2), in a throwaway environment outside the repository.
  - Resolve `JSON = "1.6"` twice, once pinned to 1.6.1 and once at the newest version the bound admits (1.13.4 when this plan was written). Inspect each Manifest for non-stdlib `_jll` packages and artifacts. Record the package lists and versions in the commit body and in `docs/src/dependencies.md` (spec 01 assigns this closure check to step 01).
  - Route 1: `JSON.parse(text; dicttype = T)` with an `AbstractDict{String,Any}` subtype `T` whose `setindex!` throws on a key already present. Route 2: a walk over `JSON.lazy(text)` (`LazyValue`) using only names that JSON 1.6.1 declares `public`. For each route, on both versions: does `{"a":1,"a":2}` throw (also nested); does the walk produce the `Node` typing of Step 1; does TrimCheck pass on `json_node(::String)`.
  - Take the first route that detects duplicates on both versions; its trim result is D4's. If neither detects duplicates on both versions, stop and ask the user (D2's alternative: compat `"~1.6"` and the internal `applyobject`/`applyarray` walk). Record the commands and results in the commit body.
- [ ] **Step 1: Write the failing tests.**
  - **`Fixtures.oracle_input("benchmarks/fe2_dmft.json")`**: 2 Fe atoms at z = ±1.89 Å (`== ±1.89 * ANGSTROM_TO_BOHR`); `basis == "sto-3g"`; `scf_convergence == 1e-6`; `max_iterations == 100`; `use_dmft`; `dmft_temperature == 1000.0`; `dmft_n_freq == 512`; `dmft_n_bath == 2`; `dmft_u == [HubbardU("Fe", 4.0)]`; `dmft_j == [HubbardU("Fe", 0.9)]`.
  - **Duplicates.** `{"a":1,"a":2}` throws `"input: duplicate key \"a\""`; the same holds nested.
  - **One item per row of the table above**, strict and lenient. Each test builds its `Node` with the constructors, so the items test the walker that JSON text, `parse_dict` and the YAML extension (Task 11) all reach through `Node`.
  - **Shared checks on the structured path** (`"json: shared checks"`), one `@test_throws` each, with the `.inp` messages at line 0: a document with no `geometry`/`atoms` throws `"no atoms: the input has no geometry block"`; `"method": "rhf"` with `"multiplicity": 3` throws "requests a restricted reference" (lenient: AutoReference with one `:rhf_open_shell` warning); H2 with `"multiplicity": 2` throws "2 electrons cannot have multiplicity 2".
  - **Units, class B** (`"json: units au and ang"`, D6). `"unit": "au"` keeps the positions in bohr and `"unit": "ang"` multiplies them by `ANGSTROM_TO_BOHR` (`==`), in both modes with no warning.
  - **`json_node` scalar typing.** `1` is an `IntNode`; `1.0` and `1e-6` are `FloatNode`s; a string is a `StringNode`; `true` is a `BoolNode`; `null` is a `NullNode`.
- [ ] **Step 2: Run** with `QUANTUMENGINE_ONLY=json_tests`. Expected: FAIL.
- [ ] **Step 3: Implement.** The walker reads only `Node`s, so it is concrete and trimmable. `json_node` uses the route of Step 0 and only public JSON.jl names, so ExplicitImports' public-access check passes.
- [ ] **Step 4: Measure trim.**
  - Run TrimCheck on `parse_structured(::Node)`. Expected: PASS; a failure is reported with its chain.
  - Run it on `json_node(::String)`. Record PASS or FAIL in the commit body.
  - On FAIL, `json_node` is left out of the TrimCheck item with the comment `# json_node is not trim-safe: JSON.jl's reader has unresolved calls under TrimCheck; the trimmed entry reads .inp only.` This is D4. Report the outcome.
- [ ] **Step 5: Run** the full suite. Expected: PASS. Add the entries to the workload and the coverage rows.
- [ ] **Step 6: Commit** `Read JSON input through a typed tree`.

### Task 11: Structured input from a Dict, and YAML files through an extension

User decision D1: the reader that turns a file into a Dict is not this package's concern; what the program does with the Dict is. The core therefore accepts any `AbstractDict` (as YAML.jl, JSON.jl and other readers produce), and YAML files are read by YAML.jl in a package extension.

**Files:**
- Create: `QuantumEngine/src/input/dict.jl`, `QuantumEngine/ext/QuantumEngineYAMLExt.jl`, `QuantumEngine/test/dict_tests.jl`, `QuantumEngine/test/yaml_tests.jl`.
- Modify: `QuantumEngine/Project.toml` (`[weakdeps] YAML`, `[extensions] QuantumEngineYAMLExt = "YAML"`, `[compat] YAML` at the current release), `QuantumEngine/test/Project.toml` (YAML), `Input.jl`, `src/QuantumEngine.jl` (export `parse_dict` and `parse_yaml`), workload, trim, coverage, `docs/src/input.md`, `docs/src/dependencies.md` (YAML.jl is outside the core closure; record its closure and that the "core closure is binary-free" item does not load the extension).

**Interfaces:**
- Consumes: Task 10's `Node` and `parse_structured`.
- Produces:
  - `node(d::AbstractDict)::Node`, recursively: `AbstractDict` → map node, keys converted with `String` (a non-string key throws `InputError` naming it); `AbstractVector` → array node; `Bool`, `Integer` (in `Int` range), `AbstractFloat`, `AbstractString`, `nothing`/`missing` → the matching scalar node; anything else throws `InputError` naming the key path and the Julia type. Key order: the Dict's iteration order; the schema walker does not depend on order.
  - `parse_dict(d::AbstractDict; source, lenient = false)::ParsedInput = parse_structured(node(d); source, lenient)`, exported.
  - In `Input` (`dict.jl`): `function parse_yaml end`, with its docstring (`parse_yaml(text::AbstractString; source::AbstractString = "<string>", lenient::Bool = false)::ParsedInput` reads YAML text; it has a method only when YAML.jl is loaded), exported. The core defines no method.
  - In the extension: the method `Input.parse_yaml(text::AbstractString; source, lenient)`, which reads the text with YAML.jl into a Dict and calls `parse_dict`.
  - In `Input` (`dict.jl`): `_yaml_unavailable(source::AbstractString)::InputError`, which returns `InputError(source, 0, "reading YAML needs `using YAML` (QuantumEngineYAMLExt)")`, printed `input: <source>: reading YAML needs `using YAML` (QuantumEngineYAMLExt)`. The `.yaml`/`.yml` dispatch (Task 12a) calls `parse_yaml` when `hasmethod(parse_yaml, Tuple{AbstractString})`, and otherwise throws `_yaml_unavailable(source)`, in both modes.
- Typing: the reader decides scalar types. Where YAML.jl's Dict types a scalar differently from JSON.jl's (Step 2 measures `017`, `True`, `nan` and any other difference found), the schema walker's type checks decide: a value of the wrong type for its key is an `InputError` in strict mode and handled as Task 10's lenient table says. The differences found are listed in the task report and in `docs/src/input.md`.

- [ ] **Step 1: Failing tests** in `dict_tests.jl`: `node` on every scalar kind, nested maps and vectors, a non-string key, an unsupported value type (`InputError` with the path), `missing`; `parse_dict` on a Dict built in the test equals `parse_json` on the same content (`Fixtures.fields_equal`); `"unit" => "au"` keeps the positions in bohr and `"unit" => "ang"` multiplies them by `ANGSTROM_TO_BOHR` (`==`), in both modes with no warning (class B, D6; `"dict: units au and ang"`); `_yaml_unavailable("a.yaml")` shown with `showerror` is `"input: a.yaml: reading YAML needs `using YAML` (QuantumEngineYAMLExt)"`.
- [ ] **Step 2: Failing tests** in `yaml_tests.jl` (with `using YAML`): `tests/si.yaml`, `examples/legacy_inputs/input_hf.yaml`, `src/input/input.yaml` and the two YAML strings embedded in the oracle tests (`parser.rs:1257`, `tests/test_smd_solvation.rs:239`) give the oracle's values (Task 12b's rows); the two `.inp`-in-`.yaml` files throw "top-level YAML document is not a mapping; this file looks like .inp syntax" (D26); a repeated key throws in both modes (D3) if YAML.jl reports it, and otherwise the test records that YAML.jl keeps the last value and the docs say so. Measure each typing difference (`017`, `True`, `nan`, and others found) between YAML.jl's Dict and JSON.jl's, and record it in the task report and in docs/src/input.md; each difference found has one test stating the outcome.
- [ ] **Step 3: Implement** `dict.jl` and the extension. TrimCheck `@validate` covers `parse_dict` on a concrete `Dict{String, Any}` input only if it passes; otherwise the result is recorded for step 17 (the trimmed CLI reads `.inp` and JSON).
- [ ] **Step 4: Run** the new items and the closure item (the core closure still has no `_jll`).
- [ ] **Step 5: Commit** `Read structured input from a Dict; YAML through an extension`.

### Task 12a: Input dispatch and the fixture files

**Files:**
- Create: `QuantumEngine/src/input/dispatch.jl`, `test/inp_fixture_tests.jl`.
- Modify: `Input.jl`, `QuantumEngine.jl` (export `parse_input`), workload, trim, coverage.

**Interfaces:**
- Produces:
  - **`input_format(path::AbstractString)::InputFormat`:** the extension, lowercased (`parser.rs:15-26`). `xyz`, `pdb` and `cif` are geometry files; `yaml` and `yml` are YAML; `json` is JSON; anything else is `.inp`.
  - **`parse_input(path::AbstractString; lenient::Bool = false)::ParsedInput`.** It reads the file and dispatches.
    - A geometry file gives an `InputFile` holding only its atoms, with every other field at its default, as in the oracle.
    - A multi-frame xyz is the case `:xyz_frames`, through `IO.reject_or_warn`: strict throws `InputError(source, 0, "xyz: the file has 3 frames; an input geometry needs exactly one")`; lenient takes the first frame (the oracle's `io/molecule.rs` reader; the implementer cites the line) with one `:xyz_frames` warning.
    - `.yaml`/`.yml` goes to `parse_yaml` (Task 11) when the extension has added its method; otherwise it throws `_yaml_unavailable(path)` (Task 11) in both modes.
  - StrictMode: `parse_input` and `input_format` are exempted by name as entry points if they allocate.

- [ ] **Step 1: Write the failing tests.** `inp_fixture_tests.jl` checks the six fixture files field by field, one item per file, named `"fixture: <relative path>"`. The expected values are read from `parser.rs` (cite the lines in comments; D30); every field not listed is the default, checked with `fields_equal` against an `InputFile` built with only the listed fields. Each file is read with `parse_input(Fixtures.oracle_input_path(relpath))`, a Fixtures helper that returns the vendored path.
  1. examples/h2.inp (strict): Task 8's values.
  2. examples/h2o_631g_sp.inp (strict; `631-g` is accepted as an alias, D28 decided):
     - title `Water single-point PBE0/6-31G`; `basis == "631-g"`; method PBE0, Auto; `scf_convergence == 1e-10`; `max_iterations == 100`;
     - atoms O(0, 0, 0.1173), H(0, ±0.7572, −0.4692) Å × ANGSTROM_TO_BOHR.
  3. examples/o2_triplet.inp (lenient):
     - method HF, Auto; multiplicity 3; atoms O(0,0,0) and O(1.208,0,0) Å;
     - warnings `[:rhf_open_shell]`; strict error at line 5.
  4. examples/h2o_tddft.inp (strict): method LDA; `run_mode == TDDFT`; `tddft_states == 3` (the oracle CLI then overrides it to 10; spec 00 makes overrides apply only when given, step 17).
  5. tests/h2o_dimer_bsse.inp (strict): title `Water Dimer BSSE Test`; `run_mode == BSSE`; 6 atoms; `fragments == [[1,2,3],[4,5,6]]`; `nbo == true`.
  6. tests/o2_hubbard.in (lenient):
     - title `O2 Hubbard Test`; `basis == "6-31g"`; method PBE; multiplicity 3; 2 atoms; `hubbard == [HubbardU("O", 4.0)]`;
     - warnings: the list Task 9 derived for this file from the oracle source (`"lenient: o2_hubbard.in cases"`); strict error at line 10.
  - Dispatch items (`"dispatch: formats"`): `input_format` for each extension, case-insensitive; a vendored `.xyz` gives an `InputFile` with only atoms; the "needs `using YAML`" message, checked by calling `_yaml_unavailable` directly, not through whether YAML is loaded in the test process.
  - **`:xyz_frames`** (`"dispatch: xyz frames"`): `parse_input` on the vendored `examples/legacy_inputs/neb_test.neb.xyz` (5 frames, appendix E1) throws `"xyz: the file has 5 frames; an input geometry needs exactly one"` in strict mode; lenient gives the first frame's 3 atoms with exactly one `:xyz_frames` warning.
- [ ] **Step 2: Run** with `QUANTUMENGINE_ONLY=inp_fixture_tests`. Expected: FAIL.
- [ ] **Step 3: Implement** `dispatch.jl`.
- [ ] **Step 4: Run** the full suite. Expected: PASS. Add `parse_input` to the workload and TrimCheck (`.inp` path), and the coverage row.
- [ ] **Step 5: Commit** `Dispatch input files by extension`.

### Task 12b: The acceptance set of oracle inputs

**Files:**
- Create: `QuantumEngine/test/oracle/inputs/EXPECTED.toml`, `test/oracle/inputs_tests.jl`.

**Interfaces:**
- Produces, in `EXPECTED.toml`: one table per vendored file, keyed by its relative path. Its keys:
  - `strict = "ok" | "error"`;
  - `strict_message` (a substring, when an error);
  - `lenient = "ok" | "error"`;
  - `lenient_message`;
  - `lenient_warnings`, a sorted list of case names.

- [ ] **Step 1: Write the failing tests.**
  - **Coverage of the vendored set.** "every vendored input has an expected outcome" enumerates the files: a file without a row fails, and so does a row without a file. For each file it then asserts the strict and lenient outcomes. No row is `"pending"`.
  - **Outcomes known now** (the rest are seeded in Step 3):
    - `use_diis` (`:unknown_keyword`) in examples/{h2_ediis, h2o, h2o_test, h2o_cdiis, h2o_ediis, h2o_nodiss}.inp and legacy_inputs/water_ediis.inp: strict error, lenient ok.
    - examples/ch3_radical.inp and o2_triplet.inp: `:rhf_open_shell`.
    - legacy_inputs/h2o_opt.inp: `convergence` and `maxiter` are `:unknown_keyword`.
    - legacy_inputs/neb_test.inp: strict error at `END_GEOMETRY`. Lenient ok with both geometries of 3 atoms, `neb_images == 5`, `run_mode == NEB` and `charge == 1`, with warnings `:dropped_block_line` (×2), `:unknown_keyword` (`MAX_ITER`) and `:block_keyword`.
    - tests/{argon_vdw, h2o, h2o_vdw, h2o_vdw_d2}.in: `:block_keyword`/`:unclosed_block`, by Task 9's unclosed-block rule.
    - tests/o2_hubbard.in and tests/stress_h2o_complex.in: the lists Task 9 derived from the oracle source (`"lenient: <file name> cases"`), copied with their cited lines.
    - molecules/{h2_ccsd, oh_uccsd}.inp: strict "looks like YAML", lenient error "no atoms".
    - examples/h2o_opt.inp and h2o_631g_sp.inp: `basis 631-g` parses (`basis == "631-g"`; D28, decided by the user: accepted as the oracle's 6-31G alias).
    - legacy_inputs/neb_test.neb.xyz: `:xyz_frames`.
    - The YAML files: their outcomes through Task 11's extension, with `using YAML` in the item.
- [ ] **Step 2: Run** with `QUANTUMENGINE_ONLY=inputs_tests`. Expected: FAIL.
- [ ] **Step 3: Fill `EXPECTED.toml`** for the files not listed above. Each row, `ok`/`ok` rows included, is seeded from the oracle: the oracle source (cited lines in a `# reason` comment) or a captured oracle run (appendix E2, or Task 13's captures), never from the implementation's own output. Then run both modes against the rows. A row is never edited to make a failing parser pass; a disagreement is reported.
- [ ] **Step 4: Run** the full suite. Expected: PASS.
- [ ] **Step 5: Commit** `Pin the outcome of every vendored oracle input`.
### Task 13: The parity harness

**Files:**
- Create:
  - `QuantumEngine/test/oracle/harness.jl` (`@testmodule OracleHarness`), `test/oracle/capture.jl`.
  - Case and reference data: `test/oracle/references.toml`, `test/oracle/tolerances.toml`.
  - Captured outputs: `test/oracle/captured/<case>.{log,stdout}` and `captured/water_pbe_631g.molden`.
  - Items: `test/oracle/harness_tests.jl`, `test/oracle/live_tests.jl`.
  - Docs: `docs/src/parity.md`.
- Modify: `test/format_tests.jl` and `test/logger_tests.jl` (re-pin to the captured files), `test/molden_tests.jl` (oracle file), `test/inp_fixture_tests.jl` (the run-derived cross-check, D30), `test/Project.toml` (SHA, which `capture.jl` and the harness use for the file hashes, with the stdlib compat form of Task 1), `docs/make.jl` (pages).

**Interfaces:**
- Produces, in `@testmodule OracleHarness`:
  - **Locating the oracle.**
    - `oracle_binary()::Union{Nothing,String}` reads `ENV["QUANTUM_ENGINE_ORACLE"]`.
    - `oracle_basis_dir()::String` reads `QUANTUM_ENGINE_ORACLE_BASIS`. It throws `"oracle: QUANTUM_ENGINE_ORACLE is set but QUANTUM_ENGINE_ORACLE_BASIS (the oracle's basis_sets directory) is not"`.
  - **`run_oracle(case::String)::NamedTuple{(:log, :stdout, :exitcode), Tuple{String,String,Int}}`.**
    - It writes the case input into a fresh `mktempdir()`, symlinks `basis_sets` there, and runs `quantum_engine run <name>.inp` with that directory as the working directory: the oracle resolves basis files relative to cwd.
    - It reads `<name>.log` and the captured stdout. The post-SCF modules print with `println!`, so their lines appear only in stdout.
  - **Extractors**, written against the formats in the oracle source. A missing line throws `"oracle log: no `Total Energy:` line"`.
    - `total_energy(log)::Float64` is the last `  Total Energy:` line (`engine/results.rs:18`). The `SCF converged … E =` line excludes nuclear repulsion and is never used.
    - `scf_iterations(log)::Vector{Float64}` reads the `SCF iter` lines. They exclude E_nuc for UHF, so they are for convergence-history checks only.
    - `opt_steps(log)::Vector{NamedTuple{(:step,:energy,:gnorm),Tuple{Int,Float64,Float64}}}` reads the `[OPT] Step` lines.
    - `mp2_correlation(stdout)::Float64` reads `Total MP2 Correlation: x Hartree` (`scf/mp2.rs:74`).
    - `ccsd_correlation(stdout)::Float64` reads the CCSD block, whose format the implementer locates in `scf/ccsd*.rs` and cites.
    - `frequencies(log)::Vector{Float64}` and `ext_report(path)::Dict{String,Float64}` are written against the formats in the oracle source. The implementer locates them by searching the vibrations and analysis modules and cites file:line in the docstring.
    - `nuclear_repulsion_from(log)::Float64` is the `Total Energy` value minus the `SCF converged` value, closed-shell cases only.
  - **Stored data.** `reference(case, key)::Float64` reads `references.toml`. `tolerance(case, key)::Float64` looks up the case's `tolerance` map for `key`, then that row of `tolerances.toml`; a quantity with no entry in the map, or an entry naming a row that does not exist, throws `"oracle: no tolerance row <row> for <case>.<key>"`. This is the DFT rule (no row, no comparison) applied to every class.
- Produces, as data files:
  - **`references.toml`:** `[meta] oracle_commit` (the full hash, from `QUANTUM_ENGINE_ORACLE_COMMIT`, which `capture.jl` requires). One `[case.<name>]` table per case, with:
    - `input`: the verbatim input text, as a TOML multiline literal; it parses with strict `parse_inp`;
    - `source`: the oracle file it is copied or derived from, and for a derived input the lines changed (e.g. `"derived from examples/o2_triplet.inp: basis 6-31g, method uhf"`);
    - `extract`: the extractor names; `exports`: for example `["molden"]`;
    - `tolerance`: a map from each extracted quantity to its `tolerances.toml` row, e.g. `tolerance = {total_energy = "hf.energy", mp2_correlation = "mp2.energy"}`;
    - the extracted values and the SHA-256 of each captured file, which `capture.jl` writes.
    Only machine-independent fields are stored.
  - **`[fixture."<relative path>"]` tables** in `references.toml` for the six fixture files of Task 12a (D30): the echoed settings lines the oracle prints (`Title:`, `Method:`, `Basis Set:`, `Multiplicity:`, `Charge:`, `Convergence:`, and the `Atoms:` count), extracted from their captured stdout. These inputs are the vendored files themselves (two of them parse only with `--lenient`), so they are not `[case]` tables and are not compared numerically.
  - **`tolerances.toml`:** one row per (class, quantity), with `value` and `basis`, the tolerance justification:
    - `hf.energy = 1e-9` (kind (a): spec 01, measured closed-shell noise ~1e-12);
    - `mp2.energy = 1e-9`;
    - `uhf.energy = 1e-7`;
    - `geometry.angstrom = 1e-4`;
    - `frequency.cm1 = 1.0`;
    - `nuclear_repulsion.energy`: half a unit in the last digit the oracle prints (kind (a)); the implementer reads the printed digit count from the line `nuclear_repulsion_from` parses, cites the format's file:line, and writes the value.
    - The DFT rows are added by step 05 with its measured defect. Until then a DFT comparison throws the missing-row error.

**Who provides oracle outputs (D17).** The user (or the owner of a built oracle binary) runs:
```
QUANTUM_ENGINE_ORACLE=<binary> QUANTUM_ENGINE_ORACLE_BASIS=<basis_sets dir> \
QUANTUM_ENGINE_ORACLE_COMMIT=<full hash> \
julia --project=QuantumEngine/test QuantumEngine/test/oracle/capture.jl [--fixtures] <case>...
```
The arguments are case names. `--fixtures` also runs the six fixture files of Task 12a, unchanged, and writes their `[fixture."<relative path>"]` tables. The script runs each case and writes `captured/<case>.log` and `.stdout` plus any exports. It removes the lines that name local paths ("Reading input:", "Logging to:"), which are rewritten to the case name, and the timing lines. It then updates `references.toml`. Agents never build the oracle. They run `capture.jl` only on a binary the user names, at the user's request. The captured files are committed and are the stored oracle outputs; CI parses them without a binary.

**The step-01 capture list**, each input copied from the named oracle file. The tolerance column is each case's `tolerance` map; a row marked "(no row)" does not exist in step 01, so a comparison of that quantity throws the missing-row error until the step that owns it adds the row.

| case | input | quantities | tolerance rows |
|---|---|---|---|
| `water_pbe_631g` | examples/h2o_631g_sp.inp with `method pbe` and `export_molden true` | total energy (00-overview: −76.299269743150), nuclear repulsion, the atom table, the Molden file | `total_energy = "dft_closed.energy"` (no row; step 05), `nuclear_repulsion = "nuclear_repulsion.energy"` |
| `water_rhf_631g` | the same with `method rhf` | total energy −75.983974472322 | `total_energy = "hf.energy"` |
| `water_mp2` | examples/legacy_inputs/test_mp2.inp | `mp2_correlation`; the total energy is the SCF energy (trap) | `mp2_correlation = "mp2.energy"`, `total_energy = "hf.energy"` |
| `water_ccsd` | examples/legacy_inputs/h2o_ccsd_t.inp | `ccsd_correlation`; the total energy is the SCF energy (trap) | `ccsd_correlation = "ccsd.energy"` (no row; step 09), `total_energy = "hf.energy"` |
| `h2_opt` | examples/h2_opt.inp | `[OPT] Step` lines | `opt_energy = "hf.energy"`, `opt_gnorm = "gradient.norm"` (no row; step 06) |
| `water_optfreq` | examples/legacy_inputs/h2o_optfreq.inp (PBE0) | `opt_steps`, `frequencies` | `opt_energy = "dft_closed.energy"` (no row; step 05), `frequencies = "frequency.cm1"` |
| `o2_uhf_631g` | derived from examples/o2_triplet.inp: `basis 6-31g` and `method uhf` (the oracle maps `uhf` to `Method::RHF` and selects UHF by multiplicity, `parser.rs:1127`, so its result is the same as with `method rhf`) | total energy −149.545553662520; iteration lines without E_nuc | `total_energy = "uhf.energy"` |
| `h2_qtaim` | examples/h2.inp with `run qtaim` | `ext_report` | each key `"qtaim.<key>"` (no row; step 13) |

The six fixture files of Task 12a are captured too, unchanged, for their echoed settings (D30), with `capture.jl --fixtures`.

**How later steps plug in.**
1. Add a `[case.<name>]` table to `references.toml`, with its `tolerance` map.
2. Ask the user to capture it.
3. Add an extractor to `OracleHarness` if the quantity is new, with an item that parses the captured file.
4. Write the step's comparison item, `@testitem "<step>: <case> matches the oracle" setup = [OracleHarness]`. It computes the Julia value and asserts `abs(julia - reference(case, key)) ≤ tolerance(case, key)`.
5. Add the `tolerances.toml` rows the step's spec fixes, with their basis.

The live item re-runs the binary and checks that it still reproduces `references.toml`.

- [ ] **Step 1: Write the failing tests.**
  - **Self-test** (`harness_tests.jl`, `"harness: parses the stored water PBE log"`): `total_energy(read("captured/water_pbe_631g.log", String)) == reference("water_pbe_631g", "total_energy")`. This is `==` because the printed decimal is parsed and `capture.jl` stored the same parse; it tests the parser, not the oracle.
  - **00-overview energies** (`harness_tests.jl`, `"harness: captured energies equal the 00-overview values"`): `total_energy` of the captured `water_pbe_631g` log `== -76.299269743150`, of `water_rhf_631g` `== -75.983974472322` and of `o2_uhf_631g` `== -149.545553662520`. Each literal carries a comment citing its source, the "Reference energies measured from the oracle" paragraph of `00-overview.md` (lines 189–195 when this plan was written). `==`: 00-overview quotes the oracle's printed 12-decimal value, and both sides parse the same decimal. This is spec 01's "recovers −76.299269743150".
  - **Case inputs parse strictly** (`"harness: case inputs parse strictly"`): every `[case]` input in `references.toml` parses with strict `parse_inp`, with no warning.
  - **The other reference energies.** `total_energy` of the `water_rhf_631g` and `o2_uhf_631g` logs `==` their `references.toml` values.
  - **Fixture cross-check** (`inp_fixture_tests.jl`, `"fixture: echoed settings match the oracle"`): for each `[fixture]` table, the Julia parse of that file (strict, or lenient where Task 12a says so) gives the same title, multiplicity, charge, basis, convergence and atom count, and a method that maps to the echoed `Method:` name (the oracle prints its own enum name, e.g. `RHF` for `hf`; the item holds that map as a literal).
  - **Extractors.** `opt_steps` on `h2_opt` returns its step count and last energy as printed; `mp2_correlation` parses; `frequencies` returns 3 values ≥ 0 for water (3N − 6 = 3); `ext_report` parses.
  - **Nuclear repulsion cross-check.** `|nuclear_repulsion(Fixtures.water()) − nuclear_repulsion_from(water_pbe log)| ≤ 2e-12`. Kind (a): two values printed at 12 decimals, each ±5e-13, plus a Float64 sum of 3 terms (~6e-15).
  - **Re-pins.** The Task 5 atom-table line is re-pinned to `captured/water_pbe_631g.log` (the same geometry), and the Task 6 Molden reader item reads `captured/water_pbe_631g.molden`.
  - **Missing tolerance.** `tolerance("water_pbe_631g", "total_energy")` throws the missing-row message; so does a quantity with no entry in the map.
  - **Live** (`live_tests.jl`, tag `:oracle`): for each case, `run_oracle` gives `exitcode == 0`, and every compared quantity is looked up with `tolerance(case, key)` and asserted within it of `references.toml`. `tolerance` throws on a missing map entry or row, so no quantity is skipped silently: until its owning step adds the row, a "(no row)" quantity is an error of the live item, in its own `@testset` so that the other quantities are still compared. With `QUANTUM_ENGINE_ORACLE` set and the basis variable unset, `oracle_basis_dir()` throws its message.
- [ ] **Step 2: Run** with `QUANTUMENGINE_ONLY=harness_tests`. Expected: FAIL (no captured files).
- [ ] **Step 3: Implement** `harness.jl`, `capture.jl` and the TOML files with the eight cases' inputs and the six fixture entries.
- [ ] **Step 4: USER STEP. Ask the user to run `capture.jl --fixtures`** with the eight case names (so the six fixture files run too), and to confirm the oracle commit. Nothing in this task proceeds until the files exist; the plan does not create, edit or synthesize any captured file.
- [ ] **Step 5: Check the captures.**
  - Check that the captured files contain no local path (`grep` for `/Users/`, `/home/` and the host name; expect none).
  - Compare the captured values with 00-overview: `water_pbe_631g` −76.299269743150, `water_rhf_631g` −75.983974472322, `o2_uhf_631g` −149.545553662520. Report any difference to the user, with both values; it is a fact about the oracle run, not a parser failure, and no test or tolerance is changed for it (the item `"harness: captured energies equal the 00-overview values"` then fails until the user rules).
  - Run the harness items, then the full suite. Expected: PASS.
- [ ] **Step 6: Document** `docs/src/parity.md`: the capture protocol, the plug-in steps, the traps (MP2/CCSD total = SCF energy; the UHF iteration lines and the `SCF converged` line exclude E_nuc; the CLI overrides `verbosity` and `tddft_states`) and the tolerance table.
- [ ] **Step 7: Commit** `Add the oracle parity harness and captured references`.

### Task 14: The checkpoint-format comparison (Q-CHK)

**Files:**
- Create: `docs/superpowers/specs/2026-11-25-qe-checkpoint-formats.md` (repository root; shared directory, D20).

No format is implemented until the user chooses one.


- [ ] **Step 0: Measure.** The research behind the figures below is not durable (see "Evidence"), so the figures are measured again before they are written, in throwaway environments outside the repository, and the document records each version and command:
  - JLD2 0.6.7: resolve it, list the Manifest's packages, and name each `_jll` and whether it is a stdlib; build a juliac `--trim=safe` program that writes and reads a small struct of arrays, and count the lines of the build log that contain "unresolved call" (the earlier count was 484 such lines, appendix E3).
  - Serialization (stdlib): the same round-trip program and count (appendix E3 has 82 matching lines by `grep -c`; an earlier note said 88, so the figure written is the new count, stated with the command that produced it).
  - Own flat binary: the same program with `write`/`read` of a documented header and arrays; expect a clean build and an exact round trip.
  - QCSchema-style JSON through JSON.jl: the same program with typed JSON writing and the D2 reading route.
- [ ] **Step 1: Write the comparison document** from the Step 0 measurements and the oracle source. It states:
  - **What the oracle stores:** `ScfSnapshot {energy, density (total P), density_beta, mo_coeffs (alpha), mo_energies (alpha), n_iter, is_unrestricted}` in bincode.
    - It is written only on convergence (`scf_cycle.rs:626-640`, `uhf.rs:345-358`, `aim.rs:800-809`).
    - A restart reads only the densities (`engine/config.rs:320-342`) and tells total P from P_α by a trace heuristic (`uhf.rs:33-46`); a failed load does not stop the run: the SCF path warns through `writer.warn` and continues with the default guess (`engine/config.rs:341`), and the AIM path drops the load silently and runs the SCF again (`aim.rs:774`).
    - There is no geometry, basis, charge, multiplicity, occupations or magic, so the AIM path needs the matching `.inp`.
    - Rust `.chk` compatibility is out of scope (00, 13).
  - **The comparison table**, with each option's line estimate (marked as an estimate), closure, trim result, Julia-version stability, the programs that read it, and its fidelity to the oracle checkpoint. The closure and trim cells are the Step 0 results; the expectations below come from the earlier research (appendix E3) and are replaced by the measurement:
    - own flat binary: trim clean, round trip exact;
    - JLD2 0.6.7: `--trim=safe` fails (484 unresolved-call lines earlier); brings Pkg/LibCURL through FileIO; stdlib jlls only;
    - Serialization: fails trim; the format can change in 1.x minors;
    - Molden: no energy, density or n_iter;
    - Gaussian `.fchk`: 5E16.8, i.e. 8 significant digits;
    - QCSchema JSON: typed JSON.jl verified clean under trim; the AO order needs a permutation;
    - TREXIO, text back end: no Julia package; verification needs an external C build.
  - **The recommendation, stated as the researcher's and not as a decision:** an own documented flat binary `QECHK001` for restart, in the style of `PATRAJ01`, plus Molden export (Task 6) for interoperability.
  - **The open questions for the user:** the format; the contents (the oracle's fields only, or also P_α/P_β separately, beta C/ε/occupations, geometry, charge, multiplicity and a basis record); which interop exports beyond Molden, and whether readers are needed; Molden precision against the harness (D18); whether a failed load throws (Q-LENIENT), and whether `--lenient` restores the oracle's warning.
  - **The AO-order finding:** the oracle's Cartesian order equals the Molden document's for d, f and g (the correction to spec 02 is in D18). Molden has no h shells.
- [ ] **Step 2: Review.** Check every figure in the document against the Step 0 records (versions, commands, outputs) or the cited oracle line. No figure is added that Step 0 did not measure or the oracle source does not show.
- [ ] **Step 3: Commit** `Compare checkpoint formats for the user's choice`.
- [ ] **Step 4: STOP: ask the user** to choose the format and the contents (the open questions above). Record the answer in this plan. No checkpoint code is written in any task until then.

### Task 15: Step completion gate

**Files:**
- Modify: `QuantumEngine/test/coverage/thresholds.toml` (final rows), `QuantumEngine/docs/src/index.md`, `api.md` (every exported and public name in an `@docs` block).

- [ ] **Step 0: Preconditions.** Every row under "Decisions needed" except D2 has the user's answer; D2's route is the one Task 10 Step 0 measured and recorded. The Task 14 Step 4 answer is recorded.
- [ ] **Step 1: Run the full suite** under `--check-bounds=yes`. Expected: every item passes, with zero skipped outside the tagged groups of the `runtests.jl` filter (`:juliac`, `:oracle`, `:oracle_src`, `:slow`, `:blocked_on_purexc` and `:gpu`).
- [ ] **Step 2: Run the juliac item** with `QUANTUMENGINE_JULIAC_BUILD=1`. Expected: PASS.
- [ ] **Step 3: Check coverage.** Run `Pkg.test(coverage = true)`, then `check.jl`. Expected: PASS for every `src` file, each row at its measured floor.
- [ ] **Step 4: Run the mutations** with `julia QuantumEngine/test/mutation/run.jl`. Expected: the control passes and every entry is killed by its `expect` item.
- [ ] **Step 5: Build the docs locally** with `julia --project=QuantumEngine/docs QuantumEngine/docs/make.jl`, without deploying. Expected: no missing-docstring warning for an exported name.
- [ ] **Step 6: Check formatting.** Run `runic --check QuantumEngine`. Expected: clean.
- [ ] **Step 7: Commit** `Complete QuantumEngine step 01` (the docs and coverage rows only, if any changed).
- [ ] **Step 8: Review.** Request one review of the `qe-foundations` branch against spec 01, PLAN.md's "Gates between steps" and this plan (PLAN.md: a step is done when "one review has approved it"). Step 01 is not complete until that review approves; findings are fixed on the branch, the gates of Steps 1–6 are rerun, and the review is requested again for the changes.
- [ ] **Step 9: Draft the notice and the report.**
  - Draft (do not send) the one-line notice to the PureAtomistic session. It lists the shared files changed (the three workflows, AGENTS.md, README.md, the root `.gitignore`, `docs/superpowers/`) and the stale AGENTS.md lines 24 and 68 (D24).
  - Report to the coordinator: the review outcome (approved, with the reviewer and the commit reviewed); the TrimCheck outcome of `json_node` (D4) and the JSON route chosen (D2); the YAML extension (Task 11) results; the capture SHA-256s and any difference between the captures and the 00-overview values (Task 13 Step 5).

---

## Appendix: Evidence

The figures the tasks rely on, with their source. "Read" means read in the oracle checkout at 4a78a1d or in a package source; "run" means a command run while this plan was written; "earlier research" means a figure from the planning research whose raw output is not kept, and which the named task measures again before relying on it.

**E1. The oracle input files** (read).
- `find examples tests molecules benchmarks src/input \( -name '*.inp' -o -name '*.in' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' -o -name '*.xyz' \) | wc -l` → 72.
- `grep -rlE '/Users/|/home/'` over those files → no match.
- `examples/water.xyz` declares 3 atoms; `examples/legacy_inputs/adenina.xyz` 32; `examples/legacy_inputs/neb_test.neb.xyz` has 25 lines, 5 frames of 3 atoms.
- `examples/h2.inp` line 1 is `# Hydrogen molecule`. `examples/h2o_opt.inp:12` and `examples/h2o_631g_sp.inp:11` are `basis 631-g`. `examples/o2_triplet.inp` has `multiplicity 3`, `basis sto-3g`, `method rhf`. `tests/stress_dimer_bsse.in:15-17` is a `fragments` block with the line `0,1,2 ; 3,4,5`.

**E2. Oracle probes** (run: the oracle's existing release binary, not built for this plan, on small inputs; outcomes from the probes' stdout).

| probe input | oracle outcome |
|---|---|
| `.inp` with `dispersion False`, `foo_bar 3`, `use_diis true`, `method rhf`, bohr H2 | runs; `Dispersion: D3 (BJ)` (so `False` gave true); unknown keywords ignored |
| `.inp` with `hubbard O:5.0` followed by `use_diis true` | `Error: Input error: Invalid U value: true` |
| YAML content in a `.inp` file (`molecules/h2_ccsd.inp`) | title from the `#` line; `Error: Input error: No atoms found in input geometry…` |
| `.inp` content in a `.yaml` file | `Error: Input error: No atoms found in input geometry…` |
| YAML `scf_convergence: 1`, `method: dft`, `mode: bogus`, `max_iterations: 7` | runs; `Convergence: 1.0e-6` (the integer is ignored), `Method: RHF` (unknown method ignored), unknown mode ignored |
| xyz with an empty comment line 2 and 3 atoms | `Atoms: 2` (the first atom line is taken as the comment) |
| xyz with `Xx 0 0 0` | parsed; then `Error: Basis set error: Basis 'sto-3g' not found for element Xx` |
| `.inp` with a cell block | runs the plane-wave path (`PW SCF: Basis size 7…`) |

**E3. Checkpoint trim builds** (earlier research; Task 14 Step 0 measures again). Build logs of juliac `--trim=safe` round-trip programs: JLD2 484 lines containing "unresolved call"; Serialization 82 such lines by `grep -c` (an earlier note said 88); JSON.jl typed writing 8; the own flat binary none.

**E4. JSON.jl 1.6.1** (read: the package source). `src/JSON.jl:14-23` declares `parse`, `parse!`, `parsefile`, `parsefile!`, `lazy`, `lazyfile`, `LazyValue`, `isvalidjson`, `json`, `print`, `lower`, `lift`, `omit_null`, `omit_empty`, `Object`, `Null`, `Omit` and `JSONStyle` public; `applyobject` and `applyarray` are not. `parse.jl:221` takes a `dicttype` keyword. The difference in `applyobject`'s key conversion between 1.6.1 (`convert(String, k)`) and 1.13.4 (`String(k)`) is from earlier research; Task 10 Step 0 measures again.

**E5. YAML** (read: the General registry and the oracle's crate source). `Y/YAML/Deps.toml`: versions 0.4 onward depend on Base64, Dates and Printf; StringEncodings is added from 0.4.7. yaml-rust2 0.9.0 `src/yaml.rs:223-228` returns `"…: duplicated key in mapping"` for a repeated key. The YAML.jl 0.4.6 TrimCheck count (6 findings) and its scalar bugs (folding, `0o17`) are from earlier research; no task relies on them, since YAML.jl is loaded only by `QuantumEngineYAMLExt` and the trimmed entry reads no YAML (D1). Task 11 Step 2 measures each scalar typing difference between YAML.jl's Dict and JSON.jl's (`017`, `True`, `nan` and any other found).

**E6. StrictMode behavior** (run in a scratch module, StrictMode 0.5.21, StrictModeTest 0.5.19, Julia 1.13.4).
- A module that owns no functions: `test_compiled` checks 0 signatures and warns "green because it checked nothing"; `audit(...; sweep = true)` returns 0 findings; `test_calls` returns 0.
- `Errors` with an exempted, single-argument-call `Base.showerror(::Base.IO, ::InputError)` (with a `String` message): `test_compiled` judges 1 signature for each of `:typestable`, `:noalloc` and `:noboxing` without throwing; the noboxing scan passes; `test_calls(Errors)` returns 1; `test_exemptions(Errors)` returns 0 findings.
- `@enum V a b` adds to its module methods of `Base.Enums.namemap`, `instances`, `typemin`, `typemax` and `Base.Enums._enum_hash`.
