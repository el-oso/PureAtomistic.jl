# Parity with the oracle

QuantumEngine reproduces the Rust `quantum_engine`'s reading of input files. The cases below are
known to read differently. Each is listed with the input that shows it and what each program
gives.

## Input words

| input | QuantumEngine | oracle | cause |
|---|---|---|---|
| a word holding the Kelvin sign U+212A, such as `bacK` for `back` | not recognized (`nothing`) | recognized: Rust's `to_lowercase` maps U+212A to `k` | the word lookups lowercase ASCII bytes only |
| the exact-exchange fraction of `libxc_hgga` or `libxc_hmgga` beyond the `Float64` range, such as `1e400` | not a number (`method_word` returns `nothing`) | `inf` (`-1e400` gives `-inf`) | the fraction is read with Parsers.jl's `tryparse(Float64, ·)`, which rejects an out-of-range decimal as Julia's `tryparse` does; Rust's `f64` parser rounds it to an infinity |

## Defaults

| input | QuantumEngine | oracle | cause |
|---|---|---|---|
| an SMD solvation block without a probe radius | `probe_radius` = 0.40 Å = `0.7558904498503081` bohr | `0.7558903954315693` bohr, relative difference 7.2e-8 | QuantumEngine converts with `Units.ANGSTROM_TO_BOHR` (CODATA 2018, 0.529177210903 Å per bohr); the oracle's SMD module divides by its own constant, 0.529177249 Å per bohr (`smd.rs`), which it also uses for a YAML `probe_radius` and for the solvent-accessible surface's 1.4 Å probe |

## The parity harness

The test suite compares QuantumEngine with outputs of the oracle stored in the repository, so
continuous integration needs no oracle binary. The harness lives in `test/oracle/`:

- `harness.jl`, the test module `OracleHarness`: it locates and runs the oracle, reads its
  outputs, and reads the stored references and tolerances;
- `references.toml`: one `[case.<name>]` table per case, with the input text, the oracle file it
  is copied or derived from, the oracle command that runs it (`run` or `aimext`), the extractors
  and exports, the map from each compared quantity to its tolerance row, the extracted values and
  the SHA-256 of each captured file; and one `[fixture."<path>"]` table per vendored fixture file,
  with the settings the oracle echoes for it;
- `tolerances.toml`: one row per class and quantity, with its value and its justification;
- `captured/`: the oracle's standard output of each case, its log for a `run` case and the files
  it exports, and in `captured/fixtures/` the standard output of each fixture file.

### Capturing the oracle's outputs

The owner of a built oracle binary runs, from the repository root,

```
QUANTUM_ENGINE_ORACLE=<binary> QUANTUM_ENGINE_ORACLE_BASIS=<basis_sets dir> \
QUANTUM_ENGINE_ORACLE_COMMIT=<full hash> \
julia --project=QuantumEngine/test QuantumEngine/test/oracle/capture.jl [--fixtures] <case>...
```

with the names of the cases to capture. For each case, `capture.jl` writes the case input as
`<case>.inp` into a fresh temporary directory, links the basis directory there as `basis_sets`
(the oracle resolves basis files relative to its working directory) and runs the oracle command
that the case's `command` names in that directory: `quantum_engine run <case>.inp` for every case
but `h2_qtaim`, which runs `quantum_engine aimext h2_qtaim.inp` (see below). It then writes
`captured/<case>.stdout`, `captured/<case>.log` for a `run`, and the exported files, rewriting the
`Reading input:` and `Logging to:` lines to name the case's files and dropping the lines that
report elapsed time; it stops if an output still names the temporary directory, the home
directory or the host name. Last it stores the extracted values, the files' SHA-256 and the commit
in `references.toml`. `--fixtures` also runs the six fixture files of the input tests unchanged
and stores their echoed settings. A capture at a commit other than the stored one must name every
case and `--fixtures`, so the file never mixes two oracle commits. The captured files are
committed.

The stored outputs are those of the oracle at commit 4a78a1d235688c5af10ccd94be75d7292073ff8d
(`meta.oracle_commit`), built from a clean checkout of that commit with
`cargo build --release --no-default-features`: a headless build, without the GUI and GPU
features (the GUI's dependency `objc2` 0.3.0-beta.3 does not compile with rustc 1.100.0), linked
against Libxc 7.1.2 (shared library version 15) from Homebrew.

The case `o2_uhf_631g` sets `scf_convergence 1e-8`, the convergence of the oracle's reference
value −149.545553662520; with the `1e-6` of its source file `examples/o2_triplet.inp` the oracle
stops at −149.545553539349, 1.2e-7 Ha away and beyond the `uhf.energy` tolerance.

The live items (tag `:oracle`) run only when `QUANTUM_ENGINE_ORACLE` names the binary. They re-run
every case and check that each compared quantity is still within its tolerance of
`references.toml`.

### Adding a case

1. Add a `[case.<name>]` table to `references.toml`, with its `input`, `source`, `extract`,
   `exports` and `tolerance` map.
2. Capture it with `capture.jl`.
3. If the quantity is new, add an extractor to `OracleHarness`, written against the format string
   in the oracle source and citing it, with a test item that parses the captured file.
4. Write the comparison item, `@testitem "<step>: <case> matches the oracle" setup =
   [OracleHarness]`, which computes the QuantumEngine value and asserts
   `abs(value - reference(case, key)) ≤ tolerance(case, key)`.
5. Add the `tolerances.toml` rows the step's specification fixes, with their justification.

`tolerance(case, key)` throws for a quantity that has no entry in the case's map, or whose entry
names a row that does not exist: a quantity without a row is never compared. It throws as well
for a quantity in the case's `defects` table (see below).

### Traps in the oracle's output

- For MP2 and CCSD runs the `Total Energy:` line is the SCF energy. The correlation energies are
  printed with `println!` by the post-SCF modules, so they appear only in the standard output, not
  in the log: `Total MP2 Correlation:` (`scf/mp2.rs:74`) and `FINAL CCSD CORRELATION:`
  (`scf/ccsd.rs:365`).
- The `SCF converged … E =` line excludes the nuclear repulsion (`scf/scf_cycle.rs:591`); the
  harness reads the total energy only from the `Total Energy:` line, and the nuclear repulsion of
  a closed-shell run as the difference of the two.
- The restricted `SCF iter` lines include the nuclear repulsion (`scf/scf_cycle.rs:500-503`); the
  unrestricted `UHF iter` lines do not (`scf/uhf.rs:260`). An unrestricted run prints no `SCF
  converged … E =` line.
- The oracle reads `uhf` as its one Hartree-Fock method and chooses UHF by the multiplicity
  (`parser.rs:1127`).
- The command line overrides the input's `verbosity` and `num_tddft_states` on every run, with its
  defaults 1 and 10 when the options are not given (`engine/config.rs:157-158`).
- `run qtaim` prints the QTAIM analysis to the log and writes no `.ext` report; the `.ext` report
  is written by the oracle's `aimext` command (`analysis/aim.rs:334-336`), so `h2_qtaim` is
  captured with `quantum_engine aimext h2_qtaim.inp`, run in the case's directory with
  `basis_sets` linked. Given an `.inp` file, `aimext` parses it, builds the basis, and loads
  `h2_qtaim.chk` from the input's directory if it exists; otherwise it runs the SCF itself and saves
  that checkpoint (`analysis/aim.rs:722-821`). It writes `h2_qtaim.ext` into the working directory
  and no log file (`main.rs:229-258`), so the case's captured files are `h2_qtaim.stdout` and
  `h2_qtaim.ext`. The input's `run qtaim` line has no effect on `aimext`.

### Known oracle defects

A captured quantity whose value is wrong is stored as captured and listed in its case's `defects`
table with what is wrong; `tolerance` throws for it, so no comparison requires QuantumEngine to
reproduce it.

| case | quantity | oracle | why it is wrong |
|---|---|---|---|
| `h2_qtaim` | `ext_report.n_nacp` | 4 | H2 has two nuclei, and the report's own Poincaré–Hopf line reads `N_NACP - N_BCP + N_RCP - N_CCP = 3 [INCOMPLETE / NON-CONVERGED]` (`analysis/aim.rs:229-237`) instead of 1. The correct count is decided with the QTAIM analysis. |

The other `h2_qtaim` quantities (the counts of bond, ring and cage critical points and the bond
critical point's ρ, Laplacian, `R_e`, `R_b`, `Delta_R` and ellipticity) are references.

### Tolerances

| row | value | justification |
|---|---|---|
| `hf.energy` | 1e-9 Ha | spec 01 parity tolerance for closed-shell HF; measured closed-shell oracle noise ~1e-12 |
| `mp2.energy` | 1e-9 Ha | spec 01 parity tolerance for MP2 |
| `uhf.energy` | 1e-7 Ha | spec 01 parity tolerance for UHF |
| `geometry.angstrom` | 1e-4 Å | spec 01 parity tolerance for geometries |
| `frequency.cm1` | 1 cm⁻¹ | spec 01 parity tolerance for frequencies |
| `nuclear_repulsion.energy` | 5e-13 Ha | half a unit in the last of the 12 decimals the `Total Energy:` and `SCF converged … E =` lines print |

The DFT rows (closed shell: the measured defect of the oracle's angular tables + 1e-8; open
shell: the measured defect + 2e-6) are added with that measurement. Until then a DFT comparison
throws for its missing row, as do the `ccsd.energy`, `gradient.norm` and `qtaim.*` rows that the
steps implementing those quantities add.
