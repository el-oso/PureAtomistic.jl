# QuantumEngine checkpoint formats: a comparison for the user's choice

Status: **no format is chosen.** This document compares the options so that the user can
choose the format and its contents. No checkpoint code is written until then. The
recommendation below is the researcher's, not a decision. Issue #6.

Every figure here comes from the measurements under "Measurements" (versions, commands and
outputs stated there), from a cited line of the Rust `quantum_engine` source at commit
`4a78a1d235688c5af10ccd94be75d7292073ff8d` (the oracle), or from a cited format document.
Anything not checked says so.

## What the oracle stores

The oracle's checkpoint is the Rust struct `ScfSnapshot` (`src/io/checkpoint.rs:9-17`),
written and read with bincode 1.3 (`Cargo.toml:17`, `src/io/checkpoint.rs:20-33`):

| field | type | content |
|---|---|---|
| `energy` | `f64` | the converged total energy |
| `density` | `DMatrix<f64>` | the total density P (P_α + P_β for UHF, `src/scf/uhf.rs:319`) |
| `density_beta` | `Option<DMatrix<f64>>` | P_β; `None` for RHF |
| `mo_coeffs` | `Option<DMatrix<f64>>` | the MO coefficients; the alpha set for UHF (`src/scf/uhf.rs:324`) |
| `mo_energies` | `Option<Vec<f64>>` | the MO energies; the alpha set for UHF (`src/scf/uhf.rs:325`) |
| `n_iter` | `usize` | the SCF iteration count |
| `is_unrestricted` | `bool` | true for UHF/UKS |

- It is written only on convergence: `src/scf/scf_cycle.rs:626-640` (restricted,
  `is_unrestricted: false`, `density_beta: None`), `src/scf/uhf.rs:345-358` (unrestricted),
  and `src/analysis/aim.rs:800-809` (the AIM path, which ignores a failed save).
- A restart reads only the densities (`src/engine/config.rs:320-342`): `density` becomes the
  initial density, and `density_beta` is used only when `is_unrestricted` is true. The energy,
  MOs, MO energies and `n_iter` are printed or ignored.
- The UHF start tells total P from P_α by a trace heuristic (`src/scf/uhf.rs:33-46`): it
  computes tr(P S) and takes P − P_β as P_α when that trace is closer to N_α + N_β than to N_α.
- A failed load does not stop the run. In the SCF path it prints
  `Failed to load checkpoint: …` through `writer.warn` and continues with the default guess
  (`src/engine/config.rs:341`). In the AIM path the load of `<stem>.chk` beside an input file is
  dropped silently and the SCF is run again (`src/analysis/aim.rs:774`).
- The file holds no geometry, basis, charge, multiplicity, occupations or magic number. The AIM
  path reading a `.chk` directly therefore needs the matching `<stem>.inp` beside it to rebuild
  the basis shells, and fails without it (`src/analysis/aim.rs:732-740`).
- Reading or writing the Rust `.chk` (bincode) is out of scope for the rewrite.

## The comparison

The line counts are **estimates** for a reader and writer of the oracle's fields in
QuantumEngine, not measurements; the round-trip programs of the measurements are much
smaller (8 to 52 lines each, `wc -l`). "Trim" is the result of a juliac `--trim=safe` build of
a program that writes and reads a small struct of arrays (see "Measurements").

| option | lines (estimate) | dependency closure | `--trim=safe` | stability across Julia versions | programs that read it | fidelity to the oracle checkpoint |
|---|---|---|---|---|---|---|
| Own flat binary `QECHK001` | ~200 | none | clean build, 0 "unresolved call" lines; the trimmed program's file equals the untrimmed one byte for byte and the round trip is exact | defined by its own documented layout | QuantumEngine only | all fields, bit-exact |
| JLD2 0.6.7 | ~50 | 12 registered packages and 34 stdlibs (11 of them `_jll`); Pkg (with Downloads, LibCURL, LibGit2) enters through FileIO; every `_jll` is a stdlib, and no package has an `Artifacts.toml` | fails: 482 "unresolved call" lines, no executable written | not checked | HDF5 tools (JLD2 README: "JLD2 files adhere to the HDF5 format specification"); not checked here | all fields, exact round trip untrimmed |
| Serialization (stdlib) | ~30 | the stdlib only | fails: 82 "unresolved call" lines (88 verifier errors: the other 6 are "unresolved finalizer"), no executable written | "The data format can change in minor (1.x) Julia releases" (its `serialize` docstring) | Julia only | all fields, exact round trip untrimmed |
| Molden | 0 new (the Molden writer planned for `src/io/molden.jl` exports it) | none | not built here | text format | molecular viewers; not checked here | none of energy, density or `n_iter` as fields: the format has no section for the density, and energies appear only per SCF iteration in the optional `[SCFCONV]` or per geometry in `[GEOCONV]` (Molden format document) |
| Gaussian `.fchk` | ~300 | none | not built here | text format | Gaussian's utilities; others not checked | real arrays in Fortran `5E16.8` (Gaussian's "Interfacing" document), i.e. 8 significant digits, not a Float64 round trip; stores "Total Energy" and the total density (spin density for open shell) |
| QCSchema-style JSON through JSON.jl 1.13.4 | ~250 | JSON, Parsers, PrecompileTools, Preferences, StructUtils and 8 stdlibs; no `_jll` | typed writing (`JSON.json` of a concrete struct): clean, file equal to the untrimmed one. Typed reading (`JSON.parse(text, T)` with a concrete `T`): clean, exact round trip. `JSON.parse(text; dicttype = T)`: 2 lines. `JSON.lazy` with `getproperty` and `JSON.parse(::LazyValue, T)`: 4 lines | text format | QCSchema consumers; not checked here | densities a/b, orbitals and eigenvalues exist in the wavefunction schema; the AO order needs a permutation (see "AO order") |
| TREXIO, text back end | not estimated | no Julia package in the General registry; the library is C (with Fortran and Python interfaces) | not built here | not checked | TREXIO users | not checked; verifying it needs an external C build of the TREXIO library |

## Recommendation (the researcher's, not a decision)

An own documented flat binary, magic `QECHK001`, for restart, in the style of PureAtomistic's
`PATRAJ01` trajectory (`PureAtomistic/src/io/trajectory.jl`): a magic number, a version, a
flags word and length fields in a fixed little-endian header, then the arrays column-major. It
is the only option measured here that is dependency-free, builds clean under `--trim=safe` and
round-trips exactly. For interoperability, the Molden export planned for QuantumEngine
carries the geometry, basis and MOs to other programs.

## Open questions for the user

1. **The format:** the flat binary, or another row of the table.
2. **The contents:** the oracle's fields only, or also any of P_α and P_β stored separately
   (which removes the trace heuristic), the beta MO coefficients, beta MO energies and
   occupations, the geometry, charge, multiplicity, and a basis record (which would let the
   AIM path run without the matching `.inp`).
3. **Interoperability exports beyond Molden** (for example `.fchk` or QCSchema JSON), and
   whether readers for them are needed or writers suffice.
4. **Molden precision against the parity harness:** the Molden writer is specified at full
   precision (shortest round trip, decision D18 of the step-01 plan in
   `docs/superpowers/plans/`); whether that is kept when the harness compares with the oracle's
   Molden files.
5. **A failed load (Q-LENIENT, decision D6 of the step-01 plan in `docs/superpowers/plans/`:
   unknown keywords and bad values are errors unless `--lenient` is given):** whether a failed
   load throws, and whether `--lenient` restores the oracle's behavior of a warning and the
   default guess.

## AO order

The oracle's Cartesian order (`src/basis/shell.rs:84-117`) equals the Molden format
document's for d, f and g:

- d: xx, yy, zz, xy, xz, yz
- f: xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
- g: xxxx, yyyy, zzzz, xxxy, xxxz, yyyx, yyyz, zzzx, zzzy, xxyy, xxzz, yyzz, xxyz, yyxz, zzxy

So a Molden file needs no permutation of the oracle's AOs up to l = 4 (the correction to the
rewrite spec's Molden sentence is decision D18 of the step-01 plan). The Molden document's shell
labels are s, p, d, f, sp and g: it has no h shells, while the oracle has l = 5
(`src/basis/shell.rs:106-112`).

QCSchema orders AOs "according to the CCA standard as implemented in libint" (the QCSchema
wavefunction documentation), and libint's CCA Cartesian order is lexicographic (d: xx, xy, xz,
yy, yz, zz), so a QCSchema export permutes the oracle's order for d and above.

## Measurements

Run on 2026-11-25 with Julia 1.13.4 (aarch64, macOS), in throwaway environments outside the
repository, the General registry at tree `4f87f0fa7cc692b3a78ae682af0bd98b3f3f8dbb`. The
juliac driver is JuliaC.jl 0.3.12 from `PureMLIP/juliac/toolenv`.

### Environments

- JLD2: `Pkg.add(name = "JLD2", version = "0.6.7")`. Manifest, registered packages:
  ChunkCodecCore 1.0.2, ChunkCodecLibZlib 1.1.0, ChunkCodecLibZstd 1.0.0, FileIO 1.20.0,
  HashArrayMappedTries 0.2.0, JLD2 0.6.7, MacroTools 0.5.16, OrderedCollections 2.0.3,
  PrecompileTools 1.4.0, Preferences 1.6.1, Requires 1.3.1, ScopedValues 1.6.2. Stdlibs:
  ArgTools, Artifacts, Base64, Dates, Downloads, FileWatching, JuliaSyntaxHighlighting,
  LibCURL, LibGit2, Libdl, Logging, Markdown, Mmap, NetworkOptions, Pkg, Printf, Random, SHA,
  StyledStrings, TOML, Tar, UUIDs, Unicode, and the `_jll` stdlibs
  CompilerSupportLibraries_jll, LibCURL_jll, LibGit2_jll, LibSSH2_jll, MozillaCACerts_jll,
  OpenSSL_jll, PCRE2_jll, Zlib_jll, Zstd_jll, nghttp2_jll, p7zip_jll. FileIO depends on Pkg,
  Requires and UUIDs; Pkg brings Downloads (and through it LibCURL) and LibGit2. A package was
  counted as a stdlib when its directory exists under `Sys.STDLIB`, and no package directory
  holds an `Artifacts.toml`.
- Serialization: `Pkg.add("Serialization")`; the Manifest holds Serialization 1.11.0 only.
- JSON: `Pkg.add("JSON"); Pkg.compat("JSON", "1.6")`, which resolves JSON 1.13.4 with
  Parsers 3.0.6, PrecompileTools 1.4.0, Preferences 1.6.1, StructUtils 2.12.1 and the stdlibs
  Dates, Logging, Printf, Random, SHA, TOML, UUIDs, Unicode. No `_jll`.
- Flat binary: an empty project.

### Programs

Each program defines the struct below and writes it to the file named by its argument. Every
program except JSON writing then reads it back and prints `roundtrip exact` (exit 0) when every
field is `===`/`==` equal; the JSON-writing program writes only and prints `written`.

```julia
struct Snap
    energy::Float64
    density::Matrix{Float64}      # 4×4, entries 1/(i+j-1) + 0.1 sin(ij)
    mo_energies::Vector{Float64}  # 4 entries
    n_iter::Int
    is_unrestricted::Bool
end
```

- JLD2: `jldopen(path, "w") do f; f["snap"] = s; end`, then `jldopen(path, "r") do f;
  f["snap"]::Snap; end`.
- Serialization: `serialize(io, s)`, then `deserialize(io)::Snap`.
- Flat binary: the magic `QECHK001`, `UInt32` version 1, `UInt32` flags (bit 0: unrestricted),
  `Int64` n_ao, n_mo and n_iter, `Float64` energy, then the density column-major and the MO
  energies, each value through `htol`/`ltoh`; the reader checks the magic, the version and
  the end of the file. The file is 208 bytes.
- JSON, writing: `JSON.json(io, s)` (the matrix is written as a list of columns).
- JSON, `dicttype` reading: `JSON.parse(text; dicttype = UniqueDict)`, where `UniqueDict <:
  AbstractDict{String, Any}` throws on a repeated key, then typed asserts per field.
- JSON, lazy reading: `l = JSON.lazy(text)`, then `JSON.parse(l.energy, Float64)`,
  `JSON.parse(l.density, Vector{Vector{Float64}})` and so on per field.
- JSON, typed reading: `JSON.parse(text, SnapJ)`, where `SnapJ` is `Snap` with the density as
  `Vector{Vector{Float64}}`.

Every program except JSON writing prints `roundtrip exact` when run untrimmed (`julia
--project=<env> entry.jl out.chk`); the JSON-writing program prints `written`, and its trimmed
output file is checked with `cmp` against the untrimmed run's.

### Build and count

```sh
julia --startup-file=no --project=PureMLIP/juliac/toolenv -e "using JuliaC; JuliaC.main(ARGS)" -- \
    --project=<env> --output-exe rt --bundle <env>/build --experimental --trim=safe --verbose \
    <env>/entry.jl > build.log 2>&1
grep -c "unresolved call" build.log
grep "Trim verify finished" build.log
```

| program | build exit | `grep -c "unresolved call"` | verifier summary | trimmed run |
|---|---|---|---|---|
| flat binary | 0 | 0 | none printed | `roundtrip exact`; file `cmp`-identical to the untrimmed run's |
| JLD2 0.6.7 | 1 | 482 | "Trim verify finished with 482 errors, 0 warnings." | no executable |
| Serialization | 1 | 82 | "Trim verify finished with 88 errors, 0 warnings." (6 are "unresolved finalizer") | no executable |
| JSON writing | 0 | 0 | none printed | file `cmp`-identical to the untrimmed run's |
| JSON `dicttype` reading | 1 | 2 | "Trim verify finished with 2 errors, 0 warnings." (`#parse#72` called with `dicttype::DataType`) | no executable |
| JSON lazy reading | 1 | 4 | "Trim verify finished with 4 errors, 0 warnings." (`applyobject`/`applyarray` through `getproperty(::LazyValue, ::Symbol)`) | no executable |
| JSON typed reading | 0 | 0 | none printed | `roundtrip exact` |

The `dicttype` and lazy reading programs are one way each to write those routes; another way
of writing them may build clean, which was not checked.

## Decision (the user's, 2026-11-25)

1. **Format:** "own flat binary, but make it hdf5-like in pure julia". The checkpoint is an own
   binary format written and read in pure Julia, structured like HDF5: a self-describing tree of
   named groups holding named, typed n-dimensional datasets and small attributes, with a fixed
   header (magic, version) and an index of the tree, so that a reader can list and read one
   dataset without knowing the layout in advance. It is not the HDF5 file format and needs no
   HDF5 library; it builds under `--trim=safe` and round-trips exactly.
2. **Contents:** the oracle's fields, plus P_α and P_β stored separately, the beta MO
   coefficients, energies and occupations, the geometry, charge and multiplicity, and a basis
   record.
3. **Exports:** Molden only.
4. **A failed load:** an error; `--lenient` restores the oracle's warning and default guess.
