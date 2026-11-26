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
