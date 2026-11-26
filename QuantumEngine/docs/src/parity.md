# Parity with the oracle

QuantumEngine reproduces the Rust `quantum_engine`'s reading of input files. The cases below are
known to read differently. Each is listed with the input that shows it and what each program
gives.

## Input words

| input | QuantumEngine | oracle | cause |
|---|---|---|---|
| a word holding the Kelvin sign U+212A, such as `bacK` for `back` | not recognized (`nothing`) | recognized: Rust's `to_lowercase` maps U+212A to `k` | the word lookups lowercase ASCII bytes only |
| the exact-exchange fraction of `libxc_hgga` or `libxc_hmgga` beyond the `Float64` range, such as `1e400` | not a number (`method_word` returns `nothing`) | `inf` (`-1e400` gives `-inf`) | the fraction is read with Julia's `tryparse(Float64, ·)`, which rejects an out-of-range decimal; Rust's `f64` parser rounds it to an infinity |
| the same fraction in a C `strtod` spelling that Rust rejects, such as `nan(1)` | read | rejected | Julia's `tryparse(Float64, ·)` follows `strtod` |
