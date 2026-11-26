# Vendored oracle input files

These files are copied unchanged from the repository
https://github.com/boleklolek/quantum_engine at commit
4a78a1d235688c5af10ccd94be75d7292073ff8d: every `*.inp`, `*.in`, `*.yaml`,
`*.yml`, `*.json` and `*.xyz` file under `examples/` (including
`legacy_inputs/`), `tests/`, `molecules/`, `benchmarks/` and `src/input/`, at
the same relative paths.

The source repository declares the Apache-2.0 license in its `Cargo.toml`; the
license text is in `LICENSE` next to this file.

`EXPECTED.toml` is not one of them: it records the outcome of `parse_input` on
each file in strict and lenient mode, with the oracle source lines behind it.
