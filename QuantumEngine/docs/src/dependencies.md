# Dependencies

Every dependency of QuantumEngine is resolved in a fresh environment before it is added, and the
manifest is checked for `_jll` packages outside the standard libraries and for packages with an
`Artifacts.toml`. The test item "core closure is binary-free" repeats the check for the whole
package.

| dependency | compat | resolved closure (outside the standard libraries) | standard libraries | `_jll` / artifacts |
|---|---|---|---|---|
| StrictMode | `0.5.21` | StrictMode 0.5.22, PrecompileTools 1.4.0, Preferences 1.6.1, TypeContracts 0.14.7 | Base64, Dates, InteractiveUtils, JuliaSyntaxHighlighting, Markdown, Printf, Serialization, StyledStrings, TOML, Unicode | none |
| StaticArrays | `1.9` | StaticArrays 1.9.25, StaticArraysCore 1.4.5, PrecompileTools 1.4.0, Preferences 1.6.1 | Artifacts, Dates, Libdl, LinearAlgebra, Printf, Random, SHA, TOML, Unicode; the standard-library `_jll` packages CompilerSupportLibraries_jll, OpenBLAS_jll and libblastrampoline_jll (through LinearAlgebra) | none outside the standard libraries |
| LinearAlgebra | `1.13` | none: a standard library, already in the closure through StaticArrays | Artifacts, Libdl; the standard-library `_jll` packages CompilerSupportLibraries_jll, OpenBLAS_jll and libblastrampoline_jll | none outside the standard libraries |
| Printf | `1.11` | none: a standard library (version 1.11.0 in Julia 1.13), already in the closure through StrictMode and StaticArrays | Unicode | none |
| Parsers | `3` | Parsers 3.0.6, PrecompileTools 1.4.0, Preferences 1.6.1 | Dates, Printf, TOML, Unicode | none |

Parsers 3.0.6 comes from the General registry. It reads the exact-exchange fraction of a Libxc
hybrid method. Its exact decimal path, the one that places a value the 64-bit paths cannot,
allocates a digit buffer, so `method_word` and `_fraction` are exempt from `:noalloc`. The
exemption is removed when a Parsers release contains JuliaData/Parsers.jl#244.
