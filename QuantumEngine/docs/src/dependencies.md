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
