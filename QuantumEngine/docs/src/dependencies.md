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
| JSON | `1.6` | at JSON 1.6.1: JSON 1.6.1, Parsers 2.8.11, StructUtils 2.12.1, PrecompileTools 1.4.0, Preferences 1.6.1; at JSON 1.13.4, the newest the bound admits: JSON 1.13.4, Parsers 3.0.6, StructUtils 2.12.1, PrecompileTools 1.4.0, Preferences 1.6.1 | Dates, Logging, Printf, Random, SHA, TOML, UUIDs, Unicode | none |

Parsers 3.0.6 comes from the General registry. It reads the exact-exchange fraction of a Libxc
hybrid method. Its exact decimal path, the one that places a value the 64-bit paths cannot,
allocates a digit buffer, so `method_word` and `_fraction` are exempt from `:noalloc`. The
exemption is removed when a Parsers release contains JuliaData/Parsers.jl#244.

JSON (JSON.jl) reads JSON input. Its closure was resolved twice in fresh environments, pinned to
1.6.1 and at 1.13.4; neither has a `_jll` package outside the standard libraries or an
`Artifacts.toml`. JSON 1.6 and 1.7 require Parsers 1 or 2, and QuantumEngine requires Parsers 3,
so the compat bound `"1.6"` resolves to JSON 1.8 or later (1.13.4 here). `json_node` reads
through the public `JSON.parse(text; dicttype = T)`, with an object type `T` whose `setindex!`
rejects a repeated key; JSON 1.6.1 and 1.13.4 both build objects through `T()` and `setindex!`
alone. TrimCheck rejects `json_node` on both versions, so the trimmed program reads `.inp` input
only. `JSON.parse` returns its values as `Any`, so `json_node`, `parse_json` and the helpers that
read JSON.jl's values are exempt from the StrictMode guarantees; `parse_structured`, which reads
only `Node`s, is held to all of them.

## Weak dependencies

| weak dependency | extension | compat | resolved closure (outside the standard libraries) | `_jll` / artifacts |
|---|---|---|---|---|
| YAML | `QuantumEngineYAMLExt` | `0.4.18` | YAML 0.4.18, StringEncodings 0.3.7, LazyJLLWrappers 1.2.7, Preferences 1.6.1, Libiconv_jll 1.18.1 | Libiconv_jll 1.18.1, with an `Artifacts.toml` |

YAML (YAML.jl) reads YAML input files. It is a `[weakdeps]` entry, not a dependency: the
extension `QuantumEngineYAMLExt` loads only when a program loads YAML.jl itself, and the package
reads structured input from any `AbstractDict` without it. Its closure, resolved in a fresh
environment, brings the binary library Libiconv_jll (through StringEncodings), so it stays outside
the core closure. The item "core closure is binary-free" resolves a fresh environment holding only
QuantumEngine, which installs no weak dependency, so it checks the core without the extension. A
trimmed program does not load the extension.
