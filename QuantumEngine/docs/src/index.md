# QuantumEngine

QuantumEngine is a quantum-chemistry program in pure Julia, under the MIT license: a rewrite of the
Rust program `quantum_engine`, whose outputs serve as its test oracle. It requires Julia 1.13 or
later. Its dependencies are JSON, LinearAlgebra, Parsers, Printf, StaticArrays and StrictMode, and
their closure holds no `_jll` package and no artifact outside the standard libraries. YAML.jl is an
optional dependency, read through a package extension once it is loaded; its closure brings
Libiconv_jll and so stays outside the core (see [Dependencies](dependencies.md)).

## Errors

Every error QuantumEngine throws, besides `ArgumentError`, is a subtype of
[`QuantumEngineError`](@ref). Each prints its message after a prefix that names its kind:

| type | printed as |
|---|---|
| [`InputError`](@ref) | `input: <source>:<line>: <msg>` (`input: <source>: <msg>` without a line) |
| [`BasisError`](@ref) | `basis: <msg>` |
| [`SCFNotConverged`](@ref) | `scf: <msg>` |
| [`OptimizationFailed`](@ref) | `optimization: <msg>` |
| [`NotImplemented`](@ref) | `not implemented (<section>): <msg>` |
| [`FunctionalUnavailable`](@ref) | `functional: <msg>` |
| [`NumericalError`](@ref) | `numerical: <msg>` |

An [`InputWarning`](@ref) records an input that lenient mode accepted where strict mode throws; it
is shown as the `InputError` of the same source, line and message.
