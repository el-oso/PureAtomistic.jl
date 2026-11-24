# PureXC

PureXC is a library of exchange-correlation (XC) functionals in pure Julia, under the MIT license.
It evaluates the energy per volume of a functional and its partial derivatives with respect to the
density variables to any order, spin-restricted and spin-polarized, in any floating-point type.
The derivatives come from the truncated Taylor number type of
[PureTaylor](https://el-oso.github.io/PureAtomistic.jl/PureTaylor/). Libxc is used only as a test
oracle and a speed reference; it is not a dependency.

PureXC requires Julia 1.13 or later. Its dependencies are PureTaylor, StrictMode and
TypeContracts; its closure holds no `_jll` package and no artifact.

## Functionals

| constructor | functional | Libxc identifier | family | kind |
|---|---|---|---|---|
| [`lda_x`](@ref) | Slater exchange | `lda_x` | LDA | exchange |
| [`lda_c_pw`](@ref) | Perdew–Wang 1992 correlation, constants of the paper | `lda_c_pw` | LDA | correlation |
| [`lda_c_pw_mod`](@ref) | Perdew–Wang 1992 correlation, constants used by PBE | `lda_c_pw_mod` | LDA | correlation |
| [`pbe_x`](@ref) | Perdew–Burke–Ernzerhof exchange | `gga_x_pbe` | GGA | exchange |
| [`pbe_c`](@ref) | Perdew–Burke–Ernzerhof correlation | `gga_c_pbe` | GGA | correlation |
| [`pbe`](@ref) | `pbe_x` + `pbe_c` | `gga_x_pbe` + `gga_c_pbe` | GGA | exchange-correlation |

Each constructor takes the number type, `Float64` by default: `pbe_c(Float32)`, `lda_x(BigFloat)`.
[`XC`](@ref) builds any weighted sum of functionals of one family.

## Three layers

**1. Functional values.** A functional is an immutable struct of its parameters and thresholds,
parametric in their number type: [`PureXC.PBEExchange`](@ref)`{T}` holds κ, μ and its
[`PureXC.Thresholds`](@ref). A functional whose parameters are `Taylor` numbers carries them as
series variables, so [`energy_density`](@ref) returns the derivatives with respect to them:

```julia
using PureXC, PureTaylor
f0 = pbe_x()
κ = variable(Val(1), Val(1), Val(1), f0.κ)
f = PureXC.PBEExchange{typeof(κ)}(κ, f0.μ, f0.thresholds)
derivatives(energy_density(f, 1.0, 0.5), (1.0,))   # (e, ∂e/∂κ)
```

[`xc_derivatives`](@ref) refuses such a functional with an `ArgumentError`, since it would need the
parameter's and the inputs' variables in one series.

**2. Point layer.** [`xc_derivatives`](@ref) and [`energy_density`](@ref) take one point, as numbers
(restricted) or tuples (polarized), and return a `NamedTuple` of numbers and tuples. They allocate
nothing, so they can be called inside a caller's own grid loop.

```julia
using PureXC
f = pbe()
xc_derivatives(f, Val(2), 0.3, 0.1)                            # restricted ρ, σ
xc_derivatives(f, Val(2), (0.2, 0.1), (0.05, 0.02, 0.03))      # polarized (ρα, ρβ), (σαα, σαβ, σββ)
energy_density(lda_x(), 0.3)                                   # e alone
zk(energy_density(lda_x(), 0.3), 0.3)                          # e/ρ, Libxc's zk
```

**3. Array layer.** [`xc_evaluate!`](@ref) evaluates every point of arrays of inputs into the arrays
of a `NamedTuple`, which [`xc_outputs`](@ref) allocates:

```julia
ρ = rand(2, 1000) .+ 0.1          # polarized: spin on dimension 1, points on dimension 2
out = xc_outputs(lda_c_pw(), Val(1), ρ)
xc_evaluate!(out, lda_c_pw(), Val(1), ρ)
size(out.vrho)                    # (2, 1000)
```

It checks every input before it writes the first output, and it allocates nothing. See
[Outputs and layout](outputs.md) for the array shapes and [Evaluation](evaluation.md) for the
rules on views and shared memory.

## Errors

Caller errors throw: a negative or non-finite density or σ, a polarized σαβ beyond
√σαα·√σββ·(1 + 8·eps), an input beyond the domain limits of a functional (see
[Thresholds and domain](thresholds.md)) and a `Taylor` parameter passed to `xc_derivatives` throw
an `ArgumentError` whose message starts with `"xc: "`; mismatched array axes throw a
`DimensionMismatch`. Libxc returns values for some of these inputs (0 for ρ = −1, a finite value
for σ = −1); PureXC does not clamp them.
