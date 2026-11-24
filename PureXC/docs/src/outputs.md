# Outputs and layout

## Inputs

| case | LDA | GGA |
|---|---|---|
| restricted | `ρ` | `ρ, σ` with σ = ∇ρ·∇ρ |
| polarized | `(ρα, ρβ)` | `(ρα, ρβ), (σαα, σαβ, σββ)` with σαβ = ∇ρα·∇ρβ |

ρ and the σ components are numbers of any `Real` type; the result type is the promotion of the
functional's number type and the input types.

## Output names

`xc_derivatives(f, Val(k), ...)` returns a `NamedTuple` with the energy per volume `e` (e = ρ·ε,
not Libxc's energy per particle `zk`; [`zk`](@ref) converts) and one field for each split
a + b = j of each order j = 1, …, k, with a descending: a derivatives in the densities and b in
the σ components. The field is named `v[j]rho[a]sigma[b]`, with each exponent written only when it
is above 1 and each part left out when its count is 0:

| order | LDA | GGA |
|---|---|---|
| 0 | `e` | `e` |
| 1 | `vrho` | `vrho`, `vsigma` |
| 2 | `v2rho2` | `v2rho2`, `v2rhosigma`, `v2sigma2` |
| 3 | `v3rho3` | `v3rho3`, `v3rho2sigma`, `v3rhosigma2`, `v3sigma3` |
| j | `vjrhoj` | `vjrhoj`, `vjrho(j-1)sigma`, …, `vjsigmaj` |

These are Libxc's names, so code moving from Libxc needs no renaming; Libxc provides orders 0 to 2.

## Polarized layout

A restricted field is a number. A polarized field is a tuple over the density multisets of size a,
in lexicographic order over (α, β), each followed in turn by the σ multisets of size b in
lexicographic order over (αα, αβ, ββ). This is Libxc's order:

| field | components |
|---|---|
| `vrho` | α, β |
| `vsigma` | αα, αβ, ββ |
| `v2rho2` | αα, αβ, ββ |
| `v2rhosigma` | α·αα, α·αβ, α·ββ, β·αα, β·αβ, β·ββ |
| `v2sigma2` | αα·αα, αα·αβ, αα·ββ, αβ·αβ, αβ·ββ, ββ·ββ |
| `v3rho3` | ααα, ααβ, αββ, βββ |

A field of a densities and b σ components has binomial(a + 1, a)·binomial(b + 2, b) components.

## Array layout

The array layer takes the inputs of all points at once:

- restricted: vectors `ρ` (and `σ`) with the same axes;
- polarized: a `2 × n` matrix `ρ` and a `3 × n` matrix `σ`, the spin on dimension 1 in the order
  above and the points on dimension 2.

Every output field is a matrix with one row per component of the field (one row for a scalar
field) and the point axis of the inputs as its columns. [`xc_outputs`](@ref) allocates them with
`similar`, so the array type and the point axis follow `ρ`, including the offset axes of an
`OffsetArray`.
