# PureTaylor

`Taylor{N,K,M,T,L}` is a truncated Taylor polynomial in `N` variables up to total degree `K`, with
coefficients of type `T`. It is a `Real` whose coefficients live in an `NTuple{L,T}`, so a value
is an immutable object with no heap allocation. The package has no dependency besides
StrictMode, which checks its type stability and allocation freedom when the policy is enabled.

Arithmetic (`+`, `-`, `*`, `/`, `inv`, integer and real powers) and the elementary functions
`sqrt`, `cbrt`, `exp`, `expm1`, `log` and `log1p` act on the whole polynomial. `substitute`
changes variables. The `fast_*` functions are scalar kernels without calls or data-dependent
branches, so that loops over many points can be vectorized; they also act on `Taylor` numbers.

```julia
using PureTaylor

# f(x, y) = x²·y at (2, 3), with every partial derivative up to order 2.
x = variable(Val(2), Val(2), Val(1), 2.0)
y = variable(Val(2), Val(2), Val(2), 3.0)
f = x^2 * y
value(f)                       # 12.0
derivatives(f, (1.0, 1.0))     # (12.0, 12.0, 4.0, 6.0, 4.0, 0.0), in multi-index order
```

## Layout

The type holds the coefficient `c_α = (∂^α f)/α!` of every multi-index `α`, the normalized Taylor
coefficients, so that the product of two series is the plain convolution `c_γ = Σ_{α+β=γ} a_α·b_β`
with no binomial factors. `c[1]` is the value.

**Multi-index order.** The multi-indices of `N` variables up to total degree `K` are graded by
degree: degree 0 first, then degree 1, and so on. Within one degree they are in descending
lexicographic order of the exponent tuple. For `N = 2`, `K = 2` the order is
`(0,0)`, `(1,0)`, `(0,1)`, `(2,0)`, `(1,1)`, `(0,2)`. This order is the contract other code builds
on: coefficient planes are indexed by it, and `derivatives` returns its results in it.

**Masks.** The parameter `M` is a bit mask over the variables: bit `i-1` set means variable `i`
carries nonzero higher-order terms. Only the multi-indices supported on those variables are
stored, so `L == binomial(count_ones(M) + K, K)`. A constant has mask 0 and one coefficient. The
mask of a sum or product is the union of the masks of its operands, computed at compile time, so a
function of many variables that depends on a few of them at a time stores and multiplies only the
coefficients that can be nonzero. `coefficient(a, α)` reads one coefficient and gives zero for an
index that is not stored.

**Scaled seeds.** `variable(Val(N), Val(K), Val(i), x, s)` is `x + s·uᵢ`: the series variable is
`uᵢ`, and the derivative with respect to `uᵢ` is `s` times the derivative with respect to `x`. With
`s = x` every coefficient stays of order 1 in size, whatever the magnitude of `x`. `derivatives(a, invscale)`
undoes the scaling: `∂^α f = α! · c_α · Π invscale_i^α_i` with `invscale_i = 1/s_i`.
`substitute` takes the same convention: a variable seeded with scale `s` is the increment divided
by `s`, so pass `u[i] / s`.

## Fast functions

`fast_sqrt`, `fast_cbrt`, `fast_invcbrt`, `fast_log`, `fast_log1p`, `fast_exp` and `fast_expm1`
for `Float32` and `Float64` compile to code with no calls other than LLVM intrinsics, no throws
and no data-dependent branches. Their domains are documented and not checked; outside a domain the
result is unspecified. Any other `Real` falls back to the Base function. On a `Taylor` number the
value comes from the scalar kernel and the higher coefficients from the same formulas as the Base
methods.

| function | domain, `Float32` and `Float64` |
|:---------|:--------------------------------|
| `fast_sqrt` | `x ≥ 0` |
| `fast_cbrt`, `fast_invcbrt`, `fast_log` | positive normal `x` |
| `fast_log1p` | `x > -1` with `1 + x` normal |
| `fast_exp`, `fast_expm1` | `abs(x) ≤ 700` for `Float64`, `abs(x) ≤ 85` for `Float32` |

`fast_sqrt` is the correctly rounded hardware square root, equal to Base's `sqrt` on its domain.

The tests bound the other kernels at 4 ulps. Largest error measured over every `Float32` of each
domain (Julia 1.13, Apple M-series), in ulps of the correctly rounded result: `cbrt` 1, `invcbrt` 1,
`log` 1, `log1p` 1, `exp` 1, `expm1` 2. For `Float64`, over 10⁷ log-uniform samples per domain
plus the ends, the powers of two and, for `log`, the region near 1, against a 128-bit `BigFloat`
reference, in ulps: `cbrt` 1.3249, `invcbrt` 1.5603, `log` 0.8489, `log1p` 1.4369, `exp` 1.0404,
`expm1` 2.1782. The `Float64` figures are sampled, not exhaustive.
