# Evaluation

## Derivatives

Each functional defines one function, its energy per volume, on `Taylor` numbers. The driver seeds
one `Taylor` variable per input (ρ, or ρα and ρβ, and the σ components), evaluates the energy
once, and reads every partial derivative to order k off the resulting series. Each input x is
seeded with a scale s, as x + s·u, so that the coefficients of the series stay of moderate size
whatever the size of x. By default a density is seeded at max(|x|, n_thr, floatmin) and a σ
component at the larger of |x| and the σ at which the reduced gradient is 1; PW92 and the PBE
functionals define their own scales. The scale changes only the
representation of the series, not the derivatives.

PureTaylor accumulates each coefficient of a product as a chain of `fma` calls (`_fma`), with one
rounding per step for the IEEE floats and `BigFloat` and the plain `x*y + z` for other coefficient
types. `muladd` would let the compiler decide at each call site whether to fuse, so the same
kernel could round differently inlined and out of line, unrolled and looped; with `fma` every path
gives identical results. The Taylor operations on series of at most 21 coefficients (PureTaylor's
`_INLINE_MAX`, the largest series PureXC evaluates at orders 0 to 2) are forced inline, so that the
point loop of the array layer has no call and vectorizes; larger series are left to the compiler,
since forced inlining of them costs compile time that grows with the order.

## Evaluation type

A functional built for `Float32` and given `Float32` inputs is evaluated in `Float32` up to the
largest order measured to stay within its bound in pure `Float32` (`_float32_max_order`), and in
`Float64` above it, with each output rounded once to `Float32`:

| functional | evaluated in Float32 up to order |
|---|---|
| `lda_x` | 10 |
| `lda_c_pw`, `lda_c_pw_mod` | 2 |
| `pbe_x`, `pbe_c` | 3 |
| `XC`, including `pbe` | 1 |

Above these orders the mixed partials spread over many orders of magnitude (at (1, 1e-6), order 9,
the order-9 partials of `lda_c_pw` range from 19.8 to 1.4e48), so that intermediate coefficients
overflow `Float32` where the output fits. An output beyond `floatmax(Float32)` becomes the infinity
of its sign. The choice depends on the types alone, so it is made when the method is compiled.

An [`XC`](@ref) evaluates each term in its own evaluation type (every term in `Float64` when a
`Float32` XC is above order 1), sums the weighted terms in the widest of those types and rounds
each output once. A sum of terms rounded first could turn two entries
beyond `floatmax(Float32)` with opposite signs into `Inf - Inf = NaN` where their sum fits.
The order of an XC is measured on the XC, not taken from its terms: for `pbe(Float32)` at order 2,
each term evaluated in `Float32` gives an infinity of opposite sign in `v2sigma2` at densities from
1e-12 to about 1e-11 and small σ, where the exact sum is beyond `floatmax(Float32)` (at (1e-12, 0),
5.93e43), so a `Float32` sum would be `NaN` (`PureXC/test/scans/xc_float32_order.jl`).

## Array layer

[`xc_evaluate!`](@ref) runs in this order: it checks the axes and the output fields, checks that
no view lies outside its parent and that no output shares memory with an input or another output,
checks every input point, and only then evaluates the points. An invalid input therefore leaves
the outputs untouched.

The input check first runs a branch-free predicate over all points, which vectorizes, and walks
the points again with the throwing check only when one fails, so that the error names the first
invalid point.

**Dense path.** When every input and output is an `Array` of IEEE floats, the point loop reads
and writes through raw pointers under `@simd ivdep`. The overlap check is what makes this sound:
it has established that the iterations write disjoint memory. Every other array type (views,
`OffsetArray`, `reshape`, other wrappers) takes the generic loop over the point axis, with the
same results.

**Shared memory.** No output may share memory with an input or another output, and no output may
write one address from two points. `xc_evaluate!` resolves every argument through `parent` to its
root and throws `ArgumentError("xc: an output shares memory with an input or another output")`
before writing anything when two roots may share memory. The test is exact for `Array`s and views
of an `Array` indexed by integers and ranges, so disjoint rows or column blocks of one matrix are
accepted; any other array over the same storage as another argument is rejected, even when the
elements it reaches are disjoint. An array type with no known address range is compared by
`Base.dataids`. The full rule, including what an array type must define for its memory to be
seen, is in the docstring of [`xc_evaluate!`](@ref).

## Guarantees

- **Allocation and type stability.** `xc_derivatives`, `energy_density` and `xc_evaluate!` allocate
  nothing and are type-stable; the module ends with `@strict_policy (:typestable, :noalloc)` of
  StrictMode (0.5.20 or later). The only exemptions are `xc_outputs` and its helper
  `_outputs_like`: they return new arrays.
- **Trimming.** `energy_density`, `xc_derivatives` and `xc_evaluate!` at order 2 pass TrimCheck for
  every shipped functional in `Float64` and `Float32`, and the program `PureXC/juliac/entry.jl`,
  which evaluates `pbe()` with `xc_evaluate!`, builds with `juliac --trim=safe` and writes the same
  output as the untrimmed run, bit for bit.
- **Functional interface.** The methods a functional type must define (`family`, `kind`,
  `_with_number_type`, `_float32_max_order`, and `_energy_density` and `_cutoff` for each spin case)
  are TypeContracts (0.14.7 or later) contracts of [`AbstractFunctional`](@ref),
  [`AbstractLDA`](@ref) and [`AbstractGGA`](@ref); `check_contract` reports a missing method.
