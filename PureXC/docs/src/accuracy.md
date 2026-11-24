# Accuracy and speed

Every figure on this page is a bound or a measurement stated in a committed test or record of the
repository; each table names its source.

## Oracle grid

Source: `PureXC/test/libxc/grid.jl`.

- ρ, the total density: 131 points log-spaced over [1e-10, 1e3];
- s, the reduced gradient: 0 and 41 points log-spaced over [1e-3, 10];
- ζ, the spin polarization: −1, −(1 − eps), −0.6, 0, 0.25, 0.95, 1 − eps, 1 (eps = eps(Float64)).

A polarized GGA point has the same s in both channels and parallel gradients, σαβ = √σαα·√σββ. A
point is *above* when it is above every threshold of the functional (every term, for a sum), and
*below* otherwise. Every gated comparison takes the above points; the rule items of the main suite
gate the below points.

## Float64 against Libxc and BigFloat

Source: `PureXC/test/libxc/runtests.jl`, run by the CI job `purexc-libxc`. The oracles are Libxc
7.0.0 at orders 0 to 2 and the same Julia functional converted exactly to 256-bit BigFloat. Every
component of every output field is compared in Libxc's layout. Where the BigFloat value of a
component is exactly 0, its magnitude is replaced by the largest magnitude in its field at that
point.

Bounds on every above point:

| item | orders | bound |
|---|---|---|
| (A) against Libxc | 0 to 2 | \|jl − libxc\| ≤ 1e-6·\|libxc\| per component |
| (B) conditioning against BigFloat | 0 to 2 | \|jl − big\| ≤ c_B·max(\|libxc − big\|, 4·eps·\|big\|), c_B below |
| against BigFloat | 3 and 4 | relative error ≤ 1e-6 (order 4 on ten points per population) |
| Richardson differences of Libxc's order 2 against BigFloat | 3 | relative error ≤ 1e-3 (each entry against the larger of its \|BigFloat\| value and 1e-6 times the field maximum) |

Item (B) measures PureXC's error against Libxc's own distance from the exact value of the same
formula, so the bound at each point scales with the conditioning of the formula there. The constant c_B and the largest ratio measured over the above points (`C_B`):

| functional | c_B restricted | measured | c_B polarized | measured |
|---|---|---|---|---|
| `lda_x` | 13 | 1.251 | 14 | 1.322 |
| `lda_c_pw` | 21 | 2.021 | 50 | 4.972 |
| `lda_c_pw_mod` | 22 | 2.145 | 58 | 5.758 |
| `pbe_x` | 490 | 48.89 | 690 | 68.24 |
| `pbe_c` | 940 | 35.12 | 3400 | 33.14 |
| `pbe` | 1.2e4 | 1121 | 1400 | 132.8 |

Each c_B is about 10 times its measurement, except `pbe_c`, about 27 times (restricted) and 100
times (polarized).

## Float32

Source: `PureXC/test/libxc/runtests.jl`. A `Float32` call is compared with the exact function of
the `Float32` functional (its parameters rounded to `Float32`), so the comparison measures the
evaluation error alone. The bound, at orders 0 to 4, is

max(|jl32 − big32| − floatmin(Float32), 0) ≤ c_32·eps(Float32)·max(1, κ̂)·|big32|,

with κ̂ the `Float64` error at the same point in ulps of `Float64`: where the formula is
ill-conditioned, the `Float64` evaluation shows it, and the `Float32` bound grows with it. An
infinity is accepted only where the exact value is beyond `floatmax(Float32)` with the same sign;
no other output may be non-finite. The constant and the largest ratio measured (`C_32`):

| functional | c_32 restricted | measured | c_32 polarized | measured |
|---|---|---|---|---|
| `lda_x` | 33 | 3.266 | 39 | 3.874 |
| `lda_c_pw` | 79 | 7.818 | 320 | 31.45 |
| `lda_c_pw_mod` | 73 | 7.236 | 220 | 21.97 |
| `pbe_x` | 1600 | 160.0 | 1600 | 159.0 |
| `pbe_c` | 1.8e4 | 1736 | 1900 | 187.2 |
| `pbe` | 4.7e4 | 4653 | 5500 | 549.1 |

The measured values exclude order 3 of polarized correlation, whose maxima are within c_32:
`lda_c_pw` 0.45, `lda_c_pw_mod` 0.47, `pbe_c` 870.5. The largest `pbe_c` values, 1736 (restricted)
and 870.5 (polarized), both at `v3rho3`, are at outputs whose condition numbers in the inputs,
measured in BigFloat, are 142.5 and 974.

## Known differences

Source: `PureXC/test/libxc/runtests.jl` (`KNOWN_COUNT`). At a polarized point within one rounding
of 1 ± ζ = ε_ζ, one evaluation keeps the factor (1 ± ζ)^p and another replaces it by the constant
of the threshold rule, so their derivatives differ by the series part of that factor. These points
are excluded from the gated items and listed with their count and largest differences. They are
exactly the grid rows ζ = ±(1 − eps):

| functional | restricted | polarized |
|---|---|---|
| `lda_x`, `lda_c_pw`, `lda_c_pw_mod` | 0 | 262 (2·131) |
| `pbe_x`, `pbe_c`, `pbe` | 0 | 11004 (2·131·42) |

## Bounds of the main suite

The main suite compares every functional with its exact BigFloat function at orders 0 to 10, at
sweeps of densities, spin splits and reduced gradients up to the domain limits. The bounds are in
eps of the result type: for the LDA relative to the largest entry of each output, for the PBE
functionals per entry against the scale of its order. Each bound above order 3 is about 10 times
the largest error that `PureXC/test/scans/point_bounds.jl` measures over every point it covers.

LDA (source: `point_bound` in `PureXC/test/lda_tests.jl`):

| case | bound | measured |
|---|---|---|
| orders 0 to 3 | 2100 | rounding bound, not a measurement |
| orders 4 to 10 | 1000 | at most 12.8, 15.6, 15.6, 43.5, 43.5, 58.6, 58.6 at orders 4 to 10 |
| PW92 where the factor (1 ± ζ)^(4/3) of the smaller channel is kept | the above + 8κ, κ = 2/(1 − \|ζ\|) | at most 3.1·κ |
| Float32 above `_float32_max_order` (PW92, orders 3 to 10) | 5 | at most 0.49 |

PBE (source: `point_bound` in `PureXC/test/pbe_tests.jl`):

| case | bound | measured |
|---|---|---|
| orders 0 to 3 | 760 (`pbe_x`), 7800 (`pbe_c`) | rounding bound, not a measurement |
| orders 4 to 7 | 300 | at most 29.9 (polarized `pbe_x`, order 6; `pbe_c` 20.5) |
| orders 8 to 10 | 700 | at most 69.8 (restricted `pbe_c`, order 10; `pbe_x` 40.2) |
| `pbe_c` where the factors (1 ± ζ)^p of the smaller channel are kept | the above + 8κ_ζ | |
| Float32 above `_float32_max_order` (orders 4 to 10) | 5 | at most 0.50 (polarized `pbe_c`, order 4; `pbe_x` 0.47) |

The 8κ terms are the conditioning of 1 ± ζ formed from ζ: where one channel is far smaller than
the other, κ is large, and errors of order 1 relative are the documented conditioning of the
formula, not an evaluation error.

## Speed against Libxc

Source: `PureXC/bench/results/libxc_speed_mac.json`, written by `PureXC/bench/libxc_speed.jl`.
Apple M6, Julia 1.13.4, Libxc 7.0.0 through Libxc.jl 0.3.27, one thread in each arm, 10^6 points
drawn from the above points of the oracle grid, PureXC at commit a114766. Each of 7 rounds runs
one child process per arm, in alternating order, and reports the median of three samples per cell;
a fixed calibration loop discards rounds in which the children ran at different speeds (none was
discarded). A cell computes every order from 0 to k into preallocated outputs, and the PureXC arm
includes its input check.

Speed-up is the Libxc time over the PureXC time, so above 1 means PureXC is faster: the median and
[min, max] over the rounds. The `Float32` column divides Libxc's `Float64` time on the same points
(Libxc runs in `Float64` only) by PureXC's `Float32` time. The gate is on the `Float64` cells: the
minimum speed-up of each must be above 1.

| functional | spin | k | Float64 | Float32 |
|---|---|---|---|---|
| `lda_x` | restricted | 0 | 1.44 [1.44, 1.48] | 3.59 [3.57, 3.74] |
| | | 1 | 1.46 [1.45, 1.52] | 3.51 [3.47, 3.63] |
| | | 2 | 1.43 [1.42, 1.48] | 3.37 [3.35, 3.50] |
| | polarized | 0 | 2.47 [2.47, 2.53] | 5.60 [5.50, 5.73] |
| | | 1 | 2.82 [2.82, 2.91] | 6.38 [6.35, 6.58] |
| | | 2 | 4.30 [4.28, 4.42] | 10.01 [9.85, 10.22] |
| `lda_c_pw` | restricted | 0 | 10.07 [9.86, 10.16] | 27.01 [25.99, 27.32] |
| | | 1 | 7.33 [7.22, 7.47] | 19.02 [18.65, 19.42] |
| | | 2 | 6.68 [6.57, 6.76] | 16.38 [16.17, 16.93] |
| | polarized | 0 | 4.49 [4.41, 4.58] | 10.56 [10.37, 10.73] |
| | | 1 | 3.86 [3.84, 3.97] | 8.93 [8.88, 9.17] |
| | | 2 | 3.84 [3.84, 3.96] | 8.93 [8.84, 9.08] |
| `lda_c_pw_mod` | restricted | 0 | 10.14 [9.97, 10.30] | 27.17 [26.75, 27.23] |
| | | 1 | 7.46 [7.39, 7.49] | 19.53 [19.45, 19.63] |
| | | 2 | 6.70 [6.66, 6.84] | 16.91 [16.64, 17.02] |
| | polarized | 0 | 4.46 [4.44, 4.55] | 10.63 [10.47, 10.80] |
| | | 1 | 3.86 [3.82, 3.97] | 8.94 [8.91, 9.16] |
| | | 2 | 3.84 [3.81, 3.94] | 8.89 [8.83, 9.16] |
| `pbe_x` | restricted | 0 | 1.40 [1.38, 1.46] | 3.40 [3.37, 3.57] |
| | | 1 | 1.15 [1.15, 1.20] | 2.66 [2.64, 2.76] |
| | | 2 | 1.05 [1.05, 1.09] | 2.35 [2.34, 2.44] |
| | polarized | 0 | 1.53 [1.50, 1.56] | 3.55 [3.50, 3.71] |
| | | 1 | 1.39 [1.39, 1.44] | 3.18 [3.15, 3.23] |
| | | 2 | 1.11 [1.11, 1.14] | 1.34 [1.33, 1.37] |
| `pbe_c` | restricted | 0 | 2.18 [2.17, 2.19] | 6.40 [6.36, 6.45] |
| | | 1 | 1.79 [1.77, 1.79] | 4.60 [4.55, 4.62] |
| | | 2 | 1.67 [1.65, 1.69] | 3.97 [3.92, 4.00] |
| | polarized | 0 | 1.52 [1.52, 1.53] | 4.04 [4.01, 4.06] |
| | | 1 | 1.45 [1.43, 1.46] | 3.45 [3.44, 3.52] |
| | | 2 | 1.92 [1.91, 1.94] | 4.35 [4.32, 4.40] |
| `pbe` | restricted | 0 | 2.30 [2.30, 2.31] | 5.85 [5.80, 5.86] |
| | | 1 | 1.62 [1.61, 1.62] | 4.04 [4.03, 4.05] |
| | | 2 | 1.65 [1.65, 1.70] | 3.86 [3.84, 3.97] |
| | polarized | 0 | 1.63 [1.62, 1.65] | 4.04 [3.99, 4.10] |
| | | 1 | 1.54 [1.53, 1.58] | 3.59 [3.58, 3.69] |
| | | 2 | 1.13 [1.12, 1.15] | 1.38 [1.38, 1.40] |

Known bias: Libxc evaluates `pbe` as `gga_x_pbe` and `gga_c_pbe` into separate outputs and does
not sum them, while PureXC returns the sum, so the Libxc arm of the `pbe` cells does less work
than the PureXC arm. How much this changes the `pbe` speed-ups is not measured. The times are
those of one machine; other machines have not been measured.
