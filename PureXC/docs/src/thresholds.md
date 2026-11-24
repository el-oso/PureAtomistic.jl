# Thresholds and domain

## Thresholds

Each functional applies its own threshold rule, the "PureXC rule" of its spec in
`PureXC/specs/`. The rules apply the thresholds the way Libxc 7.0.0 applies them, checked against
Libxc's outputs, so that the two agree up to the thresholds. The thresholds are a field of the
functional, a [`PureXC.Thresholds`](@ref) in its number type: the density threshold n_thr, the
reported gradient threshold σ_thr, whose square σ_thr² is the floor applied to σ, and the
spin-polarization threshold ε_ζ.

| functional | n_thr | σ_thr | ε_ζ |
|---|---|---|---|
| `lda_x` | 1e-15 | (not used) | (not used) |
| `lda_c_pw`, `lda_c_pw_mod` | 1e-15 | (none) | 2.220446049250313e-16 |
| `pbe_x` | 1e-15 | 1.0000000000000027e-20 | (not used) |
| `pbe_c` | 1e-12 | 1.0000000000000021e-16 | 2.220446049250313e-16 |

The values are Libxc 7.0.0's defaults as its API reports them. For `Float32`
([`PureXC.default_thresholds`](@ref)), ε_ζ is `eps(Float32)`, and a density threshold or σ floor
below `floatmin(Float32)` is raised to `floatmin(Float32)`.

Where a rule raises an input to a floor, the derivatives are those at the floor, not zero. Where a
rule cuts a point off, every output is an exact zero.

### `lda_x`

1. A total density n < n_thr gives zero energy and zero derivatives.
2. Otherwise each spin channel is separate: a channel with n_σ ≤ n_thr contributes zero to e and to
   every derivative; the other contributes −(3/4)(6/π)^(1/3)·n_σ^(4/3).
3. Restricted input is n_α = n_β = n/2, so rule 2 makes every output zero for n ≤ 2·n_thr.

### `lda_c_pw` and `lda_c_pw_mod`

1. A total density n < n_thr (strict) gives zero energy and zero derivatives.
2. Otherwise, polarized, each channel is raised to max(n_σ, n_thr) before r_s and ζ are formed.
   Every derivative is that of n_r·ε_c at the raised densities, with n_r the sum of the raised
   channels; the energy is e = n·ε_c with the caller's n, so that e/n is Libxc's zk.
3. In f(ζ), a factor (1 ± ζ)^(4/3) with 1 ± ζ < ε_ζ (strict, on the rounded ζ) is replaced by the
   constant ε_ζ^(4/3).

### `pbe_x`

1. A total density n < n_thr gives zero energy and zero derivatives.
2. Otherwise each spin channel is separate: a channel with n_σ ≤ n_thr contributes zero.
   Restricted, every output is zero for n ≤ 2·n_thr.
3. σ_σσ (restricted: σ) is raised to max(σ_σσ, σ_thr²); the floor is the square of the reported
   threshold.

### `pbe_c`

1. A total density n < n_thr (strict) gives zero energy and zero derivatives. There is no
   per-channel cutoff: restricted input is cut off at n < n_thr.
2. Otherwise, polarized, each channel is raised to max(n_σ, n_thr) before r_s, ζ and φ are formed,
   with derivatives and energy as in rule 2 of `lda_c_pw`.
3. σ_σσ (restricted: σ) is raised to max(σ_σσ, σ_thr²).
4. In f(ζ) and φ(ζ), a factor (1 ± ζ)^p with 1 ± ζ < ε_ζ (strict, on the rounded ζ) is replaced by
   the constant ε_ζ^p.
5. Polarized, the total σαα + 2σαβ + σββ is raised to σ_thr². It falls below it only by rounding.
   This rule is PureXC's own and not checked against Libxc.

Libxc clamps σαβ to ±(σαα + σββ)/2; PureXC does not, and throws instead for |σαβ| beyond
√σαα·√σββ·(1 + 8·eps).

### Sums

An [`XC`](@ref) applies the rule of each term to that term, and checks its inputs against every
term.

## Domain limits

Above these limits the outputs lose their accuracy, so the functionals throw an `ArgumentError`
there. Each limit is a power of two, measured as the largest at which every output at the order is
within its test bound against BigFloat (the docstrings of the internal functions named below, in
the source, give each measurement).
Orders above 10 are not measured.

| limit | Float64 (and other types) | Float32 |
|---|---|---|
| total density of an LDA, `_lda_density_max` | 2^767 at orders 0 to 6; 2^731, 2^679, 2^627, 2^575 at orders 7 to 10 | 2^95 |
| total density of a GGA, `_gga_density_max` | 2^381 | 2^45 |
| s² of `pbe_x`, per channel, `_pbe_x_s2_max` | 2^1023 | 2^127 |
| t² of `pbe_c`, `_pbe_c_t2_max` | 2^131 | 2^19 at orders 0 to 3; 2^127 at 4 and 5; 2^17, 2^17, 2^16, 2^14, 2^13 at 6 to 10 |

For the LDA, the 2^767 is ⌊3e/4⌋ with e = exponent(floatmax), where n^(4/3) of `lda_x` overflows;
the decrease from order 7 comes from PW92 near full polarization. BigFloat and the other types use
the Float64 values.
