# lda_c_pw: Perdew–Wang 1992 correlation of the uniform electron gas

## 1. Identifier

- Libxc identifier `lda_c_pw`, number 12.
- Family LDA, kind correlation.
- Spin cases: spin-restricted (input n) and spin-polarized (inputs n_α, n_β).
- This spec uses the constants as printed in [PW92] Table I. `lda/lda_c_pw_mod.md` is the same
  formula with the more precise constants that PBE uses.

## 2. References

- [PW92] J. P. Perdew and Y. Wang, Phys. Rev. B 45, 13244 (1992),
  doi:10.1103/PhysRevB.45.13244. Erratum: Phys. Rev. B 98, 079904 (2018),
  doi:10.1103/PhysRevB.98.079904 (corrects Eq. (B3) only; see section 8).

Equation and table numbers below are those of [PW92] itself (read for this spec).

## 3. Variables and units

Hartree atomic units.

| symbol | meaning | unit |
|---|---|---|
| n_α, n_β, n = n_α + n_β | spin and total densities | bohr⁻³ |
| r_s = (3 / (4π n))^{1/3} | Seitz radius, [PW92] Eq. (1) | bohr |
| ζ = (n_α − n_β) / n | relative spin polarization, [PW92] Eq. (2); in [−1, 1] | — |
| ε_c(r_s, ζ) | correlation energy per electron | hartree |
| e = n ε_c | correlation energy per volume | hartree bohr⁻³ |

Outputs follow Libxc's layout: `zk` = e/n; `vrho` = ∂e/∂n or (∂e/∂n_α, ∂e/∂n_β);
`v2rho2` ordered αα, αβ, ββ. Derivatives are with respect to the densities, not r_s or ζ.

## 4. Equations

(E1) The fitting function, [PW92] Eq. (10):

    G(r_s; A, α₁, β₁, β₂, β₃, β₄, p) =
        −2A (1 + α₁ r_s) ln[ 1 + 1 / (2A (β₁ r_s^{1/2} + β₂ r_s + β₃ r_s^{3/2} + β₄ r_s^{p+1})) ]

(E2) Three instances of G with the parameter sets of section 5 ([PW92] Table I):

    ε_c(r_s, 0) = G(r_s; set 0)          (unpolarized)
    ε_c(r_s, 1) = G(r_s; set 1)          (fully polarized)
    −α_c(r_s)   = G(r_s; set 2)          (minus the spin stiffness)

Set 2 parametrizes −α_c, not α_c: the column of [PW92] Table I is headed −α_c(r_s). The
spin stiffness α_c is defined by the small-ζ expansion ε_c(r_s, ζ) = ε_c(r_s, 0) + ½ α_c(r_s) ζ² + …
([PW92] Eq. (3)), i.e. α_c = ∂²ε_c/∂ζ² at ζ = 0.

(E3) Spin interpolation function, [PW92] Eq. (9):

    f(ζ) = [ (1 + ζ)^{4/3} + (1 − ζ)^{4/3} − 2 ] / (2^{4/3} − 2)

so f(0) = 0, f(±1) = 1, and f''(0) = (8/9) / (2^{4/3} − 2) = 4 / (9 (2^{1/3} − 1)) = 1.70992093…
[PW92] states f(0) = 0, f(1) = 1 and prints f''(0) = 1.709921 in the text after Eq. (9).

(E4) Correlation energy per electron, [PW92] Eq. (8):

    ε_c(r_s, ζ) = ε_c(r_s, 0)
                + α_c(r_s) · f(ζ)/f''(0) · (1 − ζ⁴)
                + [ε_c(r_s, 1) − ε_c(r_s, 0)] · f(ζ) · ζ⁴

with α_c(r_s) = −G(r_s; set 2) and f''(0) the constant of section 5 (for this functional the
rounded value 1.709921, not the exact expression; see section 8).

(E5) Energy density: e = n ε_c(r_s(n), ζ). Spin-restricted input is ζ = 0, where
e(n) = n G(r_s; set 0).

[PW92] Appendix A gives the corresponding potential, ∂(n ε_c)/∂n_σ, in Eqs. (A1)–(A8)
(Eq. (A4) is f'(ζ); Eqs. (A5)–(A8) are ∂G/∂r_s). They are an independent check of `vrho`;
nothing in this spec depends on them.

## 5. Constants

The parameters of [PW92] Table I, beyond-RPA columns ε_c(r_s,0), ε_c(r_s,1), −α_c(r_s).
Status: **paper**; every value below was compared with the table and matches as printed.
Table I marks p and A = c₀ as constrained to exact values. Libxc 7.0.0 reports exactly these
values as the default external parameters of `lda_c_pw`.

| parameter | set 0: ε_c(r_s,0) | set 1: ε_c(r_s,1) | set 2: −α_c(r_s) |
|---|---|---|---|
| p | 1 | 1 | 1 |
| A | 0.031091 | 0.015545 | 0.016887 |
| α₁ | 0.21370 | 0.20548 | 0.11125 |
| β₁ | 7.5957 | 14.1189 | 10.357 |
| β₂ | 3.5876 | 6.1977 | 3.6231 |
| β₃ | 1.6382 | 3.3662 | 0.88026 |
| β₄ | 0.49294 | 0.62517 | 0.49671 |

| constant | value | source | status |
|---|---|---|---|
| f''(0) | 1.709921 | [PW92] text after Eq. (9) (the exact value from Eq. (9) is 4/(9(2^{1/3} − 1)) = 1.7099209341613657…); also Libxc API `_fz20` | paper |
| 2^{4/3} − 2 | 0.5198420997897464… | exact, from (E3) | paper |

Measured: using the exact f''(0) instead of 1.709921 changes the polarized Libxc comparison
from ≤ 1.1e-13 to 5.2e-9 (zk), 3.0e-8 (vrho), 1.6e-7 (v2rho2) relative, so Libxc's
`lda_c_pw` uses the rounded 1.709921.

## 6. Building blocks

None. Used (with the constants of `lda/lda_c_pw_mod.md`) by `gga/gga_c_pbe.md`.

## 7. Domain, thresholds and limits

Domain: n_α, n_β ≥ 0, n > 0. A negative density, NaN or Inf is a caller error (design:
throws).

### PureXC rule

Parameters: density threshold n_thr (default 1e-15) and spin-polarization threshold ε_ζ
(default 2.220446049250313e-16 = eps(Float64)); both are the Libxc 7.0.0 defaults read from its
API. Applied as follows (application checked against Libxc, see below):

1. If the total density n < n_thr (strict), the energy and every derivative are zero.
2. Otherwise, in the spin-polarized case each channel density is raised to max(n_σ, n_thr)
   before r_s and ζ are formed.
   The raised channels also give the total density n_r = max(n_α, n_thr) + max(n_β, n_thr).
   Every derivative is that of n_r·ε_c at the raised densities. The energy output is
   e = n·ε_c with the caller's n, so that e/n equals Libxc's zk (checked against Libxc, below).
3. In (E3), a factor (1 ± ζ)^{4/3} with 1 ± ζ < ε_ζ is replaced by the constant ε_ζ^{4/3}. The comparison is strict and uses the rounded ζ = (n_α − n_β)/n: at ρ = (1000, 1e-13) the rounded 1 − ζ equals ε_ζ, and Libxc keeps the factor (v2rho2_ββ = 4.65e6).

### Libxc behavior (information; checked against Libxc 7.0.0 outputs)

- Rule 1: unpolarized n = 0.999e-15 gives zk = 0; n = 1.0e-15 gives zk = −6.8960e-6 (so
  equality is not cut off). Polarized: the test is on n_α + n_β.
- Which density multiplies ε_c (checked against Libxc 7.0.0 outputs): zk and vrho are
  bit-identical at (0.01, 0), (0.01, 0.5e-15) and (0.01, 1e-15), and at n_α = n_β = 0.5005e-15,
  0.9995e-15 and 1e-15 (`lda_c_pw`: zk = −8.674368563248956e-6, vrho = −1.1544342326421484e-5 for
  each channel). So zk is ε_c at the raised densities, and vrho is the derivative of n_r·ε_c
  there; a product with the caller's n would change vrho at the 1e-13 level at the first three
  points and by a factor near 2 in its second term at the symmetric points.
- Rule 2: polarized (n_α, n_β) = (0.01, 0), (0.01, 0.5e-15) and (0.01, 1e-15) give identical
  outputs (zk = −0.020296494557675415, vrho_β = −0.15101284284991515,
  v2rho2_ββ = 1.740992732755484e9). The derivative with respect to a raised channel is the
  derivative at n_σ = n_thr, not zero. Polarized n_α = n_β = 0.5005e-15 and
  n_α = n_β = 0.9995e-15 give the same zk: both channels are raised to n_thr.
- Rule 3 (read in Libxc's source, effect seen in outputs): at n_α = 1000,
  v2rho2_ββ = 4.6e6 for n_β = 1e-13 (1 − ζ = 2e-16 after rounding of n) and 2.6e-3 for
  n_β = 1e-14 (1 − ζ = 2e-17), where the replaced factor contributes no derivative. At such
  points ζ is at the rounding resolution of n, so only the qualitative effect is checked.
- Libxc does not report a σ or τ threshold for an LDA in any way that affects it.

### Limits

- ε_c is finite for all r_s > 0 and tends to 0 as r_s → ∞ (n → 0).
- f(ζ) has infinite second derivative at ζ = ±1 ((1 ∓ ζ)^{−2/3}), so the second density
  derivatives diverge as a channel empties; Libxc gives finite values there only because of
  rules 2 and 3 (measured v2rho2_ββ = 1.74e9 at (0.01, 0)).

## 8. Known misprints and ambiguities

- 1 ± ζ is formed from ζ = (n_α − n_β)/n, as the ζ-factor rule of section 7 needs, so it
  loses digits at nearly full polarization. Measured: at (0.01, 0), with n_β raised to n_thr,
  v2rho2_ββ is 5.3e-4 relative from the value with 1 − ζ formed exactly as 2n_β/n (BigFloat),
  and agrees with Libxc 7.0.0. Forming 1 ± ζ exactly would also move the points where the
  ζ-factor rule applies, so this spec keeps the ζ form. Such points are below the thresholds.
- Erratum (2018): Eq. (B3) of [PW92] should read tan⁻¹(1/u) instead of tan⁻¹(u). (B3) is
  not used here; it does not affect (E1)–(E4).
- f''(0): [PW92] prints the rounded 1.709921 after Eq. (9), while Eq. (9) itself gives
  1.7099209341613657…. This spec, like Libxc's `lda_c_pw`, uses the printed value;
  `lda_c_pw_mod` uses the exact expression. Not a misprint: the printed value is the exact one
  rounded to 7 digits.
- The sign convention of set 2 (it fits −α_c) is a common source of error; (E2) states it.
- p = 1 in all three sets, so the last term of (E1) is β₄ r_s².

## 9. Libxc oracle

- Identifier `lda_c_pw` (number 12), Libxc 7.0.0 through Libxc.jl, `n_spin = 1` and `2`.
- Flags: exc, vxc, fxc, enforce_fhc, dim3. Supported derivatives [0, 1, 2].
- External parameters reported by the API (name: default): `_pp[0..2]` = 1, 1, 1;
  `_a[0..2]` = 0.031091, 0.015545, 0.016887; `_alpha1[0..2]` = 0.2137, 0.20548, 0.11125;
  `_beta1[0..2]` = 7.5957, 14.1189, 10.357; `_beta2[0..2]` = 3.5876, 6.1977, 3.6231;
  `_beta3[0..2]` = 1.6382, 3.3662, 0.88026; `_beta4[0..2]` = 0.49294, 0.62517, 0.49671;
  `_fz20` = 1.709921.
- Thresholds reported by the API: dens_threshold 1.0e-15, zeta_threshold
  2.220446049250313e-16.

Agreement of (E1)–(E5), evaluated in 256-bit BigFloat, with Libxc in Float64 (orders 0–2,
n log-spaced over [1e-6, 1e3], ζ ∈ {0, 0.25, −0.6, 0.95}): largest relative difference
9.3e-15 (zk), 3.0e-14 (vrho), 1.1e-13 (v2rho2).

Reference values (Libxc 7.0.0, Float64; last column as in `lda/lda_x.md`):

| input | zk; vrho | rel. diff |
|---|---|---|
| n = 1.0e-6 | −0.00477162315902391; −0.0060858858656315734 | 8.9e-15 |
| n = 0.01 | −0.03769770328922326; −0.04387606205358233 | 2.8e-16 |
| n = 1.0 | −0.07120031359839032; −0.0794572203196884 | 2.0e-17 |
| n = 100.0 | −0.1125111809948515; −0.12205070626815523 | 2.4e-17 |
| (n_α, n_β) = (0.3, 0.1) | −0.05824958052852496; −0.05385591326647189, −0.10055877615182211 | 5.4e-16 |
| (1.0, 0.001) | −0.03768321214182925; −0.04168267033646179, −0.2870635770635308 | 3.3e-15 |
| (0.02, 0.02) | −0.04675320448391642; −0.0536287333531124, −0.0536287333531124 | 1.1e-16 |
