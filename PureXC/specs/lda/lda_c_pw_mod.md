# lda_c_pw_mod: Perdew–Wang 1992 correlation with the constants used by PBE

## 1. Identifier

- Libxc identifier `lda_c_pw_mod`, number 13 (Libxc name "Perdew & Wang (modified)").
- Family LDA, kind correlation.
- Spin cases: spin-restricted (input n) and spin-polarized (inputs n_α, n_β).

## 2. References

- [PW92] J. P. Perdew and Y. Wang, Phys. Rev. B 45, 13244 (1992),
  doi:10.1103/PhysRevB.45.13244 (equations; see `lda/lda_c_pw.md` for the access note and
  erratum).
- [PBE-code] K. Burke, reference Fortran routine for PBE ("PBE alpha2.1", July 1996),
  `PBE.asc` (dft.uci.edu/pubs/PBE.asc). Libxc's second reference for this functional reads
  "Added extra digits to some constants as in the PBE routine
  (http://dft.rutgers.edu/pubs/PBE.asc)". Only its comments and constant values were used.

## 3. Variables and units

As in `lda/lda_c_pw.md`.

## 4. Equations

Identical to `lda/lda_c_pw.md`, equations (E1)–(E5), with two changes of constants:

1. the three A values of section 5 below replace those of `lda_c_pw`;
2. f''(0) in (E4) is the exact value 4 / (9 (2^{1/3} − 1)), which follows from [PW92]
   Eq. (9) by differentiation, instead of the rounded 1.709921.

All other parameters (p, α₁, β₁–β₄ of the three sets) are those of `lda_c_pw`.

## 5. Constants

| parameter | set 0: ε_c(r_s,0) | set 1: ε_c(r_s,1) | set 2: −α_c(r_s) |
|---|---|---|---|
| p | 1 | 1 | 1 |
| A | 0.0310907 | 0.01554535 | 0.0168869 |
| α₁ | 0.21370 | 0.20548 | 0.11125 |
| β₁ | 7.5957 | 14.1189 | 10.357 |
| β₂ | 3.5876 | 6.1977 | 3.6231 |
| β₃ | 1.6382 | 3.3662 | 0.88026 |
| β₄ | 0.49294 | 0.62517 | 0.49671 |

| constant | value | source | status |
|---|---|---|---|
| A (three sets) | 0.0310907, 0.01554535, 0.0168869 | [PBE-code] (the values it passes for the three sets); Libxc API `_a[0..2]` | checked against Libxc |
| f''(0) | 4 / (9 (2^{1/3} − 1)) = 8 / (9 (2^{4/3} − 2)) = 1.7099209341613657… (Float64: 1.7099209341613657, printed by Libxc's API; the Float64 evaluation of either expression gives 1.7099209341613653, 2 ulp lower) | exact from [PW92] Eq. (9); [PBE-code] uses 8/(9(2^{4/3} − 2)); Libxc API `_fz20` | paper (form); checked against Libxc (value) |
| α₁, β₁–β₄, p | as in `lda/lda_c_pw.md` | [PW92] Table I via Libxc API; same in [PBE-code] | checked against Libxc |

The A values agree with A = (1 − ln 2)/π² = 0.0310906908696549… (set 0) and A/2 (set 1),
the exact high-density coefficients, to the digits given; whether [PW92] or [PBE-code] gives
this as their origin is not checked.

## 6. Building blocks

`lda/lda_c_pw.md` (the formula). Used by `gga/gga_c_pbe.md`, which requires exactly this
variant.

## 7. Domain, thresholds and limits

Identical to `lda/lda_c_pw.md` section 7: same domain, same PureXC rule with the same
defaults (n_thr = 1e-15, ε_ζ = 2.220446049250313e-16, read from Libxc's API for
`lda_c_pw_mod`), and the same Libxc behavior. Measured for this functional: polarized
(n_α, n_β) = (0.01, 0), (0.01, 0.5e-15), (0.01, 1e-15) give identical outputs
(zk = −0.020296576673299066, vrho_β = −0.15101107190123725); unpolarized n = 0.999e-15 gives
zk = 0, n = 1.0e-15 gives zk = −6.8960262e-6.

## 8. Known misprints and ambiguities

- The A values differ from those of `lda_c_pw` in the 6th–7th significant digit. Measured: Libxc's
  `lda_c_pw` and `lda_c_pw_mod` differ by up to 5.9e-6 relative in zk and 3.2e-5 in vrho
  (polarized, n log-spaced over [1e-6, 1e3], ζ ∈ {0, 0.25, −0.6, 0.95}). Evaluating either
  functional with the other's f''(0) alone leaves a 5.2e-9 relative difference in zk.
- Libxc's API prints `_fz20` = 1.7099209341613657, which is the double nearest the exact
  value; a plain Float64 evaluation of 4/(9(cbrt(2) − 1)) gives 1.7099209341613653. The
  2-ulp difference is below the comparison tolerance; an implementation should state which
  it uses.

## 9. Libxc oracle

- Identifier `lda_c_pw_mod` (number 13), Libxc 7.0.0 through Libxc.jl, `n_spin = 1` and `2`.
- Flags: exc, vxc, fxc, enforce_fhc, dim3. Supported derivatives [0, 1, 2].
- External parameters reported by the API: as `lda_c_pw` except `_a[0..2]` = 0.0310907,
  0.01554535, 0.0168869 and `_fz20` = 1.7099209341613657.
- Thresholds reported by the API: dens_threshold 1.0e-15, zeta_threshold
  2.220446049250313e-16.

Agreement of the equations, evaluated in 256-bit BigFloat, with Libxc in Float64 (orders 0–2,
same grid as `lda_c_pw`): largest relative difference 9.9e-15 (zk), 5.2e-14 (vrho),
2.9e-13 (v2rho2).

Reference values (Libxc 7.0.0, Float64; last column as in `lda/lda_x.md`):

| input | zk; vrho | rel. diff |
|---|---|---|
| n = 1.0e-6 | −0.004771623035336743; −0.006085885635601924 | 5.5e-15 |
| n = 0.01 | −0.03769764282445714; −0.04387597615794007 | 2.8e-16 |
| n = 1.0 | −0.07120005886619186; −0.07945690779111174 | 1.1e-16 |
| n = 100.0 | −0.11251060623211576; −0.12205005230913385 | 1.8e-16 |
| (n_α, n_β) = (0.3, 0.1) | −0.05824939474999474; −0.05385574005674182, −0.10055836084415391 | 4.4e-16 |
| (1.0, 0.001) | −0.037683529511408645; −0.04168306420165874, −0.2870566027090974 | 1.3e-15 |
| (0.02, 0.02) | −0.046753101645079384; −0.05362859476471246, −0.05362859476471246 | 1.8e-16 |
