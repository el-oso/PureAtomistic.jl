# gga_c_pbe: Perdew–Burke–Ernzerhof correlation

## 1. Identifier

- Libxc identifier `gga_c_pbe`, number 130.
- Family GGA, kind correlation.
- Spin cases: spin-restricted (inputs n, σ) and spin-polarized (inputs n_α, n_β, σ_αα,
  σ_αβ, σ_ββ).

## 2. References

- [PBE] J. P. Perdew, K. Burke, M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996),
  doi:10.1103/PhysRevLett.77.3865. Read for this spec (author copy, dft.uci.edu/pubs/PBE96.pdf).
- [PBE-E] Erratum, Phys. Rev. Lett. 78, 1396 (1997), doi:10.1103/PhysRevLett.78.1396.
  Read; it changes nothing used here.
- [PW92] J. P. Perdew and Y. Wang, Phys. Rev. B 45, 13244 (1992),
  doi:10.1103/PhysRevB.45.13244, through `lda/lda_c_pw_mod.md`.
- [PBE-code] K. Burke, reference Fortran routine for PBE ("PBE alpha2.1", July 1996),
  `PBE.asc` (dft.uci.edu/pubs/PBE.asc). Only its comments and constant values were used.

## 3. Variables and units

Hartree atomic units (e²/a₀ = 1 in [PBE] Eqs. (7), (8)).

| symbol | meaning | unit |
|---|---|---|
| n_α, n_β, n = n_α + n_β | spin and total densities | bohr⁻³ |
| σ_αα, σ_αβ, σ_ββ | contracted spin-density gradients (Libxc's inputs) | bohr⁻⁸ |
| σ_tot = ∇n·∇n = σ_αα + 2σ_αβ + σ_ββ | squared total-density gradient; restricted input σ | bohr⁻⁸ |
| r_s = (3/(4π n))^{1/3} | Seitz radius | bohr |
| ζ = (n_α − n_β)/n | relative spin polarization | — |
| k_F = (3π² n)^{1/3} | Fermi wave number | bohr⁻¹ |
| k_s = (4 k_F / π)^{1/2} | Thomas–Fermi screening wave number | bohr⁻¹ |
| φ(ζ) | spin-scaling factor | — |
| t = \|∇n\| / (2 φ k_s n) | reduced gradient for correlation | — |
| ε_c(r_s, ζ) | uniform-gas correlation energy per electron, `lda/lda_c_pw_mod.md` | hartree |
| H(r_s, ζ, t) | gradient correction per electron | hartree |
| e | correlation energy per volume | hartree bohr⁻³ |

Output layout as in `gga/gga_x_pbe.md`.

## 4. Equations

(E1) Correlation energy, [PBE] Eq. (3):

    E_c = ∫ d³r n [ ε_c(r_s, ζ) + H(r_s, ζ, t) ],    so e = n [ε_c + H]

with r_s, ζ, φ, k_s and t as defined in the text after [PBE] Eq. (3):

    φ(ζ) = [ (1 + ζ)^{2/3} + (1 − ζ)^{2/3} ] / 2
    t² = σ_tot / (4 φ² k_s² n²) = π σ_tot / (16 φ² k_F n²)

(E2) Gradient correction, [PBE] Eq. (7):

    H = γ φ³ ln{ 1 + (β/γ) t² [ (1 + A t²) / (1 + A t² + A² t⁴) ] }

(E3) [PBE] Eq. (8):

    A = (β/γ) [ exp{ −ε_c / (γ φ³) } − 1 ]⁻¹

(E4) ε_c(r_s, ζ) is the PW92 correlation energy per electron **with the constants of
`lda_c_pw_mod`** (see section 6 and section 8). [PBE] cites [PW92] for ε_c (its reference
[12]); the formula is [PW92] Eqs. (8)–(10) with Table I.

(E5) Spin-restricted case: ζ = 0, φ = 1, σ_tot = σ.

The energy depends on the gradients only through σ_tot: [PBE] neglects ∇ζ corrections
("∇ζ corrections to Eq. (3), which are small for most purposes, will be derived in later
work"). Hence ∂e/∂σ_αα = ∂e/∂σ_ββ = ½ ∂e/∂σ_αβ, which Libxc's outputs show.

Properties stated in [PBE]: H → β φ³ t² as t → 0 (Eq. (4)); H → −ε_c as t → ∞ (Eq. (5)),
so e → 0 there; E_c ≤ 0.

## 5. Constants

| constant | value | source | status |
|---|---|---|---|
| β | 0.06672455060314922 | [PBE] after Eq. (4) prints β ≃ 0.066725; the full value is [PBE-code]'s `bet` and Libxc API `_beta` | checked against Libxc |
| γ | (1 − ln 2)/π² = 0.031090690869654895034940863712730… | [PBE] after Eq. (5) | paper |
| A-coefficient multiplier | 1 | Libxc API `_B` ("Multiplies the A t² term. Used in the SPBE functional"); for PBE it is 1, which is (E2) as printed | checked against Libxc |
| PW92 constants | see `lda/lda_c_pw_mod.md` | [PW92] Table I for p, α₁, β₁–β₄; f''(0) exact from [PW92] Eq. (9); A from [PBE-code] and Libxc | paper (p, α₁, β₁–β₄, form of f''(0)); checked against Libxc (A) |

On γ: Libxc's API reports 0.031090690869654894, which is the double nearest the exact
value; [PBE-code] writes the 32-digit decimal 0.03109069086965489503494086371273. A plain
Float64 evaluation of (1 − log(2))/π² gives 0.0310906908696549, 1 ulp higher. An
implementation should state which it uses.

Measured effect of the paper's rounded constants (restricted, grid of section 9): β = 0.066725
instead of the full value gives relative differences from Libxc of 1.3e-5 (zk, vrho, vsigma)
and up to 1.5e-3 (v2rhosigma); building ε_c from `lda_c_pw` (paper PW92 constants) instead of
`lda_c_pw_mod` gives 5.0e-5 (zk) to 6.0e-5 (v2rho2).

## 6. Building blocks

- `lda/lda_c_pw_mod.md`: ε_c(r_s, ζ) and its spin interpolation f(ζ). Measured: PBE
  correlation built on it matches Libxc to ≤ 9.2e-12 (section 9); built on `lda_c_pw` it
  differs by 5e-5.

## 7. Domain, thresholds and limits

Domain: n_α, n_β ≥ 0, σ_αα, σ_ββ ≥ 0, σ ≥ 0. Negative density or σ, NaN or Inf is a caller
error (design: throws), as is σ_αβ² > σ_αα σ_ββ beyond a rounding bound.

### PureXC rule

Parameters, defaults as Libxc 7.0.0 reports them for this functional: density threshold
n_thr = 1e-12; gradient threshold σ_thr = n_thr^{4/3} = 1.0000000000000021e-16 (Float64);
spin-polarization threshold ε_ζ = 2.220446049250313e-16. Applied as follows (application
checked against Libxc, see below):

1. If the total density n < n_thr (strict), the energy and every derivative are zero.
   There is no per-channel cutoff: restricted input is cut off at n < n_thr, not 2 n_thr.
2. Otherwise, in the polarized case each channel density is raised to max(n_σ, n_thr)
   before r_s, ζ and φ are formed.
   As in `lda/lda_c_pw.md` rule 2: every derivative is that of n_r·ε at the raised densities,
   with n_r the sum of the raised channels, and the energy output is e = n·ε with the caller's n.
   Checked against Libxc 7.0.0 outputs: with σ = (1e-4, 0, 0), zk, vrho and vsigma are
   bit-identical at (0.01, 0), (0.01, 0.5e-12) and (0.01, 1e-12) (zk = −0.010227071671042167,
   vrho_β = 16.23626753063028), and at n_α = n_β = 0.5005e-12 and 0.9995e-12.
3. σ_σσ (restricted: σ) is raised to max(σ_σσ, σ_thr²); the floor is the **square** of the
   reported threshold, 1.0000000000000042e-32.
4. In f(ζ) (`lda_c_pw_mod`) and in φ(ζ), a factor (1 ± ζ)^p with 1 ± ζ ≤ ε_ζ is replaced by
   the constant ε_ζ^p.
5. In the polarized case the total σ_αα + 2σ_αβ + σ_ββ is raised to the σ floor σ_thr². It can be
   below it only by rounding, since PureXC rejects σ_αβ beyond √σ_αα·√σ_ββ·(1 + 8·eps) (Libxc
   instead clamps σ_αβ, which PureXC does not copy). Without this, t² < 0 makes the logarithm in
   H undefined. This is a PureXC rule, not checked against Libxc.

### Libxc behavior (information; checked against Libxc 7.0.0 outputs)

- Rule 1: unpolarized n = 0.999e-12 gives zk = 0; n = 1.0e-12 gives zk = −6.6604e-5;
  n = 1.999e-12 and 2.001e-12 give nonzero values (no per-channel cutoff).
- Rule 2: polarized (n_α, n_β) = (0.01, 0), (0.01, 5e-13), (0.01, 1e-12) with
  σ = (1e-3, 0, 0) give identical outputs (zk = −0.0006198385727128573,
  vrho_β = 3.818163979061827, v2rho2_ββ = −1.2791281029994136e12). The derivative with
  respect to a raised channel is the derivative at n_thr, not zero.
- Rule 3: unpolarized n = 1e-11 with σ ∈ {0, 0.5, 1.0} × σ_thr² gives identical outputs
  (zk = −0.00014165595883248967, vsigma = 1.9656605414274226e12); 1.5 σ_thr² differs. Same
  per channel in the polarized case. The reported vsigma at a raised σ is the derivative at
  the floor.
- σ_αβ clamp (Libxc only): after rule 3, Libxc clamps σ_αβ to
  [−(σ_αα + σ_ββ)/2, +(σ_αα + σ_ββ)/2]. Measured at (n_α, n_β) = (0.1, 0.05),
  σ_αα = 1e-2, σ_ββ = 1e-4: σ_αβ = 0.006 gives the same outputs as 0.00505, and −0.006 the
  same as −0.00505, while 9e-4, 1e-3 and 2e-3 (the last beyond √(σ_αα σ_ββ) = 1e-3) are
  used as given. PureXC instead throws for σ_αβ² > σ_αα σ_ββ beyond a rounding bound
  (design), so the oracle grid must stay inside that bound.
- Rule 4 was read in Libxc's source and its effect seen for `lda_c_pw_mod` (see that spec);
  for this functional the effect was **not checked** separately: with n_thr = 1e-12 the
  smallest unraised channel gives 1 − ζ ≥ 2e-12/n, which reaches ε_ζ only for n ≳ 9000.

### Limits

- t → 0: H ≈ β φ³ t² ([PBE] Eq. (4)); e is analytic in σ_tot, so ∂e/∂σ is finite at σ = 0:
  restricted, ∂e/∂σ → β π / (16 (3π²)^{1/3}) n^{−4/3} = 1.9657e12 at n = 1e-11 (Libxc:
  1.9656605414274226e12, the same expression at the σ floor).
- t → ∞: H → −ε_c ([PBE] Eq. (5)), so ε_c + H → 0. The sum is a difference of nearly equal
  numbers: at s = 10 it is about 1e-4 of ε_c (design document, "Validation against Libxc"),
  so both Libxc and a direct evaluation of (E1) lose about 4 digits there. Measured here: the
  largest difference between Libxc and the BigFloat evaluation is 4.1e-12 (restricted zk)
  and 9.2e-12 (polarized zk), at large s.
- φ(ζ) has an infinite derivative at ζ = ±1 ((1 ∓ ζ)^{−1/3}), so as a channel empties the
  first density derivatives grow and the second diverge; Libxc's values there are finite only
  because of rules 2 and 4 (vrho_β = 3.82, v2rho2_ββ = −1.28e12 at (0.01, 0)).

## 8. Known misprints and ambiguities

- Which PW92: [PBE] cites [PW92] for ε_c without stating constants. Libxc and [PBE-code] use
  the constants of `lda_c_pw_mod` (more digits in A, exact f''(0)); this was settled by
  Libxc's outputs (section 6) and is marked checked against Libxc.
- β: [PBE] prints β ≃ 0.066725; Libxc and [PBE-code] use 0.06672455060314922. The origin of
  the extra digits is not checked.
- [PBE-E] corrects the Z → ∞ two-electron correlation energy quoted after Eq. (9)
  (−0.0479 instead of −0.0482 hartree); it does not change any formula used here.
- (E2) applies (β/γ) t² outside the bracket, as printed in Eq. (7); [PBE-code] computes the
  same expression.

## 9. Libxc oracle

- Identifier `gga_c_pbe` (number 130), Libxc 7.0.0 through Libxc.jl, `n_spin = 1` and `2`.
- Flags: exc, vxc, fxc, enforce_fhc, dim3. Supported derivatives [0, 1, 2].
- External parameters reported by the API: `_beta` = 0.06672455060314922 ("beta
  constant"), `_gamma` = 0.031090690869654894 ("(1 - ln(2))/Pi^2 in the PBE"), `_B` = 1.0.
- Thresholds reported by the API: dens_threshold 1.0e-12, sigma_threshold
  1.0000000000000021e-16, zeta_threshold 2.220446049250313e-16.

Agreement of (E1)–(E5) with `lda_c_pw_mod`, evaluated in 256-bit BigFloat, with Libxc in
Float64 (orders 0–2, n log-spaced over [1e-6, 1e3], s ∈ {0, 0.01, 0.3, 1, 3, 10} where
s is the exchange reduced gradient, ζ ∈ {0, 0.25, −0.6, 0.95}, σ_αβ = ½ √(σ_αα σ_ββ)):
largest relative difference, restricted: 4.1e-12 (zk), 1.8e-12 (vrho), 9.4e-13 (vsigma),
1.2e-12 (v2rho2), 1.9e-12 (v2rhosigma), 2.1e-12 (v2sigma2); polarized: 9.2e-12, 7.1e-12,
4.5e-12, 1.2e-11, 1.4e-11, 1.3e-11. These are dominated by the large-t cancellation
described in section 7.

Reference values (Libxc 7.0.0, Float64; last column: largest relative difference of the
equations in BigFloat from the Libxc values in the row):

| input | zk; vrho; vsigma | rel. diff |
|---|---|---|
| n = 1.0e-6, σ = 1.0e-14 | −0.0018251719785951246; −0.006603549782069997; 163445.18311941568 | 1.2e-14 |
| n = 0.01, σ = 0.001 | −0.002082858384624374; −0.011104703381614515; 0.03379190245692333 | 2.2e-15 |
| n = 1.0, σ = 0.5 | −0.06915172038977144; −0.08203337875329407; 0.003964180823079662 | 3.3e-16 |
| n = 100.0, σ = 1.0e4 | −0.11161135888782478; −0.12321899201646803; 8.863660356898457e-6 | 1.1e-16 |
| (n_α, n_β) = (0.3, 0.1), (σ_αα, σ_αβ, σ_ββ) = (0.02, 0.005, 0.01) | −0.05688857138579849; −0.05568950866256309, −0.10201373855798285; 0.013285863471859305, 0.026571726943718614, 0.013285863471859305 | 3.3e-16 |
| (1.0, 0.001), (0.5, 1e-4, 1e-6) | −0.036074487176590946; −0.043655184565981234, −0.27853534969333865; 0.0030625050689373832, 0.0061250101378747665, 0.0030625050689373832 | 7.7e-16 |
