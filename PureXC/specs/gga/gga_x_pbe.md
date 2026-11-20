# gga_x_pbe: Perdew–Burke–Ernzerhof exchange

## 1. Identifier

- Libxc identifier `gga_x_pbe`, number 101.
- Family GGA, kind exchange.
- Spin cases: spin-restricted (inputs n, σ) and spin-polarized (inputs n_α, n_β, σ_αα,
  σ_αβ, σ_ββ).

## 2. References

- [PBE] J. P. Perdew, K. Burke, M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996),
  doi:10.1103/PhysRevLett.77.3865. Read for this spec (author copy, dft.uci.edu/pubs/PBE96.pdf).
- [PBE-E] Erratum, Phys. Rev. Lett. 78, 1396 (1997), doi:10.1103/PhysRevLett.78.1396.
  Read; it changes nothing used here (it corrects Table I, the Z → ∞ two-electron correlation
  energy, and reference [6]).
- [PBE-code] K. Burke, reference Fortran routine for PBE ("PBE alpha2.1", July 1996),
  `PBE.asc` (dft.uci.edu/pubs/PBE.asc). Only its comments and constant values were used.

## 3. Variables and units

Hartree atomic units.

| symbol | meaning | unit |
|---|---|---|
| n_α, n_β, n | spin and total densities | bohr⁻³ |
| σ_αα = ∇n_α·∇n_α, σ_αβ = ∇n_α·∇n_β, σ_ββ = ∇n_β·∇n_β | contracted spin-density gradients (Libxc's inputs) | bohr⁻⁸ |
| σ = ∇n·∇n | restricted input | bohr⁻⁸ |
| k_F(n) = (3π² n)^{1/3} | Fermi wave number | bohr⁻¹ |
| s = \|∇n\| / (2 k_F n) | reduced density gradient | — |
| ε_x(n) = −3 k_F/(4π) | uniform-gas exchange energy per electron (`lda/lda_x.md`) | hartree |
| e | exchange energy per volume | hartree bohr⁻³ |

Outputs follow Libxc's layout: `zk` = e/n; `vrho` (α, β); `vsigma` (αα, αβ, ββ); `v2rho2`
(αα, αβ, ββ); `v2rhosigma` (α-αα, α-αβ, α-ββ, β-αα, β-αβ, β-ββ); `v2sigma2` (αα-αα, αα-αβ,
αα-ββ, αβ-αβ, αβ-ββ, ββ-ββ). Restricted: one entry each, derivatives with respect to n and σ.

## 4. Equations

(E1) Exchange energy for an unpolarized density, [PBE] Eq. (10):

    E_x = ∫ d³r n ε_x(n) F_x(s),    ε_x(n) = −3 k_F(n)/(4π) = −(3/4)(3/π)^{1/3} n^{1/3}

(E2) Enhancement factor, [PBE] Eq. (14):

    F_x(s) = 1 + κ − κ / (1 + μ s² / κ)

equivalently F_x = 1 + κ μ s² / (κ + μ s²), which has no cancellation at small s. F_x(0) = 1,
F_x → 1 + κ as s → ∞, F_x ≈ 1 + μ s² for small s ([PBE] Eq. (12)).

(E3) Spin-restricted energy density, with s from the definition after [PBE] Eq. (9):

    s² = σ / (4 (3π²)^{2/3} n^{8/3})
    e(n, σ) = n ε_x(n) F_x(s) = −(3/4)(3/π)^{1/3} n^{4/3} F_x(s)

(E4) Spin-polarized energy density by the exact spin scaling, [PBE] Eq. (11),
E_x[n_α, n_β] = (E_x[2n_α] + E_x[2n_β])/2, applied pointwise. The density 2n_σ has squared
gradient 4σ_σσ:

    e(n_α, n_β, σ_αα, σ_αβ, σ_ββ) = ½ [ e(2n_α, 4σ_αα) + e(2n_β, 4σ_ββ) ]

so each channel uses s_σ² = σ_σσ / (4 (6π²)^{2/3} n_σ^{8/3}) and contributes
−(3/4)(6/π)^{1/3} n_σ^{4/3} F_x(s_σ). The energy does not depend on σ_αβ; Libxc reports
vsigma_αβ = 0 and zero for every second derivative involving σ_αβ. With n_α = n_β = n/2 and
σ_αα = σ_αβ = σ_ββ = σ/4, (E4) equals (E3).

## 5. Constants

| constant | value | source | status |
|---|---|---|---|
| κ | 0.804 | [PBE] after Eq. (14); Libxc API `_kappa` | paper |
| μ | 0.2195149727645171 | [PBE] after Eq. (12) defines μ = β π²/3 ≃ 0.21951 with β ≃ 0.066725; the full value is [PBE-code]'s `um` and Libxc API `_mu` | checked against Libxc |
| (3/4)(3/π)^{1/3} | 0.7385587663820224… | [PBE] after Eq. (10) | paper |
| 4 (3π²)^{2/3} | from k_F; exact | [PBE] definition of s | paper |

On μ: β π²/3 with β = 0.06672455060314922 (the value in `gga/gga_c_pbe.md`) is
0.21951497276451704… when evaluated in Float64, which is 2 ulp from the decimal
0.2195149727645171 that Libxc reports and uses. Measured: the equations with μ = βπ²/3 evaluated exactly (BigFloat)
reproduce Libxc to the same 3.8e-16 in zk as with μ = 0.2195149727645171. Using the paper's
rounded μ = 0.21951 instead gives relative differences from Libxc of 2.6e-6 (zk), 5.6e-6
(vrho), 2.3e-5 (vsigma) and up to 9.4e-5 (second derivatives). PureXC uses
μ = 0.2195149727645171 as the default.

## 6. Building blocks

- `lda/lda_x.md`: ε_x(n) and its spin scaling (this functional with F_x ≡ 1 is lda_x).

## 7. Domain, thresholds and limits

Domain: n_α, n_β ≥ 0, σ_αα, σ_ββ ≥ 0, σ ≥ 0. Negative density or σ, NaN or Inf is a caller
error (design: throws), as is σ_αβ² > σ_αα σ_ββ beyond a rounding bound.

### PureXC rule

Parameters: density threshold n_thr (default 1e-15) and gradient threshold σ_thr (default
n_thr^{4/3} = 1.0000000000000027e-20 in Float64), both as Libxc 7.0.0 reports them for this
functional. Applied as follows (application checked against Libxc, see below):

1. If the total density n < n_thr, the energy and every derivative are zero.
2. Otherwise each spin channel is treated separately: a channel with n_σ ≤ n_thr contributes
   zero to e and every derivative. Restricted input is n_α = n_β = n/2, so its energy is zero
   for n ≤ 2 n_thr.
3. σ_σσ (restricted: σ) is raised to max(σ_σσ, σ_thr²). The floor is the **square** of the
   reported threshold: 1.0000000000000052e-40, not 1e-20. (At n = 1e-10, the lowest density
   of the oracle grid, σ = 1e-20 already means s² ≈ 1.2e5, so a floor of 1e-20 would change
   the result completely.)

### Libxc behavior (information; checked against Libxc 7.0.0 outputs)

- Rule 1 and 2: unpolarized zk = 0 for n ∈ {0.999, 1.0, 1.001, 1.999, 2.0} × 1e-15 and
  zk = −9.3152e-6 at n = 2.001e-15 (σ = 0). Polarized (n_α, n_β) = (0.01, 0), (0.01, 5e-16)
  and (0.01, 1e-15) with σ = (1e-3, 0, 0) give identical outputs (zk = −0.27975390539018097,
  vrho_β ≈ 7e-17, vsigma_ββ = 0). Each channel density below n_thr is first raised to n_thr.
- Rule 3: unpolarized n = 3e-15 with σ ∈ {0, 0.5, 1.0} × σ_thr² gives identical outputs
  (zk = −1.0655121955534234e-5, vsigma = −9.78023690313513e16); 1.5 σ_thr² gives a different
  zk. The same holds per channel in the polarized case. The reported vsigma at a raised σ
  is the derivative at the floor, not zero.
- Libxc additionally clamps σ_αβ to [−(σ_αα + σ_ββ)/2, +(σ_αα + σ_ββ)/2]; exchange does not
  depend on σ_αβ, so this has no effect here (see `gga/gga_c_pbe.md`).
- Libxc's ζ threshold 2.220446049250313e-16 enters only through the spin factor of each
  channel, as in `lda/lda_x.md`; no measurable effect, not part of the PureXC rule.

### Limits

- s → 0: e → lda_x, with ∂e/∂σ finite (F_x is a function of s²). At σ = 0 the derivative
  with respect to σ is −(3/4)(3/π)^{1/3} n^{4/3} μ / (4 (3π²)^{2/3} n^{8/3}) (restricted),
  large at small n: −9.7877e16 at n = 3e-15; Libxc reports −9.78023690313513e16 there,
  which is the same expression evaluated at the floor σ = σ_thr² (agreement to 16 digits).
- s → ∞: F_x → 1 + κ, the energy stays finite and bounded by the Lieb–Oxford condition
  ([PBE] Eq. (13)).
- As a channel empties, its second derivatives diverge like n_σ^{−2/3} (and faster in σ):
  measured Libxc v2rho2_ββ = −3.18e9, v2sigma2_ββ = 5.9e54 at (n_α, n_β) = (0.01, 1.5e-15),
  σ = (1e-3, 0, 0) raised to the floor.

## 8. Known misprints and ambiguities

- None in [PBE] or [PBE-E] for these formulas.
- [PBE] prints rounded β ≃ 0.066725 and μ ≃ 0.21951; Libxc and [PBE-code] use
  μ = 0.2195149727645171 (section 5).
- s in [PBE] is defined for the total density; the polarized case follows only through the
  spin-scaling relation (E4), not by inserting a spin-polarized s.

## 9. Libxc oracle

- Identifier `gga_x_pbe` (number 101), Libxc 7.0.0 through Libxc.jl, `n_spin = 1` and `2`.
- Flags: exc, vxc, fxc, enforce_fhc, dim3. Supported derivatives [0, 1, 2].
- External parameters reported by the API: `_kappa` = 0.804 ("Asymptotic value of the
  enhancement function"), `_mu` = 0.2195149727645171 ("Coefficient of the 2nd order
  expansion").
- Thresholds reported by the API: dens_threshold 1.0e-15, sigma_threshold
  1.0000000000000027e-20, zeta_threshold 2.220446049250313e-16.

Agreement of (E2)–(E4), evaluated in 256-bit BigFloat, with Libxc in Float64 (orders 0–2,
n log-spaced over [1e-6, 1e3], s ∈ {0, 0.01, 0.3, 1, 3, 10}, ζ ∈ {0, 0.25, −0.6, 0.95},
σ_αβ = ½ √(σ_αα σ_ββ)): largest relative difference, restricted: 3.9e-16 (zk), 4.3e-16
(vrho), 7.9e-16 (vsigma), 2.9e-15 (v2rho2), 3.4e-15 (v2rhosigma), 1.2e-15 (v2sigma2);
polarized: 4.4e-16, 2.3e-15, 6.1e-15, 1.8e-14, 1.5e-14, 6.1e-15.

Reference values (Libxc 7.0.0, Float64; last column: largest relative difference of the
equations in BigFloat from the nonzero Libxc values in the row):

| input | zk; vrho; vsigma | rel. diff |
|---|---|---|
| n = 1.0e-6, σ = 1.0e-14 | −0.009857529291178524; −0.0092956540460543; −144289.43783188995 | 1.2e-16 |
| n = 0.01, σ = 0.001 | −0.23661249554580363; −0.2340118340980689; −0.30551809986125983 | 2.8e-16 |
| n = 1.0, σ = 0.5 | −0.7406686863550702; −0.9819517873661594; −0.004204845830450659 | 1.6e-16 |
| n = 100.0, σ = 1.0e4 | −3.4289981999409207; −4.5695661996270385; −9.117751103206427e-6 | 2.6e-16 |
| (n_α, n_β) = (0.3, 0.1), (σ_αα, σ_αβ, σ_ββ) = (0.02, 0.005, 0.01) | −0.5777797708644895; −0.8290883168542295, −0.5668106267843886; −0.01666238899320948, 0.0, −0.06948722425183211 | 2.4e-16 |
| (1.0, 0.001), (0.5, 1e-4, 1e-6) | −0.9313874433395687; −1.238475205039433, −0.11231928807573745; −0.003346181950688029, 0.0, −16.002793294311036 | 1.2e-13 |
