# lda_x: Slater (Dirac) exchange of the uniform electron gas

## 1. Identifier

- Libxc identifier `lda_x`, number 1.
- Family LDA, kind exchange.
- Spin cases: spin-restricted (input n) and spin-polarized (inputs n_α, n_β).

## 2. References

- [D] P. A. M. Dirac, Math. Proc. Cambridge Philos. Soc. 26, 376 (1930), doi:10.1017/S0305004100016108.
- [B] F. Bloch, Z. Phys. 57, 545 (1929), doi:10.1007/BF01340281.
- [PBE] J. P. Perdew, K. Burke, M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996),
  doi:10.1103/PhysRevLett.77.3865. Used for the explicit form of the uniform-gas exchange
  energy (text after its Eq. (10)) and for the exact spin-scaling relation (its Eq. (11)).

[D] and [B] are the references Libxc lists for this functional. Their equation numbers are
**not checked**: neither paper was read for this spec. Every equation below is taken from
[PBE], which states it explicitly.

## 3. Variables and units

Hartree atomic units (energies in hartree, lengths in bohr, e = ħ = m = 1).

| symbol | meaning | unit |
|---|---|---|
| n_α, n_β | spin densities | bohr⁻³ |
| n = n_α + n_β | total density | bohr⁻³ |
| k_F(n) = (3π² n)^{1/3} | Fermi wave number of density n | bohr⁻¹ |
| ε_x(n) | exchange energy per electron of the unpolarized gas | hartree |
| e | exchange energy per volume (the energy density) | hartree bohr⁻³ |

Outputs follow Libxc's layout: `zk` = e/n (energy per electron); `vrho` = ∂e/∂n (restricted) or
(∂e/∂n_α, ∂e/∂n_β); `v2rho2` = second derivatives, ordered αα, αβ, ββ.

## 4. Equations

(E1) Unpolarized exchange energy per electron, [PBE] text after Eq. (10):

    ε_x(n) = −3 k_F(n) / (4π) = −(3/4) (3/π)^{1/3} n^{1/3}

(E2) Spin-restricted energy density:

    e(n) = n ε_x(n) = −(3/4) (3/π)^{1/3} n^{4/3}

(E3) Exact spin scaling, [PBE] Eq. (11): E_x[n_α, n_β] = (E_x[2n_α] + E_x[2n_β]) / 2. Applied
pointwise to (E2):

    e(n_α, n_β) = ½ [ e(2n_α) + e(2n_β) ] = −(3/4) (6/π)^{1/3} (n_α^{4/3} + n_β^{4/3})

The spin-polarized energy density is a sum of two independent channel terms; no spin
polarization variable ζ is needed. With n_α = n_β = n/2, (E3) equals (E2).

## 5. Constants

| constant | value | source | status |
|---|---|---|---|
| prefactor in (E2) | (3/4)(3/π)^{1/3} = 0.7385587663820224… | [PBE] after Eq. (10), from ε_x = −3k_F/(4π) | paper |
| prefactor in (E3) | (3/4)(6/π)^{1/3} = 2^{1/3} × the above | [PBE] Eq. (11) applied to (E2) | paper |

No external parameters: Libxc 7.0.0 reports none for this functional.

## 6. Building blocks

None. This exchange energy per electron, ε_x, is used by `gga/gga_x_pbe.md`.

## 7. Domain, thresholds and limits

Domain: n_α, n_β ≥ 0. A negative density, NaN or Inf is a caller error (design: throws).

### PureXC rule

Parameter: density threshold n_thr, default 1e-15 (Libxc 7.0.0 default for this functional,
read from its API). Applied as follows (application checked against Libxc, see below):

1. If the total density n < n_thr, the energy and every derivative are zero.
2. Otherwise each spin channel is treated separately: a channel whose density n_σ ≤ n_thr
   contributes zero to e and to every derivative; the other channel contributes
   −(3/4)(6/π)^{1/3} n_σ^{4/3}.
3. Spin-restricted input is the case n_α = n_β = n/2, so rule 2 makes the energy zero for
   n ≤ 2 n_thr.

### Libxc behavior (information; checked against Libxc 7.0.0 outputs)

- Total density below n_thr (strict `<`): all outputs zero. Measured: unpolarized
  n = 0.999e-15 gives zk = 0.
- Each channel density is first raised to max(n_σ, n_thr). A channel with n_σ ≤ n_thr then
  contributes zero. Measured, polarized: (n_α, n_β) = (0.01, 0), (0.01, 0.5e-15) and
  (0.01, 1e-15) give identical outputs, with vrho_β ≈ 2e-17 (zero up to rounding) and
  zk = −0.20047569263577633; n_β = 1.5e-15 gives vrho_β = −1.42e-5.
- Unpolarized: zk = 0 for n ∈ {1.0, 1.001, 1.999, 2.0} × 1e-15 and
  zk = −9.3068e-6 at n = 2.001e-15, i.e. the per-channel test of rule 2 with n_σ = n/2.
- Polarized with n_α = n_β: zk = 0 at n_σ = 0.9995e-15, nonzero at n_σ = 1.0005e-15.
- Libxc's ζ threshold (2.220446049250313e-16) enters this functional only through factors
  (1 ± ζ)^{4/3}, which it replaces by ζ_thr^{4/3} when 1 ± ζ ≤ ζ_thr (read in Libxc's source).
  A channel this small is already screened by rule 2 whenever n ≲ 9, and above that its
  contribution is below Float64 rounding of the other channel; no measurable effect was
  found. Not part of the PureXC rule.

### Limits

- e → 0 as n → 0 like n^{4/3}; derivatives of order k behave like n_σ^{4/3−k}, so the
  second derivative in a channel diverges as n_σ^{−2/3} when that channel empties.
  Measured: Libxc v2rho2_ββ = −3.16e9 at (n_α, n_β) = (0.01, 1.5e-15).

## 8. Known misprints and ambiguities

- None known in [PBE] for these formulas.
- The equation numbers in [D] and [B] are not checked (papers not read).

## 9. Libxc oracle

- Identifier `lda_x` (number 1), Libxc 7.0.0 through Libxc.jl, `n_spin = 1` and `2`.
- Flags: exc, vxc, fxc, enforce_fhc, dim3. Supported derivatives [0, 1, 2].
- External parameters: none.
- Thresholds reported by the API: dens_threshold 1.0e-15, zeta_threshold
  2.220446049250313e-16, sigma_threshold 1.0000000000000027e-20 (unused by an LDA),
  tau_threshold 1.0e-20 (unused).

Agreement of (E2)/(E3), evaluated in 256-bit BigFloat, with Libxc in Float64 (orders 0–2, ρ
log-spaced over [1e-6, 1e3], ζ ∈ {0, 0.25, −0.6, 0.95}): largest relative difference
2.2e-16 (zk), 1.4e-15 (vrho), 2.7e-15 (v2rho2).

Reference values (Libxc 7.0.0, Float64; last column: largest relative difference of
(E2)/(E3) in BigFloat from the Libxc values in the row):

| input | zk, vrho | rel. diff |
|---|---|---|
| n = 1.0e-6 | −0.0073855876638202235, −0.009847450218426965 | 6.1e-17 |
| n = 0.01 | −0.15911766269205826, −0.212156883589411 | 1.9e-16 |
| n = 1.0 | −0.7385587663820223, −0.9847450218426964 | 1.3e-16 |
| n = 100.0 | −3.4280861230056243, −4.570781497340832 | 1.0e-16 |
| (n_α, n_β) = (0.3, 0.1) | −0.575171388289353; −0.8305661184154147, −0.5758823822969721 | 2.0e-16 |
| (1.0, 0.001) | −0.929689099822912; −1.2407009817987997, −0.1240700981798798 | 1.7e-15 |
| (0.02, 0.02) | −0.2525835451440944; −0.33677806019212586, −0.33677806019212586 | 2.3e-16 |
