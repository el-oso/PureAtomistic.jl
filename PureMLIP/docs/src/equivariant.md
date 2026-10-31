# Equivariant math

PureMLIP provides the rotation-equivariant building blocks of MACE in pure Julia:
real spherical harmonics, real Clebsch-Gordan coefficients, real Wigner matrices, and the
conversion to the conventions used by e3nn.

## Native convention

The real spherical harmonics ``Y_{lm}`` are orthonormal on the unit sphere
(``\int Y_{lm} Y_{l'm'}\, d\Omega = \delta_{ll'}\delta_{mm'}``), use ``z`` as the polar axis,
and carry no Condon-Shortley phase. For ``m > 0``,
``Y_{l,m} = \sqrt{2} N_{lm} P_l^m(\cos\theta)\cos(m\varphi)`` and
``Y_{l,-m} = \sqrt{2} N_{lm} P_l^m(\cos\theta)\sin(m\varphi)``, with
``Y_{l0} = N_{l0} P_l(\cos\theta)``.

`spherical_harmonics!` and
`spherical_harmonics_and_gradient!`
evaluate them at the direction of `v` (the input need not be normalized). The gradient is
taken with respect to the unnormalized `v`.

## Index layout

All degrees up to `lmax` are stored in one vector of length `(lmax + 1)^2`. The degree-`l`
block is contiguous, ordered ``m = -l, \dots, l``, and ``Y_{lm}`` sits at the one-based
position

```math
l^2 + l + m + 1,
```

which is `sh_index(l, m)`; `sh_length(lmax)` gives the total length.

## Coupling coefficients and Wigner matrices

`real_clebsch_gordan(l1, l2, l3)` returns the real-basis coupling tensor of shape
`(2l1+1, 2l2+1, 2l3+1)` in the native component order, and `clebsch_gordan_complex` gives the
standard complex coefficients it is built from.

`wigner_D(l, R)` is the real Wigner matrix defined by ``Y_l(Rv) = D\,Y_l(v)``. It is computed
by exact quadrature: ``D_{ij} = \int Y_i(Ru)\,Y_j(u)\,d\Omega``, using Gauss-Legendre nodes in
``\cos\theta`` and uniform points in ``\varphi``, which integrates the degree-``2l`` integrand
exactly up to rounding. No recursion or angle parametrization is involved, so any proper
rotation matrix is accepted.

## Mapping to e3nn

e3nn orders the Cartesian axes as ``(y, z, x)`` for its ``l = 1`` components, which amounts to
the axis permutation ``R: v \mapsto (v_3, v_1, v_2)``. With `normalization="component"`:

```math
Y^{\mathrm{e3nn}}(v) = \sqrt{4\pi}\; Y^{\mathrm{native}}(v_3, v_1, v_2).
```

Equivalently ``Y^{\mathrm{e3nn}}_l = \sqrt{4\pi}\,M_l\,Y_l`` with
`M_l = e3nn_basis_change(l) = wigner_D(l, R)`. The coupling tensors convert as

```math
C^{\mathrm{e3nn}} = s\,(M_{l_1}\otimes M_{l_2}\otimes M_{l_3})\,C^{\mathrm{native}},
\qquad s = (-1)^{(l_1 + l_2 - l_3)/2},
```

where ``s`` is the sign rule that reconciles the overall phase of the two conventions.
`e3nn_spherical_harmonics!`, `e3nn_spherical_harmonics_and_gradient!` and `e3nn_wigner_3j`
implement these formulas. Coefficients of magnitude below ``10^{-13}`` are set to exact zero,
so the sparsity pattern matches e3nn's.

## Even-sum restriction

`real_clebsch_gordan` and `e3nn_wigner_3j` are defined for triples satisfying the triangle
inequality ``|l_1 - l_2| \le l_3 \le l_1 + l_2`` **and** with ``l_1 + l_2 + l_3`` even. Triples
that violate either condition throw an `ArgumentError`.

## Reference

```@docs
PureMLIP.sh_length
PureMLIP.sh_index
PureMLIP.spherical_harmonics!
PureMLIP.spherical_harmonics_and_gradient!
PureMLIP.clebsch_gordan_complex
PureMLIP.real_clebsch_gordan
PureMLIP.wigner_D
PureMLIP.e3nn_spherical_harmonics!
PureMLIP.e3nn_spherical_harmonics_and_gradient!
PureMLIP.e3nn_basis_change
PureMLIP.e3nn_wigner_3j
```
