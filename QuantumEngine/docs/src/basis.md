# Basis sets

`QuantumEngine.Basis` reads Gaussian basis sets in the Gaussian94 `.gbs` format and builds the
molecular basis of a molecule with [`build_basis`](@ref QuantumEngine.Basis.build_basis). The package ships 72 basis files from
the Basis Set Exchange in `data/basis_sets/`; their source, license and citation are in that
directory's `README.md`.

## Cartesian functions only

Every shell is expanded in Cartesian Gaussians `x^lx y^ly z^lz exp(-α r²)`; no spherical (pure)
functions are formed. A d shell therefore carries 6 functions, an f shell 10 and a g shell 15
(`ncart(l) = (l + 1)(l + 2)/2`), where a code that uses spherical functions has 5, 7 and 9. The
number of basis functions, and so the energies, differ from those of a code that uses 5d (or 7f)
functions with the same basis name; compare only with a calculation that also uses Cartesian
functions.

Each Cartesian component has its own normalization, [`primitive_norm`](@ref QuantumEngine.Basis.primitive_norm)`(α, lx, ly, lz)`, so
`xy` and `xx` differ. The contraction of a shell is renormalized with
[`contraction_renorm`](@ref QuantumEngine.Basis.contraction_renorm), and the stored coefficients of a [`Shell`](@ref QuantumEngine.Basis.Shell) include that
factor but not the per-component norm, which the consumer applies.

## Component order

[`cartesian_component`](@ref QuantumEngine.Basis.cartesian_component)`(l, i)` gives the exponents `(lx, ly, lz)` of the `i`-th function of
a shell, in the oracle's order, which is the order of the Molden format for `l ≤ 4`. No rule
generates it (l = 3 lists `xyy` before `xxy`), so it is a table:

| l | shell | components, in order |
|---|---|---|
| 0 | s | `1` |
| 1 | p | `x y z` |
| 2 | d | `xx yy zz xy xz yz` |
| 3 | f | `xxx yyy zzz xyy xxy xxz xzz yzz yyz xyz` |
| 4 | g | `xxxx yyyy zzzz xxxy xxxz xyyy yyyz xzzz yzzz xxyy xxzz yyzz xxyz xyyz xyzz` |
| 5 | h | `xxxxx yyyyy zzzzz xxxxy xxxxz xyyyy yyyyz xzzzz yzzzz xxxyy xxxzz xxyyy yyyzz xxzzz yyzzz xxxyz xyyyz xyzzz xxyyz xxyzz xyyzz` |

## The l limit

The component table stops at l = 5 (h). [`parse_gbs`](@ref QuantumEngine.Basis.parse_gbs) reads I shells (l = 6), which seven
shipped files hold for some elements, so every shipped file parses. [`build_basis`](@ref QuantumEngine.Basis.build_basis) throws
only when it places an I shell on an atom:

```
basis: <file>:<line>: I shell (l = 6) for C; this build supports l ≤ 5
```

So water in `def2-universal-jkfit` builds (its I shells belong to transition metals and heavier
elements), while carbon in `cc-pv6z` throws.

## Basis names

A basis name is normalized by [`normalize_basis_name`](@ref QuantumEngine.Basis.normalize_basis_name): stripped, lowercased, and mapped
through the alias table, so `6-31G*`, `631g(d)` and `6-31g(d)` all name `6-31g_star`. A name not in
the table is kept as it is. [`candidate_stems`](@ref QuantumEngine.Basis.candidate_stems) lists the file stems tried, in order: the
normalized name first, then variants with `*`, `**`, `+`, `++`, `(d)` and `(d,p)` written out
(`_star`, `_star_star`, `_dp`, `plus`, `plusplus`). A name containing `/` or `\` is rejected on
every platform; put the file in a directory on the search path instead.

Four aliases differ from the oracle's table:

| names | file stem here | the oracle's file |
|---|---|---|
| `6-31G**`, `6-31g(d,p)` and their spellings | `6-31g_star_star` | `6-31g_dp`, whose O d exponent differs |
| `6-311++G**`, `6-311++g(d,p)` and their spellings | `6-311plusplusg_star_star` | `6-311plusplusg_dp`, with one shell fewer for C, N, O, F and Ne |
| `6-311++G(2d,2p)` and its spellings | `6-311plusplusg(2d,2p)` | `6-311++g_2d2p`, which holds only H, C and N, identical to their blocks here |
| `def2-SVP-JKFIT` | `def2-universal-jkfit` | the oracle's own placeholder file (H, C, N and O only) |

The oracle's three files have no Basis Set Exchange header and are not shipped; the aliases name
the Basis Set Exchange files of the same basis. `def2-SVP-JKFIT` is the oracle's default auxiliary
basis; here it is read as the universal JK fit, and `build_basis` with a `logger` writes the note
`basis <name>: the oracle's def2-SVP-JKFIT is a placeholder; using def2-universal-jkfit`.

## Search order and shadowing

[`locate_basis_file`](@ref QuantumEngine.Basis.locate_basis_file) tries, for each candidate stem in turn, each directory of
[`basis_search_paths`](@ref QuantumEngine.Basis.basis_search_paths) in turn, and returns the absolute path of the first `<dir>/<stem>.gbs`
that exists. The directories are, in order:

1. `basis_sets`, `../basis_sets` and `../../basis_sets`, relative to the working directory at the
   time of the lookup;
2. `QUANTUM_ENGINE_BASIS_DIR`, when it is set and not empty;
3. when an executable directory is given, `basis_sets`, `../basis_sets` and `../../basis_sets`
   under it;
4. the package's `data/basis_sets`.

The first match wins, so a file in a `basis_sets` directory under the working directory, or in
`QUANTUM_ENGINE_BASIS_DIR`, shadows the shipped file of the same stem. Because the stems are the
outer loop, an earlier stem found in a later directory wins over a later stem found in an earlier
one. A name with no file throws a `BasisError` that lists the stems and the directories tried;
there are no other fallbacks.

## Building the molecular basis

[`build_basis`](@ref QuantumEngine.Basis.build_basis)`(mol, name)` reads the whole file once with [`parse_gbs`](@ref QuantumEngine.Basis.parse_gbs) and gives
each atom, in order, the shells of its element in file order. A ghost atom gets the shells of the
element it stands for. The AO offsets run from 0 in shell order; [`ao_range`](@ref QuantumEngine.Basis.ao_range) gives the AOs
of a shell and [`shell_pairs`](@ref QuantumEngine.Basis.shell_pairs) iterates the shell pairs. `build_basis(mol, name, Float32)`
builds the basis in another element type.

ECP sections of a `.gbs` file are read into [`ECPPotential`](@ref QuantumEngine.Basis.ECPPotential); ECP integrals are not
implemented.

The functions and types are listed in the [API](api.md), under "Basis sets".
