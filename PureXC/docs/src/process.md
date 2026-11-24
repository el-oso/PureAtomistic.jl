# Two-room process

PureXC is MIT-licensed; Libxc, the reference its tests compare with, is MPL-2.0. So that no Libxc
code reaches PureXC, every functional passes through two separate rooms, and the record of both
is kept in the repository under `PureXC/specs/`.

**Room 1, the spec writer**, reads the published papers first, then Libxc's public documentation,
its public API and its outputs, and opens Libxc's source only for what those leave open (which
variant of a formula, a misprint, a constant the papers round, how a threshold is applied). Every
fact taken from Libxc is marked "checked against Libxc". A spec holds facts only, equations and
numbers, never code. There is one spec per functional, `PureXC/specs/<family>/<libxc_id>.md`:

1. identifier, family, kind, spin cases;
2. references (DOI, equation numbers);
3. variables and units;
4. equations, each with its paper and equation number;
5. constants, each with its value, its source, and "paper" or "checked against Libxc";
6. building blocks (other specs);
7. domain, thresholds and limits, with the functional's PureXC threshold rule;
8. known misprints or ambiguities, and how each is resolved;
9. the Libxc oracle identifier and the parameter values its public API reports.

**Room 2, the implementer and the reviewer**, read the specs, the papers and the repository, and
never Libxc's C or Maple source, any file of the Julia packages `Libxc` and `Libxc_jll`, or any page
that shows Libxc source. The reviewer holds the same restriction as the implementer and checks the
code against the spec.

The prompts of the three roles are `PureXC/specs/prompts/spec_writer.md`, `implementer.md` and
`reviewer.md`. The specs of the shipped functionals:

| functional | spec |
|---|---|
| `lda_x` | `PureXC/specs/lda/lda_x.md` |
| `lda_c_pw` | `PureXC/specs/lda/lda_c_pw.md` |
| `lda_c_pw_mod` | `PureXC/specs/lda/lda_c_pw_mod.md` |
| `pbe_x` | `PureXC/specs/gga/gga_x_pbe.md` |
| `pbe_c` | `PureXC/specs/gga/gga_c_pbe.md` |

Libxc itself is called only through Libxc.jl's `Libxc.Functional`, `evaluate` and `evaluate!`, and
only in the comparison tests (`PureXC/test/libxc/`) and the speed bench (`PureXC/bench/`), whose
environments are the only ones that hold it. The item "Libxc is not a dependency" checks that
neither the package nor its main test environment names Libxc or Libxc_jll.
