# Input files

`parse_inp(text; source, lenient)` reads an input file in the `.inp` format into a
`ParsedInput`: the `InputFile` and the warnings of lenient reading. `molecule(inp)` builds the
`Molecule` of the result.

```julia
using QuantumEngine
p = parse_inp(read("h2.inp", String); source = "h2.inp")
mol = molecule(p.input)
```

Reading is strict by default: every input that the Rust `quantum_engine` (the oracle) accepts
silently but that is not valid is an `InputError` naming the file and the line, printed as
`input: h2.inp:9: keyword `charge` given twice (lines 2 and 9)`. With `lenient = true` such an
input is read as the oracle reads it and recorded as an `InputWarning`. An input that the oracle
itself rejects is an error in both modes.

## Lines

Each line is read in this order:

1. A UTF-8 byte-order mark on the first line, a trailing carriage return and the surrounding
   whitespace are removed, so CRLF files and tab-separated tokens read as LF, space-separated ones.
2. A line that starts with `#` is a comment. The first comment line, while no title is set, gives
   the title: the text after the `#`, stripped. A `title` keyword replaces it wherever it appears.
3. The line is split into tokens at spaces and tabs. A token that starts with `#` ends the line,
   so `charge 0 # neutral` is `charge 0`. A line with no token is skipped.

The first token is the keyword, read without regard to case. A first token that ends in `:`, or a
line that starts with `- `, is rejected as YAML: YAML input needs a `.yaml` or `.yml` extension.

## Values

- **Counts.** Each keyword takes the number of values in the table below. Another count is an
  error: `` `charge` takes 1 value, got 2 ``, `` `freeze_bond` takes 2 or 3 values, got 1 ``,
  `` `kpath` needs a value ``. A block opener that takes no value (`hubbard`, `cell`, `pbc`,
  `oniom`) and gets one is `` `hubbard` takes no value on its line ``.
- **Booleans** are exactly `true` or `false`.
- **Integers** are decimal, with an optional sign.
- **Numbers** are decimal floats, finite: `nan` and `inf` are `expected a finite number`, and a
  hexadecimal float is not a number.
- **Words** are looked up without regard to ASCII case in the keyword's vocabulary; an unknown
  word is `` unknown value `tpu` for `compute_mode`; accepted: cpu, gpu, hybrid ``.
- **Atom indices** (`freeze_*`, `fragments`, the ONIOM model atoms) are 0-based integers in the
  file and 1-based in `InputFile`.
- **Repetition.** A keyword given twice, under any of its aliases, is an error naming both lines;
  `optimize` and `run`, and `gpu` and `compute_mode`, count as one keyword. The `freeze_*` keywords
  repeat, each adding a constraint. A second `geometry` or `product_geometry` block is
  `second geometry block (first at line 4)`.

## Blocks

`geometry [unit]`, `product_geometry [unit]`, `charges [unit]`, `hubbard`, `fragments`, `cell`,
`pbc` and `oniom` open a block, which `end` closes. The unit is `angstrom` or `ang` (the default)
or `bohr` or `au`, in any case, and applies to that block alone. Inside a block, a line that is a
keyword or that is not the block's content is an error, `` `charge` inside the geometry block
opened at line 6; close the block with `end` ``, and a block still open at the end of the file is
`` the geometry block opened at line 6 is not closed by `end` ``.

| block | lines |
|---|---|
| `geometry`, `product_geometry` | `Sym x y z`, an atom (`El`, or the ghosts `El@`, `BqEl`, `Bq-El`); `Bq x y z q`, a point charge `q` |
| `charges` | `q x y z`, a point charge |
| `hubbard` | `Sym U`, each element once |
| `fragments` | fragments separated by `;`, each a list of atom indices separated by `,`; every line is added |
| `cell` | exactly 3 lines of 3 numbers: the lattice vectors in Å, whatever the geometry unit |
| `pbc` | one line of 3 Booleans |
| `oniom` | `high <method>`, `low <method>`, `model_atoms`/`model_indices <indices>`, `model_charge <Int>`, `model_multiplicity <Int>` |

The `fragments` keyword with two or more fragments on its line, `fragments 0,1,2 ; 3,4,5`, needs no
block; with none it opens one. An empty fragment or an index that is not an integer ≥ 0 is an
error.

## After the file

- An input without atoms is `no atoms: the input has no geometry block`.
- `rhf`, `rmp2`, `rccsd`, `rmp2-f12` and `rccsd-f12` request a restricted reference, and a
  multiplicity other than 1 is an error at the method line.
- The charge and multiplicity must give whole numbers of α and β electrons; the error names the
  multiplicity line.
- **Solvation** does not depend on the order of its lines. The model is the `solvation` word if
  given; otherwise `solvent` gives SMD and `epsilon` alone gives PCM. Under PCM, `epsilon`,
  `pcm_radii` and `pcm_grid` apply; under SMD, `solvent`, `pcm_grid` and `epsilon` apply, and
  `epsilon` replaces the dielectric constant of the solvent (`custom_epsilon == true`). `solvent`
  with `solvation pcm`, `pcm_radii` under SMD, `pcm_radii` or `pcm_grid` without a model, and any
  solvation option with `solvation none` are errors.
- `f12_gamma` and `f12_model` need `f12 true` or an F12 method (`mp2-f12`, `ccsd-f12`,
  `ccsd(t)-f12` and their spellings).

## Deviations from the oracle

Besides rejecting what the oracle accepts silently, the parser reads these inputs differently in
both modes, because the oracle misreads them:

| input | QuantumEngine | oracle |
|---|---|---|
| the text `cell` or `pbc` anywhere, as in `title supercell pbc test` | nothing; only a `cell` or `pbc` block sets the lattice or the periodicity | scans the whole file for the first `cell` and the first `pbc` and reads the lines after them (`parser.rs:622-657`): here the first three atom lines as lattice vectors and the first atom line as three `false` flags |
| a `fragments` block of several lines | every line is added | each line replaces the fragments of the line before |
| a block without a unit after `geometry bohr` | Å, the default | the unit of the last geometry block |
| a `charges` block | takes an optional unit, Å by default | takes no unit and uses the unit of the last geometry block |
| `title "a title"` | `a title`: one pair of surrounding double quotes is removed | the quotes are kept |
| a geometry block without `end` | an error | closed by the next block opener or the end of the file |
| the `pbc` values `1`, `0`, `T`, `F` | an error | `true`, `1` and `T` are true, every other token false (`parser.rs:649-653`) |

## Keywords

The values are those of the field of `InputFile` named in the last column. Notation: `Int`,
`Float`, `Bool`, `word` (one token, kept as written unless the column says otherwise),
`1+` (one or more), `[x]` (optional).

| keyword (aliases) | values | field |
|---|---|---|
| `title` | `1+ words`, joined by single spaces | `title` |
| `run`, `run_mode`; `optimize` | `word`; none | `run_mode` |
| `charge`; `multiplicity` | `Int`; `Int` | `charge`; `multiplicity` |
| `basis` | `word`, lowercased | `basis` |
| `method` | `word`; `libxc_gga x c`, `libxc_mgga x c`: `word Int Int`; `libxc_hgga id exx`, `libxc_hmgga id exx`: `word Int Float`; `casscf(e,o)` | `method` |
| `scf_convergence`; `max_iterations` | `Float`; `Int` | `scf_convergence`; `max_iterations` |
| `diis_method` | `cdiis` or `ediis` | `diis_method` |
| `conv_tol_density` | `Float` | `conv_tol_density` |
| `diis_reset`, `diis_reset_threshold` | `Float` | `diis_reset_threshold` |
| `damping`; `smearing`, `smear`; `level_shift` | `Float` | `damping`; `smearing`; `level_shift` |
| `grid`, `grid_level`, `xc_grid` | `coarse`, `standard`, `medium`, `fine`, `ultrafine`, `superfine` or `0`–`4` | `grid_level` |
| `auto_stabilize`, `autostabilize`; `robust_convergence`, `robust` | `Bool` | `auto_stabilize`; `robust_convergence` |
| `field`, `electric_field` | `Float Float Float` | `electric_field` |
| `density_fitting`, `ri_j`; `ri_k`; `aux_basis` | `Bool`; `Bool`; `word` | `use_ri_j`; `use_ri_k`; `aux_basis` |
| `gpu`; `compute_mode` | `Bool`; `cpu`, `gpu` or `hybrid` | `compute_mode` |
| `checkpoint`; `load_checkpoint`, `restart` | `word`; `Bool` | `checkpoint`; `load_checkpoint` |
| `ts_target_mode`; `ts_trust_radius` | `Int`; `Float` | `ts_target_mode`; `ts_trust_radius` |
| `neb_k`; `neb_images`; `climbing_image` | `Float`; `Int`; `Bool` | `neb_k`; `neb_images`; `climbing_image` |
| `temperature`, `temp`; `pressure`, `press`; `symmetry_number`, `sigma` | `Float` | `temperature`; `pressure`; `symmetry_number` |
| `dispersion`; `zora`, `zora_relativistic`; `nbo` | `Bool` | `dispersion`; `zora`; `nbo` |
| `f12`, `explicit_correlation`; `f12_gamma`, `gamma`, `gamma_f12`; `f12_model`, `f12_approximation` | `Bool`; `Float`; `word` | `use_f12`; `f12_gamma`; `f12_model` |
| `admm`; `admm_basis` | `Bool`; `word` | `admm`; `admm_basis` |
| `ecp`, `use_ecp`; `ecp_name` | `true`, `false` or a name (which sets `use_ecp`); `word` (sets `use_ecp`) | `use_ecp`, `ecp_name` |
| `vdw_kernel` | `d2`, `grimme-d2`, `d3`, `grimme-d3`, `d3bj`, `b97x-d`, `vv10`, `nl-vdw` | `vdw_kernel` |
| `solvation`; `solvent`; `epsilon`, `dielectric`; `pcm_radii`, `radii_scaling`; `pcm_grid`, `smd_grid` | `pcm`, `true`, `smd`, `none`, `false`; a solvent; `Float`; `Float`; `Int` | `solvation` |
| `output`; `verbosity` | `word`; `Int` | `output`; `verbosity` |
| `export_molden`, `export_cube`, `export_wfx` | `Bool` | `export_molden`, `export_cube`, `export_wfx` |
| `population`, `pop_analysis` | `1+` of `mulliken`, `lowdin` | `population` |
| `pw_cutoff`, `cutoff` | `Float` | `pw_cutoff` |
| `kpoints`, `k_points`, `kgrid`; `dos_k_grid`, `dos_kgrid`; `dos_sigma` | `Int Int Int`; `Int Int Int`; `Float` (eV) | `kpoints`; `dos_k_grid`; `dos_sigma` |
| `kpath`, `k_path` | `1+ words` | `kpath` |
| `oniom_high`, `oniom_low`; `oniom_model`; `oniom_model_charge`, `oniom_model_multiplicity` | method values; `1+` indices; `Int` | `oniom` |
| `freeze_atom`, `freeze` | `1+` indices | `constraints` |
| `freeze_bond`; `freeze_angle`; `freeze_dihedral`, `freeze_torsion` | 2, 3 or 4 indices and `[Float]`: a bond target in bohr, an angle in degrees (stored in radians) | `constraints` |
| `irc_step_size`, `irc_step`; `irc_max_points`, `irc_points`; `irc_direction` | `Float`; `Int`; `forward`, `fwd`, `backward`, `bwd`, `back`, `both` | `irc_step_size`; `irc_max_points`; `irc_direction` |
| `tddft_states`, `num_tddft_states`, `nstates`; `tddft_method` | `Int`; `tda`, `rpa`, `spin_flip`, `spinflip`, `sf`, `sf-tddft` | `tddft_states`; `tddft_method` |
| `compute_soc`, `soc`; `tddft_triplets`, `triplets` | `Bool` | `compute_soc`; `tddft_triplets` |
| `dmft`, `use_dmft`; `dmft_temp`; `dmft_n_freq`, `dmft_matsubara`; `dmft_n_bath`, `dmft_bath` | `Bool`; `Float`; `Int`; `Int` | `use_dmft`; `dmft_temperature`; `dmft_n_freq`; `dmft_n_bath` |
| `geometry`, `product_geometry`, `charges` | `[unit]`, then the block | `atoms`, `product_atoms`, `point_charges` |
| `hubbard`; `cell`; `pbc`; `oniom` | none, then the block | `hubbard`; `lattice`; `pbc`; `oniom` |
| `fragments` | fragments on the line, or none and a block | `fragments` |

`accepted_keywords()` lists every keyword and alias.

```@docs
QuantumEngine.parse_inp
QuantumEngine.molecule
QuantumEngine.Input.accepted_keywords
```
