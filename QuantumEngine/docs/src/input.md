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
- **Numbers** are decimal floats, finite: `nan`, `inf` and a value beyond the `Float64` range
  (`1e500`) are `expected a finite number`, a value below it (`1e-400`) reads as 0.0, and a
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

## Lenient reading

With `lenient = true`, each input of the tables below is read as the oracle reads it and recorded
as an `InputWarning` whose `case` names it; in strict mode each is an `InputError`. The tables are
the complete record of what lenient reading accepts. An input the oracle itself rejects (marked
"error") is an error in both modes.

### A bad or missing value

"Bad" is a token that does not parse as the keyword's type or a word outside its vocabulary;
"missing" is no token. "Unchanged" keeps the default or the value of an earlier line. A token that
parses as a non-finite number (`nan`, `inf`, `1e500`) is `:nonfinite` and keeps the parsed value.

| keyword (aliases) | bad | missing | case (bad / missing) |
|---|---|---|---|
| `run`, `run_mode` | error | unchanged | — / `:missing_value` |
| `title` | — | unchanged | `:missing_value` |
| `charge`, `multiplicity`, `scf_convergence`, `max_iterations`, `ts_target_mode`, `ts_trust_radius`, `neb_k`, `neb_images`, `temperature`, `pressure`, `symmetry_number` and their aliases | error | error | — |
| `basis` | — | error | — |
| `method`, `oniom_high`, `oniom_low`, and `high`, `low` in an oniom block | error | error | — |
| a `geometry` or `product_geometry` unit | error | — | — |
| `climbing_image`, `density_fitting`/`ri_j`, `ri_k`, `gpu`, `load_checkpoint`/`restart`, `dispersion`, `zora`, `f12`, `nbo`, `auto_stabilize`, `robust_convergence`, `dmft`, `admm` | true (`gpu`: GPU) | true | `:bool_fallback` |
| `export_molden`, `export_cube`, `export_wfx` | false | false | `:bool_fallback` |
| `compute_soc`/`soc`, `tddft_triplets`/`triplets` | `1` gives true, any other token false | true | `:bool_fallback` |
| `ecp`/`use_ecp` | a name, valid in both modes | `use_ecp = true` | `:bool_fallback` |
| `ecp_name` | — | `use_ecp = true`, `ecp_name = nothing` | `:missing_value` |
| `compute_mode` | unchanged | unchanged | `:unknown_value` / `:missing_value` |
| `checkpoint`, `aux_basis`, `admm_basis` | — | `nothing` | `:missing_value` |
| `f12_gamma` | 1.0 | 1.0 | `:number_fallback` / `:missing_value` |
| `f12_model` | — | `"F12b"` | `:missing_value` |
| `damping`, `smearing`, `level_shift` | 0.0 | 0.0 | `:number_fallback` / `:missing_value` |
| `diis_reset`, `dos_sigma` | 0.1 | 0.1 | `:number_fallback` / `:missing_value` |
| `verbosity` | 1 | 1 | `:number_fallback` / `:missing_value` |
| `grid` | an unsigned integer as written, any other token 1 | 1 | `:unknown_value` / `:missing_value` |
| `conv_tol_density`, `pw_cutoff` | `nothing` | `nothing` | `:number_fallback` / `:missing_value` |
| `field` | the entry is 0.0 | unchanged, also with fewer than 3 values | `:bad_list_entry` / `:too_few_values` |
| `kpoints` | the entry is 1 | unchanged, also with fewer than 3 values | `:bad_list_entry` / `:too_few_values` |
| `dos_k_grid` | the entry is 10 | unchanged, also with fewer than 3 values | `:bad_list_entry` / `:too_few_values` |
| `solvation` | unchanged | unchanged | `:unknown_value` / `:missing_value` |
| `solvent` | unchanged | `water`, under SMD | `:unknown_value` / `:missing_value` |
| `epsilon` | 78.39 | 78.39 | `:number_fallback` / `:missing_value` |
| `pcm_radii` | 1.2, under PCM only | 1.2, likewise | `:number_fallback` / `:missing_value` |
| `pcm_grid` | 110 | 110 | `:number_fallback` / `:missing_value` |
| `output` | — | `""` | `:missing_value` |
| `vdw_kernel` | `nothing` | `nothing` | `:unknown_value` / `:missing_value` |
| `kpath` | — | an empty list | `:missing_value` |
| `population` | the default list | an empty list | `:unknown_value` / `:missing_value` |
| `oniom_model`, `model_atoms`, `freeze_atom` | the entry is dropped | nothing added | `:bad_list_entry` / `:missing_value` |
| `freeze_bond`, `freeze_angle`, `freeze_dihedral` | a bad index drops the constraint; a bad target is `nothing` | the constraint is dropped, also with too few indices | `:bad_list_entry` / `:too_few_values` |
| `irc_step_size`, `irc_max_points`, `tddft_states`, `dmft_temp`, `dmft_n_freq`, `dmft_n_bath` | unchanged | unchanged | `:number_fallback` / `:missing_value` |
| `irc_direction` | `Both` | unchanged | `:unknown_value` / `:missing_value` |
| `tddft_method` | `TDA` | unchanged | `:unknown_value` / `:missing_value` |
| `diis_method` | `CDIIS` | unchanged | `:unknown_value` / `:missing_value` |
| `model_charge`, `model_multiplicity`, `oniom_model_charge`, `oniom_model_multiplicity` | `nothing` | `nothing` | `:number_fallback` / `:missing_value` |

### Other cases

| case | input | lenient value |
|---|---|---|
| `:unknown_keyword` | `use_diis true`; `end` outside a block | ignored |
| `:dropped_block_line` | a block line that is not the block's content, as `END_GEOMETRY` | dropped |
| `:block_keyword` | a keyword inside a block, as `charge 1` | read as the keyword; the block stays open |
| `:unclosed_block` | a block without `end`, at the end of the file or before a block opener | closed there, its lines kept; an opener inside a block is also `:block_keyword` |
| `:hubbard_inline_argument` | `hubbard O:5.0` | the value is ignored and the block opens; a following line of two tokens that is not a keyword is a U line, so `use_diis true` fails with `Invalid U value: true` |
| `:fragment_index` | `fragments 0,x;1,2` | the index is dropped |
| `:nonfinite` | `scf_convergence nan`, `damping inf` | the parsed value |
| `:extra_values` | `charge 0 1`, `optimize now` | the tokens after the count are ignored |
| `:duplicate_keyword` | `charge 0` … `charge 1`; an element twice in `hubbard` | the last value; `hubbard` tells symbols apart by case |
| `:duplicate_block` | a second `geometry` or `product_geometry` block | its atoms are appended |
| `:order_dependent` | solvation lines whose result depends on their order: `pcm_radii 1.1` or `epsilon 4.0` before `solvation pcm`, `solvent dcm` before `solvation smd`, `solvent dcm` with `solvation pcm` | the lines read in file order: `solvation` resets the options, `solvent` makes the model SMD, `epsilon` without a model makes it PCM, `pcm_radii` applies under PCM only |
| `:option_without_model` | `pcm_radii 1.1` without a model | ignored |
| `:casscf_shape` | `method casscf(4)` | CASSCF(2, 2); a word of two parts reads them as the counts (`casscf4,6`), which must be unsigned integers |
| `:ghost_unchecked` | `Foo@ 0 0 0`, `Bq 0 0 0` | a ghost of Z = 0, the symbol as written, mass 0; `basis_element` of it throws |
| `:rhf_open_shell` | `method rhf`, `multiplicity 3` | `AutoReference` |
| `:f12_option_alone` | `f12_gamma 0.9` without `f12 true` | the value is kept; F12 stays off |
| `:yaml_in_inp` | YAML text in a `.inp` file | no line is read, so the input has no atoms: an error in both modes |
| `:bool_fallback` | the `pbc` line `1 T F` | `true`, `1` and `T` are true, any other token false |

The oracle keeps a separate flag for each block, and lenient reading keeps one open block, so it
does not reproduce these readings of an unclosed block: an unclosed `hubbard` block stays open in
the oracle under a later `geometry`, `product_geometry`, `charges`, `cell` or `fragments` line,
whose atom lines it then reads as U lines; `oniom` and `charges` do not end an open block in the
oracle; and `pbc` is not a keyword of the oracle, which finds the `pbc` line by its text scan.

### The molecule files

| case | input | lenient value |
|---|---|---|
| `:xyz_count` | an xyz frame of count 3 with 2 atom lines | the 2 atoms |
| `:xyz_frames` | an xyz file of several frames given as an input geometry | the first frame |
| `:unknown_element` | `Xx 0 0 0` in an xyz, pdb or cif file | Z = 1, the symbol as written, not a ghost, the mass of ¹H |
| `:pdb_short_line` | an ATOM record shorter than 54 columns | skipped |
| `:pdb_element_columns` | no element in columns 77–78 | the first character of the atom name for a record shorter than 78 columns; an empty symbol (Z = 1) for a wider record whose columns 77–78 are blank |
| `:cif_row` | a row with fewer values than its loop's tags | skipped |
| `:cif_coordinate` | a coordinate that is not a number | 0.0 |
| `:nonfinite` | a coordinate `nan`, `inf` or `1e400` | the parsed value |
| `:cif_columns` | an `_atom_site` loop without `_atom_site_type_symbol` or one of `_atom_site_Cartn_x`, `_y`, `_z` | the loop's rows skipped |

Lenient reading does not reproduce the oracle for an xyz frame with more atom lines than its
count: the oracle reads one frame and ignores the lines after it, while `read_xyz` reads the next
line as the count of the next frame, an error in both modes when it is not one.

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
