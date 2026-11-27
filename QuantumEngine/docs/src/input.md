# Input files

`parse_input(path; lenient)` reads an input file with the reader its extension names: `.xyz`, `.pdb`
and `.cif` are molecule files, read into an `InputFile` holding only their atoms; `.json` goes to
`parse_json`, `.yaml` and `.yml` to `parse_yaml` (with YAML.jl loaded); any other extension is the
`.inp` format. An xyz file given as an input must hold one frame (lenient reading takes the first of
several, with an `:xyz_frames` warning) and at least one atom (lenient reading keeps a frame of
none, with an `:xyz_no_atoms` warning). A path that names a directory is an `InputError` (`io: the path is
a directory`) whatever its extension. A path that ends in a separator (`d.xyz/`) has no extension, so
it is dispatched as `.inp`, which rejects it as a directory when it is one; the oracle dispatches it as
xyz.

```@docs
QuantumEngine.parse_input
QuantumEngine.Input.input_format
```

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

The text must be valid UTF-8: otherwise it is, in both modes and for every format `parse_input`
reads, `the file is not valid UTF-8`, naming the line of the first invalid character. Each line is
read in this order:

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
- **Integers** are decimal, with an optional sign. The counts that the oracle stores unsigned
  (`multiplicity`, `max_iterations`, `ts_target_mode`, `neb_images`, `verbosity`, `pcm_grid`,
  `kpoints`, `dos_k_grid`, `irc_max_points`, `tddft_states`, `dmft_n_freq`, `dmft_n_bath` and the
  ONIOM model multiplicity) take no `-` sign: `-1` is `expected a non-negative integer`, and in
  lenient reading takes the keyword's fallback below.
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
opened at line 6; close the block with `end` ``; a line whose first word is not a keyword either
says so and lists the keywords, as an unknown keyword does (`` `END_GEOMETRY` inside the geometry
block opened at line 6 is not a keyword; the accepted keywords are … ``); and a block still open at
the end of the file is
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
| `model_charge`, `model_multiplicity`, `oniom_model_charge`, `oniom_model_multiplicity` | `nothing`; a charge outside the `Int32` range also `nothing` | `nothing` | `:number_fallback` / `:missing_value` |

### Other cases

| case | input | lenient value |
|---|---|---|
| `:unknown_keyword` | `use_diis true`; `end` outside a block | ignored |
| `:dropped_block_line` | a block line that is not the block's content, as `END_GEOMETRY`; in a `cell` block a line of other than 3 tokens whose first is not a number, in a `pbc` block a line of fewer than 3 tokens whose first is not `true`, `false`, `1`, `0`, `T` or `F`, in a `fragments` block a line that does not start with a digit, sign, `,` or `;` | dropped |
| `:block_keyword` | a keyword inside a block, as `charge 1` | read as the keyword; the block stays open |
| `:unclosed_block` | a block without `end`, at the end of the file or before a block opener | closed there, its lines kept; an opener inside a block is also `:block_keyword` |
| `:hubbard_inline_argument` | `hubbard O:5.0` | the value is ignored and the block opens; a following line of two tokens that is not a keyword is a U line, so `use_diis true` fails with `Invalid U value: true` |
| `:fragment_index` | `fragments 0,x;1,2` | the index is dropped |
| `:atom_index_range` | `freeze_atom 99`, `fragments 0,1;5,6` or `oniom_model 0 99` for an input of 2 atoms; in structured input a `constraints` or `oniom.model_atoms` index | the index as written |
| `:repeated_atom` | `freeze_bond 0 0 1.4`, `freeze_angle 0 1 0`; an atom in two fragments, `fragments 0;0,1` | as written |
| `:hubbard_element` | `Xx 4.0` in a `hubbard` block; in structured input a key of `hubbard`, `u_map` or `j_map` | the symbol as written; a bare `Bq` is an error in both modes |
| `:nonfinite` | `scf_convergence nan`, `damping inf` | the parsed value |
| `:extra_values` | `charge 0 1`, `optimize now` | the tokens after the count are ignored |
| `:duplicate_keyword` | `charge 0` … `charge 1`; an element twice in `hubbard` | the last value; `hubbard` tells symbols apart by case |
| `:duplicate_block` | a second `geometry` or `product_geometry` block | its atoms are appended |
| `:order_dependent` | solvation lines whose result depends on their order: `pcm_radii 1.1` or `epsilon 4.0` before `solvation pcm`, `solvent dcm` before `solvation smd`, `solvent dcm` with `solvation pcm` | the lines read in file order: `solvation` resets the options, `solvent` makes the model SMD, `epsilon` without a model makes it PCM, `pcm_radii` applies under PCM only; it is given beside another warning on those lines (`epsilon 4.0 x` before `solvation pcm` gives `:extra_values` and `:order_dependent`) |
| `:option_without_model` | `pcm_radii 1.1` without a model | ignored |
| `:casscf_shape` | `method casscf(4)` | CASSCF(2, 2); a word of two parts reads them as the counts (`casscf4,6`), which must be unsigned integers |
| `:ghost_unchecked` | `Foo@ 0 0 0`, `Bq 0 0 0` | a ghost of Z = 0, the symbol as written, mass 0; `basis_element` of it throws. Only the oracle's ghost forms (a prefix `Bq` or `bq`, or a suffix `@`, case-sensitive) are read so: `BQx` is an unknown element in both modes |
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
| `:xyz_no_atoms` | an xyz file whose frame has no atoms, given as an input geometry | no atoms |
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
| `charge`; `multiplicity` | `Int` in the `Int32` range (the oracle's `i32`; outside it an error in strict mode and `x % Int32` with an `:int32_truncated` warning in lenient mode); `Int` | `charge`; `multiplicity` |
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
QuantumEngine.Input._finalize_input
```

## Structured input (JSON, dictionaries, YAML)

`parse_json(text; source, lenient)` reads a JSON input file: `json_node(text)` reads the text with
JSON.jl into a tree of `Node`s, and `parse_structured(root; source, lenient)` reads the tree into
a `ParsedInput`. Any reader that builds a `Node` tree reaches the same schema. A JSON integer is
an integer and a number with a fraction or an exponent is a number; a number beyond the `Float64`
range is ±Inf. A key given twice in one object is an error in both modes, as the oracle's reader
rejects it, and so is an integer outside the `Int64` range; both name the key path
(`input: a.json: scf: duplicate key "a"`). Text that is not JSON is, in both modes,
`not valid JSON (<JSON.jl's message>)`. Each of these is an `InputError` of line 0.

```julia
using QuantumEngine
p = parse_json(read("fe2_dmft.json", String); source = "fe2_dmft.json")
```

Keys are case-sensitive. Messages name the key path, with 1-based list positions, and line 0:
`input: a.json: geometry[2].position[3]: expected a number, got a string`. The checks shared with
`.inp` (atoms given, every atom index of a constraint or of the ONIOM model atoms within the atoms
and no atom twice in one constraint, a restricted reference only for a singlet, whole numbers of
α and β electrons) run last, with the `.inp` messages.

### Schema

"Fallback" is a top-level key read when the `scf` key is absent; giving both is an error in strict
mode. A value of another type than the one listed is an error. An integer where a number is
listed is converted.

| key path | value | field |
|---|---|---|
| `title`; `charge`; `multiplicity` | string; integer in the `Int32` range (outside it an error in strict mode, wrapped with an `:int32_truncated` warning in lenient mode); integer ≥ 0 | `title`; `charge`; `multiplicity` |
| `basis` | string, lowercased | `basis` |
| `method` | a method word of `.inp` without the Libxc forms, in any case | `method` |
| `mode`, else `task.mode` | a run-mode word of `.inp` | `run_mode` |
| `dispersion`; `zora` | Boolean | `dispersion`; `zora` |
| `hubbard` | a map of element symbol to number | `hubbard` |
| `vdw_kernel` | a kernel word of `.inp` | `vdw_kernel` |
| `geometry` or `atoms` (not both) | a list of `{symbol, position: [x, y, z]}` or `{symbol, x, y, z}` | `atoms` |
| `unit` | `angstrom` or `ang` (the default), `bohr` or `au`, in any case | the geometry unit |
| `irc_step_size`, `irc_step`; `irc_max_points`, `irc_points`; `irc_direction` | number; integer ≥ 0; a direction word | `irc_step_size`; `irc_max_points`; `irc_direction` |
| `constraints` | a list of `{freeze_atom: i}`, `{freeze_bond: [i, j], target}`, `{freeze_angle: [i, j, k], target}`, 0-based; a bond target in bohr, an angle target in degrees | `constraints` (1-based, radians) |
| `f12` | Boolean, or a map of `gamma` or `f12_gamma` (number) and `model` or `approximation` (string), which turns F12 on | `use_f12`, `f12_gamma`, `f12_model` |
| `f12_gamma`, `gamma`; `f12_model` | number, which needs F12; string | `f12_gamma`; `f12_model` |
| `scf.max_iter` (fallback `max_iterations`); `scf.conv_energy` (fallback `scf_convergence`) | integer ≥ 0; number | `max_iterations`; `scf_convergence` |
| `scf.use_ri_j`, `scf.use_ri_k`; `scf.aux_basis` | Boolean; string | `use_ri_j`, `use_ri_k`; `aux_basis` |
| `scf.gpu` (fallback `gpu`); `scf.compute_mode` (fallback `compute_mode`) | Boolean, `true` selects the GPU; a compute-mode word; not both | `compute_mode` |
| `scf.diis_method`; `scf.verbosity`; `scf.grid` | `cdiis` or `ediis`; integer ≥ 0; integer ≥ 0 | `diis_method`; `verbosity`; `grid_level` |
| `scf.checkpoint`; `scf.load_checkpoint` | string; Boolean | `checkpoint`; `load_checkpoint` |
| `scf.damping`, `scf.smearing`, `scf.level_shift` | number | `damping`, `smearing`, `level_shift` |
| `scf.auto_stabilize`, `scf.robust_convergence` | Boolean | `auto_stabilize`, `robust_convergence` |
| `scf.field` | 3 numbers | `electric_field` |
| `scf.pw_cutoff` (fallback `pw_cutoff`) | number | `pw_cutoff` |
| `scf.lattice` (fallback `lattice`) | 3 rows of 3 numbers, in Å | `lattice` |
| `scf.pbc` (fallback `pbc`) | 3 Booleans | `pbc` |
| `scf.admm`; `scf.admm_basis`; `scf.population` | Boolean; string; a list of `mulliken`, `lowdin` | `admm`; `admm_basis`; `population` |
| `scf.kpoints` (fallbacks `k_grid`, then `kpoints`) | 3 integers ≥ 0 | `kpoints` |
| `scf.kpath` (fallback `k_path`) | a list of strings | `kpath` |
| `scf.use_dmft` (fallback `use_dmft`) | Boolean | `use_dmft` |
| `scf.dmft_options`, else `dmft`, else `scf.dmft` | a map of `temperature` (number), `n_matsubara`, `n_bath` (integers ≥ 0), `u_map`, `j_map` (maps of element symbol to number) | `dmft_temperature`, `dmft_n_freq`, `dmft_n_bath`, `dmft_u`, `dmft_j` |
| `solvation.model`, else `solvation.pcm`, else `solvation.smd` (one of them) | `pcm` or `smd`; Boolean; Boolean | `solvation.model` |
| `solvation` under SMD: `solvent`, `grid_order`, `probe_radius` | a solvent word; integer ≥ 0; number in Å | `solvation` |
| `solvation` under PCM: `epsilon`, `radii_scaling`, `grid_order`, `surface` | number; number; integer ≥ 0; `vdw`, `sas` or `smd` | `solvation` |
| `oniom`: `high`, `low`, `model_atoms`, `model_charge`, `model_multiplicity` | method words; 0-based indices; integer in the `Int32` range (outside it as for `charge`); integer ≥ 0 | `oniom` (any `oniom` value asks for ONIOM) |
| `tddft`: `num_states`, `method`, `soc`, `triplets` | integer ≥ 0; a TDDFT method word; Boolean; Boolean | `tddft_states`, `tddft_method`, `compute_soc`, `tddft_triplets` |
| `exports`: `molden`, `cube`, `wfx` | Boolean | `export_molden`, `export_cube`, `export_wfx` |

These are errors in both modes, as in the oracle: an empty document; a top-level value that is not
a map; a geometry entry without a string `symbol`, one with a `position` of fewer than 3 values and
one with an unknown element; and an unknown ONIOM method or a malformed CASSCF active space in
one. A negative integer where the oracle stores an unsigned count (`multiplicity`, the counts and
the atom indices) is also an error in both modes: the oracle wraps it to a count near 2^64, which
no field holds.

### Lenient reading of structured input

| case | input | strict | lenient (oracle) |
|---|---|---|---|
| `:yaml_unknown_key` | `"scf": {"foo": 1}` | `scf.foo: unknown key; accepted keys: …` | ignored |
| `:yaml_wrong_type` | `"scf": {"max_iter": "many"}` | `scf.max_iter: expected an integer, got a string` | ignored, the default kept |
| `:int32_truncated` | `"charge": 3000000000` | an error: expected an integer in the `Int32` range | `x % Int32`, as the oracle's `as i32` reads it; `oniom.model_charge` likewise; an `.inp` `charge` stays an error in both modes |
| `:yaml_int_for_float` | `"scf": {"conv_energy": 1}` | accepted, converted to 1.0 | ignored, the default kept |
| `:yaml_unknown_method` | `"method": "dft"`; `"method": "casscf"` | ``unknown value `dft` for `method` `` | ignored (HF); CASSCF(2, 2) for a `casscf` word, ignored when its counts are malformed |
| `:yaml_unknown_mode` | `"mode": "zzz"` | ``unknown value `zzz` for `mode` `` | ignored |
| `:yaml_unknown_solvent` | `"solvent": "lava"` under SMD | ``unknown value `lava` for `solvation.solvent` `` | water |
| `:yaml_unknown_surface` | `"surface": "blob"` under PCM | ``unknown value `blob` for `solvation.surface` `` | vdw |
| `:yaml_solvation_without_model` | `"solvation": {"epsilon": 4.0}` | `solvation: give model, pcm or smd` | ignored |
| `:yaml_f12_gamma_enables` | `"f12_gamma": 0.9` without `f12` | `f12_gamma needs f12` | `use_f12 = true` |
| `:yaml_constraint_index` | `{"freeze_bond": [0]}`; an index that is not an integer; an entry with none or several of the `freeze_*` keys | `constraints[1].freeze_bond: expected 2 indices, got 1` | fewer indices: ignored; a bad index: 0, 1, 2 by position (0-based); several keys: the first of `freeze_atom`, `freeze_bond`, `freeze_angle` |
| `:yaml_bad_number` | `"position": [0, 0, "x"]`, and an entry of `field`, `lattice` or `kpoints` | `geometry[2].position[3]: expected a number, got a string` | 0.0 (`kpoints`: 1) |
| `:yaml_missing_coordinate` | `{"symbol": "H", "x": 0, "y": 0}` | ``geometry[2]: missing key `z` `` | 0.0 |
| `:yaml_unknown_unit` | `"unit": "nm"` | ``unit: unknown value `nm`; accepted: angstrom, ang, bohr, au`` | Å |
| `:yaml_lattice_row` | a lattice row that is not 3 values | `lattice[2]: expected 3 numbers` | a row of zeros |
| `:yaml_both_keys` | a key and its fallback (`scf.max_iter` and `max_iterations`), `geometry` and `atoms`, `gpu` and `compute_mode`, `position` and `x`, two of `model`, `pcm`, `smd` | `` `scf.max_iter` and `max_iterations` are both given; give one `` | as the oracle reads them: the key, or the fallback when the key is absent (for a scalar key, also when the key has another type); `compute_mode` after `gpu`; `position` before `x`, `y`, `z`; `model`, then `pcm`, then `smd` |
| `:yaml_list_length` | `field` of other than 3 values; `lattice`, `pbc` or `kpoints` of other than 3; a `position` of more than 3 | `scf.field: expected 3 numbers, got 2` | `field`: the first 3, ignored when shorter; `position`: the first 3; the others ignored |
| `:yaml_unknown_value` | an unknown word of `vdw_kernel`, `compute_mode`, `diis_method`, `population`, `irc_direction`, `tddft.method`, `solvation.model` | ``unknown value `d9` for `vdw_kernel` `` | `nothing`; unchanged; CDIIS; the default list; Both; TDA; no solvation |
| `:nonfinite` | `"damping": 1e400` | `scf.damping: expected a finite number, got Inf` | the value |

An entry of `pbc` that is not a Boolean is `:yaml_wrong_type` and reads as true, an entry of
`kpath` or `population` that is not a string is dropped (a `population` list with a bad entry
keeps the default analyses), and a `hubbard`, `u_map` or `j_map` value that is not a number is
dropped. An option of the other solvation model (`epsilon` under SMD, `solvent` under PCM) and a
`target` on `freeze_atom` are `:yaml_unknown_key`, ignored as the oracle ignores them.

The units `au` and `ang` mean bohr and Å in both modes; the oracle reads only `bohr` as bohr and
every other unit as Å. A method word in an ONIOM map is read without regard to case, which the
oracle does only for the top-level `method`.

### Dictionaries and YAML files

`parse_dict(d; source, lenient)` reads structured input from any `AbstractDict`, as YAML.jl,
JSON.jl and other readers build one: `node(d)` converts it to a `Node` tree and
`parse_structured` reads the tree, so a dictionary meets the schema and the lenient cases above.
Maps keep the dictionary's iteration order. Keys are converted with `String`; a key that is not a
string, a key repeated within one map (two keys that convert to the same `String`), an integer
outside the `Int64` range and a value that is not a map, a vector, a Boolean, an integer, a
floating-point number, a string, `nothing` or `missing` are errors in both modes, naming the key
path: `input: a.yaml: title: unsupported value of type Dates.Date`. `nothing` and `missing` are null.

```julia
using QuantumEngine
p = parse_dict(Dict("unit" => "bohr", "geometry" => [Dict("symbol" => "H", "position" => [0, 0, 0])],
    "multiplicity" => 2); source = "h")
```

`parse_yaml(text; source, lenient)` reads a YAML file. It has a method only once YAML.jl is
loaded (`using YAML` loads the extension `QuantumEngineYAMLExt`); YAML.jl is not a dependency of
the package. The extension reads the text with YAML.jl, keeping each mapping as written, in the
order of the file and with any repeated key, and calls `parse_dict`. A key given twice in one
mapping is an error in both modes, as the oracle's reader (yaml-rust2) rejects it; YAML.jl itself
logs an error and keeps the last value, and its log message appears before the error. An empty
document is `the document is empty` in both modes. A document that is not a mapping is
`top-level YAML document is not a mapping` in strict mode, and in lenient mode, as the oracle
reads a document without keys, `no atoms: the input has no geometry block`. Of several documents,
the first is read, as the oracle reads it. Text that YAML.jl's parser or scanner rejects (an
unterminated quoted string, or a tab between a keyword and its value) is, in both modes,
`not valid YAML (<YAML.jl's message>)`; YAML.jl's other errors are its own. That message and the
not-a-mapping message end in `; this file looks like .inp syntax` when the first line that is
neither blank nor a `#` comment (after a byte order mark, if any) is a `keyword value` line (a
word that starts with a letter, then whitespace and a value), as in an `.inp` file read as YAML. A
merge key (`<<`) is `YAML merge keys (<<) are not supported` in both modes (see the table below).

YAML.jl decides the type of each scalar. Where it differs from yaml-rust2 0.9.0, which the oracle
uses for `.yaml`, `.yml` and `.json` files, the schema's type checks decide: a value of the wrong
type for its key is an error in strict mode and the lenient case of the table above in lenient
mode. JSON.jl rejects each of these forms as invalid JSON: `not valid JSON (<JSON.jl's message>)`.

| YAML scalar | YAML.jl reads | yaml-rust2 (the oracle) reads | QuantumEngine, e.g. |
|---|---|---|---|
| `017` (leading zero) | 15, an octal integer | 17 | `max_iterations: 017` is 15 |
| `08`, `09` | a string | 8, 9 | `max_iterations: 08` is `:yaml_wrong_type` |
| `True`, `TRUE`, `False` | a Boolean | a string, which a Boolean key ignores | `zora: True` is true |
| `nan`, `NaN`, `inf`, `infinity` | a string | NaN, Inf | `scf_convergence: nan` is `:yaml_wrong_type` |
| `.NaN` | NaN | a string | `scf_convergence: .NaN` is `:nonfinite` |
| `0b101` | 5 | a string | `max_iterations: 0b101` is 5 |
| `1_000`, `1_000.5` | 1000, 1000.5 | a string | `max_iterations: 1_000` is 1000 |
| `Null`, `NULL` | null | a string | `title: Null` is `:yaml_wrong_type` |
| `2001-12-14` | a `Date` | a string | an error in both modes |
| `9223372036854775808` (beyond `Int64`) | an `Int128` | a float, which an integer key ignores | an error in both modes |
| a key that is not a string (`1: x`) | the integer key 1 | the integer key 1, which no lookup finds | an error in both modes |
| a merge key (`<<: *base`) | merges the mapping it names; an explicit key overrides a merged one | the ordinary key `<<`, which no lookup finds | `YAML merge keys (<<) are not supported`, an error in both modes |

Both readers agree on `-017`, `+017`, `0x1F`, `0o17`, `.5`, `5.`, `1e-6`, `.nan`, `.inf`,
`-.inf`, `1e400` (Inf), `yes`, `no`, `on`, `off`, `~`, `null` and an empty value.

```@docs
QuantumEngine.parse_json
QuantumEngine.Input.parse_structured
QuantumEngine.Input.json_node
QuantumEngine.parse_dict
QuantumEngine.Input.node
QuantumEngine.parse_yaml
QuantumEngine.Input.Node
QuantumEngine.Input.NodeKind
```
