# Plan 1: monorepo setup and the trim-compatible `.model` reader

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the PureAtomistic.jl monorepo with the PureMLIP package, and give PureMLIP a pure-Julia, `juliac --trim` compatible reader that turns a mace-torch `.model` file into a typed state dict plus the model's hyperparameters.

**Architecture:** Four layers in `PureMLIP/src/checkpoint/`, each consuming only the one below: a STORED-only ZIP64 reader with CRC-32 checking; a restricted protocol-2 unpickler that builds a typed node arena and accepts only an allowlist of opcodes and globals; a tensor and state-dict walker that mirrors torch's `state_dict()` naming; and a hyperparameter extractor that reads the MACE module attributes into concrete structs. No `Any`-typed containers anywhere, so `@validate` and `juliac --trim=safe` succeed on the whole path.

**Tech Stack:** Julia 1.12+, Base and stdlib only in the package (SHA is test-only). Tests: TestItems + TestItemRunner, StrictMode + StrictModeTest, TrimCheck. Build: JuliaC.jl in its own tool environment. Docs: Documenter + DocumenterVitepress.

**Spec:** `docs/superpowers/specs/2026-10-30-pureatomistic-design.md` (copied into the repo in Task 1 from the approved design). Sections that govern this plan: 4 (constraints), 6.1 (`.model` parser row), 6.3 (engineering stack), 13 (calls), 14 (trimmed binary option C).

## Global Constraints

- Repository: `PureAtomistic.jl`, one git repo; package folder `PureMLIP/`. `PureAtomistic/` is created in plan 4, not here.
- No Python in any capacity. Test inputs are built in Julia or downloaded.
- PureMLIP `[deps]`: none outside Base. No `_jll`, no `Artifacts.toml`, no Pickle.jl, no ZipFile.jl.
- `julia = "1.12"` in `[compat]`.
- Every public function is type-stable (`test_signatures(...; guarantees = (:typestable,))`) and passes TrimCheck `@validate`.
- Error messages in reachable code use `lazy"..."` when they interpolate four or more pieces.
- Generic indexing rules apply; byte-buffer functions take `Vector{UInt8}` (a file's bytes) and say so in the signature.
- No `@inbounds` (rule: only with a benchmark and a test; this plan has neither need).
- Commits: subject ≤ 72 chars, body describes the change for a reader with only the repo, trailer `Assisted-by: Claude Opus 5.5 <noreply@anthropic.com>`.
- Creating the GitHub remote is outward-facing: ask the user before `gh repo create`. Local commits need no approval.
- License: MIT.
- TypeContracts: this plan introduces no abstract types, so it has no `@contract`; contracts start with the first abstract interface (plan 2 or 3).

Verified facts this plan relies on (recon on 2026-10-30, files downloaded from the URLs below):

| File | URL | SHA-256 | Bytes |
|---|---|---|---|
| MP-0b3 | `https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model` | `2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012` | 79472952 |
| MPA-0 | `https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model` | `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638` | 79462305 |

- Both are zip archives with **every entry STORED** (93 entries in MPA-0), a ZIP64 end-of-central-directory record and locator (`PK\x06\x06`, `PK\x06\x07`) before the classic one, and a top folder that is not `archive/` (MPA-0: `mace-alex-main-branch/`). Entries: `<prefix>/data.pkl`, `<prefix>/byteorder` (content `little`), `<prefix>/version` (content `3`), `<prefix>/data/<key>`, `<prefix>/.data/serialization_id`.
- `data.pkl` starts `0x80 0x02` (protocol 2). Both files use exactly these 31 opcodes: `PROTO 0x80, GLOBAL c, BINPUT q, LONG_BINPUT r, BINGET h, LONG_BINGET j, BININT J, BININT1 K, BININT2 M, BINFLOAT G, BINUNICODE X, MARK (, EMPTY_TUPLE ), TUPLE t, TUPLE1 0x85, TUPLE2 0x86, TUPLE3 0x87, EMPTY_LIST ], APPEND a, APPENDS e, EMPTY_DICT }, SETITEM s, SETITEMS u, NONE N, NEWTRUE 0x88, NEWFALSE 0x89, REDUCE R, NEWOBJ 0x81, BUILD b, BINPERSID Q, STOP .`
- Both files use exactly these 42 globals (`module.name`): `__builtin__.set, _codecs.encode, collections.OrderedDict, e3nn.math._normalize_activation.normalize2mom, e3nn.nn._activation.Activation, e3nn.nn._fc.FullyConnectedNet, e3nn.nn._fc._Layer, e3nn.o3._irreps.Irrep, e3nn.o3._irreps.Irreps, e3nn.o3._irreps._MulIr, e3nn.o3._linear.Instruction, e3nn.o3._linear.Linear, e3nn.o3._spherical_harmonics.SphericalHarmonics, e3nn.o3._tensor_product._instruction.Instruction, e3nn.o3._tensor_product._sub.FullyConnectedTensorProduct, e3nn.o3._tensor_product._tensor_product.TensorProduct, mace.modules.blocks.AtomicEnergiesBlock, mace.modules.blocks.EquivariantProductBasisBlock, mace.modules.blocks.LinearNodeEmbeddingBlock, mace.modules.blocks.LinearReadoutBlock, mace.modules.blocks.NonLinearReadoutBlock, mace.modules.blocks.RadialEmbeddingBlock, mace.modules.blocks.RealAgnosticDensityInteractionBlock, mace.modules.blocks.RealAgnosticDensityResidualInteractionBlock, mace.modules.blocks.ScaleShiftBlock, mace.modules.irreps_tools.reshape_irreps, mace.modules.models.ScaleShiftMACE, mace.modules.radial.AgnesiTransform, mace.modules.radial.BesselBasis, mace.modules.radial.PolynomialCutoff, mace.modules.radial.ZBLBasis, mace.modules.symmetric_contraction.Contraction, mace.modules.symmetric_contraction.SymmetricContraction, torch.DoubleStorage, torch.LongStorage, torch._utils._rebuild_parameter, torch._utils._rebuild_tensor_v2, torch.fx._symbolic_trace.Tracer, torch.fx.graph_module.reduce_graph_module, torch.nn.functional.silu, torch.nn.modules.container.ModuleList, torch.nn.modules.container.ParameterList`.
- Reference fixture: llmlip.rs (Apache-2.0) `fixtures/mace/v0_3_16_4d2da09/mace_mp0b_inspection.json` at commit `6bfb0418cc5aba0a13416fd783994fa82552cc34`. Its values for MP-0b3: `full_state_tensor_count` 93; `model.num_parameters` 9063204; `model.r_max` 6.0; `model.num_interactions` 2; `model.heads` `["default"]`; 89 `atomic_numbers`, last three `92, 93, 94`; interaction 0 class `RealAgnosticDensityInteractionBlock`, interaction 1 class `RealAgnosticDensityResidualInteractionBlock`, both `avg_num_neighbors` 61.964672446250916 and `radial_MLP` `[64,64,64]`; product 0 contraction 0 `U_matrix_3` shape `[16,16,16,23]` with 353 nonzeros, `weights_max` shape `[89,23,128]`, `correlation` 3; readout 0 `LinearReadoutBlock`, readout 1 `NonLinearReadoutBlock` with `hidden_irreps` `[[16,[0,1]]]`.

## Review Focus

1. **A checkpoint the reader was not written for** (OMOL, POLAR, a cuEquivariance-converted model, any other torch file): the user expects an error naming the unknown global or opcode and its byte offset, never a partial result. Test in Task 3.
2. **A corrupted or truncated download**: expected a CRC or bounds error naming the entry, not garbage tensors. Test in Task 2.
3. **A zip with a deflated entry** (a re-zipped checkpoint): expected `compressed entries are not supported: <name> (method 8)`. Test in Task 2.
4. **A non-contiguous or offset tensor view in the pickle** (storage shared by several tensors): expected the values torch would return, not the first `prod(shape)` storage elements. Test in Task 4.
5. **An older MACE file where `avg_num_neighbors` is a plain attribute** (MP-0b3, MACE 0.3.16) versus a newer one where it is a buffer: both must give 61.964672446250916 for MP-0b3. Test in Task 5.

---

### Task 1: Monorepo, PureMLIP skeleton, CI, docs

**Files:**
- Create: `PureAtomistic.jl/README.md`, `LICENSE`, `.gitignore`, `.github/workflows/CI.yml`, `.github/workflows/docs.yml`
- Create: `PureAtomistic.jl/docs/superpowers/specs/2026-10-30-pureatomistic-design.md` (copy of the approved design), `docs/superpowers/plans/2026-10-30-plan1-model-reader.md` (this file)
- Create: `PureMLIP/Project.toml`, `PureMLIP/src/PureMLIP.jl`
- Create: `PureMLIP/test/Project.toml`, `PureMLIP/test/runtests.jl`, `PureMLIP/test/smoke_tests.jl`
- Create: `PureMLIP/docs/Project.toml`, `PureMLIP/docs/make.jl`, `PureMLIP/docs/package.json`, `PureMLIP/docs/src/index.md`

**Interfaces:**
- Produces: module `PureMLIP`; test env with `TestItemRunner, Test, StrictMode, StrictModeTest, TrimCheck, SHA, Downloads`; `[preferences.StrictMode] checks_enabled = true` in `test/Project.toml`.

- [ ] **Step 1:** `git init PureAtomistic.jl`; generate `PureMLIP/Project.toml` with a fresh UUID (`using UUIDs; uuid4()`), `version = "0.1.0"`, `[compat] julia = "1.12"`, no deps.
- [ ] **Step 2: Write the smoke test** in `test/smoke_tests.jl`:

```julia
@testitem "PureMLIP loads" begin
    @test isdefined(PureMLIP, :PureMLIP)
end
```

`test/runtests.jl` is `using TestItemRunner; using PureMLIP; @run_package_tests`. Copy the structure of PureBLAS's `test/Project.toml` (deps listed above; `[compat] StrictModeTest = "0.5", TrimCheck = "0.1.5"`; the StrictMode preference).
- [ ] **Step 3:** Run `julia --project=PureMLIP -e 'using Pkg; Pkg.precompile(); Pkg.test()'`. Expected: 1 test item passes.
- [ ] **Step 4:** Docs: `make.jl` modeled on PureBLAS's (`sitename = "PureMLIP.jl"`, `repo = "github.com/el-oso/PureAtomistic.jl"`, `devbranch = "main"`, pages `["Home" => "index.md"]`), `package.json` copied from PureBLAS's `docs/`. Run `julia --project=PureMLIP/docs -e 'using Pkg; Pkg.develop(path="PureMLIP"); Pkg.instantiate(); include("PureMLIP/docs/make.jl")'`. Expected: build finishes without errors.
- [ ] **Step 5:** CI workflow: one job, Julia `1.12` and `1.13`, `julia-actions/julia-buildpkg` and `julia-runtest` with `project: PureMLIP`, `check_bounds: 'yes'`. Docs workflow builds `PureMLIP/docs`. `.gitignore`: `Manifest*.toml`, `docs/build`, `PureMLIP/juliac/build/`, `node_modules`.
- [ ] **Step 6:** Commit: `Create PureAtomistic monorepo with PureMLIP skeleton`.

### Task 2: CRC-32 and the STORED ZIP64 reader

**Files:**
- Create: `PureMLIP/src/checkpoint/crc32.jl`, `PureMLIP/src/checkpoint/torchzip.jl`
- Create: `PureMLIP/test/zip_tests.jl`, `PureMLIP/test/testutils.jl` (`@testmodule ZipFixtures`)

**Interfaces:**
- Produces:
  - `crc32(data::AbstractVector{UInt8})::UInt32` — IEEE 802.3 polynomial `0xEDB88320`, reflected, init and final XOR `0xFFFFFFFF`, 256-entry table built at module load as a `const`.
  - `struct ZipEntry; name::String; data_offset::Int; size::Int; crc::UInt32; end`
  - `struct TorchArchive; bytes::Vector{UInt8}; prefix::String; entries::Dict{String,ZipEntry}; end` — `prefix` is the folder holding `data.pkl`, with a trailing `/`.
  - `open_torch_archive(bytes::Vector{UInt8})::TorchArchive`
  - `entry_data(a::TorchArchive, name::AbstractString)` returns a `view` of `a.bytes` for `a.prefix * name`; verifies the CRC on first access.
- Test helper (test-only): `ZipFixtures.stored_zip(entries::Vector{Pair{String,Vector{UInt8}}}; zip64::Bool=true, method::UInt16=0x0000)::Vector{UInt8}` writes local headers with no extra field (so the first data byte of the first entry is at index `31 + length(name)`), the central directory, and (when `zip64`) the ZIP64 EOCD record and locator. Task 4 and Task 5 add `ZipFixtures.zip_with_pickle(pkl::Vector{UInt8}, storages::Vector{Pair{String,Vector{UInt8}}})` (prefix `t/`, `byteorder` = `little`) and `ZipFixtures.dict_root_zip()` (a `data.pkl` whose root is an empty dict).

- [ ] **Step 1: Write the failing tests:**

```julia
@testitem "crc32 matches the IEEE check value" begin
    @test PureMLIP.crc32(codeunits("123456789")) == 0xCBF43926
    @test PureMLIP.crc32(UInt8[]) == 0x00000000
end

@testitem "stored zip round trip" setup = [ZipFixtures] begin
    z = ZipFixtures.stored_zip(["arch/data.pkl" => UInt8[0x80, 0x02, 0x2e], "arch/byteorder" => collect(codeunits("little"))])
    a = PureMLIP.open_torch_archive(z)
    @test a.prefix == "arch/"
    @test collect(PureMLIP.entry_data(a, "byteorder")) == collect(codeunits("little"))
end

@testitem "zip errors" setup = [ZipFixtures] begin
    z = ZipFixtures.stored_zip(["a/data.pkl" => UInt8[1, 2, 3]]; method = 0x0008)
    @test_throws "compressed entries are not supported: a/data.pkl (method 8)" PureMLIP.open_torch_archive(z)
    good = ZipFixtures.stored_zip(["a/data.pkl" => UInt8[1, 2, 3]])
    bad = copy(good); bad[31 + length("a/data.pkl")] ⊻= 0xff   # first data byte
    @test_throws "CRC mismatch in a/data.pkl" PureMLIP.entry_data(PureMLIP.open_torch_archive(bad), "data.pkl")
    @test_throws "no end-of-central-directory record" PureMLIP.open_torch_archive(good[1:end-30])
    @test_throws "no data.pkl entry" PureMLIP.open_torch_archive(ZipFixtures.stored_zip(["a/x" => UInt8[1]]))
end
```

- [ ] **Step 2:** Run `Pkg.test()`; expected FAIL (`crc32` not defined).
- [ ] **Step 3:** Implement. Find the classic EOCD by scanning back from the end for `0x06054b50`; if it carries `0xFFFF`/`0xFFFFFFFF` sentinels or a ZIP64 locator (`0x07064b50`) precedes it, read the ZIP64 EOCD (`0x06064b50`) for the central-directory offset and count. For each central entry read the ZIP64 extra field (id `0x0001`) when sizes or offset are `0xFFFFFFFF`. The data offset is the local header offset + 30 + local name length + local extra length (read from the local header, not the central one). Reject any method other than 0 and any entry whose range exceeds the buffer.
- [ ] **Step 4:** Run `Pkg.test()`; expected PASS.
- [ ] **Step 5:** Commit: `Add CRC-32 and a STORED-only ZIP64 reader`.

### Task 3: Restricted protocol-2 unpickler

**Files:**
- Create: `PureMLIP/src/checkpoint/pickle.jl`
- Create: `PureMLIP/test/pickle_tests.jl`

**Interfaces:**
- Produces:
  - `@enum PickleKind::UInt8 PK_NONE PK_BOOL PK_INT PK_FLOAT PK_STR PK_BYTES PK_TUPLE PK_LIST PK_DICT PK_GLOBAL PK_CALL PK_PERSID`
  - `struct PickleNode; kind::PickleKind; int::Int64; float::Float64; str::String; items::Vector{Int32}; end`
  - `struct PickleTree; nodes::Vector{PickleNode}; root::Int32; end`
  - `unpickle(bytes::AbstractVector{UInt8})::PickleTree`
  - `const ALLOWED_GLOBALS::Set{String}` — exactly the 42 names in Global Constraints.
- Node semantics (later tasks depend on these exactly):
  - `PK_TUPLE`/`PK_LIST`: `items` are the elements. `PK_DICT`: `items` are `k1, v1, k2, v2, ...` in insertion order.
  - `PK_GLOBAL`: `str` is `"module.name"`.
  - `REDUCE` and `NEWOBJ` make `PK_CALL` with `items = [callable, args]`; `BUILD` appends the state as `items[3]`. A second `BUILD` on one node is an error.
  - `REDUCE` of `collections.OrderedDict` with an empty tuple makes an empty `PK_DICT` (so `SETITEMS` can fill it). `REDUCE` of `_codecs.encode` with `(str, "latin1")` makes `PK_BYTES` whose `str` holds the bytes (each code point < 256 becomes one byte).
  - `BINPERSID` makes `PK_PERSID` with `items = [the persistent-id tuple]`.
  - Memo `BINGET` returns the stored node index, so nodes may be shared.
  - `BINFLOAT` is big-endian. `BININT` is signed little-endian 32-bit.

- [ ] **Step 1: Write the failing tests** (pickles are byte literals; `op(c) = UInt8(c)`):

```julia
@testitem "unpickle scalars and containers" begin
    b = UInt8[0x80, 0x02, UInt8('}'), UInt8('('), UInt8('X'), 0x01, 0, 0, 0, UInt8('a'),
              UInt8('K'), 0x05, UInt8('X'), 0x01, 0, 0, 0, UInt8('b'),
              UInt8('G'), reinterpret(UInt8, [hton(6.0)])..., UInt8('u'), UInt8('.')]
    t = PureMLIP.unpickle(b)
    r = t.nodes[t.root]
    @test r.kind == PureMLIP.PK_DICT && length(r.items) == 4
    @test t.nodes[r.items[2]].int == 5
    @test t.nodes[r.items[4]].float == 6.0
end

@testitem "unpickle rejects unknown globals and opcodes" begin
    g = UInt8[0x80, 0x02, UInt8('c'), codeunits("os\nsystem\n")..., UInt8('.')]
    @test_throws "unpickle: global os.system is not allowed (byte 3)" PureMLIP.unpickle(g)
    o = UInt8[0x80, 0x02, 0x8a, 0x01, 0x05, UInt8('.')]   # LONG1, not in the allowed set
    @test_throws "unpickle: unsupported opcode 0x8a at byte 3" PureMLIP.unpickle(o)
    @test_throws "unpickle: unexpected end of data" PureMLIP.unpickle(UInt8[0x80, 0x02, UInt8('K')])
end

@testitem "unpickle OrderedDict, BUILD, memo, persistent id" begin
    # c collections\nOrderedDict\n ) R q\x00 ( X..k K\x07 u  then BINGET 0, MARK, X..pid, t, Q, TUPLE2, .
    # (exact bytes written in the test; asserts: root is PK_TUPLE of [PK_DICT with one pair, PK_PERSID])
end
```

The third item builds its bytes inline in the test; it must assert: the dict made from `OrderedDict` has kind `PK_DICT` and one key/value pair; the memo `BINGET` returns the same node index that `BINPUT` stored; the `PK_PERSID` node's `items[1]` is a `PK_TUPLE`.
- [ ] **Step 2:** Run tests; expected FAIL.
- [ ] **Step 3:** Implement `unpickle` with a value stack of `Int32` node indices, a mark stack of `Int`, a memo `Dict{Int,Int32}`. Errors use the exact messages in the tests.
- [ ] **Step 4:** Run tests; expected PASS.
- [ ] **Step 5:** Commit: `Add restricted protocol-2 unpickler`.

### Task 4: Tensors and the state dict

**Files:**
- Create: `PureMLIP/src/checkpoint/tensors.jl`, `PureMLIP/src/checkpoint/statedict.jl`
- Create: `PureMLIP/test/tensor_tests.jl`

**Interfaces:**
- Consumes: `TorchArchive`, `entry_data` (Task 2); `PickleTree`, `PickleNode`, `PickleKind` (Task 3).
- Produces:
  - `struct CheckpointTensor{T}; shape::Vector{Int}; data::Vector{T}; requires_grad::Bool; end` — `data` is contiguous in torch (row-major) order; a 0-dimensional tensor has `shape == Int[]` and one element.
  - `struct StateDict; f64::Dict{String,CheckpointTensor{Float64}}; i64::Dict{String,CheckpointTensor{Int64}}; names::Vector{String}; end` — `names` in torch `state_dict()` order.
  - `read_state_dict(a::TorchArchive, t::PickleTree)::StateDict`
  - `torch_array(x::CheckpointTensor{T}, ::Val{N})::Array{T,N}` with `A[i1, ..., iN] == torch_tensor[i1-1, ..., iN-1]` (builds the reversed-dims array and `permutedims` it).
  - `numel(x::CheckpointTensor)::Int`.
- Rules:
  - A tensor is a `PK_CALL` to `torch._utils._rebuild_tensor_v2` with args `(storage, offset, size, stride, requires_grad, hooks)`; `storage` is a `PK_PERSID` of `("storage", torch.DoubleStorage | torch.LongStorage, key, location, numel)`. A `torch._utils._rebuild_parameter` call wraps such a tensor and sets `requires_grad` from its second argument.
  - Values are gathered element by element through `offset` and `stride` (in elements), so views and shared storages read correctly. Storage bytes come from `entry_data(a, "data/" * key)` read as little-endian; `byteorder` must be `little`, otherwise error `big-endian checkpoints are not supported`.
  - A module is a `PK_CALL` whose state (`items[3]`, from `BUILD`) is a `PK_DICT` with keys `_parameters`, `_buffers`, `_modules`; for `torch.fx.graph_module.reduce_graph_module` the same three keys are in the first argument dict. Walk depth-first in insertion order, joining names with `.`; skip `None` values and names listed in `_non_persistent_buffers_set` (a `__builtin__.set` call over a list) when present. Order: own parameters, own buffers, then children — torch's order.

- [ ] **Step 1: Write the failing tests:**

```julia
@testitem "torch_array index order" begin
    x = PureMLIP.CheckpointTensor{Float64}([2, 3], collect(1.0:6.0), false)
    A = PureMLIP.torch_array(x, Val(2))
    @test size(A) == (2, 3) && A[1, 2] == 2.0 && A[2, 1] == 4.0
end

@testitem "strided view of a shared storage" setup = [ZipFixtures] begin
    # storage 0 holds Float64 1:6; the pickle rebuilds a tensor with offset 1, size (2,), stride (2,)
    # expected values [2.0, 4.0]; built with ZipFixtures.stored_zip and a hand-written data.pkl
end
```

The second item writes the zip and pickle in the test and asserts `sd.f64["x"].data == [2.0, 4.0]`.
- [ ] **Step 2:** Run tests; expected FAIL.
- [ ] **Step 3:** Implement per the rules above.
- [ ] **Step 4:** Run tests; expected PASS.
- [ ] **Step 5:** Commit: `Read torch tensors and state dict from a checkpoint`.

### Task 5: MACE hyperparameters and `load_mace_checkpoint`

**Files:**
- Create: `PureMLIP/src/checkpoint/hyperparameters.jl`, `PureMLIP/src/checkpoint/load.jl`
- Create: `PureMLIP/test/fixtures/llmlip_mace_mp0b_inspection.json` (verbatim from llmlip.rs at the pinned commit), `PureMLIP/test/fixtures/NOTICE` (Apache-2.0 attribution: source repo, path, commit)
- Create: `PureMLIP/test/checkpoint_tests.jl`; add `@testmodule Checkpoints` to `testutils.jl`
- Modify: `PureMLIP/src/PureMLIP.jl` (export `load_mace_checkpoint`, `MACECheckpoint`, `MACEHyperparameters`, `CheckpointTensor`, `torch_array`)

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces:
  - `struct IrrepMul; mul::Int; l::Int; parity::Int; end`
  - `struct InteractionSpec; class::String; avg_num_neighbors::Float64; radial_MLP::Vector{Int}; node_feats_irreps::Vector{IrrepMul}; edge_attrs_irreps::Vector{IrrepMul}; edge_feats_irreps::Vector{IrrepMul}; target_irreps::Vector{IrrepMul}; hidden_irreps::Vector{IrrepMul}; end`
  - `struct ReadoutSpec; class::String; hidden_irreps::Vector{IrrepMul}; end` (empty `hidden_irreps` for a linear readout)
  - `struct MACEHyperparameters; model_class::String; r_max::Float64; num_interactions::Int; atomic_numbers::Vector{Int}; heads::Vector{String}; interactions::Vector{InteractionSpec}; correlations::Vector{Int}; readouts::Vector{ReadoutSpec}; end`
  - `struct MACECheckpoint; hyperparameters::MACEHyperparameters; state::StateDict; end`
  - `load_mace_checkpoint(bytes::Vector{UInt8})::MACECheckpoint`, `load_mace_checkpoint(path::AbstractString)::MACECheckpoint`
  - Test helper: `Checkpoints.fetch(name::Symbol)::String` downloads `:mp0b3` or `:mpa0` (URLs and SHA-256 in Global Constraints) into `get(ENV, "PUREMLIP_CHECKPOINT_DIR", joinpath(first(DEPOT_PATH), "scratchspaces", "PureMLIP-checkpoints"))`, verifies SHA-256 with stdlib SHA, and returns the path; a mismatch deletes the file and errors.
- Rules: the model class must be `mace.modules.models.ScaleShiftMACE`, otherwise error `unsupported model class <name>`. Irreps come from `e3nn.o3._irreps.Irreps` objects (a sequence of `_MulIr(mul, Irrep(l, p))`). `avg_num_neighbors` is read from the interaction's `_buffers` when present, else from its attribute. `correlations[i]` is `length(weights) + 1` of product `i`'s contractions (all contractions of one product share it; error if they differ). Attribute names are those of `mace/modules/blocks.py` and `models.py` at mace commit `7215fad`; the tests below are the arbiter.

- [ ] **Step 1: Write the failing tests:**

```julia
@testitem "MP-0b3 matches the llmlip.rs inspection" setup = [Checkpoints] begin
    ck = load_mace_checkpoint(Checkpoints.fetch(:mp0b3))
    ref = Checkpoints.inspection()          # parsed fixture JSON (Base-only reader in testutils, or TOML-free hand parse)
    h = ck.hyperparameters
    @test length(ck.state.names) == 93
    @test sum(PureMLIP.numel(ck.state.f64[n]) for n in ck.state.names if haskey(ck.state.f64, n) && ck.state.f64[n].requires_grad) == 9063204
    @test h.r_max == 6.0 && h.num_interactions == 2 && h.heads == ["default"]
    @test length(h.atomic_numbers) == 89 && h.atomic_numbers[end-2:end] == [92, 93, 94]
    @test [i.class for i in h.interactions] == ["mace.modules.blocks.RealAgnosticDensityInteractionBlock",
                                               "mace.modules.blocks.RealAgnosticDensityResidualInteractionBlock"]
    @test all(i -> i.avg_num_neighbors == 61.964672446250916 && i.radial_MLP == [64, 64, 64], h.interactions)
    @test h.correlations == [3, 3]
    @test h.readouts[2].hidden_irreps == [PureMLIP.IrrepMul(16, 0, 1)]
    U = ck.state.f64["products.0.symmetric_contractions.contractions.0.U_matrix_3"]
    @test U.shape == [16, 16, 16, 23] && count(!iszero, U.data) == 353
    @test ck.state.f64["products.0.symmetric_contractions.contractions.0.weights_max"].shape == [89, 23, 128]
    # every tensor listed in ref (state_dict_coverage names, missing_state_tensors entries with shape/dtype/nonzero,
    # and all product_blocks parameters/buffers) is present with the same shape, dtype and nonzero count
    @test Checkpoints.mismatches(ck, ref) == String[]
end

@testitem "MPA-0 loads" setup = [Checkpoints] begin
    h = load_mace_checkpoint(Checkpoints.fetch(:mpa0)).hyperparameters
    @test h.r_max == 6.0 && h.correlations == [3, 3] && length(h.atomic_numbers) == 89
    @test h.interactions[1].hidden_irreps == [PureMLIP.IrrepMul(128, 0, 1), PureMLIP.IrrepMul(128, 1, -1)]
    @test all(i -> isfinite(i.avg_num_neighbors) && i.avg_num_neighbors > 0, h.interactions)   # buffer or attribute form, whichever MPA-0 uses (unverified)
end

@testitem "non-MACE pickle is rejected" setup = [ZipFixtures] begin
    # zip holding a data.pkl that unpickles to a PK_DICT root
    @test_throws "not a MACE checkpoint: the pickle root is not a module" load_mace_checkpoint(ZipFixtures.dict_root_zip())
end
```

`Checkpoints.inspection()` and `Checkpoints.mismatches` are test helpers: the JSON is read with a small test-only JSON reader or the `JSON` package added to the **test** environment only (allowed: test-only dependency). `mismatches` returns one string per differing tensor.
- [ ] **Step 2:** Run tests; expected FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run tests; expected PASS. If the 93 count or a name differs, read the pickle structure for that module before changing the walk rule in Task 4.
- [ ] **Step 5:** Commit: `Load MACE hyperparameters and state from a .model file`.

### Task 6: Trim and StrictMode gates, juliac build

**Files:**
- Create: `PureMLIP/test/trim_tests.jl`, `PureMLIP/test/strictmode_tests.jl`
- Create: `PureMLIP/juliac/build.jl`, `PureMLIP/juliac/entry.jl`, `PureMLIP/juliac/toolenv/Project.toml` (`JuliaC = "0.3"`), modeled on PureBLAS's `juliac/`
- Modify: `PureMLIP/docs/src/index.md` (usage of `load_mace_checkpoint`), `README.md`

**Interfaces:**
- Consumes: `load_mace_checkpoint` (Task 5).
- Produces: executable `PureMLIP/juliac/build/mace_inspect`, whose `@main` reads one path from `ARGS`, loads it, and prints to `Core.stdout` exactly: `model_class=<s>`, `r_max=<f>`, `num_interactions=<n>`, `state_tensors=<n>`, `num_parameters=<n>`, one per line.

- [ ] **Step 1: Write the failing tests:**

```julia
@testitem "TrimCheck: checkpoint loading" tags = [:checks] begin
    using TrimCheck
    @validate(init = begin using PureMLIP end,
              PureMLIP.load_mace_checkpoint(Vector{UInt8}),
              PureMLIP.load_mace_checkpoint(String))
end

@testitem "StrictMode: public API is type-stable" tags = [:checks] begin
    using StrictModeTest, StrictMode
    P = PureMLIP
    test_signatures([(P.load_mace_checkpoint, (Vector{UInt8},)), (P.load_mace_checkpoint, (String,)),
                     (P.open_torch_archive, (Vector{UInt8},)), (P.unpickle, (Vector{UInt8},)),
                     (P.torch_array, (P.CheckpointTensor{Float64}, Val{3}))]; guarantees = (:typestable,))
end
```

- [ ] **Step 2:** Run `Pkg.test()`; fix every TrimCheck finding at its source (typical: an `Any` in a container, a 4+-piece interpolated error string → `lazy"..."`).
- [ ] **Step 3:** Run `julia PureMLIP/juliac/build.jl`; expected: build exits 0. Then `PureMLIP/juliac/build/mace_inspect <path to MP-0b3>`; expected output includes `num_parameters=9063204` and `state_tensors=93`.
- [ ] **Step 4:** Add a `@testitem "juliac build" tags = [:juliac]` that runs the build and the binary on MP-0b3 and checks those two lines; CI excludes `:juliac`, local runs include it (the PureBLAS `test/juliac_build_test.jl` pattern).
- [ ] **Step 5:** Commit: `Gate checkpoint loading with TrimCheck, StrictMode and a juliac build`.
- [ ] **Step 6:** Ask the user before creating the GitHub repo `el-oso/PureAtomistic.jl` and pushing.
