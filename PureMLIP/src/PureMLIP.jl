module PureMLIP

include("checkpoint/Checkpoint.jl")
include("equivariant/Equivariant.jl")
include("model/Model.jl")

using .Checkpoint, .Equivariant, .Model
using .Checkpoint: open_torch_archive, entry_data, embedded_archives, unpickle, read_state_dict, numel
using .Checkpoint: StateDict, IrrepMul, InteractionSpec, ReadoutSpec, PickleTree, PickleNode, PickleKind
using .Checkpoint: TorchArchive, ZipEntry
using .Checkpoint: Normalize2Mom, LinearInstruction, LinearSpec, TPInstruction, TPSpec
using .Checkpoint: PK_NONE, PK_BOOL, PK_INT, PK_FLOAT, PK_STR, PK_BYTES, PK_TUPLE, PK_LIST, PK_DICT
using .Checkpoint: PK_GLOBAL, PK_CALL, PK_PERSID

public Checkpoint, Equivariant, Model
export load_mace_checkpoint, MACECheckpoint, MACEHyperparameters, MACEConstants, CheckpointTensor, torch_array
export sh_length, sh_index, spherical_harmonics!, spherical_harmonics_and_gradient!
export clebsch_gordan_complex, real_clebsch_gordan, wigner_D
export e3nn_spherical_harmonics!, e3nn_spherical_harmonics_and_gradient!, e3nn_basis_change, e3nn_wigner_3j
export NeighborList, neighbor_list, neighbor_list!
export MACEModel, MACEResult, MACEWorkspace, mace_energy!

public open_torch_archive, entry_data, embedded_archives, unpickle, read_state_dict, numel
public StateDict, IrrepMul, InteractionSpec, ReadoutSpec, PickleTree, PickleNode, PickleKind
public TorchArchive, ZipEntry
public Normalize2Mom, LinearInstruction, LinearSpec, TPInstruction, TPSpec
public PK_NONE, PK_BOOL, PK_INT, PK_FLOAT, PK_STR, PK_BYTES, PK_TUPLE, PK_LIST, PK_DICT
public PK_GLOBAL, PK_CALL, PK_PERSID

include("precompile.jl")

end
