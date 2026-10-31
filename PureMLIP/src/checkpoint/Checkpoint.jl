"""
    PureMLIP.Checkpoint

Reader for mace-torch `.model` checkpoints: the torch zip container, the restricted pickle
decoder, tensors, the state dict and the model hyperparameters. It depends on no other part
of PureMLIP.
"""
module Checkpoint

export load_mace_checkpoint, MACECheckpoint, MACEHyperparameters, CheckpointTensor, torch_array

public open_torch_archive, entry_data, embedded_archives, unpickle, read_state_dict, numel
public StateDict, IrrepMul, InteractionSpec, ReadoutSpec, PickleTree, PickleNode, PickleKind
public TorchArchive, ZipEntry
public PK_NONE, PK_BOOL, PK_INT, PK_FLOAT, PK_STR, PK_BYTES, PK_TUPLE, PK_LIST, PK_DICT
public PK_GLOBAL, PK_CALL, PK_PERSID

include("crc32.jl")
include("torchzip.jl")
include("pickle.jl")
include("embedded.jl")
include("tensors.jl")
include("statedict.jl")
include("hyperparameters.jl")
include("load.jl")

end
