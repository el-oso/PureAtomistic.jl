struct MACECheckpoint
    hyperparameters::MACEHyperparameters
    state::StateDict
end

"""
    load_mace_checkpoint(bytes::Vector{UInt8}) -> MACECheckpoint
    load_mace_checkpoint(path::AbstractString) -> MACECheckpoint

Read a mace-torch `.model` file (a `torch.save`d `ScaleShiftMACE`): its hyperparameters
and every parameter and persistent buffer of its module tree.
"""
function load_mace_checkpoint(bytes::Vector{UInt8})
    a = open_torch_archive(bytes)
    t = unpickle(entry_data(a, "data.pkl"))
    t.nodes[t.root].kind == PK_CALL || error("not a MACE checkpoint: the pickle root is not a module")
    h = read_hyperparameters(a, t)
    return MACECheckpoint(h, read_state_dict(a, t))
end

load_mace_checkpoint(path::AbstractString) = load_mace_checkpoint(read(path))
