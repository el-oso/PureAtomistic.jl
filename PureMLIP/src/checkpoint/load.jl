"""
    MACECheckpoint

Fields: `hyperparameters`, a `MACEHyperparameters`, and `state`, the tensors stored in the
file. `state` is a `StateDict` with `f64` (name to `CheckpointTensor{Float64}`), `i64`
(name to `CheckpointTensor{Int64}`) and `names` (every name, in torch's `state_dict()`
order). For example, `ck.state.i64["atomic_numbers"]`.

`state` holds only the tensors present in the file. Buffers that e3nn rebuilds when the
model is constructed (the Wigner-3j buffers of the compiled tensor products) are not
stored and so are not included: for MACE-MP-0b3, 89 tensors are stored against 93 in
torch's in-memory `state_dict()`.
"""
struct MACECheckpoint
    hyperparameters::MACEHyperparameters
    state::StateDict
end

"""
    load_mace_checkpoint(bytes::Vector{UInt8}) -> MACECheckpoint
    load_mace_checkpoint(path::AbstractString) -> MACECheckpoint

Read a mace-torch `.model` file (a `torch.save`d `ScaleShiftMACE`): its hyperparameters
and the parameters and persistent buffers stored in it (see `MACECheckpoint`).
"""
function load_mace_checkpoint(bytes::Vector{UInt8})
    a = open_torch_archive(bytes)
    t = unpickle(entry_data(a, "data.pkl"))
    t.nodes[t.root].kind == PK_CALL || error("not a MACE checkpoint: the pickle root is not a module")
    h = read_hyperparameters(a, t)
    return MACECheckpoint(h, read_state_dict(a, t))
end

load_mace_checkpoint(path::AbstractString) = load_mace_checkpoint(read(path))
