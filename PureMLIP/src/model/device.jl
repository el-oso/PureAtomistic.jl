# Entry points for evaluating a model on a KernelAbstractions backend. The methods live in
# `PureMLIPGPUExt`, which loads with KernelAbstractions; without it, each call lands on the fallback
# below, which names what to load.

_no_device_backend(what) = throw(
    ArgumentError(
        lazy"model: device evaluation needs a KernelAbstractions backend (load KernelAbstractions, Metal, CUDA or AMDGPU), got $(typeof(what))"
    )
)

"""
    device_model(model::MACEModel, backend; elements)

`model` with its weights on `backend`, built for the atomic numbers in `elements`. Needs the
`PureMLIPGPUExt` extension: load KernelAbstractions, and a vendor package for a GPU.
"""
device_model(model::MACEModel, backend; elements) = _no_device_backend(backend)

"""
    device_neighbor_list(nl::NeighborList, backend)
    device_neighbor_list(positions, cell, r_cut; skin = 0, extra_capacity = 0)

A neighbor list stored on a device: a copy of `nl` on `backend`, or one built for the device arrays
`positions` and `cell`. Needs the `PureMLIPGPUExt` extension.
"""
device_neighbor_list(nl::NeighborList, backend) = _no_device_backend(backend)
device_neighbor_list(positions, cell, r_cut; skin = 0, extra_capacity = 0) = _no_device_backend(positions)

"""
    device_workspace(model, n_atoms, n_edges; edge_chunk, contraction_block)

Scratch arrays for evaluating a device model: per-edge buffers for chunks of `edge_chunk` edge slots, and
symmetric-contraction buffers for blocks of `contraction_block` atoms. Needs the `PureMLIPGPUExt` extension.
"""
device_workspace(model, n_atoms, n_edges; edge_chunk = nothing, contraction_block = nothing) = _no_device_backend(model)

"""
    device_result(model, n_atoms)

The result arrays of a device evaluation. Needs the `PureMLIPGPUExt` extension.
"""
device_result(model, n_atoms) = _no_device_backend(model)
