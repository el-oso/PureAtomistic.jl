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
    device_neighbor_list(positions, cell, r_cut; skin = 0, extra_capacity = 0, extra_capacity_fraction = 0)

A neighbor list stored on a device: a copy of `nl` on `backend`, or one built for the device arrays
`positions` and `cell` with room for `m + extra_capacity + ceil(extra_capacity_fraction · m)` edges,
where `m` is the number of edges found. Needs the `PureMLIPGPUExt` extension.
"""
device_neighbor_list(nl::NeighborList, backend) = _no_device_backend(backend)
device_neighbor_list(positions, cell, r_cut; skin = 0, extra_capacity = 0, extra_capacity_fraction = 0) =
    _no_device_backend(positions)

"""
    device_rebuild!(nl, positions, cell) -> nl

Rebuild the device neighbor list `nl` for the device arrays `positions` and `cell` under the rule of
`neighbor_list!`, without reading anything back to the host: an overflow of the capacity and the
other flags of the build stay in the list's status word until [`read_device_status!`](@ref) reads
it. Needs the `PureMLIPGPUExt` extension.
"""
device_rebuild!(nl, positions, cell) = _no_device_backend(nl)

"""
    device_energy_forces!(res, ws, model, positions, cell, atomic_numbers, nl) -> res

The device `mace_energy_forces!` without its read of the status word: every kernel is launched,
and nothing is copied to the host. The contents of `res` are valid only once
[`read_device_status!`](@ref) has read the status word after this call and
[`check_device_status`](@ref) has returned 0 for it. Needs the `PureMLIPGPUExt` extension.
"""
device_energy_forces!(res, ws, model, positions, cell, atomic_numbers, nl) = _no_device_backend(model)

"""
    read_device_status!(ws, nl) -> Vector{Int32}

Copy the status word of the device neighbor list `nl` to the host vector `ws.status` of the device
workspace `ws` and return it: the one copy from device to host of a device evaluation. Needs the
`PureMLIPGPUExt` extension.
"""
read_device_status!(ws, nl) = _no_device_backend(nl)

"""
    check_device_status(ws, model, atomic_numbers, nl) -> Int

Act on the status word that [`read_device_status!`](@ref) copied to `ws.status` last. A list that
overflowed its capacity holds no edges, and the number of edges its build found is returned
before the other flags of the evaluation are looked at; a caller rebuilds the list and evaluates
again. Every other flag (an invalid geometry, too many edges for `Int32` indices, an edge without a
reverse, an atomic number outside the model's elements, two atoms at one place) throws an
`ArgumentError`. Returns 0 when no flag is set. Needs the `PureMLIPGPUExt` extension.
"""
check_device_status(ws, model, atomic_numbers, nl) = _no_device_backend(nl)

"""
    device_sum!(out, x, n) -> out

`out[1] = Σ_{i ≤ n} x[i]` for device vectors `out` and `x`, summed on the device in a fixed-shape
tree, so that the sum repeats bit for bit. Needs the `PureMLIPGPUExt` extension.
"""
device_sum!(out, x, n) = _no_device_backend(x)

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
