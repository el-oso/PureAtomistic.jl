# Entry points for running a simulation on a KernelAbstractions backend. The methods live in
# `PureAtomisticGPUExt`, which loads with KernelAbstractions; without it, each call lands on the
# fallback below, which names what to load.

_no_device_backend(what) = throw(
    ArgumentError(
        lazy"pureatomistic: device evaluation needs a KernelAbstractions backend (load KernelAbstractions, Metal, CUDA or AMDGPU), got $(typeof(what))"
    )
)

"""
    device_potential(model::MACEModel, sys::System; skin, extra_capacity_fraction = 0.1)

A MACE potential for `sys` whose arrays live on the backend of `sys`'s arrays. Needs the
`PureAtomisticGPUExt` extension.
"""
device_potential(model::PureMLIP.MACEModel, sys::Systems.System; skin, extra_capacity_fraction = 0.1) =
    _no_device_backend(sys)

"""
    to_device(sys::System, backend) -> System

A copy of `sys` whose arrays live on `backend`. Needs the `PureAtomisticGPUExt` extension.
"""
to_device(sys::Systems.System, backend) = _no_device_backend(backend)
