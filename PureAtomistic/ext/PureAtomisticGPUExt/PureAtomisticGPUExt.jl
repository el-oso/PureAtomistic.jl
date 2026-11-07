module PureAtomisticGPUExt

using KernelAbstractions: KernelAbstractions, Backend, allocate, get_backend, @kernel, @index, @Const
using PureAtomistic: PureAtomistic, Simulation, VelocityVerlet
using PureAtomistic.Systems: Systems, System, natoms, no_hook, AMU_IN_EV_FS2_PER_A2, KB_EV_PER_K
using PureAtomistic.Potentials: Potentials, AbstractStressPotential
using PureAtomistic.Integrators: Integrators
using PureMLIP: PureMLIP, MACEModel

include("potential.jl")
include("motion.jl")

end
