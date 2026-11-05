module Potentials

using PureMLIP: MACEModel, MACEResult, MACEWorkspace, NeighborList, mace_energy_forces!, neighbor_list, neighbor_list!
using ..Systems: cell_volume
using TypeContracts: @contract, @verify, Self

export AbstractPotential, AbstractStressPotential, PotentialResult
export n_atoms, cutoff, prepare!, energy_forces!, stress, node_energies
export MACEPotential, HarmonicPotential, LennardJones

include("contracts.jl")
include("mace.jl")
include("harmonic.jl")
include("lennard_jones.jl")

# Checked at precompile time: every method of the contract exists for each concrete potential and
# infers its declared return type.
for T in (Float64, Float32)
    @verify MACEPotential{T} trim_compat = true
    @verify HarmonicPotential{T} trim_compat = true
    @verify LennardJones{T} trim_compat = true
end

end
