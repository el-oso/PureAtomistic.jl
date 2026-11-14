module Potentials

using PureMLIP: MACEModel, MACEResult, MACEWorkspace, NeighborList, mace_energy_forces!, neighbor_list, neighbor_list!
using ..Systems: cell_volume
using TypeContracts: @contract, @verify, Self
using StrictMode: @strict_exempt

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

@strict_exempt :noalloc MACEPotential reason = "entry point: potential constructor"
@strict_exempt :noalloc LennardJones reason = "entry point: potential constructor"
@strict_exempt :noalloc :_ reason = "entry point: keyword body of a parametric constructor such as `LennardJones{T}(…; kw…)`"
@strict_exempt :noalloc _sized_list reason = "workspace: builds a neighbor list"
@strict_exempt :noalloc _regrow! reason = "workspace: rebuilds the neighbor list and buffers when the edges outgrow them"
@strict_exempt :noalloc _log_regrow reason = "workspace: reports a rebuild of the neighbor list"

end
