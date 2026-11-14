module Analysis

using ..Systems: KB_EV_PER_K, cell_volume
using StrictMode: @strict_exempt, @strict_policy

include("vacancy.jl")
include("diffusion.jl")
include("arrhenius.jl")
include("rdf.jl")
include("coordination.jl")
include("msd.jl")
include("rmsd.jl")

export reference_sites, reference_sites!, wigner_seitz_occupancy!, VacancyTracker, update!, hop_rate, msd_diffusion, arrhenius_fit
export rdf, rdf!, rdf_histogram!, coordination_numbers, coordination_numbers!, unwrap!, msd, msd!, diffusion_coefficient
export rmsd, rmsd!

@strict_exempt :noalloc reference_sites reason = "entry point: allocates the site matrix for `reference_sites!`"
@strict_exempt :noalloc VacancyTracker reason = "entry point: tracker constructor"
@strict_exempt :noalloc rdf reason = "entry point: allocates the bins for `rdf!`"
@strict_exempt :noalloc coordination_numbers reason = "entry point: allocates the counts for `coordination_numbers!`"
@strict_exempt :noalloc _pick_species reason = "entry point: selects the counts `coordination_numbers` returns"
@strict_exempt :noalloc msd reason = "entry point: allocates the lags, blocks and scratch for `msd!`"
@strict_exempt :noalloc rmsd reason = "entry point: allocates the vector for `rmsd!`"

# Last, so that it checks every method defined above.
@strict_policy (:typestable, :noalloc)

end
