module Analysis

import LinearAlgebra
using ..Systems: KB_EV_PER_K, cell_volume

include("vacancy.jl")
include("diffusion.jl")
include("arrhenius.jl")
include("rdf.jl")
include("coordination.jl")
include("msd.jl")
include("rmsd.jl")

export reference_sites, wigner_seitz_occupancy!, VacancyTracker, update!, hop_rate, msd_diffusion, arrhenius_fit
export rdf, rdf_histogram!, coordination_numbers, coordination_numbers!, unwrap!, msd, diffusion_coefficient, rmsd

end
