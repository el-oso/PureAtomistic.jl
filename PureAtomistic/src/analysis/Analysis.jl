module Analysis

include("vacancy.jl")
include("diffusion.jl")
include("arrhenius.jl")

export reference_sites, wigner_seitz_occupancy!, VacancyTracker, update!, hop_rate, msd_diffusion, arrhenius_fit

end
