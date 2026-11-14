module Systems

import LinearAlgebra, Random
using StrictMode: @strict_exempt, @strict_policy

export System, natoms, copy_state!, wrap!, kinetic_energy, temperature, pressure, volume, cell_volume, total_momentum
export maxwell_boltzmann!, fcc_cell, remove_atom, atomic_mass, atomic_number
export KB_EV_PER_K, AMU_IN_EV_FS2_PER_A2, EV_PER_A3_IN_GPA
public kick!, drift!, no_hook

include("units.jl")
include("system.jl")
include("init.jl")
include("motion.jl")

@strict_exempt :noalloc fcc_cell reason = "entry point: builds a new system"
@strict_exempt :noalloc remove_atom reason = "entry point: builds a new system"

# Last, so that it checks every method defined above.
@strict_policy (:typestable, :noalloc)

end
