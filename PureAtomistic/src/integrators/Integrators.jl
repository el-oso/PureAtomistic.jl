module Integrators

using ..Systems: System, natoms, kick!, drift!, no_hook
using ..Potentials: AbstractPotential, n_atoms, prepare!, energy_forces!
using TypeContracts: @contract, @verify, Self
using LinearAlgebra: I, Diagonal, Symmetric, cholesky!, issuccess, dot, mul!
import PureDAQP
using PureDAQP: ActiveSet
using StrictMode: @strict_exempt

export AbstractIntegrator, VelocityVerlet, advance!
export FIRE, RelaxResult, relax!
export FIRE2, TrustRegionBFGS, TrustRegionLBFGS
export NEBResult, neb

include("verlet.jl")
include("../optimize/fire.jl")
include("../optimize/fire2.jl")
# lbfgs.jl first: the `relax!` of quasi_newton.jl takes either trust-region state.
include("../optimize/lbfgs.jl")
include("../optimize/quasi_newton.jl")
include("../optimize/neb.jl")

# Checked at precompile time: every integrator has the contract's method and it infers `Nothing`.
@verify VelocityVerlet trim_compat = true

@strict_exempt :noalloc FIRE reason = "entry point: relaxer constructor"
@strict_exempt :noalloc FIRE2 reason = "entry point: relaxer constructor"
@strict_exempt :noalloc TrustRegionBFGS reason = "entry point: relaxer constructor"
@strict_exempt :noalloc TrustRegionLBFGS reason = "entry point: relaxer constructor"
@strict_exempt :noalloc :_ reason = "entry point: keyword body of a parametric constructor such as `FIRE{T}(n; kw…)`"
@strict_exempt :noalloc _fixed_mask reason = "entry point: builds the fixed-atom mask of the trust-region constructors and `neb`"
@strict_exempt :noalloc _frozen_rows reason = "entry point: builds the frozen-direction rows of the trust-region constructors and `neb`"
@strict_exempt :noalloc relax! reason = "entry point: relaxation driver; `relax!(sys, potential)` builds its relaxer"
@strict_exempt :noalloc neb reason = "entry point: builds the band of a nudged-elastic-band run"

end
