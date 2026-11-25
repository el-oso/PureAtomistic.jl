"""
    QuantumEngine

A quantum-chemistry program in pure Julia. Its submodules are `Errors` (the error types).
"""
module QuantumEngine

using StrictMode: @strict_policy

include("errors/Errors.jl")

using .Errors: QuantumEngineError, InputError, BasisError, SCFNotConverged, OptimizationFailed,
    NotImplemented, FunctionalUnavailable, NumericalError, InputWarning

export QuantumEngineError, InputError, BasisError, SCFNotConverged, OptimizationFailed,
    NotImplemented, FunctionalUnavailable, NumericalError, InputWarning

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
