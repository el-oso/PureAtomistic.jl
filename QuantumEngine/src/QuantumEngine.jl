"""
    QuantumEngine

A quantum-chemistry program in pure Julia. Its submodules are `Errors` (the error types) and
`Units` (physical constants and the element table).
"""
module QuantumEngine

using StrictMode: @strict_policy

include("errors/Errors.jl")
include("units/Units.jl")

using .Errors: QuantumEngineError, InputError, BasisError, SCFNotConverged, OptimizationFailed,
    NotImplemented, FunctionalUnavailable, NumericalError, InputWarning
using .Units: Element, element

export QuantumEngineError, InputError, BasisError, SCFNotConverged, OptimizationFailed,
    NotImplemented, FunctionalUnavailable, NumericalError, InputWarning
export Element, element

public Units

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
