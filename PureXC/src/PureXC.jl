module PureXC

using PureTaylor: PureTaylor, Taylor, derivatives, value, variable
using StrictMode: @strict_policy

"""
    AbstractFunctional{T}

An exchange-correlation functional whose parameters and thresholds are stored in the number
type `T`.
"""
abstract type AbstractFunctional{T} end

"""
    AbstractLDA{T}

A functional of the density alone.
"""
abstract type AbstractLDA{T} <: AbstractFunctional{T} end

"""
    AbstractGGA{T}

A functional of the density and the squared density gradient.
"""
abstract type AbstractGGA{T} <: AbstractFunctional{T} end

"""
    family(f)

`:lda` or `:gga`, the family of the functional `f`.
"""
family(::AbstractLDA) = :lda
family(::AbstractGGA) = :gga

"""
    kind(f)

`:exchange` or `:correlation`, the kind of the functional `f`. Each concrete functional type
defines its own method.
"""
function kind end

include("thresholds.jl")
include("validate.jl")
include("driver.jl")

export AbstractFunctional, AbstractLDA, AbstractGGA, family, kind
export xc_derivatives, energy_density
public Thresholds, default_thresholds

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
