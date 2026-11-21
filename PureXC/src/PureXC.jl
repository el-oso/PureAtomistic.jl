module PureXC

using PureTaylor: PureTaylor, Taylor, derivatives, fast_cbrt, fast_invcbrt, fast_log1p, fast_sqrt, substitute, value, variable
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
include("lda/lda_x.jl")
include("lda/pw92.jl")

export AbstractFunctional, AbstractLDA, AbstractGGA, family, kind
export xc_derivatives, energy_density
export lda_x, lda_c_pw, lda_c_pw_mod
public Thresholds, default_thresholds, LDAExchange, PW92Correlation

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
