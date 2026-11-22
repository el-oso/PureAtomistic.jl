module PureXC

using PureTaylor: PureTaylor, Taylor, derivatives, fast_cbrt, fast_exp, fast_expm1, fast_invcbrt, fast_log, fast_log1p, fast_sqrt, power_series, substitute, value, variable
using StrictMode: @strict_policy
using TypeContracts: @contract, Self

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
include("gga/pbe_x.jl")
include("gga/pbe_c.jl")
include("array.jl")
include("composite.jl")
include("contracts.jl")

export AbstractFunctional, AbstractLDA, AbstractGGA, family, kind
export xc_derivatives, energy_density, xc_evaluate!, xc_outputs, XC, pbe, zk
export lda_x, lda_c_pw, lda_c_pw_mod, pbe_x, pbe_c
public Thresholds, default_thresholds, LDAExchange, PW92Correlation, PBEExchange, PBECorrelation

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
