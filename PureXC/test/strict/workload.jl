# The calls whose compiled specializations the StrictMode policy item judges. `check.jl` includes
# this file in a child process that loads only the package, so no test fixture adds a
# specialization of its own.
using PureXC
using PureTaylor: variable
using TypeContracts: TypeContracts, interface_trait

# Every shipped functional in the number type `T`.
functionals(::Type{T}) where {T} = (lda_x(T), lda_c_pw(T), lda_c_pw_mod(T), pbe_x(T), pbe_c(T), pbe(T))

# Restricted and polarized point inputs in `T`, with σαβ inside its bound.
point_inputs(::Type{T}) where {T} = ((T(0.3), T(0.02)), ((T(0.3), T(0.1)), (T(0.02), T(0.005), T(0.01))))

# The point inputs of a functional: ρ alone for an LDA, (ρ, σ) for a GGA.
point_args(f, x) = PureXC.family(f) === :gga ? x : (x[1],)

# Array inputs over `n` points: restricted vectors, polarized `2 × n` and `3 × n` matrices.
function array_inputs(::Type{T}, n) where {T}
    ρ = fill(T(0.3), n)
    σ = fill(T(0.02), n)
    ρp = repeat([T(0.3), T(0.1)], 1, n)
    σp = repeat([T(0.02), T(0.005), T(0.01)], 1, n)
    return (ρ, σ), (ρp, σp)
end

# `xc_outputs` and `xc_evaluate!` on the dense arrays `x` (the pointer path) and on views of them
# (the checked path).
function exercise_arrays(f, k::Val, x)
    args = point_args(f, x)
    out = xc_outputs(f, k, args...)
    xc_evaluate!(out, f, k, args...)
    vargs = map(a -> view(a, axes(a)...), args)
    vout = map(a -> view(a, axes(a)...), out)
    xc_evaluate!(vout, f, k, vargs...)
    return nothing
end

# Orders 0 to 3, and the first order above `_float32_max_order` of every Float32 functional whose
# limit is 3 or more (below it the order-3 calls already take the Float64 path).
orders(f) = (Val(0), Val(1), Val(2), Val(3))
orders(f::Union{PureXC.PBEExchange{Float32}, PureXC.PBECorrelation{Float32}}) = (Val(0), Val(1), Val(2), Val(3), Val(4))
orders(f::PureXC.LDAExchange{Float32}) = (Val(0), Val(1), Val(2), Val(3), Val(11))

function strict_workload(::Type{T}) where {T}
    restricted, polarized = point_inputs(T)
    arestricted, apolarized = array_inputs(T, 4)
    for f in functionals(T)
        PureXC.family(f); PureXC.kind(f)
        for x in (restricted, polarized)
            energy_density(f, point_args(f, x)...)
            zk(energy_density(f, point_args(f, x)...), x[1])
        end
        for k in orders(f), x in (restricted, polarized)
            xc_derivatives(f, k, point_args(f, x)...)
        end
        for k in (Val(0), Val(2)), x in (arestricted, apolarized)
            exercise_arrays(f, k, x)
        end
    end
    return nothing
end

function run_workload()
    for T in (Float32, Float64)
        strict_workload(T)
        PureXC.default_thresholds(T, 1.0e-12, 1.0e-16, 1.0e-16)
        XC(((1, pbe_x(T)), (1, pbe_c(T))))
    end
    # The methods `@contract` adds to TypeContracts functions, for each interface.
    for I in (AbstractFunctional, AbstractLDA, AbstractGGA)
        TypeContracts._contract_specs(I)
        TypeContracts._contract_desc(I)
        for f in functionals(Float64)
            interface_trait(I, typeof(f))
        end
    end
    # A parameter as a series variable: the energy density of PBE exchange in κ.
    f0 = pbe_x()
    κ = variable(Val(1), Val(1), Val(1), f0.κ)
    f = PureXC.PBEExchange{typeof(κ)}(κ, f0.μ, f0.thresholds)
    energy_density(f, 0.3, 0.02)
    energy_density(f, (0.3, 0.1), (0.02, 0.005, 0.01))
    xc = XC(((1, f), (1, pbe_c())))
    energy_density(xc, 0.3, 0.02)
    energy_density(xc, (0.3, 0.1), (0.02, 0.005, 0.01))
    return nothing
end

# The failing control: the same calls on a functional with the abstract number type `Real`, which
# every gate must reject.
function run_control_workload()
    t = PureXC.Thresholds{Real}(1.0e-15, 1.0e-20, 1.0e-16)
    f = PureXC.LDAExchange{Real}(t)
    xc_derivatives(f, Val(1), 0.3)
    xc_derivatives(f, Val(1), (0.3, 0.1))
    return nothing
end
