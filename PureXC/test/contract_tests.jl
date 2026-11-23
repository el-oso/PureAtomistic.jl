@testmodule IncompleteFunctionals begin
    using PureXC

    # No method of the functional contract is defined for these types.
    struct NoMethodsLDA <: PureXC.AbstractLDA{Float64} end
    struct NoMethodsGGA <: PureXC.AbstractGGA{Float64} end

    # Every method of the contracts but `kind`.
    struct NoKind <: PureXC.AbstractLDA{Float64} end
    PureXC._with_number_type(f::NoKind, ::Type) = f
    PureXC._energy_density(::NoKind, ρ) = ρ
    PureXC._cutoff(::NoKind, ρ) = false

    # Every method of the contracts but `_cutoff`.
    struct NoCutoffLDA <: PureXC.AbstractLDA{Float64} end
    PureXC.kind(::NoCutoffLDA) = :exchange
    PureXC._with_number_type(f::NoCutoffLDA, ::Type) = f
    PureXC._energy_density(::NoCutoffLDA, ρ) = ρ
    struct NoCutoffGGA <: PureXC.AbstractGGA{Float64} end
    PureXC.kind(::NoCutoffGGA) = :exchange
    PureXC._with_number_type(f::NoCutoffGGA, ::Type) = f
    PureXC._energy_density(::NoCutoffGGA, ρ, σ) = ρ * σ
end

@testitem "shipped functionals satisfy their contracts" begin
    using TypeContracts
    for T in (Float64, Float32)
        for f in (lda_x(T), lda_c_pw(T), lda_c_pw_mod(T), pbe_x(T), pbe_c(T), pbe(T))
            F = typeof(f)
            @test isconcretetype(F)
            @test check_contract(F).passed
            @test implements(F, AbstractFunctional)
        end
        @test implements(typeof(lda_x(T)), AbstractLDA)
        @test implements(typeof(pbe_c(T)), AbstractGGA)
    end
end

@testitem "an incomplete functional fails its contract" setup = [IncompleteFunctionals] begin
    using TypeContracts
    using .IncompleteFunctionals: NoMethodsLDA, NoMethodsGGA, NoKind, NoCutoffLDA, NoCutoffGGA
    for F in (NoMethodsLDA, NoMethodsGGA, NoKind)
        @test_throws "does not satisfy interface contract" check_contract(F)
        @test !implements(F, AbstractFunctional)
    end
    @test_throws "kind(::Self) :: Symbol" check_contract(NoKind)
    @test !implements(NoMethodsLDA, AbstractLDA)
    @test !implements(NoMethodsGGA, AbstractGGA)
    for F in (NoCutoffLDA, NoCutoffGGA)
        @test_throws "does not satisfy interface contract" check_contract(F)
        @test implements(F, AbstractFunctional)
    end
    @test !implements(NoCutoffLDA, AbstractLDA)
    @test !implements(NoCutoffGGA, AbstractGGA)
    @test_throws "_cutoff(::Self, ::Real)" check_contract(NoCutoffLDA)
    @test_throws "_cutoff(::Self, ::Real, ::Real)" check_contract(NoCutoffGGA)
end

@testitem "contract specs allocate nothing" begin
    using TypeContracts
    using StrictModeTest
    for T in (AbstractFunctional, AbstractLDA, AbstractGGA)
        # @test_noalloc throws a StrictViolation unless AllocCheck proves the call allocation-free.
        @test (@test_noalloc TypeContracts._contract_specs(T)) isa TypeContracts._SpecList
    end
end
