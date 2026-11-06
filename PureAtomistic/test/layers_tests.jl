# Systems is the base layer; Potentials builds on it; Integrators, Thermostats and Barostats
# build on Potentials; Analysis reads all of them; IO depends on Systems only. Each layer lists
# every layer beneath it that it may extend.
@testitem "layer graph holds" tags = [:checks] begin
    using StrictModeTest
    P = PureAtomistic
    test_layers(
        P => (P.Systems, P.Potentials, P.Integrators, P.Thermostats, P.Barostats, P.IO, P.Analysis),
        P.Systems => (),
        P.Potentials => (P.Systems,),
        P.Integrators => (P.Potentials, P.Systems),
        P.Thermostats => (P.Potentials, P.Systems),
        P.Barostats => (P.Potentials, P.Systems),
        P.Analysis => (P.Integrators, P.Thermostats, P.Barostats, P.Potentials, P.Systems),
        P.IO => (P.Systems,)
    )
end

# Package source may read only the `export`ed or `public` names of PureAtomistic's modules.
@testitem "submodules read only each other's public names" tags = [:checks] begin
    using ExplicitImports
    P = PureAtomistic
    for M in (P, P.Systems, P.Potentials, P.Integrators, P.Thermostats, P.Barostats, P.IO, P.Analysis)
        @test isnothing(
            check_all_qualified_accesses_are_public(
                M, pathof(P); from = (P,), allow_internal_accesses = false
            )
        )
    end
end

@testitem "ExplicitImports" tags = [:checks] begin
    using ExplicitImports
    @test isnothing(check_no_stale_explicit_imports(PureAtomistic))
    @test isnothing(check_all_explicit_imports_via_owners(PureAtomistic))
    @test isnothing(check_no_self_qualified_accesses(PureAtomistic))
end

@testitem "Aqua" tags = [:checks] begin
    using Aqua
    Aqua.test_all(PureAtomistic)
end
