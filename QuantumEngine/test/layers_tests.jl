# The layer graph of the submodules. Each module lists every module beneath it that it may extend.
@testitem "layer graph holds" tags = [:checks] begin
    using StrictModeTest
    QE = QuantumEngine
    test_layers(
        QE => (QE.Errors, QE.Units, QE.System),
        QE.Errors => (),
        QE.Units => (),
        QE.System => (QE.Units,),
    )
end

# Package source may read only the `export`ed or `public` names of QuantumEngine's modules.
@testitem "submodules read only each other's public names" tags = [:checks] begin
    using ExplicitImports
    QE = QuantumEngine
    for M in (QE, QE.Errors, QE.Units, QE.System)
        @test isnothing(
            check_all_qualified_accesses_are_public(
                M, pathof(QE); from = (QE,), allow_internal_accesses = false
            )
        )
    end
end

# The calls of the StrictMode workload compile the entry points; no compiled signature of a
# submodule may then carry an unresolved call. The top module owns no functions and is not swept.
@testitem "test_calls sweep finds no unresolved call" tags = [:checks] begin
    using StrictModeTest
    include(joinpath(pkgdir(QuantumEngine), "test", "strict", "workload.jl"))
    run_workload()
    for M in (QuantumEngine.Errors, QuantumEngine.Units, QuantumEngine.System)
        @test length(test_calls(M)) > 0
    end
end

# Every exemption is still needed, has a compiled specialization, and states its reason. Atoms built
# from `String` symbols compile the keyword body of `Atom` that allocates nothing, which the
# exemption of `Atom` must not cover.
@testitem "every exemption is used and reasoned" tags = [:checks] begin
    using StrictMode, StrictModeTest
    include(joinpath(pkgdir(QuantumEngine), "test", "strict", "workload.jl"))
    for T in (Float64, Float32)
        Atom("O", T[0, 0, 0])
        Atom("H", T[0, 0, 1]; mass = 2)
        Atom("O@", T[0, 0, 2])
    end
    run_workload()
    for M in (QuantumEngine.Errors, QuantumEngine.Units, QuantumEngine.System)
        @test isempty(test_exemptions(M))
    end
    @test all(e -> !isempty(e.reason), StrictMode.exemptions())
end
