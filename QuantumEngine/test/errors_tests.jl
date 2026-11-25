@testitem "error messages carry their prefix" begin
    using QuantumEngine
    @test_throws "input: a.inp:3: unknown keyword `foo`" throw(InputError("a.inp", 3, "unknown keyword `foo`"))
    @test_throws "input: a.inp:4: x = 2" throw(InputError("a.inp", 4, lazy"x = $(1 + 1)"))
    @test InputError("a.inp", 3, "m").msg isa LazyString
    @test_throws "input: a.json: scf.max_iter: expected an integer" throw(InputError("a.json", 0, "scf.max_iter: expected an integer"))
    @test_throws "basis: no shells for Xe" throw(BasisError("no shells for Xe"))
    @test_throws "scf: not converged after 50 iterations" throw(SCFNotConverged("not converged after 50 iterations"))
    @test_throws "optimization: " throw(OptimizationFailed("x"))
    @test_throws "not implemented (09-post-hf §CCSD): x" throw(NotImplemented("09-post-hf §CCSD", "x"))
    @test_throws "functional: " throw(FunctionalUnavailable("x"))
    @test_throws "numerical: " throw(NumericalError("x"))
    @test all(T -> T <: QuantumEngineError, (InputError, BasisError, SCFNotConverged, OptimizationFailed, NotImplemented, FunctionalUnavailable, NumericalError))
end

@testitem "NotImplemented is not TypeContracts' trait" begin
    using QuantumEngine, TypeContracts
    @test QuantumEngine.NotImplemented !== TypeContracts.NotImplemented
end

@testitem "InputWarning shows as the InputError of its source, line and message" begin
    using QuantumEngine
    @test sprint(show, InputWarning("a.inp", 3, :unknown_keyword, "unknown keyword `foo`")) == "input: a.inp:3: unknown keyword `foo`"
    @test sprint(show, InputWarning("a.json", 0, :ignored_key, lazy"x = $(1 + 1)")) == "input: a.json: x = 2"
    @test InputWarning("a.inp", 3, :c, "m").msg isa LazyString
end

@testitem "messages print every part type they interpolate" begin
    using QuantumEngine
    sub = SubString("abc", 2, 3)
    t, c = "t", 'c'
    m = lazy"s $(t) $(:sym) $(c) $(true) $(3) $(2.5) $(sub)"
    @test sprint(showerror, BasisError(m)) == "basis: s t sym c true 3 2.5 bc"
    @test sprint(showerror, NumericalError(LazyString(lazy"x $(1)"))) == "numerical: x 1"
    @test_throws "errors: a message part is not a String, LazyString, SubString, Symbol, Char, Bool, Int or Float64" sprint(showerror, BasisError(lazy"x $(1 // 2)"))
    @test_throws "errors: a message part is not" sprint(show, InputWarning("a.inp", 1, :c, lazy"x $(Int32(1))"))
end
