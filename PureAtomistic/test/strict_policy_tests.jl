# The StrictMode policy `(:typestable, :noalloc, :noboxing)`: `@strict_policy` in the module source
# declares it and checks the concrete signatures at precompile time; this item proves it for every
# specialization a fixed workload compiles. The workload runs in a fresh process, so the
# specializations judged are the package's own and none of a test fixture's.
@testitem "StrictMode policy holds in every module" tags = [:checks] begin
    script = joinpath(pkgdir(PureAtomistic), "test", "strict", "check.jl")
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $script`
    out = read(addenv(cmd, "JULIA_LOAD_PATH" => join(LOAD_PATH, Sys.iswindows() ? ";" : ":")), String)
    failures = filter(startswith("FAIL "), split(out, '\n'))
    isempty(failures) || foreach(println, failures)
    @test isempty(failures)
    P = PureAtomistic
    declared = Dict(m[1] => m[2] for m in eachmatch(r"^POLICY (\w+) (\S+)$"m, out))
    checked = Dict(m[1] => (parse(Int, m[2]), parse(Int, m[3]), parse(Int, m[4])) for m in eachmatch(r"^CHECKED (\w+) (\d+) (\d+) (\d+)$"m, out))
    for M in (P, P.Systems, P.Potentials, P.Integrators, P.Thermostats, P.Barostats, P.IO, P.Analysis)
        name = string(nameof(M))
        # Each module's sweep judged something, in all three guarantees.
        @test all(>(0), checked[name])
    end
    for name in ("PureAtomistic", "Systems", "Potentials", "Integrators", "Thermostats", "Barostats", "IO", "Analysis")
        @test declared[name] == "typestable,noalloc,noboxing"
    end
    # `relax!` is judged, not exempt: its keyword bodies with each relaxer state and the one that
    # builds the default state (whose signature ends in the potential) pass the scan.
    judged = [(m[1], m[2]) for m in eachmatch(r"^JUDGED (\w+) Integrators (.+)$"m, out)]
    @test !isempty(judged) && all(j -> j[1] == "pass", judged)
    for opt in ("FIRE{", "FIRE2{", "TrustRegionBFGS{", "TrustRegionLBFGS{")
        @test any(j -> occursin(opt, j[2]), judged)
    end
    @test any(j -> occursin(r"(LennardJones|MACEPotential)\{Float\d+\}\)$", j[2]), judged)
    # The methods the module sweeps cannot see were judged as well.
    m = match(r"^CHECKED OTHER (\d+)$"m, out)
    @test !isnothing(m) && parse(Int, m[1]) > 0
end

# `@contract` defines a method of `TypeContracts._contract_specs` in the module that declares the
# contract. Potentials, Integrators, Thermostats, Barostats and the top module declare contracts and
# carry `@strict_policy (:typestable, :noalloc, :noboxing)`, so the generated specs must allocate nothing. This
# item checks that they do.
@testitem "TypeContracts specs pass the allocation scan" tags = [:checks] begin
    using StrictMode, TypeContracts
    fs = StrictMode.findings(
        TypeContracts._contract_specs, (Type{PureAtomistic.Potentials.AbstractPotential},); guarantees = (:noalloc,)
    )
    @test all(f -> f.status !== :fail, fs)
end

# One exemption covers the keyword constructors of the parametric types, because Julia names the
# keyword body of `T{…}(…; kw…)` `#_#N`, which StrictMode reads as the function `_`. A callable
# object with keyword arguments has a keyword body of the same name, which the exemption would
# hide as well: none may exist. Every method named `_` must be the keyword body of methods whose
# callee is a type, constructors.
@testitem "the exemption of `_` covers only constructors" tags = [:checks] begin
    using StrictMode
    callee(m) = Base.unwrap_unionall(m.sig).parameters[1]
    leaves(x) = x isa Expr ? mapreduce(leaves, vcat, x.args; init = Any[]) : Any[x]
    value(x) = x isa GlobalRef ? (isdefined(x.mod, x.name) ? getglobal(x.mod, x.name) : nothing) :
        x isa QuoteNode ? x.value : nothing
    within(m, M) = m === M || (parentmodule(m) !== m && within(parentmodule(m), M))
    # For each method of `M` that StrictMode names `_`, whether every keyword method that calls it
    # as its body has a type as its callee. A keyword method is a method of `Core.kwcall` whose
    # third parameter is the callee; its lowered code names the body by a global or, for an inner
    # constructor, by a quoted instance.
    function constructor_bodies(M::Module)
        bodies = Dict{Any, Vector{Any}}()
        for K in methods(Core.kwcall)
            within(K.module, M) || continue
            for st in Base.uncompressed_ir(K).code, x in leaves(st)
                f = value(x)
                (f isa Function && StrictMode._demangle(nameof(f)) === :_) &&
                    push!(get!(bodies, typeof(f), Any[]), Base.unwrap_unionall(K.sig).parameters[3])
            end
        end
        ms, _ = StrictMode.module_methods(M; include_exempt = true)
        return [
            haskey(bodies, callee(m)) && all(c -> c <: Type, bodies[callee(m)])
                for m in ms if StrictMode._demangle(m.name) === :_
        ]
    end
    verdicts = constructor_bodies(PureAtomistic)
    @test !isempty(verdicts) && all(verdicts)
    # A callable object with keywords and a type argument is told apart from a constructor.
    M = Module(:CallableKeywords)
    Core.eval(M, :(struct C <: Function end))
    Core.eval(M, :((c::C)(x::Type; k = 1) = k))
    Core.eval(
        M, :(
            struct P{T}
                a::T
            end
        )
    )
    Core.eval(M, :(P{T}(x; k = 1) where {T} = P{T}(T(x + k))))
    @test sort(constructor_bodies(M)) == [false, true]
end
