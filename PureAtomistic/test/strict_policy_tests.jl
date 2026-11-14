# The StrictMode policy `(:typestable, :noalloc)`: `@strict_policy` in the module source declares
# it and checks the concrete signatures at precompile time; this item proves it for every
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
    checked = Dict(m[1] => (parse(Int, m[2]), parse(Int, m[3])) for m in eachmatch(r"^CHECKED (\w+) (\d+) (\d+)$"m, out))
    for M in (P, P.Systems, P.Potentials, P.Integrators, P.Thermostats, P.Barostats, P.IO, P.Analysis)
        name = string(nameof(M))
        # Each module's sweep judged something, in both guarantees.
        @test all(>(0), checked[name])
    end
    for name in ("Systems", "IO", "Analysis")
        @test declared[name] == "typestable,noalloc"
    end
end

# `@contract` defines a method of `TypeContracts._contract_specs` in the module that declares the
# contract, and it allocates its specs on every call, so a `@strict_policy (…, :noalloc)` placed
# after a contract warns at every precompile. Potentials, Integrators, Thermostats, Barostats and
# the top module (whose policy check covers its submodules) declare contracts and carry no policy
# until TypeContracts generates the specs without allocating (see
# https://github.com/el-oso/TypeContracts.jl/issues/4). Their code is held to the policy by the
# item above all the same. When this item reports an unexpected pass, add
# `@strict_policy (:typestable, :noalloc)` to those modules and turn it into a `@test`.
@testitem "TypeContracts specs pass the allocation scan" tags = [:checks] begin
    using StrictMode, TypeContracts
    fs = StrictMode.findings(
        TypeContracts._contract_specs, (Type{PureAtomistic.Potentials.AbstractPotential},); guarantees = (:noalloc,)
    )
    @test_broken all(f -> f.status !== :fail, fs)
end

# One exemption covers the keyword constructors of the parametric types, because Julia names the
# keyword body of `T{…}(…; kw…)` `#_#N`, which StrictMode reads as the function `_`. A callable
# object with keyword arguments has a keyword body of the same name, which the exemption would
# hide as well: none may exist.
@testitem "the exemption of `_` covers only constructors" tags = [:checks] begin
    using StrictMode
    ms, _ = StrictMode.module_methods(PureAtomistic; include_exempt = true)
    underscored = filter(m -> StrictMode._demangle(m.name) === :_, ms)
    @test !isempty(underscored)
    for m in underscored
        params = Base.unwrap_unionall(m.sig).parameters
        @test any(p -> p isa Type && p <: Type, params[2:end])
    end
end
