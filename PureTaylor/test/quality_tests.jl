@testitem "ExplicitImports" tags = [:checks] begin
    using ExplicitImports
    @test isnothing(check_no_stale_explicit_imports(PureTaylor))
    @test isnothing(check_all_explicit_imports_via_owners(PureTaylor))
    @test isnothing(check_no_self_qualified_accesses(PureTaylor))
    @test isnothing(
        check_all_qualified_accesses_are_public(PureTaylor, pathof(PureTaylor); from = (PureTaylor,), allow_internal_accesses = false)
    )
end

@testitem "Aqua" tags = [:checks] begin
    using Aqua
    Aqua.test_all(PureTaylor)
end

# The package's closure holds no binary besides the standard libraries'.
@testitem "core closure is binary-free" tags = [:checks] begin
    using Pkg, TOML
    root = dirname(dirname(pathof(PureTaylor)))
    script = """
    using Pkg, TOML
    Pkg.activate(temp = true)
    Pkg.develop(PackageSpec(path = ARGS[1]))
    Pkg.instantiate()
    man = TOML.parsefile(joinpath(dirname(Base.active_project()), "Manifest.toml"))["deps"]
    std = Set(string(u) for u in keys(Pkg.Types.stdlibs()))
    for (name, entries) in man, e in entries
        e["uuid"] in std && continue
        endswith(name, "_jll") && println("JLL ", name)
        dir = dirname(dirname(Base.locate_package(Base.PkgId(Base.UUID(e["uuid"]), name))))
        isfile(joinpath(dir, "Artifacts.toml")) && println("ARTIFACTS ", name)
    end
    println("DONE ", length(man))
    """
    out = read(
        setenv(
            `$(Base.julia_cmd()) --startup-file=no -e $script $root`,
            merge(ENV, Dict("JULIA_PKG_PRECOMPILE_AUTO" => "0"))
        ), String
    )
    @test occursin(r"^DONE \d+$"m, out)
    @test !occursin(r"^(JLL|ARTIFACTS) "m, out)
end
