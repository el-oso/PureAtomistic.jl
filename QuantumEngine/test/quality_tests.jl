@testitem "ExplicitImports" tags = [:checks] begin
    using ExplicitImports
    @test isnothing(check_no_stale_explicit_imports(QuantumEngine))
    @test isnothing(check_all_explicit_imports_via_owners(QuantumEngine))
    @test isnothing(check_no_self_qualified_accesses(QuantumEngine))
    @test isnothing(
        check_all_qualified_accesses_are_public(
            QuantumEngine, pathof(QuantumEngine); from = (QuantumEngine,), allow_internal_accesses = false
        )
    )
end

@testitem "Aqua" tags = [:checks] begin
    using Aqua
    Aqua.test_all(QuantumEngine)
end

# The package's closure holds no binary besides the standard libraries'.
@testitem "core closure is binary-free" tags = [:checks] begin
    using Pkg, TOML
    root = pkgdir(QuantumEngine)
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

# Libxc is a reference for comparison only; it never enters the package or its test environment.
@testitem "Libxc is not a dependency" tags = [:checks] begin
    using TOML
    root = pkgdir(QuantumEngine)
    for file in ("Project.toml", joinpath("test", "Project.toml"))
        proj = TOML.parsefile(joinpath(root, file))
        for section in ("deps", "weakdeps", "extras", "sources", "compat")
            @test !haskey(get(proj, section, Dict{String, Any}()), "Libxc")
            @test !haskey(get(proj, section, Dict{String, Any}()), "Libxc_jll")
        end
    end
end

# Every module prints through `IO.Logger`, so `src/io/logger.jl` is the only source file that may
# print to stdout.
@testitem "no bare printing to stdout" tags = [:checks] begin
    src = joinpath(pkgdir(QuantumEngine), "src")
    pattern = r"\b(println|print|printstyled)\((\"|lazy\"|stdout)|@show\b"
    hits = String[]
    for (dir, _, files) in walkdir(src), f in files
        endswith(f, ".jl") || continue
        path = joinpath(dir, f)
        relpath(path, src) == joinpath("io", "logger.jl") && continue
        for (n, line) in enumerate(eachline(path))
            occursin(pattern, line) && push!(hits, string(relpath(path, src), ":", n, ": ", line))
        end
    end
    isempty(hits) || foreach(println, hits)
    @test isempty(hits)
end
