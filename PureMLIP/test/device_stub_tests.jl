@testitem "device entry points without KernelAbstractions name what to load" begin
    m = MACEModel(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint()))
    msg = "model: device evaluation needs a KernelAbstractions backend"
    nl = neighbor_list([0.0 1.5; 0.0 0.0; 0.0 0.0], [6.0 0.0 0.0; 0.0 6.0 0.0; 0.0 0.0 6.0], 4.0)
    @test_throws msg PureMLIP.device_model(m, nothing; elements = [29])
    @test_throws "got Nothing" PureMLIP.device_model(m, nothing; elements = [29])
    @test_throws msg PureMLIP.device_neighbor_list(nl, nothing)
    @test_throws msg PureMLIP.device_neighbor_list([0.0 1.5; 0.0 0.0; 0.0 0.0], [6.0 0.0 0.0; 0.0 6.0 0.0; 0.0 0.0 6.0], 4.0)
    @test_throws msg PureMLIP.device_workspace(m, 2, 4; edge_chunk = 8)
    @test_throws msg PureMLIP.device_result(m, 2)
    @test isnothing(Base.get_extension(PureMLIP, :PureMLIPGPUExt))
end

# The core closure holds no binary besides the standard libraries': KernelAbstractions and the
# `_jll` packages it brings enter only through the weak dependency.
@testitem "core closure is binary-free" tags = [:checks] begin
    using Pkg, TOML
    root = dirname(dirname(pathof(PureMLIP)))
    proj = TOML.parsefile(joinpath(root, "Project.toml"))
    @test haskey(proj["weakdeps"], "KernelAbstractions")
    @test !haskey(proj["deps"], "KernelAbstractions")
    @test proj["extensions"]["PureMLIPGPUExt"] == "KernelAbstractions"
    # A fresh environment holding only this package and its dependencies.
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
