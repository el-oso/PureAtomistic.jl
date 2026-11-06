@testitem "the PureMLIP extension loads and the fallbacks stay for other backends" tags = [:gpu] setup = [GPUBackend] begin
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    @test ext isa Module
    m = MACEModel(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint()))
    @test_throws "model: device evaluation needs a KernelAbstractions backend" PureMLIP.device_model(m, nothing; elements = [29])
end

@testitem "Aqua ambiguities and piracies with the PureMLIP extension loaded" tags = [:gpu] setup = [GPUBackend] begin
    using Aqua
    @test isa(Base.get_extension(PureMLIP, :PureMLIPGPUExt), Module)
    Aqua.test_ambiguities(PureMLIP)
    Aqua.test_piracies(PureMLIP)
end

@testitem "the PureMLIP extension reads only public names" tags = [:gpu] setup = [GPUBackend] begin
    using ExplicitImports
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    file = joinpath(pkgdir(PureMLIP), "ext", "PureMLIPGPUExt", "PureMLIPGPUExt.jl")
    @test isnothing(check_all_qualified_accesses_are_public(ext, file; from = (PureMLIP,), allow_internal_accesses = false))
end
