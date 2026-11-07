@testitem "the PureAtomistic extension loads and the fallbacks stay for other backends" tags = [:gpu] setup = [GPUBackend] begin
    using PureMLIP
    ext = Base.get_extension(PureAtomistic, :PureAtomisticGPUExt)
    @test ext isa Module
    m = MACEModel(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint()))
    sys = PureAtomistic.Systems.fcc_cell(3.6, 2, :Cu)
    @test_throws "pureatomistic: device evaluation needs a KernelAbstractions backend" PureAtomistic.to_device(sys, nothing)
    # A host system is a system on the KernelAbstractions CPU backend; a model of another precision
    # than the system's stays with the core's method.
    @test_throws "pureatomistic: the model has element type Float32 and the system Float64; they must match" PureAtomistic.device_potential(
        MACEModel{Float32}(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint())), sys; skin = 0.5
    )
end

@testitem "Aqua ambiguities and piracies with both extensions loaded" tags = [:gpu] setup = [GPUBackend] begin
    using PureMLIP
    using Aqua
    @test isa(Base.get_extension(PureAtomistic, :PureAtomisticGPUExt), Module)
    @test isa(Base.get_extension(PureMLIP, :PureMLIPGPUExt), Module)
    for M in (PureAtomistic, PureMLIP)
        Aqua.test_ambiguities(M)
        Aqua.test_piracies(M)
    end
end

@testitem "the PureAtomistic extension reads only public names" tags = [:gpu] setup = [GPUBackend] begin
    using PureMLIP
    using ExplicitImports
    ext = Base.get_extension(PureAtomistic, :PureAtomisticGPUExt)
    file = joinpath(pkgdir(PureAtomistic), "ext", "PureAtomisticGPUExt", "PureAtomisticGPUExt.jl")
    @test isnothing(
        check_all_qualified_accesses_are_public(ext, file; from = (PureAtomistic, PureMLIP), allow_internal_accesses = false)
    )
end
