@testitem "device entry points without KernelAbstractions name what to load" setup = [SystemFixtures] begin
    using PureMLIP
    m = MACEModel(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint()))
    sys = SystemFixtures.small_system(Float64)
    msg = "pureatomistic: device evaluation needs a KernelAbstractions backend"
    @test_throws msg PureAtomistic.device_potential(m, sys; skin = 0.5)
    @test_throws "pureatomistic: the model has element type Float64 and the system Float32; they must match" PureAtomistic.device_potential(
        m, SystemFixtures.small_system(Float32); skin = 0.5
    )
    @test_throws msg PureAtomistic.to_device(sys, nothing)
    @test_throws "got Nothing" PureAtomistic.to_device(sys, nothing)
    @test isnothing(Base.get_extension(PureAtomistic, :PureAtomisticGPUExt))
end

@testitem "copy_state! round-trips Array, OffsetArray and view systems" setup = [SystemFixtures] begin
    using OffsetArrays, PureAtomistic.Systems
    src = SystemFixtures.small_system(Float64)
    # An OffsetArray destination: the sizes match, the axes do not.
    P = OffsetArray(zeros(3, 4), 0:2, 5:8); V = OffsetArray(zeros(3, 4), 0:2, 5:8)
    m = OffsetArray(copy(src.masses), 5:8); z = OffsetArray(copy(src.species), 5:8)
    off = System(P, V, m, z, zeros(3, 3))
    @test copy_state!(off, src) === off
    @test collect(off.positions) == src.positions && collect(off.velocities) == src.velocities
    @test off.cell == src.cell
    # A destination made of views of larger buffers.
    bp = zeros(3, 6); bv = zeros(3, 6); bc = zeros(4, 4)
    vw = System(view(bp, :, 2:5), view(bv, :, 2:5), copy(src.masses), copy(src.species), view(bc, 1:3, 1:3))
    copy_state!(vw, off)
    @test bp[:, 2:5] == src.positions && bv[:, 2:5] == src.velocities && bc[1:3, 1:3] == src.cell
    @test all(iszero, bp[:, [1, 6]]) && all(iszero, bc[4, :]) && all(iszero, bc[:, 4])
    back = SystemFixtures.small_system(Float64); back.positions .= 0; back.velocities .= 0; back.cell .= 0
    copy_state!(back, vw)
    @test back.positions == src.positions && back.velocities == src.velocities && back.cell == src.cell
    # Masses and species stay as they were.
    @test back.masses == src.masses && back.species == src.species
    other = System(zeros(3, 2), zeros(3, 2), [1.0, 1.0], [1, 1], SystemFixtures.CELL)
    @test_throws "positions have size (3, 2) and (3, 4)" copy_state!(other, src)
end

@testitem "copy_state! is allocation-free and trims" tags = [:checks] setup = [SystemFixtures] begin
    using StrictModeTest, StrictMode, TrimCheck, PureAtomistic.Systems
    @noinline cp!(d, s) = copy_state!(d, s)
    for T in (Float64, Float32)
        a = SystemFixtures.small_system(T); b = SystemFixtures.small_system(T)
        cp!(b, a)
        @test_noalloc cp!(b, a)
    end
    @validate(
        init = begin
            using PureAtomistic.Systems
            cp64(d, s) = copy_state!(d, s)
            cp32(d, s) = copy_state!(d, s)
            const S64 = System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}
            const S32 = System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}
        end,
        cp64(S64, S64),
        cp32(S32, S32),
    )
end

# The core closure holds no binary besides the standard libraries': KernelAbstractions and the
# `_jll` packages it brings enter only through the weak dependency.
@testitem "core closure is binary-free" tags = [:checks] begin
    using Pkg, TOML
    root = dirname(dirname(pathof(PureAtomistic)))
    mlip = joinpath(dirname(root), "PureMLIP")
    for pkg in (root, mlip)
        proj = TOML.parsefile(joinpath(pkg, "Project.toml"))
        @test haskey(proj["weakdeps"], "KernelAbstractions")
        @test !haskey(proj["deps"], "KernelAbstractions")
        @test haskey(proj["extensions"], basename(pkg) * "GPUExt")
    end
    # A fresh environment holding only the two packages and their dependencies.
    script = """
    using Pkg, TOML
    Pkg.activate(temp = true)
    Pkg.develop([PackageSpec(path = ARGS[1]), PackageSpec(path = ARGS[2])])
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
            `$(Base.julia_cmd()) --startup-file=no -e $script $root $mlip`,
            merge(ENV, Dict("JULIA_PKG_PRECOMPILE_AUTO" => "0"))
        ), String
    )
    @test occursin(r"^DONE \d+$"m, out)
    @test !occursin(r"^(JLL|ARTIFACTS) "m, out)
end
