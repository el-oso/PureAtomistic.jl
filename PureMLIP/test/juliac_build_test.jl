# Runs the juliac --trim build and the resulting executable on MP-0b3. Slow, so it is skipped
# unless PUREMLIP_JULIAC_BUILD=1 (see test/runtests.jl). Unlike TrimCheck's @validate, which
# infers each entry point in isolation, the build is the whole-program trim verification.
@testitem "juliac --trim build of mace_inspect" tags = [:juliac] setup = [Checkpoints] begin
    root = pkgdir(PureMLIP)
    buildjl = joinpath(root, "juliac", "build.jl")
    exe = joinpath(root, "juliac", "build", "bin", "mace_inspect" * (Sys.iswindows() ? ".exe" : ""))
    # A ReTestItems/TestItemRunner worker restricts JULIA_LOAD_PATH; the juliac subprocess
    # needs the default path (including @stdlib).
    cmd = addenv(
        `$(Base.julia_cmd()) --startup-file=no --project=$root $buildjl`,
        "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing
    )
    @test success(pipeline(cmd; stdout = stderr, stderr = stderr))
    @test isfile(exe)

    out = read(`$exe $(Checkpoints.fetch(:mp0b3))`, String)
    @test occursin("model_class=mace.modules.models.ScaleShiftMACE\n", out)
    @test occursin("state_tensors=89\n", out)
    @test occursin("num_parameters=9063204\n", out)

    err = IOBuffer()
    @test !success(pipeline(ignorestatus(`$exe`); stderr = err))
    @test occursin("usage:", String(take!(err)))
end
