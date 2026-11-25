# Builds `qe_inspect` with juliac --trim=safe (Julia 1.13) and requires its output to equal that of
# the untrimmed `juliac/entry.jl` byte for byte, with nothing on stderr, without an argument (exit
# 1, the usage line) and with one (exit 0). Slow, so it runs only with QUANTUMENGINE_JULIAC_BUILD=1
# (see test/runtests.jl).
@testitem "juliac --trim build of qe_inspect" tags = [:juliac] begin
    using QuantumEngine
    root = pkgdir(QuantumEngine)
    exe = joinpath(root, "juliac", "build", "bin", "qe_inspect" * (Sys.iswindows() ? ".exe" : ""))
    # A TestItemRunner worker restricts JULIA_LOAD_PATH; the juliac subprocess needs the default
    # path (including @stdlib).
    env = ("JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing)
    cmd = addenv(`$(Base.julia_cmd()) --startup-file=no --project=$root $(joinpath(root, "juliac", "build.jl"))`, env...)
    dir = mktempdir()
    # The trim verifier reports its findings as "Error #N: unresolved call ..." lines but still
    # writes the executable, so the build log is the only place a finding shows. It must have none.
    build_log = joinpath(dir, "build.log")
    @test success(pipeline(cmd; stdout = build_log, stderr = build_log))
    print(stderr, read(build_log, String))
    @test isempty(filter(l -> occursin("unresolved call", l) || occursin("Error #", l), readlines(build_log)))
    @test isfile(exe)
    @test !isfile(joinpath(root, "LocalPreferences.toml"))

    entry = joinpath(root, "juliac", "entry.jl")
    errf, outf = joinpath(dir, "stderr.txt"), joinpath(dir, "stdout.txt")
    # The untrimmed run may print the StrictMode banner to stderr; only the binary must print nothing
    # there.
    for (args, code, expected) in (((), 1, "usage: qe_inspect <file.inp>\n"), (("x.inp",), 0, "ok\n"))
        p = run(pipeline(ignorestatus(addenv(`$(Base.julia_cmd()) --startup-file=no --project=$root $entry $args`, env...)); stdout = outf, stderr = errf))
        ref = read(outf, String)
        @test p.exitcode == code
        @test ref == expected
        p = run(pipeline(ignorestatus(addenv(`$exe $args`, env...)); stdout = outf, stderr = errf))
        @test p.exitcode == code
        @test read(outf, String) == ref
        @test isempty(read(errf, String))
    end
end
