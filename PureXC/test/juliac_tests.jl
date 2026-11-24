# Builds `xc_eval` with juliac --trim=safe (Julia 1.13); it evaluates `pbe()` with `xc_evaluate!` at
# order 2, restricted and polarized, and its output file must equal that of the untrimmed
# `juliac/entry.jl` bit for bit. Slow, so it runs only with PUREXC_JULIAC_BUILD=1 (see
# test/runtests.jl).
@testitem "juliac --trim build of xc_eval" tags = [:juliac] begin
    using PureXC
    root = pkgdir(PureXC)
    exe = joinpath(root, "juliac", "build", "bin", "xc_eval" * (Sys.iswindows() ? ".exe" : ""))
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

    ref, out = joinpath(dir, "ref.bin"), joinpath(dir, "bin.bin")
    entry = joinpath(root, "juliac", "entry.jl")
    @test success(addenv(`$(Base.julia_cmd()) --startup-file=no --project=$root $entry $ref`, env...))
    errf, outf = joinpath(dir, "stderr.txt"), joinpath(dir, "stdout.txt")
    # 64 points: 6 restricted fields of one row and 21 polarized rows, in Float64.
    @test filesize(ref) == 64 * (6 + 21) * 8
    # The binary writes the same file with nothing on stdout or stderr, under the default load path
    # and under one whose load path and active project name StrictMode (the test environment),
    # which makes a StrictMode built with its banner on print it.
    testenv = joinpath(root, "test")
    strictmode_env = ("JULIA_LOAD_PATH" => string("@:", testenv, ":@stdlib"), "JULIA_PROJECT" => testenv)
    for e in (env, strictmode_env)
        rm(out; force = true)
        @test success(pipeline(addenv(`$exe $out`, e...); stdout = outf, stderr = errf))
        @test isempty(read(outf, String))
        @test isempty(read(errf, String))
        @test read(out) == read(ref)
    end
    # A wrong argument count exits 1 with the usage line.
    p = run(pipeline(ignorestatus(addenv(`$exe`, env...)); stderr = errf))
    @test p.exitcode == 1
    @test startswith(read(errf, String), "usage: xc_eval")
end
