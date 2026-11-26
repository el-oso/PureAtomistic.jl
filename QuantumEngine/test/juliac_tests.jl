# Builds `qe_inspect` with juliac --trim=safe (Julia 1.13) and requires its output to equal that of
# the untrimmed `juliac/entry.jl` byte for byte, with nothing on stderr: without an argument (exit
# 1, the usage line), on `examples/h2.inp` (exit 0, the five lines of the molecule), on an input
# with no atoms (exit 1, the error) and on a missing file (exit 1). Slow, so it runs only with
# QUANTUMENGINE_JULIAC_BUILD=1 (see test/runtests.jl).
@testitem "juliac --trim build of qe_inspect" tags = [:juliac] setup = [Fixtures] begin
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

    h2 = joinpath(dir, "h2.inp")
    write(h2, Fixtures.oracle_input("examples/h2.inp"))
    bad = joinpath(dir, "bad.inp")
    write(bad, "charge 0\n")
    missing_file = joinpath(dir, "missing.inp")
    h2_out = join(["Hydrogen molecule", "2", "1", "1", QuantumEngine.IO.fmt_fixed(1 / 1.4, 20, 12)], "\n") * "\n"
    entry = joinpath(root, "juliac", "entry.jl")
    errf, outf = joinpath(dir, "stderr.txt"), joinpath(dir, "stdout.txt")
    # The untrimmed run may print the StrictMode banner to stderr; only the binary must print nothing
    # there.
    for (args, code, expected) in (
            ((), 1, "usage: qe_inspect <file.inp>\n"), ((h2,), 0, h2_out),
            ((bad,), 1, "input: $bad: no atoms: the input has no geometry block\n"),
            ((missing_file,), 1, "qe_inspect: cannot read $missing_file\n"),
        )
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
