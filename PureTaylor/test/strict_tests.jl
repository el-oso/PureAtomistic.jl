# `test/strict/check.jl` in a fresh process, once after the package workload and once after the
# failing control's workload. Its output lines are described at the top of that file.
@testmodule StrictRun begin
    using PureTaylor
    function run_check(args...)
        script = joinpath(pkgdir(PureTaylor), "test", "strict", "check.jl")
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $script $args`
        return read(addenv(cmd, "JULIA_LOAD_PATH" => join(LOAD_PATH, Sys.iswindows() ? ";" : ":")), String)
    end
    checked(out) = Dict((m[1], m[2]) => parse(Int, m[3]) for m in eachmatch(r"^CHECKED (\w+) (\w+) (\d+)$"m, out))
    # The gates: each engine with the guarantees it judges.
    const GATES = (
        ("proof", "typestable"), ("proof", "noalloc"), ("proof", "noboxing"),
        ("scan", "noalloc"), ("scan", "noboxing"),
        ("other", "typestable"), ("other", "noalloc"), ("other", "noboxing"),
    )
    const OTHER = (
        "+", "-", "*", "/", "inv", "^", "literal_pow", "sqrt", "cbrt", "exp", "expm1", "log", "log1p",
        "zero", "one", "isfinite", "isnan",
    )
end

@testitem "StrictMode policy holds in PureTaylor" tags = [:checks] setup = [StrictRun] begin
    using .StrictRun: run_check, checked, GATES, OTHER
    out = run_check()
    failures = filter(startswith("FAIL "), split(out, '\n'))
    isempty(failures) || foreach(println, failures)
    @test isempty(failures)
    @test occursin(r"^POLICY typestable,noalloc$"m, out)
    n = checked(out)
    for gate in GATES
        # Each gate judged something.
        @test get(n, gate, 0) > 0
    end
    # Every Base function the package extends had a specialization judged.
    judged = Set(m[1] for m in eachmatch(r"^JUDGED other (\S+)$"m, out))
    @test judged == Set(OTHER)
end

# Coefficients of the abstract type `Real` are type-unstable and boxed, so each gate must report a
# finding of its own guarantee.
@testitem "StrictMode gates reject the failing control" tags = [:checks] setup = [StrictRun] begin
    using .StrictRun: run_check, GATES
    out = run_check("control")
    for (engine, g) in GATES
        @test occursin(Regex("^FAIL $engine $g ", "m"), out)
    end
    @test !occursin(r"^FAIL unlisted"m, out)
end
