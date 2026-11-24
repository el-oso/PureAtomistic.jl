# `test/strict/check.jl` in a fresh process, after the failing control's workload. Its output lines
# are described at the top of that file.
@testmodule StrictRun begin
    using PureXC
    function run_check(args...)
        script = joinpath(pkgdir(PureXC), "test", "strict", "check.jl")
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $script $args`
        return read(addenv(cmd, "JULIA_LOAD_PATH" => join(LOAD_PATH, Sys.iswindows() ? ";" : ":")), String)
    end
    # The gates: each engine with the guarantees it judges.
    const GATES = (
        ("proof", "typestable"), ("proof", "noalloc"), ("proof", "noboxing"),
        ("scan", "noalloc"), ("scan", "noboxing"),
        ("other", "typestable"), ("other", "noalloc"), ("other", "noboxing"),
    )
end

# A functional with the abstract number type `Real`, and the control's method of the listed
# `interface_trait` on thresholds of type `Real`, are type-unstable and boxed, so each gate must
# report a finding whose own guarantee is the gate's (`[✗ <guarantee> <code>]` in the finding's
# text). The control's method of `abs2`, a Base function the listing does not name, must be
# reported unlisted.
@testitem "StrictMode gates reject the failing control" tags = [:checks] setup = [StrictRun] begin
    using .StrictRun: run_check, GATES
    out = run_check("control")
    for (engine, g) in GATES
        @test occursin(Regex("^FAIL $engine $g .*\\[✗ $g SM\\d{4}\\]", "m"), out)
    end
    @test occursin(r"^FAIL unlisted other Tuple\{typeof\(abs2\), "m, out)
end
