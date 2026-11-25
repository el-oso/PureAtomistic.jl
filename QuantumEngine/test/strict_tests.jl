# `test/strict/check.jl` in a fresh process, once after the package workload and once after the
# failing control's workload. Its output lines are described at the top of that file.
@testmodule StrictRun begin
    using QuantumEngine
    function run_check(args...)
        script = joinpath(pkgdir(QuantumEngine), "test", "strict", "check.jl")
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $script $args`
        return read(addenv(cmd, "JULIA_LOAD_PATH" => join(LOAD_PATH, Sys.iswindows() ? ";" : ":")), String)
    end
    checked(out) = Dict((m[1], m[2]) => parse(Int, m[3]) for m in eachmatch(r"^CHECKED (\w+) (\w+) (\d+)$"m, out))
    # The gates: each engine with the guarantees it judges. Every function of another package that
    # the package extends is listed for `:typestable` only.
    const GATES = (
        ("proof", "typestable"), ("proof", "noalloc"), ("proof", "noboxing"),
        ("scan", "noalloc"), ("scan", "noboxing"),
        ("other", "typestable"),
    )
    const OTHER = ("showerror", "show")
    const MODULES = ("Errors",)
end

@testitem "StrictMode policy holds in QuantumEngine" tags = [:checks] setup = [StrictRun] begin
    using .StrictRun: run_check, checked, GATES, OTHER, MODULES
    out = run_check()
    failures = filter(startswith("FAIL "), split(out, '\n'))
    isempty(failures) || foreach(println, failures)
    @test isempty(failures)
    for M in MODULES
        @test occursin(Regex("^POLICY $M typestable,noalloc\$", "m"), out)
    end
    n = checked(out)
    for gate in GATES
        # Each gate judged something, summed over the swept modules. Every function in Errors is an
        # exempted entry point, so the noalloc scan judges nothing until a module with
        # allocation-free functions is swept.
        if gate == ("scan", "noalloc")
            @test_broken get(n, gate, 0) > 0
        else
            @test get(n, gate, 0) > 0
        end
    end
    # Every function of another package that the package extends had a specialization judged.
    judged = Set(m[1] for m in eachmatch(r"^JUDGED other (\S+)$"m, out))
    @test judged == Set(OTHER)
end

# The control's function of `Errors` on a struct with a field of the abstract type `Real` is
# type-unstable, allocates and boxes, and its `showerror` method is type-unstable, so each gate
# must report a finding whose own guarantee is the gate's (`[✗ <guarantee> <code>]` in the
# finding's text). The control's method of `abs2`, a Base function the listing does not name, must
# be reported unlisted.
@testitem "StrictMode gates reject the failing control" tags = [:checks] setup = [StrictRun] begin
    using .StrictRun: run_check, GATES
    out = run_check("control")
    for (engine, g) in GATES
        @test occursin(Regex("^FAIL $engine $g .*\\[✗ $g SM\\d{4}\\]", "m"), out)
    end
    @test occursin(r"^FAIL unlisted other Tuple\{typeof\(abs2\), "m, out)
end
