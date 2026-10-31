# Measure-only: wall-clock budgets depend on the host, so this reports and never fails on
# time. A budget belongs on a pinned host (see AGENTS.md).
@testitem "cold-start latency report" tags = [:latency] begin
    using StrictModeTest
    calls = [
        "PureMLIP.spherical_harmonics!(zeros(16), [0.1, 0.2, 0.9], 3)",
        "PureMLIP.e3nn_wigner_3j(1, 2, 3)",
    ]
    r = test_latency(:PureMLIP; calls, precompile = false)
    # first_call_ms under-measures on Julia 1.13: it does not include first-call compilation,
    # so the figures are indicative only (see the latency note in AGENTS.md).
    @info "PureMLIP cold-start latency" r
    @test r.target == "PureMLIP"
    @test isfinite(r.load_ms) && r.load_ms > 0
    @test length(r.first_call_ms) == length(calls)
end
