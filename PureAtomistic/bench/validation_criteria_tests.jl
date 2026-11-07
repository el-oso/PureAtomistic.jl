# Unit tests of validation_criteria.jl:
#   julia --project=PureAtomistic/bench PureAtomistic/bench/validation_criteria_tests.jl
using Test, Random
include(joinpath(@__DIR__, "validation_criteria.jl"))

nve(; drift = 1.0e-5, rms = 1.0e-5, status = "completed", E = [0.0, 1.0]) = Dict{String, Any}(
    "dt_fs" => 2.0, "status" => status, "drift_eV_per_atom_per_ps" => drift,
    "rms_residual_eV_per_atom" => rms, "total_eV_per_atom" => E,
)

@testset "binomial_exact_p" begin
    @test binomial_exact_p(1, 5, 0.5) ≈ 0.375
    @test binomial_exact_p(0, 5, 0.5) ≈ 0.0625
    @test binomial_exact_p(5, 5, 0.5) ≈ 0.0625
    @test binomial_exact_p(3, 6, 0.5) ≈ 1.0
    @test binomial_exact_p(1, 2, 0.2) ≈ 0.36  # outcomes no likelier than 1: P(1) = 0.32 and P(2) = 0.04
    @test binomial_exact_p(2, 2, 0.2) ≈ 0.04
    @test_throws "outside 0:5" binomial_exact_p(6, 5, 0.5)
    @test_throws "outside (0, 1)" binomial_exact_p(1, 5, 1.0)
end

@testset "criteria NVE" begin
    @test criteria(nve())["pass"]
    c = criteria(nve(rms = 2 * STABLE_RMS))
    @test !c["pass"] && !c["rms"] && c["drift"] && c["completed"] && c["finite"]
    c = criteria(nve(drift = -2 * STABLE_DRIFT))
    @test !c["pass"] && !c["drift"] && c["rms"]
    @test !criteria(nve(status = "failed"))["pass"]
    @test !criteria(nve(E = [0.0, Inf]))["finite"]
    @test criteria(nve(drift = STABLE_DRIFT, rms = STABLE_RMS))["pass"]
    @test !criteria(Dict{String, Any}("dt_fs" => 2.0, "status" => "completed", "total_eV_per_atom" => [0.0]))["pass"]
    @test_throws "unknown kind" criteria(Dict{String, Any}("x" => 1))
end

@testset "compare_to_baseline NVE" begin
    @test compare_to_baseline(nve(), nve())["pass"]
    @test !compare_to_baseline(nve(rms = 1.0), nve())["pass"]
    # An unstable baseline puts no demand on the record.
    @test compare_to_baseline(nve(rms = 1.0), nve(rms = 1.0))["pass"]
    @test_throws "time steps differ" compare_to_baseline(nve(), merge(nve(), Dict("dt_fs" => 4.0)))
end

nvt(z, zv, m, se) = Dict{String, Any}(
    "observed" => Dict("variance_z" => zv, "mean_K" => m, "autocorrelation" => Dict("mean_z" => z, "mean_error_K" => se)),
)

@testset "NVT" begin
    @test criteria(nvt(NVT_Z_MAX, -NVT_Z_MAX, 1200.0, 7.0))["pass"]
    @test !criteria(nvt(3.1, 0.0, 1200.0, 7.0))["mean_z"]
    @test !criteria(nvt(0.0, -3.1, 1200.0, 7.0))["variance_z"]
    @test compare_to_baseline(nvt(0, 0, 1200.0, 3.0), nvt(0, 0, 1212.0, 4.0))["pass"]   # 12 K = 3 × 4 combined
    @test !compare_to_baseline(nvt(0, 0, 1200.0, 3.0), nvt(0, 0, 1216.0, 4.0))["pass"]
end

function vacancy(hops, ps; status = "completed", n = 5)
    return Dict{String, Any}(
        "hops" => hops, "duration_ps" => ps, "status" => status, "frames" => n, "t_fs" => collect(1.0:n),
        "vacancy_site" => collect(1:n), "vacancy_R" => [zeros(3) for _ in 1:n],
        "temperature_K" => 1300.0 .+ 10 .* randn(Xoshiro(1), 200),
    )
end

@testset "vacancy" begin
    @test criteria(vacancy(4, 22.0))["pass"]
    @test !criteria(vacancy(4, 21.0))["length"]
    @test !criteria(vacancy(4, 22.0; status = "stopped"))["completed"]
    @test !criteria(merge(vacancy(4, 22.0), Dict("frames" => 4)))["one_vacancy_per_frame"]
    h = hop_test(vacancy(1, 22.0), vacancy(4, 22.0))
    @test h["p"] ≈ 0.375 && h["p_min"] ≈ 0.0625
    @test compare_to_baseline(vacancy(1, 22.0), vacancy(4, 22.0))["hop_counts"]
    @test !compare_to_baseline(vacancy(0, 22.0), vacancy(8, 22.0))["hop_counts"]
    # Equal rates: 40 hops in 440 ps against 4 in 44 ps.
    @test hop_test(vacancy(40, 440.0), vacancy(4, 44.0))["p"] > 0.9
    @test compare_to_baseline(vacancy(4, 22.0), vacancy(4, 22.0))["pass"]
end
