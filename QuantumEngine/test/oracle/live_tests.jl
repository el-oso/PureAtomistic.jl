# Re-run the oracle binary named by QUANTUM_ENGINE_ORACLE and check that it still reproduces
# references.toml. Every compared quantity is looked up with `tolerance`, which throws for a
# quantity whose row does not exist yet; each quantity has its own testset, so such an error
# leaves the other quantities compared. A case's known oracle defects have no tolerance and are
# checked for exact reproduction.
@testitem "oracle: live runs reproduce references.toml" tags = [:oracle] setup = [OracleHarness] begin
    using .OracleHarness: run_oracle, tolerance, quantity, reference, reference_vector
    @testset "$case" for case in OracleHarness.case_names()
        dir = mktempdir()
        r = run_oracle(case; dir)
        @test r.exitcode == 0
        @testset "$key" for key in sort!(collect(keys(OracleHarness.case_table(case)["tolerance"])))
            tol = tolerance(case, key)
            v = quantity(case, key, r.log, r.stdout, dir)
            if v isa Vector{Float64}
                ref = reference_vector(case, key)
                @test length(v) == length(ref)
                @test all(abs(a - b) ≤ tol for (a, b) in zip(v, ref))
            else
                @test abs(v - reference(case, key)) ≤ tol
            end
        end
        # A known oracle defect is the oracle's own output, reproduced exactly.
        @testset "defect $key" for key in sort!(collect(keys(get(OracleHarness.case_table(case), "defects", Dict{String, Any}()))))
            @test quantity(case, key, r.log, r.stdout, dir) == reference(case, key)
        end
    end
end

@testitem "oracle: the basis directory must be named" tags = [:oracle] setup = [OracleHarness] begin
    @test OracleHarness.oracle_binary() !== nothing
    withenv("QUANTUM_ENGINE_ORACLE_BASIS" => nothing) do
        @test_throws "oracle: QUANTUM_ENGINE_ORACLE is set but QUANTUM_ENGINE_ORACLE_BASIS (the oracle's basis_sets directory) is not" OracleHarness.oracle_basis_dir()
    end
end
