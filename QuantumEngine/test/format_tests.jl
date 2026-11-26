# The oracle prints with Rust's `format!`; the expected strings are Rust's output for the same
# format specification, so the comparisons are exact.
@testitem "Rust exponent format" begin
    using QuantumEngine.IO: fmt_exp, fmt_fixed
    @test fmt_exp(0.0, 3; width = 8) == " 0.000e0"
    @test fmt_exp(1.26, 3; width = 8) == " 1.260e0"
    @test fmt_exp(0.5369, 3; width = 8) == "5.369e-1"
    @test fmt_exp(8.304e-9, 3; width = 8) == "8.304e-9"
    @test fmt_exp(-2.5e10, 3) == "-2.500e10"
    @test fmt_fixed(-76.301896228438, 20, 12) == "    -76.301896228438"
end

# Rust's `{:.3e}`, `{:.0e}` and `{:12.8}` on three-digit exponents, precision 0, Float32 and the
# non-finite values (`inf`, `-inf`, `NaN`), and its padding of integers and strings: numbers on
# the left, strings on the right, both counted in characters and never truncated.
@testitem "Rust fixed, integer and string formats" begin
    using QuantumEngine.IO: fmt_exp, fmt_fixed, fmt_int, fmt_str
    @test fmt_exp(1.0e100, 3) == "1.000e100"
    @test fmt_exp(-1.0e-100, 3) == "-1.000e-100"
    @test fmt_exp(2.5e-4, 3; width = 12) == "    2.500e-4"
    @test fmt_exp(3.0, 0) == "3e0"
    @test fmt_exp(Inf, 3; width = 5) == "  inf"
    @test fmt_exp(NaN, 3) == "NaN"
    @test fmt_fixed(0.22166487, 12, 8) == "  0.22166487"
    @test fmt_fixed(-0.8866595, 12, 8) == " -0.88665950"
    @test fmt_fixed(1.0f0, 12, 8) == "  1.00000000"
    @test fmt_fixed(123456.0, 4, 2) == "123456.00"
    @test fmt_fixed(-Inf, 6, 2) == "  -inf"
    @test fmt_fixed(NaN, 4, 2) == " NaN"
    @test fmt_fixed(2, 5, 1) == "  2.0"
    # Exact decimal ties: Rust rounds each to even.
    @test [fmt_fixed(x, 0, p) for (x, p) in ((0.25, 1), (0.125, 2), (0.375, 2), (2.5, 0), (1.5, 0), (0.5, 0), (3.5, 0), (0.0625, 3))] ==
        ["0.2", "0.12", "0.38", "2", "2", "0", "4", "0.062"]
    @test [fmt_exp(x, p) for (x, p) in ((2.5, 0), (1.25, 1), (1.75, 1), (3.5, 0))] == ["2e0", "1.2e0", "1.8e0", "4e0"]
    @test fmt_int(3, 3) == "  3"
    @test fmt_int(-12, 2) == "-12"
    @test fmt_str("O", 4) == "O   "
    @test fmt_str("Bq-H", 2) == "Bq-H"
    @test fmt_str("Å", 3) == "Å  "
    @test fmt_str(SubString("xHe", 2), 4) == "He  "
end

@testitem "format: errors" begin
    using QuantumEngine.IO: fmt_exp, fmt_fixed
    @test_throws "io: precision must be one of 0 to 8, 10, 12 and 15, got -1" fmt_fixed(1.0, 4, -1)
    @test_throws "io: precision must be one of 0 to 8, 10, 12 and 15, got -2" fmt_exp(1.0, -2)
    @test_throws "io: precision must be one of 0 to 8, 10, 12 and 15, got 9" fmt_fixed(1.0, 4, 9)
    @test_throws "io: precision must be one of 0 to 8, 10, 12 and 15, got 16" fmt_exp(1.0, 16)
    # The precision is checked before a non-finite value returns early.
    @test_throws "io: precision must be one of 0 to 8, 10, 12 and 15, got 11" fmt_fixed(Inf, 4, 11)
    @test_throws "io: precision must be one of 0 to 8, 10, 12 and 15, got 13" fmt_exp(NaN, 13)
end

# Each precision has its own Printf call: 1/3 shows how many digits each one writes.
@testitem "format: every precision" begin
    using QuantumEngine.IO: fmt_exp, fmt_fixed, FORMAT_PRECISIONS
    @test [fmt_fixed(1 / 3, 0, p) for p in FORMAT_PRECISIONS] == [p == 0 ? "0" : "0." * "3"^p for p in FORMAT_PRECISIONS]
    @test [fmt_exp(1 / 3, p) for p in FORMAT_PRECISIONS] == [(p == 0 ? "3" : "3." * "3"^p) * "e-1" for p in FORMAT_PRECISIONS]
end
