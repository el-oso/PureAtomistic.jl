# The Gaussian94 `.gbs` reader: the grammar of basis blocks and ECP sections, symbol and shell-letter
# case, line endings, and one error item per failure the reader reports.

@testmodule GBSFiles begin
    using QuantumEngine.Basis: parse_gbs, ShellSpec

    parse(text) = parse_gbs(IOBuffer(text); source = "t.gbs")

    # `ShellSpec` has no `==`; two shells are equal when their `l`, `line` and primitives are.
    same_shell(a::ShellSpec, b::ShellSpec) = a.l == b.l && a.line == b.line && a.primitives == b.primitives
    same_shells(a, b) = length(a) == length(b) && all(same_shell(x, y) for (x, y) in zip(a, b))

    # Line 1 a `****` outside any block, then comments, an H block, blank lines and an O block with
    # S, SP and D shells. The H rows are those of STO-3G, the O rows those of STO-3G and the D shell
    # of 6-31G*, in `D` notation.
    const GRAMMAR_LINES = [
        "****",                                                                         # 1
        "! Basis Set Exchange",                                                         # 2
        "# a comment",                                                                  # 3
        "H     0",                                                                      # 4
        "S    3   1.00",                                                                # 5
        "      0.3425250914D+01       0.1543289673D+00",                                # 6
        "      0.6239137298D+00       0.5353281423D+00",                                # 7
        "      0.1688554040D+00       0.4446345422D+00",                                # 8
        "S    1   1.00",                                                                # 9
        "      0.1000000000D+00       0.1000000000D+01",                                # 10
        "****",                                                                         # 11
        "",                                                                             # 12
        "",                                                                             # 13
        "O     0",                                                                      # 14
        "S    3   1.00",                                                                # 15
        "      0.1307093214D+03       0.1543289673D+00",                                # 16
        "      0.2380886605D+02       0.5353281423D+00",                                # 17
        "      0.6443608313D+01       0.4446345422D+00",                                # 18
        "SP   3   1.00",                                                                # 19
        "      0.5033151319D+01      -0.9996722919D-01       0.1559162750D+00",         # 20
        "      0.1169596125D+01       0.3995128261D+00       0.6076837186D+00",         # 21
        "      0.3803889600D+00       0.7001154689D+00       0.3919573931D+00",         # 22
        "D    1   1.00",                                                                # 23
        "      0.8000000000D+00       1.0000000",                                       # 24
        "****",                                                                         # 25
    ]
    const GRAMMAR = join(GRAMMAR_LINES, "\n") * "\n"

    # The lanl2dz Na ECP section (`lanl2dz.gbs`, Basis Set Exchange), typed in.
    const NA_ECP_LINES = [
        "NA     0",
        "NA-ECP     2     10",
        "d potential",
        "  5",
        "1    175.5502590            -10.0000000",
        "2     35.0516791            -47.4902024",
        "2      7.9060270            -17.2283007",
        "2      2.3365719             -6.0637782",
        "2      0.7799867             -0.7299393",
        "s-d potential",
        "  5",
        "0    243.3605846              3.0000000",
        "1     41.5764759             36.2847626",
        "2     13.2649167             72.9304880",
        "2      3.6797165             23.8401151",
        "2      0.9764209              6.0123861",
        "p-d potential",
        "  6",
        "0   1257.2650682              5.0000000",
        "1    189.6248810            117.4495683",
        "2     54.5247759            423.3986704",
        "2     13.7449955            109.3247297",
        "2      3.6813579             31.3701656",
        "2      0.9461106              7.1241813",
    ]
    const NA_ECP = join(NA_ECP_LINES, "\n") * "\n"
    const NA_BLOCK = "Na     0\nS    1   1.00\n      0.5000000      1.0000000\n****\n"
end

@testitem "gbs: grammar" setup = [GBSFiles] begin
    using QuantumEngine.Basis: ElementBasis
    data = GBSFiles.parse(GBSFiles.GRAMMAR)
    @test data isa Dict{String, ElementBasis{Float64}}
    @test sort(collect(keys(data))) == ["H", "O"]
    h, o = data["H"], data["O"]
    @test (h.symbol, o.symbol) == ("H", "O")
    @test isnothing(h.ecp) && isnothing(o.ecp)
    @test [s.l for s in h.shells] == [0, 0]
    @test [s.l for s in o.shells] == [0, 0, 1, 2]
    @test [s.line for s in h.shells] == [5, 9]
    @test [s.line for s in o.shells] == [15, 19, 19, 23]
    p(x) = parse(Float64, x)
    values(s) = [(q.exponent, q.coefficient) for q in s.primitives]
    @test values(h.shells[1]) == [
        (p("0.3425250914E+01"), p("0.1543289673E+00")),
        (p("0.6239137298E+00"), p("0.5353281423E+00")),
        (p("0.1688554040E+00"), p("0.4446345422E+00")),
    ]
    @test values(h.shells[2]) == [(p("0.1000000000E+00"), p("0.1000000000E+01"))]
    @test values(o.shells[1]) == [
        (p("0.1307093214E+03"), p("0.1543289673E+00")),
        (p("0.2380886605E+02"), p("0.5353281423E+00")),
        (p("0.6443608313E+01"), p("0.4446345422E+00")),
    ]
    @test values(o.shells[2]) == [
        (p("0.5033151319E+01"), p("-0.9996722919E-01")),
        (p("0.1169596125E+01"), p("0.3995128261E+00")),
        (p("0.3803889600E+00"), p("0.7001154689E+00")),
    ]
    @test values(o.shells[3]) == [
        (p("0.5033151319E+01"), p("0.1559162750E+00")),
        (p("0.1169596125E+01"), p("0.6076837186E+00")),
        (p("0.3803889600E+00"), p("0.3919573931E+00")),
    ]
    # The S and P shells of the SP entry share their exponents.
    @test [q.exponent for q in o.shells[2].primitives] == [q.exponent for q in o.shells[3].primitives]
    @test values(o.shells[4]) == [(p("0.8000000000E+00"), p("1.0000000"))]
    # The last basis block ended by EOF instead of `****`: the same shells, values and lines.
    open_end = join(GBSFiles.GRAMMAR_LINES[1:(end - 1)], "\n") * "\n"
    @test !endswith(rstrip(open_end), "****")
    d = GBSFiles.parse(open_end)
    @test sort(collect(keys(d))) == ["H", "O"]
    @test GBSFiles.same_shells(d["H"].shells, h.shells)
    @test GBSFiles.same_shells(d["O"].shells, o.shells)
    # A one-block file ending at EOF, with and without a final newline.
    one = "C 0\nS 1 1.00\n 0.5 1.0\nP 1 1.00\n 0.4 1.0\n"
    closed = GBSFiles.parse(one * "****\n")["C"].shells
    @test GBSFiles.same_shells(GBSFiles.parse(one)["C"].shells, closed)
    @test GBSFiles.same_shells(GBSFiles.parse(String(chop(one)))["C"].shells, closed)
    # A lower-case `d` exponent marker reads as `E`, as an upper-case one does.
    lower = GBSFiles.parse(replace(GBSFiles.GRAMMAR, "D+" => "d+", "D-" => "d-"))
    @test !occursin("D+", replace(GBSFiles.GRAMMAR, "D+" => "d+", "D-" => "d-"))
    @test GBSFiles.same_shells(lower["H"].shells, h.shells)
    @test GBSFiles.same_shells(lower["O"].shells, o.shells)
    @test only(GBSFiles.parse("H 0\nS 1 1.00\n 0.5d+00 0.1d+01\n****\n")["H"].shells).primitives[1].exponent == 0.5
    # The stream's default source name.
    @test_throws "basis: <stream>:1: unknown element symbol \"Xx\"" QuantumEngine.Basis.parse_gbs(IOBuffer("Xx 0\n"))
end

@testitem "gbs: ECP section (inline)" setup = [GBSFiles] begin
    using QuantumEngine.Basis: ECPPotential
    p(x) = parse(Float64, x)
    data = GBSFiles.parse(GBSFiles.NA_BLOCK * "\n" * GBSFiles.NA_ECP)
    na = data["Na"]
    @test length(na.shells) == 1
    ecp = na.ecp
    @test ecp isa ECPPotential{Float64}
    @test (ecp.symbol, ecp.ncore, ecp.lmax) == ("Na", 10, 2)
    @test [c.l for c in ecp.channels] == [2, 0, 1]
    @test [length(c.powers) for c in ecp.channels] == [5, 5, 6]
    ch = ecp.channels[1]
    @test length(ch.exponents) == length(ch.coefficients) == 5
    @test (ch.powers[1], ch.exponents[1], ch.coefficients[1]) == (1, p("175.5502590"), p("-10.0000000"))
    @test ecp.channels[3].powers == [0, 1, 2, 2, 2, 2]
    @test (ecp.channels[3].exponents[6], ecp.channels[3].coefficients[6]) == (p("0.9461106"), p("7.1241813"))
    # An ECP section for an element without a basis block in the file: an element basis with no
    # shells and that ECP.
    alone = GBSFiles.parse(GBSFiles.NA_ECP)
    @test collect(keys(alone)) == ["Na"]
    @test isempty(alone["Na"].shells)
    @test alone["Na"].ecp.lmax == 2
    @test [c.l for c in alone["Na"].ecp.channels] == [2, 0, 1]
    # The ECP section before the basis block, and ended by a header, a blank line and EOF.
    for text in (
            GBSFiles.NA_ECP * GBSFiles.NA_BLOCK, GBSFiles.NA_ECP * "\n" * GBSFiles.NA_BLOCK,
            GBSFiles.NA_BLOCK * GBSFiles.NA_ECP * "****\n",
        )
        d = GBSFiles.parse(text)
        @test length(d["Na"].shells) == 1
        @test d["Na"].ecp.ncore == 10
    end
    # Channel titles are matched without regard to case.
    upper = replace(GBSFiles.NA_ECP, "d potential" => "D POTENTIAL", "s-d" => "S-D")
    @test [c.l for c in GBSFiles.parse(upper)["Na"].ecp.channels] == [2, 0, 1]
end

@testitem "header word boundary" setup = [GBSFiles] begin
    text = "CS     0\nS    1   1.00\n  0.5  1.0\n****\nC     0\nS    1   1.00\n  0.5  1.0\nP    1   1.00\n  0.4  1.0\n****\n"
    data = GBSFiles.parse(text)
    @test sort(collect(keys(data))) == ["C", "Cs"]
    @test length(data["C"].shells) == 2
    @test length(data["Cs"].shells) == 1
    @test [s.line for s in data["C"].shells] == [6, 8]
end

@testitem "gbs: case" setup = [GBSFiles] begin
    upper = "AG     0\nS    3   1.00\n  3.0  0.2\n  1.0  0.5\n  0.3  0.4\n****\n"
    data = GBSFiles.parse(upper)
    @test collect(keys(data)) == ["Ag"]
    @test data["Ag"].symbol == "Ag"
    lower = GBSFiles.parse(replace(upper, "S    3" => "s    3"))
    @test GBSFiles.same_shells(lower["Ag"].shells, data["Ag"].shells)
    @test lower["Ag"].shells[1].l == 0
    # A lowercase `sp` gives the S and P shells of `SP`.
    sp = "C 0\nSP 1 1.00\n 0.5 0.3 0.7\n****\n"
    @test GBSFiles.same_shells(GBSFiles.parse(replace(sp, "SP" => "sp"))["C"].shells, GBSFiles.parse(sp)["C"].shells)
    @test GBSFiles.same_shells(GBSFiles.parse(replace(sp, "SP" => "Sp"))["C"].shells, GBSFiles.parse(sp)["C"].shells)
    # Every shell letter of l = 0 to 5 in both cases.
    for (l, letter) in enumerate("SPDFGH")
        for c in (letter, lowercase(letter))
            s = GBSFiles.parse("C 0\n$(c) 1 1.00\n 0.5 1.0\n****\n")["C"].shells
            @test only(s).l == l - 1
        end
    end
end

@testitem "gbs: I shell" setup = [GBSFiles] begin
    s = only(GBSFiles.parse("C     0\nI    1   1.00\n  0.5  1.0\n****\n")["C"].shells)
    @test s.l == 6
    @test s.line == 2
    @test only(GBSFiles.parse("C 0\ni 1 1.00\n 0.5 1.0\n****\n")["C"].shells).l == 6
end

@testitem "gbs: line endings" setup = [GBSFiles] begin
    lf = GBSFiles.parse(GBSFiles.GRAMMAR)
    lines = [replace(l, r" +" => "\t") * "  \t" for l in GBSFiles.GRAMMAR_LINES]
    crlf = "﻿" * join(lines, "\r\n") * "\r\n"
    @test occursin('\t', crlf) && occursin("\r\n", crlf) && startswith(crlf, "﻿")
    data = GBSFiles.parse(crlf)
    @test sort(collect(keys(data))) == sort(collect(keys(lf)))
    for k in keys(lf)
        @test GBSFiles.same_shells(data[k].shells, lf[k].shells)
    end
    # A byte-order mark before a header on line 1.
    @test GBSFiles.same_shells(
        GBSFiles.parse("﻿H 0\r\nS 1 1.00\r\n 0.5 1.0\r\n****\r\n")["H"].shells,
        GBSFiles.parse("H 0\nS 1 1.00\n 0.5 1.0\n****\n")["H"].shells,
    )
end

@testitem "parse errors are loud" setup = [GBSFiles] begin
    using QuantumEngine.Basis: read_element_basis
    P = GBSFiles.parse
    ecp(lines...) = join(["NA 0", "NA-ECP 2 10", lines...], "\n") * "\n"
    d_channel = ["d potential", "1", "1 175.5 -10.0"]
    sd_channel = ["s-d potential", "1", "0 243.4 3.0"]
    @test_throws BasisError P("Xx 0\n")
    @test_throws "basis: t.gbs:2: unknown element symbol \"Xx\"" P("\nXx     0\nS 1 1.00\n 1.0 1.0\n****\n")
    @test_throws "basis: t.gbs:2: unknown shell type \"Q\"" P("H 0\nQ 1 1.00\n 1.0 1.0\n****\n")
    @test_throws "basis: t.gbs:2: unknown shell type \"L\"" P("H 0\nL 1 1.00\n 1.0 1.0 1.0\n****\n")
    @test_throws "basis: t.gbs:2: unknown shell type \"SPD\"" P("H 0\nSPD 1 1.00\n 1.0 1.0\n****\n")
    @test_throws "basis: t.gbs:2: expected the primitive count, got \"x\"" P("H 0\nS x 1.00\n 1.0 1.0\n****\n")
    @test_throws "basis: t.gbs:2: expected the primitive count, got \"0\"" P("H 0\nS 0 1.00\n****\n")
    @test_throws "basis: t.gbs:2: expected the primitive count, got \"-1\"" P("H 0\nS -1 1.00\n****\n")
    @test_throws "basis: t.gbs:2: expected the primitive count, got \"1.5\"" P("H 0\nS 1.5 1.00\n****\n")
    @test_throws "basis: t.gbs:2: scale factor 1.2 is not supported" P("H 0\nS 1 1.2\n 1.0 1.0\n****\n")
    @test_throws "basis: t.gbs:2: \"1.0x\" is not a number" P("H 0\nS 1 1.0x\n 1.0 1.0\n****\n")
    @test_throws "basis: t.gbs:3: expected 2 numbers (exponent, coefficient), got 1" P("H 0\nS 1 1.00\n 1.0\n****\n")
    @test_throws "basis: t.gbs:3: expected 2 numbers (exponent, coefficient), got 3" P("H 0\nS 1 1.00\n 1.0 1.0 1.0\n****\n")
    @test_throws "basis: t.gbs:3: expected 3 numbers (exponent, s-coefficient, p-coefficient), got 2" P("C 0\nSP 1 1.00\n 1.0 1.0\n****\n")
    @test_throws "basis: t.gbs:3: \"0.3x\" is not a number" P("H 0\nS 1 1.00\n 0.3x 1.0\n****\n")
    @test_throws "basis: t.gbs:3: \"0.3x\" is not a number" P("H 0\nS 1 1.00\n 1.0 0.3x\n****\n")
    # A comment line inside a primitive list is read as a primitive line.
    @test_throws "basis: t.gbs:3: \"!\" is not a number" P("H 0\nS 1 1.00\n! c\n****\n")
    # A primitive list ended by `****`, by a header and by EOF; the line is the shell's.
    @test_throws "basis: t.gbs:2: the P shell ends after 2 of 3 primitives" P("C 0\nP 3 1.00\n 1.0 1.0\n 2.0 1.0\n****\n")
    @test_throws "basis: t.gbs:2: the P shell ends after 2 of 3 primitives" P("C 0\nP 3 1.00\n 1.0 1.0\n 2.0 1.0\nN 0\n")
    @test_throws "basis: t.gbs:2: the P shell ends after 2 of 3 primitives" P("C 0\nP 3 1.00\n 1.0 1.0\n 2.0 1.0\n")
    @test_throws "basis: t.gbs:2: the SP shell ends after 0 of 1 primitives" P("C 0\nSP 1 1.00\n")
    for (bad, shown) in (("-1.0", "-1.0"), ("0.0", "0.0"), ("NaN", "NaN"), ("Inf", "Inf"), ("-0.1D+01", "-1.0"))
        @test_throws "basis: t.gbs:3: exponent must be positive and finite, got $(shown)" P("H 0\nS 1 1.00\n $(bad) 1.0\n****\n")
    end
    for bad in ("NaN", "Inf")
        @test_throws "basis: t.gbs:3: coefficient must be finite, got $(bad)" P("H 0\nS 1 1.00\n 1.0 $(bad)\n****\n")
        @test_throws "basis: t.gbs:3: coefficient must be finite, got $(bad)" P("C 0\nSP 1 1.00\n 1.0 0.5 $(bad)\n****\n")
        @test_throws "basis: t.gbs:3: coefficient must be finite, got $(bad)" P("C 0\nSP 1 1.00\n 1.0 $(bad) 0.5\n****\n")
    end
    @test_throws "basis: t.gbs:2: the S shell of H has zero norm" P("H 0\nS 2 1.00\n 1.0 0.0\n 2.0 0.0\n****\n")
    @test_throws "basis: t.gbs:2: the P shell of C has zero norm" P("C 0\nSP 1 1.00\n 1.0 0.5 0.0\n****\n")
    @test_throws "basis: t.gbs:2: the S shell of C has zero norm" P("C 0\nSP 1 1.00\n 1.0 0.0 0.5\n****\n")
    # A norm sum that overflows, underflows or cancels in Float64, with nonzero coefficients.
    for row in ("1e300 1.0", "1e-300 1.0", "1.0 1e200")
        @test_throws "basis: t.gbs:2: the norm of the S shell of H is not a positive finite Float64" P("H 0\nS 1 1.00\n $(row)\n****\n")
    end
    @test_throws "basis: t.gbs:2: the norm of the P shell of C is not a positive finite Float64" P("C 0\nP 2 1.00\n 1.0 1.0\n 1.0 -1.0\n****\n")
    # An exponent whose products with the others underflow in Float64.
    @test_throws "basis: t.gbs:2: the norm of the S shell of H is not a positive finite Float64" P("H 0\nS 2 1.00\n 1e-200 1.0\n 1.0 1.0\n****\n")
    @test_throws "basis: t.gbs:1: expected an element header \"<symbol> 0\", got \"S 3 1.00\"" P("S 3 1.00\n 1.0 1.0\n")
    @test_throws "basis: t.gbs:6: expected an element header \"<symbol> 0\", got \"1.0 1.0\"" P("H 0\nS 1 1.00\n 1.0 1.0\n****\n\n1.0 1.0\n")
    @test_throws "basis: t.gbs:1: the basis block of C has no shells" P("C 0\n****\n")
    @test_throws "basis: t.gbs:1: the basis block of C has no shells" P("C 0\n! c\n\n")
    @test_throws "basis: t.gbs:4: the basis block of C is not closed by \"****\" before the header of N" P("C 0\nS 1 1.00\n 1.0 1.0\nN 0\n")
    @test_throws "basis: t.gbs:2: the basis block of C is not closed by \"****\" before the header of S" P("C 0\nS 0\n")
    @test_throws "basis: t.gbs:5: second basis block for C (first at line 1)" P("C 0\nS 1 1.00\n 1.0 1.0\n****\nC 0\nS 1 1.00\n 1.0 1.0\n****\n")
    @test_throws "basis: t.gbs:25: second ECP section for Na (first at line 1)" P(GBSFiles.NA_ECP * GBSFiles.NA_ECP)
    @test_throws "basis: t.gbs:2: ECP section for Mg under the header of Na" P("NA 0\nMG-ECP 2 10\n")
    @test_throws "basis: t.gbs:6: expected the \"s-d potential\" channel, got \"p-d potential\"" P(ecp(d_channel..., "p-d potential"))
    @test_throws "basis: t.gbs:3: expected the \"d potential\" channel, got \"s-d potential\"" P(ecp(sd_channel...))
    @test_throws "basis: t.gbs:7: expected the row count of the \"s-d potential\" channel, got \"x\"" P(ecp(d_channel..., "s-d potential", "x"))
    @test_throws "basis: t.gbs:7: expected the row count of the \"s-d potential\" channel, got \"0\"" P(ecp(d_channel..., "s-d potential", "0"))
    @test_throws "basis: t.gbs:7: expected the row count of the \"s-d potential\" channel, got \"5 1\"" P(ecp(d_channel..., "s-d potential", "5 1"))
    @test_throws "basis: t.gbs:5: expected 3 numbers (power, exponent, coefficient), got 2" P(ecp("d potential", "1", "1 175.5"))
    @test_throws "basis: t.gbs:5: \"1.5\" is not an integer power" P(ecp("d potential", "1", "1.5 175.5 -10.0"))
    @test_throws "basis: t.gbs:5: \"0.3x\" is not a number" P(ecp("d potential", "1", "1 0.3x -10.0"))
    @test_throws "basis: t.gbs:5: \"0.3x\" is not a number" P(ecp("d potential", "1", "0.3x 175.5 -10.0"))
    # ECP rows carry the checks of primitive rows, and a power is at least 0.
    for (bad, shown) in (("-2.0", "-2.0"), ("0.0", "0.0"), ("NaN", "NaN"), ("Inf", "Inf"))
        @test_throws "basis: t.gbs:5: exponent must be positive and finite, got $(shown)" P(ecp("d potential", "1", "1 $(bad) -10.0"))
    end
    for bad in ("NaN", "Inf", "-Inf")
        @test_throws "basis: t.gbs:5: coefficient must be finite, got $(bad)" P(ecp("d potential", "1", "1 175.5 $(bad)"))
    end
    @test_throws "basis: t.gbs:5: power must be non-negative, got -1" P(ecp("d potential", "1", "-1 175.5 -10.0"))
    @test_throws "basis: t.gbs:2: the ECP section of Na has ncore 12, above the atomic number 11" P("NA 0\nNA-ECP 2 12\n")
    @test P("NA 0\nNA-ECP 0 11\ns potential\n1\n0 1.0 1.0\n")["Na"].ecp.ncore == 11
    # An ECP channel ended by a header, by `****` and by EOF; the line is the channel title's.
    rows3 = ["s-d potential", "5", "0 1.0 1.0", "1 1.0 1.0", "2 1.0 1.0"]
    @test_throws "basis: t.gbs:6: the \"s-d potential\" channel ends after 3 of 5 rows" P(ecp(d_channel..., rows3..., "C 0"))
    @test_throws "basis: t.gbs:6: the \"s-d potential\" channel ends after 3 of 5 rows" P(ecp(d_channel..., rows3..., "****"))
    @test_throws "basis: t.gbs:6: the \"s-d potential\" channel ends after 3 of 5 rows" P(ecp(d_channel..., rows3...))
    # Too few channels, ended by a header, by `****`, by EOF and inside a channel's count.
    @test_throws "basis: t.gbs:1: the ECP section of Na has 2 of 3 channels" P(ecp(d_channel..., sd_channel..., "C 0"))
    @test_throws "basis: t.gbs:1: the ECP section of Na has 2 of 3 channels" P(ecp(d_channel..., sd_channel..., "****"))
    @test_throws "basis: t.gbs:1: the ECP section of Na has 2 of 3 channels" P(ecp(d_channel..., sd_channel...))
    @test_throws "basis: t.gbs:1: the ECP section of Na has 1 of 3 channels" P(ecp(d_channel..., "s-d potential"))
    @test_throws "basis: t.gbs:1: the ECP section of Na has 0 of 3 channels" P(ecp())
    # Inputs outside the message table: a shell line with a missing count or extra tokens, and an
    # ECP lmax above the highest shell letter.
    @test_throws "basis: t.gbs:2: expected a shell line \"<type> <count> [<scale>]\", got \"S\"" P("H 0\nS\n****\n")
    @test_throws "basis: t.gbs:2: expected a shell line \"<type> <count> [<scale>]\", got \"S 1 1.00 0.0\"" P("H 0\nS 1 1.00 0.0\n 1.0 1.0\n****\n")
    @test_throws "basis: t.gbs:2: the ECP section of Na has lmax 7; channel titles stop at I (l = 6)" P("NA 0\nNA-ECP 7 10\n")
    # The missing element of a file, named by its path.
    mktempdir() do dir
        path = joinpath(dir, "w.gbs")
        write(path, "H 0\nS 1 1.00\n 0.5 1.0\n****\n")
        @test only(read_element_basis(path, "h").shells).l == 0
        @test read_element_basis(path, SubString("xH", 2)).symbol == "H"
        @test_throws "basis: $(path): no basis for element Xe" read_element_basis(path, "XE")
        @test_throws BasisError read_element_basis(path, "Xe")
        bad = joinpath(dir, "b.gbs")
        write(bad, "H 0\nQ 1 1.00\n")
        @test_throws "basis: $(bad):2: unknown shell type \"Q\"" read_element_basis(bad, "H")
    end
end
