# The shipped basis data in `data/basis_sets/`: every file parses, its counts agree with a header
# scan that shares no code with the reader, its symbols resolve, and the files are the Basis Set
# Exchange exports they claim to be.

@testmodule BasisData begin
    using QuantumEngine
    using QuantumEngine.Basis: parse_gbs

    const DIR = joinpath(pkgdir(QuantumEngine), "data", "basis_sets")
    const FILES = sort!(filter(f -> endswith(f, ".gbs"), readdir(DIR)))

    read_gbs(file) = open(io -> parse_gbs(io; source = file), joinpath(DIR, file))
    file_lines(file) = readlines(joinpath(DIR, file))

    # An element header: one or two letters and a 0, nothing else on the line.
    const HEADER = r"^\s*([A-Za-z]{1,2})\s+0\s*$"
    # The line after a header that opens an ECP section, `NA-ECP     2     10`.
    const ECP_LINE = r"^\s*[A-Za-z]{1,2}-ECP\s"
    # An I shell header (l = 6) with its scale column. The iodine element header `I     0` has no
    # scale column, so it never matches.
    const I_SHELL = r"^I\s+\d+\s+1\.00"

    titlecase_symbol(s) = uppercase(s[1:1]) * lowercase(s[2:end])

    """
        header_scan(lines) -> (blocks, ecps)

    The number of headers whose next line is not an `-ECP` line (basis blocks) and of those whose
    next line is one (ECP sections).
    """
    function header_scan(lines)
        blocks = 0
        ecps = 0
        for i in eachindex(lines)
            occursin(HEADER, lines[i]) || continue
            if i < lastindex(lines) && occursin(ECP_LINE, lines[i + 1])
                ecps += 1
            else
                blocks += 1
            end
        end
        return blocks, ecps
    end

    """
        i_shell_scan(lines) -> Set{String}

    The symbols, in title case, of the element headers followed by at least one I shell header
    before the next element header.
    """
    function i_shell_scan(lines)
        found = Set{String}()
        current = ""
        for line in lines
            m = match(HEADER, line)
            if m !== nothing
                current = titlecase_symbol(m.captures[1])
            elseif occursin(I_SHELL, line)
                push!(found, current)
            end
        end
        return found
    end
end

@testitem "gbs: every shipped file parses" setup = [BasisData] begin
    @test length(BasisData.FILES) == 72
    counts = map(BasisData.FILES) do file
        data = BasisData.read_gbs(file)
        blocks = count(e -> !isempty(e.shells), values(data))
        ecps = count(e -> e.ecp !== nothing, values(data))
        @test (file, blocks, ecps) == (file, BasisData.header_scan(BasisData.file_lines(file))...)
        (blocks, ecps)
    end
    @test (sum(first, counts), sum(last, counts)) == (2873, 542)
end

@testitem "gbs: every header symbol resolves" setup = [BasisData] begin
    using QuantumEngine.Units: element
    symbols = Set{String}()
    for file in BasisData.FILES
        for k in keys(BasisData.read_gbs(file))
            @test (file, element(k).symbol) == (file, k)
            push!(symbols, k)
        end
    end
    @test symbols == Set(element(z).symbol for z in 1:96)
end

@testitem "gbs: ECP section" setup = [BasisData] begin
    na = BasisData.read_gbs("lanl2dz.gbs")["Na"]
    ecp = na.ecp
    @test ecp !== nothing
    @test (ecp.ncore, ecp.lmax) == (10, 2)
    @test length(ecp.channels) == 3
    ch = ecp.channels[1]
    @test ch.l == 2
    @test length(ch.powers) == length(ch.exponents) == length(ch.coefficients) == 5
    @test (ch.powers[1], ch.exponents[1], ch.coefficients[1]) ==
        (1, parse(Float64, "175.5502590"), parse(Float64, "-10.0000000"))
end

@testitem "gbs: shipped files are the BSE exports" setup = [BasisData] begin
    using QuantumEngine
    files = BasisData.FILES
    @test length(files) == 72
    for file in files
        lines = BasisData.file_lines(file)
        @test (file, "! Basis Set Exchange" in lines, "! Version 0.12" in lines) == (file, true, true)
    end
    for absent in ("def2-svp-jkfit.gbs", "6-31g_dp.gbs", "6-311plusplusg_dp.gbs", "6-311++g_2d2p.gbs")
        @test !(absent in files)
    end

    # No shell header with a lowercase letter: a letter or two, a count and the scale column.
    lowercase_shell = r"^\s*[a-z]{1,2}\s+\d+\s+1\.00"
    for file in files
        @test (file, count(l -> occursin(lowercase_shell, l), BasisData.file_lines(file))) == (file, 0)
    end

    # The elements with an I shell (l = 6), by the scan and by the reader.
    with_i = String[]
    for file in files
        scanned = BasisData.i_shell_scan(BasisData.file_lines(file))
        parsed = Set(k for (k, e) in BasisData.read_gbs(file) if any(s -> s.l == 6, e.shells))
        @test (file, parsed) == (file, scanned)
        isempty(scanned) || push!(with_i, file)
    end
    @test sort(with_i) == sort(
        [
            "cc-pv6z.gbs", "cc-pv5z.gbs", "aug-cc-pv5z.gbs", "pc-4.gbs", "aug-pc-4.gbs",
            "pcseg-4.gbs", "def2-universal-jkfit.gbs",
        ]
    )
    jkfit_i = Set(
        split(
            "Ag At Au Bi Cd Ce Co Cr Cu Dy Er Eu Fe Gd Hf Hg Ho Ir La Lu Mn Mo Nb Nd Ni Os Pb Pd " *
                "Pm Po Pr Pt Re Rh Rn Ru Sc Sm Ta Tb Tc Ti Tl Tm V W Y Yb Zn Zr"
        )
    )
    @test length(jkfit_i) == 50
    @test BasisData.i_shell_scan(BasisData.file_lines("def2-universal-jkfit.gbs")) == jkfit_i

    @test isfile(joinpath(pkgdir(QuantumEngine), "data", "basis_sets", "sto-3g.gbs"))
end

@testitem "gbs: shipped files equal the oracle's" setup = [BasisData] tags = [:oracle_src] begin
    src = joinpath(ENV["QUANTUM_ENGINE_ORACLE_SRC"], "basis_sets")
    @test length(BasisData.FILES) == 72
    for file in BasisData.FILES
        @test (file, read(joinpath(BasisData.DIR, file)) == read(joinpath(src, file))) == (file, true)
    end
end
