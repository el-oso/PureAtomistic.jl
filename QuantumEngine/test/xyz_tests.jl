@testmodule XYZFiles begin
    using QuantumEngine
    using QuantumEngine.IO: XYZFrame, read_xyz, write_xyz

    # The text `write_xyz` writes for `atoms` and `comment`.
    xyz_text(atoms; comment = "") = sprint(io -> write_xyz(io, atoms; comment))

    # The frames of the xyz text `s`.
    frames(s; kw...) = read_xyz(IOBuffer(s); source = "t.xyz", kw...)

    # The water frame of `examples/water.xyz`, in LF form.
    const WATER = """
    3
    Water molecule
    O   0.000000   0.000000   0.000000
    H   0.757000   0.586000   0.000000
    H  -0.757000   0.586000   0.000000
    """
end

@testitem "xyz: round trip" setup = [Fixtures, XYZFiles] begin
    using QuantumEngine.Units: ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
    atoms = Fixtures.water().atoms
    fr = XYZFiles.frames(XYZFiles.xyz_text(atoms; comment = "water, Å"))
    @test length(fr) == 1
    @test fr[1].comment == "water, Å"
    back = fr[1].atoms
    @test [a.symbol for a in back] == [a.symbol for a in atoms]
    # Oracle: the input positions. Tolerance (b): the product with BOHR_TO_ANGSTROM (1 ulp), the
    # shortest round-trip printing and parsing (exact), the product with ANGSTROM_TO_BOHR (1 ulp),
    # and the constants' product, within 1 ulp of 1 (asserted below): 3 eps, rounded up to 4 eps.
    @test abs(BOHR_TO_ANGSTROM * ANGSTROM_TO_BOHR - 1) <= eps()
    for i in eachindex(atoms, back)
        @test isapprox(back[i].position, atoms[i].position; rtol = 4eps())
    end
    @test [a.Z for a in back] == [a.Z for a in atoms]
end

@testitem "xyz: multi-frame" setup = [Fixtures, XYZFiles] begin
    atoms = Fixtures.water().atoms
    shifted(d) = [Atom(a.symbol, a.position .+ d) for a in atoms]
    sets = (atoms, shifted(0.5), shifted(-1.25))
    text = join(XYZFiles.xyz_text(s; comment = "frame $(k)") for (k, s) in enumerate(sets))
    fr = XYZFiles.frames(text)
    @test length(fr) == 3
    for (k, s) in enumerate(sets)
        @test fr[k].comment == "frame $(k)"
        @test [a.symbol for a in fr[k].atoms] == [a.symbol for a in s]
        # Oracle and tolerance as in "xyz: round trip".
        for i in eachindex(s, fr[k].atoms)
            @test isapprox(fr[k].atoms[i].position, s[i].position; rtol = 4eps())
        end
    end
end

@testitem "xyz: blank comment line" setup = [XYZFiles] begin
    # Line 2 is the comment even when blank; the oracle drops blank lines before counting
    # (io/molecule.rs:36) and so reads the first atom line as the comment.
    text = replace(XYZFiles.WATER, "Water molecule" => "")
    for lenient in (false, true)
        w = InputWarning[]
        fr = XYZFiles.frames(text; lenient, warnings = w)
        @test length(fr) == 1
        @test fr[1].comment == ""
        @test length(fr[1].atoms) == 3
        @test fr[1].atoms[1].symbol == "O"
        @test isempty(w)
    end
end

@testitem "xyz: line endings" setup = [Fixtures, XYZFiles] begin
    lf = XYZFiles.WATER * XYZFiles.WATER
    other = "\ufeff" * replace(replace(lf, "   " => "\t", "  " => " \t"), "\n" => "\r\n")
    @test occursin('\t', other) && occursin("\r\n", other)
    a = XYZFiles.frames(lf)
    b = XYZFiles.frames(other)
    @test length(a) == length(b) == 2
    for k in eachindex(a, b)
        @test Fixtures.fields_equal(a[k], b[k])
        for i in eachindex(a[k].atoms, b[k].atoms)
            @test Fixtures.fields_equal(a[k].atoms[i], b[k].atoms[i])
        end
    end
end

@testitem "xyz: errors" setup = [XYZFiles] begin
    @test_throws "input: t.xyz:1: xyz: expected the atom count" XYZFiles.frames("three\nc\nH 0 0 0\n")
    @test_throws "input: t.xyz:1: xyz: the frame declares 3 atoms but has 2" XYZFiles.frames("3\nc\nH 0 0 0\nH 0 0 1\n")
    bad = "2\nc\nH 0 0 0\nXx 0 0 1\n"
    @test_throws "input: t.xyz:4: xyz: unknown element symbol \"Xx\"" XYZFiles.frames(bad)
    @test_throws "input: t.xyz:3: xyz: unknown element symbol \"Xx\"" XYZFiles.frames("1\nc\nXx 0 0 0\n")
    err = try
        XYZFiles.frames("1\nc\nXx 0 0 0\n")
    catch e
        e
    end
    @test err isa InputError
    @test err.line == 3
    @test !occursin("system: ", sprint(showerror, err))
    # A System error other than an unknown element is an error in both modes.
    @test_throws "input: t.xyz:3: xyz: a Bq atom needs an element" XYZFiles.frames("1\nc\nBq 0 0 0\n"; lenient = true)
    @test_throws "input: t.xyz:3: xyz: coordinate \"y\" is not a number" XYZFiles.frames("1\nc\nH 0 y 0\n"; lenient = true)
    @test_throws "input: t.xyz:3: xyz: expected a symbol and three coordinates" XYZFiles.frames("1\nc\nH 0 0\n")
    @test_throws "input: t.xyz: xyz: the file has no frame" XYZFiles.frames("\n\n")
    @test_throws "io: an xyz comment is one line" XYZFiles.xyz_text([Atom("H", [0.0, 0.0, 0.0])]; comment = "a\nb")
end

@testitem "xyz: lenient" setup = [XYZFiles] begin
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    @testset "L19a :xyz_count" begin
        # Oracle: io/molecule.rs:50-51 stops at the end of the atom lines and returns the atoms read.
        text = "3\nc\nH 0 0 0\nH 0 0 1\n"
        @test_throws "input: t.xyz:1: xyz: the frame declares 3 atoms but has 2" XYZFiles.frames(text)
        w = InputWarning[]
        fr = XYZFiles.frames(text; lenient = true, warnings = w)
        @test length(fr[1].atoms) == 2
        @test length(w) == 1
        @test w[1].case === :xyz_count
        @test w[1].line == 1
    end
    @testset "L19b :unknown_element" begin
        # Oracle: io/molecule.rs:66 gives an unknown symbol Z = 1 (`unwrap_or(1)`).
        text = "1\nc\nXx 0 0 0.5\n"
        @test_throws "input: t.xyz:3: xyz: unknown element symbol \"Xx\"" XYZFiles.frames(text)
        w = InputWarning[]
        a = only(XYZFiles.frames(text; lenient = true, warnings = w)[1].atoms)
        @test a.Z == 1
        @test a.symbol == "Xx"
        @test !a.ghost
        @test a.mass == element(1).mass_most_abundant
        @test a.position == [0.0, 0.0, 0.5 * ANGSTROM_TO_BOHR]
        @test length(w) == 1
        @test w[1].case === :unknown_element
        @test w[1].line == 3
        # Every ghost spelling of `Atom`, in any case, names its element in both modes.
        ghosts = "4\nc\nO@ 0 0 0\nbqh 0 0 1\nBq-He 0 0 2\nfe 0 0 3\n"
        for lenient in (false, true)
            w = InputWarning[]
            a = XYZFiles.frames(ghosts; lenient, warnings = w)[1].atoms
            @test [x.ghost for x in a] == [true, true, true, false]
            @test basis_element.(a) == [8, 1, 2, 26]
            @test isempty(w)
        end
        @test_throws "input: t.xyz:3: xyz: unknown element symbol \"Bq@\"" XYZFiles.frames("1\nc\nBq@ 0 0 0\n")
    end
end

@testitem "xyz: oracle files" setup = [Fixtures, XYZFiles] begin
    read(rel) = XYZFiles.frames(Fixtures.oracle_input(rel))
    @test length(read("examples/water.xyz")[1].atoms) == 3
    @test length(read("examples/legacy_inputs/adenina.xyz")[1].atoms) == 32
    neb = read("examples/legacy_inputs/neb_test.neb.xyz")
    @test length(neb) == 5
    @test all(f -> length(f.atoms) == 3, neb)
end

@testitem "xyz: generic indexing" setup = [Fixtures, XYZFiles] begin
    using OffsetArrays
    atoms = Fixtures.water().atoms
    ref = XYZFiles.xyz_text(atoms; comment = "w")
    @test XYZFiles.xyz_text(OffsetArray(atoms, -2); comment = "w") == ref
    @test XYZFiles.xyz_text(view(atoms, :); comment = "w") == ref
    @test XYZFiles.xyz_text(view(atoms, [3, 1]); comment = "w") == XYZFiles.xyz_text(atoms[[3, 1]]; comment = "w")
    @test startswith(ref, "3\nw\nO ")
end
