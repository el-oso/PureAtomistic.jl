@testmodule PDBFiles begin
    using QuantumEngine
    using QuantumEngine.IO: read_pdb

    # A fixed-width `record` line (`ATOM  ` or `HETATM`) with atom name `name`, the coordinate
    # fields `x`, `y`, `z` (columns 31–38, 39–46, 47–54) and the element `el` in columns 77–78.
    function record(rec, name, x, y, z, el)
        line = rpad(rec, 6) * "    1 " * rpad(name, 4) * " WAT A   1    " * lpad(x, 8) * lpad(y, 8) *
            lpad(z, 8) * "  1.00  0.00" * " "^10 * lpad(el, 2)
        length(line) == 78 || error("record of $(length(line)) columns")
        return line
    end

    atoms(s; kw...) = read_pdb(IOBuffer(s); source = "t.pdb", kw...)
end

@testitem "pdb: values" setup = [PDBFiles] begin
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    r = PDBFiles.record
    text = join(
        [
            "HEADER    WATER",
            r("ATOM  ", " O", "0.000", "0.000", "0.117", "O"),
            r("ATOM  ", " H1", "0.757", "0.586", "0.000", "H"),
            r("HETATM", "CL", "-0.757", "0.586", "0.000", "CL"),
            "END",
        ], "\n"
    )
    a = PDBFiles.atoms(text)
    @test [x.symbol for x in a] == ["O", "H", "CL"]
    @test [x.Z for x in a] == [8, 1, 17]
    # Oracle: the Å coordinates of the record times ANGSTROM_TO_BOHR, the one product the reader
    # computes: exact by construction.
    @test a[1].position == [0.0, 0.0, 0.117 * ANGSTROM_TO_BOHR]
    @test a[3].position == [-0.757, 0.586, 0.0] .* ANGSTROM_TO_BOHR
    # CRLF endings read the same atoms.
    b = PDBFiles.atoms(replace(text, "\n" => "\r\n"))
    @test [x.position for x in b] == [x.position for x in a]
    # A left-justified element ("C " in columns 77–78) is read in both modes, with no warning,
    # whether or not the line keeps its trailing blank; the oracle trims the two columns
    # (io/molecule.rs:95-96).
    left = r("ATOM  ", " C1", "0.000", "0.000", "0.000", "C ")
    for text in (left, left * "\nEND", rstrip(left))
        for lenient in (false, true)
            w = InputWarning[]
            c = only(PDBFiles.atoms(text; lenient, warnings = w))
            @test c.symbol == "C"
            @test c.Z == 6
            @test isempty(w)
        end
    end
end

@testitem "pdb: errors" setup = [PDBFiles] begin
    r = PDBFiles.record
    line = r("ATOM  ", " O", "0.000", "0.000", "0.000", "O")
    @test_throws "input: t.pdb:1: pdb: ATOM record shorter than 54 columns" PDBFiles.atoms(first(line, 40))
    @test_throws "input: t.pdb:1: pdb: no element in columns 77–78" PDBFiles.atoms(first(line, 66))
    @test_throws "input: t.pdb:2: pdb: coordinate \"0.0x0\" is not a number" PDBFiles.atoms(
        "REMARK\n" * r("ATOM  ", " O", "0.0x0", "0.000", "0.000", "O"); lenient = true
    )
    @test_throws "input: t.pdb:1: pdb: unknown element symbol \"Xx\"" PDBFiles.atoms(r("ATOM  ", " X", "0", "0", "0", "Xx"))
    @test_throws "input: t.pdb: pdb: no ATOM or HETATM records" PDBFiles.atoms("HEADER\nEND\n"; lenient = true)
end

@testitem "pdb: lenient" setup = [PDBFiles] begin
    r = PDBFiles.record
    @testset ":pdb_short_line, a 40-column ATOM record" begin
        # Oracle: io/molecule.rs:79 skips an ATOM record shorter than 54 columns.
        text = first(r("ATOM  ", " O", "0.000", "0.000", "0.000", "O"), 40) * "\n" *
            r("ATOM  ", " H", "0.000", "0.000", "1.000", "H")
        @test_throws "input: t.pdb:1: pdb: ATOM record shorter than 54 columns" PDBFiles.atoms(text)
        w = InputWarning[]
        a = PDBFiles.atoms(text; lenient = true, warnings = w)
        @test [x.symbol for x in a] == ["H"]
        @test length(w) == 1
        @test w[1].case === :pdb_short_line
        @test w[1].line == 1
    end
    @testset ":pdb_element_columns, a 66-column record with atom name CL" begin
        # Oracle: io/molecule.rs:95-102 takes the first character of the atom name when the line
        # has no columns 77–78.
        text = first(r("ATOM  ", "CL", "0.000", "0.000", "0.000", "CL"), 66)
        @test_throws "input: t.pdb:1: pdb: no element in columns 77–78" PDBFiles.atoms(text)
        w = InputWarning[]
        a = only(PDBFiles.atoms(text; lenient = true, warnings = w))
        @test a.symbol == "C"
        @test a.Z == 6
        @test length(w) == 1
        @test w[1].case === :pdb_element_columns
        @test w[1].line == 1
    end
    @testset ":pdb_element_columns, a 78-column record with blank columns 77–78" begin
        # Oracle: io/molecule.rs:95-96,106 reads columns 77–78 of a record of 78 columns or more,
        # counted with its trailing blanks, and gives the empty symbol Z = 1.
        line = r("ATOM  ", "CL", "0.000", "0.000", "0.000", "  ")
        for text in (line, line * "\nEND", line * "   ")
            @test_throws "input: t.pdb:1: pdb: no element in columns 77–78" PDBFiles.atoms(text)
            w = InputWarning[]
            a = only(PDBFiles.atoms(text; lenient = true, warnings = w))
            @test a.symbol == ""
            @test a.Z == 1
            @test !a.ghost
            @test a.mass == element(1).mass_most_abundant
            @test length(w) == 1
            @test w[1].case === :pdb_element_columns
            @test w[1].line == 1
        end
    end
    @testset ":nonfinite, the coordinate nan" begin
        # Oracle: io/molecule.rs:89-91 parses each coordinate field as an `f64`, which reads `nan`.
        text = r("ATOM  ", " O", "nan", "0.000", "0.000", "O")
        @test_throws "input: t.pdb:1: pdb: expected a finite number, got \"nan\"" PDBFiles.atoms(text)
        w = InputWarning[]
        a = only(PDBFiles.atoms(text; lenient = true, warnings = w))
        @test isnan(a.position[1])
        @test a.Z == 8
        @test length(w) == 1
        @test w[1].case === :nonfinite
        @test w[1].line == 1
    end
end
