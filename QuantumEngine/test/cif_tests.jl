@testmodule CIFFiles begin
    using QuantumEngine
    using QuantumEngine.IO: read_cif

    # A CIF file whose `_atom_site` loop has the four tags `tags` and the data lines `rows`; the
    # first data line is line 7.
    loop(rows; tags = ("_atom_site_type_symbol", "_atom_site_Cartn_x", "_atom_site_Cartn_y", "_atom_site_Cartn_z")) =
        join(["data_t", "loop_", tags..., rows...], "\n") * "\n"

    atoms(s; kw...) = read_cif(IOBuffer(s); source = "t.cif", kw...)
end

@testitem "cif: values" setup = [CIFFiles] begin
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    text = """
    # iron oxide fragment
    data_feo
    _cell_length_a 4.3
    loop_
    _symmetry_equiv_pos_as_xyz
    'x, y, z'
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_Cartn_x
    _atom_site_Cartn_y
    _atom_site_Cartn_z
    Fe1 Fe2+ 0.0 0.0 0.0
    O1  O2-  1.5 0.0 -0.25
    """
    a = CIFFiles.atoms(text)
    @test [x.symbol for x in a] == ["Fe", "O"]
    @test [x.Z for x in a] == [26, 8]
    # Oracle: the Å coordinates of the row times ANGSTROM_TO_BOHR, the one product the reader
    # computes: exact by construction.
    @test a[2].position == [1.5, 0.0, -0.25] .* ANGSTROM_TO_BOHR
    @test a[1].position == [0.0, 0.0, 0.0]
end

@testitem "cif: errors" setup = [CIFFiles] begin
    @test_throws "input: t.cif:7: cif: row has 3 values for 4 tags" CIFFiles.atoms(CIFFiles.loop(["O 0.0 0.0"]))
    @test_throws "input: t.cif:7: cif: coordinate \"x\" is not a number" CIFFiles.atoms(CIFFiles.loop(["O x 0.0 0.0"]))
    fract = CIFFiles.loop(
        ["O 0.0 0.0 0.0"];
        tags = ("_atom_site_type_symbol", "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")
    )
    for lenient in (false, true)
        @test_throws "input: t.cif:7: cif: only Cartesian _atom_site coordinates are supported" CIFFiles.atoms(fract; lenient)
    end
    nosym = CIFFiles.loop(
        ["O1 0.0 0.0 0.0"];
        tags = ("_atom_site_label", "_atom_site_Cartn_x", "_atom_site_Cartn_y", "_atom_site_Cartn_z")
    )
    @test_throws "input: t.cif:7: cif: the _atom_site loop needs _atom_site_type_symbol and _atom_site_Cartn_x, _y and _z" CIFFiles.atoms(nosym)
    # Lenient mode skips that loop; with no other loop, the file has no atom.
    @test_throws "input: t.cif: cif: no _atom_site loop with Cartesian coordinates" CIFFiles.atoms(nosym; lenient = true)
    @test_throws "input: t.cif:7: cif: unknown element symbol \"Xx\"" CIFFiles.atoms(CIFFiles.loop(["Xx 0.0 0.0 0.0"]))
    @test_throws "input: t.cif: cif: no _atom_site loop with Cartesian coordinates" CIFFiles.atoms("data_t\n_cell_length_a 4.3\n"; lenient = true)
end

@testitem "cif: lenient" setup = [CIFFiles] begin
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    @testset ":cif_row, a row of 3 values for 4 tags" begin
        # Oracle: io/molecule.rs:143 skips a row whose value count differs from the tag count.
        text = CIFFiles.loop(["O 0.0 0.0", "H 0.0 0.0 1.0"])
        @test_throws "input: t.cif:7: cif: row has 3 values for 4 tags" CIFFiles.atoms(text)
        w = InputWarning[]
        a = CIFFiles.atoms(text; lenient = true, warnings = w)
        @test [x.symbol for x in a] == ["H"]
        @test length(w) == 1
        @test w[1].case === :cif_row
        @test w[1].line == 7
    end
    @testset ":cif_coordinate, the coordinate x" begin
        # Oracle: io/molecule.rs:164-166 reads a coordinate that does not parse as 0.0.
        text = CIFFiles.loop(["O x 0.0 1.0"])
        @test_throws "input: t.cif:7: cif: coordinate \"x\" is not a number" CIFFiles.atoms(text)
        w = InputWarning[]
        a = only(CIFFiles.atoms(text; lenient = true, warnings = w))
        @test a.position == [0.0, 0.0, 1.0 * ANGSTROM_TO_BOHR]
        @test length(w) == 1
        @test w[1].case === :cif_coordinate
        @test w[1].line == 7
    end
    @testset ":cif_columns, a second loop without the type symbol" begin
        # Oracle: io/molecule.rs:159-173 adds no atom for a row of a loop that lacks the type symbol
        # or a Cartesian coordinate, and returns the atoms of the other loops.
        text = CIFFiles.loop(["O 0.0 0.0 1.0"]) * join(
            ["loop_", "_atom_site_label", "_atom_site_Cartn_x", "_atom_site_Cartn_y", "_atom_site_Cartn_z", "H1 0 0 0", "H2 0 0 1"],
            "\n",
        )
        msg = "input: t.cif:13: cif: the _atom_site loop needs _atom_site_type_symbol and _atom_site_Cartn_x, _y and _z"
        @test_throws msg CIFFiles.atoms(text)
        w = InputWarning[]
        a = only(CIFFiles.atoms(text; lenient = true, warnings = w))
        @test a.symbol == "O"
        @test a.position == [0.0, 0.0, 1.0 * ANGSTROM_TO_BOHR]
        @test length(w) == 1
        @test w[1].case === :cif_columns
        @test w[1].line == 13
    end
    @testset ":nonfinite, the coordinate nan" begin
        # Oracle: io/molecule.rs:164-166 parses each coordinate as an `f64`, which reads `nan`.
        text = CIFFiles.loop(["O nan 0.0 1.0"])
        @test_throws "input: t.cif:7: cif: expected a finite number, got \"nan\"" CIFFiles.atoms(text)
        w = InputWarning[]
        a = only(CIFFiles.atoms(text; lenient = true, warnings = w))
        @test isnan(a.position[1])
        @test a.position[3] == 1.0 * ANGSTROM_TO_BOHR
        @test length(w) == 1
        @test w[1].case === :nonfinite
        @test w[1].line == 7
    end
end
