@testmodule MoldenFiles begin
    using QuantumEngine
    using QuantumEngine.IO: MoldenShell, MoldenOrbitals, MoldenData, write_molden, read_molden
    using Random: Xoshiro, randn, rand

    # A water molecule in bohr: O with the shells s, s, p and d, each H with one s shell; 1 + 1 +
    # 3 + 6 + 1 + 1 = 13 Cartesian atomic orbitals. The exponents, contraction coefficients and
    # orbitals come from a seeded generator, with `T` elements and `nmo` orbitals per spin.
    function water(::Type{T} = Float64; beta = true, nmo = 13, seed = 20261125) where {T}
        rng = Xoshiro(seed)
        atoms = [
            Atom("O", T[0, 0, 0.2216648]), Atom("H", T[0, 1.4308, -0.8866]), Atom("H", T[0, -1.4308, -0.8866]),
        ]
        shell(atom, l, n) = MoldenShell(atom, l, T.(100 .* rand(rng, n) .+ 0.1), T.(randn(rng, n)))
        shells = [shell(1, 0, 6), shell(1, 0, 3), shell(1, 1, 3), shell(1, 2, 1), shell(2, 0, 3), shell(3, 0, 3)]
        orbitals() = MoldenOrbitals(
            T.(sort(randn(rng, nmo))), T[i <= 5 ? 1 : 0 for i in 1:nmo], T.(randn(rng, 13, nmo))
        )
        return MoldenData("water, seeded", atoms, shells, orbitals(), beta ? orbitals() : nothing)
    end

    text(d) = sprint(write_molden, d)
    load(s) = read_molden(IOBuffer(s); source = "t.molden")

    # Whether the Molden data `a` and `b` are equal field by field, through the atoms, shells and
    # orbitals, by the Fixtures helper `fields_equal` passed in, which also requires equal types.
    function same(fields_equal, a::MoldenData, b::MoldenData)
        typeof(a) === typeof(b) && a.title == b.title || return false
        length(a.atoms) == length(b.atoms) && all(fields_equal(s, t) for (s, t) in zip(a.atoms, b.atoms)) ||
            return false
        length(a.shells) == length(b.shells) && all(fields_equal(s, t) for (s, t) in zip(a.shells, b.shells)) ||
            return false
        fields_equal(a.alpha, b.alpha) || return false
        isnothing(a.beta) && return isnothing(b.beta)
        return !isnothing(b.beta) && fields_equal(a.beta, b.beta)
    end
end

@testitem "molden: round trip" setup = [Fixtures, MoldenFiles] begin
    # Shortest round-trip printing reads back every Float64 exactly, so the data compare with `==`.
    for beta in (true, false)
        d = MoldenFiles.water(; beta)
        back = MoldenFiles.load(MoldenFiles.text(d))
        @test back isa QuantumEngine.IO.MoldenData{Float64}
        @test MoldenFiles.same(Fixtures.fields_equal, back, d)
        @test isnothing(back.beta) == !beta
        @test size(back.alpha.coefficients) == (13, 13)
    end
    # Float32 data reads back as the Float64 values of its elements, exactly.
    d32 = MoldenFiles.water(Float32)
    back = MoldenFiles.load(MoldenFiles.text(d32))
    @test back.alpha.coefficients == Float64.(d32.alpha.coefficients)
    @test back.shells[3].exponents == Float64.(d32.shells[3].exponents)
    @test [a.position for a in back.atoms] == [Float64.(a.position) for a in d32.atoms]
    # An empty title and a ghost atom (Z = 0) read back.
    w = MoldenFiles.water(; beta = false)
    atoms = [w.atoms; Atom("H@", [0.0, 0.0, 3.0])]
    g = QuantumEngine.IO.MoldenData("", atoms, w.shells, w.alpha)
    @test MoldenFiles.same(Fixtures.fields_equal, MoldenFiles.load(MoldenFiles.text(g)), g)
end

@testitem "molden: written text" setup = [MoldenFiles] begin
    using QuantumEngine.IO: MoldenShell, MoldenOrbitals, MoldenData
    d = MoldenData(
        "H2", [Atom("H", [0.0, 0.0, -0.7]), Atom("H", [0.0, 0.0, 0.7])],
        [MoldenShell(1, 0, [3.42525091, 0.62391373], [0.15432897, 0.53532814]), MoldenShell(2, 0, [1.0], [1.0])],
        MoldenOrbitals([-0.5], [2.0], reshape([0.5, -1.0e-5], 2, 1)),
        MoldenOrbitals([-0.25], [0.0], reshape([0.0, 1.0], 2, 1)),
    )
    @test MoldenFiles.text(d) == """
        [Molden Format]
        [Title]
        H2
        [Atoms] AU
        H 1 1 0.0 0.0 -0.7
        H 2 1 0.0 0.0 0.7
        [GTO]
        1 0
        s 2 1.00
        3.42525091 0.15432897
        0.62391373 0.53532814

        2 0
        s 1 1.00
        1.0 1.0

        [MO]
        Sym= A
        Ene= -0.5
        Spin= Alpha
        Occup= 2.0
        1 0.5
        2 -1.0e-5
        Sym= A
        Ene= -0.25
        Spin= Beta
        Occup= 0.0
        1 0.0
        2 1.0
        """
end

@testitem "molden: shell labels" setup = [Fixtures, MoldenFiles] begin
    using QuantumEngine.IO: MoldenShell, MoldenOrbitals, MoldenData
    labels = String[]
    for l in 0:4
        d = MoldenData(
            "t", [Atom("Ne", [0.0, 0.0, 0.0])], [MoldenShell(1, l, [1.0], [1.0])],
            MoldenOrbitals([0.0], [0.0], ones((l + 1) * (l + 2) ÷ 2, 1)),
        )
        s = MoldenFiles.text(d)
        push!(labels, split(s, '\n')[8])
        @test MoldenFiles.same(Fixtures.fields_equal, MoldenFiles.load(s), d)
    end
    @test labels == ["s 1 1.00", "p 1 1.00", "d 1 1.00", "f 1 1.00", "g 1 1.00"]
    @test_throws "molden: shell with l = 5 has no Molden label" MoldenShell(1, 5, [1.0], [1.0])
    @test_throws "molden: shell with l = -1 has no Molden label" MoldenShell(1, -1, [1.0], [1.0])
end

@testitem "molden: reading" setup = [MoldenFiles] begin
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    # The layout of the oracle's writer (io/molden.rs): no [Title], `[Atoms] (Angstrom)`, shells
    # without a scale factor and `Sym=A`. The numbers are made up.
    oracle_layout = """
    [Molden Format]
    [Atoms] (Angstrom)
     O     1  8     0.000000     0.000000     0.117300
     H     2  1     0.000000     0.757200    -0.469200
    [GTO]
      1 0
     s 1
        130.7093200000       0.1543289700

      2 0
     s 1
          3.4252509100       0.1543289700

    [MO]
     Sym=A
     Ene= -20.251577
     Spin= Alpha
     Occup= 2.000000
        1  -0.99421000
        2   0.02593200
    """
    d = MoldenFiles.load(oracle_layout)
    @test d.title == ""
    @test [a.symbol for a in d.atoms] == ["O", "H"]
    # Oracle: the Å values times ANGSTROM_TO_BOHR, the one product the reader computes: exact.
    @test d.atoms[1].position == [0.0, 0.0, 0.1173 * ANGSTROM_TO_BOHR]
    @test d.atoms[2].position == [0.0, 0.7572, -0.4692] .* ANGSTROM_TO_BOHR
    @test [(s.atom, s.l, s.exponents, s.coefficients) for s in d.shells] ==
        [(1, 0, [130.70932], [0.15432897]), (2, 0, [3.42525091], [0.15432897])]
    @test d.alpha.energies == [-20.251577]
    @test d.alpha.occupations == [2.0]
    @test d.alpha.coefficients == reshape([-0.99421, 0.025932], 2, 1)
    @test isnothing(d.beta)
    # Each unit spelling, in any case; bohr values are kept as written.
    for u in ("AU", "(AU)", "Bohr", "au", "(au)", "BOHR")
        @test MoldenFiles.load(replace(oracle_layout, "(Angstrom)" => u)).atoms[2].position == [0.0, 0.7572, -0.4692]
    end
    for u in ("Angs", "(Angstrom)", "Angstrom", "ANGS", "angstrom", "(ANGSTROM)")
        @test MoldenFiles.load(replace(oracle_layout, "(Angstrom)" => u)).atoms[2].position ==
            [0.0, 0.7572, -0.4692] .* ANGSTROM_TO_BOHR
    end
    # Fortran D exponents; section names and keywords in any case; Spin= absent means Alpha;
    # omitted AO lines are zero; a Beta orbital; CRLF line endings.
    other = """
    [MOLDEN FORMAT]
    [title]
    water
    [atoms] au
    O 1 8 0.0 0.0 0.2D0
    H 2 1 0.0 1.0d0 -0.5
    [gto]
    1 0
    P 1 1.00
    1.5D+01 1.0D-01
    2 0
    s 1 1.0
    0.5 1.0
    [mo]
    ene= -1.5D0
    OCCUP= 1.0
    2 0.25
    4 -0.5
    Sym= A
    Ene= 0.5
    SPIN= beta
    Occup= 0
    1 1.0
    """
    d = MoldenFiles.load(replace(other, "\n" => "\r\n"))
    @test d.title == "water"
    @test d.atoms[1].position == [0.0, 0.0, 0.2]
    @test d.atoms[2].position == [0.0, 1.0, -0.5]
    @test d.shells[1].l == 1
    @test d.shells[1].exponents == [15.0]
    @test d.shells[1].coefficients == [0.1]
    @test d.alpha.energies == [-1.5]
    @test d.alpha.coefficients == reshape([0.0, 0.25, 0.0, -0.5], 4, 1)
    @test d.beta.energies == [0.5]
    @test d.beta.occupations == [0.0]
    @test d.beta.coefficients == reshape([1.0, 0.0, 0.0, 0.0], 4, 1)
end

@testitem "molden: reader errors" setup = [MoldenFiles] begin
    head = "[Molden Format]\n[Atoms] AU\nH 1 1 0 0 0\n[GTO]\n1 0\ns 1 1.00\n1.0 1.0\n\n[MO]\n"
    orbital = "Ene= -0.5\nOccup= 1.0\n1 1.0\n"
    @test MoldenFiles.load(head * orbital).alpha.energies == [-0.5]
    err(s) = MoldenFiles.load(s)
    @test_throws "input: t.molden:3: molden: unknown [Atoms] unit \"nm\"" err("[Molden Format]\n\n[Atoms] nm\n")
    @test_throws "input: t.molden:2: molden: unknown [Atoms] unit \"\"" err("[Molden Format]\n[Atoms]\n")
    @test_throws "input: t.molden:9: molden: unknown section [5D]" err(replace(head, "[MO]" => "[5D]"))
    @test_throws "input: t.molden:1: molden: the file starts with [Molden Format]" err("[Atoms] AU\n")
    @test_throws "input: t.molden:1: molden: the file starts with [Molden Format]" err("H 1 1 0 0 0\n")
    @test_throws "input: t.molden: molden: the file starts with [Molden Format]" err("\n\n")
    @test_throws "input: t.molden:2: molden: the file starts with [Molden Format]" err("[Molden Format]\n[Molden Format]\n")
    @test_throws "input: t.molden:2: molden: a section header ends with ]" err("[Molden Format]\n[Atoms AU\n")
    @test_throws "input: t.molden:2: molden: a line outside a section" err("[Molden Format]\nx\n")
    @test_throws "input: t.molden:4: molden: [Title] holds one line" err("[Molden Format]\n[Title]\na\nb\n")
    @test_throws "input: t.molden:4: molden: [Title] comes before [Atoms]" err("[Molden Format]\n[Atoms] AU\nH 1 1 0 0 0\n[Title]\n")
    @test_throws "input: t.molden:1: molden: unexpected text after [Molden Format]" err("[Molden Format] x\n")
    @test_throws "input: t.molden:4: molden: section [Atoms] is given twice" err("[Molden Format]\n[Atoms] AU\nH 1 1 0 0 0\n[Atoms] AU\n")
    @test_throws "input: t.molden:2: molden: [GTO] comes after [Atoms]" err("[Molden Format]\n[GTO]\n")
    @test_throws "input: t.molden:4: molden: [MO] comes after [GTO]" err("[Molden Format]\n[Atoms] AU\nH 1 1 0 0 0\n[MO]\n")
    # [Atoms]
    @test_throws "input: t.molden:3: molden: expected <symbol> <index> <Z> <x> <y> <z>" err("[Molden Format]\n[Atoms] AU\nH 1 1 0 0\n")
    @test_throws "input: t.molden:3: molden: atom index 2, expected 1" err("[Molden Format]\n[Atoms] AU\nH 2 1 0 0 0\n")
    @test_throws "input: t.molden:3: molden: expected an integer, got \"x\"" err("[Molden Format]\n[Atoms] AU\nH 1 x 0 0 0\n")
    # Integers are decimal: a base prefix, which `tryparse(Int, _)` accepts, is rejected.
    for t in ("0x1", "0b1", "0o1")
        @test_throws "input: t.molden:3: molden: expected an integer, got \"$(t)\"" err("[Molden Format]\n[Atoms] AU\nH $(t) 1 0 0 0\n")
    end
    @test_throws "input: t.molden:3: molden: atomic number 2 does not match the symbol \"H\"" err("[Molden Format]\n[Atoms] AU\nH 1 2 0 0 0\n")
    @test_throws "input: t.molden:3: molden: unknown element symbol \"Xx\"" err("[Molden Format]\n[Atoms] AU\nXx 1 0 0 0 0\n")
    @test_throws "input: t.molden:3: molden: a Bq atom needs an element" err("[Molden Format]\n[Atoms] AU\nBq 1 0 0 0 0\n")
    @test_throws "input: t.molden:3: molden: expected a number, got \"0x1p3\"" err("[Molden Format]\n[Atoms] AU\nH 1 1 0x1p3 0 0\n")
    @test_throws "input: t.molden:3: molden: expected a finite number, got \"nan\"" err("[Molden Format]\n[Atoms] AU\nH 1 1 nan 0 0\n")
    @test_throws "input: t.molden:3: molden: expected a finite number, got \"1D400\"" err("[Molden Format]\n[Atoms] AU\nH 1 1 1D400 0 0\n")
    # [GTO]
    gto(body) = "[Molden Format]\n[Atoms] AU\nH 1 1 0 0 0\nH 2 1 0 0 1\n[GTO]\n" * body
    @test_throws "input: t.molden:6: molden: a shell line comes after its <atom> 0 line" err(gto("s 1 1.00\n"))
    @test_throws "input: t.molden:6: molden: an atom line of [GTO] is <atom> 0" err(gto("1 1\n"))
    @test_throws "input: t.molden:6: molden: [GTO] names atom 3, but there are 2 atoms" err(gto("3 0\n"))
    @test_throws "input: t.molden:7: molden: [GTO] atom 1 follows atom 2" err(gto("2 0\n1 0\n"))
    @test_throws "input: t.molden:7: molden: unknown shell label \"sp\"" err(gto("1 0\nsp 1 1.00\n"))
    @test_throws "input: t.molden:7: molden: expected <label> <nprim> [1.00]" err(gto("1 0\ns 1 1.00 x\n"))
    @test_throws "input: t.molden:7: molden: a shell has at least one primitive" err(gto("1 0\ns 0 1.00\n"))
    @test_throws "input: t.molden:7: molden: expected an integer, got \"0x1\"" err(gto("1 0\ns 0x1 1.00\n"))
    # A prefixed atom index is not an atom line, so it is read as a shell line.
    @test_throws "input: t.molden:6: molden: a shell line comes after its <atom> 0 line" err(gto("0x1 0\n"))
    @test_throws "input: t.molden:9: molden: unknown shell label \"0x2\"" err(gto("1 0\ns 1\n1.0 1.0\n0x2 0\n"))
    @test_throws "input: t.molden:7: molden: shell scale factor \"1.5\" is not 1" err(gto("1 0\ns 1 1.5\n"))
    @test_throws "input: t.molden:8: molden: expected an exponent and a coefficient" err(gto("1 0\ns 1\n1.0\n"))
    @test_throws "input: t.molden:8: molden: expected a positive exponent, got \"-1.0\"" err(gto("1 0\ns 1\n-1.0 1.0\n"))
    @test_throws "input: t.molden:7: molden: the shell declares 2 primitives but has 1" err(gto("1 0\ns 2\n1.0 1.0\n[MO]\n"))
    @test_throws "input: t.molden:7: molden: the shell declares 2 primitives but has 1" err(gto("1 0\ns 2\n1.0 1.0\n"))
    # [MO]
    @test_throws "input: t.molden:10: molden: unknown [MO] keyword \"energy\"" err(head * "Energy= 1\n")
    @test_throws "input: t.molden:10: molden: unknown spin \"Gamma\"" err(head * "Spin= Gamma\n")
    @test_throws "input: t.molden:10: molden: a coefficient line comes after its orbital's keyword lines" err(head * "1 1.0\n")
    @test_throws "input: t.molden:12: molden: expected <ao> <coefficient>" err(head * "Ene= 0\nOccup= 1\n1\n")
    @test_throws "input: t.molden:12: molden: atomic orbital 2 is outside 1:1" err(head * "Ene= 0\nOccup= 1\n2 1.0\n")
    @test_throws "input: t.molden:12: molden: atomic orbital 0 is outside 1:1" err(head * "Ene= 0\nOccup= 1\n0 1.0\n")
    @test_throws "input: t.molden:12: molden: expected an integer, got \"0x1\"" err(head * "Ene= 0\nOccup= 1\n0x1 1.0\n")
    @test_throws "input: t.molden:10: molden: the orbital has no Occup= line" err(head * "Ene= 0\n1 1.0\n")
    @test_throws "input: t.molden:10: molden: the orbital has no Ene= line" err(head * "Occup= 0\n1 1.0\nEne= 0\n")
    @test_throws "input: t.molden:10: molden: the orbital has no Occup= line" err(head * "Ene= 0\nEne= 1\nOccup= 1\n")
    @test_throws "input: t.molden: molden: the file has no alpha orbitals" err(head)
    @test_throws "input: t.molden: molden: the file has no alpha orbitals" err(head * "Spin= Beta\n" * orbital)
    @test_throws "input: t.molden: molden: the file has no atoms" err("[Molden Format]\n[Atoms] AU\n[GTO]\n[MO]\n")
    @test_throws "input: t.molden:12: molden: expected a finite number, got \"inf\"" err(head * "Ene= 0\nOccup= 1\n1 inf\n")
end

@testitem "molden: reader errors, atomic orbital order" setup = [MoldenFiles] begin
    head = "[Molden Format]\n[Atoms] AU\nH 1 1 0 0 0\n[GTO]\n1 0\np 1 1.00\n1.0 1.0\n\n[MO]\n"
    @test_throws "input: t.molden:13: molden: atomic orbital 1 follows 2" MoldenFiles.load(head * "Ene= 0\nOccup= 1\n2 1.0\n1 1.0\n")
end

@testitem "molden: constructor errors" setup = [MoldenFiles] begin
    using QuantumEngine.IO: MoldenShell, MoldenOrbitals, MoldenData
    @test_throws "molden: 3 exponents but 2 coefficients" MoldenShell(1, 0, [1.0, 2.0, 3.0], [1.0, 2.0])
    @test_throws "molden: a shell's atom index is at least 1, got 0" MoldenShell(0, 0, [1.0], [1.0])
    @test_throws "molden: a shell has at least one primitive" MoldenShell(1, 0, Float64[], Float64[])
    @test_throws "molden: shell exponents must be finite and positive" MoldenShell(1, 0, [0.0], [1.0])
    @test_throws "molden: shell exponents must be finite and positive" MoldenShell(1, 0, [Inf], [1.0])
    @test_throws "molden: shell coefficients must be finite" MoldenShell(1, 0, [1.0], [NaN])
    @test_throws "molden: 2 energies but 1 occupations" MoldenOrbitals([1.0, 2.0], [1.0], ones(1, 2))
    @test_throws "molden: 2 energies but the coefficient matrix has 3 columns" MoldenOrbitals([1.0, 2.0], [1.0, 1.0], ones(1, 3))
    @test_throws "molden: orbital energies, occupations and coefficients must be finite" MoldenOrbitals([NaN], [1.0], ones(1, 1))
    @test_throws "molden: orbital energies, occupations and coefficients must be finite" MoldenOrbitals([1.0], [1.0], fill(Inf, 1, 1))
    w = MoldenFiles.water(; beta = false)
    build(; title = w.title, atoms = w.atoms, shells = w.shells, alpha = w.alpha, beta = nothing) =
        MoldenData(title, atoms, shells, alpha, beta)
    @test_throws "molden: a title is one line" build(; title = "a\nb")
    @test_throws "molden: a title is one line" build(; title = "a\rb")
    @test_throws "molden: a title has no leading or trailing whitespace" build(; title = " a")
    @test_throws "molden: a title has no leading or trailing whitespace" build(; title = "a ")
    @test_throws "molden: a title does not start with [" build(; title = "[a]")
    @test_throws "molden: the data has no atoms" build(; atoms = Atom{Float64}[], shells = MoldenShell{Float64}[])
    @test_throws "molden: the data has no alpha orbitals" build(; alpha = MoldenOrbitals(Float64[], Float64[], zeros(13, 0)))
    @test_throws "molden: atom 2 has a non-finite position" build(; atoms = [w.atoms[1], Atom("H", [0.0, NaN, 0.0]), w.atoms[3]])
    @test_throws "molden: shell 6 is on atom 3, but there are 2 atoms" build(; atoms = w.atoms[1:2])
    @test_throws "molden: shells are ordered by atom; shell 2 on atom 1 follows atom 2" build(; shells = [w.shells[5]; w.shells[1:4]; w.shells[6]])
    @test_throws "molden: the shells have 12 atomic orbitals but the alpha coefficients have 13 rows" build(; shells = w.shells[1:5])
    beta = MoldenOrbitals([0.0], [0.0], ones(12, 1))
    @test_throws "molden: the shells have 13 atomic orbitals but the beta coefficients have 12 rows" build(; beta)
end

@testitem "molden: generic indexing" setup = [Fixtures, MoldenFiles] begin
    using QuantumEngine.IO: MoldenShell, MoldenOrbitals, MoldenData
    using OffsetArrays
    msg = "offset arrays are not supported but got an array with index other than 1"
    @test_throws msg MoldenOrbitals([1.0], [1.0], OffsetArray(ones(1, 1), 0:0, 1:1))
    @test_throws msg MoldenOrbitals(OffsetArray([1.0], 0:0), [1.0], ones(1, 1))
    @test_throws msg MoldenShell(1, 0, OffsetArray([1.0], 0:0), [1.0])
    w = MoldenFiles.water(; beta = false)
    @test_throws msg MoldenData(w.title, OffsetArray(w.atoms, 0:2), w.shells, w.alpha)
    @test_throws msg MoldenData(w.title, w.atoms, OffsetArray(w.shells, 0:5), w.alpha)
    # Views are copied into the fields and give the same data.
    c = ones(13, 4)
    o = MoldenOrbitals(view([1.0, 2.0, 3.0], 1:2), view([1.0, 0.0, 9.0], 1:2), view(c, :, 2:3))
    @test o.energies == [1.0, 2.0] && o.coefficients == ones(13, 2)
    s = MoldenShell(1, 0, view([1.0, 2.0], 2:2), view([3.0, 4.0], 1:1))
    @test s.exponents == [2.0] && s.coefficients == [3.0]
    d = MoldenData(w.title, view(w.atoms, 1:3), view(w.shells, 1:6), w.alpha)
    @test MoldenFiles.same(Fixtures.fields_equal, d, w)
end

# The water PBE/6-31G Molden file the oracle wrote (Å coordinates, alpha orbitals only). Kind (a):
# the oracle prints Å with 6 decimals (io/molden.rs:23, {:12.6}), half a unit 5e-7 Å, ×1/0.529177
# ≈ 9.5e-7 bohr, within 1e-6 of the O position in bohr.
@testitem "molden: the oracle's file" setup = [OracleHarness] begin
    using QuantumEngine.IO: read_molden
    path = OracleHarness.captured("water_pbe_631g.molden")
    d = open(io -> read_molden(io; source = basename(path)), path)
    @test length(d.atoms) == 3
    @test abs(d.atoms[1].position[3] - 0.22166487) ≤ 1.0e-6
    @test size(d.alpha.coefficients, 1) == 13
    @test isnothing(d.beta)
end
