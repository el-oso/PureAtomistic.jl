# The molecular basis: the shells and AO counts of `build_basis` on the shipped data, the shell and
# pair order, the basis-name aliases, ghost atoms, the element type, generic indexing, the errors
# and the log lines.

@testmodule BuildData begin
    using QuantumEngine
    using QuantumEngine: Atom, Molecule
    using QuantumEngine.Units: ANGSTROM_TO_BOHR

    "The shipped basis directory alone, so that no search directory of the runner is read."
    const PKGDATA = [joinpath(pkgdir(QuantumEngine), "data", "basis_sets")]

    """
        adenine() -> Molecule{Float64}

    Adenine (C5H5N5), the 15 atoms of the oracle test `tests/test_adenine_m062x.rs:26-42`, its Å
    coordinates converted to bohr.
    """
    adenine() = Molecule(
        [
            Atom(s, [x, y, z] .* ANGSTROM_TO_BOHR) for (s, x, y, z) in (
                    ("N", -0.5905, -1.3533, 0.0), ("C", -0.8412, -0.0895, 0.0), ("N", 0.2205, 0.7297, 0.0),
                    ("C", 1.5367, 0.5054, 0.0), ("C", 2.1265, -0.8175, 0.0), ("N", 3.4839, -0.9575, 0.0),
                    ("C", 3.8055, 0.2199, 0.0), ("N", 2.8808, 1.1444, 0.0), ("C", 1.4285, -1.8906, 0.0),
                    ("N", 0.1065, -1.8105, 0.0), ("H", -1.4883, -1.785, 0.0), ("H", 4.0538, -1.8385, 0.0),
                    ("H", 4.7797, 0.6272, 0.0), ("H", 2.1121, -2.9238, 0.0), ("H", -0.6358, -2.5202, 0.0),
                )
        ]
    )

    "A one-atom molecule of element `sym` at the origin."
    single(sym) = Molecule([Atom(sym, [0.0, 0.0, 0.0])])
end

@testitem "gbs: 6-31g water shells" setup = [Fixtures, BuildData] begin
    using QuantumEngine.Basis: build_basis, BasisSet
    bs = build_basis(Fixtures.water(), "6-31g"; paths = BuildData.PKGDATA)
    @test bs isa BasisSet{Float64}
    # The oracle's log of water with 6-31g reports `AO basis: 13`.
    @test bs.nao == 13
    @test length(bs.shells) == 9
    @test [s.atom for s in bs.shells] == [1, 1, 1, 1, 1, 2, 2, 3, 3]
    @test [s.l for s in bs.shells] == [0, 0, 1, 0, 1, 0, 0, 0, 0]
    @test [s.ao_offset for s in bs.shells] == [0, 1, 2, 5, 6, 9, 10, 11, 12]
    @test bs.name == "6-31g"
    mol = Fixtures.water()
    @test all(s.center == mol.atoms[s.atom].position for s in bs.shells)
end

@testitem "gbs: sto-3g water" setup = [Fixtures, BuildData] begin
    using QuantumEngine.Basis: build_basis
    @test build_basis(Fixtures.water(), "sto-3g"; paths = BuildData.PKGDATA).nao == 7
    @test build_basis(Fixtures.h2(), "sto-3g"; paths = BuildData.PKGDATA).nao == 2
end

@testitem "basis: more AO counts" setup = [Fixtures, BuildData] begin
    using QuantumEngine.Basis: build_basis
    star = build_basis(Fixtures.water(), "6-31g*"; paths = BuildData.PKGDATA)
    @test (star.nao, length(star.shells)) == (19, 10)
    star2 = build_basis(Fixtures.water(), "6-31G**"; paths = BuildData.PKGDATA)
    @test (star2.nao, length(star2.shells)) == (25, 12)
    @test star2.name == "6-31G**"
end

@testitem "adenine 6-311++g(2d,2p) nao" setup = [BuildData] begin
    using QuantumEngine.Basis: build_basis
    # The oracle test asserts 340 AOs for this molecule and basis.
    bs = build_basis(BuildData.adenine(), "6-311++g(2d,2p)"; paths = BuildData.PKGDATA)
    @test bs.nao == 340
    @test length(bs.shells) == 140
end

@testitem "shell and pair order" setup = [Fixtures, BuildData] begin
    using QuantumEngine.Basis: build_basis, parse_gbs, locate_basis_file, ao_range, shell_pairs, ncart
    mol = Fixtures.water()
    bs = build_basis(mol, "6-31g"; paths = BuildData.PKGDATA)
    # Atom order, then file order within an atom.
    file = locate_basis_file("6-31g"; paths = BuildData.PKGDATA)
    data = open(io -> parse_gbs(io; source = file), file)
    expected = [(i, spec.l) for (i, sym) in enumerate(("O", "H", "H")) for spec in data[sym].shells]
    @test [(s.atom, s.l) for s in bs.shells] == expected
    # Contiguous AO ranges in shell order.
    @test first(ao_range(bs.shells[1])) == 1
    @test all(last(ao_range(bs.shells[k])) + 1 == first(ao_range(bs.shells[k + 1])) for k in 1:8)
    @test last(ao_range(bs.shells[end])) == bs.nao
    @test sum(s -> ncart(s.l), bs.shells) == bs.nao
    @test collect(shell_pairs(bs)) == [(i, j) for i in 1:9 for j in 1:i]
end

@testitem "aliases" setup = [BuildData] begin
    using QuantumEngine.Basis: locate_basis_file
    base(n) = basename(locate_basis_file(n; paths = BuildData.PKGDATA))
    @test base("6-31g*") == "6-31g_star.gbs"
    @test base("6-31g(d)") == "6-31g_star.gbs"
    @test base("6-31G**") == "6-31g_star_star.gbs"
    @test base("6-31+g*") == "6-31plusg_star.gbs"
    @test base("6-311++g(d,p)") == "6-311plusplusg_star_star.gbs"
    @test base("def2-SVP-JKFIT") == "def2-universal-jkfit.gbs"
    @test base("631-g") == "6-31g.gbs"
end

@testitem "ghost basis" setup = [Fixtures, BuildData] begin
    using QuantumEngine: Atom, Molecule
    using QuantumEngine.Basis: build_basis
    real = Fixtures.water()
    ref = build_basis(real, "6-31g"; paths = BuildData.PKGDATA)
    real_o = [s for s in ref.shells if s.atom == 1]
    for sym in ("O@", "BqO", "Bq-O")
        mol = Molecule([Atom(sym, real.atoms[1].position); real.atoms[2:3]])
        @test mol.atoms[1].Z == 0
        bs = build_basis(mol, "6-31g"; paths = BuildData.PKGDATA)
        o = [s for s in bs.shells if s.atom == 1]
        @test length(o) == length(real_o)
        @test [s.l for s in o] == [s.l for s in real_o]
        @test all(a.exponents == b.exponents for (a, b) in zip(o, real_o))
        @test all(a.coefficients == b.coefficients for (a, b) in zip(o, real_o))
        @test bs.nao == ref.nao
    end
end

@testitem "basis: Float32 flows through" setup = [Fixtures, BuildData] begin
    using QuantumEngine.Basis: build_basis, BasisSet
    bs = build_basis(Fixtures.water(), "6-31g", Float32; paths = BuildData.PKGDATA)
    @test bs isa BasisSet{Float32}
    @test eltype(bs.shells[1].coefficients) === Float32
    @test eltype(bs.shells[1].center) === Float32
    ref = build_basis(Fixtures.water(), "6-31g"; paths = BuildData.PKGDATA)
    @test all(a.exponents == Float32.(b.exponents) for (a, b) in zip(bs.shells, ref.shells))
    @test bs.nao == ref.nao
end

@testitem "basis: generic indexing" setup = [Fixtures, BuildData] begin
    using QuantumEngine: Molecule
    using QuantumEngine.Basis: build_basis
    using OffsetArrays: OffsetArray
    atoms = Fixtures.water().atoms
    ref = build_basis(Molecule(atoms), "6-31g"; paths = BuildData.PKGDATA)
    off = build_basis(Molecule(OffsetArray(atoms, -1:1)), "6-31g"; paths = BuildData.PKGDATA)
    @test [s.atom for s in off.shells] == [-1, -1, -1, -1, -1, 0, 0, 1, 1]
    @test off.nao == ref.nao
    @test [s.ao_offset for s in off.shells] == [s.ao_offset for s in ref.shells]
    v = build_basis(Molecule(view(atoms, :)), "6-31g"; paths = BuildData.PKGDATA)
    @test v.nao == ref.nao && v.name == ref.name
    @test length(v.shells) == length(ref.shells)
    @test all(Fixtures.fields_equal(a, b) for (a, b) in zip(v.shells, ref.shells))
end

@testitem "basis: build errors" setup = [Fixtures, BuildData] begin
    using QuantumEngine: Atom, Molecule
    using QuantumEngine.Basis: build_basis, basis_search_paths
    PKGDATA = BuildData.PKGDATA
    # An element the file lacks; no other file is tried.
    f631 = joinpath(PKGDATA[1], "6-31g.gbs")
    @test_throws "basis: $(f631): no basis for element Xe" build_basis(BuildData.single("Xe"), "6-31g"; paths = PKGDATA)
    # An I shell placed on an atom.
    @test_throws r"^basis: .*cc-pv6z\.gbs:406: I shell \(l = 6\) for C; this build supports l ≤ 5$" build_basis(
        BuildData.single("C"), "cc-pv6z"; paths = PKGDATA
    )
    # A file with I shells for other elements serves water.
    jk = build_basis(Fixtures.water(), "def2-universal-jkfit"; paths = PKGDATA)
    @test jk.nao > 0
    @test_throws "def2-universal-jkfit.gbs:4461: I shell (l = 6) for Ag; this build supports l ≤ 5" build_basis(
        BuildData.single("Ag"), "def2-universal-jkfit"; paths = PKGDATA
    )
    valid_h = "****\nH     0\nS   2   1.00\n      3.5   0.4\n      0.6   0.7\n****\n"
    # A file that fails to parse is not skipped in favor of a later path.
    d1, d2 = realpath(mktempdir()), realpath(mktempdir())
    write(joinpath(d1, "sto-3g.gbs"), "H 0\nQ 1\n")
    write(joinpath(d2, "sto-3g.gbs"), valid_h)
    @test_throws "basis: $(joinpath(d1, "sto-3g.gbs")):2: unknown shell type" build_basis(
        Fixtures.h2(), "sto-3g"; paths = [d1, d2]
    )
    # A `basis_sets` directory in the working directory shadows the package data, and a partial
    # file there is not completed from it.
    tmp = realpath(mktempdir())
    mkpath(joinpath(tmp, "basis_sets"))
    local_file = joinpath(tmp, "basis_sets", "sto-3g.gbs")
    write(local_file, valid_h)
    cd(tmp) do
        paths = basis_search_paths(env = Dict{String, String}())
        @test last(paths) == PKGDATA[1]
        bs = build_basis(Fixtures.h2(), "sto-3g"; paths)
        @test all(s.exponents == [3.5, 0.6] for s in bs.shells)
        @test_throws "basis: $(local_file): no basis for element O" build_basis(Fixtures.water(), "sto-3g"; paths)
    end
    # An element with an ECP section and no basis block has no basis.
    ecp_dir = realpath(mktempdir())
    na_ecp = join(readlines(joinpath(PKGDATA[1], "lanl2dz.gbs"))[1521:1544], '\n') * "\n"
    @test startswith(na_ecp, "NA     0\nNA-ECP")
    write(joinpath(ecp_dir, "na-ecp.gbs"), na_ecp)
    @test_throws "basis: $(joinpath(ecp_dir, "na-ecp.gbs")): no basis for element Na" build_basis(
        BuildData.single("Na"), "na-ecp"; paths = [ecp_dir]
    )
    # Data that is valid in Float64 but not in the element type of the build.
    big = realpath(mktempdir())
    write(joinpath(big, "e-big.gbs"), "H 0\nS 1 1.00\n 1e39 1.0\n****\n")
    write(joinpath(big, "e-tiny.gbs"), "H 0\nS 1 1.00\n 1e-50 1.0\n****\n")
    write(joinpath(big, "c-big.gbs"), "H 0\nS 2 1.00\n 1.0 1e20\n 2.0 1e20\n****\n")
    write(joinpath(big, "c-tiny.gbs"), "H 0\nP 1 1.00\n 1.0 1e-50\n****\n")
    write(joinpath(big, "e-small.gbs"), "H 0\nS 2 1.00\n 1e-20 1.0\n 1.0 1.0\n****\n")
    h = BuildData.single("H")
    @test build_basis(h, "e-big"; paths = [big]).nao == 1
    @test_throws "basis: $(joinpath(big, "e-big.gbs")):2: exponent 1.0e39 of the S shell of H is not finite and positive in Float32" build_basis(
        h, "e-big", Float32; paths = [big]
    )
    @test_throws "e-tiny.gbs:2: exponent 1.0e-50 of the S shell of H is not finite and positive in Float32" build_basis(
        h, "e-tiny", Float32; paths = [big]
    )
    @test_throws "basis: $(joinpath(big, "c-big.gbs")):2: the norm of the S shell of H is not a positive finite Float32" build_basis(
        h, "c-big", Float32; paths = [big]
    )
    @test_throws "c-tiny.gbs:2: the norm of the P shell of H is not a positive finite Float32" build_basis(
        h, "c-tiny", Float32; paths = [big]
    )
    # 1e-20 is a normal Float32, but its square is not.
    @test_throws "e-small.gbs:2: the norm of the S shell of H is not a positive finite Float32" build_basis(
        h, "e-small", Float32; paths = [big]
    )
    @test build_basis(h, "e-small"; paths = [big]).nao == 1
    @test build_basis(h, "c-big"; paths = [big]).nao == 1
end

@testitem "basis: logs the resolved file" setup = [Fixtures, BuildData] begin
    using QuantumEngine.Basis: build_basis
    using QuantumEngine.IO: Logger
    buf = IOBuffer()
    build_basis(Fixtures.water(), "6-31g"; paths = BuildData.PKGDATA, logger = Logger(buf, devnull))
    @test split(String(take!(buf)), '\n'; keepempty = false) ==
        ["basis 6-31g: " * joinpath(BuildData.PKGDATA[1], "6-31g.gbs")]
    # Without a logger nothing is written: not to stdout, not to stderr, not as a log record.
    dir = mktempdir()
    out, err = joinpath(dir, "out"), joinpath(dir, "err")
    open(out, "w") do o
        open(err, "w") do e
            @test_logs redirect_stdio(stdout = o, stderr = e) do
                build_basis(Fixtures.water(), "6-31g"; paths = BuildData.PKGDATA, logger = nothing)
                build_basis(Fixtures.water(), "6-31g"; paths = BuildData.PKGDATA)
            end
        end
    end
    @test isempty(read(out)) && isempty(read(err))
    build_basis(Fixtures.water(), "def2-SVP-JKFIT"; paths = BuildData.PKGDATA, logger = Logger(buf, devnull))
    @test split(String(take!(buf)), '\n'; keepempty = false) == [
        "basis def2-SVP-JKFIT: the oracle's def2-SVP-JKFIT is a placeholder; using def2-universal-jkfit",
        "basis def2-SVP-JKFIT: " * joinpath(BuildData.PKGDATA[1], "def2-universal-jkfit.gbs"),
    ]
    # The real name of the file it resolves to writes no note.
    build_basis(Fixtures.water(), "def2-universal-jkfit"; paths = BuildData.PKGDATA, logger = Logger(buf, devnull))
    @test length(split(String(take!(buf)), '\n'; keepempty = false)) == 1
end

@testitem "basis: build from parsed data" setup = [Fixtures, BuildData] begin
    using QuantumEngine.Basis: build_basis, parse_gbs
    file = joinpath(BuildData.PKGDATA[1], "6-31g.gbs")
    data = open(io -> parse_gbs(io; source = file), file)
    bs = QuantumEngine.Basis._build_basis(Fixtures.water(), data, Float32, file, "6-31g")
    ref = build_basis(Fixtures.water(), "6-31g", Float32; paths = BuildData.PKGDATA)
    @test bs.nao == ref.nao == 13 && bs.name == ref.name
    @test all(Fixtures.fields_equal(a, b) for (a, b) in zip(bs.shells, ref.shells))
    @test length(bs.shells) == length(ref.shells)
    @test_throws "basis: <data>: no basis for element Xe" QuantumEngine.Basis._build_basis(
        BuildData.single("Xe"), data, Float64, "<data>", "6-31g"
    )
end
