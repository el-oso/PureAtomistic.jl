@testitem "electron counts" setup = [Fixtures] begin
    using QuantumEngine
    o2 = Fixtures.o2_triplet()
    @test (nelectrons(o2), nalpha(o2), nbeta(o2)) == (16, 9, 7)
    w = Fixtures.water()
    cation = Molecule(w.atoms; charge = 1, multiplicity = 2)
    @test (nalpha(cation), nbeta(cation)) == (5, 4)
    bad = Molecule(Fixtures.h2().atoms; multiplicity = 2)
    @test_throws "system: 2 electrons cannot have multiplicity 2" nalpha(bad)
    @test_throws "system: charge 3 exceeds the nuclear charge 2" nelectrons(Molecule(Fixtures.h2().atoms; charge = 3, multiplicity = 2))
    @test_throws "system: multiplicity must be at least 1, got 0" Molecule(Fixtures.h2().atoms; multiplicity = 0)
    # The parity rule in both spin counts, and an ArgumentError for it.
    @test_throws "system: 2 electrons cannot have multiplicity 2" nbeta(bad)
    @test_throws ArgumentError nalpha(bad)
    # More unpaired electrons than electrons: n − (mult − 1) is negative.
    @test_throws "system: 2 electrons cannot have multiplicity 5" nalpha(Molecule(Fixtures.h2().atoms; multiplicity = 5))
    @test (nalpha(Fixtures.h2()), nbeta(Fixtures.h2())) == (1, 1)
    @test (nelectrons(w), nalpha(w), nbeta(w)) == (10, 5, 5)
end

@testitem "nuclear repulsion" setup = [Fixtures] begin
    using QuantumEngine
    h2 = Fixtures.h2()
    # Oracle: the closed form 1/R at R = 1.4 bohr. Tolerance (b): one subtraction, one square root
    # and one division, each at most 1 ulp: 4 eps.
    @test isapprox(nuclear_repulsion(h2), 1 / 1.4; rtol = 4eps())
    # Oracle: a BigFloat recomputation from the literal Å coordinates of the input file and the
    # CODATA 2018 bohr radius, independent of the package code. Tolerance (b): 3 pairs of at most 4
    # roundings each, plus 1 ulp from the Å-to-bohr conversion: 8 eps (rounded up from 13 eps/2,
    # the relative error of a sum of positive terms being at most the largest term's).
    setprecision(BigFloat, 256) do
        a0 = big"0.529177210903"
        O = [big"0.0", big"0.0", big"0.1173"] ./ a0
        H1 = [big"0.0", big"0.7572", big"-0.4692"] ./ a0
        H2 = [big"0.0", big"-0.7572", big"-0.4692"] ./ a0
        dist(p, q) = sqrt(sum(abs2, p .- q))
        exact = 8 / dist(O, H1) + 8 / dist(O, H2) + 1 / dist(H1, H2)
        @test isapprox(nuclear_repulsion(Fixtures.water()), Float64(exact); rtol = 8eps())
    end
    @test nuclear_repulsion(h2) isa Float64
    # Two H atoms 300 bohr apart in Float16, where the plain sum of squares overflows (300^2 is
    # past floatmax(Float16)). Oracle: the closed form 1/R. Tolerance (b): the subtraction, the
    # scaling division, the square root, the rescaling product and the final division, each at
    # most 1 ulp: 5 eps(Float16).
    far = Molecule([Atom("H", Float16[0, 0, -150]), Atom("H", Float16[0, 0, 150])])
    @test nuclear_repulsion(far) isa Float16
    @test isapprox(nuclear_repulsion(far), 1 / 300; rtol = 5eps(Float16))
    # A molecule with one real atom has no pair.
    @test nuclear_repulsion(Molecule([Atom("He", [0.0, 0.0, 0.0])])) === 0.0
end

@testitem "Float32 flows through" setup = [Fixtures] begin
    using QuantumEngine
    w = Fixtures.water()
    atoms32 = [Atom(a.symbol, Float32.(a.position)) for a in w.atoms]
    w32 = Molecule(atoms32)
    @test w32 isa Molecule{Float32}
    @test eltype(atoms32) === Atom{Float32}
    @test atoms32[1].mass isa Float32
    e32 = nuclear_repulsion(w32)
    @test e32 isa Float32
    # Oracle: the Float64 value. Tolerance (b): rounding the positions to Float32 (1/2 ulp each)
    # and the 3 pairs of at most 4 roundings each in Float32: 8 eps(Float32).
    @test isapprox(e32, nuclear_repulsion(w); rtol = 8eps(Float32))
    @test (nelectrons(w32), nalpha(w32), nbeta(w32)) == (10, 5, 5)
end

@testitem "ghost atoms" setup = [Fixtures] begin
    using QuantumEngine
    g = Atom("O@", [0, 0, 0])
    @test g.Z == 0
    @test g.ghost
    @test basis_element(g) == 8
    @test g.mass == 0
    @test g.symbol == "O@"
    @test g isa Atom{Float64}
    for s in ("bq-h", "BqH", "BQ-H", "h@", "H@")
        a = Atom(s, [0.0, 0.0, 0.0])
        @test (a.Z, a.ghost, basis_element(a), a.mass) == (0, true, 1, 0.0)
    end
    # Real atoms: the case of the symbol does not matter, and the default mass is the most
    # abundant isotope's.
    for s in ("O", "o")
        a = Atom(s, [0.0, 0.0, 0.0])
        @test (a.Z, a.ghost, basis_element(a), a.mass) == (8, false, 8, element(8).mass_most_abundant)
    end
    @test Atom("Fe", [0.0, 0.0, 0.0]).Z == 26
    @test Atom("O", [0.0, 0.0, 0.0]; mass = 17.99915961).mass == 17.99915961
    @test Atom("bqfe", [0.0, 0.0, 0.0]) |> basis_element == 26
    # A ghost at a real atom's position: the repulsion is the H2 value, and nothing throws.
    h2 = Fixtures.h2()
    withghost = Molecule([h2.atoms[1], Atom("H@", h2.atoms[1].position), h2.atoms[2]])
    @test nuclear_repulsion(withghost) == nuclear_repulsion(h2)
    @test nelectrons(withghost) == 2
end

@testitem "system errors" begin
    using QuantumEngine
    using StaticArrays: SVector
    @test_throws "system: unknown element symbol \"Foo\"" Atom("Foo", [0.0, 0.0, 0.0])
    @test_throws "system: unknown element symbol \"Xx@\"" Atom("Xx@", [0.0, 0.0, 0.0])
    @test_throws "system: unknown element symbol \"Bq-Xx\"" Atom("Bq-Xx", [0.0, 0.0, 0.0])
    @test_throws "system: a Bq atom needs an element (BqO, O@); a point charge is `Bq x y z q`" Atom("Bq", [0.0, 0.0, 0.0])
    @test_throws "system: a Bq atom needs an element (BqO, O@); a point charge is `Bq x y z q`" Atom("bq-", [0.0, 0.0, 0.0])
    @test_throws "system: an atom position has 3 coordinates, got 2" Atom("H", [0.0, 0.0])
    @test_throws "system: a ghost atom has no mass, got 2.0" Atom("H@", [0.0, 0.0, 0.0]; mass = 2.0)
    @test_throws ArgumentError Atom("Foo", [0.0, 0.0, 0.0])
    coincide = Molecule([Atom("H", [0.0, 0.0, 0.0]), Atom("H", [0.0, 0.0, 0.0])])
    @test_throws "system: atoms 1 and 2 coincide" nuclear_repulsion(coincide)
    @test_throws "system: ghost atom \"Foo@\" stands for no element" basis_element(Atom{Float64}(0, "Foo@", SVector(0.0, 0.0, 0.0), true, 0.0))
end

@testitem "system: element_number" begin
    using QuantumEngine.System: element_number
    @test element_number("O") == 8
    @test element_number("fe") == 26
    @test element_number(SubString("xCl", 2)) == 17
    @test element_number("O@") == 8
    @test element_number("bqh") == 1
    @test element_number("Bq-He") == 2
    @test element_number("Xx") == 0
    @test element_number("Bq@") == 0
    @test element_number("") == 0
    @test_throws "system: a Bq atom needs an element (BqO, O@); a point charge is `Bq x y z q`" element_number("Bq")
    @test_throws "system: a Bq atom needs an element" element_number("bq-")
end

@testitem "system: generic indexing" setup = [Fixtures] begin
    using QuantumEngine, OffsetArrays
    using StaticArrays: SVector
    atoms = Fixtures.water().atoms
    ref = Molecule(atoms)
    for m in (Molecule(OffsetArray(atoms, -1:1)), Molecule(view(atoms, 1:3)))
        @test nuclear_repulsion(m) == nuclear_repulsion(ref)
        @test nelectrons(m) == nelectrons(ref)
        @test (nalpha(m), nbeta(m)) == (nalpha(ref), nbeta(ref))
    end
    h = Atom("H", [0.0, 0.0, 0.0])
    shifted = Molecule(OffsetArray([h, h, Atom("H", [0.0, 0.0, 1.4])], -1:1))
    @test_throws "system: atoms -1 and 0 coincide" nuclear_repulsion(shifted)
    @test Atom("H", OffsetArray([0.0, 0.0, 1.0], 0:2)).position == SVector(0.0, 0.0, 1.0)
    @test Atom("H", view([9.0, 0.0, 0.0, 1.0], 2:4)).position == SVector(0.0, 0.0, 1.0)
end

@testitem "system: allocation-free counts" setup = [Fixtures] begin
    using QuantumEngine, StrictModeTest
    w = Fixtures.water()
    w32 = Molecule([Atom(a.symbol, Float32.(a.position)) for a in w.atoms])
    # @test_noalloc throws a StrictViolation unless AllocCheck proves the call allocation-free.
    for m in (w, w32)
        @test (@test_noalloc nuclear_repulsion(m)) == nuclear_repulsion(m)
        @test (@test_noalloc nelectrons(m)) == 10
        @test (@test_noalloc nalpha(m)) == 5
        @test (@test_noalloc nbeta(m)) == 5
    end
end
