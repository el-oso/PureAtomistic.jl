# Structured input: the `Node` tree, the schema walker `parse_structured` and the JSON reader. Each
# item builds its tree with the `Node` constructors (through `nd`), so the items test the walker
# that JSON text and every `AbstractDict` reach. The oracle (`quantum_engine` at 4a78a1d) reads
# structured input in `parse_yaml_string` (`src/input/parser.rs:677-1076`); each lenient value
# below is the one it takes.

@testmodule Structured begin
    using QuantumEngine
    using QuantumEngine.Input: Node, parse_structured

    # The `Node` of a Julia value: a scalar, a vector (a list) or `key => value` pairs (a map).
    nd(x::Node) = x
    nd(::Nothing) = Node(nothing)
    nd(x::Union{Bool, Int, Float64, String}) = Node(x)
    nd(v::AbstractVector) = Node(Node[nd(x) for x in v])
    nd(ps::Pair...) = Node(String[first(p) for p in ps], Node[nd(last(p)) for p in ps])

    # H2 in bohr, the atoms in the position form.
    const H2 = [nd("symbol" => "H", "position" => [0.0, 0.0, -0.7]), nd("symbol" => "H", "position" => [0.0, 0.0, 0.7])]

    doc(ps::Pair...) = nd("unit" => "bohr", "geometry" => H2, ps...)
    strict(ps::Pair...) = parse_structured(doc(ps...); source = "t.json").input
    lenient(ps::Pair...) = parse_structured(doc(ps...); source = "t.json", lenient = true)
    # H2 and a He atom, for the constraints on three atoms.
    const H2HE = [H2..., nd("symbol" => "He", "position" => [0.0, 2.0, 0.0])]
    strict3(ps::Pair...) = parse_structured(nd("unit" => "bohr", "geometry" => H2HE, ps...); source = "t.json").input
    lenient3(ps::Pair...) = parse_structured(nd("unit" => "bohr", "geometry" => H2HE, ps...); source = "t.json", lenient = true)
    cases(p) = [w.case for w in p.warnings]
end

@testitem "json: fe2_dmft.json" setup = [Fixtures] begin
    using QuantumEngine
    using QuantumEngine.Input: HubbardU
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    p = parse_json(Fixtures.oracle_input("benchmarks/fe2_dmft.json"); source = "fe2_dmft.json")
    i = p.input
    @test isempty(p.warnings)
    @test length(i.atoms) == 2
    @test all(a -> a.symbol == "Fe" && a.Z == 26, i.atoms)
    @test i.atoms[1].position == [0.0, 0.0, 1.89 * ANGSTROM_TO_BOHR]
    @test i.atoms[2].position == [0.0, 0.0, -1.89 * ANGSTROM_TO_BOHR]
    @test i.basis == "sto-3g"
    @test i.charge == 0
    @test i.multiplicity == 1
    @test i.scf_convergence == 1.0e-6
    @test i.max_iterations == 100
    @test i.use_dmft
    @test i.dmft_temperature == 1000.0
    @test i.dmft_n_freq == 512
    @test i.dmft_n_bath == 2
    @test i.dmft_u == [HubbardU("Fe", 4.0)]
    @test i.dmft_j == [HubbardU("Fe", 0.9)]
end

@testitem "json: duplicate keys" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using QuantumEngine
    using QuantumEngine.Input: Node, json_node
    for lenient in (false, true)
        @test_throws "input: t.json: duplicate key \"a\"" parse_json("{\"a\":1,\"a\":2}"; source = "t.json", lenient)
        @test_throws "input: t.json: b: duplicate key \"a\"" parse_json("{\"b\":{\"a\":1,\"a\":2}}"; source = "t.json", lenient)
        @test_throws "input: t.json: b[1]: duplicate key \"a\"" parse_json("{\"b\":[{\"a\":1,\"a\":2}]}"; source = "t.json", lenient)
        @test_throws InputError parse_json("{\"scf\":{\"a\":1,\"a\":2}}"; source = "t.json", lenient)
    end
    @test_throws "input: <string>: duplicate key \"a\"" json_node("{\"a\":1,\"a\":2}")
    @test_throws "input: duplicate key \"a\"" Node(["a", "b", "a"], [Node(1), Node(2), Node(3)])
    @test_throws "input: a map has 2 keys and 1 values" Node(["a", "b"], [Node(1)])
end

@testitem "json: text that is not JSON" begin
    using QuantumEngine
    using QuantumEngine.Input: json_node
    for lenient in (false, true)
        @test_throws "input: t.json: not valid JSON (invalid JSON at byte position 14" parse_json("{\"charge\": 1,}"; source = "t.json", lenient)
        @test_throws InputError parse_json("{\"charge\": 1,}"; source = "t.json", lenient)
    end
    @test_throws "input: <string>: not valid JSON" json_node("[1, ")
    mktempdir() do dir
        path = joinpath(dir, "bad.json")
        write(path, "{\"charge\": 1,}")
        for lenient in (false, true)
            err = try
                parse_input(path; lenient)
                nothing
            catch e
                e
            end
            @test err isa InputError
            @test err.source == path && err.line == 0
            @test startswith(sprint(showerror, err), "input: $(path): not valid JSON (invalid JSON at byte position 14")
        end
    end
end

@testitem "json: json_node scalar typing" begin
    using QuantumEngine.Input
    t = json_node("{\"i\":1,\"f\":1.0,\"e\":1e-6,\"s\":\"x\",\"t\":true,\"n\":null,\"a\":[1,{\"k\":false}],\"m\":{}}")
    @test t.kind == MapNode
    @test t.keys == ["i", "f", "e", "s", "t", "n", "a", "m"]
    @test [n.kind for n in t.items] == [IntNode, FloatNode, FloatNode, StringNode, BoolNode, NullNode, ArrayNode, MapNode]
    @test t.items[1].i === 1
    @test t.items[2].f === 1.0
    @test t.items[3].f === 1.0e-6
    @test t.items[4].s == "x"
    @test t.items[5].b
    a = t.items[7]
    @test [n.kind for n in a.items] == [IntNode, MapNode]
    @test a.items[2].keys == ["k"] && !a.items[2].items[1].b
    @test isempty(t.items[8].keys)
    @test json_node("1").kind == IntNode
    @test json_node("1.0").kind == FloatNode
    @test json_node("1e-6").kind == FloatNode
    @test json_node("\"s\"").s == "s"
    @test json_node("true").kind == BoolNode
    @test json_node("null").kind == NullNode
    @test_throws "input: <string>: the integer 12345678901234567890 is outside the Int64 range" json_node("12345678901234567890")
    @test_throws "input: a.json: scf.max_iterations: the integer 12345678901234567890 is outside the Int64 range" json_node(
        "{\"scf\": {\"max_iterations\": 12345678901234567890}}"; source = "a.json",
    )
    # Beyond the Float64 range JSON.jl reads a BigFloat, and Rust's f64 parser, which the oracle uses, ±Inf.
    @test json_node("[1e400, -1e400]").items[1].f == Inf
    @test json_node("[1e400, -1e400]").items[2].f == -Inf
    # Below the Float64 range JSON.jl reads 0.0, as Rust's f64 parser does.
    @test json_node("1e-400").f === 0.0
end

@testitem "json: errors in both modes" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using QuantumEngine
    using QuantumEngine.Input: Node
    using .Structured: nd, H2
    run(t, lenient) = parse_structured(t; source = "t.json", lenient)
    for lenient in (false, true)
        @test_throws "input: t.json: the document is empty" run(Node(nothing), lenient)
        @test_throws "input: t.json: the top-level value is a list, not a map" run(nd([1]), lenient)
        @test_throws "input: t.json: geometry[1]: missing key `symbol`" run(nd("geometry" => [nd("x" => 0.0)]), lenient)
        @test_throws "input: t.json: geometry[1].symbol: expected a string, got an integer" run(nd("geometry" => [nd("symbol" => 1)]), lenient)
        @test_throws "input: t.json: geometry[1]: expected a map, got a string" run(nd("geometry" => ["H"]), lenient)
        @test_throws "input: t.json: geometry[1].position: expected 3 numbers, got 2" run(
            nd("geometry" => [nd("symbol" => "H", "position" => [0.0, 0.0])]), lenient,
        )
        @test_throws "input: t.json: geometry[1].symbol: unknown element symbol \"Xx\"" run(
            nd("geometry" => [nd("symbol" => "Xx", "position" => [0.0, 0.0, 0.0])]), lenient,
        )
        @test_throws "input: t.json: geometry[1].symbol: a Bq atom needs an element" run(
            nd("geometry" => [nd("symbol" => "Bq", "position" => [0.0, 0.0, 0.0])]), lenient,
        )
        @test_throws "input: t.json: unknown value `zzz` for `oniom.high`" run(nd("geometry" => H2, "oniom" => nd("high" => "zzz")), lenient)
        @test_throws "input: t.json: unknown value `zzz` for `oniom.low`" run(nd("geometry" => H2, "oniom" => nd("low" => "zzz")), lenient)
        # The oracle casts a negative integer to an unsigned count near 2^64, which no field holds.
        @test_throws "input: t.json: multiplicity: expected a non-negative integer, got -1" run(nd("geometry" => H2, "multiplicity" => -1), lenient)
        @test_throws "input: t.json: kpoints[2]: expected a non-negative integer, got -2" run(nd("geometry" => H2, "kpoints" => [1, -2, 1]), lenient)
        @test_throws "input: t.json: constraints[1].freeze_atom: expected an atom index (an integer ≥ 0), got -1" run(
            nd("geometry" => H2, "constraints" => [nd("freeze_atom" => -1)]), lenient,
        )
        # The 1-based index of typemax(Int) is not an Int.
        @test_throws "input: t.json: constraints[1].freeze_atom: expected an atom index (an integer ≥ 0), got 9223372036854775807" run(
            nd("geometry" => H2, "constraints" => [nd("freeze_atom" => typemax(Int))]), lenient,
        )
        @test_throws "input: t.json: oniom.model_atoms[2]: expected an atom index (an integer ≥ 0), got 9223372036854775807" run(
            nd("geometry" => H2, "oniom" => nd("model_atoms" => [0, typemax(Int)])), lenient,
        )
        # The oracle reads `oniom` before `geometry`, so its error comes first.
        @test_throws "input: t.json: unknown value `zzz` for `oniom.high`" run(
            nd("oniom" => nd("high" => "zzz"), "geometry" => [nd("symbol" => "Xx", "position" => [0.0, 0.0, 0.0])]), lenient,
        )
    end
    # A malformed CASSCF active space in an ONIOM method: strict rejects its shape, and the oracle fails.
    @test_throws "input: t.json: oniom.high: casscf needs (electrons,orbitals), got `casscf(x,2)`" run(
        nd("geometry" => H2, "oniom" => nd("high" => "casscf(x,2)")), false,
    )
    @test_throws "input: t.json: Invalid CASSCF electron count" run(nd("geometry" => H2, "oniom" => nd("high" => "casscf(x,2)")), true)
    @test_throws "input: t.json: Invalid CASSCF orbital count" run(nd("geometry" => H2, "oniom" => nd("low" => "casscf(2,x)")), true)
end

# One item per lenient case of structured input: strict mode throws the message, lenient mode
# takes the oracle's value with one warning of the case.
@testitem "json: yaml_unknown_key" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using .Structured: nd, strict, lenient, cases
    @test_throws "input: t.json: scf.foo: unknown key; accepted keys: max_iter, conv_energy" strict("scf" => nd("foo" => 1))
    @test_throws "input: t.json: foo: unknown key; accepted keys: title" strict("foo" => 1)
    p = lenient("scf" => nd("foo" => 1, "max_iter" => 7))
    @test cases(p) == [:yaml_unknown_key]
    @test p.input.max_iterations == 7
end

@testitem "json: yaml_wrong_type" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using .Structured: nd, strict, lenient, cases
    @test_throws "input: t.json: scf.max_iter: expected an integer, got a string" strict("scf" => nd("max_iter" => "many"))
    @test_throws "input: t.json: title: expected a string, got a number" strict("title" => 1.5)
    @test_throws "input: t.json: charge: expected an integer, got a number" strict("charge" => 1.0)
    p = lenient("scf" => nd("max_iter" => "many"))
    @test cases(p) == [:yaml_wrong_type]
    @test p.input.max_iterations == 50
end

@testitem "json: yaml_int_for_float" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using .Structured: nd, strict, lenient, cases
    @test strict("scf" => nd("conv_energy" => 1)).scf_convergence === 1.0
    p = parse_structured(Structured.doc("scf" => nd("conv_energy" => 1)); source = "t.json")
    @test isempty(p.warnings)
    p = lenient("scf" => nd("conv_energy" => 1))
    @test cases(p) == [:yaml_int_for_float]
    @test p.input.scf_convergence == 1.0e-6
end

@testitem "json: yaml_unknown_method" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using QuantumEngine.Input: MethodSpec, HF, CASSCF, AutoReference
    using .Structured: nd, strict, lenient, cases
    @test_throws "input: t.json: unknown value `dft` for `method`; accepted: rhf" strict("method" => "dft")
    p = lenient("method" => "dft")
    @test cases(p) == [:yaml_unknown_method]
    @test p.input.method == MethodSpec(HF, AutoReference, (0, 0), 0.0)
    # A `casscf` word without its active space: the oracle reads (2, 2), and a malformed count in
    # the top-level method makes it ignore the method.
    @test_throws "input: t.json: method: casscf needs (electrons,orbitals), got `casscf`" strict("method" => "casscf")
    p = lenient("method" => "casscf")
    @test cases(p) == [:yaml_unknown_method]
    @test p.input.method == MethodSpec(CASSCF, AutoReference, (2, 2), 0.0)
    p = lenient("method" => "casscf(x,2)")
    @test cases(p) == [:yaml_unknown_method]
    @test p.input.method == MethodSpec(HF, AutoReference, (0, 0), 0.0)
    @test strict("method" => "CASSCF(4,4)").method == MethodSpec(CASSCF, AutoReference, (4, 4), 0.0)
    @test strict("oniom" => nd("high" => "casscf(4,4)")).oniom.high == MethodSpec(CASSCF, AutoReference, (4, 4), 0.0)
    p = lenient("oniom" => nd("high" => "casscf"))
    @test cases(p) == [:yaml_unknown_method]
    @test p.input.oniom.high == MethodSpec(CASSCF, AutoReference, (2, 2), 0.0)
end

@testitem "json: yaml_unknown_mode" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using QuantumEngine.Input: SinglePoint, Frequency
    using .Structured: nd, strict, lenient, cases
    @test_throws "input: t.json: unknown value `zzz` for `mode`; accepted: sp" strict("mode" => "zzz")
    @test_throws "input: t.json: unknown value `zzz` for `task.mode`" strict("task" => nd("mode" => "zzz"))
    p = lenient("mode" => "zzz")
    @test cases(p) == [:yaml_unknown_mode]
    @test p.input.run_mode == SinglePoint
    @test strict("mode" => "Freq").run_mode == Frequency
    @test strict("task" => nd("mode" => "freq")).run_mode == Frequency
end

@testitem "json: yaml_unknown_solvent" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using QuantumEngine.Input: SMD
    using .Structured: nd, strict, lenient, cases
    @test_throws "input: t.json: unknown value `lava` for `solvation.solvent`; accepted: water" strict(
        "solvation" => nd("model" => "smd", "solvent" => "lava"),
    )
    p = lenient("solvation" => nd("model" => "smd", "solvent" => "lava"))
    @test cases(p) == [:yaml_unknown_solvent]
    @test p.input.solvation.model == SMD
    @test p.input.solvation.solvent === :water
end

@testitem "json: yaml_unknown_surface" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using QuantumEngine.Input: PCM, VdwSurface
    using .Structured: nd, strict, lenient, cases
    @test_throws "input: t.json: unknown value `blob` for `solvation.surface`; accepted: vdw, sas, smd" strict(
        "solvation" => nd("model" => "pcm", "surface" => "blob"),
    )
    p = lenient("solvation" => nd("model" => "pcm", "surface" => "blob"))
    @test cases(p) == [:yaml_unknown_surface]
    @test p.input.solvation.model == PCM
    @test p.input.solvation.surface == VdwSurface
end

@testitem "json: yaml_solvation_without_model" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using QuantumEngine.Input: NoSolvation
    using .Structured: nd, strict, lenient, cases
    @test_throws "input: t.json: solvation: give model, pcm or smd" strict("solvation" => nd("epsilon" => 4.0))
    p = lenient("solvation" => nd("epsilon" => 4.0))
    @test cases(p) == [:yaml_solvation_without_model]
    @test p.input.solvation.model == NoSolvation
end

@testitem "json: yaml_f12_gamma_enables" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using .Structured: nd, strict, lenient, cases
    @test_throws "input: t.json: f12_gamma needs f12" strict("f12_gamma" => 0.9)
    @test_throws "input: t.json: gamma needs f12" strict("f12" => false, "gamma" => 0.9)
    p = lenient("f12_gamma" => 0.9)
    @test cases(p) == [:yaml_f12_gamma_enables]
    @test p.input.use_f12
    @test p.input.f12_gamma == 0.9
    i = strict("f12" => true, "f12_gamma" => 0.9)
    @test i.use_f12 && i.f12_gamma == 0.9
end

@testitem "json: yaml_constraint_index" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using QuantumEngine.Input: Constraint, FreezeBond, FreezeAngle
    using .Structured: nd, strict, lenient, cases
    @test_throws "input: t.json: constraints[1].freeze_bond: expected 2 indices, got 1" strict("constraints" => [nd("freeze_bond" => [0])])
    @test_throws "input: t.json: constraints[1].freeze_angle[2]: expected an atom index (an integer ≥ 0), got a string" strict(
        "constraints" => [nd("freeze_angle" => [0, "b", 1])],
    )
    # A list of fewer than 2 indices is ignored; an entry that is not an integer is 0, 1 or 2 by
    # position (0-based), stored 1-based.
    p = lenient("constraints" => [nd("freeze_bond" => [0])])
    @test cases(p) == [:yaml_constraint_index]
    @test isempty(p.input.constraints)
    p = Structured.lenient3("constraints" => [nd("freeze_angle" => ["a", "b", "c"])])
    @test cases(p) == fill(:yaml_constraint_index, 3)
    @test p.input.constraints == [Constraint(FreezeAngle, (1, 2, 3, 0), nothing)]
    p = lenient("constraints" => [nd("freeze_bond" => [1, 0, 1], "target" => 1.5)])
    @test cases(p) == [:yaml_constraint_index]
    @test p.input.constraints == [Constraint(FreezeBond, (2, 1, 0, 0), 1.5)]
end

@testitem "json: atom indices and hubbard symbols" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using QuantumEngine.Input: Constraint, FreezeAtom, FreezeBond, HubbardU
    using .Structured: nd, strict, lenient, cases
    # parser.rs:677-1076: the oracle stores the indices and symbols as written. H2 has 2 atoms.
    range_msg(path, i) = "input: t.json: $(path): atom index $(i) is out of range; the input has 2 atoms (indices 0 to 1)"
    rows = [
        (("constraints" => [nd("freeze_atom" => 0), nd("freeze_atom" => 5)],), range_msg("constraints[2].freeze_atom", 5), :atom_index_range),
        (("constraints" => [nd("freeze_bond" => [0, 2])],), range_msg("constraints[1].freeze_bond", 2), :atom_index_range),
        (("oniom" => nd("model_atoms" => [0, 7]),), range_msg("oniom.model_atoms[2]", 7), :atom_index_range),
        (("constraints" => [nd("freeze_bond" => [1, 1])],), "input: t.json: constraints[1].freeze_bond: atom index 1 is given twice", :repeated_atom),
        (("hubbard" => nd("Fe" => 4.0, "Xx" => 1.0),), "input: t.json: hubbard.Xx: unknown element symbol \"Xx\"", :hubbard_element),
        (("dmft" => nd("u_map" => nd("Qq" => 4.0)),), "input: t.json: dmft.u_map.Qq: unknown element symbol \"Qq\"", :hubbard_element),
    ]
    for (ps, msg, case) in rows
        @test_throws msg strict(ps...)
        @test cases(lenient(ps...)) == [case]
    end
    p = lenient("constraints" => [nd("freeze_atom" => 5), nd("freeze_bond" => [1, 1], "target" => 1.4)])
    @test p.input.constraints == [Constraint(FreezeAtom, (6, 0, 0, 0), nothing), Constraint(FreezeBond, (2, 2, 0, 0), 1.4)]
    @test lenient("oniom" => nd("model_atoms" => [0, 7])).input.oniom.model_atoms == [1, 8]
    @test lenient("hubbard" => nd("Xx" => 1.0)).input.hubbard == [HubbardU("Xx", 1.0)]
    for lenient in (false, true)
        @test_throws "input: t.json: hubbard.Bq: a Bq atom needs an element" parse_structured(
            Structured.doc("hubbard" => nd("Bq" => 1.0)); source = "t.json", lenient,
        )
    end
end

@testitem "json: yaml_bad_number" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using .Structured: nd, lenient, cases
    g = [nd("symbol" => "H", "position" => [0.0, 0.0, -0.7]), nd("symbol" => "H", "position" => [0.0, 0.0, "x"])]
    @test_throws "input: t.json: geometry[2].position[3]: expected a number, got a string" parse_structured(
        nd("unit" => "bohr", "geometry" => g); source = "t.json",
    )
    p = parse_structured(nd("unit" => "bohr", "geometry" => g); source = "t.json", lenient = true)
    @test cases(p) == [:yaml_bad_number]
    @test p.input.atoms[2].position == [0.0, 0.0, 0.0]
end

@testitem "json: yaml_missing_coordinate" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using .Structured: nd, cases
    g = [nd("symbol" => "H", "x" => 0.0, "y" => 0.0, "z" => -0.7), nd("symbol" => "H", "x" => 0.5, "y" => 0.25)]
    @test_throws "input: t.json: geometry[2]: missing key `z`" parse_structured(nd("unit" => "bohr", "geometry" => g); source = "t.json")
    p = parse_structured(nd("unit" => "bohr", "geometry" => g); source = "t.json", lenient = true)
    @test cases(p) == [:yaml_missing_coordinate]
    @test p.input.atoms[1].position == [0.0, 0.0, -0.7]
    @test p.input.atoms[2].position == [0.5, 0.25, 0.0]
end

@testitem "json: yaml_unknown_unit" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    using .Structured: nd, H2, cases
    t = nd("unit" => "nm", "geometry" => H2)
    @test_throws "input: t.json: unit: unknown value `nm`; accepted: angstrom, ang, bohr, au" parse_structured(t; source = "t.json")
    p = parse_structured(t; source = "t.json", lenient = true)
    @test cases(p) == [:yaml_unknown_unit]
    @test p.input.atoms[2].position == [0.0, 0.0, 0.7 * ANGSTROM_TO_BOHR]
end

@testitem "json: yaml_lattice_row" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using StaticArrays: SMatrix
    using .Structured: nd, strict, lenient, cases
    lat = [[5.0, 0.0, 0.0], [0.0, 5.0], [0.0, 0.0, 5.0]]
    @test_throws "input: t.json: lattice[2]: expected 3 numbers" strict("lattice" => lat)
    p = lenient("lattice" => lat)
    @test cases(p) == [:yaml_lattice_row]
    @test p.input.lattice == SMatrix{3, 3}(5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0)
end

@testitem "json: shared checks" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using QuantumEngine
    using QuantumEngine.Input: AutoReference, HF
    using .Structured: nd, H2, strict, lenient, cases
    @test_throws "input: t.json: no atoms: the input has no geometry block" parse_structured(nd("charge" => 0); source = "t.json")
    @test_throws "input: t.json: method `rhf` requests a restricted reference, but multiplicity is 3" strict(
        "method" => "rhf", "multiplicity" => 3, "charge" => -2,
    )
    p = lenient("method" => "rhf", "multiplicity" => 3, "charge" => -2)
    @test cases(p) == [:rhf_open_shell]
    @test p.input.method.reference == AutoReference && p.input.method.kind == HF
    @test_throws "input: t.json: system: 2 electrons cannot have multiplicity 2" strict("multiplicity" => 2)
end

@testitem "json: units au and ang" setup = [Structured] begin
    using QuantumEngine.Input: parse_structured
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    using .Structured: nd, H2
    for lenient in (false, true)
        for (u, f) in (("au", 1.0), ("AU", 1.0), ("bohr", 1.0), ("ang", ANGSTROM_TO_BOHR), ("Angstrom", ANGSTROM_TO_BOHR))
            p = parse_structured(nd("unit" => u, "geometry" => H2); source = "t.json", lenient)
            @test isempty(p.warnings)
            @test p.input.atoms[2].position == [0.0, 0.0, 0.7 * f]
        end
        p = parse_structured(nd("geometry" => H2); source = "t.json", lenient)
        @test p.input.atoms[2].position == [0.0, 0.0, 0.7 * ANGSTROM_TO_BOHR]
    end
end

# Every key of the schema with a valid value, read strictly.
@testitem "json: schema" setup = [Structured] begin
    using QuantumEngine
    using QuantumEngine.Input
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    using StaticArrays: SMatrix, SVector
    using .Structured: nd, H2, strict, strict3
    i = strict3(
        "title" => "t", "charge" => 0, "multiplicity" => 1, "basis" => "6-31G*", "method" => "B3LYP",
        "dispersion" => true, "zora" => true, "mode" => "opt", "hubbard" => nd("Fe" => 4.0, "Ni" => 6),
        "vdw_kernel" => "d3bj",
        "scf" => nd(
            "max_iter" => 75, "conv_energy" => 1.0e-8, "use_ri_j" => true, "use_ri_k" => true, "aux_basis" => "def2-jkfit",
            "compute_mode" => "Hybrid", "diis_method" => "ediis", "verbosity" => 2, "checkpoint" => "a.chk",
            "load_checkpoint" => true, "damping" => 0.2, "smearing" => 0.01, "grid" => 3, "auto_stabilize" => false,
            "robust_convergence" => true, "level_shift" => 0.5, "field" => [0.0, 0.0, 0.01], "pw_cutoff" => 400,
            "lattice" => [[5, 0, 0], [0, 5, 0], [0, 0, 5]], "pbc" => [true, true, false], "admm" => true,
            "admm_basis" => "cfit", "population" => ["Lowdin"], "kpoints" => [2, 3, 4], "kpath" => ["G", "X"],
            "use_dmft" => true, "dmft" => nd("temperature" => 500.0, "n_matsubara" => 256, "n_bath" => 3),
        ),
        "solvation" => nd("model" => "PCM", "epsilon" => 4.0, "radii_scaling" => 1.1, "grid_order" => 194, "surface" => "sas"),
        "oniom" => nd("high" => "pbe0", "low" => "hf", "model_atoms" => [0, 1], "model_charge" => -1, "model_multiplicity" => 2),
        "tddft" => nd("num_states" => 5, "method" => "rpa", "soc" => true, "triplets" => true),
        "exports" => nd("molden" => true, "cube" => true, "wfx" => true),
        "f12" => nd("f12_gamma" => 1.2, "approximation" => "F12a"), "f12_model" => "F12b*",
        "irc_step" => 0.05, "irc_points" => 12, "irc_direction" => "fwd",
        "constraints" => [
            nd("freeze_atom" => 1), nd("freeze_bond" => [0, 1], "target" => 1.4),
            nd("freeze_angle" => [0, 1, 2], "target" => 90), nd("freeze_angle" => [1, 0, 2]),
        ],
    )
    @test i.title == "t"
    @test i.basis == "6-31g*"
    @test i.method == MethodSpec(B3LYP, AutoReference, (0, 0), 0.0)
    @test i.dispersion && i.zora
    @test i.run_mode == Optimization
    @test i.hubbard == [HubbardU("Fe", 4.0), HubbardU("Ni", 6.0)]
    @test i.vdw_kernel == D3BJ
    @test i.max_iterations == 75
    @test i.scf_convergence == 1.0e-8
    @test i.use_ri_j && i.use_ri_k
    @test i.aux_basis == "def2-jkfit"
    @test i.compute_mode == Hybrid
    @test i.diis_method == EDIIS
    @test i.verbosity == 2
    @test i.checkpoint == "a.chk" && i.load_checkpoint
    @test i.damping == 0.2 && i.smearing == 0.01 && i.level_shift == 0.5
    @test i.grid_level == 3
    @test !i.auto_stabilize && i.robust_convergence
    @test i.electric_field == SVector(0.0, 0.0, 0.01)
    @test i.pw_cutoff == 400.0
    @test i.lattice == SMatrix{3, 3}(5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0)
    @test i.pbc == (true, true, false)
    @test i.admm && i.admm_basis == "cfit"
    @test i.population == [Lowdin]
    @test i.kpoints == (2, 3, 4)
    @test i.kpath == ["G", "X"]
    @test i.use_dmft
    @test (i.dmft_temperature, i.dmft_n_freq, i.dmft_n_bath) == (500.0, 256, 3)
    s = i.solvation
    @test (s.model, s.epsilon, s.custom_epsilon, s.radii_scaling, s.grid_order, s.surface) == (PCM, 4.0, false, 1.1, 194, SASSurface)
    o = i.oniom
    @test o.high == MethodSpec(PBE0, AutoReference, (0, 0), 0.0)
    @test o.low == MethodSpec(HF, AutoReference, (0, 0), 0.0)
    @test o.model_atoms == [1, 2]
    @test (o.model_charge, o.model_multiplicity) == (-1, 2)
    @test (i.tddft_states, i.tddft_method, i.compute_soc, i.tddft_triplets) == (5, RPA_TD, true, true)
    @test i.export_molden && i.export_cube && i.export_wfx
    @test i.use_f12 && i.f12_gamma == 1.2 && i.f12_model == "F12b*"
    @test (i.irc_step_size, i.irc_max_points, i.irc_direction) == (0.05, 12, Forward)
    @test i.constraints == [
        Constraint(FreezeAtom, (2, 0, 0, 0), nothing), Constraint(FreezeBond, (1, 2, 0, 0), 1.4),
        Constraint(FreezeAngle, (1, 2, 3, 0), deg2rad(90.0)), Constraint(FreezeAngle, (2, 1, 3, 0), nothing),
    ]
    # The top-level fallbacks of the `scf` keys, the other spellings and the `{symbol, x, y, z}` form.
    i = parse_structured(
        nd(
            "atoms" => [nd("symbol" => "H", "x" => 0, "y" => 0, "z" => -0.7), nd("symbol" => "H", "x" => 0.0, "y" => 0.0, "z" => 0.7)],
            "unit" => "Bohr", "max_iterations" => 20, "scf_convergence" => 1.0e-7, "gpu" => true, "pw_cutoff" => 300.0,
            "lattice" => [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]], "pbc" => [false, false, false],
            "k_grid" => [2, 2, 2], "k_path" => ["L"], "use_dmft" => true,
            "dmft" => nd("u_map" => nd("Fe" => 4.0), "j_map" => nd("Fe" => 0.9)),
            "solvation" => nd("smd" => true, "solvent" => "Di-chloro_methane", "grid_order" => 302, "probe_radius" => 0.5),
            "tddft" => nd("method" => "sf"), "f12" => nd("gamma" => 0.8, "model" => "F12c"), "gamma" => 0.7,
            "irc_step_size" => 0.2, "irc_max_points" => 40, "irc_direction" => "back",
        ); source = "t.json",
    ).input
    @test [a.position for a in i.atoms] == [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]
    @test (i.max_iterations, i.scf_convergence, i.compute_mode, i.pw_cutoff) == (20, 1.0e-7, GPU, 300.0)
    @test i.lattice == SMatrix{3, 3}(4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0)
    @test i.pbc == (false, false, false) && i.kpoints == (2, 2, 2) && i.kpath == ["L"]
    @test i.use_dmft && i.dmft_u == [HubbardU("Fe", 4.0)] && i.dmft_j == [HubbardU("Fe", 0.9)]
    s = i.solvation
    @test (s.model, s.solvent, s.grid_order, s.probe_radius) == (SMD, :dcm, 302, 0.5 * ANGSTROM_TO_BOHR)
    @test i.tddft_method == SpinFlip
    @test i.use_f12 && i.f12_gamma == 0.7 && i.f12_model == "F12c"
    @test (i.irc_step_size, i.irc_max_points, i.irc_direction) == (0.2, 40, Backward)
    # `kpoints` at the top level, `scf.dmft_options` and `scf.dmft`, `f12` as a Boolean, the
    # default SMD options, `pcm: true` with the default PCM options, and `pcm: false`.
    i = strict("kpoints" => [3, 3, 3], "scf" => nd("dmft_options" => nd("n_bath" => 5)), "f12" => true, "solvation" => nd("model" => "smd"))
    @test i.kpoints == (3, 3, 3) && i.dmft_n_bath == 5 && i.use_f12
    @test i.solvation == SolvationSpec(; model = SMD)
    @test strict("scf" => nd("dmft" => nd("n_bath" => 6))).dmft_n_bath == 6
    @test strict("solvation" => nd("pcm" => true)).solvation == SolvationSpec(; model = PCM)
    @test strict("solvation" => nd("pcm" => false)).solvation == SolvationSpec()
    @test strict("solvation" => nd("smd" => false)).solvation == SolvationSpec()
    @test strict("scf" => nd("gpu" => false)).compute_mode == CPU
    o = strict("oniom" => nd()).oniom
    @test o.high == o.low == MethodSpec(HF, AutoReference, (0, 0), 0.0)
    @test isempty(o.model_atoms) && isnothing(o.model_charge) && isnothing(o.model_multiplicity)
end

# The other inputs the oracle reads silently: strict mode throws the message, and lenient mode
# takes the oracle's value with the warnings listed. `:yaml_both_keys` is a key given together
# with its fallback or another key that sets the same field, `:yaml_list_length` a list of another
# length than the field takes, `:yaml_unknown_value` an unknown word of a keyword other than those
# with a case of their own, and `:nonfinite` a number that is not finite.
@testitem "json: other lenient readings" setup = [Structured] begin
    using QuantumEngine
    using QuantumEngine.Input
    using StaticArrays: SVector
    using .Structured: nd, H2, strict, lenient, cases
    S(ps...) = strict(ps...)
    L(ps...) = lenient(ps...)
    # (input pairs, strict message, lenient field, lenient value, lenient cases)
    rows = [
        (("scf" => nd("max_iter" => 7), "max_iterations" => 8), "`scf.max_iter` and `max_iterations` are both given", i -> i.max_iterations, 7, [:yaml_both_keys]),
        (("scf" => nd("max_iter" => "x"), "max_iterations" => 8), "are both given", i -> i.max_iterations, 8, [:yaml_both_keys, :yaml_wrong_type]),
        (("scf" => nd("conv_energy" => 1), "scf_convergence" => 1.0e-7), "are both given", i -> i.scf_convergence, 1.0e-7, [:yaml_both_keys, :yaml_int_for_float]),
        (("mode" => "opt", "task" => nd("mode" => "freq")), "`task.mode` and `mode` are both given", i -> i.run_mode, Frequency, [:yaml_both_keys]),
        (("task" => nd("compute_gradients" => true)), "task.compute_gradients: unknown key; accepted keys: mode", i -> i.run_mode, SinglePoint, [:yaml_unknown_key]),
        (("task" => "opt",), "task: expected a map, got a string", i -> i.run_mode, SinglePoint, [:yaml_wrong_type]),
        (("hubbard" => [1],), "hubbard: expected a map, got a list", i -> i.hubbard, HubbardU[], [:yaml_wrong_type]),
        (("hubbard" => nd("Fe" => "x")), "hubbard.Fe: expected a number, got a string", i -> i.hubbard, HubbardU[], [:yaml_wrong_type]),
        (("vdw_kernel" => "d9",), "unknown value `d9` for `vdw_kernel`", i -> i.vdw_kernel, nothing, [:yaml_unknown_value]),
        (("scf" => nd("gpu" => true), "compute_mode" => "cpu"), "`scf.gpu` and `compute_mode` are both given", i -> i.compute_mode, CPU, [:yaml_both_keys]),
        (("scf" => nd("compute_mode" => "tpu")), "unknown value `tpu` for `scf.compute_mode`", i -> i.compute_mode, CPU, [:yaml_unknown_value]),
        (("scf" => nd("diis_method" => "kdiis")), "unknown value `kdiis` for `scf.diis_method`", i -> i.diis_method, CDIIS, [:yaml_unknown_value]),
        (("scf" => nd("damping" => NaN)), "scf.damping: expected a finite number, got NaN", i -> isnan(i.damping), true, [:nonfinite]),
        (("scf" => nd("field" => 1.0)), "scf.field: expected a list, got a number", i -> i.electric_field, nothing, [:yaml_wrong_type]),
        (("scf" => nd("field" => [1.0, 2.0])), "scf.field: expected 3 numbers, got 2", i -> i.electric_field, nothing, [:yaml_list_length]),
        (("scf" => nd("field" => [1.0, "x", 3.0, 4.0])), "scf.field: expected 3 numbers, got 4", i -> i.electric_field, SVector(1.0, 0.0, 3.0), [:yaml_list_length, :yaml_bad_number]),
        (("lattice" => [[1.0, 0.0, 0.0]],), "lattice: expected 3 rows, got 1", i -> i.lattice, nothing, [:yaml_list_length]),
        (("lattice" => [[1.0, 0.0, 0.0]], "scf" => nd("lattice" => [[1.0, 0.0, 0.0]])), "`scf.lattice` and `lattice` are both given", i -> i.lattice, nothing, [:yaml_both_keys, :yaml_list_length]),
        (("pbc" => [true, false],), "pbc: expected 3 Booleans, got 2", i -> i.pbc, (true, true, true), [:yaml_list_length]),
        (("pbc" => Any[false, 0, false],), "pbc[2]: expected a Boolean, got an integer", i -> i.pbc, (false, true, false), [:yaml_wrong_type]),
        (("scf" => nd("population" => ["mulliken", 1])), "scf.population[2]: expected a string, got an integer", i -> i.population, [Mulliken, Lowdin], [:yaml_wrong_type]),
        (("scf" => nd("population" => ["hirshfeld"])), "unknown value `hirshfeld` for `scf.population[1]`", i -> i.population, [Mulliken, Lowdin], [:yaml_unknown_value]),
        (("kpoints" => [2, 2],), "kpoints: expected 3 integers, got 2", i -> i.kpoints, (1, 1, 1), [:yaml_list_length]),
        (("kpoints" => Any[2, 2.0, 2],), "kpoints[2]: expected an integer, got a number", i -> i.kpoints, (2, 1, 2), [:yaml_bad_number]),
        (("k_grid" => [2, 2, 2], "kpoints" => [3, 3, 3]), "`k_grid` and `kpoints` are both given", i -> i.kpoints, (2, 2, 2), [:yaml_both_keys]),
        (("k_path" => ["G", 1, "X"],), "k_path[2]: expected a string, got an integer", i -> i.kpath, ["G", "X"], [:yaml_wrong_type]),
        (("dmft" => 1,), "dmft: expected a map, got an integer", i -> i.dmft_n_bath, 4, [:yaml_wrong_type]),
        (("solvation" => 1,), "solvation: expected a map, got an integer", i -> i.solvation.model, NoSolvation, [:yaml_wrong_type]),
        (("solvation" => nd("model" => "pcm", "smd" => true)), "solvation: give one of `model`, `pcm` and `smd`", i -> i.solvation.model, PCM, [:yaml_both_keys]),
        (("solvation" => nd("model" => "cosmo")), "unknown value `cosmo` for `solvation.model`; accepted: pcm, smd", i -> i.solvation.model, NoSolvation, [:yaml_unknown_value]),
        (("solvation" => nd("model" => 1, "pcm" => true)), "solvation: give one of", i -> i.solvation.model, PCM, [:yaml_both_keys, :yaml_wrong_type]),
        (("solvation" => nd("model" => "smd", "epsilon" => 4.0)), "solvation.epsilon: unknown key; accepted keys: model, pcm, smd, solvent", i -> i.solvation.epsilon, 78.39, [:yaml_unknown_key]),
        (("solvation" => nd("model" => "pcm", "solvent" => "water")), "solvation.solvent: unknown key; accepted keys: model, pcm, smd, epsilon", i -> i.solvation.solvent, :water, [:yaml_unknown_key]),
        (("solvation" => nd("pcm" => false, "foo" => 1)), "solvation.foo: unknown key; accepted keys: model, pcm, smd, solvent, grid_order, probe_radius, epsilon", i -> i.solvation.model, NoSolvation, [:yaml_unknown_key]),
        (("oniom" => 1,), "oniom: expected a map, got an integer", i -> isnothing(i.oniom), false, [:yaml_wrong_type]),
        (("oniom" => nd("model_atoms" => [0, "x"])), "oniom.model_atoms[2]: expected an integer, got a string", i -> i.oniom.model_atoms, [1], [:yaml_wrong_type]),
        (("tddft" => nd("method" => "cis")), "unknown value `cis` for `tddft.method`", i -> i.tddft_method, TDA, [:yaml_unknown_value]),
        (("f12" => 1,), "f12: expected a map, got an integer", i -> i.use_f12, true, [:yaml_wrong_type]),
        (("f12" => nd("gamma" => 0.8, "f12_gamma" => 0.9)), "`f12.gamma` and `f12.f12_gamma` are both given", i -> i.f12_gamma, 0.8, [:yaml_both_keys]),
        (("atoms" => H2,), "`geometry` and `atoms` are both given", i -> length(i.atoms), 2, [:yaml_both_keys]),
        (("irc_direction" => "sideways",), "unknown value `sideways` for `irc_direction`", i -> i.irc_direction, Both, [:yaml_unknown_value]),
        (("constraints" => [1],), "constraints[1]: expected a map, got an integer", i -> i.constraints, Constraint[], [:yaml_wrong_type]),
        (("constraints" => [nd("target" => 1.0)],), "constraints[1]: give one of `freeze_atom`, `freeze_bond` and `freeze_angle`", i -> i.constraints, Constraint[], [:yaml_constraint_index]),
        (("constraints" => [nd("freeze_atom" => 0, "target" => 1.0)],), "constraints[1].target: a target applies to freeze_bond and freeze_angle", i -> i.constraints, [Constraint(FreezeAtom, (1, 0, 0, 0), nothing)], [:yaml_unknown_key]),
        (("constraints" => [nd("freeze_atom" => "a", "freeze_bond" => [0, 1])],), "constraints[1]: give one of", i -> i.constraints, [Constraint(FreezeBond, (1, 2, 0, 0), nothing)], [:yaml_constraint_index, :yaml_wrong_type]),
    ]
    for (ps, msg, field, value, cs) in rows
        ps isa Pair && (ps = (ps,))
        @test_throws msg S(ps...)
        p = L(ps...)
        @test cases(p) == cs
        @test isequal(field(p.input), value)
    end
    # The atom forms: a list of more than 3 numbers, a position given with `x`, and a position that
    # is not a list, with which the oracle reads `x`, `y` and `z`.
    run(g, lenient) = parse_structured(nd("unit" => "bohr", "geometry" => [H2[1], g]); source = "t.json", lenient)
    for (g, msg, pos, cs) in (
            (nd("symbol" => "H", "position" => [0.0, 0.0, 0.7, 9.0]), "geometry[2].position: expected 3 numbers, got 4", [0.0, 0.0, 0.7], [:yaml_list_length]),
            (nd("symbol" => "H", "position" => [0.0, 0.0, 0.7], "x" => 1.0), "`geometry[2].position` and `geometry[2].x` are both given", [0.0, 0.0, 0.7], [:yaml_both_keys]),
            (nd("symbol" => "H", "position" => "p", "x" => 1.0, "y" => 0.0, "z" => 0.7), "are both given", [1.0, 0.0, 0.7], [:yaml_both_keys, :yaml_wrong_type]),
            (nd("symbol" => "H", "position" => "p", "z" => 0.7), "are both given", [0.0, 0.0, 0.7], [:yaml_both_keys, :yaml_wrong_type, :yaml_missing_coordinate, :yaml_missing_coordinate]),
            (nd("symbol" => "H", "y" => 0.5, "z" => 0.7), "geometry[2]: missing key `x`", [0.0, 0.5, 0.7], [:yaml_missing_coordinate]),
        )
        @test_throws msg run(g, false)
        p = run(g, true)
        @test cases(p) == cs
        @test p.input.atoms[2].position == pos
    end
    # A unit that is not a string: the oracle reads Å.
    @test_throws "unit: expected a string, got an integer" parse_structured(nd("unit" => 1, "geometry" => H2); source = "t.json")
    p = parse_structured(nd("unit" => 1, "geometry" => H2); source = "t.json", lenient = true)
    @test cases(p) == [:yaml_wrong_type]
    @test p.input.atoms[2].position == [0.0, 0.0, 0.7 * QuantumEngine.Units.ANGSTROM_TO_BOHR]
    @test_throws "geometry: expected a list, got a map" parse_structured(nd("geometry" => nd()); source = "t.json")
    @test_throws "no atoms" parse_structured(nd("geometry" => nd()); source = "t.json", lenient = true)
end

# A number beyond the Float64 range reaches the walker as ±Inf and meets the finite-number rule of
# every other number: strict mode rejects it, lenient mode keeps it with a `:nonfinite` warning.
@testitem "json: number beyond the Float64 range" begin
    using QuantumEngine
    h2 = """"unit": "bohr", "geometry": [{"symbol": "H", "position": [0.0, 0.0, -0.7]}, {"symbol": "H", "position": [0.0, 0.0, 0.7]}]"""
    for (x, s) in (("1e400", "Inf"), ("-1e400", "-Inf"))
        text = "{$h2, \"scf\": {\"damping\": $x}}"
        @test_throws "input: t.json: scf.damping: expected a finite number, got $s" parse_json(text; source = "t.json")
        p = parse_json(text; source = "t.json", lenient = true)
        @test p.input.damping == parse(Float64, s)
        @test [w.case for w in p.warnings] == [:nonfinite]
    end
end

# The JSON front end (`json_node`, `parse_json`) is exempt from the StrictMode guarantees because
# JSON.jl returns its values as `Any`; the walker it calls is not. `parse_structured` on a tree
# built from `Node` constructors is type-stable and boxes nothing in both modes; `parse_json`
# failing the same check shows that the check judges the call.
@testitem "json: parse_structured passes the StrictMode checks" begin
    using QuantumEngine, StrictMode, StrictModeTest
    using QuantumEngine.Input: Node, parse_structured
    StrictMode.assert_enabled()
    atom(z) = Node(["symbol", "position"], [Node("H"), Node(Node[Node(0.0), Node(0.0), Node(z)])])
    root = Node(["unit", "geometry"], [Node("bohr"), Node(Node[atom(-0.7), atom(0.7)])])
    @test length(parse_structured(root; source = "t.json").input.atoms) == 2
    @test isempty(parse_structured(root; source = "t.json", lenient = true).warnings)
    for kw in (NamedTuple{(:source,), Tuple{String}}, NamedTuple{(:source, :lenient), Tuple{String, Bool}})
        sig = (Core.kwcall, (kw, typeof(parse_structured), Node))
        for g in (:typestable, :noboxing)
            @test (test_signatures([sig]; guarantees = (g,)); true)
        end
    end
    @test_throws StrictMode.StrictViolation test_signatures(
        [(Core.kwcall, (NamedTuple{(:source,), Tuple{String}}, typeof(parse_json), String))]; guarantees = (:typestable,),
    )
end
