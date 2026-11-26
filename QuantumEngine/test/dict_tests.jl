# Structured input from any `AbstractDict`: `node` converts the dictionary to a `Node` tree, and
# `parse_dict` reads the tree with the schema walker that JSON text reaches.

@testitem "dict: node scalars" begin
    using QuantumEngine.Input
    d = Dict{String, Any}(
        "b" => true, "i" => 3, "i32" => Int32(-4), "u8" => UInt8(5), "big" => big(6), "f" => 1.5,
        "f32" => 0.25f0, "f16" => Float16(0.5), "s" => "x", "ss" => SubString("xyz", 2, 3),
        "n" => nothing, "m" => missing,
    )
    t = node(d)
    @test t.kind == MapNode
    @test sort(t.keys) == sort(collect(keys(d)))
    v(k) = t.items[findfirst(==(k), t.keys)]
    @test v("b").kind == BoolNode && v("b").b
    @test v("i").kind == IntNode && v("i").i === 3
    @test v("i32").kind == IntNode && v("i32").i === -4
    @test v("u8").kind == IntNode && v("u8").i === 5
    @test v("big").kind == IntNode && v("big").i === 6
    @test v("f").kind == FloatNode && v("f").f === 1.5
    # 0.25 and 0.5 are exact in Float32 and Float16, so the conversion to Float64 is exact.
    @test v("f32").kind == FloatNode && v("f32").f === 0.25
    @test v("f16").kind == FloatNode && v("f16").f === 0.5
    @test v("s").kind == StringNode && v("s").s == "x"
    @test v("ss").kind == StringNode && v("ss").s === "yz"
    @test v("n").kind == NullNode
    @test v("m").kind == NullNode
    # A Bool is an Integer; it stays a Boolean.
    @test node(Dict("t" => false)).items[1].kind == BoolNode
    # BigFloat beyond the Float64 range converts to ±Inf.
    @test node(Dict("x" => big"1e400")).items[1].f == Inf
end

@testitem "dict: node nested maps and vectors" begin
    using QuantumEngine.Input
    using OffsetArrays: OffsetVector
    d = Dict{String, Any}("a" => Any[1, Dict("k" => [0.5, 1.5]), Any[]], "m" => Dict{String, Any}())
    t = node(d)
    a = t.items[findfirst(==("a"), t.keys)]
    @test a.kind == ArrayNode
    @test [n.kind for n in a.items] == [IntNode, MapNode, ArrayNode]
    @test a.items[2].keys == ["k"]
    @test [n.f for n in a.items[2].items[1].items] == [0.5, 1.5]
    @test isempty(a.items[3].items)
    m = t.items[findfirst(==("m"), t.keys)]
    @test m.kind == MapNode && isempty(m.keys)
    # A vector of any axes is read in its iteration order, and key paths count from 1.
    o = node(Dict("v" => OffsetVector([7, 8], -1:0))).items[1]
    @test [n.i for n in o.items] == [7, 8]
    @test_throws "input: <dict>: v[2]: unsupported value of type Symbol" node(Dict("v" => OffsetVector([1, :x], -1:0)))
    @test_throws "input: <dict>: v[2]: unsupported value of type Symbol" node(Dict("v" => view([0, 1, :x], 2:3)))
    # The map keeps the dictionary's iteration order.
    @test node(Dict("p" => 1, "q" => 2)).keys == collect(keys(Dict("p" => 1, "q" => 2)))
end

@testitem "dict: node errors" begin
    using QuantumEngine
    using QuantumEngine.Input
    @test_throws "input: <dict>: the key 1 of type Int64 is not a string" node(Dict{Any, Any}(1 => "x"))
    @test_throws "input: <dict>: scf: the key :a of type Symbol is not a string" node(Dict("scf" => Dict(:a => 1)))
    @test_throws "input: a.yaml: geometry[1].position[2]: unsupported value of type Char" node(
        Dict("geometry" => [Dict("position" => Any[0.0, 'c', 0.0])]); source = "a.yaml",
    )
    @test_throws "input: <dict>: t: unsupported value of type Tuple{Int64, Int64}" node(Dict("t" => (1, 2)))
    @test_throws "input: <dict>: n: the integer 9223372036854775808 is outside the Int64 range" node(
        Dict("n" => Int128(typemax(Int)) + 1),
    )
    @test_throws "input: <dict>: n: the integer -9223372036854775809 is outside the Int64 range" node(
        Dict("n" => Int128(typemin(Int)) - 1),
    )
    @test_throws InputError node(Dict("n" => big(2)^70))
    # Two keys that convert to one String: an `IdDict` keeps both.
    @test_throws "input: <dict>: duplicate key \"a\"" node(IdDict{Any, Any}("a" => 1, SubString("ab", 1, 1) => 2))
end

@testitem "dict: node missing" begin
    using QuantumEngine
    using QuantumEngine.Input
    @test node(Dict("x" => missing)).items[1].kind == NullNode
    # A null value reaches the walker as null: a wrong type for `title`.
    h2 = [Dict("symbol" => "H", "position" => [0.0, 0.0, -0.7]), Dict("symbol" => "H", "position" => [0.0, 0.0, 0.7])]
    d = Dict{String, Any}("unit" => "bohr", "geometry" => h2, "title" => missing)
    @test_throws "input: t.yaml: title: expected a string, got null" parse_dict(d; source = "t.yaml")
    p = parse_dict(d; source = "t.yaml", lenient = true)
    @test p.input.title == "" && [w.case for w in p.warnings] == [:yaml_wrong_type]
end

@testitem "dict: parse_dict equals parse_json" setup = [Fixtures] begin
    using QuantumEngine
    d = Dict{String, Any}(
        "title" => "w", "charge" => 0, "multiplicity" => 1, "basis" => "STO-3G", "method" => "b3lyp",
        "mode" => "opt", "dispersion" => true, "hubbard" => Dict("O" => 4.0), "unit" => "angstrom",
        "scf" => Dict{String, Any}(
            "max_iter" => 75, "conv_energy" => 1.0e-8, "damping" => 0.1, "field" => [0.0, 0.0, 0.01],
            "lattice" => [Any[5.0, 0, 0], Any[0, 5.0, 0], Any[0, 0, 5.0]], "pbc" => [true, true, false],
            "population" => ["mulliken"], "kpoints" => [2, 2, 2],
        ),
        "solvation" => Dict("model" => "smd", "solvent" => "water", "grid_order" => 110),
        "tddft" => Dict("num_states" => 3, "method" => "tda"),
        "constraints" => [Dict("freeze_atom" => 0), Dict("freeze_bond" => [0, 1], "target" => 1.8)],
        "geometry" => [
            Dict("symbol" => "O", "position" => [0.0, 0.0, 0.1173]),
            Dict("symbol" => "H", "x" => 0.0, "y" => 0.7572, "z" => -0.4692),
            Dict("symbol" => "H", "position" => [0.0, -0.7572, -0.4692]),
        ],
    )
    json = """
    {"title": "w", "charge": 0, "multiplicity": 1, "basis": "STO-3G", "method": "b3lyp", "mode": "opt",
     "dispersion": true, "hubbard": {"O": 4.0}, "unit": "angstrom",
     "scf": {"max_iter": 75, "conv_energy": 1e-8, "damping": 0.1, "field": [0.0, 0.0, 0.01],
             "lattice": [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]], "pbc": [true, true, false],
             "population": ["mulliken"], "kpoints": [2, 2, 2]},
     "solvation": {"model": "smd", "solvent": "water", "grid_order": 110},
     "tddft": {"num_states": 3, "method": "tda"},
     "constraints": [{"freeze_atom": 0}, {"freeze_bond": [0, 1], "target": 1.8}],
     "geometry": [{"symbol": "O", "position": [0.0, 0.0, 0.1173]}, {"symbol": "H", "x": 0.0, "y": 0.7572, "z": -0.4692},
                  {"symbol": "H", "position": [0.0, -0.7572, -0.4692]}]}
    """
    for lenient in (false, true)
        p = parse_dict(d; source = "t", lenient)
        q = parse_json(json; source = "t", lenient)
        @test Fixtures.fields_equal(p.input, q.input)
        # The warnings follow the key order, which a `Dict` does not keep.
        @test sort(string.(p.warnings)) == sort(string.(q.warnings))
        lenient || @test isempty(p.warnings)
    end
    # The fixture file read by JSON.jl into a Dict, and read as text.
    # JSON.jl is a dependency of the package, reached through `Input`.
    JSON = QuantumEngine.Input.JSON
    text = Fixtures.oracle_input("benchmarks/fe2_dmft.json")
    @test Fixtures.fields_equal(
        parse_dict(JSON.parse(text; dicttype = Dict{String, Any}); source = "f").input, parse_json(text; source = "f").input,
    )
end

@testitem "dict: units au and ang" begin
    using QuantumEngine
    using QuantumEngine.Units: ANGSTROM_TO_BOHR
    h2 = [Dict("symbol" => "H", "position" => [0.0, 0.0, -0.7]), Dict("symbol" => "H", "position" => [0.0, 0.0, 0.7])]
    for lenient in (false, true)
        p = parse_dict(Dict("unit" => "au", "geometry" => h2); source = "t", lenient)
        @test isempty(p.warnings)
        @test p.input.atoms[2].position == [0.0, 0.0, 0.7]
        p = parse_dict(Dict("unit" => "ang", "geometry" => h2); source = "t", lenient)
        @test isempty(p.warnings)
        @test p.input.atoms[2].position == [0.0, 0.0, 0.7 * ANGSTROM_TO_BOHR]
    end
end

@testitem "dict: YAML unavailable" begin
    using QuantumEngine
    using QuantumEngine.Input: _yaml_unavailable
    e = _yaml_unavailable("a.yaml")
    @test e isa InputError
    @test sprint(showerror, e) == "input: a.yaml: reading YAML needs `using YAML` (QuantumEngineYAMLExt)"
end
