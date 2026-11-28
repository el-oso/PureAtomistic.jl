# The reader of the oracle's ECP JSON format: the oracle's built-in Br entry read back, and one
# error item per failure the reader reports.

@testmodule ECPFiles begin
    using QuantumEngine.Basis: parse_ecp_json, ECPChannel

    parse(text) = parse_ecp_json(IOBuffer(text); source = "t.json")

    # `ECPChannel` has no `==`; two channels are equal when all their fields are.
    same_channel(a::ECPChannel, b::ECPChannel) =
        a.l == b.l && a.powers == b.powers && a.exponents == b.exponents && a.coefficients == b.coefficients

    # The oracle's built-in Br potential, typed in as JSON: z_core 28, l_max 2, and one primitive
    # `(n, alpha, coeff)` per projector, l 0 `(2, 2.85, 14.2)`, l 1 `(2, 2.15, 8.6)` and
    # l 2 `(1, 1.45, 28.0)`.
    const BR = """
    {"symbol": "Br", "z_core": 28, "l_max": 2, "projectors": [
      {"l": 0, "primitives": [{"n": 2, "alpha": 2.85, "coeff": 14.2}]},
      {"l": 1, "primitives": [{"n": 2, "alpha": 2.15, "coeff": 8.6}]},
      {"l": 2, "primitives": [{"n": 1, "alpha": 1.45, "coeff": 28.0}]}]}"""

    # A one-entry document: `BR` with the first occurrence of `from` replaced by `to`.
    with(from, to) = "[" * replace(BR, from => to; count = 1) * "]"
end

@testitem "ecp json: oracle format" setup = [ECPFiles] begin
    using QuantumEngine.Basis: ECPChannel, ECPPotential, parse_ecp_json, read_ecp_json
    P = ECPFiles.parse
    same = ECPFiles.same_channel
    expected = [
        ECPChannel{Float64}(0, [2], [2.85], [14.2]),
        ECPChannel{Float64}(1, [2], [2.15], [8.6]),
        ECPChannel{Float64}(2, [1], [1.45], [28.0]),
    ]
    check(d) = d isa Dict{String, ECPPotential{Float64}} && collect(keys(d)) == ["Br"] &&
        d["Br"].symbol == "Br" && d["Br"].ncore == 28 && d["Br"].lmax == 2 &&
        length(d["Br"].channels) == 3 && all(same(a, b) for (a, b) in zip(d["Br"].channels, expected))
    @test check(P("[" * ECPFiles.BR * "]"))
    # A lowercase symbol is keyed and stored capitalized.
    @test check(P(ECPFiles.with("\"Br\"", "\"br\"")))
    # The default source and a file read through `read_ecp_json`.
    @test check(parse_ecp_json(IOBuffer("[" * ECPFiles.BR * "]")))
    mktempdir() do dir
        path = joinpath(dir, "w.json")
        write(path, "[" * ECPFiles.BR * "]")
        @test check(read_ecp_json(path))
        write(path, "{}")
        @test_throws "basis: $(path): \$: expected an array of ECP entries" read_ecp_json(path)
    end
    # The channels keep the order of the projectors, and an entry's primitives theirs.
    two = """[{"symbol": "Ag", "z_core": 28, "l_max": 1, "projectors": [
    {"l": 1, "primitives": [{"n": 1, "alpha": 3.0, "coeff": -1.0}, {"n": 0, "alpha": 0.5, "coeff": 2.5}]},
    {"l": 0, "primitives": []}]},
    {"z_core": 2, "l_max": 0, "projectors": [], "symbol": "LI"}]"""
    d = P(two)
    @test sort!(collect(keys(d))) == ["Ag", "Li"]
    @test same(d["Ag"].channels[1], ECPChannel{Float64}(1, [1, 0], [3.0, 0.5], [-1.0, 2.5]))
    @test same(d["Ag"].channels[2], ECPChannel{Float64}(0, Int[], Float64[], Float64[]))
    @test d["Li"].symbol == "Li" && d["Li"].ncore == 2 && d["Li"].lmax == 0 && isempty(d["Li"].channels)
    @test isempty(P("[]"))
    # An integer is a valid number for `alpha` and `coeff`.
    @test P(ECPFiles.with("2.85", "3"))["Br"].channels[1].exponents == [3.0]
end

@testitem "ecp json: errors" setup = [ECPFiles] begin
    P = ECPFiles.parse
    W = ECPFiles.with
    prim = "[1].projectors[1].primitives[1]"
    # The top level is not an array.
    @test_throws "basis: t.json: \$: expected an array of ECP entries" P("{}")
    @test_throws "basis: t.json: \$: expected an array of ECP entries" P("3")
    # An unknown key, in an entry, a projector and a primitive.
    @test_throws "basis: t.json: [1]: unexpected key \"zcore\"" P(W("\"z_core\"", "\"zcore\""))
    @test_throws "basis: t.json: [1].projectors[1]: unexpected key \"L\"" P(W("\"l\":", "\"L\":"))
    @test_throws "basis: t.json: $(prim): unexpected key \"exp\"" P(W("\"n\": 2,", "\"n\": 2, \"exp\": 1,"))
    # A missing key.
    @test_throws "basis: t.json: [1]: missing key \"l_max\"" P(W("\"l_max\": 2, ", ""))
    @test_throws "basis: t.json: [1].projectors[1]: missing key \"primitives\"" P(W(", \"primitives\": [{\"n\": 2, \"alpha\": 2.85, \"coeff\": 14.2}]", ""))
    @test_throws "basis: t.json: $(prim): missing key \"coeff\"" P(W(", \"coeff\": 14.2", ""))
    # A value of the wrong type.
    @test_throws "basis: t.json: $(prim).alpha: expected a number" P(W("2.85", "\"2.85\""))
    @test_throws "basis: t.json: $(prim).coeff: expected a number" P(W("14.2", "true"))
    @test_throws "basis: t.json: $(prim).n: expected an integer" P(W("\"n\": 2", "\"n\": 2.0"))
    @test_throws "basis: t.json: [1].z_core: expected an integer" P(W("28", "null"))
    @test_throws "basis: t.json: [1].l_max: expected an integer" P(W("\"l_max\": 2", "\"l_max\": true"))
    @test_throws "basis: t.json: [1].projectors[1].l: expected an integer" P(W("\"l\": 0", "\"l\": \"0\""))
    @test_throws "basis: t.json: [1].symbol: expected a string" P(W("\"Br\"", "35"))
    @test_throws "basis: t.json: [1].projectors: expected an array" P("[{\"symbol\": \"Br\", \"z_core\": 28, \"l_max\": 2, \"projectors\": {}}]")
    @test_throws "basis: t.json: [1].projectors[1].primitives: expected an array" P(W("\"primitives\": [{\"n\": 2, \"alpha\": 2.85, \"coeff\": 14.2}]", "\"primitives\": 1"))
    @test_throws "basis: t.json: [2]: expected an object" P("[" * ECPFiles.BR * ", []]")
    @test_throws "basis: t.json: [1].projectors[1]: expected an object" P(W("{\"l\": 0, \"primitives\": [{\"n\": 2, \"alpha\": 2.85, \"coeff\": 14.2}]}", "0"))
    @test_throws "basis: t.json: $(prim): expected an object" P(W("{\"n\": 2, \"alpha\": 2.85, \"coeff\": 14.2}", "[2, 2.85, 14.2]"))
    # An integer beyond the Int64 range.
    @test_throws "basis: t.json: [1].z_core: the integer 99999999999999999999 is outside the Int64 range" P(W("28", "99999999999999999999"))
    # A negative `z_core` or `l_max`.
    @test_throws "basis: t.json: [1].z_core: must be ≥ 0, got -1" P(W("28", "-1"))
    @test_throws "basis: t.json: [1].l_max: must be ≥ 0, got -1" P(W("\"l_max\": 2", "\"l_max\": -1"))
    # A negative power `n`.
    @test_throws "basis: t.json: $(prim).n: must be ≥ 0, got -1" P(W("\"n\": 2", "\"n\": -1"))
    # A `z_core` above the atomic number; equal to it is accepted.
    @test_throws "basis: t.json: [1].z_core: z_core 36 is above the atomic number 35" P(W("28", "36"))
    @test P(W("28", "35"))["Br"].ncore == 35
    # A projector `l` outside `0:l_max`.
    @test_throws "basis: t.json: [1].projectors[2].l: l = 3 is outside 0:2" P(W("\"l\": 1", "\"l\": 3"))
    @test_throws "basis: t.json: [1].projectors[1].l: l = -1 is outside 0:2" P(W("\"l\": 0", "\"l\": -1"))
    # A non-finite `alpha` or `coeff`: JSON has no NaN or Inf literal, and a number beyond the
    # Float64 range reads as ±Inf.
    @test_throws "basis: t.json: $(prim).coeff: must be finite, got Inf" P(W("14.2", "1e400"))
    @test_throws "basis: t.json: $(prim).alpha: must be positive and finite, got -Inf" P(W("2.85", "-1e400"))
    @test_throws "basis: t.json: $(prim).alpha: must be positive and finite, got Inf" P(W("2.85", "1e400"))
    # An `alpha` of zero or below; a number below the Float64 range reads as 0.
    @test_throws "basis: t.json: $(prim).alpha: must be positive and finite, got 0.0" P(W("2.85", "0"))
    @test_throws "basis: t.json: $(prim).alpha: must be positive and finite, got -1.5" P(W("2.85", "-1.5"))
    @test_throws "basis: t.json: $(prim).alpha: must be positive and finite, got 0.0" P(W("2.85", "1e-400"))
    # A negative or zero `coeff` is accepted.
    @test P(W("14.2", "-14.2"))["Br"].channels[1].coefficients == [-14.2]
    @test P(W("14.2", "0"))["Br"].channels[1].coefficients == [0.0]
    # An unknown element symbol.
    @test_throws "basis: t.json: [1].symbol: unknown element symbol \"Xx\"" P(W("\"Br\"", "\"Xx\""))
    # A repeated symbol, in any ASCII case.
    br = ECPFiles.BR
    li = """{"symbol": "Li", "z_core": 2, "l_max": 0, "projectors": []}"""
    @test_throws "basis: t.json: [3]: second entry for Br (first at [1])" P("[$(br), $(li), $(replace(br, "\"Br\"" => "\"BR\""))]")
    # A repeated key, in an entry and in a primitive.
    @test_throws "basis: t.json: [1]: duplicate key \"symbol\"" P(W("\"z_core\"", "\"symbol\": \"Br\", \"z_core\""))
    @test_throws "basis: t.json: $(prim): duplicate key \"n\"" P(W("\"n\": 2,", "\"n\": 2, \"n\": 3,"))
    # Text that is not JSON.
    @test_throws "basis: t.json: \$: not valid JSON (" P("[{")
    @test_throws "basis: t.json: \$: not valid JSON (" P("[NaN]")
end
