@testmodule Fixtures begin
    using QuantumEngine
    using QuantumEngine.Units: ANGSTROM_TO_BOHR

    """
        oracle_input_path(relpath) -> String

    Path of the vendored oracle input file `relpath`, relative to
    `test/oracle/inputs`.
    """
    oracle_input_path(relpath::AbstractString) =
        joinpath(pkgdir(QuantumEngine), "test", "oracle", "inputs", relpath)::String

    """
        oracle_input(relpath) -> String

    Text of the vendored oracle input file `relpath`.
    """
    oracle_input(relpath::AbstractString) = read(oracle_input_path(relpath), String)

    """
        fields_equal(a, b) -> Bool

    Whether every field of `a` and `b` is `==`. Structs with `Vector` fields have
    no `==` method; apply this recursively where a field is itself such a struct.
    """
    fields_equal(a, b) = typeof(a) === typeof(b) && all(f -> getfield(a, f) == getfield(b, f), fieldnames(typeof(a)))

    """
        h2() -> Molecule{Float64}

    The molecule of `examples/h2.inp` (coordinates in bohr).
    """
    h2() = Molecule(
        [
            Atom("H", [0.0, 0.0, -0.7]),
            Atom("H", [0.0, 0.0, 0.7]),
        ]
    )

    """
        water() -> Molecule{Float64}

    The molecule of `examples/h2o_631g_sp.inp`, its Å coordinates converted to bohr.
    """
    water() = Molecule(
        [
            Atom("O", [0.0, 0.0, 0.1173] .* ANGSTROM_TO_BOHR),
            Atom("H", [0.0, 0.7572, -0.4692] .* ANGSTROM_TO_BOHR),
            Atom("H", [0.0, -0.7572, -0.4692] .* ANGSTROM_TO_BOHR),
        ]
    )

    """
        o2_triplet() -> Molecule{Float64}

    The triplet molecule of `examples/o2_triplet.inp` (multiplicity 3), its Å coordinates
    converted to bohr.
    """
    o2_triplet() = Molecule(
        [
            Atom("O", [0.0, 0.0, 0.0] .* ANGSTROM_TO_BOHR),
            Atom("O", [1.208, 0.0, 0.0] .* ANGSTROM_TO_BOHR),
        ];
        multiplicity = 3,
    )
end
