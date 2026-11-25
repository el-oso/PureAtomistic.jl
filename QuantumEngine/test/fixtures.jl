@testmodule Fixtures begin
    using QuantumEngine

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
end
