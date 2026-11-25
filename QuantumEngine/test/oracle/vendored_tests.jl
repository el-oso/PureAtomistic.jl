@testitem "vendored inputs: count" begin
    root = joinpath(pkgdir(QuantumEngine), "test", "oracle", "inputs")
    exts = (".inp", ".in", ".yaml", ".yml", ".json", ".xyz")
    files = [f for (_, _, fs) in walkdir(root) for f in fs if splitext(f)[2] in exts]
    @test length(files) == 72
end

@testitem "vendored inputs are the oracle's files" tags = [:oracle_src] begin
    src = ENV["QUANTUM_ENGINE_ORACLE_SRC"]
    root = joinpath(pkgdir(QuantumEngine), "test", "oracle", "inputs")
    exts = (".inp", ".in", ".yaml", ".yml", ".json", ".xyz")
    rels = [
        relpath(joinpath(dir, f), root) for (dir, _, fs) in walkdir(root) for
            f in fs if splitext(f)[2] in exts
    ]
    @test length(rels) == 72
    for rel in rels
        @test read(joinpath(root, rel)) == read(joinpath(src, rel))
    end
end
