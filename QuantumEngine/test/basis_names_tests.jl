# Basis names and the search path: the name normalization and its alias table, the candidate
# stems, the order of the search directories and of the lookup, and the lookup's errors.

@testitem "basis: name normalization" begin
    using QuantumEngine.Basis: normalize_basis_name, BASIS_ALIASES, _ALIAS_GROUPS
    @test normalize_basis_name("6-31G*") == "6-31g_star"
    @test normalize_basis_name("  6-31G**  ") == "6-31g_star_star"
    @test normalize_basis_name("6-311++g(d,p)") == "6-311plusplusg_star_star"
    @test normalize_basis_name("6-311++G(2d,2p)") == "6-311plusplusg(2d,2p)"
    @test normalize_basis_name("def2-SVP-JKFIT") == "def2-universal-jkfit"
    @test normalize_basis_name("631-g") == "6-31g"
    @test normalize_basis_name("MyBasis") == "mybasis"
    @test normalize_basis_name(SubString("x STO3G ", 2)) == "sto-3g"
    @test normalize_basis_name("6-31g(d,p)") isa String
    # The oracle's 139 names and the one added row, none in two groups.
    @test sum(length(first(g)) for g in _ALIAS_GROUPS) == length(BASIS_ALIASES) == 140
    @test all(k == lowercase(strip(k)) for k in keys(BASIS_ALIASES))
    # Idempotence on the values.
    @test all(normalize_basis_name(v) == v for v in values(BASIS_ALIASES))
end

@testitem "basis: candidate stems" begin
    using QuantumEngine.Basis: candidate_stems
    stem_variants = QuantumEngine.Basis.stem_variants
    @test candidate_stems("6-31+G**") == ["6-31+g**", "6-31+g_star_star", "6-31+g_dp", "6-31plusg_star_star"]
    @test stem_variants("6-311++g_2d2p") == ["6-311++g_2d2p", "6-311plusplusg_2d2p"]
    @test candidate_stems("x(d)") == ["x(d)", "x_star"]
    @test candidate_stems("def2-SVP-JKFIT") == ["def2-universal-jkfit"]
    @test stem_variants("def2-svp-jkfit") == ["def2-svp-jkfit"]
    # An aliased name gives its stem first.
    @test first(candidate_stems("6-31G**")) == "6-31g_star_star"
    @test candidate_stems(SubString(" 6-31G* ", 1)) == ["6-31g_star"]
end

@testitem "search path order" begin
    using QuantumEngine.Basis: basis_search_paths, locate_basis_file
    using OffsetArrays: OffsetVector
    pkgdata = joinpath(pkgdir(QuantumEngine), "data", "basis_sets")
    root = realpath(mktempdir())
    cwd = mkpath(joinpath(root, "a", "b"))
    envdir = mkpath(joinpath(root, "env"))
    mkpath(joinpath(cwd, "basis_sets"))
    mkpath(joinpath(root, "basis_sets"))
    touch(joinpath(cwd, "basis_sets", "x_star.gbs"))
    touch(joinpath(envdir, "x(d).gbs"))
    touch(joinpath(root, "basis_sets", "z.gbs"))
    env = Dict("QUANTUM_ENGINE_BASIS_DIR" => envdir)
    cd(cwd) do
        # Stem order wins over path order: the first stem `x(d)` is found in the env directory
        # before the second stem `x_star` in the working directory's `basis_sets`.
        @test locate_basis_file("x(d)"; paths = basis_search_paths(env = env)) == joinpath(envdir, "x(d).gbs")
        @test locate_basis_file("x(d)"; paths = basis_search_paths(env = Dict{String, String}())) ==
            joinpath(cwd, "basis_sets", "x_star.gbs")
        # A relative directory resolves against the working directory at the lookup.
        @test locate_basis_file("Z"; paths = basis_search_paths(env = env)) == joinpath(root, "basis_sets", "z.gbs")
    end
    rel = ["basis_sets", "../basis_sets", "../../basis_sets"]
    @test basis_search_paths(env = env) == [rel; envdir; pkgdata]
    exe = joinpath(root, "bin")
    @test basis_search_paths(env = env, exe_dir = exe) == [
        rel; envdir; joinpath(exe, "basis_sets"); joinpath(exe, "..", "basis_sets");
        joinpath(exe, "..", "..", "basis_sets"); pkgdata
    ]
    @test basis_search_paths(env = Dict("QUANTUM_ENGINE_BASIS_DIR" => "")) == [rel; pkgdata]
    @test basis_search_paths(env = Dict{String, String}()) == [rel; pkgdata]
    # Path order within one stem.
    p1 = mkpath(joinpath(root, "p1"))
    p2 = mkpath(joinpath(root, "p2"))
    touch(joinpath(p1, "y.gbs"))
    touch(joinpath(p2, "y.gbs"))
    @test locate_basis_file("y"; paths = [p1, p2]) == joinpath(p1, "y.gbs")
    @test locate_basis_file("y"; paths = [p2, p1]) == joinpath(p2, "y.gbs")
    # The search paths as an OffsetArray and as a view.
    @test locate_basis_file("y"; paths = OffsetVector([p2, p1], -3)) == joinpath(p2, "y.gbs")
    @test locate_basis_file("y"; paths = view([p1, p2, p1], 2:3)) == joinpath(p2, "y.gbs")
    # A directory with the file's name is not a file.
    mkpath(joinpath(p1, "w.gbs"))
    touch(joinpath(p2, "w.gbs"))
    @test locate_basis_file("w"; paths = [p1, p2]) == joinpath(p2, "w.gbs")
    # No file: every stem and every directory, in order.
    @test_throws "basis: no file for \"6-31+G**\" (stems 6-31+g**, 6-31+g_star_star, 6-31+g_dp, 6-31plusg_star_star; searched: $p1, $p2)" locate_basis_file("6-31+G**"; paths = [p1, p2])
    @test_throws "basis: no file for \"q\" (stems q; searched: )" locate_basis_file("q"; paths = String[])
    # A name with a path separator, on every platform.
    for name in ("/tmp/x", "../x", "..\\x")
        @test_throws "basis: basis name \"$name\" contains a path separator; put the file in a directory on the search path or set QUANTUM_ENGINE_BASIS_DIR" locate_basis_file(name; paths = [p1])
    end
    rm(root; recursive = true)
end
