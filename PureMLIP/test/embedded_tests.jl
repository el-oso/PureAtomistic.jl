@testitem "embedded archives are found in pickle bytes" setup = [ZipFixtures] begin
    inner = ZipFixtures.stored_zip(["i/data.pkl" => UInt8[0x80, 0x02, 0x2e], "i/data/0" => UInt8[1, 2, 3]])
    nodes = [
        PureMLIP.PickleNode(PureMLIP.PK_BYTES; str = String(copy(inner))),
        PureMLIP.PickleNode(PureMLIP.PK_BYTES; str = "not a zip"),
        PureMLIP.PickleNode(PureMLIP.PK_BYTES; str = String(copy(inner))),
        PureMLIP.PickleNode(PureMLIP.PK_STR; str = String(copy(inner))),
    ]
    archives = PureMLIP.embedded_archives(PureMLIP.PickleTree(nodes, Int32(1)))
    @test length(archives) == 2
    @test archives[1].prefix == "i/"
    @test collect(PureMLIP.entry_data(archives[2], "data/0")) == UInt8[1, 2, 3]
    @test isempty(PureMLIP.embedded_archives(PureMLIP.PickleTree(nodes[2:2], Int32(1))))
end

@testitem "MP-0b3 embeds one archive with Wigner tables" setup = [Checkpoints] begin
    a = PureMLIP.open_torch_archive(read(Checkpoints.fetch(:mp0b3)))
    archives = PureMLIP.embedded_archives(PureMLIP.unpickle(PureMLIP.entry_data(a, "data.pkl")))
    @test length(archives) == 18
    with_data = filter(x -> haskey(x.entries, x.prefix * "data/0"), archives)
    @test length(with_data) == 1
    @test [with_data[1].entries[with_data[1].prefix * "data/$k"].size for k in 0:3] == [360, 360, 840, 840]
end

@testitem "MP-0b3 embedded data entries are deflated and inflate with a valid CRC" setup = [Checkpoints] begin
    a = PureMLIP.open_torch_archive(read(Checkpoints.fetch(:mp0b3)))
    archives = PureMLIP.embedded_archives(PureMLIP.unpickle(PureMLIP.entry_data(a, "data.pkl")))
    inner = only(filter(x -> haskey(x.entries, x.prefix * "data/0"), archives))
    code = filter(e -> startswith(e.name, inner.prefix * "code/"), collect(values(inner.entries)))
    @test !isempty(code)
    for e in code
        @test e.method == 8
        @test !isempty(PureMLIP.entry_data(inner, e.name[(ncodeunits(inner.prefix) + 1):end]))
    end
end
