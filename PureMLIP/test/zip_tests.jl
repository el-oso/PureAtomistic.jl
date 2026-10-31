@testitem "crc32 matches the IEEE check value" begin
    @test PureMLIP.Checkpoint.crc32(codeunits("123456789")) == 0xCBF43926
    @test PureMLIP.Checkpoint.crc32(UInt8[]) == 0x00000000
end

@testitem "stored zip round trip" setup = [ZipFixtures] begin
    z = ZipFixtures.stored_zip(["arch/data.pkl" => UInt8[0x80, 0x02, 0x2e], "arch/byteorder" => collect(codeunits("little"))])
    a = PureMLIP.open_torch_archive(z)
    @test a.prefix == "arch/"
    @test collect(PureMLIP.entry_data(a, "byteorder")) == collect(codeunits("little"))
    # Classic (non-ZIP64) archives read the same way.
    a32 = PureMLIP.open_torch_archive(ZipFixtures.stored_zip(["arch/data.pkl" => UInt8[0x80, 0x02, 0x2e], "arch/byteorder" => collect(codeunits("little"))]; zip64 = false))
    @test collect(PureMLIP.entry_data(a32, "byteorder")) == collect(codeunits("little"))
end

@testitem "zip errors" setup = [ZipFixtures] begin
    z = ZipFixtures.stored_zip(["a/data.pkl" => UInt8[1, 2, 3]]; method = 0x000c)
    @test_throws "compressed entries are not supported: a/data.pkl (method 12)" PureMLIP.open_torch_archive(z)
    good = ZipFixtures.stored_zip(["a/data.pkl" => UInt8[1, 2, 3]])
    bad = copy(good); bad[31 + length("a/data.pkl")] ⊻= 0xff   # first data byte
    @test_throws "CRC mismatch in a/data.pkl" PureMLIP.entry_data(PureMLIP.open_torch_archive(bad), "data.pkl")
    @test_throws "no end-of-central-directory record" PureMLIP.open_torch_archive(good[1:(end - 30)])
    @test_throws "no data.pkl entry" PureMLIP.open_torch_archive(ZipFixtures.stored_zip(["a/x" => UInt8[1]]))
end

@testitem "zip entry access" setup = [ZipFixtures] begin
    a = PureMLIP.open_torch_archive(ZipFixtures.stored_zip(["a/data.pkl" => UInt8[1, 2, 3]]))
    @test_throws "no entry a/missing" PureMLIP.entry_data(a, "missing")
    @test PureMLIP.entry_data(a, "data.pkl") == UInt8[1, 2, 3]
    @test "a/data.pkl" in a.verified
    # A second access does not re-verify: corrupting the buffer after the first access goes unnoticed.
    a.bytes[31 + length("a/data.pkl")] ⊻= 0xff
    @test PureMLIP.entry_data(a, "data.pkl")[1] == 0xfe
end

@testitem "zip duplicate entries" setup = [ZipFixtures] begin
    pkl = UInt8[1, 2, 3]
    @test_throws "duplicate zip entry: a/x" PureMLIP.open_torch_archive(ZipFixtures.stored_zip(["a/data.pkl" => pkl, "a/x" => UInt8[1], "a/x" => UInt8[2]]))
    @test_throws "more than one data.pkl entry in the archive: a/data.pkl and b/data.pkl" PureMLIP.open_torch_archive(ZipFixtures.stored_zip(["a/data.pkl" => pkl, "b/data.pkl" => pkl]))
end

@testitem "deflated entry is decoded and CRC-checked" setup = [ZipFixtures] begin
    z = ZipFixtures.stored_zip(["a/data.pkl" => UInt8[0x80, 0x02, 0x2e], "a/code.py" => collect(codeunits("x = 1\n"))]; deflate = ["a/code.py"])
    a = PureMLIP.open_torch_archive(z)
    @test collect(PureMLIP.entry_data(a, "code.py")) == collect(codeunits("x = 1\n"))
    @test collect(PureMLIP.entry_data(a, "data.pkl")) == UInt8[0x80, 0x02, 0x2e]
    @test a.entries["a/code.py"].method == 0x0008
    @test a.entries["a/code.py"].compressed_size == 6 + 5
    # The decoded bytes are cached: a second access returns the same buffer.
    @test parent(PureMLIP.entry_data(a, "code.py")) === parent(PureMLIP.entry_data(a, "code.py"))
    z32 = ZipFixtures.stored_zip(["a/data.pkl" => UInt8[1], "a/code.py" => collect(codeunits("y"))]; deflate = ["a/code.py"], zip64 = false)
    @test collect(PureMLIP.entry_data(PureMLIP.open_torch_archive(z32), "code.py")) == collect(codeunits("y"))
end

@testitem "deflated entry errors" setup = [ZipFixtures] begin
    mk() = ZipFixtures.stored_zip(["a/data.pkl" => UInt8[1], "a/code.py" => collect(codeunits("abcd"))]; deflate = ["a/code.py"])
    z = mk()
    pos = findfirst(codeunits("abcd"), z)
    bad = copy(z); bad[first(pos)] ⊻= 0xff
    @test_throws "CRC mismatch in a/code.py" PureMLIP.entry_data(PureMLIP.open_torch_archive(bad), "code.py")
    # Corrupt the stored-block LEN/NLEN pair so the stream length disagrees with the recorded size.
    short = copy(z); short[first(pos) - 4] = 0x03; short[first(pos) - 2] = 0xfc
    @test_throws "inflated size mismatch in a/code.py" PureMLIP.entry_data(PureMLIP.open_torch_archive(short), "code.py")
end

@testitem "entry_data has one concrete return type" begin
    T = SubArray{UInt8, 1, Vector{UInt8}, Tuple{UnitRange{Int}}, true}
    @test Base.return_types(PureMLIP.entry_data, (PureMLIP.TorchArchive, String)) == [T]
end
