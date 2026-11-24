# Mutated inputs must either load or fail with a reader-raised error: an ArgumentError thrown
# from a frame under PureMLIP/src, or an ErrorException whose message starts with "checkpoint:"
# or "unpickle:". An ArgumentError raised by Base (for example from an allocation size) is a
# failure. Failures print the trial number and the mutation so they can be reproduced from the
# fixed seed. Each item counts its outcomes and asserts that both loads and clean errors
# occur, so a loop in which every input fails the same way cannot pass.

@testmodule Fuzz begin
    using Random
    import PureMLIP

    const SRC_DIR = dirname(pathof(PureMLIP))

    # The throw site is the first non-C frame of the backtrace: a reader-raised ArgumentError
    # starts in a PureMLIP source file, a Base-raised one in a Base file.
    function thrown_in_src(bt)
        frames = filter(f -> !f.from_c, stacktrace(bt))
        return !isempty(frames) && startswith(String(first(frames).file), SRC_DIR * "/")
    end

    allowed(e, bt) = (e isa ArgumentError && thrown_in_src(bt)) ||
        (e isa ErrorException && occursin(r"^(checkpoint|unpickle):", e.msg))

    mutable struct Tally
        loaded::Int
        clean_error::Int
    end
    Tally() = Tally(0, 0)

    # Counts a clean outcome; returns whether `r` is one.
    function count!(tally::Tally, r)
        r === :loaded && (tally.loaded += 1; return true)
        r === :clean_error && (tally.clean_error += 1; return true)
        return false
    end

    # Applies one random mutation to a copy of `good`; returns the copy and a description.
    # `lo:hi` is the byte range that flips and window overwrites target.
    # Truncations keep at least `trunc_lo` bytes.
    function mutate(rng, good::Vector{UInt8}, kind::Integer, lo::Integer = 1, hi::Integer = length(good); trunc_lo::Integer = 0)
        bad = copy(good)
        if kind == 0
            n = rand(rng, 1:8)
            idx = [rand(rng, lo:hi) for _ in 1:n]
            for i in idx
                bad[i] ⊻= rand(rng, UInt8(1):UInt8(255))
            end
            return bad, "flip bytes at $idx"
        elseif kind == 1
            len = rand(rng, trunc_lo:(length(bad) - 1))
            resize!(bad, len)
            return bad, "truncate to $len bytes"
        else
            w = rand(rng, 1:min(64, hi - lo + 1))
            start = rand(rng, lo:(hi - w + 1))
            bad[start:(start + w - 1)] = rand(rng, UInt8, w)
            return bad, "overwrite $w random bytes at $start"
        end
    end

    # Overwrites the 4-byte operand of a random BININT (J) or BINUNICODE (X) opcode with an
    # extreme value: the sizes, shapes and offsets that drive allocation and indexing.
    function mutate_operand(rng, good::Vector{UInt8})
        bad = copy(good)
        ops = [i for i in eachindex(good) if (good[i] == UInt8('J') || good[i] == UInt8('X')) && i + 4 <= length(good)]
        isempty(ops) && return bad, "no operand to overwrite"
        p = rand(rng, ops)
        v = rand(rng, Int32[typemax(Int32), typemin(Int32), -1, 0, 1, typemax(Int32) - 1, 65536, 1 << 24])
        bad[(p + 1):(p + 4)] = collect(reinterpret(UInt8, [htol(v)]))
        return bad, "set the operand after byte $p ($(Char(good[p]))) to $v"
    end

    # Fills a random 1-8 byte window with 0xff or 0x00: the sentinels and extreme sizes of
    # container headers.
    function mutate_fill(rng, good::Vector{UInt8})
        bad = copy(good)
        w = rand(rng, 1:8)
        start = rand(rng, 1:(length(bad) - w + 1))
        v = rand(rng, (0x00, 0xff))
        bad[start:(start + w - 1)] .= v
        return bad, "fill $w bytes at $start with $v"
    end

    # (:loaded | :clean_error | the unexpected exception, seconds)
    function attempt(f, arg)
        t = @elapsed r = try
            f(arg)
            :loaded
        catch e
            allowed(e, catch_backtrace()) ? :clean_error : e
        end
        return r, t
    end
end

# Mutations of the whole archive: flips in the data.pkl bytes or the central directory,
# truncations within the last 64 KiB (the central directory and end-of-central-directory
# record), and overwrites in the data.pkl bytes. Every entry is CRC-checked before use, so a
# mutated data.pkl is rejected by the CRC check and never reaches the pickle parser; this
# item covers the central-directory parsing, the EOCD search and the CRC check. The pickle
# parser and the readers behind it are fuzzed by the items below.
@testitem "fuzzed MP-0b3 checkpoints fail cleanly" setup = [Fuzz, Checkpoints] begin
    using Random
    rng = Xoshiro(20261031)
    good = read(Checkpoints.fetch(:mp0b3))
    a = PureMLIP.open_torch_archive(good)
    pkl = a.entries[a.prefix * "data.pkl"]
    cd_start = Int(PureMLIP.Checkpoint.read_le(good, PureMLIP.Checkpoint.find_eocd(good) + 16, UInt32)) + 1
    tail = 64 * 1024
    tally = Fuzz.Tally()
    total = Ref(0.0)
    for trial in 1:300
        kind = trial % 3
        bad, what = if kind == 0       # flips in data.pkl (when the trial is odd) or the central directory
            iseven(trial) ? Fuzz.mutate(rng, good, 0, cd_start, length(good)) :
                Fuzz.mutate(rng, good, 0, pkl.data_offset, pkl.data_offset + pkl.size - 1)
        elseif kind == 1
            Fuzz.mutate(rng, good, 1; trunc_lo = length(good) - tail)
        else
            Fuzz.mutate(rng, good, 2, pkl.data_offset, pkl.data_offset + pkl.size - 1)
        end
        r, t = Fuzz.attempt(PureMLIP.load_mace_checkpoint, bad)
        total[] += t
        ok = Fuzz.count!(tally, r)
        ok || @info "fuzz failure" trial what r
        @test ok
        @test t < 5.0
    end
    @info "MP-0b3 fuzz" total_seconds = total[] tally
    @test tally.loaded > 0
    @test tally.clean_error > 0
end

@testitem "fuzzed pickle sections of a small checkpoint fail cleanly" setup = [Fuzz, ZipFixtures] begin
    using Random
    rng = Xoshiro(20261101)
    a = PureMLIP.open_torch_archive(ZipFixtures.tiny_mace_zip())
    pkl = collect(PureMLIP.entry_data(a, "data.pkl"))
    storages = Pair{String, Vector{UInt8}}["0" => collect(PureMLIP.entry_data(a, "data/0")), "1" => collect(PureMLIP.entry_data(a, "data/1"))]
    @test PureMLIP.load_mace_checkpoint(ZipFixtures.zip_with_pickle(pkl, storages)) isa PureMLIP.MACECheckpoint
    tally = Fuzz.Tally()
    for trial in 1:2000
        bad, what = trial % 2 == 0 ? Fuzz.mutate_operand(rng, pkl) : Fuzz.mutate(rng, pkl, trial % 3)
        r, t = Fuzz.attempt(PureMLIP.load_mace_checkpoint, ZipFixtures.zip_with_pickle(bad, storages))
        ok = Fuzz.count!(tally, r)
        ok || @info "fuzz failure" trial what r
        @test ok
        @test t < 5.0
    end
    @info "small checkpoint pickle fuzz" tally
    @test tally.loaded > 0
    @test tally.clean_error > 0
end

@testitem "fuzzed pickles fail cleanly" setup = [Fuzz] begin
    using Random
    rng = Xoshiro(20261102)
    le32(x) = collect(reinterpret(UInt8, [htol(Int32(x))]))
    good = UInt8[
        0x80, 0x02, UInt8('}'), UInt8('('), UInt8('X'), 0x01, 0, 0, 0, UInt8('a'),
        UInt8('K'), 0x05, UInt8('X'), 0x01, 0, 0, 0, UInt8('b'),
        UInt8('G'), reinterpret(UInt8, [hton(6.0)])..., UInt8('u'),
        UInt8(']'), UInt8('('), UInt8('J'), le32(-7)..., UInt8('M'), 0x34, 0x12, UInt8('N'), 0x88, 0x89,
        0x85, UInt8('e'), UInt8('q'), 0x00,
        UInt8('c'), codeunits("collections\nOrderedDict\n")..., UInt8(')'), UInt8('R'),
        UInt8('h'), 0x00, UInt8('('), UInt8('h'), 0x00, UInt8('t'), UInt8('Q'), 0x86, 0x87, 0x86, UInt8('.'),
    ]
    # The unmutated pickle parses, so a mutation's outcome is attributable to the mutation.
    @test PureMLIP.unpickle(good) isa PureMLIP.PickleTree
    tally = Fuzz.Tally()
    for trial in 1:1000
        bad, what = if trial % 4 == 0   # extra: insert a random opcode byte
            b = copy(good)
            insert!(b, rand(rng, 3:length(b)), rand(rng, UInt8))
            b, "insert a random byte"
        else
            Fuzz.mutate(rng, good, trial % 4 == 3 ? 1 : (trial % 4 == 1 ? 0 : 2))
        end
        r, t = Fuzz.attempt(PureMLIP.unpickle, bad)
        ok = Fuzz.count!(tally, r)
        ok || @info "fuzz failure" trial what r
        @test ok
        @test t < 1.0
    end
    @info "pickle fuzz" tally
    @test tally.loaded > 0
    @test tally.clean_error > 0
end

@testitem "fuzzed small checkpoint archives fail cleanly" setup = [Fuzz, ZipFixtures] begin
    using Random
    rng = Xoshiro(20261104)
    tally = Fuzz.Tally()
    for trial in 1:5000
        good = ZipFixtures.tiny_mace_zip()
        bad, what = trial % 4 == 3 ? Fuzz.mutate_fill(rng, good) : Fuzz.mutate(rng, good, trial % 4)
        r, t = Fuzz.attempt(PureMLIP.load_mace_checkpoint, bad)
        ok = Fuzz.count!(tally, r)
        ok || @info "fuzz failure" trial what r
        @test ok
        @test t < 5.0
    end
    @info "small archive fuzz" tally
    @test tally.loaded > 0
    @test tally.clean_error > 0
end

@testitem "fuzzed MP-0b3 pickle section fails cleanly" setup = [Fuzz, Checkpoints] begin
    using Random
    rng = Xoshiro(20261103)
    a = PureMLIP.open_torch_archive(read(Checkpoints.fetch(:mp0b3)))
    pkl = collect(PureMLIP.entry_data(a, "data.pkl"))
    # The mutated pickle skips the archive's CRC check, so it reaches the hyperparameter and
    # state-dict readers against the real storages.
    function read_tree(bytes)
        t = PureMLIP.unpickle(bytes)
        t.nodes[t.root].kind == PureMLIP.PK_CALL || error("checkpoint: the pickle root is not a module")
        PureMLIP.Checkpoint.read_hyperparameters(a, t)
        return PureMLIP.Checkpoint.read_state_dict(a, t)
    end
    @test read_tree(pkl) isa PureMLIP.StateDict
    tally = Fuzz.Tally()
    for trial in 1:1000
        bad, what = trial % 2 == 0 ? Fuzz.mutate_operand(rng, pkl) : Fuzz.mutate(rng, pkl, trial % 3)
        r, t = Fuzz.attempt(read_tree, bad)
        ok = Fuzz.count!(tally, r)
        ok || @info "fuzz failure" trial what r
        @test ok
        @test t < 5.0
    end
    @info "MP-0b3 pickle fuzz" tally
    @test tally.loaded > 0
    @test tally.clean_error > 0
end

# `allowed` accepts an ArgumentError thrown from PureMLIP/src and rejects one thrown by Base.
@testitem "fuzz error classifier separates reader errors from Base errors" setup = [Fuzz] begin
    function caught(f)
        try
            f()
        catch e
            return e, catch_backtrace()
        end
        error("expected an exception")
    end
    e, bt = caught(() -> Vector{UInt8}(undef, -1))
    @test e isa ArgumentError
    @test !Fuzz.allowed(e, bt)
    e, bt = caught(() -> PureMLIP.open_torch_archive(UInt8[1, 2, 3]))
    @test e isa ArgumentError
    @test Fuzz.allowed(e, bt)
end

# A tensor dimension is untrusted: a BININT operand of 2147483646 in a shape tuple must be
# rejected against the storage length, not allocated.
@testitem "oversized tensor dimension is rejected before allocation" setup = [Fuzz, ZipFixtures] begin
    a = PureMLIP.open_torch_archive(ZipFixtures.tiny_mace_zip())
    pkl = collect(PureMLIP.entry_data(a, "data.pkl"))
    storages = Pair{String, Vector{UInt8}}["0" => collect(PureMLIP.entry_data(a, "data/0")), "1" => collect(PureMLIP.entry_data(a, "data/1"))]
    huge = collect(reinterpret(UInt8, [htol(Int32(2147483646))]))
    # Every BININT operand, set to the huge value in turn, must load or fail cleanly.
    ops = [i for i in eachindex(pkl) if pkl[i] == UInt8('J') && i + 4 <= length(pkl)]
    @test !isempty(ops)
    nshape = Ref(0)
    for p in ops
        bad = copy(pkl)
        bad[(p + 1):(p + 4)] = huge
        zip = ZipFixtures.zip_with_pickle(bad, storages)
        r, _ = Fuzz.attempt(PureMLIP.load_mace_checkpoint, zip)
        @test r === :loaded || r === :clean_error
        try
            PureMLIP.load_mace_checkpoint(zip)
        catch e
            nshape[] += e isa ErrorException && occursin("its storage holds", e.msg)
        end
    end
    @test nshape[] > 0
    # The tensor reader itself, without a pickle.
    bytes = zeros(UInt8, 16)
    @test_throws "checkpoint: tensor w: dimension 1 of 2147483646 makes more than the 2 elements its storage holds" PureMLIP.Checkpoint.gather(bytes, Float64, 0, [2147483646], [1], "w")
    @test_throws "dimension 2 of 4611686018427387904" PureMLIP.Checkpoint.gather(bytes, Float64, 0, [2, 4611686018427387904], [1, 1], "w")
    @test isempty(PureMLIP.Checkpoint.gather(bytes, Float64, 0, [0, 2147483646], [1, 1], "w"))
end

# DEFLATE fixtures: archives whose `t/blob` entry holds a hand-written fixed-Huffman stream of
# zeros, which compresses about 1000:1.
@testmodule DeflateBomb begin
    using PureMLIP
    le(x::Integer) = collect(reinterpret(UInt8, [htol(x)]))

    function bits_lsb!(bits::Vector{Bool}, v::Integer, n::Integer)
        for i in 0:(n - 1)
            push!(bits, isodd(v >> i))
        end
        return bits
    end
    function bits_msb!(bits::Vector{Bool}, v::Integer, n::Integer)
        for i in (n - 1):-1:0
            push!(bits, isodd(v >> i))
        end
        return bits
    end

    # Raw DEFLATE data that inflates to `1 + 258 * nmatch` zero bytes (without the literal, a
    # match that reaches before the start of the output).
    function zero_run(nmatch::Integer; literal::Bool = true)
        bits = Bool[]
        bits_lsb!(bits, 1, 1)                      # final block
        bits_lsb!(bits, 1, 2)                      # fixed Huffman codes
        literal && bits_msb!(bits, 0x30, 8)        # literal 0
        for _ in 1:nmatch
            bits_msb!(bits, 0b11000101, 8)         # length 258
            bits_msb!(bits, 0, 5)                  # distance 1
        end
        bits_msb!(bits, 0, 7)                      # end of block
        out = zeros(UInt8, cld(length(bits), 8))
        for (i, b) in pairs(bits)
            out[cld(i, 8)] |= UInt8(b) << ((i - 1) % 8)
        end
        return out
    end

    # An archive holding a stored `t/data.pkl` and a DEFLATE entry `t/blob` with the given
    # compressed payload and declared (central directory) sizes.
    function archive(payload::Vector{UInt8}, declared_size::Integer, crc::UInt32 = 0x00000000)
        out = UInt8[]
        central = UInt8[]
        for (name, m, data, usz, c) in (("t/data.pkl", 0x0000, UInt8[0x80, 0x02, UInt8('N'), UInt8('.')], 4, PureMLIP.Checkpoint.crc32(UInt8[0x80, 0x02, UInt8('N'), UInt8('.')])), ("t/blob", 0x0008, payload, declared_size, crc))
            nm = collect(codeunits(name))
            off = length(out)
            append!(out, le(0x04034b50), le(0x000a), le(0x0000), le(m), le(0x0000), le(0x0021), le(c), le(UInt32(length(data))), le(UInt32(usz)), le(UInt16(length(nm))), le(0x0000), nm, data)
            append!(central, le(0x02014b50), le(0x002d), le(0x000a), le(0x0000), le(m), le(0x0000), le(0x0021), le(c), le(UInt32(length(data))), le(UInt32(usz)), le(UInt16(length(nm))), le(0x0000))
            append!(central, le(0x0000), le(0x0000), le(0x0000), le(0x00000000), le(UInt32(off)), nm)
        end
        cdoff = length(out)
        append!(out, central)
        append!(out, le(0x06054b50), le(0x0000), le(0x0000), le(0x0002), le(0x0002), le(UInt32(length(central))), le(UInt32(cdoff)), le(0x0000))
        return out
    end

    function read_blob(zip)
        a = PureMLIP.open_torch_archive(zip)
        return length(PureMLIP.entry_data(a, "blob"))
    end

    # Bytes allocated by a read that fails.
    function failed_read_bytes(zip)
        try
            read_blob(zip)
        catch
        end
        return @allocated try
            read_blob(zip)
        catch
        end
    end
end

@testitem "a DEFLATE entry inflates to its declared size" setup = [DeflateBomb] begin
    raw = DeflateBomb.zero_run(100)
    n = 1 + 258 * 100
    crc = PureMLIP.Checkpoint.crc32(zeros(UInt8, n))
    @test DeflateBomb.read_blob(DeflateBomb.archive(raw, n, crc)) == n
end

@testitem "a DEFLATE entry that inflates past its declared size is rejected" setup = [DeflateBomb] begin
    # 20000 matches inflate to 5,160,001 bytes from about 33 kB.
    raw = DeflateBomb.zero_run(20000)
    zip = DeflateBomb.archive(raw, 1000)
    @test_throws "checkpoint: entry t/blob inflates past its declared size of 1000 bytes" DeflateBomb.read_blob(zip)
    # One byte past the declared size is already an error.
    n = 1 + 258 * 100
    @test_throws "inflates past its declared size of $(n - 1) bytes" DeflateBomb.read_blob(DeflateBomb.archive(DeflateBomb.zero_run(100), n - 1))
    # The output stops growing at the declared size: nothing like the 5 MB it would reach.
    DeflateBomb.failed_read_bytes(zip)
    @test DeflateBomb.failed_read_bytes(zip) < 400_000
end

@testitem "a declared size beyond the DEFLATE ratio is rejected" setup = [DeflateBomb] begin
    raw = DeflateBomb.zero_run(100)
    csz = length(raw)
    @test_throws "checkpoint: entry t/blob declares $(1032 * csz + 1) bytes, more than DEFLATE can produce from its $csz compressed bytes" PureMLIP.open_torch_archive(DeflateBomb.archive(raw, 1032 * csz + 1))
    @test PureMLIP.open_torch_archive(DeflateBomb.archive(raw, 1032 * csz)) isa PureMLIP.TorchArchive
    @test_throws "more than DEFLATE can produce" PureMLIP.open_torch_archive(DeflateBomb.archive(raw, 4_000_000_000))
end

@testitem "irreps terms with an invalid multiplicity or degree are rejected" begin
    P = PureMLIP.Checkpoint.parse_irreps
    @test_throws "has multiplicity 0, expected at least 1" P("0x0e")
    @test_throws "has multiplicity -4, expected at least 1" P("128x0e + -4x1o")
    @test_throws "has degree -1, expected at least 0" P("8x-1e")
    @test_throws "has degree -2, expected at least 0" P("-2e")
    # A huge multiplicity parses; the model checks it against the stored weights.
    @test P("1000000000x0e") == [PureMLIP.IrrepMul(1_000_000_000, 0, 1)]
end

# Sizes derived from irreps multiplicities and degrees are checked against the stored weights
# or a named bound before the model allocates.
@testmodule HugeIrreps begin
    using PureMLIP

    # The exception a model build throws, and the bytes a repeat build allocates.
    function failed_build(ck)
        build(ck) = try
            PureMLIP.MACEModel(ck)
            nothing
        catch e
            e
        end
        build(ck)
        e = build(ck)
        bytes = @allocated build(ck)
        return e, bytes
    end

    fresh() = PureMLIP.load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint())
end

@testitem "a huge hidden multiplicity is rejected against the stored weights" setup = [HugeIrreps] begin
    ck = HugeIrreps.fresh()
    huge = 2^62
    hidden = ck.hyperparameters.interactions[1].hidden_irreps
    hidden[1] = PureMLIP.IrrepMul(huge, 0, 1)
    hidden[2] = PureMLIP.IrrepMul(huge, 1, -1)
    # The path shape is untrusted too; make it agree with the hidden irreps so only the stored weight can refuse.
    shape = only(ck.constants.linear["node_embedding.linear"].instructions).path_shape
    shape[2] = huge
    e, bytes = HugeIrreps.failed_build(ck)
    @test e isa ArgumentError
    @test occursin("node_embedding.linear.weight of length", e.msg)
    @test bytes < 1_000_000
end

@testitem "a huge readout multiplicity is rejected against the stored weights" setup = [HugeIrreps] begin
    ck = HugeIrreps.fresh()
    ck.hyperparameters.readouts[2].hidden_irreps[1] = PureMLIP.IrrepMul(2^40, 0, 1)
    e, bytes = HugeIrreps.failed_build(ck)
    @test e isa ArgumentError
    @test occursin("readouts.1.linear_1.weight of length", e.msg)
    @test bytes < 1_000_000
end

@testitem "a huge slot count or degree is rejected before allocating" setup = [HugeIrreps] begin
    ck = HugeIrreps.fresh()
    C = ck.hyperparameters.interactions[1].hidden_irreps[1].mul
    spec = ck.constants.linear["interactions.1.linear"]
    spec.irreps_in[1] = PureMLIP.IrrepMul(C * 10^9, spec.irreps_in[1].l, spec.irreps_in[1].parity)
    e, bytes = HugeIrreps.failed_build(ck)
    @test e isa ArgumentError
    @test occursin("interactions.1.linear slots per term", e.msg)
    @test bytes < 1_000_000

    ck = HugeIrreps.fresh()
    tp = ck.constants.tensor_product["interactions.0.conv_tp"]
    tp.irreps_in2[end] = PureMLIP.IrrepMul(1, 1_000_000, 1)
    e, bytes = HugeIrreps.failed_build(ck)
    @test e isa ArgumentError
    @test occursin("interactions.0.conv_tp degree 1000000, expected at most 4", e.msg)
    @test bytes < 1_000_000
end

@testitem "malformed DEFLATE data is rejected with a clear error" setup = [DeflateBomb] begin
    le = DeflateBomb.le
    # A final stored block whose length and its complement disagree.
    corrupt = UInt8[0x01, le(UInt16(1))..., le(UInt16(1))..., 0x00]
    @test_throws "checkpoint: entry t/blob has a corrupt DEFLATE stored block" DeflateBomb.read_blob(DeflateBomb.archive(corrupt, 1))
    # A final block of the reserved type 3.
    @test_throws "checkpoint: entry t/blob has a DEFLATE block of invalid type 3" DeflateBomb.read_blob(DeflateBomb.archive(UInt8[0x07], 1))
    # A match with nothing yet to copy from.
    @test_throws "checkpoint: entry t/blob has a DEFLATE match reaching before its output" DeflateBomb.read_blob(DeflateBomb.archive(DeflateBomb.zero_run(1; literal = false), 258))
end

@testitem "a stored or literal DEFLATE block past the declared size is rejected" setup = [DeflateBomb] begin
    le = DeflateBomb.le
    stored = UInt8[0x01, le(UInt16(5))..., le(~UInt16(5))..., 1, 2, 3, 4, 5]
    @test DeflateBomb.read_blob(DeflateBomb.archive(stored, 5, PureMLIP.Checkpoint.crc32(UInt8[1, 2, 3, 4, 5]))) == 5
    @test_throws "checkpoint: entry t/blob inflates past its declared size of 3 bytes" DeflateBomb.read_blob(DeflateBomb.archive(stored, 3))
    @test_throws "checkpoint: entry t/blob inflates past its declared size of 0 bytes" DeflateBomb.read_blob(DeflateBomb.archive(DeflateBomb.zero_run(0), 0))
end
