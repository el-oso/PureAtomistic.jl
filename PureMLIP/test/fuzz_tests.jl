# Mutated inputs must either load or fail with a reader-raised error: an ArgumentError, or
# an ErrorException whose message starts with "checkpoint:" or "unpickle:". Failures print
# the trial number and the mutation so they can be reproduced from the fixed seed.

@testmodule Fuzz begin
    using Random

    allowed(e) = e isa ArgumentError || (e isa ErrorException && occursin(r"^(checkpoint|unpickle):", e.msg))

    # Applies one random mutation to a copy of `good`; returns the copy and a description.
    # `lo:hi` is the byte range that flips and window overwrites target.
    function mutate(rng, good::Vector{UInt8}, kind::Integer, lo::Integer = 1, hi::Integer = length(good))
        bad = copy(good)
        if kind == 0
            n = rand(rng, 1:8)
            idx = [rand(rng, lo:hi) for _ in 1:n]
            for i in idx
                bad[i] ⊻= rand(rng, UInt8(1):UInt8(255))
            end
            return bad, "flip bytes at $idx"
        elseif kind == 1
            len = rand(rng, 0:(length(bad) - 1))
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
            allowed(e) ? :clean_error : e
        end
        return r, t
    end
end

@testitem "fuzzed MP-0b3 checkpoints fail cleanly" setup = [Fuzz, Checkpoints] begin
    using Random
    rng = Xoshiro(20261031)
    good = read(Checkpoints.fetch(:mp0b3))
    a = PureMLIP.open_torch_archive(good)
    pkl = a.entries[a.prefix * "data.pkl"]
    cd_start = Int(PureMLIP.Checkpoint.read_le(good, PureMLIP.Checkpoint.find_eocd(good) + 16, UInt32)) + 1
    total = Ref(0.0)
    for trial in 1:300
        kind = trial % 3
        bad, what = if kind == 0       # flips in data.pkl (when the trial is odd) or the central directory
            iseven(trial) ? Fuzz.mutate(rng, good, 0, cd_start, length(good)) :
                Fuzz.mutate(rng, good, 0, pkl.data_offset, pkl.data_offset + pkl.size - 1)
        elseif kind == 1
            Fuzz.mutate(rng, good, 1)
        else
            Fuzz.mutate(rng, good, 2, pkl.data_offset, pkl.data_offset + pkl.size - 1)
        end
        r, t = Fuzz.attempt(PureMLIP.load_mace_checkpoint, bad)
        total[] += t
        ok = r === :loaded || r === :clean_error
        ok || @info "fuzz failure" trial what r
        @test ok
        @test t < 5.0
    end
    @info "MP-0b3 fuzz" total_seconds = total[]
end

@testitem "fuzzed pickle sections of a small checkpoint fail cleanly" setup = [Fuzz, ZipFixtures] begin
    using Random
    rng = Xoshiro(20261101)
    a = PureMLIP.open_torch_archive(ZipFixtures.tiny_mace_zip())
    pkl = collect(PureMLIP.entry_data(a, "data.pkl"))
    storages = Pair{String, Vector{UInt8}}["0" => collect(PureMLIP.entry_data(a, "data/0")), "1" => collect(PureMLIP.entry_data(a, "data/1"))]
    @test PureMLIP.load_mace_checkpoint(ZipFixtures.zip_with_pickle(pkl, storages)) isa PureMLIP.MACECheckpoint
    for trial in 1:2000
        bad, what = trial % 2 == 0 ? Fuzz.mutate_operand(rng, pkl) : Fuzz.mutate(rng, pkl, trial % 3)
        r, t = Fuzz.attempt(PureMLIP.load_mace_checkpoint, ZipFixtures.zip_with_pickle(bad, storages))
        ok = r === :loaded || r === :clean_error
        ok || @info "fuzz failure" trial what r
        @test ok
        @test t < 5.0
    end
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
        UInt8('h'), 0x00, UInt8('('), UInt8('h'), 0x00, UInt8('t'), UInt8('Q'), 0x86, 0x87, UInt8('.'),
    ]
    for trial in 1:1000
        bad, what = if trial % 4 == 0   # extra: insert a random opcode byte
            b = copy(good)
            insert!(b, rand(rng, 3:length(b)), rand(rng, UInt8))
            b, "insert a random byte"
        else
            Fuzz.mutate(rng, good, trial % 4 == 3 ? 1 : (trial % 4 == 1 ? 0 : 2))
        end
        r, t = Fuzz.attempt(PureMLIP.unpickle, bad)
        ok = r === :loaded || r === :clean_error
        ok || @info "fuzz failure" trial what r
        @test ok
        @test t < 1.0
    end
end

@testitem "fuzzed small checkpoint archives fail cleanly" setup = [Fuzz, ZipFixtures] begin
    using Random
    rng = Xoshiro(20261104)
    for trial in 1:5000
        good = ZipFixtures.tiny_mace_zip()
        bad, what = trial % 4 == 3 ? Fuzz.mutate_fill(rng, good) : Fuzz.mutate(rng, good, trial % 4)
        r, t = Fuzz.attempt(PureMLIP.load_mace_checkpoint, bad)
        ok = r === :loaded || r === :clean_error
        ok || @info "fuzz failure" trial what r
        @test ok
        @test t < 5.0
    end
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
    for trial in 1:1000
        bad, what = trial % 2 == 0 ? Fuzz.mutate_operand(rng, pkl) : Fuzz.mutate(rng, pkl, trial % 3)
        r, t = Fuzz.attempt(read_tree, bad)
        ok = r === :loaded || r === :clean_error
        ok || @info "fuzz failure" trial what r
        @test ok
        @test t < 5.0
    end
end
