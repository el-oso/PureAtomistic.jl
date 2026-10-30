@testitem "unpickle scalars and containers" begin
    b = UInt8[
        0x80, 0x02, UInt8('}'), UInt8('('), UInt8('X'), 0x01, 0, 0, 0, UInt8('a'),
        UInt8('K'), 0x05, UInt8('X'), 0x01, 0, 0, 0, UInt8('b'),
        UInt8('G'), reinterpret(UInt8, [hton(6.0)])..., UInt8('u'), UInt8('.'),
    ]
    t = PureMLIP.unpickle(b)
    r = t.nodes[t.root]
    @test r.kind == PureMLIP.PK_DICT && length(r.items) == 4
    @test t.nodes[r.items[1]].str == "a"
    @test t.nodes[r.items[2]].int == 5
    @test t.nodes[r.items[4]].float == 6.0
end

@testitem "unpickle integers, bools, None, tuples, lists" begin
    le32(x) = collect(reinterpret(UInt8, [htol(Int32(x))]))
    b = UInt8[
        0x80, 0x02, UInt8(']'), UInt8('('), UInt8('J'), le32(-7)..., UInt8('M'), 0x34, 0x12,
        UInt8('N'), 0x88, 0x89, 0x85, UInt8('e'), UInt8('.'),
    ]
    t = PureMLIP.unpickle(b)
    r = t.nodes[t.root]
    @test r.kind == PureMLIP.PK_LIST && length(r.items) == 5
    @test t.nodes[r.items[1]].int == -7
    @test t.nodes[r.items[2]].int == 0x1234
    @test t.nodes[r.items[3]].kind == PureMLIP.PK_NONE
    @test t.nodes[r.items[4]].kind == PureMLIP.PK_BOOL && t.nodes[r.items[4]].int == 1
    last = t.nodes[r.items[5]]
    @test last.kind == PureMLIP.PK_TUPLE && length(last.items) == 1
    @test t.nodes[last.items[1]].kind == PureMLIP.PK_BOOL && t.nodes[last.items[1]].int == 0
end

@testitem "unpickle rejects unknown globals and opcodes" begin
    g = UInt8[0x80, 0x02, UInt8('c'), codeunits("os\nsystem\n")..., UInt8('.')]
    @test_throws "unpickle: global os.system is not allowed (byte 3)" PureMLIP.unpickle(g)
    o = UInt8[0x80, 0x02, 0x8a, 0x01, 0x05, UInt8('.')]   # LONG1, not in the allowed set
    @test_throws "unpickle: unsupported opcode 0x8a at byte 3" PureMLIP.unpickle(o)
    @test_throws "unpickle: unexpected end of data" PureMLIP.unpickle(UInt8[0x80, 0x02, UInt8('K')])
    @test_throws "unpickle: unexpected end of data" PureMLIP.unpickle(UInt8[0x80, 0x02, UInt8('N')])
end

@testitem "unpickle protocol, trailing bytes, structural errors" begin
    @test_throws "unpickle: unsupported pickle protocol 3" PureMLIP.unpickle(UInt8[0x80, 0x03, UInt8('N'), UInt8('.')])
    @test_throws "unpickle: trailing data after STOP at byte 5" PureMLIP.unpickle(UInt8[0x80, 0x02, UInt8('N'), UInt8('.'), 0x00])
    # SETITEM on a list, APPEND on a dict, BUILD on a non-call
    @test_throws "unpickle: SETITEM at byte 6 requires a dict target" PureMLIP.unpickle(UInt8[0x80, 0x02, UInt8(']'), UInt8('N'), UInt8('N'), UInt8('s'), UInt8('.')])
    @test_throws "unpickle: APPEND at byte 5 requires a list target" PureMLIP.unpickle(UInt8[0x80, 0x02, UInt8('}'), UInt8('N'), UInt8('a'), UInt8('.')])
    @test_throws "unpickle: BUILD at byte 5 requires a call target" PureMLIP.unpickle(UInt8[0x80, 0x02, UInt8('N'), UInt8('N'), UInt8('b'), UInt8('.')])
    @test_throws "unpickle: BINGET of unset memo index 3 at byte 3" PureMLIP.unpickle(UInt8[0x80, 0x02, UInt8('h'), 0x03, UInt8('.')])
end

@testitem "unpickle OrderedDict, BUILD, memo, persistent id" begin
    # c collections\nOrderedDict\n ) R q 0 ( X "k" K 7 u        -> OrderedDict filled with k=>7, memo[0]
    # ( h 0 t  Q                                                 -> persistent id wrapping a tuple of the dict
    # TUPLE2 .                                                   -> (dict, persid)
    b = UInt8[
        0x80, 0x02, UInt8('c'), codeunits("collections\nOrderedDict\n")..., UInt8(')'), UInt8('R'),
        UInt8('q'), 0x00, UInt8('('), UInt8('X'), 0x01, 0, 0, 0, UInt8('k'), UInt8('K'), 0x07, UInt8('u'),
        UInt8('('), UInt8('h'), 0x00, UInt8('t'), UInt8('Q'), 0x86, UInt8('.'),
    ]
    t = PureMLIP.unpickle(b)
    r = t.nodes[t.root]
    @test r.kind == PureMLIP.PK_TUPLE && length(r.items) == 2
    d = t.nodes[r.items[1]]
    @test d.kind == PureMLIP.PK_DICT && length(d.items) == 2
    @test t.nodes[d.items[1]].str == "k" && t.nodes[d.items[2]].int == 7
    p = t.nodes[r.items[2]]
    @test p.kind == PureMLIP.PK_PERSID
    pid = t.nodes[p.items[1]]
    @test pid.kind == PureMLIP.PK_TUPLE
    @test pid.items[1] == r.items[1]   # BINGET returns the node BINPUT stored
end

@testitem "unpickle REDUCE, NEWOBJ, BUILD, encode" begin
    # c torch.nn.functional\nsilu\n  (X "x" t R  -> PK_CALL(silu, ("x",)), then BUILD with state None
    glob = UInt8[UInt8('c'), codeunits("torch.nn.functional\nsilu\n")...]
    str1(s) = UInt8[UInt8('X'), UInt8(length(s)), 0, 0, 0, codeunits(s)...]
    b = UInt8[0x80, 0x02, glob..., str1("x")..., 0x85, UInt8('R'), UInt8('N'), UInt8('b'), UInt8('.')]
    t = PureMLIP.unpickle(b)
    r = t.nodes[t.root]
    @test r.kind == PureMLIP.PK_CALL && length(r.items) == 3
    @test t.nodes[r.items[1]].kind == PureMLIP.PK_GLOBAL && t.nodes[r.items[1]].str == "torch.nn.functional.silu"
    @test t.nodes[r.items[2]].kind == PureMLIP.PK_TUPLE
    @test t.nodes[r.items[3]].kind == PureMLIP.PK_NONE
    # A second BUILD on the same node is an error.
    b2 = UInt8[0x80, 0x02, glob..., UInt8(')'), 0x81, UInt8('N'), UInt8('b'), UInt8('N'), UInt8('b'), UInt8('.')]
    @test_throws "unpickle: second BUILD at byte 34" PureMLIP.unpickle(b2)
    b3 = UInt8[0x80, 0x02, glob..., UInt8(')'), 0x81, UInt8('.')]
    @test PureMLIP.unpickle(b3).nodes[PureMLIP.unpickle(b3).root].kind == PureMLIP.PK_CALL
    # _codecs.encode(str, "latin1") makes bytes, one per code point
    enc = UInt8[
        0x80, 0x02, UInt8('c'), codeunits("_codecs\nencode\n")...,
        UInt8('X'), 0x03, 0, 0, 0, 0x41, 0xc3, 0xbf, str1("latin1")..., 0x86, UInt8('R'), UInt8('.'),
    ]
    te = PureMLIP.unpickle(enc)
    n = te.nodes[te.root]
    @test n.kind == PureMLIP.PK_BYTES && codeunits(n.str) == UInt8[0x41, 0xff]
    # __builtin__.set stays a call
    s = UInt8[0x80, 0x02, UInt8('c'), codeunits("__builtin__\nset\n")..., UInt8(']'), 0x85, UInt8('R'), UInt8('.')]
    ts = PureMLIP.unpickle(s)
    @test ts.nodes[ts.root].kind == PureMLIP.PK_CALL
end

@testitem "unpickle accepts views" begin
    b = UInt8[0x80, 0x02, UInt8('N'), UInt8('.')]
    @test PureMLIP.unpickle(view(b, 1:4)).nodes[1].kind == PureMLIP.PK_NONE
end
