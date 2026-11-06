@testmodule IOHelpers begin
    using PureAtomistic.IO

    bits(a::AbstractArray{Float64}) = reinterpret(UInt64, vec(a))
    bits(a::AbstractArray{Float32}) = reinterpret(UInt32, vec(a))
    bits(x::Float64) = reinterpret(UInt64, x)
    bits(x::Float32) = reinterpret(UInt32, x)

    # Field-by-field equality on bit patterns (NaN == NaN, -0.0 != 0.0).
    function same(a::Frame, b::Frame)
        return a.species == b.species && size(a.positions) == size(b.positions) &&
            bits(a.positions) == bits(b.positions) && bits(a.forces) == bits(b.forces) &&
            bits(a.velocities) == bits(b.velocities) && bits(a.cell) == bits(b.cell) &&
            bits(a.stress) == bits(b.stress) && bits(a.virial) == bits(b.virial) &&
            a.pbc == b.pbc && bits(a.energy) == bits(b.energy) && bits(a.time) == bits(b.time) &&
            a.extras == b.extras
    end

    # Finite Float64 from random bits, so exponents span the whole range including subnormals.
    function randfloat(rng)
        while true
            x = reinterpret(Float64, rand(rng, UInt64))
            isfinite(x) && return x
        end
    end

    function randframe(rng, n; forces = true, velocities = true, cell = true)
        r(m, k) = [randfloat(rng) for _ in 1:m, _ in 1:k]
        return Frame{Float64}(
            ["Cu", "Zr", "H", "Xx9"][rand(rng, 1:4, n)], r(3, n);
            forces = forces ? r(3, n) : zeros(3, 0), velocities = velocities ? r(3, n) : zeros(3, 0),
            cell = cell ? r(3, 3) : zeros(0, 0), stress = cell ? r(3, 3) : zeros(0, 0),
            virial = cell ? r(3, 3) : zeros(0, 0), pbc = Tuple(rand(rng, Bool, 3)),
            energy = randfloat(rng), time = randfloat(rng),
            extras = ["config_type" => "bulk fcc", "tag" => "x1", "empty" => ""]
        )
    end

    function extxyz_string(frames)
        io = IOBuffer()
        for f in frames
            write_extxyz(io, f)
        end
        return String(take!(io))
    end
end

@testitem "extxyz: bitwise round trip" tags = [:io] setup = [IOHelpers] begin
    using PureAtomistic.IO, Random
    rng = MersenneTwister(7)
    frames = [
        IOHelpers.randframe(rng, 5),
        IOHelpers.randframe(rng, 1; forces = false),
        IOHelpers.randframe(rng, 4; velocities = false, cell = false),
        IOHelpers.randframe(rng, 0),
        Frame(["Cu"], [-0.0; 5.0e-324; 1.7976931348623157e308;;]),
    ]
    back = read_extxyz(IOBuffer(IOHelpers.extxyz_string(frames)))
    @test length(back) == length(frames)
    for (a, b) in zip(frames, back)
        @test IOHelpers.same(a, b)
    end
    # Absent quantities stay absent; a missing cell means non-periodic.
    @test isempty(back[3].cell) && back[3].pbc == (false, false, false) || back[3].pbc == frames[3].pbc
    @test isnan(back[5].energy) && isnan(back[5].time) && isempty(back[5].forces)

    # A Float32 frame is written as its Float64 widening and read back exactly.
    f32 = Frame{Float32}(["Cu", "Cu"], Float32[0.1 0.2; 0.3 0.4; 0.5 0.6]; energy = 0.1f0, cell = Float32[1 0 0; 0 2 0; 0 0 3])
    b32 = only(read_extxyz(IOBuffer(IOHelpers.extxyz_string([f32]))))
    @test IOHelpers.bits(b32.positions) == IOHelpers.bits(Float64.(f32.positions))
    @test b32.energy === Float64(0.1f0)
    @test b32.cell == Float64.(f32.cell)

    # File path method.
    path = tempname()
    write(path, IOHelpers.extxyz_string(frames[1:1]))
    @test IOHelpers.same(only(read_extxyz(path)), frames[1])
    rm(path)

    # CRLF line endings and trailing blank lines parse.
    crlf = replace(IOHelpers.extxyz_string(frames[1:1]), "\n" => "\r\n") * "\r\n\r\n"
    @test IOHelpers.same(only(read_extxyz(IOBuffer(crlf))), frames[1])
end

@testitem "extxyz: hand-written fixtures parse" tags = [:io] begin
    using PureAtomistic.IO
    dir = joinpath(@__DIR__, "fixtures", "extxyz")

    w = only(read_extxyz(joinpath(dir, "water.extxyz")))
    @test w.species == ["O", "H", "H"]
    @test w.positions == [0.0 0.0 0.0; 0.0 0.763239 -0.763239; 0.119262 -0.477047 -0.477047]
    @test isempty(w.cell) && w.pbc == (false, false, false) && isnan(w.energy)
    @test w.extras == ["comment" => "water molecule", "converged" => "T"]

    cu = read_extxyz(joinpath(dir, "cu_forces.extxyz"))
    @test length(cu) == 2
    @test cu[1].cell == 3.61 * [1.0 0 0; 0 1 0; 0 0 1]
    @test cu[1].energy == -14.2344
    @test cu[1].stress == [0.01 0.002 0.0; 0.002 0.01 0.0; 0.0 0.0 0.012]
    @test cu[1].pbc == (true, true, true)
    @test cu[1].forces[:, 1] == [0.001, -0.002, 0.0005]
    @test cu[1].positions[:, 2] == [1.805, 1.805, 0.0]
    @test cu[1].extras == ["config_type" => "bulk fcc"]
    # Blanks around '=' and the default pbc for a periodic frame.
    @test cu[2].energy == -14.21 && cu[2].cell[1, 1] == 3.62 && cu[2].pbc == (true, true, true)
    @test isempty(cu[2].stress) && isempty(cu[2].extras)

    md = only(read_extxyz(joinpath(dir, "md_snapshot.extxyz")))
    # Lattice numbers fill the cell by columns: the second vector is (1, 6, 0).
    @test md.cell == [5.0 1.0 0.5; 0.0 6.0 0.25; 0.0 0.0 7.0]
    @test md.time == 12.5 && md.energy == -0.35 && md.pbc == (true, false, true)
    @test md.virial == [1.0 0 0; 0 2 0; 0 0 3]
    @test md.velocities == [1.0e-3 -1.0e-3; -2.0e-3 2.0e-3; 3.0e-3 -3.0e-3]
    @test isempty(md.forces)
end

@testitem "extxyz: malformed input fails with a named line" tags = [:io] begin
    using PureAtomistic.IO
    rd(s) = read_extxyz(IOBuffer(s))
    good_atoms = "Cu 0 0 0\n"
    @test_throws "extxyz: input holds no frames" rd("")
    @test_throws "extxyz: input holds no frames" rd("\n\n")
    @test_throws "extxyz: line 1: expected an atom count, got \"two\"" rd("two\n\n")
    @test_throws "extxyz: line 1: expected an atom count, got \"-1\"" rd("-1\n\n")
    @test_throws "extxyz: line 1: frame of 2 atoms is truncated" rd("2\nProperties=species:S:1:pos:R:3\n" * good_atoms)
    @test_throws "extxyz: line 2: '=' without a key" rd("1\n=3\n" * good_atoms)
    @test_throws "extxyz: line 2: key 'a' has no value" rd("1\na=\n" * good_atoms)
    @test_throws "extxyz: line 2: unterminated quote in the value of 'a'" rd("1\na=\"x\n" * good_atoms)
    @test_throws "extxyz: line 2: cannot parse \"abc\" as a number in energy" rd("1\nenergy=abc\n" * good_atoms)
    @test_throws "extxyz: line 2: 'energy' is NaN" rd("1\nenergy=NaN\n" * good_atoms)
    @test_throws "extxyz: line 2: 'Time' is NaN" rd("1\nTime=NaN\n" * good_atoms)
    @test_throws "extxyz: line 2: key 'energy' appears twice" rd("1\nenergy=1 energy=2\n" * good_atoms)
    @test_throws "extxyz: line 2: 'Lattice' needs 9 numbers, got 3" rd("1\nLattice=\"1 2 3\"\n" * good_atoms)
    @test_throws "extxyz: line 2: cannot parse \"x\" as a number in Lattice" rd("1\nLattice=\"1 2 3 4 5 6 7 8 x\"\n" * good_atoms)
    @test_throws "extxyz: line 2: 'pbc' needs 3 booleans, got 2" rd("1\npbc=\"T F\"\n" * good_atoms)
    @test_throws "extxyz: line 2: cannot parse \"maybe\" as a boolean in 'pbc'" rd("1\npbc=\"T F maybe\"\n" * good_atoms)
    @test_throws "extxyz: line 2: Properties must be name:type:count triplets" rd("1\nProperties=species:S\n" * good_atoms)
    @test_throws "extxyz: line 2: Properties must be name:type:count triplets" rd("1\nProperties=\"\"\n" * good_atoms)
    @test_throws "extxyz: line 2: bad column count \"x\" for 'pos'" rd("1\nProperties=species:S:1:pos:R:x\n" * good_atoms)
    @test_throws "extxyz: line 2: unsupported property column 'Z'" rd("1\nProperties=species:S:1:pos:R:3:Z:I:1\n" * good_atoms)
    @test_throws "extxyz: line 2: property 'pos' must be R:3, got R:2" rd("1\nProperties=species:S:1:pos:R:2\n" * good_atoms)
    @test_throws "extxyz: line 2: property 'pos' is listed twice" rd("1\nProperties=species:S:1:pos:R:3:pos:R:3\n" * good_atoms)
    @test_throws "extxyz: line 2: Properties must include species and pos" rd("1\nProperties=pos:R:3\n" * good_atoms)
    @test_throws "extxyz: line 3: expected 4 columns, got 3" rd("1\nProperties=species:S:1:pos:R:3\nCu 0 0\n")
    @test_throws "extxyz: line 3: cannot parse \"y\" as a number in pos" rd("1\nProperties=species:S:1:pos:R:3\nCu 0 y 0\n")
    @test_throws "extxyz: line 3: cannot parse \"y\" as a number in forces" rd("1\nProperties=species:S:1:pos:R:3:forces:R:3\nCu 0 0 0 y 0 0\n")
    @test_throws "extxyz: line 3: cannot parse \"y\" as a number in velocities" rd("1\nProperties=species:S:1:pos:R:3:velocities:R:3\nCu 0 0 0 y 0 0\n")
    # A fault in the second frame names its own line.
    @test_throws "extxyz: line 4: expected an atom count" rd("1\n\n" * good_atoms * "junk\n\n" * good_atoms)

    # Mutating a valid file at random never raises anything but the named error.
    using Random
    rng = MersenneTwister(11)
    dir = joinpath(@__DIR__, "fixtures", "extxyz")
    seeds = [read(joinpath(dir, f), String) for f in ("water.extxyz", "cu_forces.extxyz", "md_snapshot.extxyz")]
    alphabet = collect("=\" :0123456789.-eTFx\n\t[],")
    nfail = Ref(0)
    for _ in 1:3000
        s = collect(rand(rng, seeds))
        for _ in 1:rand(rng, 1:4)
            i = rand(rng, 1:length(s))
            r = rand(rng, 1:3)
            r == 1 ? deleteat!(s, i) : r == 2 ? insert!(s, i, rand(rng, alphabet)) : (s[i] = rand(rng, alphabet))
            isempty(s) && push!(s, ' ')
        end
        try
            read_extxyz(IOBuffer(String(s)))
        catch err
            err isa ArgumentError && startswith(err.msg isa String ? err.msg : string(err.msg), "extxyz:") || rethrow()
            nfail[] += 1
        end
    end
    @test 0 < nfail[] < 3000
end

@testitem "extxyz: writer and Frame reject unusable input" tags = [:io] begin
    using PureAtomistic.IO
    wr(f) = write_extxyz(IOBuffer(), f)
    @test_throws "extxyz: species name must be non-empty without whitespace" wr(Frame(["C u"], zeros(3, 1)))
    @test_throws "extxyz: species name must be non-empty without whitespace" wr(Frame([""], zeros(3, 1)))
    @test_throws "extxyz: key must be non-empty without whitespace" wr(Frame(["Cu"], zeros(3, 1); extras = ["a b" => "x"]))
    @test_throws "extxyz: value of 'k' must not contain a quote or a line break" wr(Frame(["Cu"], zeros(3, 1); extras = ["k" => "a\"b"]))
    @test_throws "extxyz: value of 'k' must not contain a quote or a line break" wr(Frame(["Cu"], zeros(3, 1); extras = ["k" => "a\nb"]))
    @test_throws "pureatomistic: positions must be 3×2 for 2 species, got (3, 1)" Frame(["Cu", "Cu"], zeros(3, 1))
    @test_throws "pureatomistic: forces must be 3×1 or empty (3×0), got (3, 2)" Frame(["Cu"], zeros(3, 1); forces = zeros(3, 2))
    @test_throws "pureatomistic: velocities must be 3×1 or empty (3×0), got (2, 1)" Frame(["Cu"], zeros(3, 1); velocities = zeros(2, 1))
    @test_throws "pureatomistic: cell must be 3×3 or empty (0×0), got (3, 2)" Frame(["Cu"], zeros(3, 1); cell = zeros(3, 2))
    @test_throws "pureatomistic: stress must be 3×3 or empty (0×0), got (2, 2)" Frame(["Cu"], zeros(3, 1); stress = zeros(2, 2))
    @test_throws "pureatomistic: virial must be 3×3 or empty (0×0), got (1, 1)" Frame(["Cu"], zeros(3, 1); virial = zeros(1, 1))
end

@testitem "io: OffsetArray and view inputs match plain matrices" tags = [:io] setup = [IOHelpers] begin
    using PureAtomistic.IO, OffsetArrays, Random
    rng = MersenneTwister(3)
    n = 4
    X = randn(rng, 3, n); V = randn(rng, 3, n); C = randn(rng, 3, 3)
    Xo = OffsetArray(X, 0:2, 0:(n - 1)); Vo = OffsetArray(V, -1:1, 5:(4 + n)); Co = OffsetArray(C, 0:2, 0:2)
    big = randn(rng, 6, 2n); big[1:3, 1:n] .= X; big[4:6, (n + 1):2n] .= V; Xv = view(big, 1:3, 1:n); Vv = view(big, 4:6, (n + 1):2n)
    sp = fill("Cu", n)

    f_plain = Frame(sp, X; velocities = V, cell = C)
    f_off = Frame(sp, Xo; velocities = Vo, cell = Co)
    @test IOHelpers.same(f_plain, f_off)
    f_view = Frame(sp, Xv; velocities = Vv, cell = C)
    @test IOHelpers.same(f_view, Frame(sp, big[1:3, 1:n]; velocities = big[4:6, (n + 1):2n], cell = C))

    dir = mktempdir()
    function write_all(path, x, v, c)
        w = TrajectoryWriter(path, n, sp)
        write_frame(w, 2.5, x; velocities = v, cell = c, energy = -1.0)
        close(w)
        return read(path)
    end
    ref = write_all(joinpath(dir, "a"), X, V, C)
    @test write_all(joinpath(dir, "b"), Xo, Vo, Co) == ref
    @test write_all(joinpath(dir, "c"), Xv, Vv, C) == ref
    rm(dir; recursive = true)
end

@testitem "trajectory: round trip, layout and memory mapping" tags = [:io] setup = [IOHelpers] begin
    using PureAtomistic.IO, Random, Mmap
    rng = MersenneTwister(5)
    n = 7
    sp = ["Cu", "Zr", "H", "Xx9", "O", "Cu", "Cu"]
    path = tempname()
    frames = [
        Frame{Float64}(
            sp, Float64.(randn(rng, Float32, 3, n));
            velocities = Float64.(randn(rng, Float32, 3, n)), cell = randn(rng, 3, 3),
            energy = randn(rng), time = 0.5k
        ) for k in 1:6
    ]
    w = TrajectoryWriter(path, n, sp)
    # 24 header bytes + 4n symbol bytes padded to 8, then frames of 16 + 72 + 24n bytes padded to 8.
    header = 24 + 32
    frame_bytes = 16 + 72 + 24n
    foreach(f -> write_frame(w, f), frames)
    flush(w)
    @test filesize(path) == header + 6 * frame_bytes
    close(w)
    @test filesize(path) == header + 6 * frame_bytes

    r = TrajectoryReader(path)
    @test length(r) == 6
    @test r.species == sp
    for k in (6, 1, 3)   # random access in any order
        @test IOHelpers.same(read_frame(r, k), frames[k])
    end
    # The Float32 reading is the same values, narrowed back.
    g = read_frame(r, 2, Float32)
    @test g isa Frame{Float32} && IOHelpers.bits(g.positions) == IOHelpers.bits(Float32.(frames[2].positions))
    @test g.pbc == (true, true, true)

    # Memory mapping: the positions of frame 4 are readable in place at their offset.
    off = header + 3 * frame_bytes + 16 + 72
    io = open(path)
    M = Mmap.mmap(io, Matrix{Float32}, (3, n), off)
    @test IOHelpers.bits(M) == IOHelpers.bits(Float32.(frames[4].positions))
    finalize(M)
    close(io)
    close(r)

    # Optional blocks left out: no velocities, no cell.
    path2 = tempname()
    w = TrajectoryWriter(path2, n, sp; velocities = false, cell = false)
    write_frame(w, 1.0, frames[1].positions)
    write_frame(w, 2.0, frames[2].positions; energy = 3.0)
    close(w)
    @test filesize(path2) == 24 + 32 + 2 * (16 + 12n + 4)
    r = TrajectoryReader(path2)
    f = read_frame(r, 2)
    @test isempty(f.velocities) && isempty(f.cell) && f.pbc == (false, false, false)
    @test f.energy == 3.0 && f.time == 2.0 && isnan(read_frame(r, 1).energy)
    close(r)
    rm(path); rm(path2)

    # Zero atoms.
    path3 = tempname()
    w = TrajectoryWriter(path3, 0, String[])
    write_frame(w, 0.0, zeros(3, 0); velocities = zeros(3, 0), cell = [1.0 0 0; 0 1 0; 0 0 1])
    close(w)
    r = TrajectoryReader(path3)
    @test length(r) == 1 && size(read_frame(r, 1).positions) == (3, 0)
    close(r)
    rm(path3)
end

@testitem "trajectory: malformed files and misuse fail with a message" tags = [:io] begin
    using PureAtomistic.IO
    dir = mktempdir()
    p(name) = joinpath(dir, name)
    sp = ["Cu", "Cu"]
    good = p("good")
    w = TrajectoryWriter(good, 2, sp)
    write_frame(w, 1.0, zeros(3, 2); velocities = zeros(3, 2), cell = zeros(3, 3))
    close(w)
    bytes = read(good)

    put(name, b) = (write(p(name), b); p(name))
    @test_throws "pureatomistic: \"$(p("short"))\" is too short for a trajectory header" TrajectoryReader(put("short", bytes[1:10]))
    bad = copy(bytes); bad[1] = 0x00
    @test_throws "(bad magic)" TrajectoryReader(put("magic", bad))
    bad = copy(bytes); bad[9] = 0x02
    @test_throws "pureatomistic: unsupported trajectory version 2" TrajectoryReader(put("version", bad))
    bad = copy(bytes); bad[13] = 0x08
    @test_throws "pureatomistic: unknown trajectory flags 0x00000008" TrajectoryReader(put("flags", bad))
    bad = copy(bytes); bad[24] = 0xff
    @test_throws "pureatomistic: implausible atom count" TrajectoryReader(put("count", bad))
    @test_throws "has a truncated header" TrajectoryReader(put("header", bytes[1:30]))
    @test_throws "ends in a partial frame (10 extra bytes)" TrajectoryReader(put("partial", bytes[1:(32 + 10)]))
    @test_throws "ends in a partial frame" TrajectoryReader(put("partial2", bytes[1:(end - 1)]))

    r = TrajectoryReader(good)
    @test_throws "pureatomistic: frame 2 out of range 1:1" read_frame(r, 2)
    @test_throws "pureatomistic: frame 0 out of range 1:1" read_frame(r, 0)
    close(r)

    @test_throws "pureatomistic: 1 species for 2 atoms" TrajectoryWriter(p("w"), 2, ["Cu"])
    @test_throws "pureatomistic: n_atoms must be non-negative, got -1" TrajectoryWriter(p("w"), -1, String[])
    @test_throws "must be 1 to 4 printable ASCII characters" TrajectoryWriter(p("w"), 1, ["Cu12x"])
    @test_throws "must be 1 to 4 printable ASCII characters" TrajectoryWriter(p("w"), 1, [""])
    @test_throws "must be 1 to 4 printable ASCII characters" TrajectoryWriter(p("w"), 1, ["Cü"])

    w = TrajectoryWriter(p("w"), 2, sp)
    x = zeros(3, 2); v = zeros(3, 2); c = zeros(3, 3)
    @test_throws "pureatomistic: positions must be 3×2, got (3, 1)" write_frame(w, 0.0, zeros(3, 1); velocities = v, cell = c)
    @test_throws "pureatomistic: this trajectory stores velocities; pass them" write_frame(w, 0.0, x; cell = c)
    @test_throws "pureatomistic: this trajectory stores the cell; pass it" write_frame(w, 0.0, x; velocities = v)
    @test_throws "pureatomistic: velocities must be 3×2, got (3, 1)" write_frame(w, 0.0, x; velocities = zeros(3, 1), cell = c)
    @test_throws "pureatomistic: cell must be 3×3, got (2, 3)" write_frame(w, 0.0, x; velocities = v, cell = zeros(2, 3))
    close(w)
    w = TrajectoryWriter(p("w2"), 2, sp; velocities = false, cell = false)
    @test_throws "pureatomistic: this trajectory stores no velocities; do not pass them" write_frame(w, 0.0, x; velocities = v)
    @test_throws "pureatomistic: this trajectory stores no cell; do not pass it" write_frame(w, 0.0, x; cell = c)
    close(w)
    rm(dir; recursive = true)
end

# A Float32 velocity-Verlet integration of independent anharmonic oscillators. Used only to
# produce a deterministic trajectory to restart from.
@testmodule Oscillators begin
    using PureAtomistic.IO

    force!(f, x) = (@. f = -2.0f0 * x - 0.3f0 * x^3; f)

    function advance!(x, v, steps; writer = nothing, t0 = 0.0, dt = 0.5f0, cell = Float32[1 0 0; 0 1 0; 0 0 1])
        f = similar(x)
        force!(f, x)
        for s in 1:steps
            @. v += 0.5f0 * dt * f
            @. x += dt * v
            force!(f, x)
            @. v += 0.5f0 * dt * f
            writer === nothing || write_frame(writer, t0 + s * Float64(dt), x; velocities = v, cell)
        end
        return x, v
    end
end

@testitem "trajectory: restart from a frame continues bit for bit" tags = [:io] setup = [IOHelpers, Oscillators] begin
    using PureAtomistic.IO, Random
    rng = MersenneTwister(21)
    n = 9
    sp = fill("Ar", n)
    x0 = randn(rng, Float32, 3, n); v0 = randn(rng, Float32, 3, n)
    cell = Float32[10 0 0; 0 10 0; 0 0 10]
    dir = mktempdir()

    # Uninterrupted run of 60 steps, frame 1 = initial state.
    pa = joinpath(dir, "a.traj")
    w = TrajectoryWriter(pa, n, sp)
    write_frame(w, 0.0, x0; velocities = v0, cell)
    xa, va = Oscillators.advance!(copy(x0), copy(v0), 60; writer = w, cell)
    close(w)

    # Restart from frame 21 (after 20 steps), run the remaining 40 and write them too.
    r = TrajectoryReader(pa)
    @test length(r) == 61
    fr = read_frame(r, 21, Float32)
    @test fr.time == 20 * 0.5
    pb = joinpath(dir, "b.traj")
    w = TrajectoryWriter(pb, n, sp)
    write_frame(w, fr.time, fr.positions; velocities = fr.velocities, cell = fr.cell)
    xb, vb = Oscillators.advance!(copy(fr.positions), copy(fr.velocities), 40; writer = w, t0 = fr.time, cell = fr.cell)
    close(w); close(r)

    @test IOHelpers.bits(xb) == IOHelpers.bits(xa)
    @test IOHelpers.bits(vb) == IOHelpers.bits(va)
    # The restarted file's frames equal the original's frames 21 to 61 byte for byte.
    header, fb = 24 + 4n + (8 - 4n % 8) % 8, 16 + 72 + 24n + (8 - (16 + 72 + 24n) % 8) % 8
    A = read(pa); B = read(pb)
    @test A[(header + 20 * fb + 1):end] == B[(header + 1):end]

    # A Float64 state is stored as Float32: the stored frame is the rounded state, not the state.
    y = randn(rng, 3, n)
    pc = joinpath(dir, "c.traj")
    w = TrajectoryWriter(pc, n, sp; velocities = false, cell = false)
    write_frame(w, 0.0, y); close(w)
    r = TrajectoryReader(pc)
    @test read_frame(r, 1).positions == Float64.(Float32.(y))
    @test read_frame(r, 1).positions != y
    close(r)
    rm(dir; recursive = true)
end

@testitem "io: packing a frame is allocation-free" tags = [:io, :checks] begin
    using PureAtomistic.IO, StrictModeTest, StrictMode
    import PureAtomistic.IO: pack_frame!
    # The IOStream write itself takes Base's stream lock, which the static allocation analysis
    # cannot see through; `pack_frame!` holds everything else write_frame does.
    n = 8
    sp = fill("Cu", n)
    for T in (Float64, Float32)
        path = tempname()
        w = TrajectoryWriter(path, n, sp)
        x = rand(T, 3, n); v = rand(T, 3, n); c = rand(T, 3, 3)
        @noinline f(w, t, x, v, c) = pack_frame!(w, t, x; velocities = v, cell = c, energy = 1.0)
        f(w, 0.0, x, v, c)
        @test_noalloc f(w, 0.0, x, v, c)

        # Whole write_frame, measured: zero bytes after warm-up. The measurement sits inside a
        # compiled function so the caller's dispatch is not charged to the callee.
        @noinline function bytes(w, x, v, c)
            write_frame(w, 0.0, x; velocities = v, cell = c, energy = 1.0)
            return @allocated write_frame(w, 0.0, x; velocities = v, cell = c, energy = 1.0)
        end
        @test bytes(w, x, v, c) == 0
        close(w)
        rm(path)
    end
end

@testitem "io: TrimCheck validates the writer and the readers" tags = [:io, :checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureAtomistic.IO
            new_writer(p, n, s) = TrajectoryWriter(p, n, s)
            new_reader(p) = TrajectoryReader(p)
            put_frame(w, t, x, v, c) = write_frame(w, t, x; velocities = v, cell = c, energy = 1.0)
            new_frame64(s, x) = Frame{Float64}(s, x)
            new_frame32(s, x) = Frame{Float32}(s, x)
            import PureAtomistic
            pack(w, t, x, v, c) = PureAtomistic.IO.pack_frame!(w, t, x; velocities = v, cell = c, energy = 1.0)
        end,
        write_extxyz(IOStream, Frame{Float64}),
        write_extxyz(IOStream, Frame{Float32}),
        read_extxyz(IOStream),
        read_extxyz(String),
        new_writer(String, Int, Vector{String}),
        new_reader(String),
        put_frame(TrajectoryWriter, Float64, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}),
        put_frame(TrajectoryWriter, Float64, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}),
        pack(TrajectoryWriter, Float64, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}),
        pack(TrajectoryWriter, Float64, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}),
        write_frame(TrajectoryWriter, Frame{Float64}),
        read_frame(TrajectoryReader, Int, Type{Float64}),
        read_frame(TrajectoryReader, Int, Type{Float32}),
        new_frame64(Vector{String}, Matrix{Float64}),
        new_frame32(Vector{String}, Matrix{Float32}),
    )
end

@testitem "io: exported functions are type-stable" tags = [:io, :checks] begin
    using PureAtomistic.IO, StrictModeTest, StrictMode
    if StrictMode.checks_enabled()
        test_signatures(
            [
                (write_extxyz, (IOStream, Frame{Float64})),
                (write_extxyz, (IOStream, Frame{Float32})),
                (read_extxyz, (IOStream,)),
                (read_extxyz, (String,)),
                (write_frame, (TrajectoryWriter, Frame{Float64})),
                (read_frame, (TrajectoryReader, Int, Type{Float64})),
                (read_frame, (TrajectoryReader, Int, Type{Float32})),
            ]; guarantees = (:typestable,)
        )
    else
        @test_skip StrictMode.checks_enabled()
    end
end

@testitem "trajectory: a big-endian host is refused" tags = [:io] begin
    import PureAtomistic.IO: _check_little_endian
    @test _check_little_endian() === nothing
    @test _check_little_endian(0x04030201) === nothing
    @test_throws "pureatomistic: the trajectory format is little endian and this host is not" _check_little_endian(0x01020304)
end
