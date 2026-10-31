@testitem "neighbor list matches brute force on the reference structures" setup = [LammpsReference, NeighborOracle] begin
    ref = LammpsReference.load(:mpa0_f64)
    for (name, count) in (("cu_vacancy", 2340), ("crconi_alloy", 2496), ("triclinic_ni3al", 176), ("multi_element", 44), ("isolated_cu", 0))
        s = LammpsReference.structure(ref, name)
        nl = neighbor_list(s.positions, s.cell, 6.0)
        got = Set((nl.senders[e], nl.receivers[e], Tuple(nl.shifts[:, e])) for e in 1:nl.n_edges)
        @test nl.n_edges == count && length(got) == count
        @test got == NeighborOracle.brute_force(s.positions, s.cell, 6.0)
        @test issorted(view(nl.receivers, 1:nl.n_edges))
    end
end

@testitem "neighbor list: unwrapped positions, refill, errors" setup = [LammpsReference] begin
    s = LammpsReference.structure(LammpsReference.load(:mpa0_f64), "triclinic_ni3al")
    vecs(nl, P) = sort([P[:, nl.receivers[e]] - P[:, nl.senders[e]] + s.cell * nl.shifts[:, e] for e in 1:nl.n_edges])
    P2 = copy(s.positions); P2[:, 3] .+= s.cell[:, 1] .- 2 .* s.cell[:, 2]
    # Rounding bound: |r| ≤ 6 Å built from coordinates ≤ 20 Å, 8 operations: 8·eps·20.
    @test all(isapprox.(vecs(neighbor_list(P2, s.cell, 6.0), P2), vecs(neighbor_list(s.positions, s.cell, 6.0), s.positions); atol = 160 * eps()))
    nl = neighbor_list(s.positions, 1.2 .* s.cell, 6.0)
    @test_throws "neighbor list: 176 edges exceed the capacity $(nl.n_edges)" neighbor_list!(nl, s.positions, s.cell)
    @test_throws "neighbor list: atoms 1 and 2 coincide" neighbor_list([1.0 1.0; 2.0 2.0; 3.0 3.0], 10.0 * [1.0 0 0; 0 1 0; 0 0 1], 6.0)
    @test_throws DimensionMismatch neighbor_list!(neighbor_list(s.positions, s.cell, 6.0), s.positions[:, 1:3], s.cell)
    using OffsetArrays
    @test_throws ArgumentError neighbor_list(OffsetArray(s.positions, 0, -1), s.cell, 6.0)
    v = neighbor_list(view(hcat(s.positions, zeros(3)), :, 1:4), s.cell, 6.0)
    @test v.n_edges == 176
end

@testitem "neighbor list: random triclinic cells, Float32, refill" setup = [NeighborOracle] begin
    using Random, LinearAlgebra
    rng = MersenneTwister(7)
    # Cells with heights from 0.3 to 2 times r_cut, so every image range from 1 to 4 occurs.
    for trial in 1:12
        cell = [3.0 + 4rand(rng) 1.5 * (rand(rng) - 0.5) 1.5 * (rand(rng) - 0.5); 0.0 3.0 + 4rand(rng) 1.5 * (rand(rng) - 0.5); 0.0 0.0 1.5 + 6rand(rng)]
        n = rand(rng, 1:6)
        pos = cell * rand(rng, 3, n)
        nl = neighbor_list(pos, cell, 5.0)
        got = Set((nl.senders[e], nl.receivers[e], Tuple(nl.shifts[:, e])) for e in 1:nl.n_edges)
        @test length(got) == nl.n_edges
        @test got == NeighborOracle.brute_force(pos, cell, 5.0)
        @test issorted(view(nl.receivers, 1:nl.n_edges))
        # Same list after a refill with the same inputs; `extra_capacity` leaves room for more edges.
        again = neighbor_list(pos, cell, 5.0; extra_capacity = 7)
        @test length(again.senders) == nl.n_edges + 7
        @test neighbor_list!(again, pos, cell) === again
        @test neighbor_list!(nl, pos, cell) === nl && nl.n_edges == length(nl.senders)
        @test again.n_edges == nl.n_edges && view(again.senders, 1:again.n_edges) == view(nl.senders, 1:nl.n_edges)
    end
    # Float32 coordinates two atoms 3 Å apart in a 7 Å cell.
    cell = Float32[7 0 0; 0 7 0; 0 0 7]; pos = Float32[0 3; 0 0; 0 0]
    nl = neighbor_list(pos, cell, 4)
    @test nl isa PureMLIP.Model.NeighborList{Float32}
    @test nl.n_edges == 2 && Set((nl.senders[e], nl.receivers[e]) for e in 1:2) == Set([(1, 2), (2, 1)])
end

@testitem "neighbor list: a grown cell reuses the buffers; further errors" setup = [LammpsReference, NeighborOracle] begin
    s = LammpsReference.structure(LammpsReference.load(:mpa0_f64), "cu_vacancy")
    nl = neighbor_list(s.positions, s.cell, 6.0)
    ref = (copy(nl.senders), copy(nl.receivers), copy(nl.shifts))
    # Tripling the lattice spacing leaves no pair within r_cut; the grid for the larger cell is coarser than
    # the one built at construction, and the buffers are reused.
    neighbor_list!(nl, 3 .* s.positions, 3 .* s.cell)
    @test nl.n_edges == 0
    neighbor_list!(nl, s.positions, s.cell)
    @test (nl.senders, nl.receivers, nl.shifts) == ref
    # Two atoms bin into one cell at construction; a 40 Å cell wants 6 bins per side, more than the
    # construction-time budget of bins, so the grid is coarsened.
    p = [0.0 1.5; 0.0 0.0; 0.0 0.0]; small = [6.0 0 0; 0 6 0; 0 0 6]; big = 40 .* [1.0 0 0; 0 1 0; 0 0 1]
    nl2 = neighbor_list(p, small, 6.0)
    neighbor_list!(nl2, p, big)
     Set((nl2.senders[e], nl2.receivers[e], Tuple(nl2.shifts[:, e])) for e in 1:nl2.n_edges) == NeighborOracle.brute_force(p, big, 6.0)
    bad = copy(s.positions); bad[2, 4] = NaN
    @test_throws "neighbor list: positions and cell must be finite" neighbor_list(bad, s.cell, 6.0)
    c = copy(s.cell); c[:, 3] = c[:, 1]
    @test_throws "neighbor list: the cell is singular" neighbor_list(s.positions, c, 6.0)
    @test_throws "neighbor list: the cell is singular" neighbor_list!(nl, s.positions, c)
    @test_throws "neighbor list: positions and cell must be finite" neighbor_list!(nl, s.positions, fill(Inf, 3, 3))
end

@testitem "neighbor list: the cutoff is exclusive" begin
    # Two atoms exactly 6 Å apart (exact in binary floating point) in a cell with no other image in range.
    P = [0.0 6.0; 0.0 0.0; 0.0 0.0]; C = [20.0 0.0 0.0; 0.0 20.0 0.0; 0.0 0.0 20.0]
    @test neighbor_list(P, C, 6.0).n_edges == 0
    @test neighbor_list(P, C, nextfloat(6.0)).n_edges == 2
end

@testitem "neighbor list: grids with several grid cells per direction" setup = [NeighborOracle] begin
    using Random
    rng = MersenneTwister(11)
    # Cells with heights from 2 to 5 times r_cut, so the grid has up to 5 cells per direction and
    # neighbors reach across the periodic boundary of the grid.
    for trial in 1:6
        cell = [10.0 + 8rand(rng) 4 * (rand(rng) - 0.5) 4 * (rand(rng) - 0.5); 0.0 10.0 + 8rand(rng) 4 * (rand(rng) - 0.5); 0.0 0.0 10.0 + 15rand(rng)]
        pos = cell * rand(rng, 3, 24)
        nl = neighbor_list(pos, cell, 4.5)
        got = Set((nl.senders[e], nl.receivers[e], Tuple(nl.shifts[:, e])) for e in 1:nl.n_edges)
        @test length(got) == nl.n_edges > 0
        @test got == NeighborOracle.brute_force(pos, cell, 4.5)
        @test issorted(view(nl.receivers, 1:nl.n_edges))
    end
end
