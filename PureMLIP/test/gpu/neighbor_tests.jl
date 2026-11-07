# The neighbor list built on the device, `device_neighbor_list(positions, cell, r_cut)` and
# `neighbor_list!` with device types, against the CPU's `neighbor_list` in the same precision.

@testmodule DeviceLists begin
    using PureMLIP, LinearAlgebra, Random
    include(pkgdir(PureMLIP, "test", "reference", "structures.jl"))

    # The edges of a list as a sorted vector of (receiver, sender, shift).
    function edge_set(receivers, senders, shifts, E)
        return sort([(Int(receivers[e]), Int(senders[e]), Int(shifts[1, e]), Int(shifts[2, e]), Int(shifts[3, e])) for e in 1:E])
    end

    # The length of edge (j, i, n) in Float64 from the positions and cell in `T`.
    edge_length(P, C, (j, i, n1, n2, n3)) = norm(Float64.(P[:, j]) - Float64.(P[:, i]) + Float64.(C) * [n1, n2, n3])

    # The CPU list of `cutoff`, the device list's edges and the edges that the two lists do not
    # share, each with its distance from the cutoff in units of eps(T)·cutoff.
    function compare(dnl, P::AbstractMatrix{T}, C, cutoff) where {T}
        nl = neighbor_list(P, C, cutoff)
        cpu = edge_set(nl.receivers, nl.senders, nl.shifts, nl.n_edges)
        E = Int(Array(dnl.n_edges)[1])
        dev = edge_set(Array(dnl.receivers), Array(dnl.senders), Array(dnl.shifts), E)
        odd = symdiff(cpu, dev)
        margins = [abs(edge_length(P, C, k) - cutoff) / (eps(T) * cutoff) for k in odd]
        return (; nl, cpu, dev, margins)
    end

    # Each list array of the device list equals that of the CPU list uploaded with the same capacity.
    function same_arrays(dnl, nl, backend)
        host = PureMLIP.device_neighbor_list(nl, backend)
        return all(f -> Array(getfield(dnl, f)) == Array(getfield(host, f)), (:row_ptr, :receivers, :senders, :shifts, :reverse, :n_edges))
    end

    # cu_bench(N) in `T` with its fractional coordinates placed in a triclinic cell.
    function triclinic_bench(::Type{T}, N) where {T}
        cu = cu_bench(N)
        L = cu.cell[1, 1]
        cell = [L 0.31L -0.22L; 0.0 0.93L 0.27L; 0.0 0.0 0.88L]
        frac = permutedims(cu.positions) ./ L
        return (; name = "triclinic cu_bench($N)", positions = T.(cell * frac), cell = T.(cell), cu.atomic_numbers)
    end

    # Two Cu atoms in a cell 2.6 Å thick along its third lattice vector, and one atom in a cell
    # thinner than the cutoff along every direction.
    thin_cells(::Type{T}) where {T} = [
        (; name = "two atoms, thin cell", positions = T[0.2 4.0; 0.1 3.9; 0.0 1.3], cell = T[8.0 0.5 0.4; 0.0 8.0 -0.3; 0.0 0.0 2.6], atomic_numbers = [29, 29]),
        (; name = "one atom, thin cell", positions = reshape(T[0.3, 0.2, 0.1], 3, 1), cell = T[2.6 0.4 0.0; 0.0 2.5 0.3; 0.1 0.0 2.7], atomic_numbers = [29]),
    ]

    # The structures whose lists are compared: the five LAMMPS references, cu_bench(5) (500 atoms, so
    # that a prefix sum covers more entries than work items), a triclinic cell and the thin cells.
    function list_structures(::Type{T}) where {T}
        five = [(; s.name, positions = T.(permutedims(s.positions)), cell = T.(permutedims(s.cell)), s.atomic_numbers) for s in reference_structures()]
        cu = cu_bench(5)
        bench = (; name = "cu_bench(5)", positions = T.(permutedims(cu.positions)), cell = T.(cu.cell), cu.atomic_numbers)
        return [five; bench; triclinic_bench(T, 3); thin_cells(T)]
    end

    # `P` with each atom moved by a lattice translation of up to ±7 cell vectors per direction.
    function unwrap(P::AbstractMatrix{T}, C, rng) where {T}
        k = rand(rng, -7:7, 3, size(P, 2))
        return T.(Float64.(P) + Float64.(C) * k)
    end

    # Prints a measured figure when PUREMLIP_GPU_VERBOSE=1.
    note(args...) = get(ENV, "PUREMLIP_GPU_VERBOSE", "") == "1" && println(args...)

    export edge_set, compare, same_arrays, list_structures, thin_cells, triclinic_bench, unwrap, note
end

# Oracle: CPU `neighbor_list` of r_cut + skin in the same precision. The device list must hold the
# same edges in the same order. Bound (b) of the tolerance policy allows the two lists to differ in
# an edge within 8·eps(T)·r_cut of the cutoff (c = 8: three squares, a sum and a square root); an
# edge set that differs nowhere must also give the same arrays, order and reverse map included.
# Measured on the CPU backend (Float32 and Float64) and on Metal (Float32): no differing edge, and
# every array equal.
@testitem "neighbor list on the device: edges equal the CPU list" tags = [:gpu] setup = [GPUBackend, DeviceLists] begin
    for T in precisions, s in list_structures(T), (r_cut, skin) in ((5.0, 0.0), (5.0, 1.0))
        P, C = s.positions, s.cell
        dnl = PureMLIP.device_neighbor_list(todevice(P), todevice(C), r_cut; skin, extra_capacity = 7)
        c = compare(dnl, P, C, r_cut + skin)
        isempty(c.margins) || note("edges, $(GPUBackend.NAME), $T, $(s.name): $(length(c.margins)) edges differ, within $(maximum(c.margins)) eps·cutoff")
        @test all(<=(8), c.margins)
        @test dnl.capacity == c.nl.n_edges + 7
        @test (dnl.r_cut, dnl.skin, dnl.cutoff) == (T(r_cut), T(skin), T(r_cut + skin))
        isempty(c.margins) && @test same_arrays(dnl, neighbor_list(P, C, r_cut + skin; extra_capacity = 7), backend)
        @test Array(dnl.status) == Int32[1, 0, 0, 0, 0]
        # The list rebuilt in place by `neighbor_list!` (skin 0: every call) is the same.
        dnl0 = PureMLIP.device_neighbor_list(todevice(P), todevice(C), r_cut + skin; extra_capacity = 7)
        @test neighbor_list!(dnl0, todevice(P), todevice(C)) === dnl0
        @test Array(dnl0.status)[1] == 1
        isempty(c.margins) && @test same_arrays(dnl0, neighbor_list(P, C, r_cut + skin; extra_capacity = 7), backend)
    end
end

# Review Focus 1: atoms far outside the cell (each moved by up to ±7 cell vectors per direction),
# in triclinic cells. The device list equals the CPU list of the moved positions, and the forces
# equal those of the wrapped configuration within bound (b), 10·eps(T)·1280 relative to the sum of
# the magnitudes of each atom's edge terms (the reference of "backward: forces, virial and stress
# agree with CPU mace_energy_forces!", from the CPU's ∂E/∂r of the wrapped positions), and the energy
# relative to itself. The moved positions differ from the wrapped ones by their rounding at seven
# cell lengths, which the edge vectors carry. Measured, largest over the three cells and over Julia
# 1.12.7 and 1.13.3, in eps(T): CPU backend Float32 energy 6.8, forces 216; Float64 energy 1.6,
# forces 266; Metal Float32 energy 7.5, forces 206.
@testitem "neighbor list on the device: atoms moved by lattice vectors give the forces of the wrapped atoms" tags = [:gpu] setup = [GPUBackend, DeviceLists, EdgeStages] begin
    using Random
    ck = mpa0()
    rng = Xoshiro(7)
    tol = 10 * 1280
    for T in precisions
        m = MACEModel{T}(ck)
        r_max = Float64(m.r_max)
        cases = [list_structures(T)[[2, 3]]; triclinic_bench(T, 2)]
        for s in cases
            P, C, Z = s.positions, s.cell, s.atomic_numbers
            n = size(P, 2)
            Q = unwrap(P, C, rng)
            @test maximum(abs, Q) > 5 * maximum(abs, C)
            nl = neighbor_list(P, C, r_max)
            res = PureMLIP.MACEResult{T}(n)
            cws = MACEWorkspace{T}(m, n, nl.n_edges; elements = unique(Z))
            mace_energy_forces!(res, cws, m, P, C, Z, nl)
            dQ, dC, dZ = todevice(Q), todevice(C), todevice(Int32.(Z))
            dnl = PureMLIP.device_neighbor_list(dQ, dC, r_max; extra_capacity = 10)
            c = compare(dnl, Q, C, r_max)
            @test all(<=(8), c.margins)
            isempty(c.margins) && @test same_arrays(dnl, neighbor_list(Q, C, r_max; extra_capacity = 10), backend)
            dm = PureMLIP.device_model(m, backend; elements = unique(Z))
            ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
            dres = PureMLIP.device_result(dm, n)
            mace_energy_forces!(dres, ws, dm, dQ, dC, dZ, dnl)
            scale = zeros(3, n)
            for e in 1:nl.n_edges, a in 1:3
                g = abs(Float64(cws.dEdr[a, e]))
                scale[a, nl.senders[e]] += g
                scale[a, nl.receivers[e]] += g
            end
            eerr = Float64(abs(Array(dres.energy)[1] - res.energy) / abs(res.energy) / eps(T))
            ferr = scalederr(Array(dres.forces), res.forces, scale)
            report("unwrapped, $(GPUBackend.NAME), $T, $(s.name): energy $eerr, forces $ferr eps")
            @test eerr <= tol
            @test ferr <= tol
        end
    end
end

# Review Focus 4: one atom alone (no edges), and cells thinner than the cutoff, where atoms see
# their own images. The edges equal the CPU's; the lone atom's energy is its E0 and its force zero.
@testitem "neighbor list on the device: one atom, and cells thinner than the cutoff" tags = [:gpu] setup = [GPUBackend, DeviceLists, EdgeStages] begin
    ck = mpa0()
    for T in precisions
        m = MACEModel{T}(ck)
        r_max = Float64(m.r_max)
        P, C = reshape(T[10, 10, 10], 3, 1), T[20 0 0; 0 20 0; 0 0 20]
        dnl = PureMLIP.device_neighbor_list(todevice(P), todevice(C), r_max)
        @test dnl.capacity == 0 && Array(dnl.n_edges) == [0] && Array(dnl.row_ptr) == [1, 1]
        @test neighbor_list!(dnl, todevice(P), todevice(C)) === dnl
        dm = PureMLIP.device_model(m, backend; elements = [29])
        ws = PureMLIP.device_workspace(dm, 1, dnl.capacity)
        res = PureMLIP.device_result(dm, 1)
        fill!(res.forces, one(T))
        mace_energy_forces!(res, ws, dm, todevice(P), todevice(C), todevice(Int32[29]), dnl)
        @test Array(res.energy) == [m.e0[m.species[30]]]
        @test all(iszero, Array(res.forces))
        for s in thin_cells(T)
            P, C = s.positions, s.cell
            dnl = PureMLIP.device_neighbor_list(todevice(P), todevice(C), r_max)
            c = compare(dnl, P, C, r_max)
            @test c.nl.n_edges > 5 * size(P, 2)
            @test isempty(c.margins)
            @test same_arrays(dnl, neighbor_list(P, C, r_max), backend)
            # The device list gives the CPU forces within bound (b), relative to the largest per-atom
            # sum of edge-term magnitudes (the forces cancel to zero by symmetry in the one-atom cell).
            n = size(P, 2)
            nl = c.nl
            cres = PureMLIP.MACEResult{T}(n)
            cws = MACEWorkspace{T}(m, n, nl.n_edges; elements = [29])
            mace_energy_forces!(cres, cws, m, P, C, s.atomic_numbers, nl)
            ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
            res = PureMLIP.device_result(dm, n)
            mace_energy_forces!(res, ws, dm, todevice(P), todevice(C), todevice(Int32.(s.atomic_numbers)), dnl)
            scale = maximum(sum(abs, Float64.(cws.dEdr[:, 1:nl.n_edges]); dims = 2))
            @test abs(Array(res.energy)[1] - cres.energy) <= 10 * 1280 * eps(T) * abs(cres.energy)
            @test maximum(abs, Array(res.forces) - cres.forces) <= 10 * 1280 * eps(T) * scale
        end
    end
end

# The skin criterion: an atom moved by less than half the skin leaves the list as it is, an atom
# moved by more than half rebuilds it, and so does a change of the cell. Between rebuilds the list
# still holds every edge within r_max, so with skin = 1 Å the forces equal those of a list without
# skin, rebuilt for the same positions, within bound (b) relative to the largest force (energy
# relative to itself). Measured, in eps(T): CPU backend Float32 energy 0.51, forces 2.4; Float64
# energy 0, forces 0.87; Metal Float32 energy 0, forces 0.61.
@testitem "neighbor list on the device: the skin criterion" tags = [:gpu] setup = [GPUBackend, DeviceLists, EdgeStages] begin
    using LinearAlgebra
    ck = mpa0()
    tol = 10 * 1280
    for T in precisions
        m = MACEModel{T}(ck)
        r_max = Float64(m.r_max)
        cu = DeviceLists.cu_bench(2)
        P, C = T.(permutedims(cu.positions)), T.(cu.cell)
        n = size(P, 2)
        dC, dZ = todevice(C), todevice(Int32.(cu.atomic_numbers))
        dnl = PureMLIP.device_neighbor_list(todevice(P), dC, r_max; skin = 1.0, extra_capacity = 200)
        rebuilt(Q, cell = dC) = (neighbor_list!(dnl, todevice(Q), cell); Array(dnl.status)[1])
        list() = map(Array, (dnl.row_ptr, dnl.senders, dnl.shifts, dnl.reverse))
        before = list()
        @test rebuilt(P) == 0
        Q = copy(P)
        Q[:, 5] .+= T[0.2, -0.2, 0.15]       # 0.32 Å, below half the skin
        @test rebuilt(Q) == 0
        @test list() == before
        Q[:, 5] .+= T[0.1, 0.0, 0.1]         # 0.44 Å from the build
        @test rebuilt(Q) == 0
        Q[:, 9] .+= T[0, 0.505, 0]           # past half the skin
        @test rebuilt(Q) == 1
        @test list() != before
        @test rebuilt(Q) == 0                 # the reference is the last build
        C2 = copy(C)
        C2[1, 2] = T(1.0e-3)
        @test rebuilt(Q, todevice(C2)) == 1
        @test rebuilt(Q, todevice(C2)) == 0
        @test rebuilt(Q) == 1

        # Forces with the skin list at positions that have moved since its build, against a list
        # without skin for those positions.
        dm = PureMLIP.device_model(m, backend; elements = [29])
        R = copy(Q)
        R[:, 1:2:end] .+= T(0.12)
        @test rebuilt(R) == 0
        ws = PureMLIP.device_workspace(dm, n, dnl.capacity; edge_chunk = 500)
        res = PureMLIP.device_result(dm, n)
        mace_energy_forces!(res, ws, dm, todevice(R), dC, dZ, dnl)
        nl0 = PureMLIP.device_neighbor_list(todevice(R), dC, r_max)
        @test nl0.capacity < Array(dnl.n_edges)[1]
        ws0 = PureMLIP.device_workspace(dm, n, nl0.capacity; edge_chunk = 500)
        res0 = PureMLIP.device_result(dm, n)
        mace_energy_forces!(res0, ws0, dm, todevice(R), dC, dZ, nl0)
        E, E0 = Array(res.energy)[1], Array(res0.energy)[1]
        eerr = Float64(abs(E - E0) / abs(E0) / eps(T))
        ferr = relerr(Array(res.forces), Array(res0.forces))
        report("skin, $(GPUBackend.NAME), $T: energy $eerr, forces $ferr eps")
        @test eerr <= tol
        @test ferr <= tol
    end
end

# An overflow of the capacity throws with the number of edges found, and leaves the list empty and
# flagged, so that a later rebuild and an evaluation with the list throw too. So do a singular
# cell, a position that is not finite, and two atoms at one place.
@testitem "neighbor list on the device: overflow and invalid geometry throw" tags = [:gpu] setup = [GPUBackend, DeviceLists, EdgeStages] begin
    T = first(precisions)
    m = MACEModel{T}(mpa0())
    cu = DeviceLists.cu_bench(2)
    P, C = T.(permutedims(cu.positions)), T.(cu.cell)
    n = size(P, 2)
    r_max = Float64(m.r_max)
    dZ = todevice(Int32.(cu.atomic_numbers))
    dnl = PureMLIP.device_neighbor_list(todevice(P), todevice(C), r_max; extra_capacity = 10)
    cap = dnl.capacity
    dm = PureMLIP.device_model(m, backend; elements = [29])
    ws = PureMLIP.device_workspace(dm, n, cap)
    res = PureMLIP.device_result(dm, n)
    # A compressed copy holds more edges than the capacity.
    s = T(0.9)
    sP, sC = todevice(s * P), todevice(s * C)
    more = neighbor_list(s * P, s * C, r_max).n_edges
    @test more > cap
    @test_throws "gpu: $more edges exceed the capacity $cap" neighbor_list!(dnl, sP, sC)
    @test Array(dnl.status)[1:2] == [1, more]
    @test Array(dnl.n_edges) == [0] && all(==(1), Array(dnl.row_ptr))
    @test all(iszero, Array(dnl.senders)) && all(iszero, Array(dnl.reverse)) && all(iszero, Array(dnl.shifts))
    @test_throws "gpu: $more edges exceed the capacity $cap" mace_energy!(res, ws, dm, sP, sC, dZ, dnl)
    @test_throws "gpu: $more edges exceed the capacity $cap" mace_energy_forces!(res, ws, dm, sP, sC, dZ, dnl)
    @test_throws "gpu: $more edges exceed the capacity $cap" neighbor_list!(dnl, todevice(P), todevice(C))

    geometry = "gpu: the neighbor list needs a finite cell at least cutoff/65536 thick and finite positions within 2^30 cell lengths"
    flat = copy(C)
    flat[:, 3] = flat[:, 1] + flat[:, 2]
    @test_throws geometry PureMLIP.device_neighbor_list(todevice(P), todevice(flat), r_max)
    bad = copy(P)
    bad[2, 3] = T(NaN)
    @test_throws geometry PureMLIP.device_neighbor_list(todevice(bad), todevice(C), r_max)
    bad[2, 3] = T(4.0e10)
    @test_throws geometry PureMLIP.device_neighbor_list(todevice(bad), todevice(C), r_max)
    dnl = PureMLIP.device_neighbor_list(todevice(P), todevice(C), r_max; skin = 0.5)
    bad[2, 3] = T(Inf)
    @test_throws geometry neighbor_list!(dnl, todevice(bad), todevice(C))
    ws = PureMLIP.device_workspace(dm, n, dnl.capacity)
    @test_throws geometry mace_energy!(res, ws, dm, todevice(P), todevice(C), dZ, dnl)
    thin = copy(C)
    thin[3, 3] = T(1.0e-5)
    @test_throws geometry PureMLIP.device_neighbor_list(todevice(P[:, 1:1]), todevice(thin), r_max)

    twin = copy(P)
    twin[:, 7] = twin[:, 4]
    @test_throws "gpu: atoms 4 and 7 coincide" PureMLIP.device_neighbor_list(todevice(twin), todevice(C), r_max)
    dnl = PureMLIP.device_neighbor_list(todevice(P), todevice(C), r_max)
    @test_throws "gpu: atoms 4 and 7 coincide" neighbor_list!(dnl, todevice(twin), todevice(C))
    # When the host finds no pair at one place in the list (its edge vectors round as the device's), the
    # message names none.
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    @test_throws "gpu: two atoms coincide" ext._throw_coincident(dnl, todevice(P), todevice(C))
end

@testitem "neighbor list on the device: inconsistent arguments are rejected before any launch" tags = [:gpu] setup = [GPUBackend, DeviceLists, FakeDevice] begin
    using OffsetArrays
    T = first(precisions)
    cu = DeviceLists.cu_bench(2)
    P, C = T.(permutedims(cu.positions)), T.(cu.cell)
    dP, dC = todevice(P), todevice(C)
    @test_throws "gpu: positions must hold floating-point numbers, got Int32" PureMLIP.device_neighbor_list(todevice(Int32.(round.(P))), todevice(Int32.(round.(C))), 5.0)
    @test_throws "gpu: positions and cell must share one element type, got $T and Int32" PureMLIP.device_neighbor_list(dP, todevice(Int32[1 0 0; 0 1 0; 0 0 1]), 5.0)
    @test_throws "gpu: positions have size (2, 32), expected 3 rows" PureMLIP.device_neighbor_list(todevice(P[1:2, :]), dC, 5.0)
    @test_throws "gpu: the cell has size (3, 2), expected (3, 3)" PureMLIP.device_neighbor_list(dP, todevice(C[:, 1:2]), 5.0)
    @test_throws "gpu: r_cut must be positive and finite, got 0.0" PureMLIP.device_neighbor_list(dP, dC, 0.0)
    @test_throws "gpu: r_cut must be positive and finite, got Inf" PureMLIP.device_neighbor_list(dP, dC, Inf)
    @test_throws "gpu: the skin must not be negative, got -0.5" PureMLIP.device_neighbor_list(dP, dC, 5.0; skin = -0.5)
    @test_throws "gpu: extra_capacity must not be negative, got -1" PureMLIP.device_neighbor_list(dP, dC, 5.0; extra_capacity = -1)
    @test_throws "gpu: all arrays must be on the backend of the positions" PureMLIP.device_neighbor_list(dP, OtherMatrix(C), 5.0)
    @test_throws "gpu: arrays must be 1-based" PureMLIP.device_neighbor_list(OffsetArray(dP, 0:2, 1:32), dC, 5.0)
    dnl = PureMLIP.device_neighbor_list(dP, dC, 5.0)
    @test_throws "gpu: positions have size (3, 31), the neighbor list has 32 atoms" neighbor_list!(dnl, dP[:, 1:31], dC)
    @test_throws "gpu: all arrays must be on the backend of the positions" neighbor_list!(dnl, dP, OtherMatrix(C))
    @test_throws "gpu: positions and cell must have element type $T" neighbor_list!(dnl, todevice(Float16.(P)), dC)
    # A view of the first columns of a wider device array is accepted and gives the same list.
    wide = todevice(hcat(P, zeros(T, 3, 3)))
    before = map(Array, (dnl.senders, dnl.shifts, dnl.reverse))
    neighbor_list!(dnl, view(wide, :, 1:32), dC)
    @test map(Array, (dnl.senders, dnl.shifts, dnl.reverse)) == before
end

# Host bytes allocated by one call of `neighbor_list!`, through a compiled function so that the
# dynamic dispatch of the call site is not counted: the fewest of five calls after two warm-up calls,
# on cu_bench(2) with a 6 Å cutoff, for a call that rebuilds (skin 0) and one that keeps the list
# (skin 1 Å, positions unchanged). Both launch the same build kernels, which skip their work on the
# device when nothing is rebuilt, and read the status word once; keeping adds the launch of the skin
# criterion. Measured:
#   CPU backend, Julia 1.12.7 and 1.13.3, Float32 and Float64: 0 bytes for both;
#   Metal, Julia 1.13.3: rebuild 10,048 bytes and keep 11,760, in each of 24 groups of five over
#     three processes.
# The gate is the measured figure on the CPU backend, 0 bytes (twice it on other platforms, where it
# has not been measured, is 0 too), and on Metal the measured figures rounded up to 10,100 and
# 11,800 bytes.
@testitem "neighbor list on the device: host bytes per call of neighbor_list!" tags = [:gpu] setup = [GPUBackend, DeviceLists] begin
    @noinline call_bytes(f::F, a, b, c) where {F} = @allocated f(a, b, c)
    function fewest_bytes(args)
        neighbor_list!(args...)
        neighbor_list!(args...)
        return minimum(call_bytes(neighbor_list!, args...) for _ in 1:5)
    end
    bounds = Dict("CPU" => (rebuild = 0, keep = 0), "Metal" => (rebuild = 10_100, keep = 11_800))
    cu = DeviceLists.cu_bench(2)
    for T in precisions
        P, C = todevice(T.(permutedims(cu.positions))), todevice(T.(cu.cell))
        rebuild = fewest_bytes((PureMLIP.device_neighbor_list(P, C, 6.0), P, C))
        keep_list = PureMLIP.device_neighbor_list(P, C, 6.0; skin = 1.0)
        keep = fewest_bytes((keep_list, P, C))
        @test Array(keep_list.status)[1] == 0
        note("host bytes, $(GPUBackend.NAME), $T, Julia $VERSION: neighbor_list! rebuild $rebuild, keep $keep")
        if haskey(bounds, GPUBackend.NAME)
            @test rebuild <= bounds[GPUBackend.NAME].rebuild
            @test keep <= bounds[GPUBackend.NAME].keep
        else
            # CUDA and AMDGPU are measured on their machines.
            @test_skip rebuild <= 0
        end
    end
end

# The entry points return concrete types, and the extension's own code in them dispatches nothing
# at run time (JET confined to the extension, as in "backward: device entry points are
# type-stable").
@testitem "neighbor list on the device: entry points are type-stable" tags = [:gpu] setup = [GPUBackend, DeviceLists] begin
    using StrictModeTest
    JET = StrictModeTest.JET
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    cu = DeviceLists.cu_bench(2)
    for T in precisions
        P, C = todevice(T.(permutedims(cu.positions))), todevice(T.(cu.cell))
        dnl = PureMLIP.device_neighbor_list(P, C, 6.0; skin = 0.5)
        neighbor_list!(dnl, P, C)
        build(P, C) = PureMLIP.device_neighbor_list(P, C, 6.0; skin = 0.5)
        for (f, tt, rt) in ((neighbor_list!, (typeof(dnl), typeof(P), typeof(C)), typeof(dnl)), (build, (typeof(P), typeof(C)), typeof(dnl)))
            @test Base.return_types(f, tt) == [rt]
            @test isempty(JET.get_reports(JET.report_opt(f, tt; target_modules = (ext,))))
        end
    end
end

# The kernels' arithmetic of the list as host code in Float32 (see "edge stages: Float32 stages do
# not widen to Float64"), and the extension's compiled functions after a Float32 build.
@testitem "neighbor list on the device: Float32 stages do not widen to Float64" tags = [:gpu] setup = [GPUBackend, DeviceLists] begin
    using StrictMode
    ext = Base.get_extension(PureMLIP, :PureMLIPGPUExt)
    T = Float32
    cu = DeviceLists.cu_bench(2)
    P, C = T.(permutedims(cu.positions)), T.(cu.cell)
    n = size(P, 2)
    dnl = PureMLIP.device_neighbor_list(todevice(P), todevice(C), 6.0; skin = 0.5)
    neighbor_list!(dnl, todevice(P), todevice(C))
    status = Int32[1, 0, 0, 0, 0]
    recip, grid = zeros(T, 3, 3), zeros(Int32, 7)
    ext._grid!(recip, grid, status, C, T(6.5), n)
    image, bin, bin_start = zeros(Int32, 3, n), zeros(Int32, 3, n), zeros(Int32, n + 1)
    bin_start[1] = 1
    for i in 1:n
        ext._bin_atom!(image, bin, bin_start, status, P, recip, grid, i)
    end
    ext._moved!(status, P, C, P, C, T(0.0625), 1)
    slots = zeros(Int32, n * ext._SEARCH_PARTS + 1)
    bin_atoms = Int32.(1:n)
    for w in eachindex(slots)
        ext._count_part!(slots, status, P, C, image, bin, cumsum(bin_start), bin_atoms, grid, T(42.25), w)
    end
    @test grid[7] > 0 && sum(slots) > 1
    @test isempty(StrictMode.conversion_report(ext; from = Float32, to = Float64))
end
