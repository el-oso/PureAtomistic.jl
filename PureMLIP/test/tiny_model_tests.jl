@testitem "tiny in-memory checkpoint loads and evaluates" begin
    ck = load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint())
    m = MACEModel(ck)
    P = [0.0 1.4 0.2; 0.0 0.1 1.3; 0.0 0.3 0.1]
    cell = [6.5 0 0; 0.4 6.2 0; 0.3 0.2 6.8]
    Z = [29, 1, 29]
    r = mace_energy_forces(m, P, cell, Z)
    @test isfinite(r.energy) && all(isfinite, r.forces)
    @test any(!iszero, r.forces)
    @test m.z_table == [1, 29] && m.C == 2 && m.r_max == 6.0
    h = 1.0e-5
    function fd(a, d)
        Pp = copy(P); Pp[d, a] += h
        Pm = copy(P); Pm[d, a] -= h
        return -(mace_energy_forces(m, Pp, cell, Z).energy - mace_energy_forces(m, Pm, cell, Z).energy) / 2h
    end
    worst = maximum(abs(fd(a, d) - r.forces[d, a]) for a in 1:3, d in 1:3)
    # Central difference (h = 1e-5) of the energy on every coordinate: the largest difference
    # observed is 1.4e-9 eV/Å, so the tolerance is 14 times that.
    @test worst < 2.0e-8
end

@testitem "documented repeated evaluation survives an edge count that grows within capacity" begin
    m = MACEModel(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint()))
    cell = [6.5 0 0; 0.4 6.2 0; 0.3 0.2 6.8]
    Z = [29, 1, 29]
    P = [0.0 1.4 0.2; 0.0 0.1 1.3; 0.0 0.3 0.1]
    P2 = [0.0 3.1 0.2; 0.0 3.0 3.3; 0.0 0.3 3.0]
    E2 = mace_energy_forces(m, P2, cell, Z).energy
    # Receiver blocks of the default size and of 64 atoms each hold all three atoms; blocks of one
    # atom split them.
    for block in (256, 64, 1)
        nl = neighbor_list(P, cell, m.r_max; extra_capacity = 64)
        n0 = nl.n_edges
        ws = MACEWorkspace{Float64}(m, size(P, 2), length(nl.senders); block)
        res = MACEResult{Float64}(size(P, 2))
        # Spreading the atoms apart brings periodic images inside the cutoff.
        neighbor_list!(nl, P2, cell)
        @test n0 < nl.n_edges <= length(nl.senders)
        r = mace_energy_forces!(res, ws, m, P2, cell, Z, nl)
        @test r === res && isfinite(r.energy)
        # The same positions through the allocating form give the same energy: the same arithmetic
        # with one block, and up to the rounding bound 10·eps·N of the matrix products with three
        # (N = 1280, the largest inner dimension of any matrix product of a full-size model; the
        # tiny model's largest is 10, the Bessel basis size, so the bound holds with a wide margin).
        if block >= size(P, 2)
            @test r.energy == E2
        else
            @test abs(r.energy - E2) <= 10 * eps() * 1280 * abs(E2)
        end
    end
end

@testitem "radial-MLP hidden widths must agree between the interactions" begin
    ck = load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint())
    f64 = copy(ck.state.f64)
    shapes = [f64["interactions.1.conv_tp_weights.layer$(j).weight"].shape for j in 0:3]
    H = shapes[2][1]
    @test all(count(==(H), s) == (j == 2 || j == 3 ? 2 : 1) for (j, s) in enumerate(shapes))
    for j in 0:3
        key = "interactions.1.conv_tp_weights.layer$(j).weight"
        s = replace(f64[key].shape, H => H + 1)
        f64[key] = PureMLIP.CheckpointTensor{Float64}(s, zeros(prod(s)), false)
    end
    bad = MACECheckpoint(ck.hyperparameters, PureMLIP.StateDict(f64, ck.state.i64, ck.state.names), ck.constants)
    @test_throws "model: unsupported checkpoint: interactions.1.conv_tp_weights hidden width" MACEModel(bad)
end
