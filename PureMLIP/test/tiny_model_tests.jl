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
