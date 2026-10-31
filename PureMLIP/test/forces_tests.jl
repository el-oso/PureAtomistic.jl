@testitem "forces and virial match LAMMPS-MACE" setup = [LammpsReference, Checkpoints] begin
    using LinearAlgebra
    for (ref, src) in ((:mpa0_f64, :mpa0), (:mp0b3_f64, :mp0b3))
        model = MACEModel(load_mace_checkpoint(Checkpoints.fetch(src)))
        for s in LammpsReference.structures(LammpsReference.load(ref))
            r = mace_energy_forces(model, s.positions, s.cell, s.atomic_numbers)
            # Targets from the design (section 9): 1e-6 eV and 1e-6 eV/Å against LAMMPS-MACE.
            @test abs(r.energy - s.energy) <= 1.0e-6
            @test maximum(abs, r.forces - s.forces; init = 0.0) <= 1.0e-6
            @test maximum(abs, r.virial - s.virial) <= 1.0e-6
            @test r.stress ≈ -r.virial ./ abs(det(s.cell)) rtol = 4 * eps()
        end
    end
end

@testitem "forces are minus the energy gradient; stress is the strain derivative" setup = [LammpsReference, Checkpoints] begin
    using LinearAlgebra
    m = MACEModel(load_mace_checkpoint(Checkpoints.fetch(:mpa0))); ref = LammpsReference.load(:mpa0_f64)
    E(P, C, Z) = mace_energy_forces(m, P, C, Z).energy
    # Measured margin: the largest |F + ΔE/2h| over these nine components is 5.2e-7 eV/Å
    # (multi_element atom 2, x; the others are below 1e-8). tol_fd is 19 times that.
    tol_fd = 1.0e-5
    for (name, atoms) in (("cu_vacancy", (1, 7)), ("multi_element", (2,)))      # atom 2 of multi_element: H, ZBL active
        s = LammpsReference.structure(ref, name); Z = s.atomic_numbers
        F = mace_energy_forces(m, s.positions, s.cell, Z).forces
        h = 1.0e-4
        for a in atoms, d in 1:3
            Pp = copy(s.positions); Pp[d, a] += h; Pm = copy(s.positions); Pm[d, a] -= h
            @test F[d, a] ≈ -(E(Pp, s.cell, Z) - E(Pm, s.cell, Z)) / 2h atol = tol_fd
        end
    end
    # Measured margin: the largest |−virial_ab − ΔE/2h| over these twelve components is 5.6e-8 eV
    # (crconi_alloy, diagonal). tol_strain is 18 times that.
    tol_strain = 1.0e-6
    for name in ("triclinic_ni3al", "crconi_alloy")
        s = LammpsReference.structure(ref, name); Z = s.atomic_numbers
        W = mace_energy_forces(m, s.positions, s.cell, Z).virial
        h = 1.0e-5
        for (a, b) in ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))
            S = zeros(3, 3); S[a, b] += 0.5; S[b, a] += 0.5
            Ep = E((I + h * S) * s.positions, (I + h * S) * s.cell, Z); Em = E((I - h * S) * s.positions, (I - h * S) * s.cell, Z)
            # ∂E/∂ε_ab = −virial_ab.
            @test -W[a, b] ≈ (Ep - Em) / 2h atol = tol_strain
        end
    end
end

@testitem "rotation, translation and unwrapping" setup = [LammpsReference, Checkpoints] begin
    using LinearAlgebra
    m = MACEModel(load_mace_checkpoint(Checkpoints.fetch(:mpa0)))
    s = LammpsReference.structure(LammpsReference.load(:mpa0_f64), "triclinic_ni3al"); Z = s.atomic_numbers
    r0 = mace_energy_forces(m, s.positions, s.cell, Z)
    b = 64 * eps() * sum(abs, r0.node_energies)          # rounding bound on the summed energy
    # Measured margin: the largest difference is 7.0e-15 eV/Å in the forces and 1.6e-14 eV in the
    # virial (rotation and translation alike); tol_rot is 12 times the larger.
    tol_rot = 2.0e-13
    R = [1 0 0; 0 cos(1.1) -sin(1.1); 0 sin(1.1) cos(1.1)] * [cos(0.7) -sin(0.7) 0; sin(0.7) cos(0.7) 0; 0 0 1]
    r1 = mace_energy_forces(m, R * s.positions, R * s.cell, Z)
    @test abs(r1.energy - r0.energy) <= b
    @test r1.forces ≈ R * r0.forces atol = tol_rot
    @test r1.virial ≈ R * r0.virial * R' atol = tol_rot
    P2 = s.positions .+ [0.37, -1.1, 2.3]; P2[:, 3] .+= s.cell[:, 1] .- s.cell[:, 2]
    r2 = mace_energy_forces(m, P2, s.cell, Z)
    @test abs(r2.energy - r0.energy) <= b
    @test r2.forces ≈ r0.forces atol = tol_rot
    @test r2.virial ≈ r0.virial atol = tol_rot
end

@testitem "forces: in-place form, Float32, zero edges, views and errors" setup = [LammpsReference, Checkpoints] begin
    using LinearAlgebra, OffsetArrays
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0)); m = MACEModel(ck)
    ref = LammpsReference.load(:mpa0_f64)

    # The allocating form builds the list, workspace and result and calls the in-place form.
    s = LammpsReference.structure(ref, "crconi_alloy"); n = size(s.positions, 2)
    nl = neighbor_list(s.positions, s.cell, m.r_max)
    ws = MACEWorkspace{Float64}(m, n, nl.n_edges)
    r = mace_energy_forces!(MACEResult{Float64}(n), ws, m, s.positions, s.cell, s.atomic_numbers, nl)
    ra = mace_energy_forces(m, s.positions, s.cell, s.atomic_numbers)
    @test r.energy == ra.energy && r.node_energies == ra.node_energies
    @test r.forces == ra.forces && r.virial == ra.virial && r.stress == ra.stress
    # The forward pass of the in-place form is mace_energy!: same arithmetic, same energies.
    re = mace_energy!(MACEResult{Float64}(n), ws, m, s.positions, s.cell, s.atomic_numbers, nl)
    @test re.energy == r.energy && re.node_energies == r.node_energies
    # Views give the same arithmetic as the arrays they view.
    Pb = zeros(5, n + 2); Pb[2:4, 2:(n + 1)] = s.positions
    Cb = zeros(4, 4); Cb[2:4, 1:3] = s.cell
    Zb = [0; s.atomic_numbers; 0]
    rv = mace_energy_forces!(
        MACEResult{Float64}(n), ws, m, view(Pb, 2:4, 2:(n + 1)), view(Cb, 2:4, 1:3), view(Zb, 2:(n + 1)), nl
    )
    @test rv.energy == r.energy && rv.forces == r.forces && rv.virial == r.virial
    @test_throws ArgumentError mace_energy_forces!(
        MACEResult{Float64}(n), ws, m, OffsetArray(s.positions, 0:2, 0:(n - 1)), s.cell, s.atomic_numbers, nl
    )
    # A chunk smaller than the edge count changes only the summation order of the radial-MLP
    # products. Rounding bound: 64·eps·Σ|F| and 64·eps·Σ|virial| (measured: identical results).
    small = MACEWorkspace{Float64}(m, n, nl.n_edges; edge_chunk = 37)
    rs = mace_energy_forces!(MACEResult{Float64}(n), small, m, s.positions, s.cell, s.atomic_numbers, nl)
    @test rs.forces ≈ r.forces atol = 64 * eps() * sum(abs, r.forces)
    @test rs.virial ≈ r.virial atol = 64 * eps() * sum(abs, r.virial)

    # No edges: E0 only, and no position or strain dependence.
    s0 = LammpsReference.structure(ref, "isolated_cu")
    r0 = mace_energy_forces(m, s0.positions, s0.cell, s0.atomic_numbers)
    @test r0.forces == zeros(3, 1) && r0.virial == zeros(3, 3) && r0.stress == zeros(3, 3)

    # Float32 against the Float32 LAMMPS run.
    m32 = MACEModel{Float32}(ck)
    # Measured margins over the five structures: the largest force difference is 9.4e-6 eV/Å and
    # the largest virial difference 8.0e-5 eV (both crconi_alloy); each tolerance is 16 and 15
    # times its measurement.
    tolf32, tolv32 = 1.5e-4, 1.2e-3
    for s in LammpsReference.structures(LammpsReference.load(:mpa0_f32))
        local r = mace_energy_forces(m32, Float32.(s.positions), Float32.(s.cell), s.atomic_numbers)
        @test r.forces isa Matrix{Float32} && r.virial isa Matrix{Float32}
        @test maximum(abs, r.forces - s.forces; init = 0.0) <= tolf32
        @test maximum(abs, r.virial - s.virial) <= tolv32
    end

    # Capacity and result-shape errors.
    sv = LammpsReference.structure(ref, "cu_vacancy")
    nlv = neighbor_list(sv.positions, sv.cell, m.r_max)
    wsv = MACEWorkspace{Float64}(m, size(sv.positions, 2), nlv.n_edges)
    @test_throws "model: the workspace holds $(size(sv.positions, 2)) atoms" mace_energy_forces!(
        MACEResult{Float64}(n), wsv, m, s.positions, s.cell, s.atomic_numbers, nl
    )
    bad = MACEResult{Float64}(n); bad.forces = zeros(3, n + 1)
    @test_throws "model: the result forces have size (3, $(n + 1)), expected (3, $n)" mace_energy_forces!(
        bad, ws, m, s.positions, s.cell, s.atomic_numbers, nl
    )
    bad = MACEResult{Float64}(n); bad.stress = zeros(2, 2)
    @test_throws "model: the result virial and stress must be 3 × 3, got (3, 3) and (2, 2)" mace_energy_forces!(
        bad, ws, m, s.positions, s.cell, s.atomic_numbers, nl
    )
end
