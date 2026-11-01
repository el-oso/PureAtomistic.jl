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
    # The message comes from Base.require_one_based_indexing, so only the type is checked.
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

# The two composed pullbacks of the backward pass, each against a ForwardDiff directional
# derivative of its forward function and a central difference along the same direction:
# ⟨pullback(dout), v⟩ = d/dt ⟨dout, f(in + t v)⟩.
@testitem "forces: density-normalization and message pullbacks" setup = [Checkpoints] begin
    using ForwardDiff, Random
    Mo = PureMLIP.Model
    struct LayerTag end
    D = ForwardDiff.Dual{LayerTag, Float64, 1}
    dual(a, v) = D.(a, ForwardDiff.Partials.(tuple.(v)))
    slope(y) = ForwardDiff.partials(y, 1)
    rng = Xoshiro(7)
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0))
    m = MACEModel(ck)
    P = [0.0 2.5 1.2 0.4; 0.0 0.0 1.9 2.6; 0.0 0.0 0.3 1.8]; C = [7.0 0.0 0.0; 0.0 7.0 0.0; 0.0 0.0 7.0]; Z = [29, 1, 8, 13]
    n = size(P, 2)
    nl = neighbor_list(P, C, 6.0)
    # A chunk smaller than the edge count: the message pullback recomputes the MLP per chunk.
    chunk = 7
    ws = MACEWorkspace{Float64}(m, n, nl.n_edges; edge_chunk = chunk)
    mace_energy!(MACEResult{Float64}(n), ws, m, P, C, Z, nl)
    @test nl.n_edges > 2 * chunk
    # Measured margins over k = 1, 2 (largest |pullback − reference|): density normalization
    # 4.3e-14 against ForwardDiff and 1.2e-9 against the central difference (h = 1e-5); messages
    # 1.1e-12 and 6.8e-7. Each tolerance is 15-19 times its measurement.
    tol_norm_ad, tol_norm_fd = 8.0e-13, 2.0e-8
    tol_msg_ad, tol_msg_fd = 2.0e-11, 1.0e-5
    for k in 1:2
        # Normalized linear: L = linear(M) / (density + 1).
        lin = m.linear[k]
        M = randn(rng, size(ws.M[k])); dens = rand(rng, n)
        vM = randn(rng, size(M)); vd = randn(rng, n)
        L = zeros(lin.C_out, lin.K_out, n)
        Mo._normalized_linear!(L, lin, M, dens)
        dL = randn(rng, size(L))
        dM = zeros(size(M)); dd = zeros(n)
        Mo._normalized_linear_pullback!(dM, dd, lin, copy(dL), L, dens)
        back = sum(dM .* vM) + sum(dd .* vd)
        LD = zeros(D, size(L))
        Mo._normalized_linear!(LD, lin, dual(M, vM), dual(dens, vd))
        fwd = slope(sum(dL .* LD))
        f(t) = sum(dL .* Mo._normalized_linear!(zeros(size(L)), lin, M .+ t .* vM, dens .+ t .* vd))
        h = 1.0e-5
        fd = (f(h) - f(-h)) / 2h
        @test back ≈ fwd atol = tol_norm_ad
        @test back ≈ fd atol = tol_norm_fd

        # Messages of interaction k, as functions of the sender features x and the edge vectors
        # r (through the harmonics and, with d = |r|, the radial features).
        E = nl.n_edges
        x = copy(view(ws.x[k], :, :, 1:n))
        r = copy(view(ws.r, :, 1:E))
        vx = randn(rng, size(x)); vr = randn(rng, 3, E)
        Mk = zeros(size(ws.M[k])); densk = zeros(n)
        dMk = randn(rng, size(Mk)); ddk = randn(rng, n)
        dx = zeros(size(x))
        fill!(ws.dEdr, 0.0)
        Mo._messages_pullback!(dx, ws, m, k, x, dMk, ddk, Z, nl)
        back = sum(dx .* vx) + sum(view(ws.dEdr, :, 1:E) .* vr)
        lengths(r) = vec(sqrt.(sum(abs2, r; dims = 1)))
        wsD = MACEWorkspace{D}(m, n, E; edge_chunk = chunk)
        wsD.r .= dual(r, vr); wsD.d .= lengths(wsD.r)
        MD = zeros(D, size(Mk)); densD = zeros(D, n)
        Mo._messages!(MD, densD, wsD, m, k, dual(x, vx), Z, nl)
        fwd = slope(sum(dMk .* MD) + sum(ddk .* densD))
        function g(t)
            ws.r[:, 1:E] .= r .+ t .* vr; ws.d[1:E] .= lengths(view(ws.r, :, 1:E))
            Mo._messages!(Mk, densk, ws, m, k, x .+ t .* vx, Z, nl)
            return sum(dMk .* Mk) + sum(ddk .* densk)
        end
        fd = (g(h) - g(-h)) / 2h
        ws.r[:, 1:E] .= r; ws.d[1:E] .= lengths(r)
        @test back ≈ fwd atol = tol_msg_ad
        @test back ≈ fd atol = tol_msg_fd
    end
end

@testitem "workspace memory per atom" setup = [Checkpoints, WorkspaceMemory] begin
    m64 = MACEModel(load_mace_checkpoint(Checkpoints.fetch(:mpa0)))
    # cu_bench(14) (test/reference/structures.jl) under mpa-0: 10,976 atoms, 856,128 directed
    # edges within r_max.
    b64 = WorkspaceMemory.workspace_bytes_per_atom(m64, Float64, 10_976, 856_128; edge_chunk = 4096)
    @test b64 <= 150_000
    GC.gc()
end

# Buffers whose lifetimes do not overlap share memory in MACEWorkspace. Every buffer of one
# workspace, sized for the largest structure, is filled with NaN before each evaluation, so a
# read that comes before any write within the call shows as a non-finite result. A read of a
# buffer after another member of its pool was written gives finite, deterministic values, which
# a fresh workspace with the same layout reproduces; the LAMMPS references catch those.
@testitem "forces: one workspace across structures, buffers filled with NaN" setup = [LammpsReference, Checkpoints] begin
    ck = load_mace_checkpoint(Checkpoints.fetch(:mpa0))
    # Float64: the LAMMPS targets of "forces and virial match LAMMPS-MACE". Float32: the measured
    # margins of "forces: in-place form, Float32, zero edges, views and errors".
    for (T, ref, tolf, tolv) in ((Float64, :mpa0_f64, 1.0e-6, 1.0e-6), (Float32, :mpa0_f32, 1.5e-4, 1.2e-3))
        m = MACEModel{T}(ck)
        structures = LammpsReference.structures(LammpsReference.load(ref))
        lists = [neighbor_list(T.(s.positions), T.(s.cell), m.r_max) for s in structures]
        ws = MACEWorkspace{T}(m, maximum(nl -> nl.n_atoms, lists), maximum(nl -> nl.n_edges, lists))
        for (s, nl) in zip(structures, lists)
            P, C, Z = T.(s.positions), T.(s.cell), s.atomic_numbers
            n = nl.n_atoms
            for f in fieldnames(typeof(ws))
                v = getfield(ws, f)
                for a in (v isa Tuple ? v : (v,))
                    a isa AbstractArray{T} && fill!(a, T(NaN))
                end
            end
            re = mace_energy!(MACEResult{T}(n), ws, m, P, C, Z, nl)
            r = mace_energy_forces!(MACEResult{T}(n), ws, m, P, C, Z, nl)
            # A fresh workspace sized for this structure runs the same arithmetic.
            rf = mace_energy_forces(m, P, C, Z)
            @test re.energy == rf.energy && re.node_energies == rf.node_energies
            @test r.energy == rf.energy && r.node_energies == rf.node_energies
            @test r.forces == rf.forces && r.virial == rf.virial && r.stress == rf.stress
            T === Float64 && @test abs(r.energy - s.energy) <= 1.0e-6
            @test maximum(abs, r.forces - s.forces; init = 0.0) <= tolf
            @test maximum(abs, r.virial - s.virial) <= tolv
        end
    end
end
