@testmodule ToolOracles begin
    using PureAtomistic.Systems, Random, LinearAlgebra

    # A System with unit masses and zero velocities.
    frame(P, species, cell) = System(P, zero(P), fill(one(eltype(P)), size(P, 2)), species, cell)

    # `n` points uniform in the cell `C`, from uniform fractional coordinates.
    uniform(rng, C, n) = C * rand(rng, eltype(C), 3, n)

    # A proper rotation about the unit axis `u` by `θ` (Rodrigues).
    function rotation(u, θ)
        u = u / norm(u)
        K = [0 -u[3] u[2]; u[3] 0 -u[1]; -u[2] u[1] 0]
        return I + sin(θ) * K + (1 - cos(θ)) * K * K
    end

    # L1₂ Cu₃Au: Au (79) on the cube corners, Cu (29) on the face centers.
    function l12(a, nrep)
        s = fcc_cell(a, nrep, :Cu)
        sp = copy(s.species)
        for k in axes(s.positions, 2)
            f = s.positions[:, k] ./ a
            all(x -> isinteger(round(x; digits = 8)), f) && (sp[k] = 79)
        end
        return frame(copy(s.positions), sp, copy(s.cell))
    end

    # Minimum RMSD over proper rotations by Horn's quaternion method: the largest eigenvalue of a
    # symmetric 4 × 4 matrix built from the covariance, independent of the SVD of Kabsch's method.
    # Re-derived here, not taken from the source.
    function horn_rmsd(X, Y)
        x = X .- sum(X; dims = 2) / size(X, 2)
        y = Y .- sum(Y; dims = 2) / size(Y, 2)
        S = x * y'
        Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz = S[1, 1], S[1, 2], S[1, 3], S[2, 1], S[2, 2], S[2, 3], S[3, 1], S[3, 2], S[3, 3]
        N = [
            Sxx + Syy + Szz Syz - Szy Szx - Sxz Sxy - Syx
            Syz - Szy Sxx - Syy - Szz Sxy + Syx Szx + Sxz
            Szx - Sxz Sxy + Syx -Sxx + Syy - Szz Syz + Szy
            Sxy - Syx Szx + Sxz Syz + Szy -Sxx - Syy + Szz
        ]
        λ = maximum(eigvals(Symmetric(N)))
        return sqrt(max(zero(λ), sum(abs2, x) + sum(abs2, y) - 2λ) / size(X, 2))
    end
end

# Ideal gas: for points uniform in the cell, the minimum-image displacement of a pair is uniform
# over the cell and independent of either point, so the pair events are pairwise independent and
# the unordered count C_k of bin k over M frames has mean μ_k = M N(N-1)/2 p_k and variance
# μ_k (1 - p_k), p_k the bin's shell volume over V. Then g_k = C_k / μ_k has σ_k = √((1 - p_k)/μ_k).
# Power: N = 64, M = 200 and r_max = h_min/2 = 4 Å in this cell give a pooled σ of 0.20 %, so the
# pooled 4σ test detects a uniform normalization bias of 0.82 % (the density N/V in place of
# (N-1)/V is a bias of 1.6 %, 7.6σ); the per-bin σ_k run from 8.2 % (first bin) to 0.47 % (last), so
# the 4σ test per bin detects a shell-volume error of 33 % in the first bin and 1.9 % in the last.
@testitem "rdf: an ideal gas gives g = 1 within its error" tags = [:analysis] setup = [ToolOracles] begin
    using PureAtomistic.Analysis, PureAtomistic.Systems, Random, LinearAlgebra
    C = [10.0 3.0 2.0; 0.0 9.0 1.0; 0.0 0.0 8.0]   # triclinic: lattice vectors in columns
    V = cell_volume(C)
    hmin = minimum(V ./ (norm(cross(C[:, 2], C[:, 3])), norm(cross(C[:, 3], C[:, 1])), norm(cross(C[:, 1], C[:, 2]))))
    N, M, nb = 64, 200, 10
    rng = Xoshiro(20261109)
    frames = [ToolOracles.frame(ToolOracles.uniform(rng, C, N), fill(29, N), C) for _ in 1:M]
    r_max = hmin / 2
    res = rdf(frames, r_max, nb)
    Δ = r_max / nb
    p = [4π / 3 * ((k * Δ)^3 - ((k - 1) * Δ)^3) / V for k in 1:nb]
    μ = M * N * (N - 1) / 2 .* p
    z = (res.g .- 1) ./ sqrt.((1 .- p) ./ μ)
    C_tot = sum(res.g .* μ)
    μ_tot = sum(μ)
    p_tot = 4π / 3 * r_max^3 / V
    @test all(abs.(z) .< 4)
    @test abs(C_tot / μ_tot - 1) < 4 * sqrt((1 - p_tot) / μ_tot)
    @test res.r ≈ ((1:nb) .- 0.5) .* Δ rtol = 4 * eps()   # (b): three operations per center
    # Selecting all atoms by species gives the same pairs.
    @test rdf(frames, r_max, nb; species_pair = (29, 29)) == res
    # `nothing` against a species that every atom has selects the same pairs.
    @test rdf(frames, r_max, nb; species_pair = (nothing, 29)) == res
    @test rdf(frames, r_max, nb; species_pair = (29, nothing)) == res
    # Two species, half each: `(nothing, 29)` overlaps the selection in the 32 Cu atoms, so the
    # ideal pair count is N·N_Cu - N_Cu. Each unordered pair is counted at most twice, so the
    # variance of the ordered count is at most twice its mean: σ_pool ≤ √(2(1 - p)/μ). Power: the
    # pair count N·N_Cu without the overlap is a bias of 1/N = 1.6 %, 5.4 times that bound.
    two = [ToolOracles.frame(f.positions, [isodd(i) ? 29 : 79 for i in 1:N], C) for f in frames]
    for pair in ((nothing, 29), (29, nothing), (79, 29))
        nA = isnothing(pair[1]) ? N : N ÷ 2
        nB = isnothing(pair[2]) ? N : N ÷ 2
        nAB = pair[1] == pair[2] || isnothing(pair[1]) || isnothing(pair[2]) ? N ÷ 2 : 0
        μp = M * (nA * nB - nAB) * p_tot
        gp = rdf(two, r_max, nb; species_pair = pair)
        pooled = sum(gp.g .* p) / p_tot
        @test abs(pooled - 1) < 4 * sqrt(2 * (1 - p_tot) / μp)
    end
end

# Perfect fcc Cu: shells of 12, 6, 24 and 12 neighbors at a/√2, a, a√(3/2) and a√2.
@testitem "rdf: the fcc shells of Cu by position and count" tags = [:analysis] setup = [ToolOracles] begin
    using PureAtomistic.Analysis, PureAtomistic.Systems
    for T in (Float64, Float32)
        a = T(3.61)
        nb, r_max = 108, T(5.4)
        Δ = r_max / nb
        big = fcc_cell(a, 4, :Cu)   # half height 7.22 Å: minimum image
        res = rdf([big, big], r_max, nb)
        shells = (a / sqrt(T(2)), a, a * sqrt(T(3) / 2), a * sqrt(T(2)))
        counts = (12, 6, 24, 12)
        bins = [floor(Int, s / Δ) + 1 for s in shells]
        steps = diff([zero(T); res.n])
        @test findall(!iszero, steps) == bins
        @test steps[bins] == T[counts...]
        @test res.n[end] == 54
        @test findall(!iszero, res.g) == bins
        # The normalization of the first shell: 12 neighbors at density (N - 1)/V.
        k = bins[1]
        N, V = natoms(big), volume(big)
        @test res.g[k] ≈ 12 / ((N - 1) / V * 4 * T(π) / 3 * ((k * Δ)^3 - ((k - 1) * Δ)^3)) rtol = 64 * eps(T)   # (b): about ten operations
        # Cells of half height below r_max need every image; the per-atom counts do not change.
        for small in (fcc_cell(a, 1, :Cu), fcc_cell(a, 2, :Cu), fcc_cell(a, (1, 2, 3), :Cu))
            @test rdf([small], r_max, nb).n == res.n
        end
        # A triclinic cell: 2 × 2 × 2 primitive fcc cells, rotated.
        R = T.(ToolOracles.rotation([1, 2, 3], 0.7))
        Cp = R * (a / 2 * T[0 1 1; 1 0 1; 1 1 0]) * 2
        sites = [R * (a / 2 * T[0 1 1; 1 0 1; 1 1 0]) * T[i, j, k] for i in 0:1, j in 0:1, k in 0:1]
        tri = ToolOracles.frame(reduce(hcat, vec(sites)), fill(29, 8), Cp)
        @test rdf([tri], r_max, nb).n == res.n
        # With every image counted the density is N/V, so cells of 1, 4 and 32 atoms give one g:
        # the one-atom primitive cell, the conventional cell and 2 × 2 × 2 of them. Bound (b):
        # a cell volume and a shell volume, about ten operations, 64·eps.
        prim = ToolOracles.frame(zeros(T, 3, 1), [29], a / 2 * T[0 1 1; 1 0 1; 1 1 0])
        g1 = rdf([fcc_cell(a, 1, :Cu)], r_max, nb).g
        @test rdf([prim], r_max, nb).g ≈ g1 rtol = 64 * eps(T)
        @test rdf([fcc_cell(a, 2, :Cu)], r_max, nb).g ≈ g1 rtol = 64 * eps(T)
        @test rdf([prim], r_max, nb).n == res.n
    end
end

@testitem "rdf: species pairs of L1₂ Cu₃Au" tags = [:analysis] setup = [ToolOracles] begin
    using PureAtomistic.Analysis, PureAtomistic.Systems
    s = ToolOracles.l12(3.75, 3)
    nb, r_max = 60, 3.0   # the first shell only, at a/√2 = 2.65 Å
    @test rdf([s], r_max, nb; species_pair = (79, 29)).n[end] == 12
    @test rdf([s], r_max, nb; species_pair = (29, 79)).n[end] == 4
    @test rdf([s], r_max, nb; species_pair = (29, 29)).n[end] == 8
    @test rdf([s], r_max, nb; species_pair = (79, 79)).n[end] == 0
end

@testitem "coordination numbers of perfect fcc are 12" tags = [:analysis] setup = [ToolOracles] begin
    using PureAtomistic.Analysis, PureAtomistic.Systems
    for T in (Float64, Float32)
        a = T(3.61)
        R = T.(ToolOracles.rotation([3, -1, 2], 1.1))
        rot(s) = ToolOracles.frame(R * s.positions, s.species, R * s.cell)
        prim = ToolOracles.frame(zeros(T, 3, 1), [29], a / 2 * T[0 1 1; 1 0 1; 1 1 0])
        for s in (fcc_cell(a, 4, :Cu), fcc_cell(a, 1, :Cu), fcc_cell(a, (2, 1, 3), :Cu), rot(fcc_cell(a, 3, :Cu)), prim, rot(prim))
            @test coordination_numbers(s, T(0.85) * a) == fill(12, natoms(s))
            @test coordination_numbers(s, T(1.1) * a) == fill(18, natoms(s))
            @test coordination_numbers(s, T(1.3) * a) == fill(42, natoms(s))
        end
    end
    s = ToolOracles.l12(3.75, 2)
    nAu, nCu = count(==(79), s.species), count(==(29), s.species)
    @test coordination_numbers(s, 3.0; species_pair = (79, 29)) == fill(12, nAu)
    @test coordination_numbers(s, 3.0; species_pair = (29, 79)) == fill(4, nCu)
    @test coordination_numbers(s, 3.0; species_pair = (29, nothing)) == fill(12, nCu)
    c = zeros(Int, natoms(s))
    coordination_numbers!(c, s.positions, s.cell, s.species, 3.0; species_pair = (79, 29))
    @test c == [z == 79 ? 12 : 0 for z in s.species]
end

# One atom at constant velocity through a triclinic cell, stored wrapped: the unwrapped
# displacement over τ is v τ, so the MSD is |v|² τ² up to rounding of the wrap.
@testitem "msd: unwrapping a straight path through a triclinic cell" tags = [:analysis] setup = [ToolOracles] begin
    using PureAtomistic.Analysis, PureAtomistic.Systems, PureAtomistic.IO
    C = [6.0 1.5 -1.0; 0.0 5.5 0.8; 0.0 0.0 5.0]
    v = [0.9, -0.7, 1.3]          # Å per frame, below half the shortest height
    Ci = inv(C)
    wrapc(x) = C * (Ci * x .- floor.(Ci * x))
    nf = 60
    x0 = [0.3, 0.2, 0.1]
    still = [2.0, 2.0, 2.0]
    frames = [ToolOracles.frame([wrapc(x0 .+ k .* v) still], [1, 2], C) for k in 0:(nf - 1)]
    m = msd(frames; species = 1, max_lag = 10, nblocks = 3, dt = 2.0)
    want = [sum(abs2, v) * τ^2 for τ in 0:10]
    # Measured: the largest error relative to max(want, 1 Å²) is 2.2e-15; the bound is 20 times that.
    @test all(abs.(m.msd_A2 .- want) .<= 4.5e-14 .* max.(want, 1))
    @test m.lag_fs == 2.0 .* (0:10)
    @test size(m.msd_A2) == (11, 3)
    both = msd(frames; max_lag = 10, nblocks = 3, dt = 2.0)
    @test both.msd_A2 ≈ m.msd_A2 ./ 2 rtol = 4 * eps()   # the still atom adds exactly 0; (b): one division
    # The same frames as IO.Frames carry their own times.
    iof = [Frame{Float64}(["H", "He"], f.positions; cell = f.cell, time = 2.0 * (k - 1)) for (k, f) in enumerate(frames)]
    @test msd(iof; species = "H", max_lag = 10, nblocks = 3).msd_A2 == m.msd_A2
end

# Brownian walkers: Gaussian steps of variance s² per component per frame interval dt give
# D = s² / (2 dt) and ⟨|Δr|²⟩ = 6 D τ with no offset. Power: N = 100 walkers over 2000 frames in
# 10 blocks give a block error σ_D of 0.93 % of D here (measured), so the 4σ test rejects a
# relative bias above 3.7 %; a wrong Einstein factor (2 or 3 in place of 6) is a bias of 100 % or
# more. The bound σ_D < 2 % of D keeps that power from eroding.
@testitem "msd: a random walk recovers its diffusion coefficient" tags = [:analysis] setup = [ToolOracles] begin
    using PureAtomistic.Analysis, PureAtomistic.Systems, Random, LinearAlgebra
    C = [12.0 2.0 1.0; 0.0 11.0 -1.5; 0.0 0.0 10.0]
    Ci = inv(C)
    N, nf, s, dt = 100, 2000, 0.15, 5.0
    rng = Xoshiro(7)
    X = ToolOracles.uniform(rng, C, N)
    frames = Vector{Any}(undef, nf)
    for k in 1:nf
        F = Ci * X
        frames[k] = ToolOracles.frame(C * (F .- floor.(F)), fill(29, N), C)
        X .+= s .* randn(rng, 3, N)
    end
    frames = identity.(frames)
    m = msd(frames; max_lag = 20, nblocks = 10, dt)
    fit = diffusion_coefficient(m.lag_fs, m.msd_A2; fit_window = (5dt, 20dt))
    D = s^2 / (2dt)
    @test abs(fit.D - D) < 4 * fit.σ_D
    @test 0 < fit.σ_D < 0.02 * D
end

# A line 6 D_b τ + c per block: the fit returns each D_b exactly up to rounding, their mean and
# the standard error of the mean.
@testitem "diffusion_coefficient: the Einstein slope and its block error" tags = [:analysis] begin
    using PureAtomistic.Analysis, Statistics
    lag = collect(0.0:10.0:200.0)
    Db = [1.0e-3, 1.4e-3, 0.8e-3, 1.2e-3]
    M = [6 * D * t + 0.3 + (t < 30 ? 0.5 : 0.0) for t in lag, D in Db]
    fit = diffusion_coefficient(lag, M; fit_window = (30.0, 200.0))
    # Measured: 0 and 4.4e-16 relative; the bounds are 20 times 1 eps and the measured 4.4e-16.
    @test fit.D ≈ mean(Db) rtol = 20 * eps()
    @test fit.σ_D ≈ std(Db) / 2 rtol = 20 * 4.4e-16
end

@testitem "rmsd: a rigidly moved frame and a mirror image" tags = [:analysis] setup = [ToolOracles] begin
    using PureAtomistic.Analysis, PureAtomistic.Systems, Random, LinearAlgebra
    rng = Xoshiro(3)
    Y = 5 .* randn(rng, 3, 20)
    R = ToolOracles.rotation([1, -2, 0.5], 2.0)
    t = [1.0, -4.0, 2.5]
    moved = R * Y .+ t
    mirror = R * (Diagonal([1.0, 1.0, -1.0]) * Y) .+ t
    fr = [ToolOracles.frame(X, fill(29, 20), Matrix(10.0I, 3, 3)) for X in (Y, moved, mirror)]
    plain = rmsd(fr, Y)
    @test plain[1] == 0
    # Bound (b): a sum of 60 squares and a square root, 2·eps·60 = 2.7e-14 relative.
    @test plain[2] ≈ sqrt(sum(abs2, moved .- Y) / 20) rtol = 2.7e-14
    @test plain[3] ≈ sqrt(sum(abs2, mirror .- Y) / 20) rtol = 2.7e-14
    al = rmsd(fr, Y; align = true)
    # Measured: 2.2e-15 Å for the reference itself and 7.5e-15 Å for the rigidly moved frame
    # (coordinates up to 12 Å); the bound is 20 times the larger.
    @test al[1] < 1.5e-13
    @test al[2] < 1.5e-13
    # A mirror image cannot be superposed by a proper rotation; Horn's method gives the minimum.
    # Measured: the two agree to the last bit here; the bound is the rounding of the eigenvalue and
    # of the difference G - 2λ, 64·eps·(G / E) with the ratio G / E below 20 in this case.
    @test al[3] ≈ ToolOracles.horn_rmsd(mirror, Y) rtol = 64 * eps() * 20
    @test al[3] > 1
end

@testitem "analysis tools read views, OffsetArrays and trajectories like plain arrays" tags = [:analysis] setup = [ToolOracles] begin
    using PureAtomistic.Analysis, PureAtomistic.Systems, PureAtomistic.IO, OffsetArrays, Random
    rng = Xoshiro(11)
    C = [9.0 1.0 0.5; 0.0 8.5 -0.7; 0.0 0.0 8.0]
    N = 12
    X0 = ToolOracles.uniform(rng, C, N)
    Xs = [X0 .+ 0.3 .* randn(rng, 3, N) for _ in 1:30]
    sp = [isodd(i) ? 29 : 79 for i in 1:N]
    plain = [ToolOracles.frame(X, sp, C) for X in Xs]
    off(X) = System(
        OffsetArray(X, 0:2, -5:(N - 6)), OffsetArray(zero(X), 0:2, -5:(N - 6)),
        OffsetArray(ones(N), -5:(N - 6)), OffsetArray(sp, -5:(N - 6)), OffsetArray(C, 3:5, -1:1)
    )
    function viewed(X)
        big = zeros(4, N + 3)
        big[2:4, 3:(N + 2)] .= X
        return ToolOracles.frame(view(big, 2:4, 3:(N + 2)), view([0; 0; sp; 0], 3:(N + 2)), view([C C], :, 4:6))
    end
    for wrap in (off, viewed)
        fs = [wrap(X) for X in Xs]
        @test rdf(fs, 6.0, 12) == rdf(plain, 6.0, 12)
        @test rdf(fs, 3.5, 12; species_pair = (29, 79)) == rdf(plain, 3.5, 12; species_pair = (29, 79))
        @test msd(fs; max_lag = 4, nblocks = 3, dt = 1.0) == msd(plain; max_lag = 4, nblocks = 3, dt = 1.0)
        @test collect(coordination_numbers(fs[1], 3.0)) == coordination_numbers(plain[1], 3.0)
        @test coordination_numbers(fs[1], 3.0; species_pair = (79, 29)) == coordination_numbers(plain[1], 3.0; species_pair = (79, 29))
        @test rmsd(fs, fs[1].positions; align = true) == rmsd(plain, Xs[1]; align = true)
        @test rmsd(fs, fs[1].positions) == rmsd(plain, Xs[1])
    end
    # The coordination numbers carry the atom axis of the frame.
    @test axes(coordination_numbers(off(X0), 3.0), 1) == -5:(N - 6)
    # A TrajectoryReader is read like the vector of its frames.
    path = tempname()
    w = TrajectoryWriter(path, N, ["Cu" for _ in 1:N]; velocities = false)
    for (k, X) in enumerate(Xs)
        write_frame(w, 10.0 * k, X; cell = C)
    end
    close(w)
    r = TrajectoryReader(path)
    @test eachindex(r) == 1:30 && firstindex(r) == 1 && lastindex(r) == 30
    @test r[3].positions == Float64.(Float32.(Xs[3]))
    frames = [read_frame(r, k) for k in 1:30]
    @test msd(r; max_lag = 4, nblocks = 3) == msd(frames; max_lag = 4, nblocks = 3)
    @test rdf(r, 6.0, 12) == rdf(frames, 6.0, 12)
    close(r)
    rm(path)
end

@testitem "analysis tools reject bad input" tags = [:analysis] setup = [ToolOracles] begin
    using PureAtomistic.Analysis, PureAtomistic.Systems
    s = fcc_cell(3.61, 2, :Cu)
    C = s.cell
    @test_throws "pureatomistic: rdf needs at least one bin, got 0" rdf([s], 5.0, 0)
    @test_throws "pureatomistic: rdf needs at least one frame" rdf(typeof(s)[], 5.0, 10)
    @test_throws "pureatomistic: r_max must be positive and finite, got -1.0" rdf([s], -1.0, 10)
    @test_throws "pureatomistic: rdf finds no pairs to count: frame 1 has 32 and 0 atoms of the pair, 0 in both" rdf([s], 5.0, 10; species_pair = (29, 79))
    one = ToolOracles.frame(zeros(3, 1), [29], C)
    @test_throws "frame 1 has 1 and 1 atoms of the pair, 1 in both" rdf([one], 3.0, 10)
    @test_throws "pureatomistic: species_pair must be nothing or a 2-tuple, got Vector{Int64}" rdf([s], 5.0, 10; species_pair = [29, 29])
    @test_throws "pureatomistic: rdf needs at least one bin" rdf_histogram!(Int[], s.positions, C, s.species, 5.0)
    @test_throws "pureatomistic: the cell must be 3 × 3, got (2, 2)" rdf_histogram!(zeros(Int, 3), s.positions, C[1:2, 1:2], s.species, 5.0)
    @test_throws "pureatomistic: positions need 3 rows, got 2" rdf_histogram!(zeros(Int, 3), s.positions[1:2, :], C, s.species, 5.0)
    @test_throws "species axes Base.OneTo(31) differ from the atom axis Base.OneTo(32)" rdf_histogram!(zeros(Int, 3), s.positions, C, s.species[1:31], 5.0)
    @test_throws "pureatomistic: cell is singular" rdf_histogram!(zeros(Int, 3), s.positions, zeros(3, 3), s.species, 5.0)
    @test_throws "pureatomistic: r_cut must be positive and finite, got Inf" coordination_numbers(s, Inf)
    @test_throws "counts axes Base.OneTo(3) differ from the atom axis Base.OneTo(32)" coordination_numbers!(zeros(Int, 3), s.positions, C, s.species, 3.0)
    @test_throws "pureatomistic: no atom has species 79" coordination_numbers(s, 3.0; species_pair = (79, 29))
    fs = [s for _ in 1:10]
    @test_throws "pureatomistic: max_lag must be at least 1, got 0" msd(fs; max_lag = 0, dt = 1.0)
    @test_throws "pureatomistic: nblocks must be at least 1, got 0" msd(fs; max_lag = 1, nblocks = 0, dt = 1.0)
    @test_throws "pureatomistic: dt must be positive, got -1.0" msd(fs; max_lag = 1, dt = -1.0)
    @test_throws "pureatomistic: msd has 10 frames in 5 blocks of 2, too few for a lag of 2 frames" msd(fs; max_lag = 2, dt = 1.0)
    @test_throws "pureatomistic: no atom has species 79" msd(fs; species = 79, max_lag = 1, dt = 1.0)
    @test_throws "pureatomistic: these frames carry no time; pass the frame interval as dt (fs)" msd(fs; max_lag = 1)
    s3 = fcc_cell(3.61, 1, :Cu)
    @test_throws "has positions (Base.OneTo(3), Base.OneTo(4)), frame 1 (Base.OneTo(3), Base.OneTo(32))" msd([fs; s3; fs]; max_lag = 1, nblocks = 1, dt = 1.0)
    other = ToolOracles.frame(copy(s.positions), [i == 7 ? 79 : 29 for i in 1:32], C)
    @test_throws "pureatomistic: frame 2 has other species than frame 1" msd([s, other, s]; max_lag = 1, nblocks = 1, dt = 1.0)
    using PureAtomistic.IO
    tf(t) = Frame{Float64}(fill("Cu", 32), s.positions; cell = C, time = t)
    @test_throws "pureatomistic: frame times must increase, got 1.0 after 1.0" msd([tf(1.0), tf(1.0), tf(2.0), tf(3.0)]; max_lag = 1, nblocks = 1)
    @test_throws "pureatomistic: frame times are not evenly spaced: 3.5 after 2.0, interval 1.0" msd([tf(1.0), tf(2.0), tf(3.5), tf(4.0)]; max_lag = 1, nblocks = 1)
    jump = copy(s.positions)
    jump[:, 5] .+= (3.0, 3.0, 0.0)   # 4.24 Å, more than half of the 7.22 Å height
    @test_throws "pureatomistic: atom 5 moved" msd([s, ToolOracles.frame(jump, s.species, C)]; max_lag = 1, nblocks = 1, dt = 1.0)
    @test_throws "must have equal axes" unwrap!(zeros(3, 2), zeros(3, 3), zeros(3, 3), C)
    @test_throws "pureatomistic: positions need 3 rows, got 2" unwrap!(zeros(2, 3), zeros(2, 3), zeros(2, 3), C)
    lag = collect(0.0:1.0:10.0)
    @test_throws "msd rows Base.OneTo(10) differ from the lags Base.OneTo(11)" diffusion_coefficient(lag, zeros(10, 3); fit_window = (1, 5))
    @test_throws "pureatomistic: diffusion_coefficient needs at least 2 blocks for an error, got 1" diffusion_coefficient(lag, zeros(11, 1); fit_window = (1, 5))
    @test_throws "pureatomistic: the fit window (4.5, 4.9) holds 0 lags; it needs at least 2" diffusion_coefficient(lag, zeros(11, 2); fit_window = (4.5, 4.9))
    @test_throws "pureatomistic: reference needs 3 rows, got 2" rmsd(fs, zeros(2, 32))
    @test_throws "pureatomistic: rmsd needs at least one atom" rmsd(fs, zeros(3, 0))
    @test_throws "pureatomistic: rmsd needs at least one frame" rmsd(typeof(s)[], zeros(3, 32))
    @test_throws "frame 1 has positions (Base.OneTo(3), Base.OneTo(32)), the reference (Base.OneTo(3), Base.OneTo(31))" rmsd(fs, zeros(3, 31))
end

@testitem "analysis tool kernels allocate nothing" tags = [:analysis, :checks] setup = [ToolOracles] begin
    using PureAtomistic.Analysis, PureAtomistic.Systems
    using StrictModeTest, StrictMode
    for T in (Float64, Float32)
        a = T(3.61)
        for s in (fcc_cell(a, 3, :Cu), fcc_cell(a, 1, :Cu))   # minimum image and all images
            h = zeros(Int, 40)
            c = zeros(Int, natoms(s))
            u, p = copy(s.positions), copy(s.positions)
            @noinline f!(h, s) = rdf_histogram!(h, s.positions, s.cell, s.species, T(4.5))
            @noinline fp!(h, s) = rdf_histogram!(h, s.positions, s.cell, s.species, T(4.5); species_pair = (29, 29))
            @noinline g!(c, s) = coordination_numbers!(c, s.positions, s.cell, s.species, T(3))
            @noinline w!(u, p, s) = unwrap!(u, p, s.positions, s.cell)
            f!(h, s)
            fp!(h, s)
            g!(c, s)
            w!(u, p, s)
            @test_noalloc f!(h, s)
            @test_noalloc fp!(h, s)
            @test_noalloc g!(c, s)
            @test_noalloc w!(u, p, s)
        end
        acc, ring = zeros(T, 3, 2), rand(T, 3, 5, 3)
        @noinline q!(acc, ring) = PureAtomistic.Analysis._accumulate_lag!(acc, ring, 2, 1, 2, 1)
        q!(acc, ring)
        @test_noalloc q!(acc, ring)
    end
end

@testitem "TrimCheck: analysis tools" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureAtomistic.Analysis, PureAtomistic.Systems, PureAtomistic.IO
            const S64 = System{Float64, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Vector{Int}, Matrix{Float64}}
            const S32 = System{Float32, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Vector{Int}, Matrix{Float32}}
            msd_dt(f, dt) = msd(f; max_lag = 4, dt)
            msd_t(f) = msd(f; max_lag = 4)
            msd_sp(f, s) = msd(f; species = s, max_lag = 4)
            dc(l, m) = diffusion_coefficient(l, m; fit_window = (1.0, 3.0))
            rdf_pair(f, r, n) = rdf(f, r, n; species_pair = (29, 29))
            cn_pair(f, r) = coordination_numbers(f, r; species_pair = (29, 29))
            rmsd_al(f, y) = rmsd(f, y; align = true)
            hist_pair(h, p, c, s, r) = rdf_histogram!(h, p, c, s, r; species_pair = ("Cu", "Cu"))
            msd_in(l, m, u, p, g, f) = msd!(l, m, u, p, g, f; dt = 1.0)
            rmsd_in_al(o, f, y) = rmsd!(o, f, y; align = true)
        end,
        rdf(Vector{S64}, Float64, Int),
        rdf(Vector{S32}, Float32, Int),
        rdf(Vector{Frame{Float64}}, Float64, Int),
        rdf(TrajectoryReader, Float64, Int),
        rdf_pair(Vector{S64}, Float64, Int),
        rdf_histogram!(Vector{Int}, Matrix{Float64}, Matrix{Float64}, Vector{Int}, Float64),
        rdf_histogram!(Vector{Int}, Matrix{Float32}, Matrix{Float32}, Vector{Int}, Float32),
        hist_pair(Vector{Int}, Matrix{Float64}, Matrix{Float64}, Vector{String}, Float64),
        coordination_numbers(S64, Float64),
        coordination_numbers(S32, Float32),
        cn_pair(S64, Float64),
        coordination_numbers!(Vector{Int}, Matrix{Float64}, Matrix{Float64}, Vector{Int}, Float64),
        unwrap!(Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}),
        unwrap!(Matrix{Float32}, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}),
        msd_dt(Vector{S64}, Float64),
        msd_dt(Vector{S32}, Float32),
        msd_t(Vector{Frame{Float64}}),
        msd_t(TrajectoryReader),
        msd_sp(Vector{Frame{Float64}}, String),
        dc(Vector{Float64}, Matrix{Float64}),
        dc(Vector{Float32}, Matrix{Float32}),
        rmsd(Vector{S64}, Matrix{Float64}),
        rmsd_al(Vector{S64}, Matrix{Float64}),
        rmsd_al(Vector{S32}, Matrix{Float32}),
        rdf!(Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Int}, Vector{S64}, Float64),
        rdf!(Vector{Float32}, Vector{Float32}, Vector{Float32}, Vector{Int}, Vector{S32}, Float32),
        msd_in(Vector{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Array{Float64, 3}, Vector{S64}),
        msd_in(Vector{Float32}, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}, Array{Float32, 3}, Vector{S32}),
        rmsd!(Vector{Float64}, Vector{S64}, Matrix{Float64}),
        rmsd_in_al(Vector{Float64}, Vector{S64}, Matrix{Float64}),
        rmsd_in_al(Vector{Float32}, Vector{S32}, Matrix{Float32}),
    )
end

@testitem "analysis tool return types are concrete" tags = [:checks] begin
    using PureAtomistic.Analysis, PureAtomistic.Systems, PureAtomistic.IO
    for T in (Float64, Float32)
        S = System{T, Matrix{T}, Matrix{T}, Vector{T}, Vector{Int}, Matrix{T}}
        @test Base.return_types(rdf, (Vector{S}, T, Int)) == [@NamedTuple{r::Vector{T}, g::Vector{T}, n::Vector{T}}]
        @test Base.return_types(coordination_numbers, (S, T)) == [Vector{Int}]
        @test Base.return_types(rmsd, (Vector{S}, Matrix{T})) == [Vector{T}]
        @test Base.return_types(rmsd!, (Vector{T}, Vector{S}, Matrix{T})) == [Vector{T}]
        @test Base.return_types(rdf!, (Vector{T}, Vector{T}, Vector{T}, Vector{Int}, Vector{S}, T)) ==
            [@NamedTuple{r::Vector{T}, g::Vector{T}, n::Vector{T}}]
        msd_in(l, m, u, p, g, f) = msd!(l, m, u, p, g, f; dt = 1.0)
        @test Base.return_types(msd_in, (Vector{T}, Matrix{T}, Matrix{T}, Matrix{T}, Array{T, 3}, Vector{S})) ==
            [@NamedTuple{lag_fs::Vector{T}, msd_A2::Matrix{T}}]
        @test Base.return_types(unwrap!, (Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T})) == [Matrix{T}]
        @test Base.return_types(rdf_histogram!, (Vector{Int}, Matrix{T}, Matrix{T}, Vector{Int}, T)) == [Vector{Int}]
        msd_dt(f, dt) = msd(f; max_lag = 4, dt)
        @test Base.return_types(msd_dt, (Vector{S}, T)) == [@NamedTuple{lag_fs::Vector{T}, msd_A2::Matrix{T}}]
        dc(l, m) = diffusion_coefficient(l, m; fit_window = (1, 3))
        @test Base.return_types(dc, (Vector{T}, Matrix{T})) == [@NamedTuple{D::T, σ_D::T}]
    end
    msd_t(f) = msd(f; max_lag = 4)
    @test Base.return_types(msd_t, (TrajectoryReader,)) == [@NamedTuple{lag_fs::Vector{Float64}, msd_A2::Matrix{Float64}}]
end

@testitem "in-place analysis kernels equal the allocating forms and allocate nothing" tags = [:analysis, :checks] setup = [ToolOracles] begin
    using PureAtomistic.Analysis, PureAtomistic.Systems, Random
    using StrictModeTest, StrictMode
    @noinline rdf_in!(r, g, n, h, fs) = rdf!(r, g, n, h, fs, 4.0)
    @noinline msd_in!(l, m, u, p, ring, fs) = msd!(l, m, u, p, ring, fs; dt = 2.0)
    @noinline rmsd_in!(o, fs, y) = rmsd!(o, fs, y; align = true)
    @noinline sites_in!(s, c, a) = reference_sites!(s, c, a, 2)
    for T in (Float64, Float32)
        rng = Xoshiro(5)
        C = T[9 1 0.5; 0 8.5 -0.7; 0 0 8]
        X0 = ToolOracles.uniform(rng, C, 12)
        fs = [ToolOracles.frame(X0 .+ T(0.2) .* randn(rng, T, 3, 12), fill(29, 12), C) for _ in 1:9]
        r, g, n, h = zeros(T, 10), zeros(T, 10), zeros(T, 10), zeros(Int, 10)
        @test rdf_in!(r, g, n, h, fs) == rdf(fs, 4.0, 10)
        @test_noalloc rdf_in!(r, g, n, h, fs)
        l, m, ring = zeros(T, 3), zeros(T, 3, 3), zeros(T, 3, 12, 3)
        u, p = similar(X0), similar(X0)
        @test msd_in!(l, m, u, p, ring, fs) == msd(fs; max_lag = 2, nblocks = 3, dt = 2.0)
        @test_noalloc msd_in!(l, m, u, p, ring, fs)
        o = zeros(T, 9)
        @test rmsd_in!(o, fs, X0) == rmsd(fs, X0; align = true)
        @test rmsd!(zeros(T, 9), fs, X0) == rmsd(fs, X0)
        @test_noalloc rmsd_in!(o, fs, X0)
        a = T(3.61)
        cell = Matrix{T}(2a * [1 0 0; 0 1 0; 0 0 1])
        s = zeros(T, 3, 32)
        @test sites_in!(s, cell, a) == reference_sites(cell, a, 2)
        @test_noalloc sites_in!(s, cell, a)
        @test_throws "pureatomistic: r (Base.OneTo(9),), g (Base.OneTo(10),), n (Base.OneTo(10),) and hist (Base.OneTo(10),) must have equal axes" rdf!(
            zeros(T, 9), g, n, h, fs, 4.0
        )
        @test_throws "pureatomistic: msd_A2 has 4 rows for 3 lags" msd!(l, zeros(T, 4, 3), u, p, ring, fs; dt = 2.0)
        @test_throws "pureatomistic: ring has size (3, 12, 2), expected (3, 12, 3)" msd!(l, m, u, p, zeros(T, 3, 12, 2), fs; dt = 2.0)
        @test_throws "pureatomistic: unwrapped (Base.OneTo(3), Base.OneTo(11)) and previous (Base.OneTo(3), Base.OneTo(12)) must have the axes (Base.OneTo(3), Base.OneTo(12)) of the positions" msd!(
            l, m, zeros(T, 3, 11), p, ring, fs; dt = 2.0
        )
        @test_throws "pureatomistic: out has 8 entries for 9 frames" rmsd!(zeros(T, 8), fs, X0)
        @test_throws "pureatomistic: sites must have axes (Base.OneTo(3), Base.OneTo(32)), got (Base.OneTo(3), Base.OneTo(31))" reference_sites!(
            zeros(T, 3, 31), cell, a, 2
        )
    end
end

@testitem "3 × 3 singular value decomposition" tags = [:analysis] begin
    using PureAtomistic.Analysis: _svd3
    using LinearAlgebra, Random
    mat(t) = [t[1] t[4] t[7]; t[2] t[5] t[8]; t[3] t[6] t[9]]
    rng = Xoshiro(17)
    for T in (Float64, Float32), trial in 1:300
        kind = trial % 4
        A = kind == 0 ? randn(rng, T, 3, 3) :
            kind == 1 ? randn(rng, T, 3, 2) * randn(rng, T, 2, 3) :       # rank 2
            kind == 2 ? randn(rng, T, 3, 1) * randn(rng, T, 1, 3) :       # rank 1
            Matrix(T(2) * qr(randn(rng, T, 3, 3)).Q)                    # equal singular values
        H = Tuple(A)
        U, σ, V = _svd3(H)
        Um, Vm = mat(U), mat(V)
        # Measured over 10⁵ matrices of these kinds in each precision: reconstruction within
        # 7.7 eps·σ₁ and orthonormality within 11.5 eps; the bounds are about 8 and 6 times those.
        tol = 64 * eps(T) * max(σ[1], floatmin(T))
        @test σ[1] >= σ[2] >= σ[3] >= 0
        @test norm(Um' * Um - I) <= 64 * eps(T)
        @test norm(Vm' * Vm - I) <= 64 * eps(T)
        # u₃ is u₁ × u₂, which is the third left singular vector up to its sign.
        s = Um[:, 3]' * A * Vm[:, 3] < 0 ? -one(T) : one(T)
        @test norm(Um * Diagonal([σ[1], σ[2], s * σ[3]]) * Vm' - A) <= tol
        @test isapprox(collect(σ), svdvals(A); atol = tol)
    end
    # The zero matrix: every singular value is zero, and U and V are still orthonormal.
    U, σ, V = _svd3(ntuple(_ -> 0.0, 9))
    @test σ == (0.0, 0.0, 0.0)
    @test mat(U)' * mat(U) ≈ I
end

@testitem "rmsd: planar and collinear frames" tags = [:analysis] setup = [ToolOracles] begin
    using PureAtomistic.Analysis, LinearAlgebra, Random
    rng = Xoshiro(23)
    R = ToolOracles.rotation([0.3, 1, -2], 0.7)
    t = [2.0, -1.0, 0.5]
    for Y in (
            [randn(rng, 2, 9); zeros(1, 9)],                       # planar
            [1.0, -2.0, 0.5] * randn(rng, 1, 9),                   # collinear
            [0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 1.0 1.0 1.0 1.0],   # planar, off the origin, two atoms coincide
            reshape([1.0, 2.0, 3.0], 3, 1),                        # one atom
        )
        n = size(Y, 2)
        X = R * Y .+ t
        Z = R * (Diagonal([1.0, -1.0, 1.0]) * Y) .+ t
        fr = [ToolOracles.frame(P, fill(29, n), Matrix(10.0I, 3, 3)) for P in (X, Z)]
        al = rmsd(fr, Y; align = true)
        # A rigid motion of the frame is undone exactly up to rounding (coordinates up to 10 Å).
        @test al[1] < 1.5e-13
        # A planar or collinear mirror image is a proper rotation of itself, so it is superposed too.
        @test al[2] < 1.5e-13
    end
end

# The Kabsch rotation is proper whatever order the singular values come out of the Jacobi sweeps:
# for a mirror image the best proper rotation leaves a residual, Horn's minimum over rotations.
@testitem "rmsd: the superposition is a proper rotation" tags = [:analysis] setup = [ToolOracles] begin
    using PureAtomistic.Analysis, LinearAlgebra, Random
    using PureAtomistic.Analysis: _kabsch
    rng = Xoshiro(31)
    for trial in 1:200
        n = rand(rng, 3:10)
        Y = 4 .* randn(rng, 3, n)
        X = ToolOracles.rotation(randn(rng, 3), 2π * rand(rng)) * (Diagonal([1.0, 1.0, -1.0]) * Y) .+ randn(rng, 3)
        R, _, _ = _kabsch(X, Y, Float64)
        @test det([R[1] R[4] R[7]; R[2] R[5] R[8]; R[3] R[6] R[9]]) ≈ 1
        fr = [ToolOracles.frame(X, fill(29, n), Matrix(10.0I, 3, 3))]
        # The squared deviations n·rmsd² = G - 2λ agree to the rounding of G, the summed squared
        # centered coordinates of both frames.
        G = sum(abs2, X .- sum(X; dims = 2) / n) + sum(abs2, Y .- sum(Y; dims = 2) / n)
        @test n * abs(rmsd(fr, Y; align = true)[1]^2 - ToolOracles.horn_rmsd(X, Y)^2) <= 64 * eps() * G
    end
end
