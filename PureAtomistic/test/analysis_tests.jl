@testmodule VacancyOracle begin
    # A lattice random walk of the vacancy on the fcc site graph, written without the tracker:
    # nearest neighbors come from plain minimum-image distances in an orthogonal cubic cell, and
    # each frame places every atom on its site plus Gaussian noise and a rigid drift.
    using Random

    export fcc_cell, neighbor_table, frame_from, walk!

    fcc_cell(T, a, nrep) = Matrix{T}(T(a * nrep) * [1 0 0; 0 1 0; 0 0 1])

    function neighbor_table(sites, L, a)
        n = size(sites, 2)
        nn = a / sqrt(2)
        table = [Int[] for _ in 1:n]
        for i in 1:n, j in 1:n
            i == j && continue
            d = sites[:, i] - sites[:, j]
            d -= L * round.(d / L)
            abs(sqrt(sum(abs2, d)) - nn) < 1.0e-3 * a && push!(table[i], j)
        end
        return table
    end

    # Positions of the atoms when the vacancy is at `vac` and `atom_site[k]` is the site of atom k.
    function frame_from(T, sites, atom_site, rng, noise, drift)
        P = Matrix{T}(undef, 3, length(atom_site))
        for (k, s) in enumerate(atom_site)
            for c in 1:3
                P[c, k] = sites[c, s] + noise * randn(rng) + drift[c]
            end
        end
        return P
    end

    # Moves the vacancy from `vac` to the site `w`: the atom on `w` takes the vacancy's place.
    function move_vacancy!(atom_site, vac, w)
        k = findfirst(==(w), atom_site)
        atom_site[k] = vac
        return w
    end

    # Per-frame hop probability `p`; calls `f(positions, t)` for each of `nframes` frames and
    # returns the number of hops made.
    function walk!(f, T, sites, L, a, rng, p, dt, nframes; noise = 0.05, drift = (0.0, 0.0, 0.0), vac = 1)
        table = neighbor_table(sites, L, a)
        atom_site = [s for s in 1:size(sites, 2) if s != vac]
        nhops = 0
        for frame in 1:nframes
            if frame > 1 && rand(rng) < p
                vac = move_vacancy!(atom_site, vac, rand(rng, table[vac]))
                nhops += 1
            end
            f(frame_from(T, sites, atom_site, rng, noise, drift), T(frame * dt))
        end
        return nhops
    end

    # Frames for a prescribed vacancy sequence `seq` (site per frame), no noise.
    function sequence_frames(T, sites, seq)
        vac = seq[1]
        atom_site = [s for s in 1:size(sites, 2) if s != vac]
        frames = Matrix{T}[]
        for s in seq
            s != vac && (vac = move_vacancy!(atom_site, vac, s))
            push!(frames, frame_from(T, sites, atom_site, Random.default_rng(), zero(T), (0, 0, 0)))
        end
        return frames
    end
    export sequence_frames
end

@testitem "reference_sites builds the fcc lattice" tags = [:analysis] setup = [VacancyOracle] begin
    using PureAtomistic.Analysis
    a, nrep = 3.61, 3
    for T in (Float64, Float32)
        cell = fcc_cell(T, a, nrep)
        sites = reference_sites(cell, a, nrep)
        @test eltype(sites) === T
        @test size(sites) == (3, 4 * nrep^3)
        # Distinct sites: the closest pair is the nearest-neighbor distance a/sqrt(2) (exact
        # up to the rounding of a few sums of size a*nrep, so 8 eps(T) * a * nrep).
        L = T(a * nrep)
        mic(d) = d - L * round.(d / L)
        dmin = minimum(
            sqrt(sum(abs2, mic(sites[:, i] - sites[:, j]))) for i in 1:size(sites, 2) for j in 1:(i - 1)
        )
        @test dmin ≈ a / sqrt(2) atol = 8 * eps(T) * a * nrep
        # 12 nearest neighbors of every site.
        table = neighbor_table(sites, L, a)
        @test all(length.(table) .== 12)
    end
end

@testitem "reference_sites rejects inconsistent input" tags = [:analysis] begin
    using PureAtomistic.Analysis
    @test_throws "pureatomistic: reference_sites needs the cubic cell nrep*a" reference_sites(
        [10.0 0 0; 0 10 0; 0 0 10], 3.61, 3
    )
    @test_throws "pureatomistic: nrep must be at least 1" reference_sites([1.0 0 0; 0 1 0; 0 0 1], 1.0, 0)
    @test_throws "pureatomistic: cell must be 3x3" reference_sites(ones(2, 3), 3.61, 1)
end

@testitem "wigner_seitz_occupancy! counts atoms on their nearest sites" tags = [:analysis] setup = [VacancyOracle] begin
    using PureAtomistic.Analysis
    using Random
    a, nrep = 3.61, 3
    for T in (Float64, Float32)
        cell = fcc_cell(T, a, nrep)
        sites = reference_sites(cell, a, nrep)
        n = size(sites, 2)
        vac = 17
        P = sites[:, [1:(vac - 1); (vac + 1):n]]
        rng = Xoshiro(1)
        P .+= T(0.1) .* randn(rng, T, size(P))
        # Wrapped positions (shifted by a whole cell vector) must give the same counts.
        Pw = copy(P)
        Pw[1, 3] += T(a * nrep)
        Pw[2, 5] -= T(2 * a * nrep)
        occ = zeros(Int, n)
        @test wigner_seitz_occupancy!(occ, P, sites, cell) === occ
        @test occ == [i == vac ? 0 : 1 for i in 1:n]
        occ2 = zeros(Int, n)
        wigner_seitz_occupancy!(occ2, Pw, sites, cell)
        @test occ2 == occ
    end
end

@testitem "wigner_seitz_occupancy! is generic over axes and views" tags = [:analysis] setup = [VacancyOracle] begin
    using PureAtomistic.Analysis
    using OffsetArrays
    a, nrep = 3.61, 2
    cell = fcc_cell(Float64, a, nrep)
    sites = reference_sites(cell, a, nrep)
    n = size(sites, 2)
    P = sites[:, 2:n] .+ 0.05
    occ = zeros(Int, n)
    wigner_seitz_occupancy!(occ, P, sites, cell)
    # Offset atom and site axes: the occupancy follows the axes of the sites.
    Po = OffsetArray(P, 1:3, 11:(10 + n - 1))
    So = OffsetArray(sites, 1:3, 0:(n - 1))
    occo = OffsetArray(zeros(Int, n), 0:(n - 1))
    wigner_seitz_occupancy!(occo, Po, So, cell)
    @test axes(occo, 1) == axes(So, 2)
    @test collect(occo) == occ
    # Views of larger arrays.
    big = vcat(P, zeros(2, n - 1))
    bigsites = vcat(sites, zeros(2, n))
    occv = zeros(Int, n)
    wigner_seitz_occupancy!(occv, view(big, 1:3, :), view(bigsites, 1:3, :), cell)
    @test occv == occ
    # Mismatched or ill-shaped input fails with a message.
    @test_throws "pureatomistic: occ axes" wigner_seitz_occupancy!(zeros(Int, n - 1), P, sites, cell)
    @test_throws "pureatomistic: positions must have 3 rows" wigner_seitz_occupancy!(
        occ, OffsetArray(P, 0:2, 1:(n - 1)), sites, cell
    )
    @test_throws "pureatomistic: cell is singular" wigner_seitz_occupancy!(occ, P, sites, zeros(3, 3))
    @test_throws "pureatomistic: no reference sites" wigner_seitz_occupancy!(
        zeros(Int, 0), P, zeros(3, 0), cell
    )
end

@testitem "vacancy tracker follows a lattice random walk" tags = [:analysis] setup = [VacancyOracle] begin
    using PureAtomistic.Analysis
    using Random
    a, nrep = 3.61, 3
    for T in (Float64, Float32)
        cell = fcc_cell(T, a, nrep)
        sites = reference_sites(cell, a, nrep)
        tr = VacancyTracker(sites; cell, window = 1, dwell = 1, capacity = 600)
        # Rigid drift of 1.7 A along x, 94% of the distance to the nearest Voronoi boundary
        # (a/2 = 1.805 A), plus 0.05 A noise: only the drift removal keeps every atom on its site.
        nh = walk!(
            (P, t) -> update!(tr, P, t), T, sites, a * nrep, a, Xoshiro(7), 0.05, 10, 600;
            drift = (1.7, 0.0, 0.0)
        )
        @test tr.nframes == 600
        @test tr.nhops == nh
        @test tr.nhops > 0
        # The unwrapped track is the sum of the nearest-neighbor displacements: each confirmed
        # hop moves it by a/sqrt(2).
        steps = [sqrt(sum(abs2, tr.R[:, i + 1] - tr.R[:, i])) for i in 1:(tr.nframes - 1)]
        @test sum(>(0), steps) == nh
        @test all(s -> s == 0 || isapprox(s, a / sqrt(2); atol = 1.0e-3), steps)
    end
end

@testitem "walk statistics recover the hop rate and diffusion coefficient" tags = [:analysis] setup = [VacancyOracle] begin
    using PureAtomistic.Analysis
    using Random
    a, nrep = 3.61, 3
    p, dt, nframes = 0.05, 10.0, 8000
    Γ_true = p / dt
    λ2 = a^2 / 2
    # The hop count is Poisson and the displacement a sum of independent steps, so the
    # expectation of the squared displacement over a lag of n frames is exactly n p λ²,
    # D = Γ λ² / 6. A statistical bound: 4 standard errors of each estimator.
    cell = fcc_cell(Float64, a, nrep)
    sites = reference_sites(cell, a, nrep)
    tr = VacancyTracker(sites; cell, window = 1, dwell = 1, capacity = nframes)
    nh = walk!((P, t) -> update!(tr, P, t), Float64, sites, a * nrep, a, Xoshiro(2024), p, dt, nframes)
    @test tr.nhops == nh
    r = hop_rate(tr)
    # Observed |z| = 0.30 with this seed (394 hops).
    @test abs(r.rate - Γ_true) <= 4 * r.σ
    d = msd_diffusion(tr)
    D_true = Γ_true * λ2 / 6
    # Observed |z| = 0.44 with this seed.
    @test abs(d.D - D_true) <= 4 * d.σ
    # The hop-counting route D = Γ λ² / 6 agrees with the displacement route within both errors.
    D_hop = r.rate * λ2 / 6
    @test abs(D_hop - d.D) <= 4 * hypot(d.σ, r.σ * λ2 / 6)
end

@testitem "msd_diffusion fits the slope of the squared displacement" tags = [:analysis] setup = [VacancyOracle] begin
    using PureAtomistic.Analysis
    a, nrep = 3.61, 2
    for T in (Float64, Float32)
        cell = fcc_cell(T, a, nrep)
        sites = reference_sites(cell, a, nrep)
        tr = VacancyTracker(sites; cell, window = 1, dwell = 1, capacity = 4)
        # Two blocks of two frames 1.5 time units apart. Block 1 moves by 3 (squared 9) and block 2
        # by 3 sqrt(2) (squared 18), so by |dR|^2 = 6 D tau the block estimates are D = 9 / 9 = 1
        # and D = 18 / 9 = 2: mean 3/2, standard error of the mean sqrt(0.5 / 1 / 2) = 1/2.
        tr.nframes = 4
        tr.t .= T[0, 1.5, 3, 4.5]
        tr.R[:, 1] .= 0
        tr.R[:, 2] .= T[3, 0, 0]
        tr.R[:, 3] .= 0
        tr.R[:, 4] .= T[3, 3, 0]
        d = msd_diffusion(tr; nblocks = 2, maxlag = 1)
        @test d.D ≈ T(1.5) rtol = 4 * eps(T)
        @test d.σ ≈ T(0.5) rtol = 4 * eps(T)
    end
end

@testitem "the dwell filter suppresses recrossings" tags = [:analysis] setup = [VacancyOracle] begin
    using PureAtomistic.Analysis
    a, nrep = 3.61, 2
    for T in (Float64, Float32)
        cell = fcc_cell(T, a, nrep)
        sites = reference_sites(cell, a, nrep)
        A, B = 1, 2
        # The vacancy sits on A, visits B for one frame, returns, then moves to B for good.
        seq = [fill(A, 5); B; fill(A, 5); fill(B, 5)]
        frames = sequence_frames(T, sites, seq)
        function track(; window, dwell)
            tr = VacancyTracker(sites; cell, window, dwell, capacity = 40)
            for (i, P) in enumerate(frames)
                update!(tr, P, T(i))
            end
            return tr
        end
        @test track(window = 1, dwell = 1).nhops == 3
        # Three hops in the 15 time units between the first and last frame: rate 3/15 with the
        # Poisson error sqrt(3)/15 (counting statistics, computed by hand).
        r = hop_rate(track(window = 1, dwell = 1))
        @test r.rate ≈ 3 / 15 rtol = 4 * eps(T)
        @test r.σ ≈ sqrt(3) / 15 rtol = 4 * eps(T)
        tr = track(window = 1, dwell = 2)
        @test tr.nhops == 1
        # A hop is confirmed at the second frame on B and displaces the track by the
        # minimum-image vector between the sites.
        @test tr.hop_t[1] == T(12 + 1)
        @test Vector(tr.R[:, tr.nframes]) ≈ Vector(sites[:, B] - sites[:, A]) atol = 8 * eps(T) * a
        @test tr.vsite[tr.nframes] == B
        @test track(window = 1, dwell = 6).nhops == 0
        # Averaging over three frames also removes a one-frame excursion.
        @test track(window = 3, dwell = 1).nhops == 1
    end
end

@testitem "a Frenkel pair fails with the frame index" tags = [:analysis] setup = [VacancyOracle] begin
    using PureAtomistic.Analysis
    a, nrep = 3.61, 2
    cell = fcc_cell(Float64, a, nrep)
    sites = reference_sites(cell, a, nrep)
    n = size(sites, 2)
    P = sites[:, 2:n]
    tr = VacancyTracker(sites; cell, window = 1, dwell = 1, capacity = 4)
    update!(tr, P, 1.0)
    # Atom 3 (site 4) moves onto site 5: site 4 and site 1 are empty, site 5 is doubly occupied.
    Q = copy(P)
    Q[:, 3] = sites[:, 5]
    @test_throws "pureatomistic: occupancy is not one vacancy at frame 2: 2 empty and 1 multiply occupied sites" update!(
        tr, Q, 2.0
    )
    @test tr.nframes == 1
end

@testitem "update! and the analysis functions reject bad input" tags = [:analysis] setup = [VacancyOracle] begin
    using PureAtomistic.Analysis
    a, nrep = 3.61, 2
    cell = fcc_cell(Float64, a, nrep)
    sites = reference_sites(cell, a, nrep)
    n = size(sites, 2)
    P = sites[:, 2:n]
    tr = VacancyTracker(sites; cell, window = 1, dwell = 1, capacity = 2)
    @test_throws "pureatomistic: no hops were recorded" (update!(tr, P, 1.0); update!(tr, P, 2.0); hop_rate(tr))
    tr2 = VacancyTracker(sites; cell, window = 1, dwell = 1)
    update!(tr2, P, 1.0)
    @test_throws "pureatomistic: frame times must increase" update!(tr2, P, 1.0)
    @test_throws "pureatomistic: tracker has 31 atoms, frame has 30" update!(tr2, P[:, 1:30], 2.0)
    @test_throws "pureatomistic: tracker capacity 2 is exhausted" update!(tr, P, 3.0)
    @test_throws "pureatomistic: hop_rate needs at least 2 frames" hop_rate(tr2)
    @test_throws "pureatomistic: msd_diffusion has 1 frames" msd_diffusion(tr2)
    @test_throws "pureatomistic: msd_diffusion needs nblocks >= 2" msd_diffusion(tr2; nblocks = 1)
    @test_throws "pureatomistic: window must be at least 1" VacancyTracker(sites; cell, window = 0, dwell = 1)
    @test_throws "pureatomistic: dwell must be at least 1" VacancyTracker(sites; cell, window = 1, dwell = 0)
    @test_throws "pureatomistic: capacity must be at least 1" VacancyTracker(sites; cell, window = 1, dwell = 1, capacity = 0)
    @test_throws "pureatomistic: a vacancy tracker needs at least 2 sites" VacancyTracker(sites[:, 1:1]; cell, window = 1, dwell = 1)
    @test_throws "pureatomistic: positions must have 3 rows" update!(tr2, P[1:2, :], 2.0)
end

@testitem "update! reads offset and view frames like plain ones" tags = [:analysis] setup = [VacancyOracle] begin
    using PureAtomistic.Analysis
    using OffsetArrays
    using Random
    a, nrep = 3.61, 2
    cell = fcc_cell(Float64, a, nrep)
    sites = reference_sites(cell, a, nrep)
    frames = Matrix{Float64}[]
    walk!((P, t) -> push!(frames, copy(P)), Float64, sites, a * nrep, a, Xoshiro(3), 0.2, 1.0, 60)
    run(wrap) = begin
        tr = VacancyTracker(sites; cell, window = 2, dwell = 1, capacity = 60)
        for (i, P) in enumerate(frames)
            update!(tr, wrap(P), Float64(i))
        end
        tr
    end
    ref = run(identity)
    off = run(P -> OffsetArray(P, 1:3, 5:(4 + size(P, 2))))
    vw = run(P -> view(vcat(P, P), 1:3, :))
    for tr in (off, vw)
        @test tr.R == ref.R
        @test tr.vsite == ref.vsite
        @test tr.nhops == ref.nhops
    end
end

@testitem "window averaging uses minimum-image displacements" tags = [:analysis] setup = [VacancyOracle] begin
    using PureAtomistic.Analysis
    a, nrep = 3.61, 2
    cell = fcc_cell(Float64, a, nrep)
    sites = reference_sites(cell, a, nrep)
    n = size(sites, 2)
    P = sites[:, 2:n]
    L = a * nrep
    # One atom sits on a site at the cell boundary and is reported on opposite periodic images in
    # successive frames; a plain average would place it mid-cell.
    k = findfirst(j -> sites[1, j + 1] == 0.0, 1:(n - 1))
    tr = VacancyTracker(sites; cell, window = 4, dwell = 1, capacity = 10)
    for f in 1:6
        Q = copy(P)
        Q[1, k] += iseven(f) ? L : -L
        update!(tr, Q, Float64(f))
    end
    @test tr.nhops == 0
    @test all(==(1), tr.vsite[1:tr.nframes])
end

@testitem "arrhenius_fit recovers a chosen activation energy" tags = [:analysis] begin
    using PureAtomistic.Analysis
    using Random
    kB = PureAtomistic.Systems.KB_EV_PER_K
    @test kB == 1.380649e-23 / 1.602176634e-19
    E, Γ0 = 0.7, 6.0e13
    temps = collect(800.0:100.0:1300.0)
    rates = Γ0 .* exp.(-E ./ (kB .* temps))
    errors = 0.05 .* rates
    fit = arrhenius_fit(temps, rates, errors)
    # Noise-free data lie on the line, so the fit is exact up to rounding. Errors
    # observed: 1.0e-14 eV in E_m and 1.2e-13 relative in the prefactor; tolerances 100 times those.
    @test fit.E_m ≈ E atol = 1.0e-12
    @test fit.prefactor ≈ Γ0 rtol = 1.2e-11
    # The standard error follows from the weights alone: sqrt(1/Σ w (x - x̄)²).
    x = 1 ./ (kB .* temps)
    @test fit.σ ≈ sqrt(1 / (sum(x .^ 2) / 0.05^2 - sum(x)^2 / 0.05^2 / length(x))) rtol = 1.0e-9
    # Noisy data: rates scattered by their stated errors recover E within 4 standard errors
    # (observed |z| = 1.7 with this seed).
    rng = Xoshiro(11)
    noisy = rates .* (1 .+ 0.05 .* randn(rng, length(rates)))
    nfit = arrhenius_fit(temps, noisy, 0.05 .* noisy)
    @test abs(nfit.E_m - E) <= 4 * nfit.σ
    # Float32 inputs give Float32 results; error observed 5.9e-6 eV, tolerance 160 eps(Float32) = 1.9e-5.
    f32 = arrhenius_fit(Float32.(temps), Float32.(rates), Float32.(errors))
    @test f32.E_m isa Float32 && f32.σ isa Float32 && f32.prefactor isa Float32
    @test f32.E_m ≈ E atol = 160 * eps(Float32)
    # A single weighted pair is the two-point line through the data.
    two = arrhenius_fit([1000.0, 1200.0], rates[[3, 5]], errors[[3, 5]])
    @test two.E_m ≈ E atol = 1.0e-12
end

@testitem "arrhenius_fit errors of the prefactor and the covariance" tags = [:analysis] begin
    using PureAtomistic.Analysis
    using LinearAlgebra
    kB = PureAtomistic.Systems.KB_EV_PER_K
    # Hop counts n in times t (ps) at four temperatures; rates n/t with the Poisson errors √n/t,
    # so the weights (rate/error)² of the fit are the counts.
    temps = [900.0, 980.0, 1080.0, 1200.0]
    n = [100.0, 100.0, 100.0, 200.0]
    t = [11780.0, 5650.0, 2630.0, 2460.0]
    rates, errors = n ./ t, sqrt.(n) ./ t
    fit = arrhenius_fit(temps, rates, errors)
    # Oracle: the covariance matrix inv(XᵀWX) of the generalized least-squares estimate of
    # (ln Γ₀, -E_m), from the design matrix X = [1 x] and W = Diagonal(n), and the estimate
    # itself, (XᵀWX) \ XᵀWy, solved by LinearAlgebra instead of the centered sums of the fit.
    x = 1 ./ (kB .* temps)
    X = [ones(4) x]
    W = Diagonal(n)
    C = inv(X' * W * X)
    β = (X' * W * X) \ (X' * W * log.(rates))
    # Largest relative error observed against the oracle: 6.7e-15 (ln Γ₀; the normal matrix has
    # condition number 9.6e3); tolerance 15 times that.
    @test fit.E_m ≈ -β[2] rtol = 1.0e-13
    @test log(fit.prefactor) ≈ β[1] rtol = 1.0e-13
    @test fit.σ ≈ sqrt(C[2, 2]) rtol = 1.0e-13
    @test fit.σ_ln_prefactor ≈ sqrt(C[1, 1]) rtol = 1.0e-13
    # E_m = -slope, so its covariance with ln Γ₀ is the negative of the slope-intercept entry.
    @test fit.cov_E_ln_prefactor ≈ -C[1, 2] rtol = 1.0e-13
    r = log.(rates) .- X * β
    # Observed relative error 1.8e-14; tolerance 11 times that.
    @test fit.chi2 ≈ sum(n .* r .^ 2) rtol = 2.0e-13
    # Data on an Arrhenius line give χ² = 0 up to rounding (observed 9.3e-25, tolerance 11 times
    # that). Their weights equal the counts up to the rounding of one division and one square
    # each, so the errors match those of the counts within 40 eps.
    E, Γ0 = 0.7, 6.0e13
    exact = Γ0 .* exp.(-E ./ (kB .* temps))
    fe = arrhenius_fit(temps, exact, exact ./ sqrt.(n))
    @test fe.chi2 <= 1.0e-23
    @test fe.σ ≈ fit.σ rtol = 40 * eps()
    @test fe.σ_ln_prefactor ≈ fit.σ_ln_prefactor rtol = 40 * eps()
    # OffsetArray and view inputs visit the points in the same order, so the result is identical.
    using OffsetArrays
    o(v) = OffsetArray(v, -2:1)
    @test arrhenius_fit(o(temps), o(rates), o(errors)) == fit
    @test arrhenius_fit(view([temps; temps], 1:4), view(rates, :), view([errors errors], :, 2)) == fit
    @test_throws DimensionMismatch arrhenius_fit(o(temps), rates, errors)
    # Float32 inputs give Float32 results.
    f32 = arrhenius_fit(Float32.(temps), Float32.(rates), Float32.(errors))
    @test f32.σ_ln_prefactor isa Float32 && f32.cov_E_ln_prefactor isa Float32 && f32.chi2 isa Float32
end

@testitem "arrhenius_fit errors cover the true values" tags = [:analysis] begin
    using PureAtomistic.Analysis
    using Random, Statistics
    kB = PureAtomistic.Systems.KB_EV_PER_K
    E, lnΓ0 = 0.7, log(6.0e13 / 1.0e12)  # rates per ps
    temps = [900.0, 980.0, 1080.0, 1200.0]
    x = 1 ./ (kB .* temps)
    Γ = exp.(lnΓ0 .- E .* x)
    M = 20_000
    # Known variance: ln rate is the line plus Gaussian noise of standard deviation s, and the errors
    # passed are rate·s, so the weights are exactly 1/s². The estimates are then normal with the
    # stated covariance, and an interval of ±1.96σ covers the true value with probability 0.95.
    s = [0.1, 0.1, 0.1, 0.07]
    rng = Xoshiro(20261108)
    cover = zeros(Int, 2)
    Es, As, chi = zeros(M), zeros(M), zeros(M)
    # The weights do not depend on the noise, so neither do the stated errors.
    f0 = arrhenius_fit(temps, Γ, Γ .* s)
    for k in 1:M
        r = Γ .* exp.(s .* randn(rng, 4))
        f = arrhenius_fit(temps, r, r .* s)
        Es[k], As[k], chi[k] = f.E_m, log(f.prefactor), f.chi2
        cover[1] += abs(f.E_m - E) <= 1.96 * f.σ
        cover[2] += abs(log(f.prefactor) - lnΓ0) <= 1.96 * f.σ_ln_prefactor
    end
    # Binomial standard error of a coverage fraction near 0.95 over M runs: 0.0015; bound 4 of them.
    # Observed 0.9497 (E_m) and 0.9508 (ln Γ₀).
    se = sqrt(0.95 * 0.05 / M)
    @test all(abs.(cover ./ M .- 0.95) .<= 4se)
    # The sample variances and covariance of the estimates against the stated ones. The relative
    # standard error of a sample variance of M normal draws is sqrt(2/M) = 0.010; bound 4 of them.
    # Observed deviations 0.0039, 0.0030 and 0.0035 (correlation 0.9935 stated); χ² has mean 2
    # (observed 2.018, standard error sqrt(2·2/M) = 0.014).
    @test var(Es) / f0.σ^2 ≈ 1 atol = 4 * sqrt(2 / M)
    @test var(As) / f0.σ_ln_prefactor^2 ≈ 1 atol = 4 * sqrt(2 / M)
    ρ = f0.cov_E_ln_prefactor / (f0.σ * f0.σ_ln_prefactor)
    # Standard error of a sample covariance of normal pairs: sqrt((1 + ρ²)/M) relative to σσ.
    @test cov(Es, As) / (f0.σ * f0.σ_ln_prefactor) ≈ ρ atol = 4 * sqrt((1 + ρ^2) / M)
    @test mean(chi) ≈ 2 atol = 4 * sqrt(4 / M)

    # Poisson hop counts at the campaign's targets (expected 100, 100, 100 and 200 hops): the weights
    # are the observed counts, so the errors hold to first order in 1/n. Each count is the number
    # of unit-rate exponential gaps that fit in the expected count. Observed coverage 0.9518 and
    # 0.9498; bound as above.
    λ = [100.0, 100.0, 100.0, 200.0]
    t = λ ./ Γ
    function poisson(rng, λ)
        k, g = 0, randexp(rng)
        while g <= λ
            k += 1
            g += randexp(rng)
        end
        return k
    end
    cover .= 0
    for k in 1:M
        c = [poisson(rng, λ[i]) for i in 1:4]
        f = arrhenius_fit(temps, c ./ t, sqrt.(c) ./ t)
        cover[1] += abs(f.E_m - E) <= 1.96 * f.σ
        cover[2] += abs(log(f.prefactor) - lnΓ0) <= 1.96 * f.σ_ln_prefactor
    end
    @test all(abs.(cover ./ M .- 0.95) .<= 4se)
end

@testitem "arrhenius_fit rejects bad input" tags = [:analysis] begin
    using PureAtomistic.Analysis
    t, r, e = [900.0, 1000.0], [1.0, 2.0], [0.1, 0.1]
    @test_throws "pureatomistic: arrhenius_fit needs at least 2 points" arrhenius_fit([900.0], [1.0], [0.1])
    @test_throws "pureatomistic: arrhenius_fit needs positive temperatures" arrhenius_fit([0.0, 1000.0], r, e)
    @test_throws "pureatomistic: arrhenius_fit needs positive rates" arrhenius_fit(t, [0.0, 2.0], e)
    @test_throws "pureatomistic: arrhenius_fit needs positive errors" arrhenius_fit(t, r, [0.1, 0.0])
    @test_throws "pureatomistic: arrhenius_fit needs at least 2 distinct temperatures" arrhenius_fit([900.0, 900.0], r, e)
    @test_throws DimensionMismatch arrhenius_fit(t, [1.0], e)
end

@testitem "analysis steps allocate nothing" tags = [:analysis, :checks] setup = [VacancyOracle] begin
    using PureAtomistic.Analysis
    using StrictModeTest, StrictMode
    a, nrep = 3.61, 2
    for T in (Float64, Float32)
        cell = fcc_cell(T, a, nrep)
        sites = reference_sites(cell, a, nrep)
        n = size(sites, 2)
        P = sites[:, 2:n]
        occ = zeros(Int, n)
        tr = VacancyTracker(sites; cell, window = 3, dwell = 2, capacity = 16)
        @noinline f!(occ, P, sites, cell) = wigner_seitz_occupancy!(occ, P, sites, cell)
        @noinline g!(tr, P, t) = update!(tr, P, t)
        f!(occ, P, sites, cell)
        @test_noalloc f!(occ, P, sites, cell)
        g!(tr, P, T(1))
        @test_noalloc g!(tr, P, T(2))
    end
end

@testitem "TrimCheck: analysis" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureAtomistic.Analysis
            new_tracker64(s, c) = VacancyTracker(s; cell = c, window = 2, dwell = 2)
            new_tracker32(s, c) = VacancyTracker(s; cell = c, window = 2, dwell = 2)
            msd64(tr) = msd_diffusion(tr)
            msd32(tr) = msd_diffusion(tr)
        end,
        reference_sites(Matrix{Float64}, Float64, Int),
        reference_sites(Matrix{Float32}, Float32, Int),
        wigner_seitz_occupancy!(Vector{Int}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}),
        wigner_seitz_occupancy!(Vector{Int}, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}),
        new_tracker64(Matrix{Float64}, Matrix{Float64}),
        new_tracker32(Matrix{Float32}, Matrix{Float32}),
        update!(VacancyTracker{Float64}, Matrix{Float64}, Float64),
        update!(VacancyTracker{Float32}, Matrix{Float32}, Float32),
        hop_rate(VacancyTracker{Float64}),
        hop_rate(VacancyTracker{Float32}),
        msd64(VacancyTracker{Float64}),
        msd32(VacancyTracker{Float32}),
        arrhenius_fit(Vector{Float64}, Vector{Float64}, Vector{Float64}),
        arrhenius_fit(Vector{Float32}, Vector{Float32}, Vector{Float32}),
    )
end

@testitem "analysis return types are concrete" tags = [:checks] begin
    using PureAtomistic.Analysis
    for T in (Float64, Float32)
        @test Base.return_types(hop_rate, (VacancyTracker{T},)) == [@NamedTuple{rate::T, σ::T}]
        @test Base.return_types(arrhenius_fit, (Vector{T}, Vector{T}, Vector{T})) ==
            [@NamedTuple{E_m::T, σ::T, prefactor::T, σ_ln_prefactor::T, cov_E_ln_prefactor::T, chi2::T}]
        @test Base.return_types(update!, (VacancyTracker{T}, Matrix{T}, T)) == [Nothing]
        @test Base.return_types(reference_sites, (Matrix{T}, T, Int)) == [Matrix{T}]
    end
end
