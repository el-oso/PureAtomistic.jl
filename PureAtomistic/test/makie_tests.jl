@testmodule MakieFixtures begin
    using PureAtomistic
    using PureAtomistic.IO: TrajectoryWriter, write_frame, Frame, write_extxyz
    using PureAtomistic.Analysis: VacancyTracker, update!, reference_sites

    export fcc_cell_for, hopping_tracker, write_ptraj, write_xyz, is_png, is_mp4

    fcc_cell_for(a, nrep) = Matrix{Float64}(a * nrep * [1 0 0; 0 1 0; 0 0 1])

    # A vacancy that sits on site 5 (away from the origin, so the first position is nonzero), hops to
    # 6, then to 7: with dwell = 2 each hop is confirmed on
    # the second frame at the new site, so the tracker records exactly 2 hops.
    function hopping_tracker()
        a, nrep = 3.61, 2
        cell = fcc_cell_for(a, nrep)
        sites = reference_sites(cell, a, nrep)
        tr = VacancyTracker(sites; cell, window = 1, dwell = 2)
        seq = [5, 5, 5, 6, 6, 6, 6, 7, 7, 7]
        n = size(sites, 2)
        for (k, v) in enumerate(seq)
            occupied = [s for s in 1:n if s != v]
            update!(tr, sites[:, occupied], Float64(k))
        end
        return tr
    end

    function write_ptraj(path; nframes = 4)
        pos = [0.0 1.0 0.0 1.0; 0.0 0.0 1.0 1.0; 0.0 0.0 0.0 0.0]
        w = TrajectoryWriter(path, 4, ["Cu", "Cu", "Au", "Au"]; velocities = false, cell = true)
        for k in 1:nframes
            write_frame(w, 0.5 * k, pos .+ 0.05 * k; cell = [4.0 0 0; 0 4 0; 0 0 4], energy = -1.0 * k)
        end
        close(w)
        return path
    end

    function write_xyz(path; nframes = 3)
        pos = [0.0 1.0 0.0 1.0; 0.0 0.0 1.0 1.0; 0.0 0.0 0.0 0.0]
        open(path, "w") do io
            for k in 1:nframes
                write_extxyz(
                    io, Frame(
                        ["Cu", "Cu", "Au", "Au"], pos .+ 0.05 * k;
                        cell = [4.0 0 0; 0 4 0; 0 0 4], pbc = (true, true, true), energy = -1.0 * k, time = 0.5 * k
                    )
                )
            end
        end
        return path
    end

    is_png(path) = filesize(path) > 8 && open(io -> read(io, 8), path) == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
    # An MP4 file starts with a box whose type, bytes 5 to 8, is "ftyp".
    is_mp4(path) = filesize(path) > 12 && open(io -> read(io, 12)[5:8], path) == codeunits("ftyp")
end

@testitem "the core carries no Makie dependency" tags = [:makie, :checks] begin
    using TOML
    proj = TOML.parsefile(joinpath(pkgdir(PureAtomistic), "Project.toml"))
    @test !haskey(proj["deps"], "Makie")
    @test proj["weakdeps"]["Makie"] == "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    @test proj["extensions"]["PureAtomisticMakieExt"] == "Makie"
    @test haskey(proj["compat"], "Makie")
    # The stubs exist without Makie and have no methods until an extension adds them.
    @test isdefined(PureAtomistic, :animate_trajectory)
    @test isempty(methods(PureAtomistic.plot_vacancy_track))
end

@testitem "the Makie extension loads with Makie" tags = [:makie] begin
    using CairoMakie
    @test Base.get_extension(PureAtomistic, :PureAtomisticMakieExt) isa Module
end

@testitem "plot_vacancy_track draws the track and the hops" tags = [:makie] setup = [MakieFixtures] begin
    using CairoMakie
    using PureAtomistic: plot_vacancy_track
    tr = hopping_tracker()
    @test tr.nhops == 2
    fig = plot_vacancy_track(tr)
    @test fig isa Makie.Figure
    axes_ = [c for c in fig.content if c isa Makie.Axis]
    @test length(axes_) == 2
    # Panel 1: x, y, z of the vacancy against time plus one marker set for the hop times.
    @test count(p -> p isa Makie.Lines, axes_[1].scene.plots) == 3
    @test count(p -> p isa Makie.VLines, axes_[1].scene.plots) == 1
    hops = only(p for p in axes_[1].scene.plots if p isa Makie.VLines)
    @test hops[1][] == tr.hop_t[1:2]
    # Panel 2: the squared displacement from the first frame.
    @test count(p -> p isa Makie.Lines, axes_[2].scene.plots) == 1
    lines2 = only(p for p in axes_[2].scene.plots if p isa Makie.Lines)
    pts = lines2[1][]
    @test length(pts) == tr.nframes
    @test first(pts)[1] == tr.t[1]
    # The squared displacement of the track, recomputed from R without Makie.
    msd = [sum(abs2, tr.R[:, k] - tr.R[:, 1]) for k in 1:tr.nframes]
    @test [p[2] for p in pts] == msd
    path = joinpath(mktempdir(), "vac.png")
    save(path, fig)
    @test is_png(path)
end

@testitem "plot_vacancy_track omits the hop marks when there is no hop" tags = [:makie] setup = [MakieFixtures] begin
    using CairoMakie
    using PureAtomistic: plot_vacancy_track
    using PureAtomistic.Analysis: VacancyTracker, update!, reference_sites
    a, nrep = 3.61, 2
    cell = fcc_cell_for(a, nrep)
    sites = reference_sites(cell, a, nrep)
    tr = VacancyTracker(sites; cell, window = 1, dwell = 2)
    for k in 1:3
        update!(tr, sites[:, 2:end], Float64(k))
    end
    fig = plot_vacancy_track(tr)
    ax = first(c for c in fig.content if c isa Makie.Axis)
    @test count(p -> p isa Makie.VLines, ax.scene.plots) == 0
    @test_throws "pureatomistic: plot_vacancy_track needs at least 2 frames" plot_vacancy_track(
        VacancyTracker(sites; cell, window = 1, dwell = 2)
    )
end

@testitem "plot_arrhenius draws the data, the fit line and the labels" tags = [:makie] setup = [MakieFixtures] begin
    using CairoMakie
    using PureAtomistic: plot_arrhenius
    using PureAtomistic.Analysis: arrhenius_fit
    kB = PureAtomistic.Systems.KB_EV_PER_K
    temps = [1000.0, 1100.0, 1200.0, 1300.0]
    Em, Γ0 = 0.8, 1.0e13
    rates = Γ0 .* exp.(-Em ./ (kB .* temps))
    errors = 0.05 .* rates
    fit = arrhenius_fit(temps, rates, errors)
    fig = plot_arrhenius(fit; temps, rates, errors)
    @test fig isa Makie.Figure
    ax = only(c for c in fig.content if c isa Makie.Axis)
    @test count(p -> p isa Makie.Errorbars, ax.scene.plots) == 1
    @test count(p -> p isa Makie.Scatter, ax.scene.plots) == 1
    @test count(p -> p isa Makie.Lines, ax.scene.plots) == 1
    # The line is ln Γ = ln Γ₀ - E/(kB T) against 1/(kB T): slope -E_m, intercept ln prefactor.
    line = only(p for p in ax.scene.plots if p isa Makie.Lines)
    pts = line[1][]
    slope = (pts[end][2] - pts[1][2]) / (pts[end][1] - pts[1][1])
    # The fit is exact on noiseless data; the line is evaluated from the fit in Float64, so the
    # slope matches -E_m to a few eps times the condition of the two-point difference.
    @test slope ≈ -fit.E_m rtol = 1.0e-9
    @test pts[1][2] + fit.E_m * pts[1][1] ≈ log(fit.prefactor) rtol = 1.0e-9
    # The points are plotted at (1/(kB T), ln rate), an exact transcription of the inputs.
    sc = only(p for p in ax.scene.plots if p isa Makie.Scatter)
    @test [p[1] for p in sc[1][]] == 1 ./ (kB .* temps)
    @test [p[2] for p in sc[1][]] == log.(rates)
    path = joinpath(mktempdir(), "arr.png")
    save(path, fig)
    @test is_png(path)
    # Without rates there is no data to draw: only the line, over the temperatures given.
    fig2 = plot_arrhenius(fit; temps)
    ax2 = only(c for c in fig2.content if c isa Makie.Axis)
    @test count(p -> p isa Makie.Lines, ax2.scene.plots) == 1
    @test count(p -> p isa Makie.Scatter, ax2.scene.plots) == 0
    @test_throws "pureatomistic: plot_arrhenius needs temps" plot_arrhenius(fit)
    @test_throws "pureatomistic: plot_arrhenius needs rates and errors together" plot_arrhenius(fit; temps, rates)
    @test_throws "pureatomistic: plot_arrhenius needs temps, rates and errors of one length" plot_arrhenius(
        fit; temps, rates = rates[1:2], errors = errors[1:2]
    )
end

@testitem "plot_energy_conservation draws the drift and the energy split" tags = [:makie] setup = [MakieFixtures] begin
    using CairoMakie
    using PureAtomistic: plot_energy_conservation
    t = collect(0.0:1.0:20.0)
    kinetic = 1.0 .+ 0.1 .* sin.(t)
    potential = -3.0 .- 0.1 .* sin.(t) .+ 1.0e-4 .* t
    log = (; t, kinetic, potential)
    fig = plot_energy_conservation(log)
    @test fig isa Makie.Figure
    axes_ = [c for c in fig.content if c isa Makie.Axis]
    @test length(axes_) == 2
    @test count(p -> p isa Makie.Lines, axes_[1].scene.plots) == 1
    @test count(p -> p isa Makie.Lines, axes_[2].scene.plots) == 2
    drift = only(p for p in axes_[1].scene.plots if p isa Makie.Lines)
    # Total energy minus its first value, in the units of the log: exact subtraction of the inputs.
    total = kinetic .+ potential
    @test [p[2] for p in drift[1][]] == total .- total[1]
    @test [p[1] for p in drift[1][]] == t
    path = joinpath(mktempdir(), "energy.png")
    save(path, fig)
    @test is_png(path)
    @test_throws "pureatomistic: plot_energy_conservation needs at least 2 samples" plot_energy_conservation(
        (; t = [0.0], kinetic = [1.0], potential = [-1.0])
    )
    @test_throws "pureatomistic: plot_energy_conservation needs t, kinetic and potential of one length" plot_energy_conservation(
        (; t = t, kinetic = kinetic[1:3], potential)
    )
end

@testitem "animate_trajectory writes an mp4 from a binary trajectory and from extxyz" tags = [:makie] setup = [MakieFixtures] begin
    using CairoMakie
    using PureAtomistic: animate_trajectory
    dir = mktempdir()
    ptraj = write_ptraj(joinpath(dir, "run.ptraj"))
    out = animate_trajectory(ptraj; output = joinpath(dir, "run.mp4"), framerate = 5)
    @test out == joinpath(dir, "run.mp4")
    @test is_mp4(out)
    xyz = write_xyz(joinpath(dir, "run.extxyz"))
    out2 = animate_trajectory(xyz; output = joinpath(dir, "xyz.mp4"), framerate = 5)
    @test is_mp4(out2)
    # Every second frame of the 4.
    out3 = animate_trajectory(ptraj; output = joinpath(dir, "s.mp4"), framerate = 5, stride = 2)
    @test is_mp4(out3)
    @test_throws "pureatomistic: animate_trajectory needs stride >= 1, got 0" animate_trajectory(
        ptraj; output = joinpath(dir, "bad.mp4"), stride = 0
    )
    @test_throws "pureatomistic: animate_trajectory needs a path ending in .mp4, .gif or .webm" animate_trajectory(
        ptraj; output = joinpath(dir, "bad.txt")
    )
    empty = joinpath(dir, "empty.ptraj")
    close(PureAtomistic.IO.TrajectoryWriter(empty, 4, ["Cu", "Cu", "Au", "Au"]))
    @test_throws "pureatomistic: animate_trajectory found no frames" animate_trajectory(
        empty; output = joinpath(dir, "e.mp4")
    )
end

@testitem "animate_trajectory draws one marker per atom, colored by species" tags = [:makie] setup = [MakieFixtures] begin
    using CairoMakie
    ext = Base.get_extension(PureAtomistic, :PureAtomisticMakieExt)
    dir = mktempdir()
    ptraj = write_ptraj(joinpath(dir, "run.ptraj"))
    fig, frame_index, nframes, close_source = ext.trajectory_figure(ptraj)
    @test fig isa Makie.Figure
    @test nframes == 4
    ax = only(c for c in fig.content if c isa Makie.Axis3)
    sc = only(p for p in ax.scene.plots if p isa Makie.MeshScatter)
    @test length(sc[1][]) == 4
    # Species Cu, Cu, Au, Au: two distinct colors, equal within species.
    cols = sc.color[]
    @test cols[1] == cols[2] && cols[3] == cols[4] && cols[1] != cols[3]
    # Stepping the frame index moves the markers to that frame's positions (Float32 storage).
    frame_index[] = 3
    pos = [0.0 1.0 0.0 1.0; 0.0 0.0 1.0 1.0; 0.0 0.0 0.0 0.0] .+ 0.05 * 3
    @test [Float32.(Tuple(p)) for p in sc[1][]] == [Float32.(Tuple(pos[:, i])) for i in 1:4]
    close_source()
end
