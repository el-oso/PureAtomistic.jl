# Methods for the plotting functions declared in `src/viz.jl`. Written against `Makie`, so any
# backend (GLMakie, CairoMakie, WGLMakie) can render the figures.
module PureAtomisticMakieExt

import Makie
import PureAtomistic
using PureAtomistic: animate_trajectory, plot_vacancy_track, plot_arrhenius, plot_energy_conservation
using PureAtomistic.Analysis: VacancyTracker
using PureAtomistic.IO: TrajectoryReader, read_frame, read_extxyz
using PureAtomistic.Systems: KB_EV_PER_K

# ---------------------------------------------------------------------------------------------
# Vacancy track

function PureAtomistic.plot_vacancy_track(tr::VacancyTracker)
    n = tr.nframes
    n >= 2 || throw(ArgumentError("pureatomistic: plot_vacancy_track needs at least 2 frames"))
    t = tr.t[1:n]
    R = tr.R[:, 1:n]
    fig = Makie.Figure()
    ax1 = Makie.Axis(fig[1, 1]; ylabel = "vacancy position (Å)")
    for (c, name) in enumerate(("x", "y", "z"))
        Makie.lines!(ax1, t, R[c, :]; label = name)
    end
    tr.nhops > 0 && Makie.vlines!(ax1, tr.hop_t[1:tr.nhops]; color = (:gray, 0.6), linestyle = :dash)
    Makie.axislegend(ax1; position = :lt)
    ax2 = Makie.Axis(fig[2, 1]; xlabel = "time", ylabel = "squared displacement (Å²)")
    msd = [sum(abs2, R[:, k] - R[:, 1]) for k in 1:n]
    Makie.lines!(ax2, t, msd)
    Makie.linkxaxes!(ax1, ax2)
    return fig
end

# ---------------------------------------------------------------------------------------------
# Arrhenius fit

function PureAtomistic.plot_arrhenius(fit; temps = nothing, rates = nothing, errors = nothing)
    temps === nothing && throw(ArgumentError("pureatomistic: plot_arrhenius needs temps"))
    (rates === nothing) == (errors === nothing) ||
        throw(ArgumentError("pureatomistic: plot_arrhenius needs rates and errors together"))
    if rates !== nothing
        length(temps) == length(rates) == length(errors) ||
            throw(ArgumentError("pureatomistic: plot_arrhenius needs temps, rates and errors of one length"))
    end
    kB = Float64(KB_EV_PER_K)
    x = [1 / (kB * Float64(T)) for T in temps]
    fig = Makie.Figure()
    ax = Makie.Axis(
        fig[1, 1]; xlabel = "1/(k_B T) (1/eV)", ylabel = "ln(rate)",
        title = "E_m = $(round(fit.E_m; digits = 3)) ± $(round(fit.σ; digits = 3)) eV"
    )
    if rates !== nothing
        y = [log(Float64(r)) for r in rates]
        Makie.errorbars!(ax, x, y, [Float64(e) / Float64(r) for (e, r) in zip(errors, rates)])
        Makie.scatter!(ax, x, y)
    end
    xe = extrema(x)
    Makie.lines!(ax, [xe[1], xe[2]], [log(fit.prefactor) - fit.E_m * xe[1], log(fit.prefactor) - fit.E_m * xe[2]])
    return fig
end

# ---------------------------------------------------------------------------------------------
# Energy conservation

function PureAtomistic.plot_energy_conservation(log)
    for name in (:t, :kinetic, :potential)
        hasproperty(log, name) ||
            throw(ArgumentError("pureatomistic: plot_energy_conservation needs t, kinetic and potential of one length"))
    end
    t, kinetic, potential = log.t, log.kinetic, log.potential
    length(t) == length(kinetic) == length(potential) ||
        throw(ArgumentError("pureatomistic: plot_energy_conservation needs t, kinetic and potential of one length"))
    length(t) >= 2 || throw(ArgumentError("pureatomistic: plot_energy_conservation needs at least 2 samples"))
    total = kinetic .+ potential
    fig = Makie.Figure()
    ax1 = Makie.Axis(fig[1, 1]; ylabel = "total energy drift (eV)")
    Makie.lines!(ax1, t, total .- first(total))
    ax2 = Makie.Axis(fig[2, 1]; xlabel = "time", ylabel = "energy (eV)")
    Makie.lines!(ax2, t, kinetic; label = "kinetic")
    Makie.lines!(ax2, t, potential; label = "potential")
    Makie.axislegend(ax2; position = :rt)
    Makie.linkxaxes!(ax1, ax2)
    return fig
end

# ---------------------------------------------------------------------------------------------
# Trajectory animation

# Frame count and a function returning frame `k`, for either file format; `close_source` releases
# the file.
function _open_trajectory(path::AbstractString)
    if endswith(lowercase(path), ".extxyz") || endswith(lowercase(path), ".xyz")
        frames = read_extxyz(path)
        return length(frames), k -> frames[k], () -> nothing
    end
    r = TrajectoryReader(path)
    return length(r), k -> read_frame(r, k), () -> close(r)
end

_points(positions) = [Makie.Point3f(positions[1, i], positions[2, i], positions[3, i]) for i in axes(positions, 2)]

# Corners of the cell as a closed polyline set (12 edges), or an empty list without a cell.
function _cell_edges(cell)
    isempty(cell) && return Makie.Point3f[]
    corner(i, j, k) = Makie.Point3f(i * cell[:, 1] + j * cell[:, 2] + k * cell[:, 3])
    edges = Makie.Point3f[]
    for (a, b) in (
            ((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
            ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
            ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1)),
        )
        push!(edges, corner(a...), corner(b...))
    end
    return edges
end

# The figure, the observable holding the index of the frame shown, the frame count, and a function
# that closes the file the frames are read from (to be called once the figure is no longer redrawn).
function trajectory_figure(path::AbstractString; stride::Integer = 1, markersize::Real = 0.3)
    n, getframe, close_source = _open_trajectory(path)
    try
        n >= 1 || throw(ArgumentError("pureatomistic: animate_trajectory found no frames in \"$path\""))
        first_frame = getframe(1)
        lo = fill(Inf, 3)
        hi = fill(-Inf, 3)
        for k in 1:stride:n
            P = getframe(k).positions
            for c in 1:3, i in axes(P, 2)
                lo[c] = min(lo[c], P[c, i])
                hi[c] = max(hi[c], P[c, i])
            end
        end
        edges = _cell_edges(first_frame.cell)
        for p in edges, c in 1:3
            lo[c] = min(lo[c], p[c])
            hi[c] = max(hi[c], p[c])
        end
        names = unique(first_frame.species)
        palette = Makie.wong_colors()
        colors = [palette[mod1(findfirst(==(s), names), length(palette))] for s in first_frame.species]
        index = Makie.Observable(1)
        frame = Makie.lift(getframe, index)
        positions = Makie.lift(f -> _points(f.positions), frame)
        title = Makie.lift(f -> isnan(f.time) ? "" : "t = $(round(f.time; digits = 3)) fs", frame)
        fig = Makie.Figure()
        ax = Makie.Axis3(
            fig[1, 1]; title, aspect = :data, limits = (lo[1], hi[1], lo[2], hi[2], lo[3], hi[3])
        )
        isempty(edges) || Makie.linesegments!(ax, edges; color = :gray, linewidth = 1)
        Makie.meshscatter!(ax, positions; color = colors, markersize)
        return fig, index, n, close_source
    catch
        close_source()
        rethrow()
    end
end

function PureAtomistic.animate_trajectory(
        path::AbstractString; output::AbstractString = first(splitext(path)) * ".mp4",
        framerate::Integer = 24, stride::Integer = 1, markersize::Real = 0.3
    )
    stride >= 1 || throw(ArgumentError("pureatomistic: animate_trajectory needs stride >= 1, got $stride"))
    lowercase(last(splitext(output))) in (".mp4", ".gif", ".webm") ||
        throw(ArgumentError("pureatomistic: animate_trajectory needs a path ending in .mp4, .gif or .webm, got \"$output\""))
    fig, index, n, close_source = trajectory_figure(path; stride, markersize)
    try
        Makie.record(fig, output, 1:stride:n; framerate) do k
            index[] = k
        end
    finally
        close_source()
    end
    return output
end

end
