# Renders the animations of the gallery pages `examples/nve_conservation.jl`,
# `examples/thermostat_check.jl` and `examples/vacancy_hop_md.jl` into `docs/src/assets/` with
# GLMakie (OpenGL, so it needs a display). The documentation build does not run it: the videos are
# committed.
#
#   julia --project=<environment with GLMakie, JSON and PureAtomistic> PureAtomistic/docs/render_animations.jl <trajectory dir> [nve] [nvt] [vacancy]
#
# <trajectory dir> holds the binary trajectories written by `bench/md_validation.jl`
# (`nve_dt2.patraj`, `nvt.patraj`, `vacancy.patraj`); the energy, temperature and vacancy records
# come from `bench/results/`.
using GLMakie, JSON, PureAtomistic
using PureAtomistic.IO: TrajectoryReader, read_frame

const TRAJ = ARGS[1]
const RESULTS = joinpath(@__DIR__, "..", "bench", "results")
const ASSETS = mkpath(joinpath(@__DIR__, "src", "assets"))
results(name) = JSON.parsefile(joinpath(RESULTS, name))

function frames(name)
    r = TrajectoryReader(joinpath(TRAJ, name))
    fs = [read_frame(r, k) for k in 1:length(r)]
    close(r)
    return fs
end

pts(P) = [Point3f(P[1, i], P[2, i], P[3, i]) for i in axes(P, 2)]

function cell_edges(L)
    c = (0, 1)
    e = Point3f[]
    for a in c, b in c
        push!(e, Point3f(0, a * L, b * L), Point3f(L, a * L, b * L))
        push!(e, Point3f(a * L, 0, b * L), Point3f(a * L, L, b * L))
        push!(e, Point3f(a * L, b * L, 0), Point3f(a * L, b * L, L))
    end
    return e
end

function atoms_axis(fig, L; title)
    ax = Axis3(fig[1, 1]; aspect = :data, title, limits = (0, L, 0, L, 0, L), elevation = 0.5, azimuth = 1.2)
    hidedecorations!(ax)
    hidespines!(ax)
    linesegments!(ax, cell_edges(L); color = :gray, linewidth = 1)
    return ax
end

# Displacement of every atom from its position in the first frame (unwrapped coordinates).
displacement(fs, k) = vec(sqrt.(sum(abs2, fs[k].positions .- fs[1].positions; dims = 1)))

wrap(P, L) = mod.(P, L)

function nve()
    fs = frames("nve_dt2.patraj")
    d = results("md_nve_conservation.json")
    L = fs[1].cell[1, 1]
    k = Observable(1)
    fig = Figure(; size = (960, 480), fontsize = 14)
    ax = atoms_axis(fig, L; title = @lift("dt = 2 fs, t = $(round(fs[$k].time / 1000; digits = 2)) ps"))
    meshscatter!(
        ax, @lift(pts(wrap(fs[$k].positions, L))); markersize = 0.7, color = @lift(displacement(fs, $k)),
        colormap = :viridis, colorrange = (0, 0.5)
    )
    Colorbar(fig[2, 1]; colormap = :viridis, limits = (0, 0.5), vertical = false, label = "distance from lattice site (Å)", flipaxis = false)
    ax2 = Axis(fig[1:2, 2]; xlabel = "time (ps)", ylabel = "E(t) − E(0) (meV/atom)", title = "total energy")
    xlims!(ax2, 0, 2)
    ylims!(ax2, -0.1, 0.6)
    for r in filter(r -> r["dt_fs"] <= 8, d["runs"])
        r["status"] == "completed" || continue
        t = Float64.(r["t_fs"]) ./ 1000
        E = 1.0e3 .* (Float64.(r["total_eV_per_atom"]) .- r["total_eV_per_atom"][1])
        lines!(ax2, @lift(Point2f.(t[t .<= fs[$k].time / 1000], E[t .<= fs[$k].time / 1000])); label = "dt = $(r["dt_fs"]) fs")
    end
    axislegend(ax2; position = :lb)
    record(k_ -> (k[] = k_), fig, joinpath(ASSETS, "nve_conservation.mp4"), 1:2:length(fs); framerate = 20)
    return
end

function nvt()
    fs = frames("nvt.patraj")
    d = results("md_thermostat_check.json")
    L = fs[1].cell[1, 1]
    t, T = Float64.(d["t_fs"]) ./ 1000, Float64.(d["temperature_K"])
    Tset, σ = d["theory"]["mean_K"], d["theory"]["std_K"]
    k = Observable(1)
    fig = Figure(; size = (960, 480), fontsize = 14)
    ax = atoms_axis(fig, L; title = @lift("1200 K, t = $(round(fs[$k].time / 1000; digits = 2)) ps"))
    meshscatter!(
        ax, @lift(pts(wrap(fs[$k].positions, L))); markersize = 0.7, color = @lift(displacement(fs, $k)),
        colormap = :viridis, colorrange = (0, 1.0)
    )
    Colorbar(fig[2, 1]; colormap = :viridis, limits = (0, 1.0), vertical = false, label = "distance from lattice site (Å)", flipaxis = false)
    ax2 = Axis(fig[1:2, 2]; xlabel = "time (ps)", ylabel = "kinetic temperature (K)", title = "Bussi thermostat")
    xlims!(ax2, 0, 2)
    ylims!(ax2, Tset - 5σ, Tset + 5σ)
    band!(ax2, [0, 2], [Tset - σ, Tset - σ], [Tset + σ, Tset + σ]; color = (:crimson, 0.15))
    hlines!(ax2, [Tset]; color = :crimson)
    lines!(ax2, @lift(Point2f.(t[t .<= fs[$k].time / 1000], T[t .<= fs[$k].time / 1000])); linewidth = 1)
    record(k_ -> (k[] = k_), fig, joinpath(ASSETS, "thermostat_check.mp4"), 1:length(fs); framerate = 20)
    return
end

function vacancy()
    fs = frames("vacancy.patraj")
    d = results("md_vacancy_hop.json")
    L = fs[1].cell[1, 1]
    t = Float64.(d["t_fs"]) ./ 1000
    R = reduce(hcat, [Float64.(r) for r in d["vacancy_R"]])
    hops = Float64.(d["hop_times_fs"]) ./ 1000
    n = min(length(fs), size(R, 2))
    k = Observable(1)
    fig = Figure(; size = (960, 480), fontsize = 14)
    ax = atoms_axis(fig, L; title = @lift("1300 K, t = $(round(t[$k]; digits = 2)) ps"))
    meshscatter!(
        ax, @lift(pts(wrap(fs[$k].positions, L))); markersize = 0.5, color = @lift(wrap(fs[$k].positions, L)[3, :]),
        colormap = (:Blues, 0.6), colorrange = (0, L)
    )
    meshscatter!(ax, @lift([Point3f(wrap(R[:, $k], L)...)]); markersize = 1.2, color = :crimson)
    ax2 = Axis(fig[1, 2]; xlabel = "time (ps)", ylabel = "vacancy position (Å)", title = "unwrapped vacancy position")
    xlims!(ax2, 0, t[n])
    ylims!(ax2, minimum(R) - 1, maximum(R) + 1)
    for (c, name) in enumerate(("x", "y", "z"))
        lines!(ax2, @lift(Point2f.(t[1:$k], R[c, 1:$k])); label = name)
    end
    isempty(hops) || vlines!(ax2, hops; color = (:gray, 0.7), linestyle = :dash)
    axislegend(ax2; position = :lt)
    stride = max(1, n ÷ 200)
    record(k_ -> (k[] = k_), fig, joinpath(ASSETS, "vacancy_hop_md.mp4"), 1:stride:n; framerate = 24)
    return
end

for f in (nve, nvt, vacancy)
    length(ARGS) > 1 && string(nameof(f)) ∉ ARGS[2:end] && continue
    f()
    println(f, " done")
end
