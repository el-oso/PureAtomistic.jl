# Plotting entry points. Each function is a generic function without methods here; the methods
# live in the extension `PureAtomisticMakieExt`, which loads when Makie (or a backend such as
# GLMakie or CairoMakie) is loaded. Without Makie, a call throws a `MethodError`.

"""
    animate_trajectory(path; output, framerate = 24, stride = 1, markersize = 0.3)

Render the trajectory file `path` (a binary trajectory, or extended XYZ when the name ends in
`.extxyz` or `.xyz`) as a video of atoms colored by species, drawn inside the cell when the
trajectory has one. Every `stride`-th frame is drawn. `output` is the video path, with the
extension `.mp4`, `.gif` or `.webm` (default: `path` with the extension replaced by `.mp4`); it is
returned. Requires Makie.
"""
function animate_trajectory end

"""
    plot_vacancy_track(tracker) -> Figure

Plot a [`PureAtomistic.Analysis.VacancyTracker`](@ref). The upper panel shows the three
Cartesian components of the unwrapped vacancy position against time, with a vertical line at
each confirmed hop; the lower panel shows the squared displacement from the first frame, whose
slope is `6D` for a diffusing vacancy. Requires Makie and at least 2 frames.
"""
function plot_vacancy_track end

"""
    plot_arrhenius(fit; temps, rates = nothing, errors = nothing) -> Figure

Plot an Arrhenius fit, the named tuple `(E_m, σ, prefactor)` of [`arrhenius_fit`](@ref), as
`ln Γ` against `1/(k_B T)`: the line `ln Γ₀ - E_m/(k_B T)` over the range of `temps` (K), and,
when `rates` and `errors` (the data the fit was made from) are given, the data with error bars
of `errors / rates` on `ln Γ`. The title states `E_m ± σ`. Requires Makie.
"""
function plot_arrhenius end

"""
    plot_energy_conservation(log) -> Figure

Plot the energy record `log`, any object with the equal-length vectors `t`, `kinetic` and
`potential` (energies in eV). The upper panel shows the drift of the total energy,
`kinetic + potential` minus its first value; the lower panel shows the kinetic and potential
energies. Requires Makie and at least 2 samples.
"""
function plot_energy_conservation end
