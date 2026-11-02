# # Spherical harmonics and tensor kernels
#
# `PureMLIP.Equivariant` evaluates real spherical harmonics, and `PureMLIP.Tensors` holds the
# kernels that couple them with node features. This example draws the harmonics up to `l = 3`
# on the unit sphere and lists each kernel with the einsum its docstring gives.

using PureMLIP, LinearAlgebra, Printf
using GLMakie
include(pkgdir(PureMLIP, "docs", "examples", "setup.jl")); #hide

# ## Real spherical harmonics
#
# `spherical_harmonics!(Y, v, lmax)` fills `Y` with `Y_lm(v / ‖v‖)` for `0 ≤ l ≤ lmax`, with
# `Y_lm` at `sh_index(l, m) = l² + l + m + 1`. They are orthonormal on the sphere. A simple
# product-rule quadrature on the grid used for the drawing reproduces the identity to the
# accuracy of that quadrature, about 1e-3.

lmax = 3
θs = range(0, π; length = 61)
φs = range(0, 2π; length = 121)
Y = zeros(sh_length(lmax), length(θs), length(φs))
for (j, θ) in pairs(θs), (k, φ) in pairs(φs)
    spherical_harmonics!(view(Y, :, j, k), [sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)], lmax)
end
w = [sin(θ) * step(θs) * step(φs) for θ in θs, φ in φs]
w[:, end] .= 0  # φ = 2π repeats φ = 0
gram = [sum(w .* Y[a, :, :] .* Y[b, :, :]) for a in axes(Y, 1), b in axes(Y, 1)]
@printf("%d harmonics; largest deviation of the quadrature Gram matrix from the identity: %.1e\n", size(Y, 1), maximum(abs, gram - I))

# ## Animation
#
# One sphere per `(l, m)`, colored by `Y_lm` (red positive, blue negative), turning about the
# polar axis z.

X = [sin(θ) * cos(φ) for θ in θs, φ in φs]
Yc = [sin(θ) * sin(φ) for θ in θs, φ in φs]
Zc = [cos(θ) for θ in θs, φ in φs]
azimuth = Observable(0.0)
fig = Figure(; size = (840, 520), fontsize = 13, figure_padding = (2, 2, 2, 16))
for l in 0:lmax, m in (-l):l
    ax = Axis3(
        fig[l + 1, m + lmax + 1]; aspect = :data, azimuth, elevation = 0.35, title = "l=$l, m=$m",
        protrusions = 0, titlegap = 0
    )
    hidedecorations!(ax)
    hidespines!(ax)
    c = maximum(abs, Y[sh_index(l, m), :, :])
    surface!(ax, X, Yc, Zc; color = Y[sh_index(l, m), :, :], colormap = Reverse(:RdBu), colorrange = (-c, c), shading = NoShading)
end
record(fig, media("tensors.mp4"), range(0, 2π; length = 61)[1:(end - 1)]; framerate = 20) do a
    azimuth[] = a
end
nothing #hide

# ```@raw html
# <video src="./tensors.mp4" controls loop muted playsinline width="640"></video>
# ```

# ## Kernels and their einsums
#
# Each kernel's docstring states what it computes as one or more einsums in the notation
# `ein"inputs->output"(arrays...)`, with sums over the labels that do not appear in the output.
# The list below is read from the docstrings of the kernels (the functions whose names end in
# `!`) of `PureMLIP.Tensors`. A docstring that gives its einsums in prose instead, as that of
# `symmetric_contraction_pullback!` does, is shown by its "As an einsum" paragraph.

T = PureMLIP.Tensors
for name in names(T)
    f = getfield(T, name)
    f isa Function && endswith(string(name), "!") || continue
    doc = string(Docs.doc(f))
    einsums = [m.match for m in eachmatch(r"ein\"[^\"]*\"\([^)]*\)", doc)]
    println(name, ":")
    if isempty(einsums)
        prose = match(r"As an einsum.*?(?:\n\n|\z)"s, doc)
        println("    ", isnothing(prose) ? "(no einsum in its docstring)" : replace(strip(prose.match), "\n" => " "))
    end
    for e in einsums
        println("    ", e)
    end
end
nothing #hide
