# # Lattice constant of Cu
#
# This example computes the energy of fcc Cu as its cubic cell expands and shrinks, and takes
# the lattice constant from the minimum.

using PureMLIP, LinearAlgebra, Printf
using GLMakie
include(pkgdir(PureMLIP, "docs", "examples", "setup.jl")); #hide

model = MACEModel(load_mace_checkpoint(checkpoint_path(:mpa0)));

# The four-atom conventional cell is periodic, so it represents the perfect crystal; the energy
# per atom is computed at 33 lattice constants from 3.45 Å to 3.85 Å.

as = range(3.45, 3.85; length = 33)
E = [mace_energy_forces(model, fcc(a, 1)..., fill(29, 4)).energy / 4 for a in as]
for (a, e) in zip(as, E)
    @printf("a = %.4f Å  E = %.5f eV/atom\n", a, e)
end

# A parabola `E(a) = c₁ + c₂ a + c₃ a²` fitted by least squares to the nine points around the
# lowest energy gives the minimum.

i = argmin(E)
window = max(firstindex(E), i - 4):min(lastindex(E), i + 4)
c = [ones(length(window)) as[window] as[window] .^ 2] \ E[window]
a0 = -c[2] / (2c[3])
E0 = c[1] + c[2] * a0 + c[3] * a0^2
@printf("lattice constant: %.4f Å (fit over %.4f–%.4f Å), energy %.5f eV/atom\n", a0, as[first(window)], as[last(window)], E0)
@printf("experimental value for Cu at room temperature: 3.615 Å; difference %+.4f Å (%+.2f%%)\n", a0 - 3.615, 100 * (a0 - 3.615) / 3.615)

# The computed value is static: the minimum of the energy at 0 K, without thermal expansion or
# zero-point motion. The measured 3.615 Å is a room-temperature value, so part of the difference
# is thermal expansion. MACE-MPA-0 is also trained on density functional theory data computed
# with the PBE functional, whose lattice constants for metals are often somewhat larger than
# measured ones; the difference above is a comparison with experiment, not a test of the code.

# ## Animation
#
# Left: the conventional cell drawn at each lattice constant. Right: the energy per atom, drawn
# point by point as the cell expands, then followed back as it shrinks; the last frames add the
# parabola and its minimum.

sweep = [collect(eachindex(as)); reverse(eachindex(as))[2:end]; fill(firstindex(as), 12)]
frame = Observable(1)
k_now = @lift(sweep[$frame])
a_now = @lift(as[$k_now])
sites = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
lattice_points = [s .+ [i, j, k] for s in sites for i in 0:1 for j in 0:1 for k in 0:1 if all(<=(1), s .+ [i, j, k])]
edges = [(a, b) for a in Iterators.product(0:1, 0:1, 0:1), b in Iterators.product(0:1, 0:1, 0:1) if a < b && sum(abs, a .- b) == 1]
fig = Figure(; size = (840, 420), fontsize = 14)
ax1 = Axis3(fig[1, 1]; aspect = :data, limits = (-0.5, 4.3, -0.5, 4.3, -0.5, 4.3), title = @lift(@sprintf("a = %.3f Å", $a_now)))
hidedecorations!(ax1)
hidespines!(ax1)
linesegments!(ax1, @lift([Point3f($a_now .* a) => Point3f($a_now .* b) for (a, b) in edges]); color = :gray40)
meshscatter!(ax1, @lift([Point3f($a_now .* p) for p in lattice_points]); markersize = 0.35, color = :darkorange)
ax2 = Axis(fig[1, 2]; xlabel = "a (Å)", ylabel = "E (eV/atom)")
limits!(ax2, 3.43, 3.87, minimum(E) - 0.02, maximum(E) + 0.02)
drawn = @lift(Point2f.(as[1:maximum(sweep[1:$frame])], E[1:maximum(sweep[1:$frame])]))
scatterlines!(ax2, drawn; color = :steelblue, markersize = 6)
scatter!(ax2, @lift([Point2f($a_now, E[$k_now])]); color = :crimson, markersize = 12)
show_fit = @lift($frame > length(sweep) - 12)
afit = range(as[first(window)] - 0.03, as[last(window)] + 0.03; length = 50)
lines!(ax2, afit, c[1] .+ c[2] .* afit .+ c[3] .* afit .^ 2; color = :crimson, visible = show_fit)
vlines!(ax2, [a0]; color = :crimson, linestyle = :dash, visible = show_fit)
vlines!(ax2, [3.615]; color = :gray, linestyle = :dot, visible = show_fit)
text!(ax2, a0, maximum(E); text = @sprintf(" fit minimum %.3f Å", a0), color = :crimson, align = (:left, :top), visible = show_fit)
text!(ax2, 3.615, maximum(E) - 0.04; text = "experiment 3.615 Å ", color = :gray30, align = (:right, :top), visible = show_fit)
record(fig, media("lattice_constant.mp4"), eachindex(sweep); framerate = 20) do s
    frame[] = s
end
nothing #hide

# ```@raw html
# <video src="./lattice_constant.mp4" controls loop muted playsinline width="640"></video>
# ```
