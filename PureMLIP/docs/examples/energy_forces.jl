# # Energy, forces and stress
#
# This example evaluates a 32-atom Cu cell with displaced atoms, then rotates the whole
# structure (positions and cell) and evaluates it again at each angle. The energy does not
# change under a rotation, the forces rotate with the atoms, and the stress rotates as a
# tensor, `σ → R σ Rᵀ`.

using PureMLIP, LinearAlgebra, Printf
using GLMakie
include(pkgdir(PureMLIP, "docs", "examples", "setup.jl")); #hide

model = MACEModel(load_mace_checkpoint(checkpoint_path(:mpa0)));

# Two by two by two conventional fcc cells at a = 3.63 Å, with each coordinate displaced by
# up to 0.08 Å so that the forces are not zero.

positions, cell = fcc(3.63, 2)
positions .+= 0.08 .* sin.(1.3 .* reshape(1:length(positions), size(positions)))
Z = fill(29, size(positions, 2))
r = mace_energy_forces(model, positions, cell, Z)
@printf("energy: %.6f eV (%.6f eV per atom)\n", r.energy, r.energy / length(Z))
@printf("largest force component: %.4f eV/Å\n", maximum(abs, r.forces))
@printf("sum of the forces: %.1e eV/Å\n", norm(sum(r.forces; dims = 2)))
println("stress (eV/Å³):")
show(stdout, "text/plain", round.(r.stress; sigdigits = 4))
println()

# ## Rotating the cell
#
# The rotation turns about the axis (1, 2, 3) by an angle that runs once around the circle.
# Each frame prints the energy and the largest differences between the rotated results of
# the first evaluation and the new ones.

axis = normalize([1.0, 2.0, 3.0])
rotation(θ) = cos(θ) * I + sin(θ) * [0 -axis[3] axis[2]; axis[3] 0 -axis[1]; -axis[2] axis[1] 0] + (1 - cos(θ)) * axis * axis'
center = cell * fill(0.5, 3)
angles = range(0, 2π; length = 49)[1:(end - 1)]
frames = map(angles) do θ
    R = rotation(θ)
    p = R * (positions .- center)
    rθ = mace_energy_forces(model, p, R * cell, Z)
    dF = maximum(abs, rθ.forces - R * r.forces)
    dσ = maximum(abs, rθ.stress - R * r.stress * R')
    @printf("θ = %5.1f°  E = %.8f eV  max|F − R F₀| = %.1e eV/Å  max|σ − R σ₀ Rᵀ| = %.1e eV/Å³\n", rad2deg(θ), rθ.energy, dF, dσ)
    (; R, p, F = rθ.forces, E = rθ.energy)
end
@printf("spread of the energy over all angles: %.1e eV\n", maximum(f.E for f in frames) - minimum(f.E for f in frames))

# ## Animation
#
# Atoms, cell edges and force arrows (2.5 Å per eV/Å) at each angle, with the energy of that
# frame in the title.

edges = [(0, 0, 0) => (1, 0, 0), (0, 0, 0) => (0, 1, 0), (0, 0, 0) => (0, 0, 1), (1, 0, 0) => (1, 1, 0), (1, 0, 0) => (1, 0, 1), (0, 1, 0) => (1, 1, 0), (0, 1, 0) => (0, 1, 1), (0, 0, 1) => (1, 0, 1), (0, 0, 1) => (0, 1, 1), (1, 1, 0) => (1, 1, 1), (1, 0, 1) => (1, 1, 1), (0, 1, 1) => (1, 1, 1)]
cell_lines(R) = [Point3f(R * (cell * collect(a) - center)) => Point3f(R * (cell * collect(b) - center)) for (a, b) in edges]
k = Observable(1)
fig = Figure(; size = (560, 560), fontsize = 14)
ax = Axis3(fig[1, 1]; aspect = :data, title = @lift(@sprintf("E = %.6f eV", frames[$k].E)), limits = (-6.5, 6.5, -6.5, 6.5, -6.5, 6.5))
hidedecorations!(ax)
hidespines!(ax)
linesegments!(ax, @lift(cell_lines(frames[$k].R)); color = :gray40)
meshscatter!(ax, @lift(Point3f.(eachcol(frames[$k].p))); markersize = 0.45, color = :darkorange)
arrows3d!(ax, @lift(Point3f.(eachcol(frames[$k].p))), @lift(Vec3f.(eachcol(2.5 .* frames[$k].F))); color = :steelblue, shaftradius = 0.035, tipradius = 0.09, tiplength = 0.22)
record(fig, media("energy_forces.mp4"), eachindex(frames); framerate = 16) do i
    k[] = i
end
nothing #hide

# ```@raw html
# <video src="./energy_forces.mp4" controls loop muted playsinline width="480"></video>
# ```
