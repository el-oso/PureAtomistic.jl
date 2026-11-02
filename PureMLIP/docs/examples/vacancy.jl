# # A vacancy in Cu
#
# This example computes the vacancy formation energy of Cu with MACE-MPA-0 and the energy of
# one nearest neighbor moved in a straight line into the vacancy.

using PureMLIP, LinearAlgebra, Printf
using GLMakie
include(pkgdir(PureMLIP, "docs", "examples", "setup.jl")); #hide

model = MACEModel(load_mace_checkpoint(checkpoint_path(:mpa0)));

# ## Lattice constant of the model
#
# The energy per atom of the four-atom conventional cell at nine lattice constants around the
# minimum, fitted by a parabola. The [lattice constant example](lattice_constant.md) draws the
# whole curve; it fits over a different window, so the last digit of the two values can differ.

as = range(3.6, 3.68; length = 9)
Es = [mace_energy_forces(model, fcc(a, 1)..., fill(29, 4)).energy / 4 for a in as]
c = [ones(length(as)) as as .^ 2] \ Es
a0 = -c[2] / (2c[3])
@printf("lattice constant: %.4f Å\n", a0)

# ## Vacancy formation energy
#
# A 4×4×4 block of conventional cells (256 atoms) at `a0`, and the same block with the atom at
# its center removed:
#
# ``E_\mathrm{vac} = E(N-1) - \frac{N-1}{N} E(N)``
#
# First with every other atom left on its lattice site (unrelaxed), then with the 255 atoms
# relaxed at fixed cell.

perfect, cell = fcc(a0, 4)
N = size(perfect, 2)
v = argmin(j -> norm(perfect[:, j] - fill(2a0, 3)), 1:N)
keep = setdiff(1:N, v)
E_N = mace_energy_forces(model, perfect, cell, fill(29, N)).energy
defect = perfect[:, keep]
Z = fill(29, N - 1)
r_unrelaxed = mace_energy_forces(model, defect, cell, Z)
E_vac_unrelaxed = r_unrelaxed.energy - (N - 1) / N * E_N
@printf("N = %d, E(N) = %.4f eV\n", N, E_N)
@printf("unrelaxed: E(N-1) = %.4f eV, E_vac = %.3f eV, largest force %.3f eV/Å\n", r_unrelaxed.energy, E_vac_unrelaxed, maximum(norm, eachcol(r_unrelaxed.forces)))

# The relaxation is FIRE (fast inertial relaxation engine) on the positions, with unit masses,
# a step of at most 0.1 Å per atom, and the stopping rule "largest force below 0.01 eV/Å".

function fire_relax(model, x, cell, Z; fmax = 0.01, maxsteps = 300)
    x = copy(x)
    vel = zero(x)
    dt, α, n_pos = 0.1, 0.1, 0
    r = mace_energy_forces(model, x, cell, Z)
    for step in 1:maxsteps
        F = r.forces
        maximum(norm, eachcol(F)) < fmax && return x, r, step - 1
        if dot(F, vel) > 0
            vel .= (1 - α) .* vel .+ α .* norm(vel) ./ norm(F) .* F
            n_pos += 1
            if n_pos > 5
                dt, α = min(1.1dt, 1.0), 0.99α
            end
        else
            vel .= 0
            dt, α, n_pos = 0.5dt, 0.1, 0
        end
        vel .+= dt .* F
        dx = dt .* vel
        dx .*= min(1, 0.1 / maximum(norm, eachcol(dx)))
        x .+= dx
        r = mace_energy_forces(model, x, cell, Z)
    end
    error("FIRE did not reach a largest force below $fmax eV/Å in $maxsteps steps")
end
relaxed, r_relaxed, steps = fire_relax(model, defect, cell, Z)
E_vac_relaxed = r_relaxed.energy - (N - 1) / N * E_N
@printf("relaxed in %d steps: E(N-1) = %.4f eV, E_vac = %.3f eV\n", steps, r_relaxed.energy, E_vac_relaxed)
@printf("largest displacement in the relaxation: %.3f Å\n", maximum(norm, eachcol(relaxed - defect)))

# Measured and computed values for Cu in the literature lie between about 1.0 and 1.3 eV. The
# values above are for one model, one cell size, and (relaxed) a fixed cell; they are a
# comparison with that range, not a validation of the model.

# ## A neighbor moved into the vacancy
#
# The nearest neighbor at `a0/2 · (1, 1, 0)` from the vacancy moves in a straight line onto the
# vacant site, in 25 steps, while every other atom stays on its lattice site. The energy along
# this path is measured from its starting value, the unrelaxed vacancy. Its maximum is the maximum
# along an unrelaxed straight-line path; the migration barrier is the maximum of the relaxed
# minimum-energy path, which this page does not compute.

j = findfirst(k -> norm(perfect[:, k] - perfect[:, v] - a0 / 2 * [1, 1, 0]) < 1.0e-8, keep)
start, stop = defect[:, j], perfect[:, v]
ts = range(0, 1; length = 25)
path = map(ts) do t
    x = copy(defect)
    x[:, j] = (1 - t) * start + t * stop
    r = mace_energy_forces(model, x, cell, Z)
    (; x, F = r.forces, E = r.energy)
end
E_path = [p.E - path[1].E for p in path]
i_max = argmax(E_path)
for (t, p, e) in zip(ts, path, E_path)
    @printf("t = %.3f  E − E(0) = %7.4f eV  force on the moving atom = %.3f eV/Å\n", t, e, norm(p.F[:, j]))
end
@printf("unrelaxed straight-line path maximum (not the migration barrier): %.3f eV at t = %.2f\n", E_path[i_max], ts[i_max])

# ## Animation
#
# Left: the (001) plane that holds the vacancy and the moving atom, with the in-plane
# components of the forces (2.5 Å per eV/Å). Right: the energy along the path.

zv = perfect[3, v]
inplane = [k for k in eachindex(keep) if abs(defect[3, k] - zv) < 0.1 && norm(defect[1:2, k] - perfect[1:2, v]) < 1.6a0]
k_now = Observable(1)
pts = @lift(Point2f.(eachcol(path[$k_now].x[1:2, inplane])))
fig = Figure(; size = (840, 420), fontsize = 14)
ax1 = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x (Å)", ylabel = "y (Å)", title = "(001) plane through the vacancy")
scatter!(ax1, [Point2f(perfect[1:2, v])]; marker = :rect, markersize = 18, color = :transparent, strokecolor = :black, strokewidth = 1.5)
scatter!(ax1, pts; markersize = 18, color = [k == j ? :crimson : :darkorange for k in inplane])
arrows2d!(ax1, pts, @lift(Vec2f.(eachcol(2.5 .* path[$k_now].F[1:2, inplane]))); color = :steelblue, shaftwidth = 2, tipwidth = 8, tiplength = 8)
ax2 = Axis(fig[1, 2]; xlabel = "fraction of the path", ylabel = "E − E(0) (eV)")
limits!(ax2, -0.02, 1.02, -0.1 * E_path[i_max], 1.15 * E_path[i_max])
lines!(ax2, @lift(Point2f.(ts[1:$k_now], E_path[1:$k_now])); color = :crimson, linewidth = 2)
scatter!(ax2, @lift([Point2f(ts[$k_now], E_path[$k_now])]); color = :crimson)
text!(
    ax2, ts[i_max], E_path[i_max]; text = @lift($k_now >= i_max ? "unrelaxed straight-line path maximum\n(not the migration barrier)" : ""),
    align = (:center, :bottom), offset = (0, 6), fontsize = 12
)
record(fig, media("vacancy.mp4"), eachindex(path); framerate = 8) do k
    k_now[] = k
end
nothing #hide

# ```@raw html
# <video src="./vacancy.mp4" controls loop muted playsinline width="640"></video>
# ```
