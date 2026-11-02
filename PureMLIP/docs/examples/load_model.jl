# # Loading a model
#
# This example loads MACE-MPA-0, prints the facts the model reads from the checkpoint, and
# shows how the radial features of one edge change as one Cu atom moves away from another.

using PureMLIP, LinearAlgebra, Printf
using GLMakie
include(pkgdir(PureMLIP, "docs", "examples", "setup.jl")); #hide

# `checkpoint_path(:mpa0)` downloads `mace-mpa-0-medium.model` on first use and checks its
# SHA-256; any path to the file works in its place.

ck = load_mace_checkpoint(checkpoint_path(:mpa0))
model = MACEModel(ck)
@printf("elements: %d, from Z = %d to Z = %d\n", length(model.z_table), first(model.z_table), last(model.z_table))
@printf("cutoff r_max = %.2f Å\n", model.r_max)
@printf("reference energy of Cu: %.4f eV\n", model.e0[findfirst(==(29), model.z_table)])

# ## Radial features of an edge
#
# Each edge of length `d` enters the model through a Bessel basis on an Agnesi-transformed
# distance, multiplied by a polynomial cutoff that goes smoothly to zero at `r_max`. The
# function that evaluates it, `PureMLIP.Model.radial_embedding!`, is internal (not part of the
# interface); it is used here only to draw the features.

emb = model.radial
nb = length(emb.bessel_weights)
ds = range(1.5, 6.5; length = 251)
features = zeros(nb, length(ds))
for (j, d) in pairs(ds)
    PureMLIP.Model.radial_embedding!(view(features, :, j), nothing, emb, d, 29, 29)
end
cutoff = [first(PureMLIP.Model.polynomial_cutoff(d, emb.r_max, emb.cutoff_p)) for d in ds]
@printf("%d radial features; all are zero beyond %.2f Å: %s\n", nb, emb.r_max, all(iszero, features[:, ds .>= emb.r_max]))

# ## Energy of a Cu dimer
#
# Two Cu atoms in a 20 Å periodic box: periodic images are more than `r_max` apart for every
# separation drawn here. The energy is measured from the energy of two separate atoms, so it
# is zero once the atoms are farther apart than `r_max`.

box = Matrix(20.0I, 3, 3)
dimer(d) = [10.0 (10.0 + d); 10.0 10.0; 10.0 10.0]
E_far = mace_energy_forces(model, dimer(7.0), box, [29, 29]).energy
dd = range(1.8, 6.5; length = 48)
E = [mace_energy_forces(model, dimer(d), box, [29, 29]).energy - E_far for d in dd]
i = argmin(E)
@printf("dimer energy minimum on this grid: %.3f eV at d = %.2f Å\n", E[i], dd[i])
@printf("energy at d = 6.2 Å: %.1e eV\n", mace_energy_forces(model, dimer(6.2), box, [29, 29]).energy - E_far)

# ## Animation
#
# The second atom moves from 1.8 Å to 6.5 Å. Top: the two atoms, with the
# range within r_max of the first shaded. Middle: the radial features, with their values at
# the current distance marked.
# Bottom: the dimer energy up to the current distance.

d_now = Observable(dd[1])
fig = Figure(; size = (720, 720), fontsize = 14)
ax1 = Axis(fig[1, 1]; title = "Cu–Cu edge", xlabel = "x (Å)", yticksvisible = false, yticklabelsvisible = false)
vspan!(ax1, [0.0], [emb.r_max]; color = (:steelblue, 0.12))
vlines!(ax1, [emb.r_max]; color = :steelblue, linestyle = :dash)
scatter!(ax1, [Point2f(0, 0)]; markersize = 22, color = :darkorange)
scatter!(ax1, @lift([Point2f($d_now, 0)]); markersize = 22, color = :darkorange)
text!(ax1, emb.r_max - 0.1, -0.9; text = "r_max", align = (:right, :bottom), color = :steelblue)
limits!(ax1, -1, 7.5, -1.2, 1.2)
rowsize!(fig.layout, 1, Relative(0.18))
ax2 = Axis(fig[2, 1]; xlabel = "d (Å)", ylabel = "radial feature")
for n in 1:nb
    lines!(ax2, ds, features[n, :])
end
lines!(ax2, ds, cutoff; color = :black, linestyle = :dash, label = "cutoff")
vlines!(ax2, @lift([$d_now]); color = :gray)
feat_now = @lift begin
    f = zeros(nb)
    PureMLIP.Model.radial_embedding!(f, nothing, emb, $d_now, 29, 29)
    Point2f.($d_now, f)
end
scatter!(ax2, feat_now; color = :black, markersize = 7)
axislegend(ax2; position = :rt)
ax3 = Axis(fig[3, 1]; xlabel = "d (Å)", ylabel = "E − E(∞) (eV)")
limits!(ax3, 1.5, 6.6, minimum(E) - 0.2, max(maximum(E), 0.5) + 0.2)
k_now = Observable(1)
lines!(ax3, @lift(Point2f.(dd[1:$k_now], E[1:$k_now])); color = :darkorange, linewidth = 2)
hlines!(ax3, [0.0]; color = :gray, linestyle = :dot)
vlines!(ax3, [emb.r_max]; color = :steelblue, linestyle = :dash)
linkxaxes!(ax2, ax3)
record(fig, media("load_model.mp4"), eachindex(dd); framerate = 16) do k
    k_now[] = k
    d_now[] = dd[k]
end
nothing #hide

# ```@raw html
# <video src="./load_model.mp4" controls loop muted playsinline width="560"></video>
# ```
