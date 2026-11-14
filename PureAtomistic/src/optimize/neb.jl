# Minimum energy paths and saddle points by the climbing-image nudged elastic band method:
# Henkelman, Uberuaga and Jónsson, J. Chem. Phys. 113, 9901 (2000) (the climbing image), with the
# tangent of Henkelman and Jónsson, J. Chem. Phys. 113, 9978 (2000). The rules below are
# re-derived here and verified by the tests, not transcribed from the papers.
#
# The band is a chain of images R₀, …, R_{M+1} (3 × n positions each) with fixed endpoints. For an
# interior image i with energies V₋ = V_{i-1}, V = V_i, V₊ = V_{i+1}:
#   tangent  τ = R_{i+1} - R_i when V₊ > V > V₋, τ = R_i - R_{i-1} when V₊ < V < V₋; otherwise
#            (a maximum or minimum along the band) τ = w₊ (R_{i+1} - R_i) + w₋ (R_i - R_{i-1})
#            with ΔV_max = max(|V₊ - V|, |V₋ - V|), ΔV_min = min(…), (w₊, w₋) = (ΔV_max, ΔV_min)
#            when V₊ > V₋ and (ΔV_min, ΔV_max) otherwise; then τ is normalized over all 3n
#            coordinates. Taking the uphill neighbor keeps the tangent from kinking where the
#            energy changes fast along the band, and the weighted mix switches smoothly at extrema.
#   force    F = F_true - (F_true · τ) τ + k (|R_{i+1} - R_i| - |R_i - R_{i-1}|) τ: the true force
#            perpendicular to the band plus a spring force along it, which spaces the images
#            equally in the 3n-dimensional distance.
#   climbing the interior image of highest energy has no spring and the parallel component of its
#            true force reversed, F = F_true - 2 (F_true · τ) τ: it climbs along the band and
#            descends across it, so at convergence it sits on the saddle point.
# The fixed atoms and frozen directions of `relax!` are removed from every image's force. The
# interior images are stacked into one 3 × nM matrix and moved by FIRE or FIRE 2.0, whose global
# power F · v and per-atom step cap apply to the whole band; the band force is not the gradient of
# an energy, so the trust-region relaxers, which test energy decrease, do not apply.

"""
    NEBResult{T}

The outcome of [`neb`](@ref): `converged` (the largest atomic band force fell below `fmax`),
`steps` (optimizer iterations; each evaluates every interior image once), `images` (the positions
of every image, endpoints included), `energies` (eV, one per image), `max_force` (largest
Euclidean norm of an atomic band force, eV/Å, over the interior images) and `climbing` (the index
into `images` of the climbing image at the last evaluation, 0 without one). With a converged
climbing image, `energies[climbing] - energies[1]` is the barrier from the first endpoint.
"""
struct NEBResult{T <: AbstractFloat}
    converged::Bool
    steps::Int
    images::Vector{Matrix{T}}
    energies::Vector{T}
    max_force::T
    climbing::Int
end

"""
    neb(images, cell, potential; spring = 0.1, climb = true, fmax = 0.05, maxsteps = 1000,
        fixed = Int[], frozen = Pair{Int, NTuple{3, Float64}}[], opt = nothing) -> NEBResult

Relax the band `images` (a vector of at least three `3 × n` position matrices in Å, endpoints
first and last) toward a minimum energy path of `potential` in the fixed `cell`, by the nudged
elastic band method with the climbing image (Henkelman, Uberuaga and Jónsson, J. Chem. Phys. 113,
9901, 2000) and the tangent of Henkelman and Jónsson (J. Chem. Phys. 113, 9978, 2000). The
endpoints do not move and should be minima; `images` is not modified.

- `spring` (eV/Å²) is the spring constant along the band.
- With `climb`, the interior image of highest energy climbs to the saddle point (chosen anew at
  every evaluation); without it every interior image carries a spring.
- The run stops when the largest atomic band force of the interior images is below `fmax` (eV/Å)
  or after `maxsteps` iterations (`converged = false`).
- `fixed` and `frozen` hold atoms and directions out of every image's motion, as for
  [`relax!`](@ref). Fixing one atom far from the event keeps the band from drifting: the
  perpendicular projection of forces that sum to zero need not sum to zero.
- `opt` is the [`FIRE`](@ref) or [`FIRE2`](@ref) state that moves the stacked interior images,
  built for `n · (length(images) - 2)` atoms; `nothing` builds `FIRE` with its defaults. It is
  reset at the start.

Positions are compared as given: the images must be unwrapped, so that the difference of
neighbors is the displacement along the band, and neighboring images must differ. A non-finite
energy or force throws, naming the image by its position in `images` (1 is the first endpoint).
The positions must be 1-based, as the potentials require.
"""
function neb(
        images::AbstractVector{<:AbstractMatrix{T}}, cell::AbstractMatrix{T}, potential::AbstractPotential;
        spring::Real = 0.1, climb::Bool = true, fmax::Real = 0.05, maxsteps::Integer = 1000,
        fixed = Int[], frozen = Pair{Int, NTuple{3, Float64}}[], opt::Union{Nothing, FIRE{T}, FIRE2{T}} = nothing
    ) where {T <: AbstractFloat}
    length(images) >= 3 ||
        throw(ArgumentError(lazy"pureatomistic: neb needs at least three images (two endpoints and one to move), got $(length(images))"))
    Base.require_one_based_indexing(cell)
    first_image = images[firstindex(images)]
    n = size(first_image, 2)
    for R in images
        Base.require_one_based_indexing(R)
        size(R) == (3, n) ||
            throw(DimensionMismatch(lazy"pureatomistic: every image must have size (3, $n), got $(size(R))"))
    end
    n_atoms(potential) == n ||
        throw(DimensionMismatch(lazy"pureatomistic: the potential has $(n_atoms(potential)) atoms, the images $n"))
    (isfinite(spring) && spring > 0) ||
        throw(ArgumentError(lazy"pureatomistic: the spring constant must be positive and finite, got $spring"))
    (isfinite(fmax) && fmax > 0) ||
        throw(ArgumentError(lazy"pureatomistic: fmax must be positive and finite, got $fmax"))
    maxsteps >= 0 || throw(ArgumentError(lazy"pureatomistic: maxsteps must not be negative, got $maxsteps"))
    m = length(images) - 2
    state = isnothing(opt) ? FIRE{T}(n * m) : opt
    size(state.velocity, 2) == n * m ||
        throw(DimensionMismatch(lazy"pureatomistic: the optimizer state has $(size(state.velocity, 2)) atoms, the band moves $(n * m)"))
    isfixed = _fixed_mask(n, fixed)
    frozen_atoms, frozen_dirs = _frozen_rows(T, n, isfixed, frozen)
    X = Matrix{T}(undef, 3, n * (m + 2))
    for (j, R) in enumerate(images)
        copyto!(view(X, :, ((j - 1) * n + 1):(j * n)), R)
    end
    band = _NEBBand{T}(X, n, m, T(spring), climb, isfixed, frozen_atoms, frozen_dirs)
    converged, steps, fm, ci = _neb!(band, cell, potential, state, T(fmax), Int(maxsteps))
    images_out = [Matrix(_image(band, j)) for j in 0:(m + 1)]
    return NEBResult{T}(converged, steps, images_out, copy(band.energies), fm, ci)
end

# The stacked images (3 × n(M + 2), endpoints in the first and last n columns), the energies, the
# true forces and band forces of the interior images (3 × nM), and the scratch of one image.
struct _NEBBand{T <: AbstractFloat}
    X::Matrix{T}
    n::Int
    m::Int
    spring::T
    climb::Bool
    fixed::Vector{Bool}
    frozen_atoms::Vector{Int}
    frozen_dirs::Matrix{T}
    energies::Vector{T}
    true_forces::Matrix{T}
    forces::Matrix{T}
    scratch::Matrix{T}
    function _NEBBand{T}(X, n, m, spring, climb, fixed, frozen_atoms, frozen_dirs) where {T}
        return new{T}(
            X, n, m, spring, climb, fixed, frozen_atoms, frozen_dirs, zeros(T, m + 2), zeros(T, 3, n * m),
            zeros(T, 3, n * m), zeros(T, 3, n)
        )
    end
end

# Columns of image j (0 = first endpoint, m + 1 = last) in X, and of interior image i in the
# interior force matrices.
_image(b::_NEBBand, j::Int) = view(b.X, :, (j * b.n + 1):((j + 1) * b.n))
_block(b::_NEBBand, F::AbstractMatrix, i::Int) = view(F, :, ((i - 1) * b.n + 1):(i * b.n))

# Evaluate `potential` at image j, store its energy, and copy its forces into F when given. The
# copies are loops: `copyto!` between a view and an array keeps an aliasing branch that allocates.
function _evaluate_image!(b::_NEBBand, potential::AbstractPotential, cell, j::Int, step::Int, F)
    R = b.scratch
    X = b.X
    off = j * b.n
    for a in axes(R, 2), d in axes(R, 1)
        R[d, a] = X[d, off + a]
    end
    prepare!(potential, R, cell)
    energy_forces!(potential, R, cell)
    E = potential.result.energy
    (isfinite(E) && isfinite(_max_force(potential.result.forces))) ||
        throw(ArgumentError(lazy"pureatomistic: neb met a non-finite energy or force at image $(j + 1) after $step steps"))
    b.energies[j + 1] = E
    isnothing(F) || _copy_forces!(F, potential.result.forces)
    return nothing
end

function _copy_forces!(F::AbstractMatrix, G::AbstractMatrix)
    for a in axes(F, 2), d in axes(F, 1)
        F[d, a] = G[d, a]
    end
    return nothing
end

# Weights (w₊, w₋) of the forward and backward differences in the tangent of an image with energy
# V between neighbors of energy Vm (before) and Vp (after). A flat neighborhood weighs both equally.
function _tangent_weights(Vm::T, V::T, Vp::T) where {T}
    Vp > V > Vm && return (one(T), zero(T))
    Vp < V < Vm && return (zero(T), one(T))
    dmax = max(abs(Vp - V), abs(Vm - V))
    dmin = min(abs(Vp - V), abs(Vm - V))
    iszero(dmax) && return (one(T), one(T))
    return Vp > Vm ? (dmax, dmin) : (dmin, dmax)
end

# The band force of every interior image into b.forces, from b.energies and b.true_forces; returns
# the index into the images (1-based, endpoints included) of the climbing image, 0 without one.
function _band_forces!(b::_NEBBand{T}) where {T}
    E = b.energies
    ci = 0
    if b.climb
        ci = 2
        for j in 3:(b.m + 1)
            E[j] > E[ci] && (ci = j)
        end
    end
    for i in 1:b.m
        Rm, R, Rp = _image(b, i - 1), _image(b, i), _image(b, i + 1)
        wp, wm = _tangent_weights(E[i], E[i + 1], E[i + 2])
        tn2 = zero(T)
        dp2 = zero(T)
        dm2 = zero(T)
        for a in axes(R, 2), d in axes(R, 1)
            fwd = Rp[d, a] - R[d, a]
            bwd = R[d, a] - Rm[d, a]
            tn2 += (wp * fwd + wm * bwd)^2
            dp2 += fwd^2
            dm2 += bwd^2
        end
        iszero(tn2) && throw(ArgumentError(lazy"pureatomistic: neb image $(i + 1) has a zero tangent: its neighbors coincide with it"))
        tn = sqrt(tn2)
        Ft = _block(b, b.true_forces, i)
        F = _block(b, b.forces, i)
        par = zero(T)
        for a in axes(R, 2), d in axes(R, 1)
            par += Ft[d, a] * (wp * (Rp[d, a] - R[d, a]) + wm * (R[d, a] - Rm[d, a])) / tn
        end
        # Climbing: the parallel force reversed and no spring; otherwise the parallel force
        # replaced by the spring force.
        c = i + 1 == ci ? -2par : b.spring * (sqrt(dp2) - sqrt(dm2)) - par
        for a in axes(R, 2), d in axes(R, 1)
            τ = (wp * (Rp[d, a] - R[d, a]) + wm * (R[d, a] - Rm[d, a])) / tn
            F[d, a] = Ft[d, a] + c * τ
        end
        _project!(F, b.fixed, b.frozen_atoms, b.frozen_dirs)
    end
    return ci
end

function _evaluate_interior!(b::_NEBBand, potential::AbstractPotential, cell, step::Int)
    for i in 1:b.m
        _evaluate_image!(b, potential, cell, i, step, _block(b, b.true_forces, i))
    end
    return nothing
end

_band_move!(opt::FIRE{T}, X::AbstractMatrix{T}, F::AbstractMatrix{T}) where {T} = (fire_step!(opt, X, F); true)
_band_move!(opt::FIRE2{T}, X::AbstractMatrix{T}, F::AbstractMatrix{T}) where {T} = fire2_step!(opt, X, F)

# The band relaxation in place; returns (converged, steps, largest band force, climbing image).
function _neb!(b::_NEBBand{T}, cell, potential, opt, tol::T, maxsteps::Int)::Tuple{Bool, Int, T, Int} where {T}
    _reset!(opt)
    interior = view(b.X, :, (b.n + 1):((b.m + 1) * b.n))
    _evaluate_image!(b, potential, cell, 0, 0, nothing)
    _evaluate_image!(b, potential, cell, b.m + 1, 0, nothing)
    _evaluate_interior!(b, potential, cell, 0)
    ci = _band_forces!(b)
    fm = _max_force(b.forces)
    for step in 0:(maxsteps - 1)
        fm < tol && return (true, step, fm, ci)
        _band_move!(opt, interior, b.forces) || return (false, step, fm, ci)
        _evaluate_interior!(b, potential, cell, step + 1)
        ci = _band_forces!(b)
        fm = _max_force(b.forces)
    end
    return (fm < tol, maxsteps, fm, ci)
end
