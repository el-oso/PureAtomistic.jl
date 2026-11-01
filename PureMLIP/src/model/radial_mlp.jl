"""
    normalized_silu(z, cst) -> (a, da)

e3nn's `normalize2mom(silu)`: `a = cst · z · sigmoid(z)` and `da = ∂a/∂z`.
"""
function normalized_silu(z::S, cst) where {S <: Real}
    c = oftype(z, cst)
    σ = inv(one(z) + exp(-z))
    s = z * σ
    return (s * c, c * (σ + s * (one(z) - σ)))
end

"""
    RadialMLP{T}(layers, cst)

e3nn `FullyConnectedNet` with three hidden layers of one width and `normalize2mom(silu)`
activations (factor `cst`), no biases. `layers[k]` is torch layer `k` as a Julia `[h_in, h_out]`
matrix; it is stored divided by `sqrt(h_in)`, the order in which e3nn applies the scale.
"""
struct RadialMLP{T}
    W::NTuple{4, Matrix{T}}
    cst::T
    function RadialMLP{T}(layers::NTuple{4, AbstractMatrix{<:Real}}, cst::Real) where {T}
        for k in 2:4
            size(layers[k], 1) == size(layers[k - 1], 2) ||
                throw(ArgumentError(lazy"model: radial MLP layer $k has $(size(layers[k], 1)) inputs, expected $(size(layers[k - 1], 2))"))
        end
        h = size(layers[1], 2)
        for k in 2:3
            size(layers[k], 2) == h ||
                throw(ArgumentError(lazy"model: radial MLP hidden layers must share one width, got $h and $(size(layers[k], 2))"))
        end
        W = ntuple(k -> Matrix{T}(layers[k] ./ sqrt(size(layers[k], 1))), Val(4))
        return new{T}(W, T(cst))
    end
end

function _check_mlp_hidden(mlp::RadialMLP, A::AbstractArray, nb::Int, depth::Int, what::String)
    sz = (size(mlp.W[1], 2), nb, depth)
    size(A) == sz || throw(DimensionMismatch(lazy"model: radial MLP $what size $(size(A)), expected $sz"))
    return nothing
end

function _check_mlp(mlp::RadialMLP, out::AbstractMatrix, pre::AbstractArray, act::AbstractArray, feats::AbstractMatrix)
    Base.require_one_based_indexing(out, pre, act, feats)
    nb = size(feats, 2)
    size(feats, 1) == size(mlp.W[1], 1) ||
        throw(DimensionMismatch(lazy"model: radial MLP features have $(size(feats, 1)) rows, expected $(size(mlp.W[1], 1))"))
    size(out) == (size(mlp.W[4], 2), nb) ||
        throw(DimensionMismatch(lazy"model: radial MLP output has size $(size(out)), expected $((size(mlp.W[4], 2), nb))"))
    _check_mlp_hidden(mlp, pre, nb, 3, "logistic buffer has")
    _check_mlp_hidden(mlp, act, nb, 3, "activations have")
    return nothing
end

function _check_mlp_pullback(mlp::RadialMLP, dfeats::AbstractMatrix, g::AbstractArray, dout::AbstractMatrix, pre::AbstractArray, act::AbstractArray)
    Base.require_one_based_indexing(dfeats, g, dout, pre, act)
    nb = size(dout, 2)
    size(dout, 1) == size(mlp.W[4], 2) ||
        throw(DimensionMismatch(lazy"model: radial MLP output cotangent has $(size(dout, 1)) rows, expected $(size(mlp.W[4], 2))"))
    size(dfeats) == (size(mlp.W[1], 1), nb) ||
        throw(DimensionMismatch(lazy"model: radial MLP feature gradient has size $(size(dfeats)), expected $((size(mlp.W[1], 1), nb))"))
    _check_mlp_hidden(mlp, pre, nb, 3, "logistic buffer has")
    _check_mlp_hidden(mlp, act, nb, 3, "activations have")
    _check_mlp_hidden(mlp, g, nb, 2, "scratch has")
    return nothing
end

# One hidden layer at the pre-activation z = Wᵀ input: act = normalized_silu(z) = cst·z·σ(z), and
# `pre` is overwritten with the logistic σ(z), from which the pullback gets the slope without `exp`.
function _mlp_layer!(pre::AbstractMatrix, act::AbstractMatrix, W::AbstractMatrix, input::AbstractMatrix, cst)
    mul!(pre, transpose(W), input)
    c = convert(eltype(act), cst)
    for q in eachindex(pre, act)
        z = pre[q]
        σ = inv(one(z) + exp(-z))
        act[q] = z * σ * c
        pre[q] = σ
    end
    return nothing
end

# g .*= ∂act/∂z = cst·σ + act·(1 - σ), turning an activation cotangent into a pre-activation
# cotangent; `σs` and `act` are the `pre` and `act` of `_mlp_layer!`.
function _mlp_layer_pullback!(g::AbstractMatrix, σs::AbstractMatrix, act::AbstractMatrix, cst)
    c = convert(eltype(g), cst)
    for q in checked_eachindex(g, σs, act)
        σ = σs[q]
        g[q] *= c * σ + act[q] * (one(σ) - σ)
    end
    return nothing
end

"""
    radial_mlp!(out, pre, act, mlp::RadialMLP, feats) -> out

`out = mlp(feats)` for `feats` (`n_basis × nb`) and `out` (`W_out × nb`), one column per edge.
`pre[:, :, k]` and `act[:, :, k]` (`hidden × nb × 3`) receive the logistic `σ(z)` of the
pre-activations `z` and the activations `cst · z · σ(z)` of hidden layer `k`, which
[`radial_mlp_pullback!`](@ref) reads.
"""
function radial_mlp!(out::AbstractMatrix{S}, pre::AbstractArray{S, 3}, act::AbstractArray{S, 3}, mlp::RadialMLP, feats::AbstractMatrix{S}) where {S}
    _check_mlp(mlp, out, pre, act, feats)
    _mlp_layer!(view(pre, :, :, 1), view(act, :, :, 1), mlp.W[1], feats, mlp.cst)
    _mlp_layer!(view(pre, :, :, 2), view(act, :, :, 2), mlp.W[2], view(act, :, :, 1), mlp.cst)
    _mlp_layer!(view(pre, :, :, 3), view(act, :, :, 3), mlp.W[3], view(act, :, :, 2), mlp.cst)
    mul!(out, transpose(mlp.W[4]), view(act, :, :, 3))
    return out
end

"""
    radial_mlp_pullback!(dfeats, g, mlp::RadialMLP, dout, pre, act) -> dfeats

The vector-Jacobian product of [`radial_mlp!`](@ref) with the cotangent `dout`, from the `pre`
(the logistic `σ`) and `act` that call stored: the slope of each activation is
`cst · σ + act · (1 - σ)`. `g` (`hidden × nb × 2`) is scratch. Overwrites `dfeats`.
"""
function radial_mlp_pullback!(dfeats::AbstractMatrix{S}, g::AbstractArray{S, 3}, mlp::RadialMLP, dout::AbstractMatrix{S}, pre::AbstractArray{S, 3}, act::AbstractArray{S, 3}) where {S}
    _check_mlp_pullback(mlp, dfeats, g, dout, pre, act)
    g1, g2 = view(g, :, :, 1), view(g, :, :, 2)
    mul!(g1, mlp.W[4], dout)
    _mlp_layer_pullback!(g1, view(pre, :, :, 3), view(act, :, :, 3), mlp.cst)
    mul!(g2, mlp.W[3], g1)
    _mlp_layer_pullback!(g2, view(pre, :, :, 2), view(act, :, :, 2), mlp.cst)
    mul!(g1, mlp.W[2], g2)
    _mlp_layer_pullback!(g1, view(pre, :, :, 1), view(act, :, :, 1), mlp.cst)
    mul!(dfeats, mlp.W[1], g1)
    return dfeats
end

"""
    density_edge(feats, w) -> (ρ, c)

MACE's edge density `ρ = tanh(z²)`, `z = feats · (w / sqrt(n_basis))`, and `c` with
`∂ρ/∂feats = c .* w`.
"""
function density_edge(feats::AbstractVector{S}, w::AbstractVector) where {S <: Real}
    z = zero(S)
    r = sqrt(oftype(z, length(feats)))
    for b in eachindex(feats, w)
        z += feats[b] * (oftype(z, w[b]) / r)
    end
    ρ = tanh(z * z)
    return (ρ, 2 * z * (one(S) - ρ * ρ) / r)
end
