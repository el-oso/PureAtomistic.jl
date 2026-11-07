# Device forms of a symmetric contraction: the coordinate lists and weights of each contraction,
# and the dense contraction matrices of the species of the device model's elements.

"""
    DeviceContraction{T, IM, V, A3}

A `Contraction{T}` on a device. Entry `q` of `U3`, `((m, w, x, i, k), v)`, is
`U3_index[:, q] == (m, w, x, i, k)` and `U3_value[q] == v`; likewise `U2` (four coordinates) and
`U1` (three). `Wmax`, `W0` and `W1` are the host weights (`C × K × n_species`).
"""
struct DeviceContraction{T, IM <: AbstractMatrix{Int32}, V <: AbstractVector{T}, A3 <: AbstractArray{T, 3}}
    l::Int
    n_ell::Int
    U1_index::IM
    U1_value::V
    U2_index::IM
    U2_value::V
    U3_index::IM
    U3_value::V
    Wmax::A3
    W0::A3
    W1::A3
end

_coordinates(U::Vector{Tuple{NTuple{K, Int}, T}}) where {K, T} = Int32[u[1][a] for a in 1:K, u in U]
_values(U::Vector{Tuple{NTuple{K, Int}, T}}) where {K, T} = T[u[2] for u in U]

function DeviceContraction(backend::Backend, ct::Contraction{T}) where {T}
    up(A) = _upload(backend, A)
    return DeviceContraction(
        ct.l, ct.n_ell, up(_coordinates(ct.U1)), up(_values(ct.U1)), up(_coordinates(ct.U2)), up(_values(ct.U2)),
        up(_coordinates(ct.U3)), up(_values(ct.U3)), up(ct.Wmax), up(ct.W0), up(ct.W1),
    )
end

"""
    DeviceSymmetricContraction{T, IM, V, A3}

The contractions of a `SymmetricContraction{T}` on a device, in output order; `host` gives the
sizes.
"""
struct DeviceSymmetricContraction{T, IM <: AbstractMatrix{Int32}, V <: AbstractVector{T}, A3 <: AbstractArray{T, 3}}
    contractions::Vector{DeviceContraction{T, IM, V, A3}}
    host::SymmetricContraction{T}
end

DeviceSymmetricContraction(backend::Backend, sc::SymmetricContraction{T}) where {T} =
    DeviceSymmetricContraction([DeviceContraction(backend, ct) for ct in sc.contractions], sc)

"""
    DeviceContractionMatrices{T, A4, IM}

The `ContractionMatrices{T}` of a symmetric contraction, built on the host for the species in
`species` and uploaded: `F[:, :, :, j]` and `G[:, :, :, j]` are the forward and pullback matrices
(`features × K_out·L × C`) of species `species[j]`, and `slot[s]` is the `j` of species `s`, or 0
for a species without matrices. `pairs[:, q]` is the `(x, i)` of feature `q`.
"""
struct DeviceContractionMatrices{T, A4 <: AbstractArray{T, 4}, IM <: AbstractMatrix{Int32}}
    C::Int
    L::Int
    K_out::Int
    species::Vector{Int}
    slot::Vector{Int}
    pairs::IM
    F::A4
    G::A4
end

function DeviceContractionMatrices(backend::Backend, sc::SymmetricContraction{T}, species::Vector{Int}) where {T}
    cm = ContractionMatrices{T}(sc)
    for s in species
        contraction_matrices!(cm, sc, s)
    end
    nf, ncol = length(cm.pairs) + cm.L + 1, cm.K_out * cm.L
    F = zeros(T, nf, ncol, cm.C, length(species))
    G = zeros(T, nf, ncol, cm.C, length(species))
    slot = zeros(Int, cm.n_species)
    for (j, s) in pairs(species)
        F[:, :, :, j] = cm.F[s]
        G[:, :, :, j] = cm.G[s]
        slot[s] = j
    end
    pairs_index = Int32[p[a] for a in 1:2, p in cm.pairs]
    return DeviceContractionMatrices(
        cm.C, cm.L, cm.K_out, copy(species), slot, _upload(backend, pairs_index), _upload(backend, F), _upload(backend, G)
    )
end
