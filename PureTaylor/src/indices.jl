# Multi-indices of N variables up to total degree K, graded by degree (degree 0 first); within one
# degree, descending lexicographic order of the exponent tuple. Used at generation time only.
function multi_indices(N::Int, K::Int)
    out = NTuple{N,Int}[]
    for d in 0:K, α in _compositions(N, d)
        push!(out, α)
    end
    return out
end

# Exponent tuples of length n summing to d, in descending lexicographic order.
function _compositions(n::Int, d::Int)
    n == 0 && return d == 0 ? [()] : Tuple{}[]
    out = NTuple{n,Int}[]
    for e in d:-1:0, r in _compositions(n - 1, d - e)
        push!(out, (e, r...))
    end
    return out
end

# The multi-indices stored by a Taylor{N,K,M}: those whose nonzero exponents lie in the variables
# of mask `M` (bit i-1 set means variable i), in `multi_indices` order.
function stored_indices(N::Int, K::Int, M::Integer)
    return filter(α -> all(i -> α[i] == 0 || isodd(M >> (i - 1)), 1:N), multi_indices(N, K))
end

