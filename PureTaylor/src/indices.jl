# Multi-indices of N variables up to total degree K, graded by degree (degree 0 first); within one
# degree, descending lexicographic order of the exponent tuple. The list is computed when the method
# is generated and returned as a constant tuple.
@generated function multi_indices(::Val{N}, ::Val{K}) where {N, K}
    # Exponent tuples of length n summing to d, in descending lexicographic order.
    compositions(n, d) = n == 0 ? (d == 0 ? [()] : Tuple{}[]) :
        [(e, r...) for e in d:-1:0 for r in compositions(n - 1, d - e)]
    return Tuple(NTuple{N, Int}(α) for k in 0:K for α in compositions(N, k))
end

# The multi-indices stored by a Taylor{N,K,M}: those whose nonzero exponents lie in the variables
# of mask `M` (bit i-1 set means variable i), in `multi_indices` order.
@generated function stored_indices(::Val{N}, ::Val{K}, ::Val{M}) where {N, K, M}
    return Tuple(filter(α -> all(i -> α[i] == 0 || isodd(M >> (i - 1)), 1:N), multi_indices(Val(N), Val(K))))
end
