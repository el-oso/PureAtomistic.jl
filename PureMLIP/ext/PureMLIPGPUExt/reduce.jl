# Sums on the device in a fixed order, so that a sum does not depend on how the work items are
# scheduled and an evaluation repeats bit for bit.

const _REDUCE_GROUP = 256

"""
    _sum!(out, x, n, backend) -> nothing

`out[1] = Σ_{i ≤ n} x[i]` in a fixed-shape tree: work item `t` of one group of 256 adds
`x[t], x[t + 256], …` in order, then the 256 partial sums are added pairwise in halving steps.
"""
function _sum!(out::AbstractVector{T}, x::AbstractVector{T}, n::Int, backend::Backend) where {T}
    0 <= n <= length(x) || throw(ArgumentError(lazy"gpu: the sum of $n entries of a vector of length $(length(x))"))
    _sum_kernel!(backend, _REDUCE_GROUP)(out, x, n; ndrange = _REDUCE_GROUP)
    return nothing
end

# The tree needs group memory and barriers, which only a kernel body can hold; its arithmetic is
# the additions written here.
@kernel function _sum_kernel!(out, @Const(x), n)
    t = @index(Local)
    buf = @localmem eltype(out) (_REDUCE_GROUP,)
    acc = zero(eltype(out))
    for i in t:_REDUCE_GROUP:n
        acc += x[i]
    end
    buf[t] = acc
    @synchronize
    h = _REDUCE_GROUP ÷ 2
    while h >= 1
        if t <= h
            buf[t] += buf[t + h]
        end
        @synchronize
        h ÷= 2
    end
    if t == 1
        out[1] = buf[1]
    end
end

function PureMLIP.device_sum!(out::AbstractVector{T}, x::AbstractVector{T}, n::Integer) where {T}
    backend = get_backend(x)
    get_backend(out) == backend || throw(ArgumentError("gpu: all arrays must be on the backend of the summed vector"))
    _sum!(out, x, Int(n), backend)
    return out
end
