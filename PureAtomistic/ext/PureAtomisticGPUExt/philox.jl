# The counter-based generator Philox4x32-10 of J. K. Salmon, M. A. Moraes, R. O. Dror and
# D. E. Shaw, "Parallel random numbers: as easy as 1, 2, 3", SC'11 (2011), section 4.3. A draw is
# a pure function of a 128-bit counter and a 64-bit key, so every work item computes its own
# numbers and no generator state lives on the device. The multipliers are those of section 4.3;
# the round keys follow the Weyl sequence of the paper's equation (6) with the 32-bit constants
# and the word order of the authors' reference implementation (Random123, include/Random123/philox.h,
# BSD-licensed), whose known-answer vectors the tests check.

const _PHILOX_M0 = 0xD2511F53
const _PHILOX_M1 = 0xCD9E8D57
const _PHILOX_W0 = 0x9E3779B9
const _PHILOX_W1 = 0xBB67AE85

# The high and low words of the 64-bit product of two 32-bit words.
@inline function _mulhilo(a::UInt32, b::UInt32)
    p = UInt64(a) * UInt64(b)
    return (p >> 32) % UInt32, p % UInt32
end

# One round: two multiplicative S-boxes on words 1 and 3, whose high halves are xored with the
# other two words and the round key, and the permutation that swaps the inputs of the S-boxes.
@inline function _philox_round(c::NTuple{4, UInt32}, k::NTuple{2, UInt32})
    hi0, lo0 = _mulhilo(_PHILOX_M0, c[1])
    hi1, lo1 = _mulhilo(_PHILOX_M1, c[3])
    return (hi1 ⊻ c[2] ⊻ k[1], lo1, hi0 ⊻ c[4] ⊻ k[2], lo0)
end

"""
    _philox(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32}) -> NTuple{4,UInt32}

Philox4x32-10: ten rounds, the key advanced by the Weyl constants before every round after the
first.
"""
@inline function _philox(ctr::NTuple{4, UInt32}, key::NTuple{2, UInt32})
    c = _philox_round(ctr, key)
    k = key
    for _ in 2:10
        k = (k[1] + _PHILOX_W0, k[2] + _PHILOX_W1)
        c = _philox_round(c, k)
    end
    return c
end

# A uniform number on (0, 1] from the top 24 bits of `hi` (Float32) or the top 53 bits of the
# 64-bit word `hi lo` (Float64): the integer k + 1 for k in [0, 2^b) times 2^-b, which is exact,
# so 0 never occurs and 1 does.
@inline _uniform(::Type{Float32}, hi::UInt32, ::UInt32) = Float32((hi >> 8) + one(UInt32)) * Float32(0x1p-24)
@inline _uniform(::Type{Float64}, hi::UInt32, lo::UInt32) =
    Float64(((UInt64(hi) << 21) ⊻ (UInt64(lo) >> 11)) + one(UInt64)) * 0x1p-53
