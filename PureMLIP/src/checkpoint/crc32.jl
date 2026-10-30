function crc32_table()
    t = Vector{UInt32}(undef, 256)
    for i in 0:255
        c = UInt32(i)
        for _ in 1:8
            c = isodd(c) ? (c >> 1) ⊻ 0xEDB88320 : c >> 1
        end
        t[i + 1] = c
    end
    return NTuple{256,UInt32}(t)
end

const CRC32_TABLE = crc32_table()

"""
    crc32(data) -> UInt32

IEEE 802.3 CRC-32 (reflected polynomial `0xEDB88320`, initial value and final XOR
`0xFFFFFFFF`) of a 1-based byte vector.
"""
function crc32(data::AbstractVector{UInt8})
    Base.require_one_based_indexing(data)
    c = 0xFFFFFFFF
    for b in data
        c = CRC32_TABLE[Int((c ⊻ b) & 0xff) + 1] ⊻ (c >> 8)
    end
    return ~c
end
