@testmodule ZipFixtures begin
    using PureMLIP

    le(x::Integer) = collect(reinterpret(UInt8, [htol(x)]))

    # Build a ZIP archive of uncompressed entries. Local headers carry no extra field.
    # With `zip64`, central records use 0xFFFFFFFF sentinels plus a ZIP64 extra field,
    # and a ZIP64 end-of-central-directory record and locator precede the classic one.
    # `method` is written to both header kinds without compressing the data.
    function stored_zip(entries::Vector{Pair{String,Vector{UInt8}}}; zip64::Bool = true, method::UInt16 = 0x0000)
        out = UInt8[]
        central = UInt8[]
        for (name, data) in entries
            off = length(out)
            crc = PureMLIP.crc32(data)
            nm = collect(codeunits(name))
            sz = length(data)
            append!(out, le(0x04034b50), le(0x000a), le(0x0000), le(method), le(0x0000), le(0x0021))
            append!(out, le(crc), le(UInt32(sz)), le(UInt32(sz)), le(UInt16(length(nm))), le(0x0000), nm, data)
            append!(central, le(0x02014b50), le(0x002d), le(0x000a), le(0x0000), le(method), le(0x0000), le(0x0021), le(crc))
            if zip64
                extra = vcat(le(0x0001), le(UInt16(24)), le(UInt64(sz)), le(UInt64(sz)), le(UInt64(off)))
                append!(central, le(0xffffffff), le(0xffffffff), le(UInt16(length(nm))), le(UInt16(length(extra))))
                append!(central, le(0x0000), le(0x0000), le(0x0000), le(0x00000000), le(0xffffffff), nm, extra)
            else
                append!(central, le(UInt32(sz)), le(UInt32(sz)), le(UInt16(length(nm))), le(0x0000))
                append!(central, le(0x0000), le(0x0000), le(0x0000), le(0x00000000), le(UInt32(off)), nm)
            end
        end
        cdoff = length(out)
        append!(out, central)
        n = length(entries)
        if zip64
            z64off = length(out)
            append!(out, le(0x06064b50), le(UInt64(44)), le(0x002d), le(0x002d), le(0x00000000), le(0x00000000))
            append!(out, le(UInt64(n)), le(UInt64(n)), le(UInt64(length(central))), le(UInt64(cdoff)))
            append!(out, le(0x07064b50), le(0x00000000), le(UInt64(z64off)), le(0x00000001))
            append!(out, le(0x06054b50), le(0x0000), le(0x0000), le(0xffff), le(0xffff), le(0xffffffff), le(0xffffffff), le(0x0000))
        else
            append!(out, le(0x06054b50), le(0x0000), le(0x0000), le(UInt16(n)), le(UInt16(n)))
            append!(out, le(UInt32(length(central))), le(UInt32(cdoff)), le(0x0000))
        end
        return out
    end
end
