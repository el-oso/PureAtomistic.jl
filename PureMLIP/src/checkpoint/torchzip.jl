# Reader for the ZIP container of a PyTorch checkpoint: STORED and DEFLATE entries,
# with optional ZIP64 records.

using Inflate: Inflate

struct ZipEntry
    name::String
    data_offset::Int   # 1-based index of the first data byte in the archive buffer
    size::Int               # uncompressed size
    compressed_size::Int
    method::UInt16          # 0 (STORED) or 8 (DEFLATE)
    crc::UInt32             # CRC-32 of the uncompressed data
end

struct TorchArchive
    bytes::Vector{UInt8}
    prefix::String                     # folder holding data.pkl, with a trailing '/'
    entries::Dict{String, ZipEntry}
    verified::Set{String}              # entries whose CRC has been checked
    inflated::Dict{String, Vector{UInt8}}   # decoded DEFLATE entries
end

const SIG_LOCAL = 0x04034b50
const SIG_CENTRAL = 0x02014b50
const SIG_EOCD = 0x06054b50
const SIG_ZIP64_LOCATOR = 0x07064b50
const SIG_ZIP64_EOCD = 0x06064b50

# Little-endian integer of type T at 1-based position `pos`.
function read_le(bytes::Vector{UInt8}, pos::Integer, ::Type{T}) where {T <: Unsigned}
    (pos >= 1 && pos + sizeof(T) - 1 <= length(bytes)) ||
        throw(ArgumentError(lazy"truncated zip: cannot read $(sizeof(T)) bytes at byte $pos of $(length(bytes))"))
    x = zero(T)
    for k in 0:(sizeof(T) - 1)
        x |= T(bytes[pos + k]) << (8k)
    end
    return x
end

# Largest accepted zip size or offset: 2^48 bytes (256 TiB) exceeds any archive, and it leaves
# Int headroom for the offset + size arithmetic on positions.
const MAX_ZIP_FIELD = UInt64(1) << 48

# Unsigned little-endian field read as an Int. Sizes and offsets above `MAX_ZIP_FIELD` cannot
# address any archive; rejecting them keeps position arithmetic from overflowing.
function read_le_int(bytes::Vector{UInt8}, pos::Integer, ::Type{T}) where {T <: Unsigned}
    x = read_le(bytes, pos, T)
    x <= MAX_ZIP_FIELD || throw(ArgumentError(lazy"zip field at byte $pos is out of range: $x"))
    return Int(x)
end

# Int value of a size or offset parsed from an entry header (see `read_le_int`).
function header_int(x::UInt64, what::AbstractString, name::AbstractString)
    x <= MAX_ZIP_FIELD || throw(ArgumentError(lazy"zip entry $name has an out-of-range $what: $x"))
    return Int(x)
end

function find_eocd(bytes::Vector{UInt8})
    last_start = length(bytes) - 21
    first_start = max(1, last_start - 0xffff)
    for p in last_start:-1:first_start
        read_le(bytes, p, UInt32) == SIG_EOCD && return p
    end
    throw(ArgumentError("not a zip archive: no end-of-central-directory record"))
end

function open_torch_archive(bytes::Vector{UInt8})
    eocd = find_eocd(bytes)
    nentries = Int(read_le(bytes, eocd + 10, UInt16))
    cd_offset = Int(read_le(bytes, eocd + 16, UInt32))
    sentinel = nentries == 0xffff || read_le(bytes, eocd + 16, UInt32) == 0xffffffff
    if sentinel || (eocd > 20 && read_le(bytes, eocd - 20, UInt32) == SIG_ZIP64_LOCATOR)
        eocd > 20 && read_le(bytes, eocd - 20, UInt32) == SIG_ZIP64_LOCATOR ||
            throw(ArgumentError("zip64 end-of-central-directory locator missing"))
        z64 = read_le_int(bytes, eocd - 20 + 8, UInt64) + 1
        read_le(bytes, z64, UInt32) == SIG_ZIP64_EOCD ||
            throw(ArgumentError(lazy"no zip64 end-of-central-directory record at byte $z64"))
        nentries = read_le_int(bytes, z64 + 32, UInt64)
        cd_offset = read_le_int(bytes, z64 + 48, UInt64)
    end

    entries = Dict{String, ZipEntry}()
    pkl_name = ""
    pos = cd_offset + 1
    for _ in 1:nentries
        read_le(bytes, pos, UInt32) == SIG_CENTRAL ||
            throw(ArgumentError(lazy"bad central directory record at byte $pos"))
        method = read_le(bytes, pos + 10, UInt16)
        crc = read_le(bytes, pos + 16, UInt32)
        csize = UInt64(read_le(bytes, pos + 20, UInt32))
        usize = UInt64(read_le(bytes, pos + 24, UInt32))
        nlen = Int(read_le(bytes, pos + 28, UInt16))
        xlen = Int(read_le(bytes, pos + 30, UInt16))
        clen = Int(read_le(bytes, pos + 32, UInt16))
        local_off = UInt64(read_le(bytes, pos + 42, UInt32))
        name_end = pos + 46 + nlen - 1
        name_end <= length(bytes) || throw(ArgumentError(lazy"truncated zip: entry name at byte $pos"))
        name = String(bytes[(pos + 46):name_end])

        # ZIP64 extra field: values appear in order, only for sentinel-valued fields.
        need_u = usize == 0xffffffff
        need_c = csize == 0xffffffff
        need_o = local_off == 0xffffffff
        if need_u || need_c || need_o
            q = name_end + 1
            xend = q + xlen - 1
            found = false
            while q + 3 <= xend
                id = read_le(bytes, q, UInt16)
                len = Int(read_le(bytes, q + 2, UInt16))
                if id == 0x0001
                    r = q + 4
                    if need_u
                        usize = read_le(bytes, r, UInt64); r += 8
                    end
                    if need_c
                        csize = read_le(bytes, r, UInt64); r += 8
                    end
                    if need_o
                        local_off = read_le(bytes, r, UInt64); r += 8
                    end
                    found = true
                    break
                end
                q += 4 + len
            end
            found || throw(ArgumentError(lazy"zip64 extra field missing for entry $name"))
        end

        (method == 0 || method == 8) ||
            throw(ArgumentError(lazy"compressed entries are not supported: $name (method $method)"))
        method == 0 && csize != usize && throw(ArgumentError(lazy"stored entry $name has differing sizes: $csize vs $usize"))
        lh = header_int(local_off, "offset", name) + 1
        read_le(bytes, lh, UInt32) == SIG_LOCAL ||
            throw(ArgumentError(lazy"bad local header for entry $name at byte $lh"))
        data_offset = lh + 30 + Int(read_le(bytes, lh + 26, UInt16)) + Int(read_le(bytes, lh + 28, UInt16))
        sz = header_int(usize, "size", name)
        csz = header_int(csize, "compressed size", name)
        data_offset + csz - 1 <= length(bytes) ||
            throw(ArgumentError(lazy"entry $name extends past the end of the archive (bytes $data_offset to $(data_offset + csz - 1) of $(length(bytes)))"))
        haskey(entries, name) && throw(ArgumentError(lazy"duplicate zip entry: $name"))
        # DEFLATE expands by at most 1032:1 (a run of 258 bytes costs 2 bits), so a larger
        # declared size cannot be honest.
        (method == 0 || sz <= DEFLATE_MAX_RATIO * csz) ||
            throw(ArgumentError(lazy"checkpoint: entry $name declares $sz bytes, more than DEFLATE can produce from its $csz compressed bytes"))
        entries[name] = ZipEntry(name, data_offset, sz, csz, method, crc)
        if name == "data.pkl" || endswith(name, "/data.pkl")
            isempty(pkl_name) || throw(ArgumentError(lazy"more than one data.pkl entry in the archive: $pkl_name and $name"))
            pkl_name = name
        end
        pos = name_end + 1 + xlen + clen
    end
    isempty(pkl_name) && throw(ArgumentError("no data.pkl entry in the archive"))
    prefix = pkl_name[1:(end - ncodeunits("data.pkl"))]
    return TorchArchive(bytes, prefix, entries, Set{String}(), Dict{String, Vector{UInt8}}())
end

# Largest output of DEFLATE per input byte.
const DEFLATE_MAX_RATIO = 1032

# Decompresses the DEFLATE data `raw` and throws as soon as the output would pass `size`
# bytes, so the output never grows beyond `size`. This is Inflate's in-memory block loop with
# a bound on every append: Inflate's own `inflate` has no limit, and its streaming reader
# holds an abstract `IO`, which makes every caller type-unstable.
function inflate_bounded(raw::Vector{UInt8}, size::Int, name::AbstractString)
    too_big() = throw(ArgumentError(lazy"checkpoint: entry $name inflates past its declared size of $size bytes"))
    data = Inflate.InflateData(raw)
    out = UInt8[]
    final_block = false
    while !final_block
        final_block = Inflate.getbits(data, 1) == 1
        mode = Inflate.getbits(data, 2)
        if mode == 0
            Inflate.skip_bits_to_byte_boundary(data)
            len = Inflate.getbits(data, 16)
            nlen = Inflate.getbits(data, 16)
            len ⊻ nlen == 0xffff || throw(ArgumentError(lazy"checkpoint: entry $name has a corrupt DEFLATE stored block"))
            length(out) + len <= size || too_big()
            append!(out, Inflate.get_input_bytes(data, len))
            continue
        elseif mode == 1
            data.literal_or_length_code = Inflate.fixed_literal_or_length_table
            data.distance_code = Inflate.fixed_distance_table
        elseif mode == 2
            Inflate.read_code_tables(data)
        else
            throw(ArgumentError(lazy"checkpoint: entry $name has a DEFLATE block of invalid type 3"))
        end
        while true
            v = Inflate.get_literal_or_length(data)
            if v < 256
                length(out) < size || too_big()
                push!(out, UInt8(v))
            elseif v == 256
                break
            else
                len = Inflate.getlength(data, v)
                distance = Inflate.getdist(data)
                distance <= length(out) ||
                    throw(ArgumentError(lazy"checkpoint: entry $name has a DEFLATE match reaching before its output"))
                length(out) + len <= size || too_big()
                # A match may overlap its own output, so it is copied one byte at a time.
                for _ in 1:len
                    push!(out, out[end - distance + 1])
                end
            end
        end
    end
    return out
end


"""
    entry_data(a::TorchArchive, name) -> view

Bytes of the entry `a.prefix * name`: a view of the archive buffer for a STORED entry, a
view of the decoded bytes for a DEFLATE entry (decoded once and cached in `a.inflated`).
The CRC-32 is checked on first access to each entry.
"""
function entry_data(a::TorchArchive, name::AbstractString)
    full = a.prefix * name
    e = get(a.entries, full, nothing)
    isnothing(e) && throw(ArgumentError(lazy"no entry $full in the archive"))
    if e.method == 0x0008
        out = get(a.inflated, full, nothing)
        if isnothing(out)
            raw = a.bytes[e.data_offset:(e.data_offset + e.compressed_size - 1)]
            out = inflate_bounded(raw, e.size, full)
            length(out) == e.size || throw(ArgumentError(lazy"inflated size mismatch in $full"))
            crc32(out) == e.crc || throw(ArgumentError(lazy"CRC mismatch in $full"))
            a.inflated[full] = out
            push!(a.verified, full)
        end
        return view(out, 1:length(out))
    end
    v = view(a.bytes, e.data_offset:(e.data_offset + e.size - 1))
    if !(full in a.verified)
        crc32(v) == e.crc || throw(ArgumentError(lazy"CRC mismatch in $full"))
        push!(a.verified, full)
    end
    return v
end
