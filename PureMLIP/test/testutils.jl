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

    # Archive laid out as torch.save writes it: t/data.pkl, t/byteorder, and one
    # t/data/<key> entry per storage.
    function zip_with_pickle(pkl::Vector{UInt8}, storages::Vector{Pair{String,Vector{UInt8}}}; byteorder::String = "little")
        entries = Pair{String,Vector{UInt8}}["t/data.pkl" => pkl, "t/byteorder" => collect(codeunits(byteorder))]
        for (key, bytes) in storages
            push!(entries, "t/data/" * key => bytes)
        end
        return stored_zip(entries)
    end

    # Protocol-2 pickle fragments; each returns the bytes that leave one value on the stack.
    cat(xs...) = reduce(vcat, xs; init = UInt8[])
    pk_str(s::String) = UInt8[UInt8('X'), le(UInt32(ncodeunits(s)))..., codeunits(s)...]
    pk_int(n::Integer) = UInt8[UInt8('J'), le(Int32(n))...]
    pk_bool(b::Bool) = UInt8[b ? 0x88 : 0x89]
    pk_none() = UInt8[UInt8('N')]
    pk_global(m::String, n::String) = UInt8[UInt8('c'), codeunits(m)..., UInt8('\n'), codeunits(n)..., UInt8('\n')]
    pk_tuple(xs::Vector{UInt8}...) = UInt8[UInt8('('), cat(xs...)..., UInt8('t')]
    pk_list(xs::Vector{UInt8}...) = UInt8[UInt8(']'), UInt8('('), cat(xs...)..., UInt8('e')]
    pk_dict(kvs::Pair{String,Vector{UInt8}}...) =
        UInt8[UInt8('}'), UInt8('('), cat([vcat(pk_str(k), v) for (k, v) in kvs]...)..., UInt8('u')]
    pk_odict() = UInt8[pk_global("collections", "OrderedDict")..., 0x29, UInt8('R')]
    pk_call(callable::Vector{UInt8}, args::Vector{UInt8}) = UInt8[callable..., args..., UInt8('R')]
    pk_build(obj::Vector{UInt8}, state::Vector{UInt8}) = UInt8[obj..., state..., UInt8('b')]
    pk_storage(cls::String, key::String, n::Integer) =
        UInt8[pk_tuple(pk_str("storage"), pk_global("torch", cls), pk_str(key), pk_str("cpu"), pk_int(n))..., UInt8('Q')]
    pk_tensor(storage::Vector{UInt8}, offset::Integer, size::Vector{Int}, stride::Vector{Int}) =
        pk_call(pk_global("torch._utils", "_rebuild_tensor_v2"),
                pk_tuple(storage, pk_int(offset), pk_tuple([pk_int(s) for s in size]...), pk_tuple([pk_int(s) for s in stride]...),
                         pk_bool(false), pk_odict()))
    pk_param(tensor::Vector{UInt8}, requires_grad::Bool) =
        pk_call(pk_global("torch._utils", "_rebuild_parameter"), pk_tuple(tensor, pk_bool(requires_grad), pk_odict()))
    # A module body with the three dicts torch keeps; `nonpersistent` names buffers to omit.
    function pk_module_state(; params = Pair{String,Vector{UInt8}}[], buffers = Pair{String,Vector{UInt8}}[],
                             modules = Pair{String,Vector{UInt8}}[], nonpersistent::Union{Nothing,Vector{String}} = nothing)
        kvs = Pair{String,Vector{UInt8}}["_parameters" => pk_dict(params...), "_buffers" => pk_dict(buffers...), "_modules" => pk_dict(modules...)]
        isnothing(nonpersistent) || push!(kvs, "_non_persistent_buffers_set" =>
            pk_call(pk_global("__builtin__", "set"), pk_tuple(pk_list([pk_str(s) for s in nonpersistent]...))))
        return pk_dict(kvs...)
    end
    pk_module(state::Vector{UInt8}) = pk_build(pk_call(pk_global("torch.nn.modules.container", "ModuleList"), UInt8[0x29]), state)
    pk_finish(root::Vector{UInt8}) = UInt8[0x80, 0x02, root..., UInt8('.')]
    f64bytes(xs) = cat([le(reinterpret(UInt64, Float64(x))) for x in xs]...)
    i64bytes(xs) = cat([le(reinterpret(UInt64, Int64(x))) for x in xs]...)
end
