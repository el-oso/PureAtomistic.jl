@testmodule ZipFixtures begin
    using PureMLIP

    le(x::Integer) = collect(reinterpret(UInt8, [htol(x)]))

    # Build a ZIP archive of uncompressed entries. Local headers carry an extra field of
    # `local_extra` bytes (padding, as torch.save writes to align data), independent of the
    # central records' extra field.
    # With `zip64`, central records use 0xFFFFFFFF sentinels plus a ZIP64 extra field,
    # and a ZIP64 end-of-central-directory record and locator precede the classic one.
    # `method` is written to both header kinds without compressing the data.
    # Entries named in `deflate` are stored with method 8 as one final DEFLATE "stored
    # block" (at most 65535 bytes): header 0x01, LEN, NLEN, then the data.
    function stored_zip(entries::Vector{Pair{String, Vector{UInt8}}}; zip64::Bool = true, method::UInt16 = 0x0000, deflate::Vector{String} = String[], local_extra::Int = 0)
        out = UInt8[]
        central = UInt8[]
        for (name, data) in entries
            off = length(out)
            crc = PureMLIP.Checkpoint.crc32(data)
            nm = collect(codeunits(name))
            sz = length(data)
            m = method
            payload = data
            if name in deflate
                sz <= 0xffff || throw(ArgumentError("deflate fixture entries are limited to 65535 bytes"))
                m = 0x0008
                payload = vcat(UInt8[0x01], le(UInt16(sz)), le(~UInt16(sz)), data)
            end
            csz = length(payload)
            append!(out, le(0x04034b50), le(0x000a), le(0x0000), le(m), le(0x0000), le(0x0021))
            append!(out, le(crc), le(UInt32(csz)), le(UInt32(sz)), le(UInt16(length(nm))), le(UInt16(local_extra)), nm, fill(0xab, local_extra), payload)
            append!(central, le(0x02014b50), le(0x002d), le(0x000a), le(0x0000), le(m), le(0x0000), le(0x0021), le(crc))
            if zip64
                extra = vcat(le(0x0001), le(UInt16(24)), le(UInt64(sz)), le(UInt64(csz)), le(UInt64(off)))
                append!(central, le(0xffffffff), le(0xffffffff), le(UInt16(length(nm))), le(UInt16(length(extra))))
                append!(central, le(0x0000), le(0x0000), le(0x0000), le(0x00000000), le(0xffffffff), nm, extra)
            else
                append!(central, le(UInt32(csz)), le(UInt32(sz)), le(UInt16(length(nm))), le(0x0000))
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
    function zip_with_pickle(pkl::Vector{UInt8}, storages::Vector{Pair{String, Vector{UInt8}}}; byteorder::String = "little")
        entries = Pair{String, Vector{UInt8}}["t/data.pkl" => pkl, "t/byteorder" => collect(codeunits(byteorder))]
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
    pk_dict(kvs::Pair{String, Vector{UInt8}}...) =
        UInt8[UInt8('}'), UInt8('('), cat([vcat(pk_str(k), v) for (k, v) in kvs]...)..., UInt8('u')]
    pk_odict() = UInt8[pk_global("collections", "OrderedDict")..., 0x29, UInt8('R')]
    pk_call(callable::Vector{UInt8}, args::Vector{UInt8}) = UInt8[callable..., args..., UInt8('R')]
    pk_build(obj::Vector{UInt8}, state::Vector{UInt8}) = UInt8[obj..., state..., UInt8('b')]
    pk_storage(cls::String, key::String, n::Integer) =
        UInt8[pk_tuple(pk_str("storage"), pk_global("torch", cls), pk_str(key), pk_str("cpu"), pk_int(n))..., UInt8('Q')]
    pk_tensor(storage::Vector{UInt8}, offset::Integer, size::Vector{Int}, stride::Vector{Int}) =
        pk_call(
        pk_global("torch._utils", "_rebuild_tensor_v2"),
        pk_tuple(
            storage, pk_int(offset), pk_tuple([pk_int(s) for s in size]...), pk_tuple([pk_int(s) for s in stride]...),
            pk_bool(false), pk_odict()
        )
    )
    pk_param(tensor::Vector{UInt8}, requires_grad::Bool) =
        pk_call(pk_global("torch._utils", "_rebuild_parameter"), pk_tuple(tensor, pk_bool(requires_grad), pk_odict()))
    # A module body with the three dicts torch keeps; `nonpersistent` names buffers to omit.
    # `attrs` are further entries of the module's `__dict__`.
    function pk_module_state(;
            params = Pair{String, Vector{UInt8}}[], buffers = Pair{String, Vector{UInt8}}[],
            modules = Pair{String, Vector{UInt8}}[], nonpersistent::Union{Nothing, Vector{String}} = nothing,
            attrs = Pair{String, Vector{UInt8}}[]
        )
        kvs = Pair{String, Vector{UInt8}}["_parameters" => pk_dict(params...), "_buffers" => pk_dict(buffers...), "_modules" => pk_dict(modules...)]
        append!(kvs, attrs)
        isnothing(nonpersistent) || push!(
            kvs, "_non_persistent_buffers_set" =>
                pk_call(pk_global("__builtin__", "set"), pk_tuple(pk_list([pk_str(s) for s in nonpersistent]...)))
        )
        return pk_dict(kvs...)
    end
    pk_module(state::Vector{UInt8}; cls::Tuple{String, String} = ("torch.nn.modules.container", "ModuleList")) =
        pk_build(pk_call(pk_global(cls...), UInt8[0x29]), state)
    pk_finish(root::Vector{UInt8}) = UInt8[0x80, 0x02, root..., UInt8('.')]
    f64bytes(xs) = cat([le(reinterpret(UInt64, Float64(x))) for x in xs]...)
    i64bytes(xs) = cat([le(reinterpret(UInt64, Int64(x))) for x in xs]...)
    pk_float(x::Real) = UInt8[UInt8('G'), reverse(le(reinterpret(UInt64, Float64(x))))...]

    # An e3nn `Irreps` object as pickled: Irreps((_MulIr((mul, Irrep((l, p)))), ...)).
    pk_irrep(l::Integer, p::Integer) = pk_call(pk_global("e3nn.o3._irreps", "Irrep"), pk_tuple(pk_tuple(pk_int(l), pk_int(p))))
    pk_mulir(mul::Integer, l::Integer, p::Integer) =
        pk_call(pk_global("e3nn.o3._irreps", "_MulIr"), pk_tuple(pk_tuple(pk_int(mul), pk_irrep(l, p))))
    pk_irreps(terms::Tuple{Int, Int, Int}...) =
        pk_call(pk_global("e3nn.o3._irreps", "Irreps"), pk_tuple(pk_tuple([pk_mulir(t...) for t in terms]...)))

    # Archive whose data.pkl unpickles to an empty dict.
    dict_root_zip() = zip_with_pickle(pk_finish(pk_dict()), Pair{String, Vector{UInt8}}[])

    # A one-interaction ScaleShiftMACE skeleton: storage "0" holds Float64 values
    # [r_max, avg_num_neighbors, weights...], storage "1" Int64 [atomic numbers..., num_interactions].
    # `avg_buffer` stores avg_num_neighbors as a 0-dim buffer instead of a float attribute.
    function tiny_mace_zip(; avg_buffer::Bool = true, root_class::Tuple{String, String} = ("mace.modules.models", "ScaleShiftMACE"), extra_modules::Vector{Pair{String, Vector{UInt8}}} = Pair{String, Vector{UInt8}}[])
        f64(off, size) = pk_tensor(pk_storage("DoubleStorage", "0", 6), off, size, size == [2] ? [1] : Int[])
        i64(off, size) = pk_tensor(pk_storage("LongStorage", "1", 3), off, size, size == [2] ? [1] : Int[])
        avg = avg_buffer ? ["avg_num_neighbors" => f64(1, Int[])] : Pair{String, Vector{UInt8}}[]
        attrs = Pair{String, Vector{UInt8}}[
            "node_feats_irreps" => pk_irreps((128, 0, 1)),
            "edge_attrs_irreps" => pk_irreps((1, 0, 1), (1, 1, -1)),
            "edge_feats_irreps" => pk_irreps((8, 0, 1)),
            "target_irreps" => pk_irreps((128, 0, 1), (128, 1, -1)),
            "hidden_irreps" => pk_str("128x0e + 128x1o"),
            "radial_MLP" => pk_list(pk_int(64), pk_int(64), pk_int(64)),
        ]
        avg_buffer || push!(attrs, "avg_num_neighbors" => pk_float(61.964672446250916))
        interaction = pk_module(
            pk_module_state(; buffers = avg, attrs);
            cls = ("mace.modules.blocks", "RealAgnosticDensityInteractionBlock")
        )
        weights = pk_module(
            pk_module_state(params = ["0" => pk_param(f64(2, Int[]), true), "1" => pk_param(f64(3, Int[]), true)]);
            cls = ("torch.nn.modules.container", "ParameterList")
        )
        contraction = pk_module(
            pk_module_state(params = ["weights_max" => pk_param(f64(4, [2]), true)], modules = ["weights" => weights]);
            cls = ("mace.modules.symmetric_contraction", "Contraction")
        )
        sc = pk_module(
            pk_module_state(modules = ["contractions" => pk_module(pk_module_state(modules = ["0" => contraction]))]);
            cls = ("mace.modules.symmetric_contraction", "SymmetricContraction")
        )
        product = pk_module(
            pk_module_state(modules = ["symmetric_contractions" => sc], attrs = ["use_sc" => pk_bool(false)]);
            cls = ("mace.modules.blocks", "EquivariantProductBasisBlock")
        )
        readout = pk_module(pk_module_state(); cls = ("mace.modules.blocks", "LinearReadoutBlock"))
        list(m) = pk_module(pk_module_state(modules = ["0" => m]))
        root = pk_module(
            pk_module_state(
                buffers = ["atomic_numbers" => i64(0, [2]), "r_max" => f64(0, Int[]), "num_interactions" => i64(2, Int[])],
                modules = ["interactions" => list(interaction), "products" => list(product), "readouts" => list(readout), extra_modules...],
                attrs = ["heads" => pk_list(pk_str("default"))]
            ); cls = root_class
        )
        return zip_with_pickle(
            pk_finish(root), [
                "0" => f64bytes([5.0, 61.964672446250916, 0.5, 0.25, 1.0, 2.0]),
                "1" => i64bytes([1, 8, 1]),
            ]
        )
    end
end

@testmodule Checkpoints begin
    using PureMLIP, SHA, Downloads, JSON

    const SOURCES = Dict(
        :mp0b3 => (
            "mace-mp-0b3-medium.model",
            "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
            "2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012",
        ),
        :mpa0 => (
            "mace-mpa-0-medium.model",
            "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
            "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638",
        ),
    )

    # Path of the checkpoint `name`, downloaded on first use and verified by SHA-256.
    # A file with the wrong digest is deleted.
    function fetch(name::Symbol)
        file, url, sha = SOURCES[name]
        dir = get(ENV, "PUREMLIP_CHECKPOINT_DIR", joinpath(first(DEPOT_PATH), "scratchspaces", "PureMLIP-checkpoints"))
        mkpath(dir)
        path = joinpath(dir, file)
        if !isfile(path)
            Downloads.download(url, path * ".part")
            mv(path * ".part", path; force = true)
        end
        digest = bytes2hex(open(sha256, path))
        if digest != sha
            rm(path)
            error("SHA-256 of $path is $digest, expected $sha; the file was deleted")
        end
        return path
    end

    inspection() = JSON.parsefile(pkgdir(PureMLIP, "test", "fixtures", "llmlip_mace_mp0b_inspection.json"))

    const COMPILED_W3J = "compiled_tensor_product_wigner_buffer"

    # Tensor name => its reference entry (with shape, dtype, nonzero), or `nothing` when
    # the reference lists only the name.
    function reference_tensors(ref)
        cov = ref["state_dict_coverage"]
        out = Dict{String, Any}()
        for n in cov["covered_state_tensor_names"]
            out[n] = nothing
        end
        for e in cov["missing_state_tensors"]
            out[e["name"]] = e
        end
        for p in ref["product_blocks"], c in p["symmetric_contractions"]["contractions"]
            pre = string("products.", p["index"], ".symmetric_contractions.", c["name"], ".")
            for (k, v) in c["buffers"]
                out[pre * k] = v
            end
            for (k, v) in c["parameters"]
                out[pre * k] = v
            end
            for w in c["weights"]
                out[string(pre, "weights.", w["index"])] = w
            end
        end
        return out
    end

    compiled_w3j(ref) = Set{String}(e["name"] for e in ref["state_dict_coverage"]["missing_state_tensors"] if e["classification"] == COMPILED_W3J)

    # Reference tensor names that the checkpoint lacks, sorted.
    absent(ck, ref) = sort([n for n in keys(reference_tensors(ref)) if !(n in ck.state.names)])

    # One line per interaction-block irreps or radial_MLP field that differs from the reference.
    function interaction_mismatches(ck, ref)
        out = String[]
        specs = ck.hyperparameters.interactions
        length(specs) == length(ref["interaction_blocks"]) || push!(out, "number of interaction blocks")
        for (spec, r) in zip(specs, ref["interaction_blocks"])
            for key in ("node_feats_irreps", "edge_attrs_irreps", "edge_feats_irreps", "target_irreps", "hidden_irreps")
                want = r[key] isa AbstractString ? PureMLIP.Checkpoint.parse_irreps(r[key]) :
                    [PureMLIP.IrrepMul(m, lp[1], lp[2]) for (m, lp) in r[key]]
                getfield(spec, Symbol(key)) == want || push!(out, string("interactions.", r["index"], ".", key))
            end
            spec.class == r["class"] || push!(out, string("interactions.", r["index"], ".class"))
            spec.radial_MLP == r["radial_MLP"] || push!(out, string("interactions.", r["index"], ".radial_MLP"))
        end
        return out
    end

    # One line per disagreement between the checkpoint state and the reference, ignoring
    # the reference's compiled tensor-product buffers.
    function mismatches(ck, ref)
        refs = reference_tensors(ref)
        skip = compiled_w3j(ref)
        out = String[]
        for (n, spec) in sort(collect(refs); by = first)
            n in skip && continue
            if haskey(ck.state.f64, n)
                x, dtype = ck.state.f64[n], "torch.float64"
            elseif haskey(ck.state.i64, n)
                x, dtype = ck.state.i64[n], "torch.int64"
            else
                push!(out, "$n: missing")
                continue
            end
            isnothing(spec) && continue
            x.shape == spec["shape"] || push!(out, "$n: shape $(x.shape), reference $(spec["shape"])")
            dtype == spec["dtype"] || push!(out, "$n: dtype $dtype, reference $(spec["dtype"])")
            count(!iszero, x.data) == spec["nonzero"] || push!(out, "$n: $(count(!iszero, x.data)) nonzeros, reference $(spec["nonzero"])")
        end
        for n in ck.state.names
            haskey(refs, n) || push!(out, "$n: not in the reference")
        end
        return out
    end
end

@testmodule E3nnFixtures begin
    using PureMLIP, JSON

    const DIR = pkgdir(PureMLIP, "test", "fixtures", "e3nn-core-rs")

    load(name::AbstractString) = JSON.parsefile(joinpath(DIR, name * ".json"))

    # A JSON array of equal-length rows as a Matrix{Float64}, one row per entry.
    rows(a) = Float64[a[i][j] for i in eachindex(a), j in eachindex(first(a))]

    # Spherical harmonics fixture for `lmax = L`: input vectors, values, cotangents and
    # the vector-Jacobian product, each with one row per input vector.
    function sh(L::Integer)
        d = load("spherical_harmonics_lmax$(L)_component_normalized_f64")
        return (; vectors = rows(d["vectors"]), y = rows(d["y"]), grad_y = rows(d["grad_y"]), grad_vectors = rows(d["grad_vectors"]))
    end

    # The degree of each irrep copy of an irreps string such as "2x0e+1x1o", in order;
    # the fixtures number segments over these copies from 0.
    function segment_degrees(s::AbstractString)
        out = Int[]
        for term in split(s, '+')
            mul, ir = split(strip(term), 'x')
            append!(out, fill(parse(Int, ir[1:(end - 1)]), parse(Int, mul)))
        end
        return out
    end

    # One `(l1, l2, l3, entries)` per tensor-product instruction of fixture `name`, with
    # `entries` its sparse Wigner-3j coefficients `(i, j, k, value)`, `m` indices 0-based.
    function cg_instructions(name::AbstractString)
        d = load(name)
        l1s, l2s, l3s = segment_degrees(d["irreps_in1"]), segment_degrees(d["irreps_in2"]), segment_degrees(d["irreps_out"])
        out = Tuple{Int, Int, Int, Vector{Tuple{Int, Int, Int, Float64}}}[]
        for n in eachindex(d["instruction_cg_offset"])
            off, len = d["instruction_cg_offset"][n], d["instruction_cg_len"][n]
            r = (off + 1):(off + len)
            entries = [(Int(d["cg_i_m"][q]), Int(d["cg_j_m"][q]), Int(d["cg_k_m"][q]), Float64(d["cg_value"][q])) for q in r]
            push!(
                out, (
                    l1s[d["instruction_in1_segment"][n] + 1], l2s[d["instruction_in2_segment"][n] + 1],
                    l3s[d["instruction_out_segment"][n] + 1], entries,
                )
            )
        end
        return out
    end
end

@testmodule LammpsReference begin
    using PureMLIP, JSON

    const DIR = pkgdir(PureMLIP, "test", "fixtures", "lammps-mace")

    load(name::Symbol) = JSON.parsefile(joinpath(DIR, string(name) * ".json"))

    # A JSON array of equal-length rows as a Matrix{Float64}, one row per entry.
    rows(a) = Float64[a[i][j] for i in eachindex(a), j in eachindex(first(a))]

    # One structure of the reference: positions and forces 3×n, cell columns = lattice
    # vectors, virial[a,b] = p_ab · volume / nktv2p in eV (symmetric 3×3).
    # virial = p·V/nktv2p = −∂E/∂ε with the kinetic part excluded and the LAMMPS sign (positive
    # pressure is compressive); the stress ∂E/(V·∂ε) is therefore −virial/V.
    function structure(ref, name::AbstractString)
        s = ref["structures"][findfirst(x -> x["name"] == name, ref["structures"])]
        pv = Float64.(s["pressure_virial_bar"])
        pxx, pyy, pzz, pxy, pxz, pyz = pv
        p = [pxx pxy pxz; pxy pyy pyz; pxz pyz pzz]
        return (;
            name = s["name"],
            positions = permutedims(rows(s["positions"])),
            cell = permutedims(rows(s["cell"])),
            atomic_numbers = Int.(s["atomic_numbers"]),
            energy = Float64(s["energy"]),
            node_energies = Float64.(s["node_energies"]),
            forces = permutedims(rows(s["forces"])),
            virial = p .* (Float64(s["volume"]) / Float64(ref["nktv2p"])),
        )
    end

    structures(ref) = [structure(ref, s["name"]) for s in ref["structures"]]
end

@testmodule NeighborOracle begin
    using LinearAlgebra

    # Every directed edge (sender, receiver, n) with 0 < ‖pos[:, receiver] - pos[:, sender] + cell * n‖ < r_cut,
    # by exhaustive search over the images |n_k| ≤ ceil(r_cut / h_k), h_k the cell height along k
    # (volume over the area of the other two vectors). Valid for positions inside the cell.
    function brute_force(positions, cell, r_cut)
        a1, a2, a3 = cell[:, 1], cell[:, 2], cell[:, 3]
        vol = abs(dot(a1, cross(a2, a3)))
        h = (vol / norm(cross(a2, a3)), vol / norm(cross(a3, a1)), vol / norm(cross(a1, a2)))
        m = ceil.(Int, r_cut ./ h)
        out = Set{Tuple{Int, Int, NTuple{3, Int}}}()
        for i in axes(positions, 2), j in axes(positions, 2)
            d = positions[:, j] - positions[:, i]
            for n1 in (-m[1]):m[1], n2 in (-m[2]):m[2], n3 in (-m[3]):m[3]
                r = d + cell * [n1, n2, n3]
                0 < norm(r) < r_cut && push!(out, (i, j, (n1, n2, n3)))
            end
        end
        return out
    end
end
