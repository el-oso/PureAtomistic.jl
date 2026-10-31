# Writes the LAMMPS-MACE reference fixtures in test/fixtures/lammps-mace/.
#
#   PUREMLIP_LAMMPS            the `lmp` binary (built with ML-MACE)
#   PUREMLIP_LAMMPS_MODEL_DIR  directory with mpa0_f64-lammps.pt, mpa0_f32-lammps.pt,
#                              mp0b3_f64-lammps.pt and the exporter's versions.txt
#
# The environment of this process is inherited by `lmp`.
using Printf, SHA, LinearAlgebra

include(joinpath(@__DIR__, "structures.jl"))

const SYMBOLS = Dict(1 => "H", 6 => "C", 7 => "N", 8 => "O", 13 => "Al", 24 => "Cr", 26 => "Fe", 27 => "Co", 28 => "Ni", 29 => "Cu", 40 => "Zr", 74 => "W")

const MODELS = [
    (
        name = "mpa0_f64", file = "mpa0_f64-lammps.pt", model = "mace-mpa-0-medium",
        url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
        sha = "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638", dtype = "float64", default = "float64",
    ),
    (
        name = "mpa0_f32", file = "mpa0_f32-lammps.pt", model = "mace-mpa-0-medium",
        url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
        sha = "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638", dtype = "float32", default = "float32",
    ),
    (
        name = "mp0b3_f64", file = "mp0b3_f64-lammps.pt", model = "mace-mp-0b3-medium",
        url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
        sha = "2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012", dtype = "float64", default = "float64",
    ),
]

const LAMMPS_REPOSITORY = "https://github.com/ACEsuit/lammps (branch mace)"
const LAMMPS_COMMIT = "4d222cb3ee2a6b14083c778968497bf9e0efc4b4"
const MACE_COMMIT = "4d2da09413ac1407f37cdbb6b81fa28e4c15655e"
const LIBTORCH_VERSION = "2.8.0"
const NKTV2P = 1.6021765e6

# Float with 17 significant digits, always recognizable as a JSON number with a fraction or exponent.
function fmt(x::Real)
    s = @sprintf("%.17g", x)
    return any(c -> c in ".eEn", s) ? s : s * ".0"
end

function write_json(io, x, indent = 0)
    pad = "  "^indent
    return if x isa AbstractString
        print(io, '"', x, '"')
    elseif x isa Integer
        print(io, x)
    elseif x isa AbstractFloat
        print(io, fmt(x))
    elseif x isa AbstractVector && all(e -> e isa Number, x)
        print(io, '[', join((e isa Integer ? string(e) : fmt(e) for e in x), ", "), ']')
    elseif x isa AbstractVector
        println(io, '[')
        for (i, e) in enumerate(x)
            print(io, pad, "  ")
            write_json(io, e, indent + 1)
            println(io, i < length(x) ? "," : "")
        end
        print(io, pad, ']')
    elseif x isa AbstractMatrix
        write_json(io, [x[i, :] for i in axes(x, 1)], indent)
    else
        println(io, '{')
        ks = collect(pairs(x))
        for (i, (k, v)) in enumerate(ks)
            print(io, pad, "  \"", k, "\": ")
            write_json(io, v, indent + 1)
            println(io, i < length(ks) ? "," : "")
        end
        print(io, pad, '}')
    end
end

# LAMMPS restricted-triclinic data file for one structure; atom types follow the sorted atomic numbers.
function write_data(path, s, types)
    c = s.cell
    (c[1, 2] == 0 && c[1, 3] == 0 && c[2, 3] == 0) || error("cell is not in LAMMPS restricted triclinic form")
    return open(path, "w") do io
        println(io, "reference structure ", s.name, "\n")
        println(io, size(s.positions, 1), " atoms\n", length(types), " atom types\n")
        println(io, "0.0 ", fmt(c[1, 1]), " xlo xhi")
        println(io, "0.0 ", fmt(c[2, 2]), " ylo yhi")
        println(io, "0.0 ", fmt(c[3, 3]), " zlo zhi")
        println(io, fmt(c[2, 1]), " ", fmt(c[3, 1]), " ", fmt(c[3, 2]), " xy xz yz\n")
        println(io, "Atoms # atomic\n")
        for (n, z) in enumerate(s.atomic_numbers)
            println(io, n, " ", findfirst(==(z), types), " ", join(fmt.(s.positions[n, :]), " "))
        end
    end
end

function versions(dir)
    v = Dict{String, String}()
    for line in eachline(joinpath(dir, "versions.txt"))
        m = match(r"^([A-Za-z0-9_.-]+)==(.+)$", line)
        isnothing(m) || (v[lowercase(m[1])] = m[2])
        occursin("mace-torch @", line) && (v["mace-torch"] = match(r"@([0-9a-f]{40})\s*$", line)[1])
    end
    return v
end

function run_structure(lmp, in_file, model, s, workdir)
    types = sort(unique(s.atomic_numbers))
    data = joinpath(workdir, s.name * ".data")
    dumpfile = joinpath(workdir, s.name * ".dump")
    write_data(data, s, types)
    symbols = join((SYMBOLS[z] for z in types), " ")
    cmd = setenv(`$lmp -in $in_file -log none -var data $data -var model $model -var symbols $symbols -var dumpfile $dumpfile`, merge(ENV, Dict("OMP_NUM_THREADS" => "1")))
    out = IOBuffer()
    run(pipeline(cmd; stdout = out, stderr = out))
    text = String(take!(out))
    return text, dumpfile
end

function main()
    lmp = ENV["PUREMLIP_LAMMPS"]
    dir = ENV["PUREMLIP_LAMMPS_MODEL_DIR"]
    in_file = joinpath(@__DIR__, "in.mace")
    outdir = joinpath(@__DIR__, "..", "fixtures", "lammps-mace")
    vers = versions(dir)
    helptext = read(setenv(`$lmp -h`, ENV), String)
    lammps_version = strip(match(r"Large-scale Atomic/Molecular Massively Parallel Simulator - (.+)", helptext)[1])
    occursin(r"(?m)^\s*(\S+\s+)*mace\b", helptext) || error("pair style mace is not in the LAMMPS binary")
    machine = strip(read(`sysctl -n machdep.cpu.brand_string`, String))
    structs = reference_structures()
    return mktempdir() do work
        for m in MODELS
            model = joinpath(dir, m.file)
            entries = []
            for s in structs
                text, dumpfile = run_structure(lmp, in_file, model, s, work)
                res = Dict(split(l)[2] => parse.(Float64, split(l)[3:end]) for l in split(text, '\n') if startswith(l, "RESULT "))
                n = length(s.atomic_numbers)
                forces = zeros(n, 3)
                pe = zeros(n)
                rows = [split(l) for l in eachline(dumpfile)]
                ih = findfirst(r -> length(r) >= 2 && r[1:2] == ["ITEM:", "ATOMS"], rows)
                body = rows[(ih + 1):end]
                length(body) == n || error("dump of $(s.name) has $(length(body)) atoms, expected $n")
                for r in body
                    i = parse(Int, r[1])
                    forces[i, :] = parse.(Float64, r[2:4])
                    pe[i] = parse(Float64, r[5])
                end
                push!(
                    entries, (;
                        name = s.name, cell = s.cell, positions = s.positions, atomic_numbers = s.atomic_numbers,
                        energy = only(res["energy"]), node_energies = pe, forces,
                        pressure_virial_bar = res["pressure"], volume = only(res["volume"]),
                    )
                )
            end
            doc = (;
                schema = "puremlip-lammps-mace-v1",
                model = (; name = m.model, source_url = m.url, source_sha256 = m.sha, export_sha256 = bytes2hex(open(sha256, model)), dtype = m.dtype, export_default_dtype = m.default),
                software = (;
                    lammps_version, lammps_repository = LAMMPS_REPOSITORY, lammps_commit = LAMMPS_COMMIT, libtorch = LIBTORCH_VERSION,
                    torch = vers["torch"], mace = "0.3.16", mace_commit = vers["mace-torch"], e3nn = vers["e3nn"],
                ),
                machine,
                nktv2p = NKTV2P,
                structures = entries,
            )
            open(joinpath(outdir, m.name * ".json"), "w") do io
                write_json(io, doc)
                println(io)
            end
            println("wrote ", m.name, ".json")
        end
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
