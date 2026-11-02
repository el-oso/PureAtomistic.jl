# Exact reference bits of a one-chunk evaluation, and the platform they depend on.
#
#   PUREMLIP_CHECKPOINT_DIR=<dir> julia PureMLIP/test/reference/one_chunk_bits.jl [git ref]
#
# writes `one_chunk_bits.json` next to this file from the PureMLIP of the git ref (default
# ef63ada, the evaluation before receiver chunks existed). The ref is checked out as a detached
# `git worktree` in a temporary directory outside the repository, removed at exit, and evaluated
# in a child process whose environment points PureMLIP at that tree. The child inherits the
# bounds-checking mode of this process (`Base.julia_cmd`): Julia's default unless given, which is
# how `Pkg.test` runs the suite on Julia 1.13 (on 1.12 it passes `--check-bounds=yes`). The test
# item "forces: one chunk reproduces the stored reference bits" includes this file for `platform`
# and `reference_bits`.
#
# Stored: energy, node energies, forces, virial and stress of `mace_energy_forces!` for
# cu_bench(2) (32 atoms) under MACE-MPA-0 with a default `MACEWorkspace`, in Float64 and Float32,
# each value as the hexadecimal bit pattern of its unsigned reinterpretation (arrays in column
# order). Rounding depends on the BLAS library, the CPU, the Julia version, bounds checking,
# coverage instrumentation (both change the compiled arithmetic: measured, each changes forces,
# virial and stress in the last bits) and the BLAS thread count, so the record names them; the
# BLAS thread count is set to 1.

using LinearAlgebra

# The platform a bit pattern depends on: BLAS library file names and the version string each
# library reports (OpenBLAS's `openblas_get_config`, or "unknown"), the CPU model, the Julia
# version, bounds checking, coverage instrumentation and the BLAS thread count.
function platform()
    libs = [l.libname for l in BLAS.get_config().loaded_libs]
    function version(lib)
        h = Libc.Libdl.dlopen(lib; throw_error = false)
        isnothing(h) && return "unknown"
        for sym in (:openblas_get_config64_, :openblas_get_config)
            p = Libc.Libdl.dlsym(h, sym; throw_error = false)
            isnothing(p) || return unsafe_string(ccall(p, Ptr{UInt8}, ()))
        end
        return "unknown"
    end
    return Dict(
        "blas_libraries" => basename.(libs),
        "blas_versions" => version.(libs),
        "cpu" => Sys.cpu_info()[1].model,
        "julia_version" => string(VERSION),
        "check_bounds" => ("default", "yes", "no")[Base.JLOptions().check_bounds + 1],
        "code_coverage" => iszero(Base.JLOptions().code_coverage) ? "off" : "on",
        "blas_threads" => string(BLAS.get_num_threads()),
    )
end

hexbits(x::AbstractFloat) = string(reinterpret(Unsigned, x); base = 16, pad = 2 * sizeof(x))

# `quantity => hex strings` per precision name, from `PM` (the PureMLIP module) on the structure
# `cu` (a `cu_bench` named tuple) under the checkpoint `ck`.
function reference_bits(PM, cu, ck)
    out = Dict{String, Dict{String, Vector{String}}}()
    for T in (Float64, Float32)
        m = PM.MACEModel{T}(ck)
        P, C, Z = Matrix{T}(cu.positions'), Matrix{T}(cu.cell'), cu.atomic_numbers
        n = length(Z)
        nl = PM.neighbor_list(P, C, m.r_max)
        r = PM.mace_energy_forces!(PM.MACEResult{T}(n), PM.MACEWorkspace{T}(m, n, nl.n_edges), m, P, C, Z, nl)
        out[string(T)] = Dict(
            "energy" => [hexbits(r.energy)], "node_energies" => hexbits.(r.node_energies),
            "forces" => vec(hexbits.(r.forces)), "virial" => vec(hexbits.(r.virial)), "stress" => vec(hexbits.(r.stress)),
        )
    end
    return out
end

# Child: prints `BITS<TAB>precision<TAB>quantity<TAB>hex,...` and `PLATFORM<TAB>key<TAB>value|...`.
function child_main(structures, checkpoint)
    PM = Base.require(Main, :PureMLIP)
    include(structures)
    return Base.invokelatest(child_print, PM, checkpoint)
end

# Runs in the world where PureMLIP and `cu_bench` are defined.
function child_print(PM, checkpoint)
    BLAS.set_num_threads(1)
    bits = reference_bits(PM, Main.cu_bench(2), PM.load_mace_checkpoint(checkpoint))
    for (T, qs) in bits, (q, v) in qs
        println("BITS\t", T, "\t", q, "\t", join(v, ","))
    end
    for (k, v) in platform()
        println("PLATFORM\t", k, "\t", v isa AbstractVector ? join(v, "|") : v)
    end
    return
end

json_string(s) = "\"" * escape_string(s) * "\""
json_list(v) = "[" * join(json_string.(v), ", ") * "]"

function main(ref)
    git(dir, args...) = readchomp(`git -C $dir $args`)
    repo = git(@__DIR__, "rev-parse", "--show-toplevel")
    sha = git(repo, "rev-parse", "--verify", ref * "^{commit}")
    checkpoint = joinpath(ENV["PUREMLIP_CHECKPOINT_DIR"], "mace-mpa-0-medium.model")
    isfile(checkpoint) || error("one_chunk_bits: no checkpoint at $checkpoint")
    tmp = mktempdir()
    tree = joinpath(tmp, "tree")
    run(`git -C $repo worktree add --detach $tree $sha`)
    out = try
        env = joinpath(tmp, "env")
        mkpath(env)
        write(
            joinpath(env, "Project.toml"),
            """
            [deps]
            LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
            PureMLIP = "e31974d8-f9c1-4d27-8220-1506e2a12d0a"

            [sources]
            PureMLIP = {path = "$(joinpath(tree, "PureMLIP"))"}
            """
        )
        julia = Base.julia_cmd()
        run(`$julia --project=$env -e "using Pkg; Pkg.instantiate()"`)
        structures = joinpath(tree, "PureMLIP", "test", "reference", "structures.jl")
        read(`$julia -t 1 --project=$env $(@__FILE__) --child $structures $checkpoint`, String)
    finally
        run(`git -C $repo worktree remove --force $tree`)
    end
    bits = Dict{String, Dict{String, Vector{String}}}()
    plat = Dict{String, Vector{String}}()
    for line in split(out, '\n')
        f = split(line, '\t')
        if f[1] == "BITS"
            get!(bits, f[2], Dict{String, Vector{String}}())[f[3]] = split(f[4], ',')
        elseif f[1] == "PLATFORM"
            plat[f[2]] = split(f[3], "|")
        end
    end
    one(k) = json_string(only(plat[k]))
    open(joinpath(@__DIR__, "one_chunk_bits.json"), "w") do io
        println(io, "{")
        println(io, "  \"pureMLIP_commit\": ", json_string(sha), ",")
        println(io, "  \"structure\": \"cu_bench(2)\",")
        println(io, "  \"checkpoint\": \"mace-mpa-0-medium.model\",")
        println(io, "  \"platform\": {")
        println(io, "    \"blas_libraries\": ", json_list(plat["blas_libraries"]), ",")
        println(io, "    \"blas_versions\": ", json_list(plat["blas_versions"]), ",")
        println(io, "    \"cpu\": ", one("cpu"), ",")
        println(io, "    \"julia_version\": ", one("julia_version"), ",")
        println(io, "    \"check_bounds\": ", one("check_bounds"), ",")
        println(io, "    \"code_coverage\": ", one("code_coverage"), ",")
        println(io, "    \"blas_threads\": ", one("blas_threads"))
        println(io, "  },")
        println(io, "  \"bits\": {")
        Ts = sort(collect(keys(bits)))
        for (i, T) in enumerate(Ts)
            println(io, "    ", json_string(T), ": {")
            qs = sort(collect(keys(bits[T])))
            for (j, q) in enumerate(qs)
                println(io, "      ", json_string(q), ": ", json_list(bits[T][q]), j < length(qs) ? "," : "")
            end
            println(io, "    }", i < length(Ts) ? "," : "")
        end
        println(io, "  }")
        println(io, "}")
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "--child"
        child_main(ARGS[2], ARGS[3])
    else
        main(isempty(ARGS) ? "ef63ada" : ARGS[1])
    end
end
