# Share of the samples of `mace_energy_forces!` spent in each stage, from the sampling profiler.
# One thread. Run it once per BLAS backend, case and precision; each run adds its sample counts
# to the JSON file, so a run that the profiler aborts (it can fail inside libunwind) is repeated
# and the counts of all runs add up.
#
# Environment: PureBLAS is not a dependency of PureMLIP or of the bench project, so use a separate
# project that has PureMLIP, JSON, Profile and (for the PureBLAS arm) PureBLAS:
#
#   julia -t 1 --project=<dir> PureMLIP/bench/stage_profile.jl
#
#   PUREMLIP_CHECKPOINT_DIR  directory with mace-mpa-0-medium.model
#   STAGE_BLAS               "openblas" (default) or "pureblas"
#   STAGE_CASES              cases as "N:precision", precision 64 or 32 (default "5:64")
#   STAGE_SECONDS            profiled seconds per case and run (default 20; at least one evaluation)
#   PUREMLIP_BENCH_OUT       JSON output path (default results/stage_profile_mac.json)
#
# A sample is attributed to the outermost frame (inlined frames included) that belongs to a stage
# function. The stage functions are the kernels of the stages; the evaluation code between them is
# classified by line of `evaluate.jl`, found from marker comments in the source. A sample whose
# stack holds no stage function falls in `evaluate_other` (evaluation code outside the named
# regions) or, when it has no `mace_energy_forces!` frame, `outside`. Shares are fractions of all
# samples. OpenBLAS's assembly kernels carry no unwind information, so a sample inside one has no
# caller: it is counted as `blas_stack_lost` (the matrix products of the radial MLP and of the
# linear layers together). The PureBLAS kernels are Julia code and unwind.
#
# The radial MLP is split into the part below a matrix product (`mul!`, BLAS) and the rest.
using PureMLIP, Profile, JSON, LinearAlgebra, Printf

BLAS.set_num_threads(1)
Threads.nthreads() == 1 || error("run with julia -t 1")
include(joinpath(@__DIR__, "..", "test", "reference", "structures.jl"))

const CK = joinpath(ENV["PUREMLIP_CHECKPOINT_DIR"], "mace-mpa-0-medium.model")
const BLAS_ARM = get(ENV, "STAGE_BLAS", "openblas")
BLAS_ARM in ("openblas", "pureblas") || error("STAGE_BLAS must be openblas or pureblas, got ", BLAS_ARM)
const SPECS = [
    (parse(Int, a), b == "64" ? Float64 : Float32) for (a, b) in
        split.(split(get(ENV, "STAGE_CASES", "5:64"), ','), ':')
]
const SECONDS = parse(Float64, get(ENV, "STAGE_SECONDS", "20"))
const OUT = get(ENV, "PUREMLIP_BENCH_OUT", joinpath(@__DIR__, "results", "stage_profile_mac.json"))
const DELAY = 0.001

const STAGES = [
    "edges", "radial_features",
    "radial_mlp_gemm_fwd", "radial_mlp_other_fwd", "radial_mlp_gemm_bwd", "radial_mlp_other_bwd",
    "spherical_harmonics_fwd", "spherical_harmonics_bwd",
    "tensor_product_fwd", "tensor_product_bwd",
    "symmetric_contraction_fwd", "symmetric_contraction_bwd",
    "linear_fwd", "linear_bwd", "readouts_fwd", "readouts_bwd",
    "zbl", "force_virial_scatter", "blas_stack_lost",
    "messages_other_fwd", "messages_other_bwd", "evaluate_other", "outside",
]

# Stage of a function name; the linear family and the radial MLP are refined by `classify`.
const FUNCTION_STAGE = Dict{Symbol, String}(
    :_edges! => "edges",
    :_chunk_radial! => "radial_features", :_radial_column! => "radial_features", :radial_embedding! => "radial_features",
    :radial_mlp! => "radial_mlp_fwd", :radial_mlp_pullback! => "radial_mlp_bwd",
    :e3nn_spherical_harmonics! => "spherical_harmonics_fwd",
    :e3nn_spherical_harmonics_and_gradient! => "spherical_harmonics_bwd",
    :tp_edge_add! => "tensor_product_fwd", :tp_edge_pullback_add! => "tensor_product_bwd",
    :symmetric_contraction! => "symmetric_contraction_fwd",
    :symmetric_contraction_pullback! => "symmetric_contraction_bwd",
    :linear! => "linear_fwd", :species_linear! => "linear_fwd", :_normalized_linear! => "linear_fwd",
    :linear_pullback! => "linear_bwd", :species_linear_pullback! => "linear_bwd",
    :_normalization_pullback! => "linear_bwd", :_normalized_linear_pullback! => "linear_bwd",
    :zbl_edge_energy => "zbl",
)

# Line ranges of `evaluate.jl`, from marker comments (each marker must occur in the source).
function line_regions()
    path = joinpath(dirname(pathof(PureMLIP)), "model", "evaluate.jl")
    lines = readlines(path)
    function first_line(marker, from = 1)
        i = findnext(l -> occursin(marker, l), lines, from)
        i === nothing && error("marker not found: ", marker)
        return i
    end
    r_fwd = first_line("# Readouts.")
    z_fwd = first_line("# ZBL pair repulsion, half of each directed edge credited to its receiver.")
    s_fwd = first_line("E = zero(T)", z_fwd)
    z_bwd = first_line("# ∂E/∂r per edge starts from the ZBL pair term")
    r_bwd = first_line("# Readouts: ∂E/∂e1")
    p_bwd = first_line("# Product 1 and the skip connection of interaction 1")
    sc_bwd = first_line("# Edge vectors: g_e = ∂E/∂r_e.")
    end_bwd = first_line("_det3(C) =", sc_bwd)
    return (; path = basename(path), r_fwd, z_fwd, s_fwd, z_bwd, r_bwd, p_bwd, sc_bwd, end_bwd)
end

in_range(l, a, b) = a <= l < b

# Stage of the evaluation code at `line` of `mace_energy!` / `mace_energy_forces!`.
function line_stage(R, line)
    in_range(line, R.r_fwd, R.z_fwd) && return "readouts_fwd"
    in_range(line, R.z_fwd, R.s_fwd) && return "zbl"
    in_range(line, R.z_bwd, R.r_bwd) && return "zbl"
    in_range(line, R.r_bwd, R.p_bwd) && return "readouts_bwd"
    in_range(line, R.sc_bwd, R.end_bwd) && return "force_virial_scatter"
    return "evaluate_other"
end

const EVAL_FUNCS = (:mace_energy!, :mace_energy_forces!)
is_eval_frame(R, f) = f.func in EVAL_FUNCS && basename(String(f.file)) == R.path

# A frame below a matrix product: LinearAlgebra's `mul!` / BLAS wrappers, or a BLAS library.
function is_gemm_frame(f)
    file = String(f.file)
    name = String(f.func)
    return occursin("matmul.jl", file) || occursin("LinearAlgebra", file) || occursin("PureBLAS", file) ||
        occursin("openblas", file) || occursin("blas", lowercase(name)) || occursin("gemm", lowercase(name))
end

# Stage of one sample's frames, innermost first.
function classify(R, frames)
    if !any(f -> f.func == :mace_energy_forces!, frames)
        # OpenBLAS's assembly kernels carry no unwind information, so a sample inside one has no caller.
        return any(f -> occursin("openblas", String(f.file)), frames) ? "blas_stack_lost" : "outside"
    end
    i = findlast(f -> haskey(FUNCTION_STAGE, f.func), frames)   # outermost stage function
    if i !== nothing
        stage = FUNCTION_STAGE[frames[i].func]
        if startswith(stage, "linear")
            # The linear layers called from the readout region are the readouts' own.
            j = findnext(f -> is_eval_frame(R, f), frames, i)
            if j !== nothing
                ls = line_stage(R, frames[j].line)
                startswith(ls, "readouts") && return ls
            end
            return stage
        elseif startswith(stage, "radial_mlp")
            kind = any(is_gemm_frame, view(frames, 1:(i - 1))) ? "gemm" : "other"
            return string("radial_mlp_", kind, endswith(stage, "fwd") ? "_fwd" : "_bwd")
        end
        return stage
    end
    j = findfirst(f -> f.func in (:_messages!, :_messages_pullback!) && basename(String(f.file)) == R.path, frames)
    if j !== nothing
        return frames[j].func === :_messages! ? "messages_other_fwd" : "messages_other_bwd"
    end
    j = findfirst(f -> is_eval_frame(R, f), frames)
    j === nothing && return "outside"
    return line_stage(R, frames[j].line)
end

function count_stages(R, data, lidict)
    counts = Dict(s => 0 for s in STAGES)
    stack = Base.StackTraces.StackFrame[]
    for ip in data
        if ip == 0
            isempty(stack) || (counts[classify(R, stack)] += 1)
            empty!(stack)
        else
            append!(stack, get(lidict, ip, Base.StackTraces.StackFrame[]))
        end
    end
    return counts
end

function dgemm_owner()
    p = BLAS.lbt_get_forward("dgemm_", :ilp64)
    info = Ref{NTuple{4, Ptr{Cvoid}}}()
    ccall(:dladdr, Cint, (Ptr{Cvoid}, Ptr{NTuple{4, Ptr{Cvoid}}}), p, info) == 0 && return "unknown"
    return basename(unsafe_string(Ptr{UInt8}(info[][1])))
end

function git_commit(dir)
    try
        c = readchomp(`git -C $dir rev-parse --short=8 HEAD`)
        dirty = !isempty(readchomp(`git -C $dir status --porcelain -- src`))
        return dirty ? c * "-dirty" : c
    catch
        return "unknown"
    end
end

# PureBLAS commit from the active project's manifest.
function pureblas_commit()
    manifest = joinpath(dirname(Base.active_project()), "Manifest.toml")
    isfile(manifest) || return "unknown"
    m = match(r"\[\[deps\.PureBLAS\]\].*?repo-rev = \"([0-9a-f]+)\""s, read(manifest, String))
    return m === nothing ? "unknown" : m[1][1:8]
end

function activate_backend()
    if BLAS_ARM == "pureblas"
        PB = Base.require(Main, :PureBLAS)
        Base.invokelatest(PB.activate)
        Base.invokelatest(PB.is_active) || error("PureBLAS did not become active")
        return (; backend = "PureBLAS", commit = pureblas_commit(), dgemm_owner = dgemm_owner())
    end
    owner = dgemm_owner()
    occursin("openblas", owner) || error("OpenBLAS is not the active BLAS: dgemm is owned by ", owner)
    return (; backend = "OpenBLAS", library = owner, config = [basename(l.libname) for l in BLAS.get_config().loaded_libs])
end

function make_case(N, T)
    cu = cu_bench(N)
    model = MACEModel{T}(load_mace_checkpoint(CK))
    positions = Matrix{T}(cu.positions')
    cell = Matrix{T}(cu.cell')
    z = cu.atomic_numbers
    nl = neighbor_list(positions, cell, model.r_max; extra_capacity = 64)
    ws = MACEWorkspace{T}(model, length(z), nl.n_edges; elements = z)
    res = MACEResult{T}(length(z))
    return (; n = length(z), n_edges = nl.n_edges, eval! = () -> mace_energy_forces!(res, ws, model, positions, cell, z, nl))
end

function profile_case(R, case)
    case.eval!()   # compile and warm up
    Profile.init(n = 10^8, delay = DELAY)
    Profile.clear()
    evals = 0
    t0 = time()
    Profile.@profile begin
        while evals == 0 || time() - t0 < SECONDS
            case.eval!()
            evals += 1
        end
    end
    data = Profile.fetch()
    lidict = Profile.getdict(data)
    return count_stages(R, data, lidict), evals
end

function merge_into(out, header, backend_name, backend_info, key, n, n_edges, counts, evals)
    doc = isfile(out) ? JSON.parsefile(out) : Dict{String, Any}()
    blas = get(get(doc, "header", Dict{String, Any}()), "blas", Dict{String, Any}())
    blas[backend_name] = backend_info
    header["blas"] = blas
    doc["header"] = header
    cases = get!(doc, "cases", Dict{String, Any}())
    entry = get!(cases, key, Dict{String, Any}("n_atoms" => n, "n_edges" => n_edges, "runs" => 0, "evaluations" => 0, "counts" => Dict{String, Any}()))
    entry["runs"] += 1
    entry["evaluations"] += evals
    for (s, c) in counts
        entry["counts"][s] = get(entry["counts"], s, 0) + c
    end
    total = sum(values(entry["counts"]))
    entry["samples"] = total
    entry["shares"] = Dict(s => entry["counts"][s] / total for s in STAGES if haskey(entry["counts"], s))
    open(out, "w") do io
        JSON.print(io, doc, 2)
    end
    return entry
end

function main()
    R = line_regions()
    backend = activate_backend()
    header = Dict{String, Any}(
        "pureMLIP_commit" => git_commit(pkgdir(PureMLIP)),
        "julia" => string(VERSION), "blas" => Dict{String, Any}(),
        "profile_delay_s" => DELAY, "threads" => 1, "blas_threads" => 1, "model" => "mace-mpa-0-medium",
    )
    for (N, T) in SPECS
        case = make_case(N, T)
        counts, evals = profile_case(R, case)
        total = sum(values(counts))
        key = string(backend.backend, "_cu", N, "_", T)
        entry = merge_into(OUT, header, backend.backend, Dict(string(k) => v for (k, v) in pairs(backend)), key, case.n, case.n_edges, counts, evals)
        @printf("%s: %d atoms, this run %d samples over %d evaluations, accumulated %d samples\n", key, case.n, total, evals, entry["samples"])
        for s in STAGES
            c = get(entry["counts"], s, 0)
            c == 0 || @printf("  %-28s %6.2f%%\n", s, 100 * c / entry["samples"])
        end
    end
    return nothing
end

main()
