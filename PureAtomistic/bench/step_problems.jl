# The step problems of TrustRegionBFGS on the two Cu cells, solved with the four PureDAQP settings
# (working_set :rows/:gram × scan :all/:window), with the conditioning of their working sets.
#
#   PUREMLIP_CHECKPOINT_DIR=<dir> julia -t 1 --project=PureAtomistic/bench PureAtomistic/bench/step_problems.jl
#
# Every step problem of a relaxation (fmax = 1e-3) is captured: the rows, bounds and linear term as
# the relaxer poses them (an identity `Diagonal` objective over rows rewritten in place), and its
# inverse Hessian, gradient and radius. Each problem is also posed with the trust radius cut to half
# and to a tenth of the largest component of its unconstrained step, where the box binds
# (`with_radius`). Each problem is then solved in two forms:
#
#   in_place  the relaxer's own path: a workspace set up once per setting, the rows copied into the
#             matrix it holds, then `update!`, `cold_start!` and `solve!`;
#   dense     the same step as `solve(B, g, I, -Δ, Δ)` with B = H⁻¹ dense, set up from scratch, which
#             is what passing the curvature as the objective costs per step.
#
# Timing interleaves the settings: each repetition times every setting once, in an order rotated by
# the repetition, and the median of REPS repetitions is kept. Speed is reported as the ratio of the
# summed medians to those of (:rows, :all), the default. `faster_scan` is run on every problem of
# the in-place form for each working set. One thread, BLAS on one thread.
#
# Writes PureAtomistic/bench/results/step_problems_mac.json (or PUREATOMISTIC_STEP_OUT).
using JSON, Printf, Statistics
import PureDAQP
using PureDAQP: ActiveSet, faster_scan
include(joinpath(@__DIR__, "cells.jl"))
using PureAtomistic.Integrators: TrustRegionBFGS

BLAS.set_num_threads(1)
const REPS = 15
const SMOKE = get(ENV, "PUREATOMISTIC_STEP_SMOKE", "") == "1"
const SETTINGS = [(ws, sc) for ws in (:rows, :gram) for sc in (:all, :window)]

function capture(model, build)
    sys = build()
    opt = TrustRegionBFGS(sys)
    problems = NamedTuple[]
    first = Ref(true)
    hook = function ()
        # The first evaluation is the starting point; every later one is a trial step, whose problem
        # is still in the state.
        first[] && (first[] = false; return)
        return push!(
            problems, (
                M = copy(opt.M), q = copy(opt.q), l = copy(opt.lower), u = copy(opt.upper),
                H = copy(opt.H), g = copy(opt.gradv), radius = opt.radius,
            )
        )
    end
    p = Counting(mace(model, sys); hook)
    info = relax!(sys, p, opt; fmax = SMOKE ? 1.0e-6 : 1.0e-3)
    return problems, info
end

function in_place_workspace(n, m, alg)
    M = zeros(m, n)
    for i in 1:m
        M[i, i <= n ? i : 1] = 1
    end
    ws = PureDAQP.setup(Diagonal(ones(n)), zeros(n), M, fill(-0.1, m), fill(0.1, m), alg; max_iter = 10 * (m + n))
    return ws, M
end

function solve_in_place!(ws, M, pr)
    copyto!(M, pr.M)
    PureDAQP.update!(ws; q = pr.q, l = pr.l, u = pr.u)
    PureDAQP.cold_start!(ws)
    return PureDAQP.solve!(ws)
end

function solve_dense(pr, alg)
    n = length(pr.g)
    B = Matrix(Symmetric(inv(Symmetric(pr.H))))
    A = Matrix{Float64}(I, n, n)
    return PureDAQP.solve(B, pr.g, A, fill(-pr.radius, n), fill(pr.radius, n), alg; max_iter = 20n)
end

# The captured problem (`:as_run`), and the same problem with the trust radius set to a fraction of
# the largest component of its unconstrained step -Hg, so that the box binds. The row bounds scale
# with the radius, since every box row is ±radius over its norm.
const RADIUS_SCALES = (:as_run, 0.5, 0.1)
function with_radius(pr, scale)
    scale === :as_run && return pr
    r = scale * maximum(abs, pr.H * pr.g)
    f = r / pr.radius
    return merge(pr, (l = pr.l .* f, u = pr.u .* f, radius = r))
end

function summarize(records)
    summary = Dict{String, Any}()
    for form in ("in_place", "dense"), s in SETTINGS
        summary["$(form)_total_s_$(s[1])_$(s[2])"] = sum(r["$(form)_s_$(s[1])_$(s[2])"] for r in records)
    end
    for form in ("in_place", "dense"), s in SETTINGS
        summary["$(form)_speedup_vs_rows_all_$(s[1])_$(s[2])"] =
            summary["$(form)_total_s_rows_all"] / summary["$(form)_total_s_$(s[1])_$(s[2])"]
    end
    summary["build_total_s"] = sum(r["build_s"] for r in records)
    summary["max_cond_active"] = maximum(r["cond_active"] for r in records)
    summary["max_cond_M"] = maximum(r["cond_M"] for r in records)
    summary["max_active_rows"] = maximum(r["active_rows"] for r in records)
    summary["all_solved"] = all(all(v == "SOLVED" for (k, v) in r if startswith(k, "status_") || startswith(k, "dense_status_")) for r in records)
    summary["max_dx_vs_rows_all"] = maximum(maximum(v for (k, v) in r if startswith(k, "dx_") && !isnothing(v); init = 0.0) for r in records)
    summary["faster_scan_window_count"] = Dict(ws => count(r -> r["faster_scan_$(ws)"] == "window", records) for ws in ("rows", "gram"))
    summary["faster_scan_statuses_differ"] = Dict(ws => count(r -> r["faster_scan_$(ws)"] == "statuses differ", records) for ws in ("rows", "gram"))
    return summary
end

# Condition number of a set of rows (largest over smallest singular value; one for no rows).
rowcond(R) = size(R, 1) == 0 ? 1.0 : (s = svdvals(R); s[1] / s[end])

function study(model, name, build)
    problems, info = capture(model, build)
    n = length(problems[1].q)
    m = size(problems[1].M, 1)
    @printf("%s: %d step problems (%d steps, converged %s), n = %d, m = %d\n", name, length(problems), info.steps, info.converged, n, m)
    flush(stdout)
    algs = Dict(s => ActiveSet(; working_set = s[1], scan = s[2]) for s in SETTINGS)
    wss = Dict(s => in_place_workspace(n, m, algs[s]) for s in SETTINGS)
    records = Dict{String, Any}[]
    for (k, pr0) in enumerate(problems), scale in RADIUS_SCALES
        pr = with_radius(pr0, scale)
        # Reference solution and its active rows, from the default setting.
        ref = solve_in_place!(wss[(:rows, :all)]..., pr)
        x_ref = copy(ref.x)
        active = findall(!iszero, ref.y)
        rec = Dict{String, Any}(
            "step" => k, "radius_scale" => string(scale), "radius" => pr.radius, "active_rows" => length(active),
            "cond_M" => rowcond(pr.M), "cond_active" => rowcond(pr.M[active, :]),
            "cond_H" => cond(Symmetric(pr.H)),
        )
        for s in SETTINGS
            sol = solve_in_place!(wss[s]..., pr)
            rec["status_$(s[1])_$(s[2])"] = string(sol.status)
            rec["iter_$(s[1])_$(s[2])"] = sol.iter
            # `nothing` (null) when the setting returned no point.
            dx = maximum(abs, sol.x - x_ref)
            rec["dx_$(s[1])_$(s[2])"] = isfinite(dx) ? dx : nothing
            dsol = solve_dense(pr, algs[s])
            rec["dense_status_$(s[1])_$(s[2])"] = string(dsol.status)
        end
        t_in = Dict(s => Float64[] for s in SETTINGS)
        t_dn = Dict(s => Float64[] for s in SETTINGS)
        for r in 1:REPS
            order = circshift(SETTINGS, r)
            for s in order
                push!(t_in[s], @elapsed solve_in_place!(wss[s]..., pr))
            end
            for s in order
                push!(t_dn[s], @elapsed solve_dense(pr, algs[s]))
            end
        end
        # The relaxer's own work around the solve: Cholesky of H, the rows and bounds, and q.
        opt = TrustRegionBFGS{Float64}(div(n, 3))
        opt.H .= pr.H
        opt.gradv .= pr.g
        opt.radius = pr.radius
        PureAtomistic.Integrators._step_problem!(opt)
        rec["build_s"] = median([@elapsed(PureAtomistic.Integrators._step_problem!(opt)) for _ in 1:REPS])
        for s in SETTINGS
            rec["in_place_s_$(s[1])_$(s[2])"] = median(t_in[s])
            rec["dense_s_$(s[1])_$(s[2])"] = median(t_dn[s])
        end
        for ws in (:rows, :gram)
            # `faster_scan` throws when the two scans reach different statuses; that is recorded.
            fs = try
                faster_scan(Diagonal(ones(n)), pr.q, pr.M, pr.l, pr.u; alg = ActiveSet(; working_set = ws), reps = 5, max_iter = 10 * (m + n))
            catch err
                err isa ArgumentError || rethrow()
                nothing
            end
            rec["faster_scan_$(ws)"] = isnothing(fs) ? "statuses differ" : string(fs.scan)
            rec["faster_scan_$(ws)_ratio"] = isnothing(fs) ? nothing : fs.ratio
        end
        # A problem some setting does not solve is kept outside the repository for a reproducer.
        if any(v -> v isa String && v != "SOLVED", (rec[k] for k in keys(rec) if occursin("status_", k)))
            dump = joinpath(tempdir(), "step_problem_$(name)_$(k)_$(scale).json")
            open(io -> JSON.json(io, Dict(pairs(pr))), dump, "w")
            println("  unsolved by some setting; problem written to ", dump)
        end
        push!(records, rec)
        @printf(
            "  step %2d %-6s: active %3d  cond(M) %.2e  cond(active) %.2e  in-place rows/all %.3f ms  dense rows/all %.2f ms\n",
            k, string(scale), length(active), rec["cond_M"], rec["cond_active"], 1.0e3 * rec["in_place_s_rows_all"], 1.0e3 * rec["dense_s_rows_all"]
        )
        flush(stdout)
    end
    summary = Dict(string(sc) => summarize(filter(r -> r["radius_scale"] == string(sc), records)) for sc in RADIUS_SCALES)
    println("  summary: ", summary)
    flush(stdout)
    return Dict("cell" => String(name), "n" => n, "m" => m, "relax_steps" => info.steps, "problems" => records, "summary" => summary)
end

# PUREATOMISTIC_STEP_SMOKE=1 runs the script end to end on PureMLIP's tiny checkpoint and a rattled
# 32-atom cell, to check it before a long run; its output is not a measurement.
smoke_cell() = (s = fcc_cell(A0, 2, :Cu); s.positions .+= 0.05 .* randn(Xoshiro(3), size(s.positions)); s)
model = SMOKE ? MACEModel(load_mace_checkpoint(PureMLIP.tiny_mace_checkpoint())) : MACEModel(load_mace_checkpoint(mpa0_path()))
cells_run = SMOKE ? (smoke = smoke_cell,) : CELLS
out = Dict{String, Any}(
    "description" => "TrustRegionBFGS step problems on MPA-0 Cu cells, four PureDAQP settings; seconds are medians of interleaved repetitions",
    "reps" => REPS, "julia" => string(VERSION), "cpu" => host_cpu(), "threads" => Threads.nthreads(),
    "puredaqp" => string(pkgversion(PureDAQP)),
    "cells" => [study(model, name, build) for (name, build) in pairs(cells_run)],
)
path = get(ENV, "PUREATOMISTIC_STEP_OUT", joinpath(@__DIR__, "results", "step_problems_mac.json"))
open(io -> (JSON.json(io, out; pretty = true); println(io)), path, "w")
println("wrote ", path)
