# Interleaved timing of PureXC's `xc_evaluate!` against `Libxc.evaluate!`.
#
#   julia --project=PureXC/bench PureXC/bench/libxc_speed.jl
#
# macOS only: it reads `sysctl` (swap usage, CPU brand). Each arm runs in its own child process
# with one thread (`-t 1`, `OMP_NUM_THREADS=1`); a round runs one child of each arm, and the arm
# order alternates from kept round to kept round (PureXC first in odd ones, Libxc first in even
# ones; a discarded round is run again in its own order) so that a steady drift of the machine
# does not favor one arm. A child builds the inputs, warms up each cell, then reports the median
# of three one-evaluation Chairmarks samples per cell. The speed-up of a round is the Libxc time
# over the PureXC time, so a value above 1 means PureXC is faster.
#
# A cell is a functional (`lda_x`, `lda_c_pw`, `lda_c_pw_mod`, `pbe_x`, `pbe_c`, and `pbe()`, which
# Libxc evaluates as `gga_x_pbe` and `gga_c_pbe` into separate outputs, without the sum), a spin
# case, a derivative order k ∈ {0, 1, 2} (both arms compute every order 0 to k into preallocated
# outputs) and a precision of the PureXC arm. The PureXC arm includes its input validation.
#
# The inputs are POINTS points drawn with replacement (fixed seed) from the "above" population of
# the oracle grid (`PureXC/test/libxc/grid.jl`) of the functional and spin case. The Float64 cells
# draw from the points above the Float64 thresholds. The Float32 cells draw from the points that,
# rounded to Float32, are also above the Float32 thresholds and accepted by the Float32
# functional at order 2 (the Float32 ε_ζ is eps(Float32), and rounding a subnormal Float32 σ can
# push σαβ past its bound); Libxc is timed on the same points in Float64, and that time is the
# reference of the Float32 speed-up.
#
# A round is discarded, and run again, when swap usage grew while it ran, or when a fixed scalar
# calibration loop (independent of both libraries) shows the children ran at different speeds:
# their calibration medians differ by more than CAL_PAIR_TOLERANCE, or one child's calibration
# varies by more than CAL_WITHIN_TOLERANCE.
#
#   PUREXC_SPEED_OUT     JSON output path (default: a new file under the system temp directory,
#                        whose path is printed; pass PureXC/bench/results/... to keep a record)
#   PUREXC_SPEED_ROUNDS  kept rounds (default 7)
#   PUREXC_SPEED_POINTS  points per cell (default 1000000)
#   PUREXC_SPEED_ONLY    comma-separated functional names to run (default: all six)
#   PUREXC_SPEED_CAL_TOLERANCE  "<within>,<pair>" fractions (default "0.03,0.01")
using Chairmarks, JSON, Pkg, Printf, Random, Statistics

const SAMPLES = 3
const WARMUP_EVALS = 2
const MAX_DISCARDS = 20
const CAL_REPEATS = 30
const SEED = 20261121
const CAL_WITHIN_TOLERANCE, CAL_PAIR_TOLERANCE =
    parse.(Float64, split(get(ENV, "PUREXC_SPEED_CAL_TOLERANCE", "0.03,0.01"), ','))
const FUNCTIONALS = ("lda_x", "lda_c_pw", "lda_c_pw_mod", "pbe_x", "pbe_c", "pbe")
const LIBXC_IDS = Dict(
    "lda_x" => (:lda_x,), "lda_c_pw" => (:lda_c_pw,), "lda_c_pw_mod" => (:lda_c_pw_mod,),
    "pbe_x" => (:gga_x_pbe,), "pbe_c" => (:gga_c_pbe,), "pbe" => (:gga_x_pbe, :gga_c_pbe),
)
const SPINS = (:restricted, :polarized)
const ORDERS = (0, 1, 2)

swap_used_mb() = parse(Float64, match(r"used = ([0-9.]+)M", read(`sysctl -n vm.swapusage`, String))[1])

selected_functionals() = Tuple(split(get(ENV, "PUREXC_SPEED_ONLY", join(FUNCTIONALS, ",")), ','))

# ---------------------------------------------------------------------------------------------
# Child: measures every cell of one arm and prints `RESULT <cell> <seconds> <cal0> <cal1>` lines.
# ---------------------------------------------------------------------------------------------

# Time of a fixed scalar loop (minimum over repeats). It touches neither library, so it moves
# only with the speed of the core the process is on.
function calibrate()
    best = Inf
    for _ in 1:CAL_REPEATS
        t = time_ns()
        x = 1.0
        for i in 1:1_000_000
            x = muladd(x, 0.9999999, 1.0e-9 * i)
        end
        best = min(best, (time_ns() - t) / 1.0e9 + 0 * x)
    end
    return best
end

constructor(PX, name) = getfield(PX, Symbol(name))

# The two populations of a functional and spin case: points above the Float64 thresholds, and the
# subset of those that stay above, and accepted, in Float32.
function populations(PX, name, spin)
    f64, f32 = constructor(PX, name)(Float64), constructor(PX, name)(Float32)
    pts = Main.grid_points(PX.family(f64), spin)
    above = filter(p -> Main.is_above(p, Main.thresholds_of(f64)), pts)
    to32(p) = map(x -> x isa Tuple ? Float32.(x) : Float32(x), p)
    function ok32(p)
        q = to32(p)
        Main.is_above(q, Main.thresholds_of(f32)) || return false
        try
            PX.xc_derivatives(f32, Val(2), q...)
        catch err
            err isa ArgumentError || rethrow()
            return false
        end
        return true
    end
    return (; f64 = above, f32 = filter(ok32, above), grid = length(pts))
end

# POINTS draws from `pop` as the arrays of the arm's inputs, in the type T.
function draw(pop, npoints, ::Type{T}, spin, seed) where {T}
    idx = rand(Xoshiro(seed), eachindex(pop), npoints)
    gga = length(first(pop)) == 2
    if spin === :restricted
        ρ = T[pop[i][1] for i in idx]
        return gga ? (ρ, T[pop[i][2] for i in idx]) : (ρ,)
    end
    ρ = Matrix{T}(undef, 2, npoints)
    σ = Matrix{T}(undef, 3, npoints)
    for (j, i) in pairs(idx)
        ρ[:, j] .= pop[i][1]
        gga && (σ[:, j] .= pop[i][2])
    end
    return gga ? (ρ, σ) : (ρ,)
end

cell_key(name, spin, k, prec) = "$name $spin k=$k $prec"

function libxc_outputs(spin, gga, k, npoints)
    pol = spin === :polarized
    o = Dict{Symbol, Array{Float64}}(:zk => zeros(npoints))
    if k >= 1
        o[:vrho] = zeros(pol ? 2 : 1, npoints)
        gga && (o[:vsigma] = zeros(pol ? 3 : 1, npoints))
    end
    if k >= 2
        o[:v2rho2] = zeros(pol ? 3 : 1, npoints)
        gga && (o[:v2rhosigma] = zeros(pol ? 6 : 1, npoints))
        gga && (o[:v2sigma2] = zeros(pol ? 6 : 1, npoints))
    end
    return (; o...)
end

function time_cell(run)
    for _ in 1:WARMUP_EVALS
        run()
    end
    cal0 = calibrate()
    t = median([s.time for s in (@be run() evals = 1 samples = SAMPLES seconds = 600).samples])
    cal1 = calibrate()
    return t, cal0, cal1
end

function child_main(PX, LX, arm, npoints)
    Threads.nthreads() == 1 || error("expected one thread, got ", Threads.nthreads())
    for (fi, name) in enumerate(selected_functionals()), (si, spin) in enumerate(SPINS)
        pops = populations(PX, name, spin)
        println("POPULATION\t", name, " ", spin, "\t", pops.grid, "\t", length(pops.f64), "\t", length(pops.f32))
        flush(stdout)
        gga = PX.family(constructor(PX, name)()) === :gga
        for (prec, pop) in (("Float64", pops.f64), ("Float32", pops.f32))
            seed = SEED + 100 * fi + 10 * si + (prec == "Float32")
            T = prec == "Float64" ? Float64 : Float32
            if arm == "purexc"
                x = draw(pop, npoints, T, spin, seed)
                f = constructor(PX, name)(T)
                for k in ORDERS
                    out = PX.xc_outputs(f, Val(k), x...)
                    r = time_cell(() -> PX.xc_evaluate!(out, f, Val(k), x...))
                    println("RESULT\t", cell_key(name, spin, k, prec), "\t", join(r, '\t'))
                    flush(stdout)
                end
            else
                x = draw(pop, npoints, Float64, spin, seed)
                fs = map(id -> LX.Functional(id; n_spin = spin === :restricted ? 1 : 2), LIBXC_IDS[name])
                kw = gga ? (; rho = x[1], sigma = x[2]) : (; rho = x[1])
                for k in ORDERS
                    outs = map(_ -> libxc_outputs(spin, gga, k, npoints), fs)
                    run = () -> foreach((g, o) -> LX.evaluate!(g; kw..., o...), fs, outs)
                    r = time_cell(run)
                    println("RESULT\t", cell_key(name, spin, k, prec), "\t", join(r, '\t'))
                    flush(stdout)
                end
            end
        end
    end
    println("PKG\t", join(("$(v.name)=$(v.version)" for (_, v) in Pkg.dependencies() if v.version !== nothing), ","))
    if arm == "libxc"
        println("LIBXC_C\t", LX.libxc_version)
    end
    return
end

# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------
git(dir, args...) = readchomp(`git -C $dir $args`)

function commit_label(dir)
    sha = git(dir, "rev-parse", "--short=12", "HEAD")
    return isempty(git(dir, "status", "--porcelain", "--untracked-files=no", "--", "PureXC", "PureTaylor")) ? sha : sha * "-dirty"
end

function run_child(arm, npoints)
    env = joinpath(@__DIR__)
    cmd = setenv(
        `$(Base.julia_cmd()) -t 1 --project=$env $(@__FILE__) --child $arm $npoints`,
        merge(ENV, Dict("OMP_NUM_THREADS" => "1")),
    )
    out = read(cmd, String)
    results = Dict{String, NTuple{3, Float64}}()
    pops = Dict{String, Vector{Int}}()
    pkgs = Dict{String, String}()
    libxc_c = ""
    for line in split(out, '\n')
        f = split(line, '\t')
        if f[1] == "RESULT"
            results[f[2]] = (parse(Float64, f[3]), parse(Float64, f[4]), parse(Float64, f[5]))
        elseif f[1] == "POPULATION"
            pops[f[2]] = parse.(Int, f[3:5])
        elseif f[1] == "PKG"
            for kv in split(f[2], ',')
                name, ver = split(kv, '=')
                pkgs[name] = ver
            end
        elseif f[1] == "LIBXC_C"
            libxc_c = String(f[2])
        end
    end
    return (; results, pops, pkgs, libxc_c)
end

calibrations(r) = Dict(k => [v[2], v[3]] for (k, v) in r)

function discard_reason(rp, rl, sw0, sw1)
    sw1 > sw0 + 1 && return "swap usage grew during the round"
    cals(r) = [c for v in values(r) for c in v[2:3]]
    cp, cl = cals(rp), cals(rl)
    for (name, c) in (("PureXC", cp), ("Libxc", cl))
        spread = maximum(c) / minimum(c) - 1
        spread > CAL_WITHIN_TOLERANCE && return @sprintf("calibration varied by %.1f%% within the %s child", 100 * spread, name)
    end
    d = median(cp) / median(cl) - 1
    abs(d) > CAL_PAIR_TOLERANCE && return @sprintf("calibration medians differ by %.1f%% between the children", 100 * d)
    return nothing
end

function cells()
    return [
        (; functional = n, spin = string(s), order = k, precision = p)
            for n in selected_functionals() for s in SPINS for p in ("Float64", "Float32") for k in ORDERS
    ]
end

function summarize(kept)
    return map(cells()) do c
        key = cell_key(c.functional, c.spin, c.order, c.precision)
        tp = [r.purexc[key][1] for r in kept]
        tl = [r.libxc[key][1] for r in kept]
        s = tl ./ tp
        (; c..., gated = c.precision == "Float64", speedups = s, median = median(s), min = minimum(s), max = maximum(s), purexc_s = tp, libxc_s = tl)
    end
end

function driver_main()
    root = git(@__DIR__, "rev-parse", "--show-toplevel")
    nrounds = parse(Int, get(ENV, "PUREXC_SPEED_ROUNDS", "7"))
    npoints = parse(Int, get(ENV, "PUREXC_SPEED_POINTS", "1000000"))
    out = get(ENV, "PUREXC_SPEED_OUT", tempname() * "_libxc_speed.json")
    head = commit_label(root)
    println("PureXC at ", head, ", ", npoints, " points, ", nrounds, " rounds")
    flush(stdout)
    kept = Any[]
    discarded = Any[]
    last = Dict{String, Any}()
    sw_first = swap_used_mb()
    write_doc = function (sw_last)
        doc = (;
            schema = "purexc-libxc-speed-v1",
            hardware = Sys.isapple() ? strip(read(`sysctl -n machdep.cpu.brand_string`, String)) : string(Sys.CPU_NAME),
            julia_version = string(VERSION), purexc_commit = head,
            purexc_version = get(get(last, "purexc", (; pkgs = Dict())).pkgs, "PureXC", nothing),
            puretaylor_version = get(get(last, "purexc", (; pkgs = Dict())).pkgs, "PureTaylor", nothing),
            libxc_jl_version = get(get(last, "libxc", (; pkgs = Dict())).pkgs, "Libxc", nothing),
            libxc_jll_version = get(get(last, "libxc", (; pkgs = Dict())).pkgs, "Libxc_jll", nothing),
            libxc_c_version = get(last, "libxc", (; libxc_c = nothing)).libxc_c,
            chairmarks_version = get(get(last, "purexc", (; pkgs = Dict())).pkgs, "Chairmarks", nothing),
            threads = 1, omp_num_threads = 1, points = npoints, seed = SEED,
            samples_per_cell = SAMPLES, warmup_evaluations = WARMUP_EVALS,
            speedup = "Libxc time / PureXC time per round; above 1 means PureXC is faster",
            gate = "Float64 cells: the minimum speed-up over the kept rounds must be above 1; Float32 cells are reported against Libxc's Float64 time on the same points and not gated",
            order = "odd rounds run PureXC then Libxc, even rounds Libxc then PureXC",
            calibration = "seconds of a fixed scalar loop before and after each cell; rounds are discarded when the two children's medians differ by more than $(100 * CAL_PAIR_TOLERANCE)% or one child varies by more than $(100 * CAL_WITHIN_TOLERANCE)%",
            populations = get(last, "purexc", (; pops = Dict())).pops,
            populations_columns = ["grid points", "above (Float64 cells)", "above in Float32 too (Float32 cells)"],
            swap_first_start_mb = sw_first, swap_last_end_mb = sw_last,
            rounds = [(; r.round, r.order, r.swap_before_mb, r.swap_after_mb, purexc_calibration_s = calibrations(r.purexc), libxc_calibration_s = calibrations(r.libxc)) for r in kept],
            discarded_rounds = discarded,
            cells = isempty(kept) ? [] : summarize(kept),
        )
        open(io -> (JSON.json(io, doc; pretty = true); println(io)), out, "w")
        return doc
    end
    attempt = 0
    while length(kept) < nrounds
        attempt += 1
        # The order follows the kept rounds, so a rerun of a discarded round keeps its order.
        purexc_first = iseven(length(kept))
        sw0 = swap_used_mb()
        r = Dict{String, Any}()
        for arm in (purexc_first ? ("purexc", "libxc") : ("libxc", "purexc"))
            r[arm] = run_child(arm, npoints)
        end
        last["purexc"], last["libxc"] = r["purexc"], r["libxc"]
        sw1 = swap_used_mb()
        order = purexc_first ? "PureXC,Libxc" : "Libxc,PureXC"
        reason = discard_reason(r["purexc"].results, r["libxc"].results, sw0, sw1)
        if reason !== nothing
            println("round discarded: ", reason)
            flush(stdout)
            push!(
                discarded, (;
                    round = attempt, order, reason, swap_before_mb = sw0, swap_after_mb = sw1,
                    purexc_s = Dict(k => v[1] for (k, v) in r["purexc"].results), libxc_s = Dict(k => v[1] for (k, v) in r["libxc"].results),
                    purexc_calibration_s = calibrations(r["purexc"].results), libxc_calibration_s = calibrations(r["libxc"].results),
                )
            )
            if length(discarded) >= MAX_DISCARDS
                write_doc(swap_used_mb())
                println("results written to ", out)
                error(length(discarded), " discards (swap growth or calibration): the machine is too busy")
            end
            continue
        end
        push!(kept, (; round = attempt, order, swap_before_mb = sw0, swap_after_mb = sw1, purexc = r["purexc"].results, libxc = r["libxc"].results))
        @printf("round %d kept (%s)\n", length(kept), order)
        flush(stdout)
        write_doc(swap_used_mb())
    end
    sw_last = swap_used_mb()
    doc = write_doc(sw_last)
    for c in doc.cells
        @printf(
            "%-13s %-10s k=%d %-7s median %6.3f  min %6.3f  max %6.3f%s\n", c.functional, c.spin, c.order, c.precision,
            c.median, c.min, c.max, c.gated && c.min <= 1 ? "  FAIL" : ""
        )
    end
    @printf("%d rounds kept, %d discarded; swap at start %.2f MB, at end %.2f MB\n", length(kept), length(discarded), sw_first, sw_last)
    println("results written to ", out)
    return
end

# Included from another script (abspath(PROGRAM_FILE) differs), it only defines the functions.
if abspath(PROGRAM_FILE) != @__FILE__
elseif !isempty(ARGS) && ARGS[1] == "--child"
    # Loaded at top level, so that the code compiled after this sees their methods.
    const PX = Base.require(Main, :PureXC)
    include(joinpath(@__DIR__, "..", "test", "libxc", "grid.jl"))
    const LX = ARGS[2] == "libxc" ? Base.require(Main, :Libxc) : nothing
    Base.invokelatest(child_main, PX, LX, ARGS[2], parse(Int, ARGS[3]))
else
    driver_main()
end
