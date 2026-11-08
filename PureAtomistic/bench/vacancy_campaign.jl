# Vacancy-diffusion campaign: the hop rate of one vacancy in Cu with MACE-MPA-0 (Float32, on a GPU)
# at 1200, 1080, 980 and 900 K, two runs per temperature running as two processes at once, for an
# Arrhenius fit of the migration energy and prefactor
# (docs/superpowers/specs/2026-11-08-vacancy-campaign-proposal.md).
#
#   VACANCY_CAMPAIGN_DIR=<directory outside the repository> PUREMLIP_CHECKPOINT_DIR=<dir> \
#   PUREATOMISTIC_CPULOCK=<lock script> HWLOC_COMPONENTS=-darwin \
#       julia --project=PureAtomistic/bench/gpu/metal PureAtomistic/bench/vacancy_campaign.jl campaign
#
# Running the same command again resumes the campaign where it stopped: each run continues from
# its last stored segment. `summary` rebuilds the summary record from the run records. To stop
# it, kill the campaign process; its workers exit when their input closes. A `cpulock.sh acquire`
# that the campaign process was waiting on must be killed with it, or it takes the lock and keeps
# it.
#
#   VACANCY_CAMPAIGN_DIR      checkpoints, trajectories, vacancy tracks and logs (required)
#   PUREATOMISTIC_CPULOCK     the shared lock script (`acquire` prints a token, `release <token>`);
#                             without it no lock is taken
#   VACANCY_CAMPAIGN_MAX_HOURS  hours of computing the campaign may spend (default 150): it stops
#                             before a cycle that would pass them, and after 1200 K when the
#                             re-estimated total at 0.70 eV exceeds them
#   VACANCY_CAMPAIGN_DT_FS    2 or 4: the production time step, in place of the dt_check rule (whose
#                             result is still recorded); stored in the campaign state, and a later
#                             value that differs is an error. VACANCY_CAMPAIGN_DT_REASON is recorded
#                             with it
#   GPU_BACKEND               Metal (default), CUDA or AMDGPU; the project is bench/gpu/<lower case>
#
# Each run: 4 × 4 × 4 fcc cells with one vacancy (255 atoms; 5 × 5 × 5, 499 atoms, for the size
# check), Maxwell-Boltzmann velocities at T from the run's seed, then 20 ps of Bussi (τ 100 fs)
# and Berendsen (P₀ 0, τ_P 500 fs, β_T 1/140 GPa⁻¹), the cell scaled to the mean volume of the
# last 10 ps, 2 ps of Bussi NVT that are discarded, and production NVT with Bussi (τ 100 fs). A
# VacancyTracker (window 5, dwell 3) reads a frame every 20 fs and skips up to 500 fs of
# consecutive frames whose occupancy is not one vacancy (each record counts them); a trajectory
# frame is written every 100 fs. Production runs in segments of 100 ps; the state at the end of
# each segment (positions, velocities, cell, generator, tracker and series) is stored, and a run
# continues from it exactly.
#
# The two runs of a stage run as two worker processes. A cycle advances both by 50,000 steps
# (100 ps at 2 fs, 200 ps at 4 fs) under one token of the lock, which is released between cycles.
# Stages, in order:
#
#   dt_check    1200 K, one run at 2 fs and one at 4 fs, until they hold DT_CHECK_HOPS hops
#               together (the stop depends on the total only, so the split stays binomial). Use
#               4 fs if the exact conditional binomial test gives p >= DT_CHECK_P and the exact
#               (Clopper-Pearson) 95 % interval of the rate ratio Γ(4 fs)/Γ(2 fs) lies within
#               DT_CHECK_RATIO; otherwise 2 fs. The run at the chosen time step is the first
#               1200 K production run.
#   t1200_rest  the second 1200 K run (255 atoms) and the 499-atom size check, both at the chosen
#               time step, each for the whole segments that hold 100 hops at the rate of the
#               first run (a length fixed before they start, so their counts stay Poisson)
#   t1080, t980, t900   two runs each at the chosen time step, extended in cycles until they hold
#               100 hops together
#
# Records (small JSON) go to bench/results/vacancy_campaign/: one per run, dt_check.json, and
# summary.json with the pooled rates, the tests and the fit.
using JSON, Statistics, Dates, Printf, Serialization, LinearAlgebra
include(joinpath(@__DIR__, "cells.jl"))
include(joinpath(@__DIR__, "autocorrelation.jl"))
include(joinpath(@__DIR__, "validation_criteria.jl"))
using PureAtomistic.Systems: maxwell_boltzmann!, temperature, pressure, cell_volume, KB_EV_PER_K, EV_PER_A3_IN_GPA
using PureAtomistic.Analysis: VacancyTracker, reference_sites, update!, hop_rate, msd_diffusion, arrhenius_fit
using PureAtomistic.Barostats: rescale!
using PureAtomistic.IO: TrajectoryWriter, write_frame

const MODE = isempty(ARGS) ? "" : ARGS[1]
MODE in ("campaign", "worker", "summary") || error("usage: vacancy_campaign.jl campaign | worker <run directory> | summary")
Sys.isapple() && get!(ENV, "HWLOC_COMPONENTS", "-darwin")

const BACKEND_NAME = get(ENV, "GPU_BACKEND", "Metal")
BACKEND_NAME in ("Metal", "CUDA", "AMDGPU") || error("GPU_BACKEND must be Metal, CUDA or AMDGPU, got ", BACKEND_NAME)
if MODE == "worker"
    @eval using KernelAbstractions
    @eval using $(Symbol(BACKEND_NAME))
end

# VACANCY_CAMPAIGN_SMOKE=1 shrinks every length (segments of 2 ps, cycles of 1,000 steps, 2 ps of NPT,
# hop targets of 1) and writes the records under VACANCY_CAMPAIGN_DIR, to try the harness end to end.
const SMOKE = get(ENV, "VACANCY_CAMPAIGN_SMOKE", "0") == "1"
const FT = Float32
const FRAME_FS = 20.0
const TRAJ_FS = 100.0
const SEGMENT_PS = SMOKE ? 2.0 : 100.0
const CYCLE_STEPS = SMOKE ? 1_000 : 50_000
const NPT_PS = SMOKE ? 2.0 : 20.0
const NPT_MEAN_PS = SMOKE ? 1.0 : 10.0
const NVT_EQ_PS = SMOKE ? 0.4 : 2.0
const TAU_T_FS = 100.0
const P0_GPA = 0.0
const TAU_P_FS = 500.0
const BETA_T = 1 / 140
const WINDOW, DWELL = 5, 3
const CAPACITY_PS = 24_000.0
const MAXLAG_FRAMES = 2_500
# A tracker frame whose Wigner-Seitz occupancy is not one vacancy is skipped while it is one of at
# most this many fs of consecutive such frames (an atom near the saddle of a jump); beyond it the
# run stops.
const MAX_TRANSIENT_FS = 500.0
const MAX_TRANSIENT = round(Int, MAX_TRANSIENT_FS / FRAME_FS)
const SEED_BASE = 2026110800

const DT_CHECK_T = 1200.0
const DT_CHECK_HOPS = SMOKE ? 1 : 800
const DT_CHECK_MAX_CYCLES = SMOKE ? 10 : 50
const DT_CHECK_P = 0.05
const DT_CHECK_RATIO = (0.75, 1.25)
const TARGETS = SMOKE ? [(1200.0, 2), (1080.0, 1), (980.0, 1), (900.0, 1)] : [(1200.0, 200), (1080.0, 100), (980.0, 100), (900.0, 100)]
const SIZE_CHECK_HOPS = SMOKE ? 1 : 100

const RESULTS = SMOKE ? joinpath(get(ENV, "VACANCY_CAMPAIGN_DIR", tempdir()), "results") : joinpath(@__DIR__, "results", "vacancy_campaign")
const HARNESS = @__FILE__
const MAX_HOURS = parse(Float64, get(ENV, "VACANCY_CAMPAIGN_MAX_HOURS", "150"))
# VACANCY_CAMPAIGN_DT_FS fixes the production time step instead of the dt_check rule; it is stored
# in the campaign state, and a later value that differs is an error. VACANCY_CAMPAIGN_DT_REASON is
# recorded with it.
const DT_OVERRIDE = get(ENV, "VACANCY_CAMPAIGN_DT_FS", "")

function campaign_dir()
    d = get(ENV, "VACANCY_CAMPAIGN_DIR", "")
    isempty(d) && error("set VACANCY_CAMPAIGN_DIR to a directory outside the repository")
    return mkpath(d)
end

# ---------------------------------------------------------------------------------------------
# Provenance: the short commit with "-dirty" appended when bench (except its results), src or ext
# differ from it or hold untracked files, the last commit that changed this file, and whether the
# file is that commit's.
git(args...) = strip(read(pipeline(setenv(`git $args`; dir = @__DIR__); stderr = devnull), String))

function provenance()
    pkg = dirname(@__DIR__)
    changed = read(setenv(`git status --porcelain --untracked-files=normal -- bench src ext ":(exclude)bench/results"`; dir = pkg), String)
    rel = relpath(HARNESS, git("rev-parse", "--show-toplevel"))
    blob = git("hash-object", HARNESS)
    committed = try
        git("rev-parse", "HEAD:$rel")
    catch
        ""
    end
    return Dict{String, Any}(
        "commit" => git("rev-parse", "--short", "HEAD") * (isempty(strip(changed)) ? "" : "-dirty"),
        "harness_commit" => git("log", "-1", "--format=%h", "--", HARNESS),
        "harness_blob" => blob, "harness_is_head" => blob == committed,
        "julia" => string(VERSION), "device" => BACKEND_NAME, "date" => string(now()),
    )
end

readjson(path) = JSON.parsefile(path; dicttype = Dict{String, Any})

function save_json(path, d)
    mkpath(dirname(path))
    open(io -> JSON.print(io, d, 1), path * ".tmp", "w")
    mv(path * ".tmp", path; force = true)
    return path
end

function atomic_serialize(path, x)
    serialize(path * ".tmp", x)
    mv(path * ".tmp", path; force = true)
    return path
end

# Standard error of the mean of `x` from `nb` consecutive blocks; `nothing` when a block would hold
# fewer than 2 samples.
function block_se(x, nb = 10)
    bs = length(x) ÷ nb
    bs >= 2 || return nothing
    return std([mean(@view x[((b - 1) * bs + 1):(b * bs)]) for b in 1:nb]) / sqrt(nb)
end

# ---------------------------------------------------------------------------------------------
# Worker: one run, driven by commands on standard input. Replies are lines starting with "@@" on
# standard output; everything else goes to standard error.
#   status            reply with the state
#   advance_to <n>    equilibrate if not done, then run production to step n (whole segments)
#   finish [status]   store the state, write the record with `status` (default "completed") and exit
reply(s) = (println(stdout, "@@ ", s); flush(stdout))
logln(s) = (println(stderr, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "  ", s); flush(stderr))

mutable struct Worker{S, H, P}
    const dir::String
    const spec::Dict{String, Any}
    const sys::S          # the device system
    const host::H         # host copy of the state, for the tracker, the series and the frames
    const pot::P
    rng::Xoshiro
    st::Dict{String, Any}  # everything a checkpoint stores
end

function initial_host(spec)
    nrep = spec["nrep"]
    perfect = fcc_cell(FT(A0), nrep, :Cu)
    v = argmin(j -> norm(perfect.positions[:, j] - fill(nrep * A0 / 2, 3)), 1:natoms(perfect))
    host = remove_atom(perfect, v)
    maxwell_boltzmann!(host, spec["T"], Xoshiro(spec["seed"]))
    return host
end

new_state() = Dict{String, Any}(
    "equilibrated" => false, "prod_steps" => 0, "tracker" => nothing, "t_fs" => Float64[],
    "T_K" => Float64[], "P_GPa" => Float64[], "segments" => Dict{String, Any}[], "compute_s" => 0.0,
    "eq" => Dict{String, Any}(), "regrowths" => 0, "status" => "running", "message" => "",
)

# The tracker is stored in a checkpoint as a Dict of its fields, so a checkpoint stays readable when
# VacancyTracker gains fields: fields the Dict lacks keep the values of a new tracker, the time of
# the last frame defaults to that of the last stored frame, and the campaign's transient limit applies.
tracker_fields(tr::VacancyTracker) = Dict{Symbol, Any}(f => getfield(tr, f) for f in fieldnames(typeof(tr)))
tracker_fields(::Nothing) = nothing

function restore_tracker(d::AbstractDict, cell)
    tr = VacancyTracker{FT}(d[:sites]; cell, window = d[:window], dwell = d[:dwell], capacity = length(d[:t]), max_transient = MAX_TRANSIENT)
    for (k, v) in d
        k === :max_transient && continue
        hasfield(typeof(tr), k) || error("checkpoint: the stored tracker has a field $k that VacancyTracker does not")
        x = getfield(tr, k)
        x isa AbstractArray ? copyto!(x, v) : setfield!(tr, k, convert(fieldtype(typeof(tr), k), v))
    end
    haskey(d, :tlast) || (tr.tlast = tr.t[tr.nframes])
    return tr
end

function load_worker(dir)
    spec = readjson(joinpath(dir, "spec.json"))
    model = MACEModel{FT}(load_mace_checkpoint(mpa0_path()))
    host = initial_host(spec)
    ck = joinpath(dir, "checkpoint.jls")
    st, rng = new_state(), Xoshiro(spec["seed"] + 1)
    if isfile(ck)
        c = deserialize(ck)
        copyto!(host.positions, c["positions"]); copyto!(host.velocities, c["velocities"]); copyto!(host.cell, c["cell"])
        st, rng = c["state"], c["rng"]
        isnothing(st["tracker"]) || (st["tracker"] = restore_tracker(st["tracker"], host.cell))
        st["status"] = "running"
        logln("resumed at production step $(st["prod_steps"]) ($(st["equilibrated"] ? "equilibrated" : "not equilibrated"))")
    end
    backend = BACKEND_NAME == "Metal" ? MetalBackend() : BACKEND_NAME == "CUDA" ? CUDABackend() : ROCBackend()
    sys = PureAtomistic.to_device(host, backend)
    pot = PureAtomistic.device_potential(model, sys; skin = 0)
    return Worker(dir, spec, sys, host, pot, rng, st)
end

function checkpoint!(w::Worker)
    copy_state!(w.host, w.sys)
    c = Dict{String, Any}(
        "positions" => copy(w.host.positions), "velocities" => copy(w.host.velocities), "cell" => copy(w.host.cell),
        "rng" => copy(w.rng), "state" => merge(w.st, Dict{String, Any}("tracker" => tracker_fields(w.st["tracker"]))),
    )
    atomic_serialize(joinpath(w.dir, "checkpoint.jls"), c)
    return nothing
end

press(w::Worker) = Float64(pressure(w.host, Array(w.pot.result.stress))) * EV_PER_A3_IN_GPA

function equilibrate!(w::Worker)
    T, dt = w.spec["T"], w.spec["dt_fs"]
    every = round(Int, FRAME_FS / dt)
    t0 = time()
    npt = Simulation(
        w.sys, w.pot, VelocityVerlet(dt); thermostat = BussiThermostat(T, TAU_T_FS),
        barostat = BerendsenBarostat(P0_GPA, TAU_P_FS, BETA_T), rng = w.rng
    )
    tv, V, P, Tk = Float64[], Float64[], Float64[], Float64[]
    n_npt = round(Int, NPT_PS * 1000 / dt)
    run!(
        npt, n_npt; every, observer = s -> begin
            copy_state!(w.host, s.system)
            push!(tv, s.step_count * dt); push!(V, cell_volume(w.host.cell)); push!(P, press(w)); push!(Tk, temperature(w.host))
        end
    )
    last = findall(>(1000 * (NPT_PS - NPT_MEAN_PS) + 1.0e-6), tv)
    Vmean = mean(V[last])
    μ = cbrt(Vmean / cell_volume(w.host.cell))
    rescale!(w.sys, FT(μ))
    copy_state!(w.host, w.sys)
    n_nvt = round(Int, NVT_EQ_PS * 1000 / dt)
    nvt = Simulation(w.sys, w.pot, VelocityVerlet(dt); thermostat = BussiThermostat(T, TAU_T_FS), rng = w.rng)
    run!(nvt, n_nvt)
    nat = natoms(w.host)
    w.st["eq"] = Dict{String, Any}(
        "npt_steps" => n_npt, "nvt_steps" => n_nvt, "volume_mean_A3" => Vmean, "volume_se_A3" => block_se(V[last], 5),
        "volume_per_site_A3" => Vmean / (nat + 1), "lattice_constant_A" => cbrt(Vmean) / w.spec["nrep"],
        "npt_last_pressure_mean_GPa" => mean(P[last]), "npt_last_pressure_se_GPa" => block_se(P[last], 5),
        "npt_last_temperature_mean_K" => mean(Tk[last]), "scale_factor" => μ,
        "npt_t_fs" => tv[1:5:end], "npt_volume_A3" => V[1:5:end], "wall_s" => time() - t0,
    )
    w.st["equilibrated"] = true
    w.st["regrowths"] = w.pot.n_regrows
    copy_state!(w.host, w.sys)
    a = Float64(w.host.cell[1, 1]) / w.spec["nrep"]
    sites = reference_sites(w.host.cell, FT(a), w.spec["nrep"])
    tr = VacancyTracker(sites; cell = w.host.cell, window = WINDOW, dwell = DWELL, capacity = round(Int, CAPACITY_PS * 1000 / FRAME_FS) + 1, max_transient = MAX_TRANSIENT)
    update!(tr, w.host.positions, 0.0)
    w.st["tracker"] = tr
    push!(w.st["t_fs"], 0.0); push!(w.st["T_K"], temperature(w.host)); push!(w.st["P_GPa"], press(w))
    w.st["compute_s"] += time() - t0
    logln(@sprintf("equilibrated: V %.2f Å³ (a %.4f Å), NPT pressure %.3f GPa, %.0f s", Vmean, w.st["eq"]["lattice_constant_A"], w.st["eq"]["npt_last_pressure_mean_GPa"], time() - t0))
    checkpoint!(w)
    write_record(w)
    return nothing
end

function production_segment!(w::Worker, nsteps)
    T, dt = w.spec["T"], w.spec["dt_fs"]
    every = round(Int, FRAME_FS / dt)
    traj_every = round(Int, TRAJ_FS / dt)
    st = w.st
    tr = st["tracker"]
    sim = Simulation(w.sys, w.pot, VelocityVerlet(dt); thermostat = BussiThermostat(T, TAU_T_FS), rng = w.rng)
    sim.step_count = st["prod_steps"]
    k = length(st["segments"]) + 1
    tdir = mkpath(joinpath(w.dir, "trajectory"))
    traj = TrajectoryWriter(joinpath(tdir, @sprintf("segment_%04d.patraj", k)), natoms(w.host), ["Cu" for _ in 1:natoms(w.host)]; velocities = false)
    h0, r0, t0 = tr.nhops, w.pot.n_regrows, time()
    function observe(s)
        copy_state!(w.host, s.system)
        t = s.step_count * dt
        update!(tr, w.host.positions, t)
        push!(st["t_fs"], t); push!(st["T_K"], temperature(w.host)); push!(st["P_GPa"], press(w))
        s.step_count % traj_every == 0 && write_frame(traj, t, w.host.positions; cell = w.host.cell)
        if tr.nframes % 1000 == 0
            logln(@sprintf("t %.1f ps  hops %d  T %.0f K  P %.3f GPa", t / 1000, tr.nhops, st["T_K"][end], st["P_GPa"][end]))
        end
        return nothing
    end
    try
        run!(sim, nsteps; every, observer = observe)
    finally
        close(traj)
    end
    wall = time() - t0
    st["prod_steps"] = sim.step_count
    st["compute_s"] += wall
    st["regrowths"] = w.pot.n_regrows
    push!(
        st["segments"], Dict{String, Any}(
            "index" => k, "steps" => nsteps, "end_ps" => sim.step_count * dt / 1000, "hops" => tr.nhops - h0,
            "wall_s" => wall, "step_ms" => 1000 * wall / nsteps, "regrowths" => w.pot.n_regrows - r0, "date" => string(now()),
        )
    )
    checkpoint!(w)
    write_record(w)
    logln(@sprintf("segment %d done: %.1f ps, %d hops (%d in the segment), %.1f ms/step", k, sim.step_count * dt / 1000, tr.nhops, tr.nhops - h0, 1000 * wall / nsteps))
    return nothing
end

function advance_to!(w::Worker, target::Int)
    seg = round(Int, SEGMENT_PS * 1000 / w.spec["dt_fs"])
    target % seg == 0 || error("the target step $target is not a whole number of $seg-step segments")
    w.st["equilibrated"] || equilibrate!(w)
    while w.st["prod_steps"] < target
        production_segment!(w, seg)
    end
    return nothing
end

# The hops and sites of the run: the site of the vacancy after each hop.
function hop_sites(tr)
    s = Int[]
    for f in 2:tr.nframes
        tr.vsite[f] != tr.vsite[f - 1] && push!(s, tr.vsite[f])
    end
    return s
end

function write_record(w::Worker)
    st, spec = w.st, w.spec
    tr = st["tracker"]
    r = merge(
        provenance(), Dict{String, Any}(
            "description" => "One vacancy in Cu with MACE-MPA-0 (Float32, $BACKEND_NAME) at $(spec["T"]) K and zero pressure; one run of the vacancy-diffusion campaign",
            "run" => spec["name"], "stage" => spec["stage"], "temperature_K" => spec["T"], "dt_fs" => spec["dt_fs"], "seed" => spec["seed"],
            "cells" => spec["nrep"], "natoms" => 4 * spec["nrep"]^3 - 1, "potential" => "MACE-MPA-0 (Float32)", "skin" => 0,
            "settings" => Dict(
                "thermostat" => "Bussi, tau $TAU_T_FS fs", "barostat_equilibration" => "Berendsen, P0 $P0_GPA GPa, tau_P $TAU_P_FS fs, beta_T 1/140 per GPa",
                "npt_ps" => NPT_PS, "volume_from_last_ps" => NPT_MEAN_PS, "nvt_discarded_ps" => NVT_EQ_PS,
                "frame_interval_fs" => FRAME_FS, "trajectory_interval_fs" => TRAJ_FS, "segment_ps" => SEGMENT_PS,
                "tracker_window_frames" => WINDOW, "tracker_dwell_frames" => DWELL, "initial_velocities" => "Maxwell-Boltzmann at T, Xoshiro(seed); thermostat generator Xoshiro(seed + 1)",
            ),
            "status" => st["status"], "message" => st["message"], "equilibration" => st["eq"], "regrowths" => st["regrowths"],
            "production_steps" => st["prod_steps"], "duration_ps" => st["prod_steps"] * spec["dt_fs"] / 1000,
            "compute_seconds" => st["compute_s"], "segments" => st["segments"],
        )
    )
    if !isnothing(tr)
        r["frames"] = tr.nframes
        r["hops"] = tr.nhops
        r["hop_times_fs"] = tr.hop_t[1:tr.nhops]
        r["hop_sites"] = hop_sites(tr)
        r["initial_site"] = tr.vsite[1]
        r["transient_limit_frames"], r["transient_limit_fs"] = tr.max_transient, tr.max_transient * FRAME_FS
        r["transient_frames"], r["transient_longest_frames"] = tr.ntransient, tr.longest_transient
        r["transient_episodes"], r["transient_episode_start_fs"] = tr.nepisodes, tr.transient_t[1:tr.nepisodes]
        r["mean_temperature_K"], r["temperature_se_K"] = mean(st["T_K"]), block_se(st["T_K"])
        r["mean_pressure_GPa"], r["pressure_se_GPa"] = mean(st["P_GPa"]), block_se(st["P_GPa"])
        a = st["eq"]["lattice_constant_A"]
        if tr.nhops >= 1
            h = hop_rate(tr)
            Γ, σ = Float64(h.rate) * 1000, Float64(h.σ) * 1000  # per ps
            r["rate_per_ps"], r["rate_se_per_ps"] = Γ, σ
            # D_v = Γ λ²/6 with the jump length λ = a/√2.
            r["D_v_from_rate_A2_per_ps"], r["D_v_from_rate_se_A2_per_ps"] = Γ * a^2 / 12, σ * a^2 / 12
        end
        blen = tr.nframes ÷ 10
        if blen >= 3
            m = msd_diffusion(tr; nblocks = 10, maxlag = min(MAXLAG_FRAMES, blen - 1))
            r["D_msd_A2_per_ps"], r["D_msd_se_A2_per_ps"] = Float64(m.D) * 1000, Float64(m.σ) * 1000
            r["msd_max_lag_frames"] = min(MAXLAG_FRAMES, blen - 1)
        end
        c = [s["hops"] for s in st["segments"]]
        if length(c) >= 2 && sum(c) > 0
            cm = mean(c)
            r["segment_dispersion_chi2"], r["segment_dispersion_dof"] = sum(abs2, c .- cm) / cm, length(c) - 1
        end
        steps = sum((s["steps"] for s in st["segments"]); init = 0)
        steps > 0 && (r["mean_step_ms"] = 1000 * sum((s["wall_s"] for s in st["segments"]); init = 0.0) / steps)
    end
    save_json(joinpath(RESULTS, spec["name"] * ".json"), r)
    return nothing
end

function status_line(w::Worker)
    tr = w.st["tracker"]
    hops = isnothing(tr) ? 0 : tr.nhops
    return "ok $(w.st["prod_steps"]) $hops $(w.st["prod_steps"] * w.spec["dt_fs"] / 1000) $(w.st["compute_s"])"
end

function worker_main(dir)
    BLAS.set_num_threads(1)
    w = load_worker(dir)
    logln("worker $(w.spec["name"]) ready on $BACKEND_NAME")
    reply(status_line(w))
    while true
        line = readline(stdin)
        isempty(line) && eof(stdin) && return
        cmd = split(line)
        try
            if cmd[1] == "status"
                reply(status_line(w))
            elseif cmd[1] == "advance_to"
                advance_to!(w, parse(Int, cmd[2]))
                reply(status_line(w))
            elseif cmd[1] == "finish"
                w.st["status"] = length(cmd) >= 2 ? String(cmd[2]) : "completed"
                checkpoint!(w)
                write_record(w)
                reply("bye")
                return
            else
                error("unknown command $(cmd[1])")
            end
        catch err
            msg = sprint(showerror, err)
            logln("stopped: " * msg)
            w.st["status"], w.st["message"] = "stopped", msg
            isnothing(w.st["tracker"]) || write_record(w)
            reply("error " * replace(msg, '\n' => ' '))
            exit(1)
        end
    end
    return
end

# ---------------------------------------------------------------------------------------------
# Campaign: stages of two runs each.
run_name(T, dt, seed_index, nrep) = @sprintf("T%04d_dt%d_s%02d%s", round(Int, T), round(Int, dt), seed_index, nrep == 4 ? "" : "_n$(4 * nrep^3 - 1)")

function run_spec(stage, T, dt, seed_index; nrep = 4, max_steps = 0)
    return Dict{String, Any}(
        "name" => run_name(T, dt, seed_index, nrep), "stage" => stage, "T" => T, "dt_fs" => dt,
        "seed" => SEED_BASE + seed_index, "seed_index" => seed_index, "nrep" => nrep, "max_steps" => max_steps,
    )
end

mutable struct Handle
    spec::Dict{String, Any}
    proc::Base.Process
    log::IOStream
    steps::Int
    hops::Int
    ps::Float64
end

function read_reply(h::Handle)
    while true
        eof(h.proc) && error("worker $(h.spec["name"]) exited without a reply; see its log")
        line = readline(h.proc)
        startswith(line, "@@ ") || continue
        s = line[4:end]
        startswith(s, "error") && error("worker $(h.spec["name"]): $s")
        if startswith(s, "ok")
            f = split(s)
            h.steps, h.hops, h.ps = parse(Int, f[2]), parse(Int, f[3]), parse(Float64, f[4])
        end
        return s
    end
    return
end

function start_worker(root, spec)
    dir = mkpath(joinpath(root, spec["name"]))
    sp = joinpath(dir, "spec.json")
    if isfile(sp)
        old = readjson(sp)
        all(old[k] == spec[k] for k in ("name", "T", "dt_fs", "seed", "nrep")) || error("run $(spec["name"]) exists with another spec")
        old["max_steps"] = spec["max_steps"]
        spec = old
    end
    save_json(sp, spec)
    log = open(joinpath(root, "logs", spec["name"] * ".log"), "a")
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $HARNESS worker $dir`
    proc = open(pipeline(cmd; stderr = log), "r+")
    return Handle(spec, proc, log, 0, 0, 0.0)
end

function stop_worker(h::Handle, status = "completed")
    println(h.proc, "finish $status"); flush(h.proc)
    read_reply(h)
    wait(h.proc)
    close(h.log)
    return nothing
end

const LOCK_SCRIPT = get(ENV, "PUREATOMISTIC_CPULOCK", "")
lock_acquire() = isempty(LOCK_SCRIPT) ? "" : strip(read(`$LOCK_SCRIPT acquire`, String))
lock_release(tok) = isempty(LOCK_SCRIPT) || run(`$LOCK_SCRIPT release $tok`)

clog(s) = (println(stdout, Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "  ", s); flush(stdout))

state_path(root) = joinpath(root, "campaign_state.json")
load_state(root) = isfile(state_path(root)) ? readjson(state_path(root)) :
    Dict{String, Any}("stage" => "dt_check", "gpu_hours" => 0.0, "cycles" => Any[], "dt_fs" => nothing, "status" => "running")
save_state(root, s) = save_json(state_path(root), s)

# One cycle: both workers advance to their targets under one lock token.
function cycle!(root, state, hs, targets)
    state["gpu_hours"] + last_cycle_hours(state) > MAX_HOURS && return false
    tok = lock_acquire()
    t0 = time()
    try
        for (h, t) in zip(hs, targets)
            println(h.proc, "advance_to $t"); flush(h.proc)
        end
        foreach(read_reply, hs)
    finally
        lock_release(tok)
    end
    wall = time() - t0
    state["gpu_hours"] += wall / 3600
    push!(
        state["cycles"], Dict{String, Any}(
            "stage" => state["stage"], "runs" => [h.spec["name"] for h in hs], "targets" => targets, "wall_s" => wall,
            "date" => string(now()), "hops" => [h.hops for h in hs], "ps" => [h.ps for h in hs],
        )
    )
    save_state(root, state)
    clog(@sprintf("%s: %s, %.0f s, %.2f h in all", state["stage"], join(["$(h.spec["name"]) $(h.ps) ps $(h.hops) hops" for h in hs], "; "), wall, state["gpu_hours"]))
    return true
end

last_cycle_hours(state) = isempty(state["cycles"]) ? 0.0 : state["cycles"][end]["wall_s"] / 3600

# Next cycle target: CYCLE_STEPS more than the furthest of the pair, on the cycle grid, capped at
# each run's fixed length.
function next_targets(hs)
    base = maximum(h.steps for h in hs)
    t = (base ÷ CYCLE_STEPS + 1) * CYCLE_STEPS
    return [h.spec["max_steps"] > 0 ? min(t, h.spec["max_steps"]) : t for h in hs]
end

function halt(root, state, why)
    state["status"] = "halted: " * why
    save_state(root, state)
    clog("halted: " * why)
    return nothing
end

# Binomial tails and the exact (Clopper-Pearson) interval of the success probability.
function binomial_tail(k, n, q, upper)
    lf = cumsum(log.(1:n)); pushfirst!(lf, 0.0)
    return sum(exp(lf[n + 1] - lf[j + 1] - lf[n - j + 1] + j * log(q) + (n - j) * log1p(-q)) for j in (upper ? (k:n) : (0:k)))
end

function bisect(f, lo, hi)
    for _ in 1:200
        mid = (lo + hi) / 2
        f(mid) > 0 ? (hi = mid) : (lo = mid)
    end
    return (lo + hi) / 2
end

function clopper_pearson(k, n; α = 0.05)
    lo = k == 0 ? 0.0 : bisect(q -> binomial_tail(k, n, q, true) - α / 2, 0.0, 1.0)
    hi = k == n ? 1.0 : bisect(q -> α / 2 - binomial_tail(k, n, q, false), 0.0, 1.0)
    return lo, hi
end

function dt_check_rule()
    return Dict{String, Any}(
        "temperature_K" => DT_CHECK_T, "time_steps_fs" => [2.0, 4.0], "stop" => "at the end of the first cycle where the two runs hold at least $DT_CHECK_HOPS hops together, or after $DT_CHECK_MAX_CYCLES cycles",
        "stop_hops" => DT_CHECK_HOPS, "max_cycles" => DT_CHECK_MAX_CYCLES, "p_min" => DT_CHECK_P, "ratio_bounds" => collect(DT_CHECK_RATIO),
        "rule" => "use 4 fs if the exact conditional binomial test of the hop counts gives p >= $DT_CHECK_P and the exact 95 % (Clopper-Pearson) interval of the rate ratio Γ(4 fs)/Γ(2 fs) lies within [$(DT_CHECK_RATIO[1]), $(DT_CHECK_RATIO[2])]; otherwise 2 fs",
        "design" => "both runs advance the same number of steps per cycle, so the 4 fs run covers twice the simulated time; with equal rates a 4 fs share q = 2/3 of $DT_CHECK_HOPS hops gives a 95 % interval of the ratio of about ×/÷ $(round(exp(1.96 / sqrt(DT_CHECK_HOPS * 2 / 9)); digits = 3)), and equal rates pass the rule with probability 0.79 (2,000 simulated checks of the exact rule at 800 hops; 0.67 for a true ratio of 0.9)",
        "counts_toward_production" => "the run at the chosen time step is the first 1200 K production run: same protocol, independent seed, and a stop that depends only on the total count",
    )
end

function dt_decision(h2, h4)
    n2, n4 = h2.hops, h4.hops
    t2, t4 = h2.ps, h4.ps
    n = n2 + n4
    q = t4 / (t2 + t4)
    p = binomial_exact_p(n4, n, q)
    lo, hi = clopper_pearson(n4, n)
    ratio(x) = x / (1 - x) * t2 / t4
    ρ = (n4 / t4) / (n2 / t2)
    ci = (ratio(lo), hi == 1 ? Inf : ratio(hi))
    use4 = p >= DT_CHECK_P && ci[1] >= DT_CHECK_RATIO[1] && ci[2] <= DT_CHECK_RATIO[2]
    return Dict{String, Any}(
        "hops_2fs" => n2, "ps_2fs" => t2, "hops_4fs" => n4, "ps_4fs" => t4, "rate_2fs_per_ps" => n2 / t2, "rate_4fs_per_ps" => n4 / t4,
        "rate_ratio" => ρ, "ratio_ci95" => collect(ci), "binomial_p" => p, "binomial_p_min" => min(binomial_exact_p(0, n, q), binomial_exact_p(n, n, q)),
        "chosen_dt_fs" => use4 ? 4.0 : 2.0, "run_2fs" => h2.spec["name"], "run_4fs" => h4.spec["name"],
    )
end

function stage_dt_check!(root, state)
    rec = merge(provenance(), Dict{String, Any}("status" => "running", "rule" => dt_check_rule()))
    path = joinpath(RESULTS, "dt_check.json")
    isfile(path) && (rec = merge(readjson(path), Dict("status" => "running")))
    haskey(state, "dt_override_fs") && (rec["override"] = override_record(state))
    save_json(path, rec)
    hs = [start_worker(root, run_spec("dt_check", DT_CHECK_T, 2.0, 1)), start_worker(root, run_spec("dt_check", DT_CHECK_T, 4.0, 2))]
    foreach(read_reply, hs)
    while sum(h.hops for h in hs) < DT_CHECK_HOPS && count(c -> c["stage"] == "dt_check", state["cycles"]) < DT_CHECK_MAX_CYCLES
        cycle!(root, state, hs, next_targets(hs)) || (foreach(h -> stop_worker(h, "paused"), hs); halt(root, state, "the next cycle would pass $MAX_HOURS hours"); return false)
        write_summary(root, state)
    end
    foreach(stop_worker, hs)
    d = dt_decision(hs[1], hs[2])
    rec["status"] = "decided"
    rec["decision"] = d
    rec["decided"] = string(now())
    save_json(path, rec)
    dt = get(state, "dt_override_fs", d["chosen_dt_fs"])
    rec["dt_fs_used"] = dt
    save_json(path, rec)
    state["dt_fs"] = dt
    state["first_1200_run"] = dt == 2.0 ? d["run_2fs"] : d["run_4fs"]
    state["stage"] = "t1200_rest"
    save_state(root, state)
    clog("dt check: $(d["hops_2fs"]) hops in $(d["ps_2fs"]) ps at 2 fs, $(d["hops_4fs"]) in $(d["ps_4fs"]) ps at 4 fs; ratio $(round(d["rate_ratio"]; digits = 3)) $(round.(d["ratio_ci95"]; digits = 3)), p $(round(d["binomial_p"]; digits = 3)); rule $(d["chosen_dt_fs"]) fs, used $dt fs")
    return true
end

function stage_t1200_rest!(root, state)
    dt = state["dt_fs"]
    first = readjson(joinpath(RESULTS, state["first_1200_run"] * ".json"))
    Γ = first["hops"] / first["duration_ps"]
    seg = round(Int, SEGMENT_PS * 1000 / dt)
    L = max(1, ceil(Int, SIZE_CHECK_HOPS / Γ / SEGMENT_PS)) * seg
    hs = [start_worker(root, run_spec("t1200_rest", 1200.0, dt, 3; max_steps = L)), start_worker(root, run_spec("t1200_rest", 1200.0, dt, 4; nrep = 5, max_steps = L))]
    foreach(read_reply, hs)
    while any(h.steps < L for h in hs)
        cycle!(root, state, hs, next_targets(hs)) || (foreach(h -> stop_worker(h, "paused"), hs); halt(root, state, "the next cycle would pass $MAX_HOURS hours"); return false)
        write_summary(root, state)
    end
    foreach(stop_worker, hs)
    state["stage"] = "t1080"
    save_state(root, state)
    write_summary(root, state)
    est = reestimate(state)
    state["reestimate_after_1200K"] = est
    save_state(root, state)
    if est["projected_total_hours"]["0.7"] > MAX_HOURS
        halt(root, state, @sprintf("the re-estimated total at 0.70 eV is %.1f h, above %.1f h; set VACANCY_CAMPAIGN_MAX_HOURS after approval", est["projected_total_hours"]["0.7"], MAX_HOURS))
        return false
    end
    return true
end

function stage_pair!(root, state, T, target, seeds)
    dt = state["dt_fs"]
    stage = state["stage"]
    hs = [start_worker(root, run_spec(stage, T, dt, seeds[1])), start_worker(root, run_spec(stage, T, dt, seeds[2]))]
    foreach(read_reply, hs)
    while sum(h.hops for h in hs) < target
        cycle!(root, state, hs, next_targets(hs)) || (foreach(h -> stop_worker(h, "paused"), hs); halt(root, state, "the next cycle would pass $MAX_HOURS hours"); return false)
        write_summary(root, state)
    end
    foreach(stop_worker, hs)
    return true
end

# Hours still to spend at each assumed E_m, from the zero-pressure rate measured at 1200 K and the
# mean step time of the 255-atom 1200 K production runs at the chosen time step, each of which ran
# as one of a pair of processes.
function reestimate(state)
    s = readjson(joinpath(RESULTS, "summary.json"))
    t12 = s["temperatures"]["1200"]
    Γ12 = t12["rate_per_ps"]
    dt = state["dt_fs"]
    step_s = mean(s["mean_step_ms"][r] for r in t12["runs"]) / 1000
    eq_steps = round(Int, (NPT_PS + NVT_EQ_PS) * 1000 / dt)
    out = Dict{String, Any}("rate_1200K_per_ps" => Γ12, "dt_fs" => dt, "step_s_per_process_in_pair" => step_s, "spent_hours" => state["gpu_hours"])
    proj = Dict{String, Any}()
    for E in (0.6, 0.7, 0.9)
        h = 0.0
        per = Dict{String, Any}()
        for (T, target) in TARGETS[2:end]
            Γ = Γ12 * exp(-E / KB_EV_PER_K * (1 / T - 1 / 1200))
            ps = target / 2 / Γ
            steps_run = ceil(Int, ps * 1000 / dt / CYCLE_STEPS) * CYCLE_STEPS + eq_steps
            hrs = steps_run * step_s / 3600
            per[string(Int(T))] = Dict("rate_per_ps" => Γ, "ps_per_run" => ps, "steps_per_run" => steps_run, "pair_hours" => hrs)
            h += hrs
        end
        out["remaining_E$(E)"] = per
        proj[string(E)] = state["gpu_hours"] + h
    end
    out["projected_total_hours"] = proj
    return out
end

# Summary of the run records: per temperature the production runs at the chosen time step, their
# pooled rate, the between-run test and the dispersion over segments; the fit over temperatures.
function write_summary(root, state)
    mkpath(RESULTS)
    recs = Dict{String, Any}[]
    for f in readdir(RESULTS)
        (endswith(f, ".json") && startswith(f, "T")) || continue
        push!(recs, readjson(joinpath(RESULTS, f)))
    end
    dt = state["dt_fs"]
    temps = Dict{String, Any}()
    for (T, target) in TARGETS
        rs = filter(r -> r["temperature_K"] == T && r["cells"] == 4 && haskey(r, "hops") && (isnothing(dt) || r["dt_fs"] == dt), recs)
        isnothing(dt) && T == 1200 && filter!(r -> r["dt_fs"] == 2.0, rs)
        isempty(rs) && continue
        n = sum(r["hops"] for r in rs)
        t = sum(r["duration_ps"] for r in rs)
        e = Dict{String, Any}(
            "runs" => [r["run"] for r in rs], "dt_fs" => rs[1]["dt_fs"], "target_hops" => target, "hops" => n, "duration_ps" => t,
            "rate_per_ps" => n / t, "rate_se_per_ps" => sqrt(n) / t, "lattice_constant_A" => mean(r["equilibration"]["lattice_constant_A"] for r in rs),
            "mean_temperature_K" => [r["mean_temperature_K"] for r in rs], "mean_pressure_GPa" => [r["mean_pressure_GPa"] for r in rs],
        )
        a = e["lattice_constant_A"]
        e["D_v_from_rate_A2_per_ps"] = e["rate_per_ps"] * a^2 / 12
        e["D_msd_A2_per_ps"] = [get(r, "D_msd_A2_per_ps", nothing) for r in rs]
        if length(rs) == 2
            e["hop_test"] = hop_test(rs[1], rs[2])
        end
        c = reduce(vcat, [[s["hops"] for s in r["segments"] if s["steps"] * r["dt_fs"] == SEGMENT_PS * 1000] for r in rs])
        if length(c) >= 2 && sum(c) > 0
            cm = mean(c)
            e["segment_dispersion_chi2"], e["segment_dispersion_dof"] = sum(abs2, c .- cm) / cm, length(c) - 1
        end
        temps[string(Int(T))] = e
    end
    s = merge(provenance(), Dict{String, Any}("stage" => state["stage"], "dt_fs" => dt, "gpu_hours" => state["gpu_hours"], "temperatures" => temps))
    s["starts"] = get(state, "starts", Any[])
    haskey(state, "dt_override_fs") && (s["dt_override"] = override_record(state))
    sz = filter(r -> r["cells"] == 5 && haskey(r, "hops"), recs)
    isempty(sz) || (s["size_check_499"] = Dict{String, Any}("run" => sz[1]["run"], "hops" => sz[1]["hops"], "duration_ps" => sz[1]["duration_ps"], "rate_per_ps" => sz[1]["hops"] / sz[1]["duration_ps"]))
    if !isempty(sz) && haskey(temps, "1200")
        s["size_check_499"]["hop_test_against_255"] = hop_test(sz[1], Dict("hops" => temps["1200"]["hops"], "duration_ps" => temps["1200"]["duration_ps"]))
    end
    # The fit uses the temperatures with at least one hop, which have a Poisson error.
    ks = sort!([k for k in keys(temps) if temps[k]["hops"] > 0])
    if length(ks) >= 2
        f = arrhenius_fit([parse(Float64, k) for k in ks], [temps[k]["rate_per_ps"] for k in ks], [temps[k]["rate_se_per_ps"] for k in ks])
        s["fit"] = Dict{String, Any}(string(k) => v for (k, v) in pairs(f))
        s["fit"]["temperatures_K"] = ks
        s["fit"]["dof"] = length(ks) - 2
    end
    steps_ms = [(r["run"], r["mean_step_ms"]) for r in recs if haskey(r, "mean_step_ms")]
    s["mean_step_ms"] = Dict(steps_ms)
    save_json(joinpath(RESULTS, "summary.json"), s)
    return nothing
end

function apply_dt_override!(state)
    isempty(DT_OVERRIDE) && return state
    dt = parse(Float64, DT_OVERRIDE)
    dt in (2.0, 4.0) || error("VACANCY_CAMPAIGN_DT_FS must be 2 or 4, got $DT_OVERRIDE")
    old = get(state, "dt_override_fs", nothing)
    isnothing(old) || old == dt || error("VACANCY_CAMPAIGN_DT_FS=$DT_OVERRIDE conflicts with the override of $old fs stored in the campaign state")
    isnothing(state["dt_fs"]) || state["dt_fs"] == dt || error("VACANCY_CAMPAIGN_DT_FS=$DT_OVERRIDE conflicts with the time step $(state["dt_fs"]) fs the campaign already uses")
    if isnothing(old)
        state["dt_override_fs"] = dt
        state["dt_override_note"] = "overridden by user: $(round(Int, dt)) fs"
        state["dt_override_reason"] = get(ENV, "VACANCY_CAMPAIGN_DT_REASON", "")
        state["dt_override_date"] = string(now())
    end
    return state
end

override_record(state) = Dict{String, Any}(
    "dt_fs" => state["dt_override_fs"], "note" => state["dt_override_note"], "reason" => state["dt_override_reason"],
    "date" => state["dt_override_date"], "effect" => "the rule's result is recorded under decision; the production runs use dt_fs",
)

function campaign_main()
    root = campaign_dir()
    mkpath(joinpath(root, "logs"))
    state = load_state(root)
    apply_dt_override!(state)
    state["status"] = "running"
    push!(get!(state, "starts", Any[]), merge(provenance(), Dict{String, Any}("dt_override_fs" => get(state, "dt_override_fs", nothing))))
    save_state(root, state)
    clog("campaign at stage $(state["stage"]), $(round(state["gpu_hours"]; digits = 2)) h spent, limit $MAX_HOURS h")
    state["stage"] == "dt_check" && (stage_dt_check!(root, state) || return)
    state["stage"] == "t1200_rest" && (stage_t1200_rest!(root, state) || return)
    for (stage, T, seeds, next) in (("t1080", 1080.0, (5, 6), "t980"), ("t980", 980.0, (7, 8), "t900"), ("t900", 900.0, (9, 10), "done"))
        state["stage"] == stage || continue
        stage_pair!(root, state, T, Dict(TARGETS)[T], seeds) || return
        state["stage"] = next
        save_state(root, state)
        write_summary(root, state)
    end
    state["status"] = "completed"
    save_state(root, state)
    clog("campaign completed, $(round(state["gpu_hours"]; digits = 1)) h")
    return nothing
end

if MODE == "worker"
    worker_main(ARGS[2])
elseif MODE == "campaign"
    campaign_main()
else
    write_summary(campaign_dir(), load_state(campaign_dir()))
end
