# Validation runs of the MD driver on 500 Cu atoms (a 5 × 5 × 5 fcc cell at the model's lattice
# constant) with MACE-MPA-0, one Julia thread, BLAS on one thread.
#
#   PUREMLIP_CHECKPOINT_DIR=<dir> julia -t 1 --project=PureAtomistic/bench PureAtomistic/bench/md_validation.jl \
#       nve <dt_fs>
#       nvt <dt_fs> <tau_fs>
#       vacancy <dt_fs> <wall_seconds>
#       vacancy <dt_fs> ps <simulated_ps>
#       verdicts      (device runs: adds the verdicts against the CPU Float32 records)
#
# PUREATOMISTIC_PRECISION=32|64 (default 64) sets the arithmetic of the system, the model and the
# potential; the results go to md_<mode>_f32.json for 32. The pass criteria are in
# validation_criteria.jl.
#
# PUREATOMISTIC_DEVICE=Metal|CUDA|AMDGPU runs the dynamics through the device MD path
# (`to_device`, `device_potential`, skin 0) in Float32, with the project bench/gpu/<device in lower
# case>; the results go to md_<mode>_f32_<device in lower case>.json together with the verdicts
# against the Float32 records on the CPU. The observers read the temperature and the energies
# as scalars every sample and copy the state to a host system only at the frame interval.
#
# PUREATOMISTIC_BLAS=pureblas loads and activates PureBLAS before anything else (Julia 1.13; PureBLAS
# must be on the load path). PUREATOMISTIC_TRAJ_DIR is the directory the trajectories are written
# to (default: a directory in the system temporary directory); they are not stored in the repository.
#
# nve      2 ps of constant-energy dynamics from Maxwell-Boltzmann velocities at 300 K (seeded).
#          The total energy per atom is sampled every 8 fs (every step when dt > 8 fs). Stable means
#          all of: the run completes with finite energies; the drift, the slope of the least-squares
#          line through the samples, is at most STABLE_DRIFT; the RMS residual of the samples about
#          that line is at most STABLE_RMS. The criterion is fixed here, before any run.
# nvt      Bussi thermostat at 1200 K for 2 ps; the instantaneous kinetic temperature of every step
#          is stored. The second half is compared with the canonical distribution of the kinetic
#          energy: Gamma with shape (3N - 3)/2 and scale k_B T, whose temperature estimate
#          2K/((3N - 3) k_B) has mean T and variance 2T²/(3N - 3).
# vacancy  One vacancy (4 × 4 × 4 cells, 255 atoms) at 1300 K, Bussi thermostat, run for the
#          wall-clock budget; a VacancyTracker follows the vacancy.
#
# Each mode writes into PureAtomistic/bench/results/md_<mode>.json (nve appends one entry per dt).
using JSON, Statistics, Dates, Printf, Serialization
include(joinpath(@__DIR__, "cells.jl"))
include(joinpath(@__DIR__, "autocorrelation.jl"))
include(joinpath(@__DIR__, "validation_criteria.jl"))
using PureAtomistic.Systems: maxwell_boltzmann!, kinetic_energy, temperature, KB_EV_PER_K
using PureAtomistic.Analysis: VacancyTracker, reference_sites, update!
using PureAtomistic.IO: TrajectoryWriter, write_frame

const BLAS_NAME = get(ENV, "PUREATOMISTIC_BLAS", "openblas")
if BLAS_NAME == "pureblas"
    pb = Base.require(Main, :PureBLAS)
    Base.invokelatest(pb.activate)
    Base.invokelatest(pb.is_active) || error("PureBLAS did not become active")
elseif BLAS_NAME != "openblas"
    error("PUREATOMISTIC_BLAS must be openblas or pureblas")
end
BLAS.set_num_threads(1)

const PRECISION = get(ENV, "PUREATOMISTIC_PRECISION", "64")
PRECISION in ("32", "64") || error("PUREATOMISTIC_PRECISION must be 32 or 64")
const FT = PRECISION == "32" ? Float32 : Float64
const DEVICE = get(ENV, "PUREATOMISTIC_DEVICE", "")
isempty(DEVICE) || DEVICE in ("Metal", "CUDA", "AMDGPU") || error("PUREATOMISTIC_DEVICE must be Metal, CUDA or AMDGPU")
isempty(DEVICE) || PRECISION == "32" || error("PUREATOMISTIC_DEVICE needs PUREATOMISTIC_PRECISION=32")
if !isempty(DEVICE)
    @eval using KernelAbstractions
    @eval using $(Symbol(DEVICE))
end
const BACKEND = isempty(DEVICE) ? nothing : DEVICE == "Metal" ? MetalBackend() : DEVICE == "CUDA" ? CUDABackend() : ROCBackend()
const SUFFIX = (PRECISION == "32" ? "_f32" : "") * (isempty(DEVICE) ? "" : "_" * lowercase(DEVICE))

const RESULTS = joinpath(@__DIR__, "results")
const TRAJ_DIR = get(ENV, "PUREATOMISTIC_TRAJ_DIR", joinpath(tempdir(), "pureatomistic-md"))
mkpath(TRAJ_DIR)

git_commit() = strip(read(setenv(`git rev-parse --short HEAD`; dir = @__DIR__), String))

function settings(extra)
    return merge(
        Dict{String, Any}(
            "commit" => git_commit(), "date" => string(Dates.today()), "julia" => string(VERSION),
            "blas" => isempty(DEVICE) ? BLAS_NAME : "not used (device)", "blas_threads" => BLAS.get_num_threads(), "julia_threads" => Threads.nthreads(),
            "device" => isempty(DEVICE) ? "CPU" : DEVICE, "noise" => isempty(DEVICE) ? "Xoshiro" : "Philox4x32-10, one key per step from Xoshiro",
            "potential" => "MACE-MPA-0 ($FT)", "precision" => PRECISION, "lattice_constant" => FT(A0), "element" => "Cu",
        ), extra,
    )
end

function save(update, name)
    path = joinpath(RESULTS, name)
    d = isfile(path) ? JSON.parsefile(path) : Dict{String, Any}()
    update(d)
    open(io -> JSON.print(io, d, 1), path, "w")
    return path
end

# Least-squares line through (t, y): slope, its standard error, and the RMS residual.
function linfit(t, y)
    n = length(t)
    tm, ym = mean(t), mean(y)
    stt = sum(abs2, t .- tm)
    slope = sum((t .- tm) .* (y .- ym)) / stt
    res = y .- ym .- slope .* (t .- tm)
    return (; slope, stderr = sqrt(sum(abs2, res) / (n - 2) / stt), rms = sqrt(mean(abs2, res)))
end

function load_model()
    return MACEModel{FT}(load_mace_checkpoint(mpa0_path()))
end

function cu_cell(n, seed, temp)
    sys = fcc_cell(FT(A0), n, :Cu)
    maxwell_boltzmann!(sys, temp, Xoshiro(seed))
    return sys
end

# The system a `Simulation` integrates: `host` itself, or its copy on the device.
on_device(host) = isempty(DEVICE) ? host : PureAtomistic.to_device(host, BACKEND)

# The potential of `sys`, which is `host` or its device copy: skin 0 on the device, so the list is
# rebuilt every step and a step reads back only the status word.
make_potential(model, host, sys) = isempty(DEVICE) ? mace(model, host) : PureAtomistic.device_potential(model, sys; skin = 0)

# Copies the state of the simulation's system into `host` (a no-op on the CPU, where they are the
# same system) and returns `host`.
pull!(host, sim) = isempty(DEVICE) ? host : copy_state!(host, sim.system)

# The potential energy of the last evaluation in eV (one scalar copy on the device).
potential_energy(sim) = isempty(DEVICE) ? sim.potential.result.energy : Float64(only(Array(sim.potential.result.energy)))

# Regrowths of the device neighbor list over the run; 0 on the CPU, where the list is the
# `MACEPotential`'s own.
regrowths(pot) = isempty(DEVICE) ? 0 : pot.n_regrows

# ---------------------------------------------------------------------------------------------
function run_nve(dt)
    n_steps = round(Int, 2000 / dt)
    every = max(1, round(Int, 8 / dt))
    seed = 20261106
    model = load_model()
    host = cu_cell(5, seed, 300.0)
    sys = on_device(host)
    N = natoms(sys)
    pot = make_potential(model, host, sys)
    sim = Simulation(sys, pot, VelocityVerlet(dt); rng = Xoshiro(seed + 1))
    t, E, K, U = Float64[], Float64[], Float64[], Float64[]
    sample!(s) = (
        push!(t, s.step_count * dt); push!(K, kinetic_energy(s) / N);
        push!(U, potential_energy(s) / N); push!(E, K[end] + U[end]); nothing
    )
    traj = dt == 2 ? TrajectoryWriter(joinpath(TRAJ_DIR, "nve_dt2.patraj"), N, ["Cu" for _ in 1:N]; velocities = false) : nothing
    frame!(s) = isnothing(traj) || (pull!(host, s); write_frame(traj, s.step_count * dt, host.positions; cell = host.cell, energy = potential_energy(s)))
    sample!(sim); frame!(sim)
    t_wall = time()
    status, message = "completed", ""
    try
        run!(sim, n_steps; every, observer = s -> (sample!(s); frame!(s); isfinite(E[end]) || error("non-finite energy at step $(s.step_count)")))
    catch err
        status, message = "failed", sprint(showerror, err)
    end
    wall = time() - t_wall
    isnothing(traj) || close(traj)
    entry = Dict{String, Any}("dt_fs" => dt, "status" => status, "message" => message, "steps_done" => sim.step_count, "wall_seconds" => wall, "regrowths" => regrowths(pot))
    if length(t) >= 3 && all(isfinite, E)
        f = linfit(t ./ 1000, E)  # per ps
        entry["drift_eV_per_atom_per_ps"] = f.slope
        entry["drift_stderr"] = f.stderr
        entry["rms_residual_eV_per_atom"] = f.rms
        entry["max_abs_deviation_eV_per_atom"] = maximum(abs, E .- E[1])
        entry["mean_temperature_K"] = mean(2 .* K .* N ./ ((3N - 3) * KB_EV_PER_K))
    end
    entry["t_fs"], entry["total_eV_per_atom"], entry["kinetic_eV_per_atom"], entry["potential_eV_per_atom"] =
        t, E, K, U
    entry["stable"] = criteria(entry)["pass"]
    @printf("nve dt %g fs: %s, %d steps, %.0f s, drift %s, rms %s, stable %s\n", dt, status, sim.step_count, wall, get(entry, "drift_eV_per_atom_per_ps", NaN), get(entry, "rms_residual_eV_per_atom", NaN), entry["stable"])
    save("md_nve_conservation$SUFFIX.json") do d
        d["description"] = "Constant-energy MD of 500 Cu atoms with MACE-MPA-0 ($FT, $(isempty(DEVICE) ? "CPU" : DEVICE)) for 2 ps from 300 K velocities, one entry per time step"
        d["settings"] = settings(
            Dict(
                "natoms" => N, "initial_temperature_K" => 300.0, "duration_fs" => 2000.0, "sample_interval_fs" => dt > 8 ? dt : 8.0,
                "seed" => seed, "stable_if" => "completes with finite energies, |drift| <= $STABLE_DRIFT eV/atom/ps and RMS residual about the line <= $STABLE_RMS eV/atom",
                "stable_drift_eV_per_atom_per_ps" => STABLE_DRIFT, "stable_rms_eV_per_atom" => STABLE_RMS,
            )
        )
        runs = get!(d, "runs", Any[])
        filter!(r -> r["dt_fs"] != dt, runs)
        push!(runs, entry)
        sort!(runs; by = r -> r["dt_fs"])
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# Standard error of the mean of `x` from non-overlapping blocks of `bs` samples.
function block_sem(x, bs)
    nb = length(x) ÷ bs
    nb >= 2 || return NaN
    m = [mean(@view x[((b - 1) * bs + 1):(b * bs)]) for b in 1:nb]
    return std(m) / sqrt(nb)
end

function run_nvt(dt, tau)
    n_steps = round(Int, 2000 / dt)
    Tset = 1200.0
    seed = 20261107
    model = load_model()
    host = cu_cell(5, seed, Tset)
    sys = on_device(host)
    N = natoms(sys)
    pot = make_potential(model, host, sys)
    th = BussiThermostat(Tset, tau)
    sim = Simulation(sys, pot, VelocityVerlet(dt); thermostat = th, rng = Xoshiro(seed + 1))
    t, T, U = Float64[], Float64[], Float64[]
    traj = TrajectoryWriter(joinpath(TRAJ_DIR, "nvt.patraj"), N, ["Cu" for _ in 1:N]; velocities = false)
    every_frame = max(1, round(Int, 20 / dt))
    t_wall = time()
    run!(
        sim, n_steps; observer = s -> begin
            push!(t, s.step_count * dt); push!(T, temperature(s)); push!(U, potential_energy(s) / N)
            s.step_count % every_frame == 0 && (pull!(host, s); write_frame(traj, s.step_count * dt, host.positions; cell = host.cell, energy = potential_energy(s)))
            isfinite(T[end]) || error("non-finite temperature at step $(s.step_count)")
        end
    )
    wall = time() - t_wall
    close(traj)
    dof = 3N - 3
    half = findall(>(1000.0), t)
    x = T[half]
    nb = 5
    bs = length(x) ÷ nb
    mean_blocks = [mean(x[((b - 1) * bs + 1):(b * bs)]) for b in 1:nb]
    # Delete-one-block jackknife of the variance, with the overall mean as the center.
    xs = x[1:(nb * bs)]
    var_all = var(xs)
    var_jack = [var(vcat((xs[((c - 1) * bs + 1):(c * bs)] for c in 1:nb if c != b)...)) for b in 1:nb]
    var_err = sqrt((nb - 1) / nb * sum(abs2, var_jack .- mean(var_jack)))
    # The 5-block error underestimates the true one when the blocks are not longer than the
    # correlation time; "autocorrelation" below is the reference error.
    mean_err = std(mean_blocks) / sqrt(nb)
    theory_var = 2 * Tset^2 / dof
    @printf("nvt: %d steps in %.0f s; mean %.2f ± %.2f K (theory %.1f); var %.0f ± %.0f K² (theory %.0f)\n", n_steps, wall, mean(xs), mean_err, Tset, var_all, var_err, theory_var)
    save("md_thermostat_check$SUFFIX.json") do d
        d["description"] = "Bussi thermostat at 1200 K on 500 Cu atoms with MACE-MPA-0 ($FT, $(isempty(DEVICE) ? "CPU" : DEVICE)) for 2 ps; kinetic-temperature distribution of the second half against the canonical one"
        d["settings"] = settings(
            Dict(
                "natoms" => N, "target_temperature_K" => Tset, "dt_fs" => dt, "tau_fs" => tau, "duration_fs" => 2000.0, "seed" => seed,
                "initial_temperature_K" => Tset, "dof" => dof, "analysis_window_fs" => [1000.0, 2000.0], "blocks" => nb,
                "block_samples" => bs, "wall_seconds" => wall, "regrowths" => regrowths(pot),
            )
        )
        d["theory"] = Dict(
            "distribution" => "kinetic energy ~ Gamma(shape = dof/2, scale = k_B T); temperature estimate 2K/(dof k_B)",
            "mean_K" => Tset, "variance_K2" => theory_var, "std_K" => sqrt(theory_var),
            "skewness" => 2 / sqrt(dof / 2),
        )
        d["observed"] = Dict(
            "n_samples" => length(xs), "mean_K" => mean(xs), "mean_error_K" => mean_err, "variance_K2" => var_all, "variance_error_K2" => var_err,
            "std_K" => sqrt(var_all), "mean_over_blocks" => mean_blocks,
            "skewness" => mean((xs .- mean(xs)) .^ 3) / var_all^1.5,
            "mean_z" => (mean(xs) - Tset) / mean_err, "variance_z" => (var_all - theory_var) / var_err,
            "autocorrelation" => autocorrelation_record(xs, Tset, mean((xs .- mean(xs)) .^ 3) / var_all^1.5, 2 / sqrt(dof / 2)),
            "blocking_sem_K" => [Dict("block_samples" => b, "sem_K" => block_sem(xs, b)) for b in (1, 2, 5, 10, 25, 50, 100)],
        )
        d["t_fs"], d["temperature_K"], d["potential_eV_per_atom"] = t, T, U
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# A MACEPotential whose evaluation splits the atoms into `nchunks` receiver chunks that run on
# Julia threads. The results depend on `nchunks` through rounding, not on the number of threads.
mutable struct ThreadedMACE{P <: MACEPotential, R} <: PP.AbstractPotential
    const inner::P
    const result::R
    const nchunks::Int
    regrows::Int
end

function chunked_workspace!(inner::MACEPotential{T}, nchunks) where {T}
    inner.ws = MACEWorkspace{T}(inner.model, PP.n_atoms(inner), length(inner.nl.senders); elements = inner.atomic_numbers, nchunks)
    return nothing
end

function ThreadedMACE(model, sys, nchunks)
    inner = mace(model, sys)
    chunked_workspace!(inner, nchunks)
    return ThreadedMACE(inner, inner.result, nchunks, inner.n_regrows)
end
PP.n_atoms(p::ThreadedMACE) = PP.n_atoms(p.inner)
PP.cutoff(p::ThreadedMACE) = PP.cutoff(p.inner)
function PP.prepare!(p::ThreadedMACE, x::AbstractMatrix, c::AbstractMatrix)
    PP.prepare!(p.inner, x, c)
    # A regrow rebuilds the single-chunk workspace; restore the chunked one.
    if p.inner.n_regrows != p.regrows
        chunked_workspace!(p.inner, p.nchunks)
        p.regrows = p.inner.n_regrows
    end
    return nothing
end
function PP.energy_forces!(p::ThreadedMACE, x::AbstractMatrix, c::AbstractMatrix)
    i = p.inner
    mace_energy_forces!(i.result, i.ws, i.model, x, c, i.atomic_numbers, i.nl, ThreadedChunks())
    return nothing
end

# The shared CPU lock, released and taken again between segments of a long run so that other jobs
# can interleave. PUREATOMISTIC_CPULOCK is the path of the lock script.
const LOCK_SCRIPT = get(ENV, "PUREATOMISTIC_CPULOCK", "")
lock_acquire() = isempty(LOCK_SCRIPT) ? "" : strip(read(`$LOCK_SCRIPT acquire`, String))
lock_release(tok) = isempty(LOCK_SCRIPT) || run(`$LOCK_SCRIPT release $tok`)

# `budget` is a wall-clock limit in seconds, or `(:ps, length)` for a fixed simulated length.
function run_vacancy(dt, budget)
    Tset = 1300.0
    seed = 20261108
    nrep = 4
    frame_fs = 20.0
    every = round(Int, frame_fs / dt)
    model = load_model()
    perfect = fcc_cell(FT(A0), nrep, :Cu)
    v = argmin(j -> norm(perfect.positions[:, j] - fill(nrep * A0 / 2, 3)), 1:natoms(perfect))
    host = remove_atom(perfect, v)
    maxwell_boltzmann!(host, Tset, Xoshiro(seed))
    sys = on_device(host)
    N = natoms(sys)
    sites = reference_sites(host.cell, FT(A0), nrep)
    window, dwell = 5, 3
    tracker = VacancyTracker(sites; cell = host.cell, window, dwell, capacity = 200_000)
    nchunks = parse(Int, get(ENV, "PUREATOMISTIC_NCHUNKS", "1"))
    isempty(DEVICE) || nchunks == 1 || error("PUREATOMISTIC_NCHUNKS applies to the CPU path only")
    pot = nchunks == 1 ? make_potential(model, host, sys) : ThreadedMACE(model, sys, nchunks)
    sim = Simulation(sys, pot, VelocityVerlet(dt); thermostat = BussiThermostat(Tset, 100.0), rng = Xoshiro(seed + 1))
    traj = TrajectoryWriter(joinpath(TRAJ_DIR, "vacancy.patraj"), N, ["Cu" for _ in 1:N]; velocities = false)
    Tseries, tseries = Float64[], Float64[]
    status, message = "completed", ""
    segment_ps = parse(Float64, get(ENV, "PUREATOMISTIC_SEGMENT_PS", "0"))
    segments = 0
    waited = 0.0
    compute = 0.0  # seconds of simulation, without the waits for the lock
    resumed_from = 0
    # State at the end of a segment: everything a continuation needs besides the potential, which is
    # evaluated again at the stored positions.
    ckpt = joinpath(TRAJ_DIR, "vacancy$(SUFFIX)_checkpoint.jls")
    if get(ENV, "PUREATOMISTIC_RESUME", "0") == "1"
        c = Serialization.deserialize(ckpt)
        copyto!(sys.positions, c.positions); copyto!(sys.velocities, c.velocities)
        sim.step_count = c.step_count; copy!(sim.rng, c.rng)
        tracker, Tseries, tseries = c.tracker, c.Tseries, c.tseries
        segments, waited, compute, resumed_from = c.segments, c.waited, c.compute, c.step_count
        PP.prepare!(pot, sys.positions, sys.cell); PP.energy_forces!(pot, sys.positions, sys.cell)
    end
    token = lock_acquire()
    t_wall = time()
    t_seg = time()
    function observe(s)
        tt = s.step_count * dt
        pull!(host, s)
        update!(tracker, host.positions, tt)
        write_frame(traj, tt, host.positions; cell = host.cell, energy = potential_energy(s))
        push!(tseries, tt); push!(Tseries, temperature(s))
        if tracker.nframes % 100 == 0
            @printf("  t %.1f ps  frames %d  hops %d  T %.0f K  wall %.0f s\n", tt / 1000, tracker.nframes, tracker.nhops, Tseries[end], compute + time() - t_seg)
            flush(stdout)
        end
        return nothing
    end
    try
        if budget isa Tuple
            total = round(Int, budget[2] * 1000 / dt)
            total % every == 0 || error("the simulated length must be a multiple of the frame interval $frame_fs fs")
            seg_steps = segment_ps > 0 ? round(Int, segment_ps * 1000 / dt) : total
            seg_steps % (100 * every) == 0 || error("the segment length must be a multiple of $(100 * every) steps")
            while sim.step_count < total
                segments += 1
                stop = min(total, segments * seg_steps)
                while sim.step_count < stop
                    run!(sim, min(100 * every, stop - sim.step_count); every, observer = observe)
                end
                compute += time() - t_seg
                state = (; positions = Array(sys.positions), velocities = Array(sys.velocities), step_count = sim.step_count, rng = copy(sim.rng), tracker, Tseries, tseries, segments, waited, compute)
                Serialization.serialize(ckpt * ".tmp", state); mv(ckpt * ".tmp", ckpt; force = true)
                @printf("segment %d done: %.1f ps, %d hops, %.0f s of simulation\n", segments, sim.step_count * dt / 1000, tracker.nhops, compute)
                flush(stdout)
                if sim.step_count < total
                    lock_release(token)
                    t0 = time()
                    token = lock_acquire()
                    waited += time() - t0
                    t_seg = time()
                end
            end
        else
            while time() - t_wall < budget
                run!(sim, 100 * every; every, observer = observe)
            end
        end
    catch err
        status, message = "stopped", sprint(showerror, err)
    end
    status == "completed" || (compute += time() - t_seg)
    wall = budget isa Tuple ? compute : time() - t_wall - waited
    lock_release(token)
    close(traj)
    n = tracker.nframes
    @printf("vacancy: %s %s; %d steps (%.1f ps) in %.0f s; %d hops\n", status, message, sim.step_count, sim.step_count * dt / 1000, wall, tracker.nhops)
    save("md_vacancy_hop$SUFFIX.json") do d
        d["description"] = "One vacancy in a 255-atom Cu cell with MACE-MPA-0 ($FT, $(isempty(DEVICE) ? "CPU" : DEVICE)) at 1300 K (constant volume, Bussi); an illustration of the tracker, not a diffusion measurement"
        d["settings"] = settings(
            Dict(
                "natoms" => N, "cells" => nrep, "temperature_K" => Tset, "dt_fs" => dt, "tau_fs" => 100.0, "seed" => seed,
                "wall_budget_seconds" => budget isa Tuple ? nothing : budget, "simulated_ps_target" => budget isa Tuple ? budget[2] : nothing, "wall_seconds" => wall, "nchunks" => nchunks, "regrowths" => regrowths(pot), "segments" => segments, "segment_ps" => segment_ps,
                "lock_released_between_segments" => segments > 1 && !isempty(LOCK_SCRIPT), "lock_wait_seconds" => waited,
                "checkpoint_each_segment" => segment_ps > 0, "resumed_from_step" => resumed_from,
                "state_across_segments" => "positions, velocities, generator, tracker and series are stored at the end of each segment and continue exactly; a resumed run starts from the last stored segment",
                "frame_interval_fs" => frame_fs,
                "tracker_window_frames" => window, "tracker_dwell_frames" => dwell, "initial_velocities_K" => Tset,
                "volume" => "fixed at the lattice constant of the zero-temperature crystal",
            )
        )
        d["status"] = status
        d["message"] = message
        d["steps"] = sim.step_count
        d["duration_ps"] = sim.step_count * dt / 1000
        d["frames"] = n
        d["hops"] = tracker.nhops
        d["hop_times_fs"] = tracker.hop_t[1:tracker.nhops]
        d["t_fs"] = tracker.t[1:n]
        d["vacancy_R"] = [tracker.R[:, k] for k in 1:n]
        d["vacancy_site"] = tracker.vsite[1:n]
        d["initial_site_position"] = sites[:, tracker.vsite[1]]
        d["temperature_K"] = Tseries
        d["mean_temperature_K"] = mean(Tseries)
    end
    return nothing
end

mode = ARGS[1]
if mode == "nve"
    run_nve(parse(Float64, ARGS[2]))
elseif mode == "nvt"
    run_nvt(parse(Float64, ARGS[2]), parse(Float64, ARGS[3]))
elseif mode == "verdicts"
    isempty(DEVICE) && error("verdicts needs PUREATOMISTIC_DEVICE")
    write_device_verdicts(RESULTS, SUFFIX)
elseif mode == "vacancy"
    ARGS[3] == "ps" ? run_vacancy(parse(Float64, ARGS[2]), (:ps, parse(Float64, ARGS[4]))) :
        run_vacancy(parse(Float64, ARGS[2]), parse(Float64, ARGS[3]))
else
    error("mode must be nve, nvt or vacancy")
end
