# Atomic diffusion coefficient of a vacancy campaign run from its stored trajectory, compared with
# the prediction from the run's vacancy hop rate.
#
#   julia --project=PureAtomistic/bench PureAtomistic/bench/trajectory_diffusion.jl \
#       <run directory with trajectory/segment_*.patraj> <run record .json> <output .json>
#
# The MSD of all atoms is taken over the concatenated segments with one block per 100 ps
# (`nblocks` = number of segments), lags up to 50 ps, and the Einstein fit over lags of 10 to
# 50 ps: long against the mean time between hops (about 5 ps at 1200 K) and against the
# vibrational relaxation, and at most half a block. The settings do not depend on the data.
#
# Prediction. Each vacancy jump moves one atom by a nearest-neighbor vector of length
# λ = a/√2. After n jumps the summed squared displacement of the N atoms is
# Σᵢ |Δrᵢ|² = Σᵢ |Σⱼ λᵢⱼ|² = n λ² + (cross terms of successive jumps of one atom). For
# uncorrelated jumps the cross terms average to zero, and with n = Γ t and Σᵢ |Δrᵢ|² = 6 N D t,
# D = Γ λ² / (6 N). An atom that has just exchanged with the vacancy is likely to exchange back,
# so the cross terms are negative; their long-time effect is the tracer correlation factor f,
# D = f Γ λ² / (6 N) = f Γ a² / (12 N). For the infinite fcc lattice f = 0.78145 (Compaan and
# Haven, 1956; the value is not taken from the source, and the lattice walk below reproduces it
# within its error). The cell here is small and periodic, so f is measured for it by a lattice
# random walk of one vacancy among N atoms in the same 4 × 4 × 4 periodic fcc lattice
# (`kmc_correlation_factor`), with the center-of-mass displacement removed as in the MD run,
# whose total momentum is zero. Known limit: Γ counts the hops the vacancy tracker confirms, and a
# jump undone within the tracker's dwell time counts as none. Such pairs move no atom, so they
# leave D unchanged, but the walk's f counts them, so f Γ λ² / (6 N) is low by the fraction of
# hops missed this way, which is not measured.
using JSON, Printf, Random, Statistics
using PureAtomistic
using PureAtomistic.Analysis: msd, diffusion_coefficient, unwrap!
using PureAtomistic.IO: TrajectoryReader

const LAG_MAX_FS = 50_000.0
const FIT_WINDOW_FS = (10_000.0, 50_000.0)
const F_INFINITE = 0.78145

# f of the periodic fcc lattice of `nrep`³ conventional cells with one vacancy: the slope of the
# summed squared atom displacement against n λ², from `nrun` walks of `njump` jumps each, fitted
# between `njump ÷ 10` and `njump` jumps. Positions are in units of a/2, so λ² = 2.
function kmc_correlation_factor(nrep, nrun, njump, rng)
    M = 2nrep
    sites = [(i, j, k) for i in 0:(M - 1), j in 0:(M - 1), k in 0:(M - 1) if iseven(i + j + k)]
    index = Dict(s => n for (n, s) in enumerate(sites))
    nbr = [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0), (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]
    ns = length(sites)
    N = ns - 1
    n0 = njump ÷ 10
    slopes = Float64[]
    for _ in 1:nrun
        occ = collect(0:(ns - 1))          # occ[site] = atom on it, 0 for the vacancy
        disp = zeros(Int, 3, ns - 1)
        v = 1
        s0 = 0.0
        x0 = 0.0
        sxy = 0.0
        sxx = 0.0
        sx = 0.0
        sy = 0.0
        m = 0
        for n in 1:njump
            d = nbr[rand(rng, 1:12)]
            p = sites[v]
            q = (mod(p[1] + d[1], M), mod(p[2] + d[2], M), mod(p[3] + d[3], M))
            w = index[q]
            atom = occ[w]
            disp[1, atom] -= d[1]
            disp[2, atom] -= d[2]
            disp[3, atom] -= d[3]
            occ[v], occ[w] = atom, 0
            v = w
            if n >= n0 && n % 100 == 0
                c = (sum(@view disp[1, :]), sum(@view disp[2, :]), sum(@view disp[3, :])) ./ N
                y = 0.0
                for a in 1:N
                    y += (disp[1, a] - c[1])^2 + (disp[2, a] - c[2])^2 + (disp[3, a] - c[3])^2
                end
                x = 2.0 * n
                m += 1
                sx += x
                sy += y
                sxx += x * x
                sxy += x * y
            end
        end
        push!(slopes, (sxy - sx * sy / m) / (sxx - sx^2 / m))
    end
    return mean(slopes), std(slopes) / sqrt(nrun)
end

# Slope over 6 of the least-squares line through the points of one block in the fit window.
function block_D(lag, y)
    w = FIT_WINDOW_FS[1] .<= lag .<= FIT_WINDOW_FS[2]
    x, v = lag[w], y[w]
    return sum((x .- mean(x)) .* (v .- mean(v))) / sum(abs2, x .- mean(x)) / 6
end

function main(rundir, record_path, out_path)
    rec = JSON.parsefile(record_path)
    segs = sort(filter(f -> endswith(f, ".patraj"), readdir(joinpath(rundir, "trajectory"); join = true)))
    frames = PureAtomistic.IO.Frame{Float64}[]
    for path in segs
        r = TrajectoryReader(path)
        append!(frames, (r[k] for k in eachindex(r)))
        close(r)
    end
    dt = frames[2].time - frames[1].time
    nblocks = length(segs)
    max_lag = round(Int, LAG_MAX_FS / dt)
    m = msd(frames; max_lag, nblocks)
    fit = diffusion_coefficient(m.lag_fs, m.msd_A2; fit_window = FIT_WINDOW_FS)
    D_b = [block_D(m.lag_fs, m.msd_A2[:, b]) for b in axes(m.msd_A2, 2)]

    # Center of mass of the unwrapped atoms: zero total momentum keeps it in place.
    N = length(frames[1].species)
    com_shift = let u = copy(frames[1].positions), prev = copy(frames[1].positions), c0 = vec(sum(u; dims = 2)) / N, worst = 0.0
        for f in frames[2:end]
            unwrap!(u, prev, f.positions, f.cell)
            worst = max(worst, sqrt(sum(abs2, vec(sum(u; dims = 2)) / N - c0)))
        end
        worst
    end

    a = rec["equilibration"]["lattice_constant_A"]
    λ2 = a^2 / 2
    Γ = rec["rate_per_ps"] / 1000          # per fs
    σΓ = rec["rate_se_per_ps"] / 1000
    fN, σf = kmc_correlation_factor(rec["cells"], 400, 20_000, Xoshiro(20261109))
    pred = fN * Γ * λ2 / (6N)
    σpred = pred * sqrt((σΓ / Γ)^2 + (σf / fN)^2)
    pred_inf = F_INFINITE * Γ * λ2 / (6N)
    z = (fit.D - pred) / sqrt(fit.σ_D^2 + σpred^2)

    to_ps(x) = x * 1000                     # Å²/fs → Å²/ps
    every = max(1, round(Int, 1000 / dt))   # one MSD point per ps
    mean_msd = vec(mean(m.msd_A2; dims = 2))
    sha(path) = readchomp(`git hash-object $path`)
    out = Dict(
        "description" => "Atomic diffusion coefficient of a vacancy campaign run from its trajectory (msd and diffusion_coefficient), against the prediction from the vacancy hop rate",
        "run" => rec["run"], "run_commit" => rec["commit"], "potential" => rec["potential"],
        "temperature_K" => rec["mean_temperature_K"], "natoms" => N, "frames" => length(frames),
        "frame_interval_fs" => dt, "nblocks" => nblocks, "max_lag_fs" => m.lag_fs[end],
        "fit_window_fs" => collect(FIT_WINDOW_FS),
        "msd_source_blob" => sha(joinpath(pkgdir(PureAtomistic), "src", "analysis", "msd.jl")),
        "script_blob" => sha(@__FILE__), "julia" => string(VERSION),
        "D_atoms_A2_per_ps" => to_ps(fit.D), "D_atoms_se_A2_per_ps" => to_ps(fit.σ_D),
        "D_atoms_blocks_A2_per_ps" => to_ps.(D_b),
        "com_max_shift_A" => com_shift,
        "lattice_constant_A" => a, "jump_length_A" => sqrt(λ2),
        "hops" => rec["hops"], "rate_per_ps" => rec["rate_per_ps"], "rate_se_per_ps" => rec["rate_se_per_ps"],
        "correlation_factor_infinite" => F_INFINITE,
        "correlation_factor_cell" => fN, "correlation_factor_cell_se" => σf,
        "correlation_factor_cell_method" => "lattice walk of one vacancy in the same periodic fcc cell, 400 walks of 20000 jumps, slope fitted from 2000 jumps, center of mass removed",
        "D_atoms_predicted_A2_per_ps" => to_ps(pred), "D_atoms_predicted_se_A2_per_ps" => to_ps(σpred),
        "D_atoms_predicted_infinite_f_A2_per_ps" => to_ps(pred_inf),
        "z_measured_minus_predicted" => z,
        "D_v_from_rate_A2_per_ps" => rec["D_v_from_rate_A2_per_ps"], "D_v_msd_A2_per_ps" => rec["D_msd_A2_per_ps"],
        "D_v_msd_se_A2_per_ps" => rec["D_msd_se_A2_per_ps"],
        "msd_lag_ps" => m.lag_fs[1:every:end] ./ 1000, "msd_mean_A2" => mean_msd[1:every:end],
    )
    open(out_path, "w") do io
        JSON.print(io, out, 1)
    end
    @printf(
        "D_atoms = %.3e ± %.3e Å²/ps; predicted %.3e ± %.3e (f_N = %.4f ± %.4f); z = %.2f; COM shift %.3g Å\n",
        to_ps(fit.D), to_ps(fit.σ_D), to_ps(pred), to_ps(σpred), fN, σf, z, com_shift
    )
    return out
end

main(ARGS[1], ARGS[2], ARGS[3])
