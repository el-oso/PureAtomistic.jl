# Analysis of the vacancy-diffusion campaign from its stored records; it runs no dynamics.
#
#   julia --project=PureAtomistic/bench PureAtomistic/bench/vacancy_analysis.jl
#
# Reads bench/results/vacancy_campaign/*.json (the run records), neb_vacancy.json (the NEB
# barriers), trajectory_diffusion_<run>.json (the atomic MSD of each run, written by
# trajectory_diffusion.jl from the trajectories) and md_vacancy_hop_f32{,_metal}.json (the 1300 K
# runs at the 0 K lattice constant), and writes bench/results/vacancy_campaign_analysis.json.
#
# Method (docs/superpowers/specs/2026-11-08-vacancy-campaign-proposal.md):
# - Γ = n/t with σ = √n/t per temperature, pooled over the two 255-atom runs; T is the mean of the
#   two runs' mean kinetic temperatures (equal durations), a the mean of their NPT lattice
#   constants.
# - Within one temperature: the exact conditional binomial test of the two counts (`hop_test`);
#   the dispersion of the per-segment counts (χ² of equal 100 ps segments) with its p-value from
#   the multinomial distribution of the counts given their total, by simulation; and the longest
#   time without a confirmed hop, with the exact probability that n uniform hop times in [0, t]
#   leave a gap at least that long.
# - `arrhenius_fit` of the four pooled rates; errors are not rescaled.
# - NEB: E_m(a) is linear in a through the two NEB points (both 255 atoms, Float64). With
#   x = 1/(k_B T) and a(T) from the runs, the rates expected from that barrier alone,
#   ln Γᵢ = ln ν − xᵢ E_m(aᵢ), fitted with the same weights as the data, give the effective barrier
#   E_eff = −d ln Γ / dx = E_m(a) + x (dE_m/da)(da/dx) that the zero-pressure fit measures, and the
#   shift of its intercept from ln ν.
# - Size check: 499 against 255 atoms at 1200 K, the exact conditional binomial test and the exact
#   (Clopper-Pearson) 95 % interval of the rate ratio, and the barrier shift δ = −k_B T ln ρ.
# - D: atomic D from the MSD against f Γ a²/(12 N), f measured for each cell by a lattice walk
#   (trajectory_diffusion.jl); the vacancy D_v = Γ a²/12 against the MSD of the tracked vacancy.
using JSON, Printf, Random, Statistics
using PureAtomistic
using PureAtomistic.Analysis: arrhenius_fit
include(joinpath(@__DIR__, "validation_criteria.jl"))

const RESULTS = joinpath(@__DIR__, "results")
const CAMPAIGN = joinpath(RESULTS, "vacancy_campaign")
const KB = 8.617333262e-5                      # eV/K
const TEMPS = ["900", "980", "1080", "1200"]
const SIZE_RUN = "T1200_dt2_s04_n499"
const NSIM = 200_000                           # multinomial draws per dispersion test

readjson(path) = JSON.parsefile(path)
record(run) = readjson(joinpath(CAMPAIGN, run * ".json"))

function chi2_equal(counts)
    m = sum(counts) / length(counts)
    return iszero(m) ? 0.0 : sum((c - m)^2 for c in counts) / m
end

# p-value of the χ² of `counts` against equal segments, given their total: the fraction of
# multinomial draws whose χ² is at least the observed one.
function dispersion_p(counts, rng)
    k, n = length(counts), sum(counts)
    obs = chi2_equal(counts)
    sim = zeros(Int, k)
    hits = 0
    for _ in 1:NSIM
        fill!(sim, 0)
        for _ in 1:n
            sim[rand(rng, 1:k)] += 1
        end
        hits += chi2_equal(sim) >= obs - 1.0e-9
    end
    return obs, k - 1, hits / NSIM
end

# Probability that the n + 1 gaps of n uniform points in [0, t] have a largest gap of at least g
# (inclusion-exclusion over the gaps, exact; BigFloat for the alternating sum).
function max_gap_p(n, t, g)
    s = big(0.0)
    for j in 1:(n + 1)
        r = 1 - j * big(g) / t
        r <= 0 && break
        s += (-1)^(j + 1) * binomial(big(n + 1), j) * r^n
    end
    return Float64(s)
end

function longest_gap(r)
    t = r["duration_ps"]
    h = Float64.(r["hop_times_fs"]) ./ 1000
    edges = [0.0; h; t]
    gaps = diff(edges)
    i = argmax(gaps)
    return (gap_ps = gaps[i], from_ps = edges[i], to_ps = edges[i + 1], p = max_gap_p(length(h), t, gaps[i]))
end

function bisect(f, lo, hi)
    for _ in 1:200
        mid = (lo + hi) / 2
        f(lo) * f(mid) <= 0 ? (hi = mid) : (lo = mid)
    end
    return (lo + hi) / 2
end
binomial_cdf(k, n, q) = sum(binomial(big(n), j) * big(q)^j * (1 - big(q))^(n - j) for j in 0:k)
function clopper_pearson(k, n; α = 0.05)
    lo = k == 0 ? 0.0 : bisect(q -> Float64(1 - binomial_cdf(k - 1, n, q)) - α / 2, 0.0, 1.0)
    hi = k == n ? 1.0 : bisect(q -> Float64(binomial_cdf(k, n, q)) - α / 2, 0.0, 1.0)
    return lo, hi
end

function main()
    rng = Xoshiro(20261112)
    summary = readjson(joinpath(CAMPAIGN, "summary.json"))
    neb = readjson(joinpath(RESULTS, "neb_vacancy.json"))["summary"]["barriers"]

    # ---- per temperature
    per_T = Dict{String, Any}[]
    for key in TEMPS
        s = summary["temperatures"][key]
        rs = record.(s["runs"])
        all(r -> r["natoms"] == 255 && r["dt_fs"] == 2.0 && r["status"] == "completed", rs) ||
            error("analysis: unexpected run in the $key K set")
        n = sum(r["hops"] for r in rs)
        t = sum(r["duration_ps"] for r in rs)
        Tm = mean(r["mean_temperature_K"] for r in rs)
        a = mean(r["equilibration"]["lattice_constant_A"] for r in rs)
        runs = Dict{String, Any}[]
        for r in rs
            segs = [sg["hops"] for sg in r["segments"]]
            c2, dof, p = dispersion_p(segs, rng)
            lg = longest_gap(r)
            push!(
                runs, Dict{String, Any}(
                    "run" => r["run"], "hops" => r["hops"], "duration_ps" => r["duration_ps"],
                    "rate_per_ps" => r["hops"] / r["duration_ps"], "rate_se_per_ps" => sqrt(r["hops"]) / r["duration_ps"],
                    "mean_temperature_K" => r["mean_temperature_K"], "temperature_se_K" => r["temperature_se_K"],
                    "mean_pressure_GPa" => r["mean_pressure_GPa"], "pressure_se_GPa" => r["pressure_se_GPa"],
                    "lattice_constant_A" => r["equilibration"]["lattice_constant_A"],
                    "segment_hops" => segs, "segment_chi2" => c2, "segment_dof" => dof, "segment_dispersion_p" => p,
                    "longest_gap_ps" => lg.gap_ps, "longest_gap_from_ps" => lg.from_ps, "longest_gap_to_ps" => lg.to_ps,
                    "longest_gap_p" => lg.p,
                    "transient_episodes" => r["transient_episodes"], "transient_frames" => r["transient_frames"],
                    "transient_longest_frames" => r["transient_longest_frames"], "regrowths" => r["regrowths"],
                    "harness_commit" => r["harness_commit"],
                )
            )
        end
        pooled = reduce(vcat, [[sg["hops"] for sg in r["segments"]] for r in rs])
        c2, dof, p = dispersion_p(pooled, rng)
        push!(
            per_T, Dict{String, Any}(
                "nominal_K" => parse(Float64, key), "mean_temperature_K" => Tm, "lattice_constant_A" => a,
                "hops" => n, "duration_ps" => t, "rate_per_ps" => n / t, "rate_se_per_ps" => sqrt(n) / t,
                "hop_test" => hop_test(rs[1], rs[2]),
                "segment_chi2" => c2, "segment_dof" => dof, "segment_dispersion_p" => p,
                "transient_episodes" => sum(r["transient_episodes"] for r in rs), "runs" => runs,
            )
        )
    end

    # ---- Arrhenius fit
    Ts = [d["mean_temperature_K"] for d in per_T]
    Γs = [d["rate_per_ps"] for d in per_T]
    σs = [d["rate_se_per_ps"] for d in per_T]
    fit = arrhenius_fit(Ts, Γs, σs)
    fit_nominal = arrhenius_fit([d["nominal_K"] for d in per_T], Γs, σs)
    # Sensitivity to the least homogeneous run: 1080 K from T1080_dt2_s05 alone.
    i1080 = findfirst(d -> d["nominal_K"] == 1080.0, per_T)
    r05 = only(r for r in per_T[i1080]["runs"] if r["run"] == "T1080_dt2_s05")
    Ts_s = copy(Ts); Γs_s = copy(Γs); σs_s = copy(σs)
    Ts_s[i1080], Γs_s[i1080], σs_s[i1080] = r05["mean_temperature_K"], r05["rate_per_ps"], r05["rate_se_per_ps"]
    fit_s05 = arrhenius_fit(Ts_s, Γs_s, σs_s)
    ns = [d["hops"] for d in per_T]
    x = 1 ./ (KB .* Ts)
    xbar = sum(ns .* x) / sum(ns)
    Tc = 1 / (KB * xbar)
    p_chi2 = exp(-fit.chi2 / 2)                 # χ² survival function on 2 degrees of freedom

    # ---- NEB comparison
    a1, E1 = neb[1]["lattice_constant_A"], neb[1]["E_m_eV"]
    a2, E2 = neb[2]["lattice_constant_A"], neb[2]["E_m_eV"]
    slope = (E2 - E1) / (a2 - a1)              # eV/Å
    E_neb(a) = E1 + slope * (a - a1)
    as = [d["lattice_constant_A"] for d in per_T]
    yneb = [-x[i] * E_neb(as[i]) for i in eachindex(x)]
    sxx = sum(ns .* (x .- xbar) .^ 2)
    E_eff = -sum(ns .* (x .- xbar) .* yneb) / sxx
    ln_shift = sum(ns .* yneb) / sum(ns) + E_eff * xbar     # intercept of the model fit minus ln ν
    # a(T) as a weighted line, for the closed form E_eff = E_m(a(0)) with a(0) the extrapolation.
    Tbar = sum(ns .* Ts) / sum(ns)
    dadT = sum(ns .* (Ts .- Tbar) .* as) / sum(ns .* (Ts .- Tbar) .^ 2)
    a_c = sum(ns .* as) / sum(ns)
    E_at_center = E_neb(a_c + dadT * (Tc - Tbar))
    ν = fit.prefactor / exp(ln_shift)
    z_eff = (fit.E_m - E_eff) / fit.σ
    z_center = (fit.E_m - E_at_center) / fit.σ

    # 1300 K at the 0 K lattice constant (about 9 GPa), against the zero-pressure fit.
    hv = [readjson(joinpath(RESULTS, f)) for f in ("md_vacancy_hop_f32.json", "md_vacancy_hop_f32_metal.json")]
    n13 = sum(h["hops"] for h in hv)
    t13 = sum(h["duration_ps"] for h in hv)
    T13 = mean(h["mean_temperature_K"] for h in hv)
    Γ13 = n13 / t13
    Γ13_fit = fit.prefactor * exp(-fit.E_m / (KB * T13))
    σlnfit = sqrt(fit.σ_ln_prefactor^2 + (fit.σ / (KB * T13))^2 - 2 * fit.cov_E_ln_prefactor / (KB * T13))
    ΔE13 = KB * T13 * log(Γ13_fit / Γ13)
    σΔE13 = KB * T13 * sqrt(1 / n13 + σlnfit^2)
    lat = readjson(joinpath(RESULTS, "neb_vacancy.json"))["lattice"]
    a13 = only(l["lattice_constant_A"] for l in lat if l["temperature_K"] == 1300.0)
    ΔE13_neb = E_neb(a1) - E_neb(a13)

    # ---- size check
    big_r = record(SIZE_RUN)
    s1200 = only(d for d in per_T if d["nominal_K"] == 1200.0)
    n_b, t_b = big_r["hops"], big_r["duration_ps"]
    n_s, t_s = s1200["hops"], s1200["duration_ps"]
    q = t_b / (t_b + t_s)
    lo, hi = clopper_pearson(n_b, n_b + n_s)
    ratio(p) = p / (1 - p) * (t_s / t_b)
    ρ = (n_b / t_b) / (n_s / t_s)
    kT = KB * s1200["mean_temperature_K"]
    size_check = Dict{String, Any}(
        "run" => SIZE_RUN, "natoms" => big_r["natoms"], "hops" => n_b, "duration_ps" => t_b,
        "rate_per_ps" => n_b / t_b, "rate_se_per_ps" => sqrt(n_b) / t_b,
        "lattice_constant_A" => big_r["equilibration"]["lattice_constant_A"],
        "mean_temperature_K" => big_r["mean_temperature_K"], "mean_pressure_GPa" => big_r["mean_pressure_GPa"],
        "transient_episodes" => big_r["transient_episodes"],
        "baseline_hops" => n_s, "baseline_duration_ps" => t_s,
        "binomial_p" => binomial_exact_p(n_b, n_b + n_s, q),
        "rate_ratio" => ρ, "rate_ratio_ci95" => [ratio(lo), ratio(hi)],
        "barrier_shift_eV" => -kT * log(ρ), "barrier_shift_ci95_eV" => [-kT * log(ratio(hi)), -kT * log(ratio(lo))],
        "barrier_shift_from_lattice_constant_eV" => slope * (big_r["equilibration"]["lattice_constant_A"] - s1200["lattice_constant_A"]),
    )

    # ---- diffusion
    diffusion = Dict{String, Any}[]
    for d in per_T
        tds = [readjson(joinpath(RESULTS, "trajectory_diffusion_$(r["run"]).json")) for r in d["runs"]]
        f = mean(td["correlation_factor_cell"] for td in tds)
        σf = mean(td["correlation_factor_cell_se"] for td in tds)
        N = only(unique(td["natoms"] for td in tds))
        Dm = mean(td["D_atoms_A2_per_ps"] for td in tds)
        σDm = sqrt(sum(td["D_atoms_se_A2_per_ps"]^2 for td in tds)) / length(tds)
        a = d["lattice_constant_A"]
        Dp = f * d["rate_per_ps"] * a^2 / (12N)
        σDp = Dp * sqrt((d["rate_se_per_ps"] / d["rate_per_ps"])^2 + (σf / f)^2)
        rs = [record(r["run"]) for r in d["runs"]]
        Dv = d["rate_per_ps"] * a^2 / 12
        Dvm = mean(r["D_msd_A2_per_ps"] for r in rs)
        σDvm = sqrt(sum(r["D_msd_se_A2_per_ps"]^2 for r in rs)) / length(rs)
        push!(
            diffusion, Dict{String, Any}(
                "nominal_K" => d["nominal_K"], "natoms" => N, "correlation_factor" => f, "correlation_factor_se" => σf,
                "D_atoms_msd_A2_per_ps" => Dm, "D_atoms_msd_se_A2_per_ps" => σDm,
                "D_atoms_msd_runs_A2_per_ps" => [td["D_atoms_A2_per_ps"] for td in tds],
                "D_atoms_from_rate_A2_per_ps" => Dp, "D_atoms_from_rate_se_A2_per_ps" => σDp,
                "z_atoms" => (Dm - Dp) / hypot(σDm, σDp),
                "D_vacancy_from_rate_A2_per_ps" => Dv, "D_vacancy_from_rate_se_A2_per_ps" => Dv / sqrt(d["hops"]),
                "D_vacancy_msd_A2_per_ps" => Dvm, "D_vacancy_msd_se_A2_per_ps" => σDvm,
                "z_vacancy" => (Dvm - Dv) / hypot(σDvm, Dv / sqrt(d["hops"])),
            )
        )
    end

    sha(path) = readchomp(`git hash-object $path`)
    out = Dict{String, Any}(
        "description" => "Analysis of the vacancy-diffusion campaign (one vacancy in Cu, MACE-MPA-0, Float32 on Metal, 255 atoms, 2 fs, zero pressure): pooled hop rates, homogeneity tests, Arrhenius fit, comparison with the NEB barrier and thermal expansion, size check and diffusion coefficients (bench/vacancy_analysis.jl)",
        "script_blob" => sha(@__FILE__), "julia" => string(VERSION),
        "campaign_gpu_hours" => summary["gpu_hours"], "campaign_completed" => summary["date"],
        "temperatures" => per_T,
        "fit" => Dict{String, Any}(
            "temperatures_K" => Ts, "E_m_eV" => fit.E_m, "σ_E_m_eV" => fit.σ,
            "prefactor_per_ps" => fit.prefactor, "σ_ln_prefactor" => fit.σ_ln_prefactor,
            "prefactor_factor_1σ" => exp(fit.σ_ln_prefactor),
            "cov_E_ln_prefactor_eV" => fit.cov_E_ln_prefactor, "chi2" => fit.chi2, "dof" => length(Ts) - 2,
            "chi2_p" => p_chi2, "weighted_center_K" => Tc,
            "rate_at_center_per_ps" => fit.prefactor * exp(-fit.E_m * xbar),
            "rate_at_center_relative_se" => 1 / sqrt(sum(ns)),
            "target_σ_E_m_eV" => 0.036,
            "nominal_temperatures" => Dict("E_m_eV" => fit_nominal.E_m, "σ_E_m_eV" => fit_nominal.σ, "prefactor_per_ps" => fit_nominal.prefactor),
            "without_T1080_dt2_s06" => Dict("E_m_eV" => fit_s05.E_m, "σ_E_m_eV" => fit_s05.σ, "prefactor_per_ps" => fit_s05.prefactor, "chi2" => fit_s05.chi2),
        ),
        "neb" => Dict{String, Any}(
            "points" => [Dict("lattice_constant_A" => a1, "E_m_eV" => E1), Dict("lattice_constant_A" => a2, "E_m_eV" => E2)],
            "method" => "E_m(a) linear in a through the two NEB points (255 atoms, Float64, fixed cell); linearity between them is assumed, not measured",
            "dE_m_da_eV_per_A" => slope,
            "E_m_at_campaign_a_eV" => Dict(string(Int(d["nominal_K"])) => E_neb(d["lattice_constant_A"]) for d in per_T),
            "da_dT_A_per_K" => dadT,
            "E_m_at_weighted_center_eV" => E_at_center,
            "E_eff_expected_eV" => E_eff,
            "a_extrapolated_to_0K_A" => a_c - dadT * Tbar,
            "ln_prefactor_shift" => ln_shift,
            "attempt_rate_implied_per_ps" => ν, "attempt_frequency_per_jump_THz" => ν / 12,
            "z_fit_minus_E_eff" => z_eff, "z_fit_minus_E_at_center" => z_center,
        ),
        "fixed_volume_1300K" => Dict{String, Any}(
            "hops" => n13, "duration_ps" => t13, "mean_temperature_K" => T13, "rate_per_ps" => Γ13,
            "lattice_constant_A" => a1, "zero_pressure_lattice_constant_1300K_A" => a13,
            "rate_from_fit_per_ps" => Γ13_fit,
            "barrier_increase_from_rates_eV" => ΔE13, "barrier_increase_from_rates_se_eV" => σΔE13,
            "barrier_increase_from_neb_line_eV" => ΔE13_neb,
            "note" => "runs at the 0 K lattice constant (about 9 GPa), not in the fit; the barrier increase from the rates assumes the same prefactor",
        ),
        "size_check" => size_check,
        "diffusion" => diffusion,
    )
    open(joinpath(RESULTS, "vacancy_campaign_analysis.json"), "w") do io
        JSON.print(io, out, 1)
    end

    for d in per_T
        @printf(
            "%5.0f K  T %.1f  a %.4f  n %3d  t %5.0f ps  Γ %.4f ± %.4f  binom p %.3f  disp χ² %.1f/%d p %.3f  trans %d\n",
            d["nominal_K"], d["mean_temperature_K"], d["lattice_constant_A"], d["hops"], d["duration_ps"],
            d["rate_per_ps"], d["rate_se_per_ps"], d["hop_test"]["p"], d["segment_chi2"], d["segment_dof"],
            d["segment_dispersion_p"], d["transient_episodes"]
        )
        for r in d["runs"]
            @printf(
                "    %s n %3d  χ² %.1f/%d p %.4f  longest gap %.1f ps (%.1f–%.1f) p %.2e  trans %d\n", r["run"], r["hops"],
                r["segment_chi2"], r["segment_dof"], r["segment_dispersion_p"], r["longest_gap_ps"],
                r["longest_gap_from_ps"], r["longest_gap_to_ps"], r["longest_gap_p"], r["transient_episodes"]
            )
        end
    end
    @printf(
        "fit: E_m %.4f ± %.4f eV, Γ₀ %.1f /ps (×/÷ %.2f), χ² %.3f / 2 (p %.3f), center %.0f K; nominal T: %.4f ± %.4f\n",
        fit.E_m, fit.σ, fit.prefactor, exp(fit.σ_ln_prefactor), fit.chi2, p_chi2, Tc, fit_nominal.E_m, fit_nominal.σ
    )
    @printf("fit with 1080 K from s05 alone: E_m %.4f ± %.4f eV, Γ₀ %.1f /ps, χ² %.3f\n", fit_s05.E_m, fit_s05.σ, fit_s05.prefactor, fit_s05.chi2)
    @printf(
        "NEB: slope %.4f eV/Å, da/dT %.3e Å/K, E_m(a) %s; at center %.4f; E_eff %.4f (z %.2f; vs center z %.2f); ln shift %.3f, ν %.1f /ps (%.2f THz per jump)\n",
        slope, dadT, join([@sprintf("%.4f", E_neb(a)) for a in as], " "), E_at_center, E_eff, z_eff, z_center, ln_shift, ν, ν / 12
    )
    @printf("1300 K fixed volume: Γ %.4f, fit %.4f; ΔE %.3f ± %.3f eV; NEB line %.3f eV\n", Γ13, Γ13_fit, ΔE13, σΔE13, ΔE13_neb)
    @printf(
        "size: ρ %.3f [%.3f, %.3f], p %.3f, δ %.4f [%.4f, %.4f] eV, δ from Δa %.4f eV\n", ρ, size_check["rate_ratio_ci95"]...,
        size_check["binomial_p"], size_check["barrier_shift_eV"], size_check["barrier_shift_ci95_eV"]...,
        size_check["barrier_shift_from_lattice_constant_eV"]
    )
    for d in diffusion
        @printf(
            "%5.0f K D_atoms msd %.3e ± %.3e, rate %.3e ± %.3e (z %.2f); D_v rate %.4f ± %.4f, msd %.4f ± %.4f (z %.2f)\n",
            d["nominal_K"], d["D_atoms_msd_A2_per_ps"], d["D_atoms_msd_se_A2_per_ps"], d["D_atoms_from_rate_A2_per_ps"],
            d["D_atoms_from_rate_se_A2_per_ps"], d["z_atoms"], d["D_vacancy_from_rate_A2_per_ps"],
            d["D_vacancy_from_rate_se_A2_per_ps"], d["D_vacancy_msd_A2_per_ps"], d["D_vacancy_msd_se_A2_per_ps"], d["z_vacancy"]
        )
    end
    return out
end

main()
