# Pass criteria of the MD validation records (bench/results/md_*.json), applied to one record at a
# time and, for a run in another arithmetic or on another device, against a baseline record. The
# constants and rules are fixed here; a run never changes them.
#
#   criteria(record)                    absolute criteria of one record
#   compare_to_baseline(record, base)   agreement of a record with a baseline record
#
# A record is one entry of the "runs" list of md_nve_conservation*.json (an NVE run at one time
# step), the whole of md_thermostat_check*.json (NVT) or the whole of md_vacancy_hop*.json
# (vacancy). The kind is read from the keys. Both functions return Dict{String,Bool}; the key
# "pass" is the conjunction of the others.
using Statistics, JSON
isdefined(@__MODULE__, :correlated_errors) || include(joinpath(@__DIR__, "autocorrelation.jl"))

const STABLE_DRIFT = 1.0e-3  # eV/atom/ps
const STABLE_RMS = 5.0e-4    # eV/atom
const NVT_Z_MAX = 3.0        # |z| of the mean and the variance of the kinetic temperature
const MEAN_T_SES = 3.0       # allowed difference of second-half mean temperatures, in combined SEs
const HOP_P_MIN = 0.01       # exact conditional binomial test of the hop counts
const NVE_DTS = (1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0)
const VACANCY_PS = 22.0      # simulated length of the vacancy run

record_kind(r) = haskey(r, "dt_fs") && haskey(r, "stable") ? :nve :
    haskey(r, "observed") ? :nvt :
    haskey(r, "hops") ? :vacancy :
    throw(ArgumentError("validation record of unknown kind with keys $(sort!(collect(keys(r))))"))

with_pass!(d::Dict{String, Bool}) = (d["pass"] = all(values(d)); d)

"""
    criteria(record)::Dict{String,Bool}

NVE: the run completed, every energy is finite, `|drift| <= STABLE_DRIFT` and
`RMS <= STABLE_RMS`; `"pass"` is stability. NVT: `|mean_z| <= NVT_Z_MAX` with the
autocorrelation error and `|variance_z| <= NVT_Z_MAX`. Vacancy: the run completed, the tracker
holds one vacancy in each of its frames, and the run covers `VACANCY_PS`.
"""
function criteria(r::AbstractDict)::Dict{String, Bool}
    kind = record_kind(r)
    d = Dict{String, Bool}()
    if kind === :nve
        d["completed"] = r["status"] == "completed"
        d["finite"] = haskey(r, "total_eV_per_atom") && all(x -> x isa Real && isfinite(x), r["total_eV_per_atom"])
        d["drift"] = haskey(r, "drift_eV_per_atom_per_ps") && abs(r["drift_eV_per_atom_per_ps"]) <= STABLE_DRIFT
        d["rms"] = haskey(r, "rms_residual_eV_per_atom") && r["rms_residual_eV_per_atom"] <= STABLE_RMS
    elseif kind === :nvt
        o = r["observed"]
        d["mean_z"] = abs(o["autocorrelation"]["mean_z"]) <= NVT_Z_MAX
        d["variance_z"] = abs(o["variance_z"]) <= NVT_Z_MAX
    else
        n = length(r["vacancy_site"])
        d["completed"] = r["status"] == "completed"
        # The tracker rejects a frame without exactly one empty site, so a stored frame has one vacancy.
        d["one_vacancy_per_frame"] = r["frames"] == n == length(r["t_fs"]) == length(r["vacancy_R"]) && n > 0
        d["length"] = r["duration_ps"] >= VACANCY_PS - 1.0e-9
    end
    return with_pass!(d)
end

# Exact two-sided p-value of `k` successes in `n` Bernoulli(q) trials: the total probability of
# the outcomes no more likely than `k`.
function binomial_exact_p(k::Integer, n::Integer, q::Real)
    0 <= k <= n || throw(ArgumentError("binomial_exact_p: k = $k outside 0:$n"))
    (0 < q < 1) || throw(ArgumentError("binomial_exact_p: q = $q outside (0, 1)"))
    lf = cumsum(log.(1:n)); pushfirst!(lf, 0.0)  # lf[j+1] = log(j!)
    logpmf(j) = lf[n + 1] - lf[j + 1] - lf[n - j + 1] + j * log(q) + (n - j) * log1p(-q)
    lk = logpmf(k)
    return min(1.0, sum(exp(logpmf(j)) for j in 0:n if logpmf(j) <= lk + 1.0e-7))
end

# Second-half mean and its autocorrelation standard error of a temperature series.
function second_half(T::AbstractVector)
    x = Float64.(T[(length(T) ÷ 2 + 1):end])
    return (mean = mean(x), se = correlated_errors(x).mean_error)
end

"""
    compare_to_baseline(record, baseline)::Dict{String,Bool}

NVE: the record is stable whenever the baseline is (`"pass"` for this dt). NVT: the second-half
mean temperatures differ by at most `MEAN_T_SES` combined autocorrelation SEs. Vacancy: the hop
counts are consistent by the exact conditional binomial test (`p >= HOP_P_MIN`, the counts
split in proportion to the simulated lengths) and the second-half mean temperatures agree
within `MEAN_T_SES` combined SEs.
"""
function compare_to_baseline(r::AbstractDict, base::AbstractDict)::Dict{String, Bool}
    kind = record_kind(r)
    kind === record_kind(base) || throw(ArgumentError("record is $kind but the baseline is $(record_kind(base))"))
    d = Dict{String, Bool}()
    if kind === :nve
        r["dt_fs"] == base["dt_fs"] || throw(ArgumentError("time steps differ: $(r["dt_fs"]) and $(base["dt_fs"])"))
        d["stable_where_baseline_stable"] = !criteria(base)["pass"] || criteria(r)["pass"]
    elseif kind === :nvt
        a, b = r["observed"]["autocorrelation"], base["observed"]["autocorrelation"]
        se = hypot(a["mean_error_K"], b["mean_error_K"])
        d["mean_temperature"] = abs(r["observed"]["mean_K"] - base["observed"]["mean_K"]) <= MEAN_T_SES * se
    else
        n1, n2 = r["hops"], base["hops"]
        t1, t2 = r["duration_ps"], base["duration_ps"]
        d["hop_counts"] = n1 + n2 == 0 || binomial_exact_p(n1, n1 + n2, t1 / (t1 + t2)) >= HOP_P_MIN
        a, b = second_half(r["temperature_K"]), second_half(base["temperature_K"])
        d["mean_temperature"] = abs(a.mean - b.mean) <= MEAN_T_SES * hypot(a.se, b.se)
    end
    return with_pass!(d)
end

# Adds `md_float32_baseline` to the three Float32 records: the verdict of `criteria` on the
# Float32 data (the Float64 rules, so "Float32 alone meets the Float64 criterion"), and the
# agreement with the Float64 records. `dir` holds md_*.json and md_*_f32.json.
function write_float32_baseline(dir::AbstractString)
    load(f) = JSON.parsefile(joinpath(dir, f))
    function put(f32file, record)
        d = load(f32file)
        d["md_float32_baseline"] = record
        open(io -> JSON.print(io, d, 1), joinpath(dir, f32file), "w")
        return record
    end
    f64, f32 = load("md_nve_conservation.json"), load("md_nve_conservation_f32.json")
    runs64 = Dict(r["dt_fs"] => r for r in f64["runs"])
    nve = Dict{String, Any}()
    for r in f32["runs"]
        base = runs64[r["dt_fs"]]
        nve[string(r["dt_fs"])] = Dict(
            "criteria_float32" => criteria(r), "criteria_float64" => criteria(base),
            "float32_alone_meets_float64_criteria" => criteria(r)["pass"] == criteria(base)["pass"],
            "versus_float64" => compare_to_baseline(r, base),
        )
    end
    put("md_nve_conservation_f32.json", nve)
    for (name, f64name) in (("md_thermostat_check", "md_thermostat_check.json"), ("md_vacancy_hop", "md_vacancy_hop.json"))
        r, base = load(name * "_f32.json"), load(f64name)
        put(
            name * "_f32.json", Dict(
                "criteria_float32" => criteria(r), "criteria_float64" => criteria(base),
                "float32_alone_meets_float64_criteria" => criteria(r)["pass"],
                "versus_float64" => compare_to_baseline(r, base),
            )
        )
    end
    return nothing
end
