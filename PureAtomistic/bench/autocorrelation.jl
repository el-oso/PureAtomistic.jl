# Statistical error of the mean of a correlated series from its integrated autocorrelation time.
using Statistics

# Integrated autocorrelation time `τ = 1/2 + Σ_{t=1}^{M} ρ(t)` in samples, with the window `M` the
# smallest lag that satisfies `M ≥ c τ(M)` (Sokal's automatic windowing). `ρ` is the biased
# sample autocorrelation, which is positive semidefinite, so `τ` stays positive. The standard
# error of the mean is `std(x) sqrt(2τ/N)`; `N_eff = N/(2τ)` is the number of independent samples.
function integrated_autocorrelation(x::AbstractVector; c = 6)
    N = length(x)
    xc = x .- mean(x)
    c0 = sum(abs2, xc) / N
    τ = 0.5
    for M in 1:(N - 1)
        τ += sum(xc[i] * xc[i + M] for i in 1:(N - M)) / (N * c0)
        M >= c * τ && return (; tau = τ, window = M)
    end
    throw(ArgumentError("the autocorrelation window does not close within the series; the series is too short for c = $c"))
end

# Standard error of the mean, effective sample size and the standard error of the skewness
# (`sqrt(6/N_eff)`, the large-sample value for independent Gaussian samples) of a series.
function correlated_errors(x::AbstractVector; c = 6)
    N = length(x)
    (; tau, window) = integrated_autocorrelation(x; c)
    n_eff = N / (2tau)
    return (;
        c, tau_samples = tau, window_samples = window, n_eff, mean_error = std(x) * sqrt(2tau / N),
        skewness_error = sqrt(6 / n_eff),
    )
end

# The JSON record of the autocorrelation analysis of the series `x` against the theoretical mean
# and skewness, with the sensitivity of the mean error to the window constant `c`.
function autocorrelation_record(x::AbstractVector, mean_theory, skewness_observed, skewness_theory)
    e = correlated_errors(x)
    return Dict(
        "method" => "integrated autocorrelation time with Sokal's automatic window M >= c tau; SEM = std sqrt(2 tau / N); skewness SE = sqrt(6 / N_eff)",
        "n_samples" => length(x), "c" => e.c, "tau_samples" => e.tau_samples, "window_samples" => e.window_samples,
        "n_eff" => e.n_eff, "mean_error_K" => e.mean_error, "mean_z" => (mean(x) - mean_theory) / e.mean_error,
        "skewness_error" => e.skewness_error, "skewness_z" => (skewness_observed - skewness_theory) / e.skewness_error,
        "window_sensitivity" => [
            let r = correlated_errors(x; c)
                Dict(
                    "c" => c, "tau_samples" => r.tau_samples, "mean_error_K" => r.mean_error,
                    "mean_z" => (mean(x) - mean_theory) / r.mean_error,
                )
            end for c in (4, 5, 6, 8)
        ],
    )
end
