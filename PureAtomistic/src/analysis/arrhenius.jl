# Arrhenius fit of a thermally activated rate.

"""
    arrhenius_fit(temps, rates, errors)

Weighted least-squares fit of `ln rate = ln Γ₀ - E_m / (k_B T)` to rates (`1/time`) with one
standard error each, at temperatures `temps` (K). Returns the named tuple
`(E_m, σ, prefactor, σ_ln_prefactor, cov_E_ln_prefactor, chi2)`: the activation energy `E_m` in
eV and its standard error `σ`, `Γ₀` in the units of `rates`, the standard error of `ln Γ₀`, the
covariance of `E_m` and `ln Γ₀` (eV), and the weighted sum of squared residuals `χ²`.

With `x = 1/(k_B T)` and `y = ln rate`, the model is `y = a + b x` with `a = ln Γ₀` and
`b = -E_m`. The weights `wᵢ = (rateᵢ / errorᵢ)²` are the inverse variances of `yᵢ` to first
order (`Var ln Γ ≈ Var Γ / Γ²`). For a hop count `n` in a time `t`, with the Poisson rate `n/t`
and error `√n/t` of [`hop_rate`](@ref), the weight is `n` itself. With the weighted means `x̄`
and `ȳ` and `Sxx = Σ wᵢ (xᵢ - x̄)²`, the estimates are `b = Σ wᵢ (xᵢ - x̄) yᵢ / Sxx` and
`a = ȳ - b x̄`. Since `b` is a linear combination of the `yᵢ` with coefficients
`wᵢ (xᵢ - x̄)/Sxx`, `Var b = 1/Sxx`; the same coefficients sum to zero, so `Cov(ȳ, b) = 0`,
which gives `Var a = 1/Σwᵢ + x̄²/Sxx` and `Cov(a, b) = -x̄/Sxx`, hence
`Cov(E_m, ln Γ₀) = x̄/Sxx`. These follow from the stated errors and are not rescaled by the fit:
`χ² = Σ wᵢ (yᵢ - a - b xᵢ)²` has `length(temps) - 2` degrees of freedom when the errors are
right, and a large value says the errors or the Arrhenius form are not.
"""
function arrhenius_fit(temps::AbstractVector, rates::AbstractVector, errors::AbstractVector)
    T = float(promote_type(eltype(temps), eltype(rates), eltype(errors)))
    eachindex(temps, rates, errors)
    length(temps) >= 2 ||
        throw(ArgumentError("pureatomistic: arrhenius_fit needs at least 2 points"))
    all(>(0), temps) || throw(ArgumentError("pureatomistic: arrhenius_fit needs positive temperatures"))
    all(>(0), rates) || throw(ArgumentError("pureatomistic: arrhenius_fit needs positive rates"))
    all(>(0), errors) || throw(ArgumentError("pureatomistic: arrhenius_fit needs positive errors"))
    kB = T(KB_EV_PER_K)
    sw = zero(T)
    swx = zero(T)
    swy = zero(T)
    for i in eachindex(temps, rates, errors)
        w = (T(rates[i]) / T(errors[i]))^2
        sw += w
        swx += w / (kB * T(temps[i]))
        swy += w * log(T(rates[i]))
    end
    xbar = swx / sw
    ybar = swy / sw
    sxx = zero(T)
    sxy = zero(T)
    for i in eachindex(temps, rates, errors)
        w = (T(rates[i]) / T(errors[i]))^2
        dx = 1 / (kB * T(temps[i])) - xbar
        sxx += w * dx^2
        sxy += w * dx * log(T(rates[i]))
    end
    sxx > 0 || throw(ArgumentError("pureatomistic: arrhenius_fit needs at least 2 distinct temperatures"))
    slope = sxy / sxx
    chi2 = zero(T)
    for i in eachindex(temps, rates, errors)
        w = (T(rates[i]) / T(errors[i]))^2
        dx = 1 / (kB * T(temps[i])) - xbar
        chi2 += w * (log(T(rates[i])) - ybar - slope * dx)^2
    end
    return (
        E_m = -slope, σ = sqrt(1 / sxx), prefactor = exp(ybar - slope * xbar),
        σ_ln_prefactor = sqrt(1 / sw + xbar^2 / sxx), cov_E_ln_prefactor = xbar / sxx, chi2,
    )
end
