# Arrhenius fit of a thermally activated rate.

# Boltzmann constant in eV/K: k = 1.380649e-23 J/K (exact in the 2019 SI) divided by
# e = 1.602176634e-19 C (exact), giving 8.617333262145179e-5 eV/K.
const BOLTZMANN_EV_PER_K = 8.617333262145179e-5

"""
    arrhenius_fit(temps, rates, errors)

Weighted least-squares fit of `ln rate = ln Γ₀ - E / (k_B T)` to rates (`1/time`) with one
standard error each, at temperatures `temps` (K). The weights are `(rate / error)²`, the
inverse variances of `ln rate` to first order. Returns `(E_m, σ, prefactor)`: the activation
energy in eV, its standard error from the stated errors (not rescaled by the fit's χ²), and
`Γ₀` in the units of `rates`.
"""
function arrhenius_fit(temps::AbstractVector, rates::AbstractVector, errors::AbstractVector)
    T = float(promote_type(eltype(temps), eltype(rates), eltype(errors)))
    eachindex(temps, rates, errors)
    length(temps) >= 2 ||
        throw(ArgumentError("pureatomistic: arrhenius_fit needs at least 2 points"))
    all(>(0), temps) || throw(ArgumentError("pureatomistic: arrhenius_fit needs positive temperatures"))
    all(>(0), rates) || throw(ArgumentError("pureatomistic: arrhenius_fit needs positive rates"))
    all(>(0), errors) || throw(ArgumentError("pureatomistic: arrhenius_fit needs positive errors"))
    kB = T(BOLTZMANN_EV_PER_K)
    sw = zero(T)
    swx = zero(T)
    for i in eachindex(temps, rates, errors)
        w = (T(rates[i]) / T(errors[i]))^2
        sw += w
        swx += w / (kB * T(temps[i]))
    end
    xbar = swx / sw
    sxx = zero(T)
    sxy = zero(T)
    swy = zero(T)
    for i in eachindex(temps, rates, errors)
        w = (T(rates[i]) / T(errors[i]))^2
        dx = 1 / (kB * T(temps[i])) - xbar
        y = log(T(rates[i]))
        sxx += w * dx^2
        sxy += w * dx * y
        swy += w * y
    end
    sxx > 0 || throw(ArgumentError("pureatomistic: arrhenius_fit needs at least 2 distinct temperatures"))
    slope = sxy / sxx
    intercept = swy / sw - slope * xbar
    return (E_m = -slope, σ = sqrt(1 / sxx), prefactor = exp(intercept))
end
