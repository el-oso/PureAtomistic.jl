# Vacancy hop rate and diffusion coefficient from a `VacancyTracker`.
#
# For a vacancy that makes uncorrelated jumps of length λ at rate Γ, the number of jumps in
# time t has mean Γt, so ⟨|R(t)|²⟩ = Γ t λ² = 6 D t (three dimensions), i.e. D = Γ λ² / 6
# (Einstein's relation for a random walk). `msd_diffusion` estimates D from the slope of
# the mean-square displacement without using the jump length.

"""
    hop_rate(tracker)

Hop rate of the vacancy, `(rate, σ)`: confirmed hops per unit time between the first and last
frame, with the Poisson error `√n / t_total` for `n` hops. Throws if no hop was recorded, since
the Poisson error of a zero count is undefined.
"""
function hop_rate(tr::VacancyTracker{T}) where {T}
    tr.nframes >= 2 || throw(ArgumentError("pureatomistic: hop_rate needs at least 2 frames"))
    tr.nhops >= 1 ||
        throw(ArgumentError("pureatomistic: no hops were recorded, so the hop rate has no Poisson error"))
    total = tr.t[tr.nframes] - tr.t[1]
    return (rate = tr.nhops / total, σ = sqrt(T(tr.nhops)) / total)
end

"""
    msd_diffusion(tracker; nblocks = 10, maxlag = 0)

Vacancy diffusion coefficient `(D, σ)` from the mean-square displacement `⟨|R(t+τ) - R(t)|²⟩ = 6Dτ`
of the unwrapped vacancy track. The frames split into `nblocks` consecutive blocks; each block
gives `D` as the least-squares slope through the origin of the squared displacements against
the lag `τ`, over every pair of frames in the block at most `maxlag` frames apart (default: a
quarter of the block). `D` is the mean over blocks and `σ` the standard error of that mean.
Units are those of the positions squared per unit of `t`.
"""
function msd_diffusion(tr::VacancyTracker{T}; nblocks::Integer = 10, maxlag::Integer = 0) where {T}
    nblocks >= 2 || throw(ArgumentError(lazy"pureatomistic: msd_diffusion needs nblocks >= 2, got $nblocks"))
    blen = tr.nframes ÷ nblocks
    lag = maxlag > 0 ? Int(maxlag) : blen ÷ 4
    (lag >= 1 && lag < blen) ||
        throw(
        ArgumentError(
            lazy"pureatomistic: msd_diffusion has $(tr.nframes) frames in $nblocks blocks of $blen, too few for a lag of $lag frames"
        )
    )
    nb = 0
    mean = zero(T)
    m2 = zero(T)
    for b in 1:nblocks
        lo = (b - 1) * blen + 1
        hi = lo + blen - 1
        num = zero(T)
        den = zero(T)
        for i in lo:(hi - 1), j in (i + 1):min(i + lag, hi)
            τ = tr.t[j] - tr.t[i]
            d = _sub3(_col(T, tr.R, j), _col(T, tr.R, i))
            num += τ * _dot3(d, d)
            den += τ * τ
        end
        # Welford update of the mean and sum of squared deviations of the block estimates.
        x = num / (6 * den)
        nb += 1
        delta = x - mean
        mean += delta / nb
        m2 += delta * (x - mean)
    end
    return (D = mean, σ = sqrt(m2 / (nb - 1) / nb))
end
