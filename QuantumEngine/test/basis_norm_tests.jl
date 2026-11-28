# The contracted self-overlap of every Cartesian component of the shipped shells, renormalized in
# `T` and evaluated in BigFloat from the closed form, against 1.
#
# Bound, tolerance rule (b), with u = eps(T)/2. Every operation of `contraction_renorm` and of the
# final scaling is a correctly rounded `+`, `*`, `/` or `sqrt`, so each contributes at most u
# relative error, and `sqrt` halves the relative error of its argument. The data are the `T` values
# themselves, so their conversion from the file adds nothing to the computation the bound covers.
#
# - The base `b = 2√(α_i α_j)/(α_i + α_j)`: the product (u), the sqrt (u/2 + u), the sum (u) and
#   the division (u) give at most 3.5u; the factor 2 is exact.
# - `t = b^{l+1} · sqrt(b)`, by l multiplications, a sqrt and a final product: the error of b times
#   p = l + 3/2, plus l + 2 roundings, at most (3.5p + l + 2)u = (4.5l + 7.25)u.
# - Each term `c_i c_j t_ij` adds two products, 2u.
# - The naive sum of the n² terms, in `eachindex` order, adds (n² − 1)u relative to Σ|terms|,
#   which is κ·D with κ = Σ|c_i c_j t_ij| / |Σ c_i c_j t_ij|.
# - So the sum D carries at most (n² + 4.5l + 8.25)·u·κ (k_D = n² + 5l + 9 after rounding up, the
#   bound of the STO-3G item in `basis_geometry_tests.jl`).
# - The factor f = 1/sqrt(D) carries half of that, plus u for the sqrt and u for the division;
#   S ∝ f², so S carries the error of D plus 4u. The final scaling c_i·f rounds once per
#   coefficient, which adds 2u to each term, at most 2uκ relative to S.
# - Total, first order: |S − 1| ≤ (n² + 4.5l + 8.25)uκ + 4u + 2uκ ≤ (n² + 4.5l + 14.25)·u·κ, since
#   κ ≥ 1. Rounded up, k = n² + 5l + 15, used as γ_k = k·u/(1 − k·u), and the assertion is
#   |S − 1| ≤ γ_k·κ.
#
# κ is computed in BigFloat from the `T` exponents and raw coefficients that `contraction_renorm`
# reads. In Float64 the items also assert |S − 1| ≤ 1e-12, tolerance rule (a): the
# normalization criterion of the project overview for this step.

@testmodule NormClosedForm begin
    using QuantumEngine
    using QuantumEngine.Basis: parse_gbs, contraction_renorm, primitive_norm, ncart, cartesian_component

    const DIR = joinpath(pkgdir(QuantumEngine), "data", "basis_sets")

    read_gbs(file) = open(io -> parse_gbs(io; source = file), joinpath(DIR, file))

    """
        shells(file, keep) -> Vector{Tuple{String, ShellSpec{Float64}}}

    The `(symbol, shell)` pairs of `file`, in symbol order and then file order, for the shells
    with `keep(shell)`.
    """
    function shells(file, keep)
        data = read_gbs(file)
        out = Tuple{String, eltype(first(values(data)).shells)}[]
        for sym in sort!(collect(keys(data)))
            for spec in data[sym].shells
                keep(spec) && push!(out, (sym, spec))
            end
        end
        return out
    end

    "(2m − 1)!! as a BigInt, with (−1)!! = 1."
    dfact(m) = prod((big(k) for k in 1:2:(2m - 1)); init = big(1))

    """
        check(T, spec) -> Vector{NamedTuple}

    For the shell `spec` with its data converted to `T` and renormalized in `T`, one row per
    Cartesian component: the BigFloat self-overlap `S`, κ, the count `k` and `γ_k`. Call under
    `setprecision(BigFloat, 256)`.
    """
    function check(::Type{T}, spec) where {T}
        e = [T(p.exponent) for p in spec.primitives]
        c = [T(p.coefficient) for p in spec.primitives]
        l = spec.l
        n = length(e)
        f = contraction_renorm(e, c, l)
        cr = c .* f
        # κ from the raw `T` coefficients, with t_ij = b^{l+3/2}.
        sabs = big(0)
        ssum = big(0)
        for i in eachindex(e), j in eachindex(e)
            b = 2 * sqrt(big(e[i]) * big(e[j])) / (big(e[i]) + big(e[j]))
            term = big(c[i]) * big(c[j]) * b^(l + 1) * sqrt(b)
            sabs += abs(term)
            ssum += term
        end
        κ = sabs / abs(ssum)
        u = big(eps(T)) / 2
        # Rule (b): k = n² + 5l + 15 roundings, derived in the file header.
        k = n^2 + 5l + 15
        γ = k * u / (1 - k * u)
        # The component-independent part of each pair, c_i c_j (π/p)^{3/2} / (2p)^l, p = α_i + α_j.
        pair = Matrix{BigFloat}(undef, n, n)
        for i in eachindex(e), j in eachindex(e)
            p = big(e[i]) + big(e[j])
            x = big(π) / p
            pair[i, j] = big(cr[i]) * big(cr[j]) * x * sqrt(x) / (2p)^l
        end
        rows = NamedTuple{(:S, :κ, :k, :γ), Tuple{BigFloat, BigFloat, Int, BigFloat}}[]
        for m in 1:ncart(l)
            lx, ly, lz = cartesian_component(l, m)
            N = [primitive_norm(big(a), lx, ly, lz) for a in e]
            df = dfact(lx) * dfact(ly) * dfact(lz)
            S = big(0)
            for i in eachindex(e), j in eachindex(e)
                S += N[i] * N[j] * pair[i, j] * df
            end
            push!(rows, (; S, κ, k, γ))
        end
        return rows
    end

    """
        run_checks(test, files) -> nothing

    Calls `test(T, file, sym, spec, row)` for every row of `check` over `T ∈ (Float64, Float32)` and
    the `(file, keep)` pairs of `files`.
    """
    function run_checks(test, files)
        setprecision(BigFloat, 256) do
            for T in (Float64, Float32), (file, keep) in files, (sym, spec) in shells(file, keep)
                for row in check(T, spec)
                    test(T, file, sym, spec, row)
                end
            end
        end
        return nothing
    end
end

@testitem "normalization (closed form)" setup = [NormClosedForm] begin
    using .NormClosedForm: shells, run_checks
    # def2-tzvpp's contracted g shells: 14, two primitives each, Ce–Lu. The g and h shells of
    # cc-pvqz each have one primitive.
    tz_g(s) = s.l == 4 && length(s.primitives) >= 2
    @test length(shells("def2-tzvpp.gbs", tz_g)) == 14
    @test all(s -> length(s[2].primitives) == 2, shells("def2-tzvpp.gbs", tz_g))
    files = (
        ("sto-3g.gbs", s -> s.l <= 5),
        ("6-31g.gbs", s -> s.l <= 5),
        ("6-31g_star.gbs", s -> s.l <= 5),
        ("cc-pvtz.gbs", s -> s.l <= 5),
        ("cc-pvqz.gbs", s -> s.l == 4),
        ("def2-tzvpp.gbs", tz_g),
    )
    run_checks(files) do T, file, sym, spec, row
        # Rule (b): γ_k·κ, k = n² + 5l + 15, u = eps(T)/2 (file header).
        @test abs(row.S - 1) <= row.γ * row.κ
        # Rule (a): the 1e-12 normalization criterion of the project overview.
        T === Float64 && @test abs(row.S - 1) <= 1.0e-12
    end
end

@testitem "basis: normalization at high κ" tags = [:slow] setup = [NormClosedForm] begin
    using .NormClosedForm: run_checks
    files = (("def2-qzvpp.gbs", s -> s.l <= 5), ("ano-rcc.gbs", s -> s.l <= 5))
    run_checks(files) do T, file, sym, spec, row
        # Rule (b): γ_k·κ, k = n² + 5l + 15, u = eps(T)/2 (file header).
        @test abs(row.S - 1) <= row.γ * row.κ
        # Rule (a): the 1e-12 normalization criterion of the project overview.
        T === Float64 && @test abs(row.S - 1) <= 1.0e-12
    end
end
