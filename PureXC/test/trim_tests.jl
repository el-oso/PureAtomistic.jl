# Every shipped functional in Float64 (suffix 64) and Float32 (suffix 32): `energy_density`,
# `xc_derivatives` at order 2 and `xc_evaluate!` at order 2 into the outputs of `xc_outputs`,
# restricted (R) and polarized (P).
@testitem "TrimCheck: energy_density, xc_derivatives and xc_evaluate!" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureXC
            for T in (Float64, Float32), (name, build) in ((:LX, lda_x), (:PW, lda_c_pw), (:PM, lda_c_pw_mod), (:BX, pbe_x), (:BC, pbe_c), (:PB, pbe))
                s = T === Float64 ? "64" : "32"
                f = build(T)
                ρ, σ = ones(T, 2), ones(T, 2)
                ρp, σp = ones(T, 2, 2), ones(T, 3, 2)
                gga = PureXC.family(f) === :gga
                outr = gga ? xc_outputs(f, Val(2), ρ, σ) : xc_outputs(f, Val(2), ρ)
                outp = gga ? xc_outputs(f, Val(2), ρp, σp) : xc_outputs(f, Val(2), ρp)
                @eval const $(Symbol(name, s)) = $(typeof(f))
                @eval const $(Symbol(name, "OR", s)) = $(typeof(outr))
                @eval const $(Symbol(name, "OP", s)) = $(typeof(outp))
            end
            const V64, M64 = Vector{Float64}, Matrix{Float64}
            const V32, M32 = Vector{Float32}, Matrix{Float32}
        end,
        energy_density(LX64, Float64), energy_density(LX64, NTuple{2, Float64}),
        energy_density(PW64, Float64), energy_density(PW64, NTuple{2, Float64}),
        energy_density(PM64, Float64), energy_density(PM64, NTuple{2, Float64}),
        energy_density(BX64, Float64, Float64), energy_density(BX64, NTuple{2, Float64}, NTuple{3, Float64}),
        energy_density(BC64, Float64, Float64), energy_density(BC64, NTuple{2, Float64}, NTuple{3, Float64}),
        energy_density(PB64, Float64, Float64), energy_density(PB64, NTuple{2, Float64}, NTuple{3, Float64}),
        energy_density(LX32, Float32), energy_density(LX32, NTuple{2, Float32}),
        energy_density(PW32, Float32), energy_density(PW32, NTuple{2, Float32}),
        energy_density(PM32, Float32), energy_density(PM32, NTuple{2, Float32}),
        energy_density(BX32, Float32, Float32), energy_density(BX32, NTuple{2, Float32}, NTuple{3, Float32}),
        energy_density(BC32, Float32, Float32), energy_density(BC32, NTuple{2, Float32}, NTuple{3, Float32}),
        energy_density(PB32, Float32, Float32), energy_density(PB32, NTuple{2, Float32}, NTuple{3, Float32}),
        xc_derivatives(LX64, Val{2}, Float64), xc_derivatives(LX64, Val{2}, NTuple{2, Float64}),
        xc_derivatives(PW64, Val{2}, Float64), xc_derivatives(PW64, Val{2}, NTuple{2, Float64}),
        xc_derivatives(PM64, Val{2}, Float64), xc_derivatives(PM64, Val{2}, NTuple{2, Float64}),
        xc_derivatives(BX64, Val{2}, Float64, Float64), xc_derivatives(BX64, Val{2}, NTuple{2, Float64}, NTuple{3, Float64}),
        xc_derivatives(BC64, Val{2}, Float64, Float64), xc_derivatives(BC64, Val{2}, NTuple{2, Float64}, NTuple{3, Float64}),
        xc_derivatives(PB64, Val{2}, Float64, Float64), xc_derivatives(PB64, Val{2}, NTuple{2, Float64}, NTuple{3, Float64}),
        xc_derivatives(LX32, Val{2}, Float32), xc_derivatives(LX32, Val{2}, NTuple{2, Float32}),
        xc_derivatives(PW32, Val{2}, Float32), xc_derivatives(PW32, Val{2}, NTuple{2, Float32}),
        xc_derivatives(PM32, Val{2}, Float32), xc_derivatives(PM32, Val{2}, NTuple{2, Float32}),
        xc_derivatives(BX32, Val{2}, Float32, Float32), xc_derivatives(BX32, Val{2}, NTuple{2, Float32}, NTuple{3, Float32}),
        xc_derivatives(BC32, Val{2}, Float32, Float32), xc_derivatives(BC32, Val{2}, NTuple{2, Float32}, NTuple{3, Float32}),
        xc_derivatives(PB32, Val{2}, Float32, Float32), xc_derivatives(PB32, Val{2}, NTuple{2, Float32}, NTuple{3, Float32}),
        xc_evaluate!(LXOR64, LX64, Val{2}, V64), xc_evaluate!(LXOP64, LX64, Val{2}, M64),
        xc_evaluate!(PWOR64, PW64, Val{2}, V64), xc_evaluate!(PWOP64, PW64, Val{2}, M64),
        xc_evaluate!(PMOR64, PM64, Val{2}, V64), xc_evaluate!(PMOP64, PM64, Val{2}, M64),
        xc_evaluate!(BXOR64, BX64, Val{2}, V64, V64), xc_evaluate!(BXOP64, BX64, Val{2}, M64, M64),
        xc_evaluate!(BCOR64, BC64, Val{2}, V64, V64), xc_evaluate!(BCOP64, BC64, Val{2}, M64, M64),
        xc_evaluate!(PBOR64, PB64, Val{2}, V64, V64), xc_evaluate!(PBOP64, PB64, Val{2}, M64, M64),
        xc_evaluate!(LXOR32, LX32, Val{2}, V32), xc_evaluate!(LXOP32, LX32, Val{2}, M32),
        xc_evaluate!(PWOR32, PW32, Val{2}, V32), xc_evaluate!(PWOP32, PW32, Val{2}, M32),
        xc_evaluate!(PMOR32, PM32, Val{2}, V32), xc_evaluate!(PMOP32, PM32, Val{2}, M32),
        xc_evaluate!(BXOR32, BX32, Val{2}, V32, V32), xc_evaluate!(BXOP32, BX32, Val{2}, M32, M32),
        xc_evaluate!(BCOR32, BC32, Val{2}, V32, V32), xc_evaluate!(BCOP32, BC32, Val{2}, M32, M32),
        xc_evaluate!(PBOR32, PB32, Val{2}, V32, V32), xc_evaluate!(PBOP32, PB32, Val{2}, M32, M32),
    )
end
