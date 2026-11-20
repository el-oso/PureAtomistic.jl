@testitem "TrimCheck: exported functions" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using PureTaylor
            const T64 = Taylor{2, 3, 3, Float64, 10}
            const S64 = Taylor{2, 3, 1, Float64, 4}
            const T32 = Taylor{2, 3, 3, Float32, 10}
            const S32 = Taylor{2, 3, 1, Float32, 4}
        end,
        variable(Val{2}, Val{3}, Val{1}, Float64, Float64),
        variable(Val{2}, Val{3}, Val{2}, Float32, Float32),
        constant(Val{2}, Val{3}, Float64),
        constant(Val{2}, Val{3}, Float32),
        value(T64),
        value(T32),
        value(Float64),
        value(Float32),
        derivatives(T64, NTuple{2, Float64}),
        derivatives(T32, NTuple{2, Float32}),
        PureTaylor.coefficient(T64, NTuple{2, Int}),
        PureTaylor.coefficient(T32, NTuple{2, Int}),
        substitute(T64, Tuple{T64, T64}),
        substitute(T64, Tuple{S64, Float64}),
        substitute(T32, Tuple{T32, T32}),
        substitute(T32, Tuple{S32, Float32}),
        fast_sqrt(Float64), fast_cbrt(Float64), fast_invcbrt(Float64), fast_log(Float64),
        fast_log1p(Float64), fast_exp(Float64), fast_expm1(Float64),
        fast_sqrt(Float32), fast_cbrt(Float32), fast_invcbrt(Float32), fast_log(Float32),
        fast_log1p(Float32), fast_exp(Float32), fast_expm1(Float32),
        fast_sqrt(T64), fast_cbrt(T64), fast_invcbrt(T64), fast_log(T64),
        fast_log1p(T64), fast_exp(T64), fast_expm1(T64),
        fast_sqrt(T32), fast_cbrt(T32), fast_invcbrt(T32), fast_log(T32),
        fast_log1p(T32), fast_exp(T32), fast_expm1(T32),
    )
end
