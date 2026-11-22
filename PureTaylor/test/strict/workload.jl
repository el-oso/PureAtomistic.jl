# The calls whose compiled specializations the StrictMode policy item judges. `check.jl` includes
# this file in a child process that loads only the package, so no test fixture adds a
# specialization of its own.
using PureTaylor
using PureTaylor: coefficient, power_series

# Every exported and public function and every Base method of the package, on the Taylor numbers
# `numbers` (all over the same `N` and `K`, with coefficients of type `T`) and the scalar `x::T`.
# The binary operations take every ordered pair of `numbers`, so that the masks combine.
function exercise(numbers, x::T) where {T}
    for a in numbers, b in numbers
        a + b; a - b; a * b; a / b
    end
    for a in numbers
        exercise_one(a, x)
    end
    return nothing
end

function exercise_one(a::Taylor{N, K}, x::T) where {N, K, T}
    -a; inv(a)
    a + x; x + a; a - x; x - a; a * x; x * a; a / x; x / a
    a + 1; 1 + a; a - 1; 1 - a; 2 * a; a * 2; a / 2; 2 / a
    a^0; a^1; a^2; a^3; a^-1; a^-2
    pow(a, 3); pow(a, -2)
    a^T(0.5); a^T(2); a^(1 // 3)
    a^a; x^a; 2^a; ℯ^a
    sqrt(a); cbrt(a); exp(a); expm1(a); log(a); log1p(a)
    fast_sqrt(a); fast_cbrt(a); fast_invcbrt(a); fast_log(a); fast_log1p(a); fast_exp(a); fast_expm1(a)
    zero(a); one(a); zero(typeof(a)); one(typeof(a)); isfinite(a); isnan(a)
    value(a)
    derivatives(a, ntuple(_ -> inv(x), Val(N)))
    derivatives(a, ntuple(_ -> 1, Val(N)))
    coefficient(a, ntuple(_ -> 0, Val(N))); coefficient(a, ntuple(i -> i == 1 ? 1 : 0, Val(N)))
    power_series(a, T(4) / 3, x); power_series(a, 1 // 3, x)
    substitute(a, ntuple(_ -> a, Val(N)))
    substitute(a, ntuple(i -> isodd(i) ? a : x, Val(N)))
    return nothing
end

# `a^n` with an exponent known only at run time (a literal exponent lowers to `literal_pow`).
pow(a, n::Int) = a^n

# Taylor numbers over `N` variables to degree `K` with the full mask, the mask of variable 1 alone,
# and mask 0, and the scalar kernels in `T`.
function strict_workload(::Val{N}, ::Val{K}, ::Type{T}) where {N, K, T}
    x = T(0.7)
    full = reduce(+, ntuple(i -> variable(Val(N), Val(K), Val(i), x + T(i) / 10, x), Val(N)))
    single = variable(Val(N), Val(K), Val(1), x)
    none = constant(Val(N), Val(K), x)
    exercise((full, single, none), x)
    # A change of variables onto two new variables.
    v = variable(Val(2), Val(K), Val(1), x) + variable(Val(2), Val(K), Val(2), x, T(2))
    substitute(full, ntuple(_ -> v, Val(N)))
    substitute(single, ntuple(i -> isodd(i) ? v : x, Val(N)))
    for f in (fast_sqrt, fast_cbrt, fast_invcbrt, fast_log, fast_log1p, fast_exp, fast_expm1)
        f(x)
    end
    value(x)
    return nothing
end

function run_workload()
    # (3, 8) and (1, 130) are above `_UNROLL_MAX` (L = 165 and 131), so their kernels are the loop
    # forms; (1, 130) also takes the loop form of the recurrence of the power series (K + 1 = 131).
    for (N, K) in ((1, 0), (1, 3), (2, 2), (2, 3), (5, 2), (5, 3), (3, 8)), T in (Float32, Float64)
        strict_workload(Val(N), Val(K), T)
    end
    for T in (Float32, Float64)
        x = T(0.7)
        exercise((variable(Val(1), Val(130), Val(1), x),), x)
    end
    # The fallbacks of the fast kernels for other `Real`s.
    for f in (fast_sqrt, fast_cbrt, fast_invcbrt, fast_log, fast_log1p, fast_exp, fast_expm1)
        f(2)
    end
    return nothing
end

# The failing control: the same calls on Taylor numbers with the abstract coefficient type `Real`,
# which every gate must reject.
function run_control_workload()
    a = Taylor{2, 2, 3, Real, 6}((0.7, 1.0, 0.5, 0.1, 0.2, 0.3))
    exercise((a,), 0.7)
    return nothing
end
