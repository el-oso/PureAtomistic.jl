module PureTaylor

using StrictMode: @strict_exempt, @strict_policy

include("indices.jl")
include("loops.jl")
include("type.jl")
include("arith.jl")
include("univariate.jl")
include("fast.jl")
include("substitute.jl")

export Taylor, variable, constant, value, derivatives
export fast_sqrt, fast_cbrt, fast_invcbrt, fast_log, fast_log1p, fast_exp, fast_expm1
export substitute
public coefficient, power_series

# Code generation: these build the bodies and tables of generated methods, so they run when a
# method is generated, never when a kernel is called.
for f in (
        :_addsub_unrolled_body, :_mul_unrolled_body, :_literal_pow_unrolled_body, :_compose_unrolled_body,
        :_recurrence_unrolled_body, :_coefficient_unrolled_body, :_derivatives_unrolled_body,
        :_scale_from_unrolled_body, :_div_unrolled_body, :_div_type_error, :_substitute_body, :_substitute_plan, :_substitute_masks, :_int32_tuple,
    )
    @eval @strict_exempt $f reason = "code generation: runs when a method is generated"
end
# The stack arrays of the loop kernels (see `loops.jl`); their `:noalloc` is proved by AllocCheck on
# the kernels that call them.
@strict_exempt :noalloc _loop_tuple reason = "stack array: a Ref written through a pointer"
@strict_exempt :noalloc _loop_recurrence reason = "stack array: a Ref written through a pointer"
@strict_exempt :noalloc _with_scratch reason = "stack array: a Ref written through a pointer"

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
