module PureTaylor

using StrictMode: @strict_policy

include("indices.jl")
include("type.jl")
include("arith.jl")
include("univariate.jl")
include("fast.jl")
include("substitute.jl")

export Taylor, variable, constant, value, derivatives
export fast_sqrt, fast_cbrt, fast_invcbrt, fast_log, fast_log1p, fast_exp, fast_expm1
export substitute
public coefficient, power_series

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
