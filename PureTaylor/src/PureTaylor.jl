module PureTaylor

# Declared dependency; imported here so Aqua sees it used.
import StrictMode

include("indices.jl")
include("type.jl")
include("arith.jl")
include("univariate.jl")
include("fast.jl")

export Taylor, variable, constant, value, derivatives
export fast_sqrt, fast_cbrt, fast_invcbrt, fast_log, fast_log1p, fast_exp, fast_expm1
public coefficient

end
