module PureTaylor

# Declared dependency; imported here so Aqua sees it used.
import StrictMode

include("indices.jl")
include("type.jl")
include("arith.jl")
include("univariate.jl")

export Taylor, variable, constant, value, derivatives
public coefficient

end
