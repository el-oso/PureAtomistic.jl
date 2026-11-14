module IO

using StrictMode: @strict_exempt, @strict_policy

include("extxyz.jl")
include("trajectory.jl")

export Frame, write_extxyz, read_extxyz
export TrajectoryWriter, TrajectoryReader, write_frame, read_frame

@strict_exempt :noalloc Frame reason = "entry point: frame constructor"
@strict_exempt :noalloc :_ reason = "entry point: keyword body of the parametric constructor `Frame{T}(…; kw…)`"
@strict_exempt :noalloc _dense reason = "entry point: converts the arrays of the `Frame` constructor and of `read_frame`"
@strict_exempt :noalloc read_extxyz reason = "entry point: extended XYZ reader"
@strict_exempt :noalloc _read_frame reason = "entry point: reads one frame for `read_extxyz`"
@strict_exempt :noalloc _comment_pairs reason = "entry point: parses the comment line for `read_extxyz`"
@strict_exempt :noalloc _parse_matrix3 reason = "entry point: parses a 3 × 3 value for `read_extxyz`"
@strict_exempt :noalloc _parse_properties reason = "entry point: parses the column layout for `read_extxyz`"
@strict_exempt :noalloc write_extxyz reason = "entry point: extended XYZ writer"
@strict_exempt :noalloc _write_value reason = "entry point: formats a comment-line value for `write_extxyz`"
@strict_exempt :noalloc _write_array_value reason = "entry point: formats an array value for `write_extxyz`"
@strict_exempt :noalloc _write_numbers reason = "entry point: formats numbers for `write_extxyz`"
@strict_exempt :noalloc read_frame reason = "entry point: trajectory reader, which returns a new frame"
@strict_exempt :noalloc getindex types = (TrajectoryReader, Int) reason = "entry point: `read_frame` by index"

# Last, so that it checks every method defined above.
@strict_policy (:typestable, :noalloc)

end
