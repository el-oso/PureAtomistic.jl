module IO

include("extxyz.jl")
include("trajectory.jl")

export Frame, write_extxyz, read_extxyz
export TrajectoryWriter, TrajectoryReader, write_frame, read_frame

end
