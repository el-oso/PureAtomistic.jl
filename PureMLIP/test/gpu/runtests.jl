# Runs the :gpu test items on the backend named by PUREMLIP_GPU_BACKEND (CPU, Metal, CUDA or AMDGPU):
#
#   PUREMLIP_GPU_BACKEND=Metal julia --check-bounds=yes --project=PureMLIP/test/gpu/metal PureMLIP/test/gpu/runtests.jl
#
# The environment is PureMLIP/test/gpu/<backend in lower case>. The main suite filters these items out.
using TestItemRunner

get(ENV, "PUREMLIP_GPU_BACKEND", "") in ("CPU", "Metal", "CUDA", "AMDGPU") ||
    error("set PUREMLIP_GPU_BACKEND to CPU, Metal, CUDA or AMDGPU")
# KernelAbstractions 0.10's CPU backend hangs at hwloc's topology detection on Apple M-series chips
# with performance levels; hwloc's Darwin component is skipped. The test processes inherit this.
Sys.isapple() && get!(ENV, "HWLOC_COMPONENTS", "-darwin")

TestItemRunner.run_tests(joinpath(@__DIR__, "..", ".."); filter = ti -> :gpu in ti.tags)
