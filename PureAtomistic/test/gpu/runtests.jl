# Runs the :gpu test items on the backend named by PUREMLIP_GPU_BACKEND (CPU, Metal, CUDA or AMDGPU):
#
#   PUREMLIP_GPU_BACKEND=Metal julia --check-bounds=yes --project=PureAtomistic/test/gpu/metal PureAtomistic/test/gpu/runtests.jl
#
# The environment is PureAtomistic/test/gpu/<backend in lower case>. The main suite filters these items out.
using TestItemRunner

get(ENV, "PUREMLIP_GPU_BACKEND", "") in ("CPU", "Metal", "CUDA", "AMDGPU") ||
    error("set PUREMLIP_GPU_BACKEND to CPU, Metal, CUDA or AMDGPU")
# hwloc's Darwin component is skipped on macOS; see the KernelAbstractions rule in AGENTS.md.
Sys.isapple() && get!(ENV, "HWLOC_COMPONENTS", "-darwin")

TestItemRunner.run_tests(joinpath(@__DIR__, "..", ".."); filter = ti -> :gpu in ti.tags)
