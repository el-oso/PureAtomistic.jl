# Runs the :gpu test items on the backend named by PUREMLIP_GPU_BACKEND (CPU, Metal, CUDA or AMDGPU):
#
#   PUREMLIP_GPU_BACKEND=Metal julia --check-bounds=yes --project=PureMLIP/test/gpu/metal PureMLIP/test/gpu/runtests.jl
#
# The environment is PureMLIP/test/gpu/<backend in lower case>. The main suite filters these items out.
using TestItemRunner

get(ENV, "PUREMLIP_GPU_BACKEND", "") in ("CPU", "Metal", "CUDA", "AMDGPU") ||
    error("set PUREMLIP_GPU_BACKEND to CPU, Metal, CUDA or AMDGPU")
# hwloc's Darwin component is skipped on macOS; see the KernelAbstractions rule in AGENTS.md.
Sys.isapple() && get!(ENV, "HWLOC_COMPONENTS", "-darwin")
# JET (through StrictModeTest) loads Revise. In its automatic mode Revise polls for a REPL every
# 50 ms for a second after it loads, and each poll allocates a timer (112 bytes) on the thread that
# runs the items, which the host-byte gates then count when they measure within that second. The
# items need no REPL integration.
ENV["JULIA_REVISE"] = "manual"

TestItemRunner.run_tests(joinpath(@__DIR__, "..", ".."); filter = ti -> :gpu in ti.tags)
