# Runs PureMLIP's reference and allocation checks with PureBLAS routing the process's BLAS calls.
# PureBLAS requires Julia 1.13, so this environment is separate from the main test environment:
#
#   julia +1.13 --project=PureMLIP/test/pureblas -e 'using Pkg; Pkg.instantiate()'
#   julia +1.13 --project=PureMLIP/test/pureblas PureMLIP/test/pureblas/runtests.jl
#
# The items carry the tag :pureblas, which the main suite filters out.
using TestItemRunner, PureBLAS

PureBLAS.activate()
PureBLAS.is_active() || error("PureBLAS did not become active")

TestItemRunner.run_tests(joinpath(@__DIR__, "..", ".."); filter = ti -> :pureblas in ti.tags)
