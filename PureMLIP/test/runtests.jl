using TestItemRunner
using PureMLIP

# The juliac build is slow; it runs only when PUREMLIP_JULIAC_BUILD=1. Items tagged :pureblas need
# PureBLAS (Julia 1.13) and run from test/pureblas.
@run_package_tests filter = ti -> (:pureblas ∉ ti.tags) && ((:juliac ∉ ti.tags) || get(ENV, "PUREMLIP_JULIAC_BUILD", "") == "1")
