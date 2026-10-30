using TestItemRunner
using PureMLIP

# The juliac build is slow; it runs only when PUREMLIP_JULIAC_BUILD=1.
@run_package_tests filter = ti -> (:juliac ∉ ti.tags) || get(ENV, "PUREMLIP_JULIAC_BUILD", "") == "1"
