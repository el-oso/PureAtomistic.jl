using TestItemRunner
using PureAtomistic

# The juliac build is slow; it runs only when PUREATOMISTIC_JULIAC_BUILD=1.
@run_package_tests filter = ti -> (:juliac ∉ ti.tags) || get(ENV, "PUREATOMISTIC_JULIAC_BUILD", "") == "1"
