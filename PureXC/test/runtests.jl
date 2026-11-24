using TestItemRunner
using PureXC

# The items tagged :exhaustive take tens of minutes; PUREXC_EXHAUSTIVE=1 runs only them, otherwise
# they are skipped. The juliac build is slow; it runs only when PUREXC_JULIAC_BUILD=1.
const EXHAUSTIVE = get(ENV, "PUREXC_EXHAUSTIVE", "") == "1"
const JULIAC = get(ENV, "PUREXC_JULIAC_BUILD", "") == "1"
@run_package_tests filter = ti -> (:exhaustive in ti.tags) == EXHAUSTIVE && ((:juliac ∉ ti.tags) || JULIAC)
