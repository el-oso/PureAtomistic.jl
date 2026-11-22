using TestItemRunner
using PureXC

# The items tagged :exhaustive take tens of minutes; PUREXC_EXHAUSTIVE=1 runs only them, otherwise
# they are skipped.
const EXHAUSTIVE = get(ENV, "PUREXC_EXHAUSTIVE", "") == "1"
@run_package_tests filter = ti -> (:exhaustive in ti.tags) == EXHAUSTIVE
