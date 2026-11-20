using TestItemRunner
using PureTaylor

# The items tagged :exhaustive take minutes; PURETAYLOR_EXHAUSTIVE=1 runs only them, otherwise they
# are skipped.
const EXHAUSTIVE = get(ENV, "PURETAYLOR_EXHAUSTIVE", "") == "1"
@run_package_tests filter = ti -> (:exhaustive in ti.tags) == EXHAUSTIVE
