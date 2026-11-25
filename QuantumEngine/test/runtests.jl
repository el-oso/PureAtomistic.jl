using TestItemRunner
using QuantumEngine

# Items tagged :juliac (slow builds) run only with QUANTUMENGINE_JULIAC_BUILD=1, :oracle items only
# when QUANTUM_ENGINE_ORACLE names the oracle binary, :slow items only with QUANTUMENGINE_SLOW=1,
# :blocked_on_purexc items (waiting for a PureXC functional) only with
# QUANTUMENGINE_PUREXC_BLOCKED=1, and :gpu items never in this suite. QUANTUMENGINE_ONLY=<stem>
# runs only the items of the test file `<stem>.jl`.
const JULIAC = get(ENV, "QUANTUMENGINE_JULIAC_BUILD", "") == "1"
const ORACLE = !isempty(get(ENV, "QUANTUM_ENGINE_ORACLE", ""))
const SLOW = get(ENV, "QUANTUMENGINE_SLOW", "") == "1"
const PUREXC_BLOCKED = get(ENV, "QUANTUMENGINE_PUREXC_BLOCKED", "") == "1"
const ONLY = get(ENV, "QUANTUMENGINE_ONLY", "")

function selected(ti)
    (:juliac in ti.tags) && !JULIAC && return false
    (:oracle in ti.tags) && !ORACLE && return false
    (:slow in ti.tags) && !SLOW && return false
    (:blocked_on_purexc in ti.tags) && !PUREXC_BLOCKED && return false
    (:gpu in ti.tags) && return false
    return isempty(ONLY) || splitext(basename(ti.filename))[1] == ONLY
end

@run_package_tests filter = selected
