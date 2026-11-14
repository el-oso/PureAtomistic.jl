# Run in a fresh process by the item "StrictMode policy holds in every module": loads the package,
# runs `workload.jl`, then judges every compiled specialization of each module against the policy
# `(:typestable, :noalloc)`. Type stability is proved by StrictModeTest (JET); `:noalloc` is the
# value-free scan that `@strict_policy` itself applies, and the `@test_noalloc` items prove the
# hot paths with AllocCheck. Functions under `@strict_exempt` are skipped for what they exempt.
#
# Prints one line per finding and a summary line per module:
#   POLICY <module> <declared guarantees>
#   FAIL <guarantee> <module> <function and signature>
#   CHECKED <module> <typestable signatures> <noalloc signatures>
using StrictMode, StrictModeTest
StrictMode.assert_enabled()
include(joinpath(@__DIR__, "workload.jl"))

const P = PureAtomistic
# The neighbor-list rebuild is a workspace path (exempt in Potentials); the scan stops there
# instead of reporting it inside every `prepare!` that can reach it.
register_alloc_barrier!(P.Potentials._regrow!)

for M in (P, P.Systems, P.Potentials, P.Integrators, P.Thermostats, P.Barostats, P.IO, P.Analysis)
    name = nameof(M)
    println("POLICY ", name, " ", join(StrictMode.module_policy(M), ","))
    ntypestable = try
        length(test_compiled(M; guarantees = (:typestable,)))
    catch err
        err isa StrictMode.StrictViolation || rethrow()
        println("FAIL typestable ", name, " ", replace(sprint(showerror, err), '\n' => " | "))
        0
    end
    fs = audit(M; sweep = true, guarantees = (:noalloc,), dispatch_suggest = false)
    for f in fs
        f.status === :fail && println("FAIL noalloc ", name, " ", f.func, f.signature)
    end
    println("CHECKED ", name, " ", ntypestable, " ", count(f -> f.status === :pass, fs))
end
