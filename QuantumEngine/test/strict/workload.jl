# The calls whose compiled specializations the StrictMode gates judge. `check.jl` includes this file
# in a child process that loads only the package, so no test fixture adds a specialization of its
# own; the items "test_calls sweep finds no unresolved call" and "every exemption is used and
# reasoned" include it as well.
using QuantumEngine

# Every error type and `InputWarning`, each built from a `String` and from a `LazyString` message,
# rendered into an `IOBuffer` (`showerror` for the errors, `show` for the warning).
function run_workload()
    io = IOBuffer()
    sub = SubString("ab", 1, 1)
    for msg in ("m", lazy"m $(1) $(:s) $(2.5) $('c') $(true) $(sub)")
        for e in (
                InputError("a.inp", 3, msg), InputError("a.inp", 0, msg), BasisError(msg),
                SCFNotConverged(msg), OptimizationFailed(msg), NotImplemented("01 §Errors", msg),
                FunctionalUnavailable(msg), NumericalError(msg),
            )
            showerror(io, e)
        end
        show(io, InputWarning("a.inp", 3, :case, msg))
        show(io, InputWarning("a.inp", 0, :case, msg))
    end
    return take!(io)
end
