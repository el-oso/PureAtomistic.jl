# The calls whose compiled specializations the StrictMode gates judge. `check.jl` includes this file
# in a child process that loads only the package, so no test fixture adds a specialization of its
# own; the items "test_calls sweep finds no unresolved call" and "every exemption is used and
# reasoned" include it as well.
using QuantumEngine

# Every error type and `InputWarning`, each built from a `String` and from a `LazyString` message,
# rendered into an `IOBuffer` (`showerror` for the errors, `show` for the warning); then the
# element lookups and radius accessors of `Units`.
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
    # The element lookups by symbol (String and SubString) and by Z, and the radius accessors.
    Units = QuantumEngine.Units
    for e in (element("Fe"), element("fe"), element(SubString("xCu", 2, 3)), element(26), element(Int32(6)))
        Units.covalent_radius(e)
        Units.bragg_slater_radius(e)
    end
    Units.bondi_radius(element("C"))
    Units.nelements()
    return take!(io)
end
