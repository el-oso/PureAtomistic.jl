# Run in a fresh process by the item "StrictMode policy holds in PureTaylor": loads the package,
# runs the workload of `workload.jl` (with the argument `control`, the failing control's workload
# instead), then judges every compiled specialization of the package. Each guarantee is judged on
# its own, by two engines: the proof (`test_compiled`, JET for `:typestable` and AllocCheck for
# `:noalloc` and `:noboxing`) and the value-free scan that `@strict_policy` itself applies (`audit`
# with `sweep = true`, for `:noalloc` and `:noboxing`).
#
# The module sweeps judge the functions the package owns. The methods it adds to functions it does
# not own are judged afterwards, each compiled specialization against the guarantees
# `OTHER_METHODS` lists for it, with JET and AllocCheck (`test_signatures`).
#
# Prints one line per finding and a summary line per gate:
#   POLICY <declared guarantees>
#   FAIL <engine> <guarantee> <finding>
#   CHECKED <engine> <guarantee> <judged signatures>
#   FAIL other <guarantee> <specialization> | <finding>
#   FAIL unlisted other <method signature>
#   CHECKED other <guarantee> <specializations>
#   JUDGED other <function>                      (each function of OTHER_METHODS with a specialization)
using StrictMode, StrictModeTest
StrictMode.assert_enabled()
include(joinpath(@__DIR__, "workload.jl"))

const CONTROL = "control" in ARGS
CONTROL ? run_control_workload() : run_workload()

const P = PureTaylor
const GUARANTEES = (:typestable, :noalloc, :noboxing)
oneline(err) = replace(sprint(showerror, err), '\n' => " | ")

println("POLICY ", join(StrictMode.module_policy(P), ","))

for g in GUARANTEES
    n = try
        length(test_compiled(P; guarantees = (g,)))
    catch err
        err isa StrictMode.StrictViolation || rethrow()
        println("FAIL proof ", g, " ", oneline(err))
        0
    end
    println("CHECKED proof ", g, " ", n)
end

for g in (:noalloc, :noboxing)
    fs = audit(P; sweep = true, guarantees = (g,), dispatch_suggest = false)
    for f in fs
        f.status === :fail && println("FAIL scan ", g, " ", f.func, f.signature, " | ", f.reason)
    end
    println("CHECKED scan ", g, " ", count(f -> f.status === :pass, fs))
end

# Every Base function the package adds methods to, with the guarantees its methods are held to. A
# method of a function missing here fails the item, so a new one is listed when it is written.
const OTHER_FUNCTIONS = (
    +, -, *, /, inv, ^, Base.literal_pow, sqrt, cbrt, exp, expm1, log, log1p, zero, one, isfinite, isnan,
)
const OTHER_METHODS = Dict{Any, Tuple{Vararg{Symbol}}}(f => GUARANTEES for f in OTHER_FUNCTIONS)

# The function of method `m` when the package does not own it, `:swept` for a function the
# package owns, or `:skip` for a method that no listing can name (keyword sorters and
# constructors).
function other_function(m::Method)
    ft = Base.unwrap_unionall(m.sig).parameters[1]
    (ft === typeof(Core.kwcall) || ft <: Type) && return :skip
    isdefined(ft, :instance) || return :skip
    f = ft.instance
    return Base.moduleroot(parentmodule(f)) === P ? :swept : f
end

nother = Dict(g => 0 for g in GUARANTEES)
judged = Set{Any}()
Base.visit(Core.methodtable) do m
    m isa Method && Base.moduleroot(m.module) === P || return nothing
    f = other_function(m)
    f in (:skip, :swept) && return nothing
    gs = get(OTHER_METHODS, f, nothing)
    if isnothing(gs)
        println("FAIL unlisted other ", m.sig)
        return nothing
    end
    for mi in Base.specializations(m)
        (mi.specTypes isa DataType && Base.isdispatchtuple(mi.specTypes)) || continue
        sig = (f, Tuple(mi.specTypes.parameters[2:end]))
        push!(judged, f)
        for g in gs
            nother[g] += 1
            try
                test_signatures([sig]; guarantees = (g,))
            catch err
                err isa StrictMode.StrictViolation || rethrow()
                println("FAIL other ", g, " ", mi.specTypes, " | ", oneline(err))
            end
        end
    end
    return nothing
end
for g in GUARANTEES
    println("CHECKED other ", g, " ", nother[g])
end
for f in judged
    println("JUDGED other ", nameof(f))
end
