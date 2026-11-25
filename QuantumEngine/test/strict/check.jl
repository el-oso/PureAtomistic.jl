# Run in a fresh process by the items "StrictMode policy holds in QuantumEngine" (no argument) and
# "StrictMode gates reject the failing control" (argument `control`): loads the package, runs the
# workload of `workload.jl` (with `control`, the failing control's workload as well: a method of an
# unlisted Base function, `abs2`, and a type-unstable, allocating and boxed function and
# `showerror` method of `Errors`), then judges every compiled specialization of the package. Each
# guarantee is judged on its own, by two engines: the proof (`test_compiled`, JET for
# `:typestable` and AllocCheck for `:noalloc` and `:noboxing`) and the value-free scan that
# `@strict_policy` itself applies (`audit` with `sweep = true`, for `:noalloc` and `:noboxing`).
#
# The module sweeps judge the functions the package owns, one submodule at a time; the top module
# owns no functions and is not swept. The methods the package adds to functions it does not own are
# judged afterwards, each compiled specialization against the guarantees `OTHER_METHODS` lists for
# it, with JET (`test_signatures`).
#
# Prints one line per finding and a summary line per gate, summed over the swept modules. Each
# finding's text carries the guarantee and code the finding itself reports, as
# `[✗ <guarantee> <code>]`:
#   POLICY <module> <declared guarantees>
#   FAIL <engine> <guarantee> <finding>
#   CHECKED <engine> <guarantee> <signatures judged and passing; exempt ones are not counted>
#   FAIL other <guarantee> <specialization> | <finding>
#   FAIL unlisted other <method signature>
#   CHECKED other <guarantee> <specializations>
#   JUDGED other <function>                      (each function of OTHER_METHODS with a specialization)
using StrictMode, StrictModeTest
StrictMode.assert_enabled()
include(joinpath(@__DIR__, "workload.jl"))

const CONTROL = "control" in ARGS
const P = QuantumEngine
const MODULES = (P.Errors, P.Units, P.System, P.IO)
const GUARANTEES = (:typestable, :noalloc, :noboxing)
oneline(err) = replace(sprint(showerror, err), '\n' => " | ")

# The failing control of the listing: a method of a Base function that `OTHER_METHODS` does not
# list, on a type of the package.
# The failing control of the gates: a struct of `Errors` with an abstractly typed field, a function
# on it that is type-unstable, allocates and boxes, and a `showerror` method that is type-unstable.
if CONTROL
    @eval P.Errors struct _ControlReal
        x::Float64
    end
    @eval P.Errors Base.abs2(c::_ControlReal) = c.x * c.x
    @eval P.Errors struct _Control <: Exception
        x::Real
    end
    @eval P.Errors _control_value(c::_Control) = [c.x][1] * c.x
    @eval P.Errors Base.showerror(io::Base.IO, c::_Control) = print(io, c.x * c.x)
end

run_workload()
if CONTROL
    Base.invokelatest(abs2, Base.invokelatest(P.Errors._ControlReal, 2.0))
    c = Base.invokelatest(P.Errors._Control, 2.0)
    Base.invokelatest(P.Errors._control_value, c)
    Base.invokelatest(showerror, IOBuffer(), c)
end

for M in MODULES
    println("POLICY ", nameof(M), " ", join(StrictMode.module_policy(M), ","))
end

for g in GUARANTEES
    n = 0
    for M in MODULES
        n += try
            count(f -> f.status === :pass, test_compiled(M; guarantees = (g,)))
        catch err
            err isa StrictMode.StrictViolation || rethrow()
            println("FAIL proof ", g, " ", oneline(err))
            0
        end
    end
    println("CHECKED proof ", g, " ", n)
end

for g in (:noalloc, :noboxing)
    n = 0
    for M in MODULES
        fs = audit(M; sweep = true, guarantees = (g,), dispatch_suggest = false)
        for f in fs
            f.status === :fail && println("FAIL scan ", g, " ", f.func, f.signature, " | [✗ ", f.guarantee, " ", f.code, "] ", f.reason)
        end
        n += count(f -> f.status === :pass, fs)
    end
    println("CHECKED scan ", g, " ", n)
end

# Every function of another package the package adds methods to, with the guarantees its methods
# are held to here. A method of a function missing here fails the item, so a new one is listed when
# it is written. The module sweeps already hold `showerror` and `show` to every guarantee, with
# `:noalloc` and `:noboxing` exempt; this listing judges their type stability a second time.
const OTHER_FUNCTIONS = (Base.showerror, Base.show)
const OTHER_GUARANTEES = (:typestable,)
const OTHER_METHODS = Dict{Any, Tuple{Vararg{Symbol}}}(f => OTHER_GUARANTEES for f in OTHER_FUNCTIONS)

# Methods that `@enum` writes into the defining module (`Base.Enums.namemap`,
# `Base.Enums._enum_hash`, `instances`, `typemin`, `typemax`); no listing names them.
const ENUM_FUNCTIONS = (Base.Enums.namemap, Base.Enums._enum_hash, instances, typemin, typemax)

# The function of method `m` when the package does not own it, `:swept` for a function the
# package owns, or `:skip` for a method that no listing can name (keyword sorters, constructors
# and the methods `@enum` writes).
function other_function(m::Method)
    ft = Base.unwrap_unionall(m.sig).parameters[1]
    (ft === typeof(Core.kwcall) || ft <: Type) && return :skip
    isdefined(ft, :instance) || return :skip
    f = ft.instance
    f in ENUM_FUNCTIONS && return :skip
    return Base.moduleroot(parentmodule(f)) === P ? :swept : f
end

nother = Dict(g => 0 for g in OTHER_GUARANTEES)
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
for g in OTHER_GUARANTEES
    println("CHECKED other ", g, " ", nother[g])
end
for f in judged
    println("JUDGED other ", nameof(f))
end
