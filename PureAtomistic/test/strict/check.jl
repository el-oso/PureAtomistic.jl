# Run in a fresh process by the item "StrictMode policy holds in every module": loads the package,
# runs `workload.jl`, then judges every compiled specialization of each module against the policy
# `(:typestable, :noalloc, :noboxing)`. Type stability is proved by StrictModeTest (JET); `:noalloc`
# and `:noboxing` are the value-free scans that `@strict_policy` itself applies, each judged on its
# own, and the `@test_noalloc` items prove the hot paths with AllocCheck. Functions under
# `@strict_exempt` are skipped for what they exempt.
#
# The module sweeps judge the functions a module owns. The methods it adds to functions it does not
# own (`Base.getindex` of a `TrajectoryReader`) and the methods of its callable structs are judged
# afterwards, each compiled specialization against the guarantees `OTHER_METHODS` lists for it,
# with JET and AllocCheck (`test_signatures`).
#
# Prints one line per finding and a summary line per module:
#   POLICY <module> <declared guarantees>
#   FAIL <guarantee> <module> <function and signature>
#   JUDGED <status> <module> <function and signature>   (the `relax!` findings, for the item)
#   CHECKED <module> <typestable signatures> <noalloc signatures> <noboxing signatures>
#   OTHER <status> <guarantees> <function and signature>
#   CHECKED OTHER <specializations>
using StrictMode, StrictModeTest
StrictMode.assert_enabled()
include(joinpath(@__DIR__, "workload.jl"))

const P = PureAtomistic
# The scan stops at these calls instead of reporting them inside every caller that can reach them:
# the neighbor-list rebuild and the default relaxer's state are workspace creation, exempt in
# their modules, and `read_frame` is the trajectory reader's entry point, which returns a new frame
# (an analysis kernel over a `TrajectoryReader` allocates that frame per access, and nothing else).
register_alloc_barrier!(P.Potentials._regrow!)
register_alloc_barrier!(P.Integrators._default_relaxer)
register_alloc_barrier!(P.IO.read_frame)

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
    npass = Int[]
    for g in (:noalloc, :noboxing)
        fs = audit(M; sweep = true, guarantees = (g,), dispatch_suggest = false)
        for f in fs
            f.status === :fail && println("FAIL ", f.guarantee, " ", name, " ", f.func, f.signature)
            g === :noalloc && occursin("relax!", string(f.func)) &&
                println("JUDGED ", f.status, " ", name, " ", f.func, f.signature)
        end
        push!(npass, count(f -> f.status === :pass, fs))
    end
    println("CHECKED ", name, " ", ntypestable, " ", join(npass, " "))
end

# Every method of the package that the module sweeps cannot see, by function and first argument,
# with the guarantees it is held to. A method missing here fails the item, so a new one is listed
# when it is written.
const TR, TW = P.IO.TrajectoryReader, P.IO.TrajectoryWriter
const OTHER_METHODS = Dict{Tuple{Any, Any}, Tuple{Vararg{Symbol}}}(
    (length, TR) => (:typestable, :noalloc, :noboxing),
    (keys, TR) => (:typestable, :noalloc, :noboxing),
    (firstindex, TR) => (:typestable, :noalloc, :noboxing),
    (lastindex, TR) => (:typestable, :noalloc, :noboxing),
    # Entry point: reads frame k into a new `Frame` (`read_frame`).
    (getindex, TR) => (:typestable,),
    # Entry points: they close or flush the file.
    (close, TR) => (:typestable,),
    (close, TW) => (:typestable,),
    (flush, TW) => (:typestable,),
    (P._Rescale, nothing) => (:typestable, :noalloc, :noboxing),
    (P._IntegratorStep, nothing) => (:typestable, :noalloc, :noboxing),
)

in_package(m::Module) = Base.moduleroot(m) === P

# The key of `OTHER_METHODS` that covers method `m`, `:swept` for a method the module sweeps see,
# or `:skip` for one that no listing can name (keyword sorters, constructors, closures, and the
# methods `@contract` and `@verify` generate in TypeContracts' functions, which the item
# "TypeContracts specs pass the allocation scan" covers).
function other_key(m::Method)
    ft = Base.unwrap_unionall(m.sig).parameters[1]
    (ft === typeof(Core.kwcall) || ft <: Type) && return :skip
    if isdefined(ft, :instance)
        f = ft.instance
        in_package(parentmodule(f)) && return :swept
        nameof(Base.moduleroot(parentmodule(f))) === :TypeContracts && return :skip
        return (f, Base.unwrap_unionall(m.sig).parameters[2])
    end
    occursin(r"^#", string(nameof(Base.unwrap_unionall(ft)))) && return :skip
    return (Base.unwrap_unionall(ft).name.wrapper, nothing)
end

# A callable struct is judged through a function that calls it: StrictMode names what it judges
# by its function, and a struct value has no name.
call_with(f, a) = f(a)
call_with(f, a, b) = f(a, b)

nother = 0
Base.visit(Core.methodtable) do m
    m isa Method && in_package(m.module) || return nothing
    key = other_key(m)
    key in (:skip, :swept) && return nothing
    gs = get(OTHER_METHODS, key, nothing)
    if isnothing(gs)
        println("FAIL unlisted other ", m.sig)
        return nothing
    end
    for mi in Base.specializations(m)
        (mi.specTypes isa DataType && Base.isdispatchtuple(mi.specTypes)) || continue
        tt = mi.specTypes.parameters
        sig = isdefined(tt[1], :instance) ? (tt[1].instance, Tuple(tt[2:end])) : (call_with, Tuple(tt))
        global nother += 1
        try
            test_signatures([sig]; guarantees = gs)
            println("OTHER pass ", join(gs, ","), " ", mi.specTypes)
        catch err
            err isa StrictMode.StrictViolation || rethrow()
            println("FAIL other ", join(gs, ","), " ", mi.specTypes, " | ", replace(sprint(showerror, err), '\n' => " | "))
        end
    end
    return nothing
end
println("CHECKED OTHER ", nother)
