# juliac entry point: run a molecular-dynamics simulation described by a TOML file.
#   md_run <model.model> <config.toml> <out.traj>
# Writes the binary trajectory `out.traj` and the final frame `out.traj.extxyz`.
# Every failure prints one message to stderr and exits with code 1.
using PureAtomistic

# Writes `x` to stderr through the runtime's own printer, which needs no method dispatch on the
# value's type and so is the one way to show an arbitrary object in a trimmed program.
static_show(x) = ccall(:jl_static_show, Csize_t, (Ptr{Cvoid}, Any), Core.io_pointer(Core.stderr), x)

# Prints an exception message. The pieces of a `LazyString` are printed one by one, `String`
# pieces directly and any other piece through `static_show`: printing or materializing the
# `LazyString` itself is not resolvable in a trimmed program. The argument is a small union
# rather than `Any` because a call with an abstractly typed argument does not trim.
function print_message(m::Union{String, LazyString})
    if m isa String
        print(Core.stderr, m)
    else
        for p in m.parts
            p isa String ? print(Core.stderr, p) : static_show(p)
        end
    end
    return nothing
end

function (@main)(args::Vector{String})::Cint
    if length(args) != 3
        print(Core.stderr, "usage: md_run <model.model> <config.toml> <out.traj>\n")
        return 1
    end
    try
        PureAtomistic.run_md(args[1], args[2], args[3])
    catch err
        if err isa ArgumentError
            print_message(err.msg::Union{String, LazyString})
        elseif err isa ErrorException
            print_message(err.msg::Union{String, LazyString})
        elseif err isa DimensionMismatch
            print_message(err.msg::Union{String, LazyString})
        else
            static_show(err)
        end
        print(Core.stderr, "\n")
        return 1
    end
    return 0
end
