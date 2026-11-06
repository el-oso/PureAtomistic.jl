# juliac entry point: run a molecular-dynamics simulation described by a TOML file.
#   md_run <model.model> <config.toml> <out.traj>
# Writes the binary trajectory `out.traj` and the final frame `out.traj.extxyz`.
using PureAtomistic

# An `ArgumentError` whose message is a `String` (the config reader's errors) is printed and ends
# the run with exit code 1. A lazily interpolated message cannot be rendered in a trimmed
# program, so any other error propagates and aborts the run.
function (@main)(args::Vector{String})::Cint
    if length(args) != 3
        print(Core.stderr, "usage: md_run <model.model> <config.toml> <out.traj>\n")
        return 1
    end
    try
        PureAtomistic.run_md(args[1], args[2], args[3])
    catch err
        msg = err isa ArgumentError ? err.msg : nothing
        msg isa String || rethrow()
        print(Core.stderr, msg, "\n")
        return 1
    end
    return 0
end
