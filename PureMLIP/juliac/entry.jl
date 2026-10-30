# juliac entry point: print a summary of a MACE checkpoint.
#   mace_inspect <path to .model>
using PureMLIP

function print_summary(ck::MACECheckpoint)
    h = ck.hyperparameters
    nparams = 0
    for name in ck.state.names
        haskey(ck.state.f64, name) || continue
        x = ck.state.f64[name]
        x.requires_grad && (nparams += length(x.data))
    end
    print(Core.stdout, "model_class=", h.model_class, "\n")
    print(Core.stdout, "r_max=", string(h.r_max), "\n")
    print(Core.stdout, "num_interactions=", string(h.num_interactions), "\n")
    print(Core.stdout, "state_tensors=", string(length(ck.state.names)), "\n")
    print(Core.stdout, "num_parameters=", string(nparams), "\n")
    return nothing
end

function (@main)(args::Vector{String})::Cint
    if length(args) != 1
        print(Core.stderr, "usage: mace_inspect <path to .model checkpoint>\n")
        return 1
    end
    print_summary(load_mace_checkpoint(args[1]))
    return 0
end
