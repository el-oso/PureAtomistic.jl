# juliac entry point.
#   qe_inspect <file.inp>
# Prints `ok` for one argument. Any other argument count prints the usage line and exits with 1.
# Both go to stdout, and run untrimmed (`julia entry.jl <file.inp>`) it prints the same.
using QuantumEngine

function (@main)(args::Vector{String})::Cint
    if length(args) != 1
        print(Core.stdout, "usage: qe_inspect <file.inp>\n")
        return 1
    end
    print(Core.stdout, "ok\n")
    return 0
end
