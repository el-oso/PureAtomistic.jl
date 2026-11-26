# juliac entry point.
#   qe_inspect <file.inp>
# Reads the `.inp` file strictly and prints, one per line, its title, the electron count, the α and
# β electron counts and the nuclear repulsion energy (hartree, `fmt_fixed(e, 20, 12)`), and exits
# with 0. An input error is printed with `showerror` and exits with 1, as does a file that cannot be
# read; any other argument count prints the usage line and exits with 1. Everything goes to stdout,
# and run untrimmed (`julia entry.jl <file.inp>`) it prints the same.
using QuantumEngine
using QuantumEngine.IO: fmt_fixed

function (@main)(args::Vector{String})::Cint
    if length(args) != 1
        print(Core.stdout, "usage: qe_inspect <file.inp>\n")
        return 1
    end
    path = args[1]
    if !isfile(path)
        print(Core.stdout, "qe_inspect: cannot read ")
        print(Core.stdout, path)
        print(Core.stdout, "\n")
        return 1
    end
    buf = IOBuffer()
    status = Cint(0)
    try
        inp = parse_inp(read(path, String); source = path).input
        mol = molecule(inp)
        println(buf, inp.title)
        println(buf, nelectrons(mol))
        println(buf, nalpha(mol))
        println(buf, nbeta(mol))
        println(buf, fmt_fixed(nuclear_repulsion(mol), 20, 12))
    catch e
        e isa InputError || rethrow()
        showerror(buf, e)
        write(buf, "\n")
        status = Cint(1)
    end
    write(Core.stdout, take!(buf))
    return status
end
