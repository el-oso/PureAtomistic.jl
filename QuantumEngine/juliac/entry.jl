# juliac entry point.
#   qe_inspect <file.inp>
# Reads the `.inp` file strictly and prints, one per line, its title, the electron count, the α and
# β electron counts and the nuclear repulsion energy (hartree, `fmt_fixed(e, 20, 12)`), and exits
# with 0. An input error, or an `ArgumentError` from the molecule (two atoms at one position)
# shown as an input error of the file, is printed with `showerror` and exits with 1, as does a file
# that cannot be read; any other argument count prints the usage line and exits with 1. Everything
# goes to stdout, and run untrimmed (`julia entry.jl <file.inp>`) it prints the same.
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
        # Computed before any line is printed, so an error prints alone.
        vnn = nuclear_repulsion(mol)
        println(buf, inp.title)
        println(buf, nelectrons(mol))
        println(buf, nalpha(mol))
        println(buf, nbeta(mol))
        println(buf, fmt_fixed(vnn, 20, 12))
    catch e
        # An `ArgumentError` of the molecule is shown as an input error of the file: `showerror` of
        # an `ArgumentError` prints its abstract `msg` through a runtime dispatch, which does not
        # trim, and an `InputError` prints a `String` or `LazyString` message part by part.
        if e isa ArgumentError
            m = e.msg
            if m isa LazyString
                e = InputError(path, 0, m)
            elseif m isa String
                e = InputError(path, 0, m)
            end
        end
        e isa InputError || rethrow()
        showerror(buf, e)
        write(buf, "\n")
        status = Cint(1)
    end
    write(Core.stdout, take!(buf))
    return status
end
