# juliac entry point: evaluate `pbe()` with `xc_evaluate!` at order 2 on fixed inputs.
#   xc_eval <out.bin>
# Writes the output fields of the restricted and then the polarized evaluation, each field's
# Float64 values in column-major order, to `out.bin`. Run untrimmed (`julia entry.jl <out.bin>`)
# it writes the same file. A wrong argument count prints a usage line to stderr and exits with 1.
using PureXC

# n points with densities from 0.05 to about 1.6 and σαβ inside its bound.
function inputs(n)
    ρ = [0.05 + 0.025 * j for j in 1:n]
    σ = [0.01 + 0.004 * j for j in 1:n]
    ρp = [i == 1 ? 0.03 + 0.02 * j : 0.02 + 0.005 * j for i in 1:2, j in 1:n]
    σp = [i == 2 ? 0.0 : 0.002 * j for i in 1:3, j in 1:n]
    for j in 1:n
        σp[2, j] = 0.5 * sqrt(σp[1, j] * σp[3, j])
    end
    return ρ, σ, ρp, σp
end

function write_fields(io::IO, out::NamedTuple)
    for a in values(out)
        write(io, a)
    end
    return nothing
end

function (@main)(args::Vector{String})::Cint
    if length(args) != 1
        print(Core.stderr, "usage: xc_eval <out.bin>\n")
        return 1
    end
    f = pbe()
    ρ, σ, ρp, σp = inputs(64)
    out = xc_evaluate!(xc_outputs(f, Val(2), ρ, σ), f, Val(2), ρ, σ)
    outp = xc_evaluate!(xc_outputs(f, Val(2), ρp, σp), f, Val(2), ρp, σp)
    io = open(args[1], "w")
    try
        write_fields(io, out)
        write_fields(io, outp)
    finally
        close(io)
    end
    return 0
end
