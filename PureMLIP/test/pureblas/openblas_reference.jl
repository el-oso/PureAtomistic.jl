# Child process of pureblas_tests.jl: evaluates the generated reference structures with Julia's
# default BLAS (PureBLAS is never loaded here) and serializes energy, forces and virial.
#
#   julia --project=<env> openblas_reference.jl <checkpoint> <output>
using PureMLIP, LinearAlgebra, Serialization

include(joinpath(@__DIR__, "..", "reference", "structures.jl"))

function main(checkpoint, output)
    libs = [basename(l.libname) for l in BLAS.get_config().loaded_libs]
    any(occursin("openblas", lowercase(l)) for l in libs) || error("expected OpenBLAS, found ", libs)
    ck = load_mace_checkpoint(checkpoint)
    out = Dict{Tuple{DataType, String}, Any}()
    for T in (Float64, Float32)
        m = MACEModel{T}(ck)
        for g in reference_structures()
            P, C = T.(permutedims(g.positions)), T.(permutedims(g.cell))
            r = mace_energy_forces(m, P, C, g.atomic_numbers)
            out[(T, g.name)] = (; energy = r.energy, forces = r.forces, virial = r.virial)
        end
    end
    serialize(output, out)
    return
end

main(ARGS[1], ARGS[2])
