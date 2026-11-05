# The cases of the items "forces: threaded chunks give the serial chunk result" and "PureBLAS:
# threaded chunks give the serial chunk result", and the child process that evaluates them with
# `ThreadedChunks()`. The test process runs one thread, so the items start this file as a script
# under `julia -t 4,0`; they include it too, for the cases.
#
#   julia -t 4,0 --project=<test environment> threaded_child.jl <checkpoint> [pureblas]
#
# With `pureblas`, the child activates PureBLAS first. For each case the child prints the bits of
# `mace_energy_forces!(…, ThreadedChunks())` and of the energy of `mace_energy!(…, ThreadedChunks())`,
# whether a second threaded evaluation gave the same bits, and for each entry point the fewest
# bytes one of five further evaluations allocated. Then, in Float64, it prints the threaded
# results on the LAMMPS reference structures and the threaded forces and energies of the
# finite-difference checks.
using PureMLIP, LinearAlgebra

include(joinpath(@__DIR__, "..", "reference", "structures.jl"))

# (name, T, nchunks, positions, cell, atomic_numbers). cu_bench(2) has 32 atoms in one receiver
# block: 1 chunk runs no task, 2 chunks are fewer than the 4 threads, 35 are more than the atoms.
# The mixed structure has three elements and a vacancy, so its chunks differ in species and edge
# count. The last case is one atom without edges.
function threaded_cases()
    cu = cu_bench(2)
    P, C = Matrix(cu.positions'), Matrix(cu.cell')
    keep = [i for i in axes(P, 2) if i != 5]
    mixed = (P[:, keep], C, [(29, 28, 13)[mod1(i, 3)] for i in eachindex(keep)])
    atom = (zeros(3, 1), Matrix(20.0I, 3, 3), [29])
    cases = []
    for T in (Float64, Float32)
        conv((P, C, Z)) = (Matrix{T}(P), Matrix{T}(C), Z)
        for nchunks in (1, 2, 3, 8, 35)
            push!(cases, ("cu_bench(2)", T, nchunks, conv((P, C, cu.atomic_numbers))...))
        end
        push!(cases, ("mixed", T, 3, conv(mixed)...))
        push!(cases, ("one atom", T, 4, conv(atom)...))
    end
    return cases
end

# The receiver chunks of the threaded evaluations of the LAMMPS reference structures and of the
# finite-difference checks.
const REFERENCE_CHUNKS = 3

# The finite-difference checks: (structure, atoms), every Cartesian component of each atom, and the
# step. Atom 2 of multi_element is H, where the ZBL pair term is active.
const FD_ATOMS = (("cu_vacancy", (1, 7)), ("multi_element", (2,)))
const FD_STEP = 1.0e-4

hexbits(x::AbstractFloat) = string(reinterpret(Unsigned, x); base = 16)
hexbits(A::AbstractArray) = join(map(hexbits, vec(A)), ",")
frombits(::Type{T}, s::AbstractString) where {T} = reinterpret(T, parse(Base.uinttype(T), s; base = 16))
frombits(::Type{T}, s::AbstractString, dims) where {T} =
    reshape(T[frombits(T, x) for x in split(s, ',') if !isempty(x)], dims)

# A threaded evaluation on a workspace of its own, whose workers are stopped after it.
function threaded_forces(m::MACEModel{T}, P, C, Z, nchunks) where {T}
    nl = neighbor_list(P, C, m.r_max)
    n = nl.n_atoms
    ws = MACEWorkspace{T}(m, n, nl.n_edges; nchunks, elements = Z)
    r = mace_energy_forces!(MACEResult{T}(n), ws, m, P, C, Z, nl, ThreadedChunks())
    close(ws)
    return r
end

function child_main(checkpoint, blas)
    BLAS.set_num_threads(1)
    if blas == "pureblas"
        pb = Base.require(Main, :PureBLAS)
        Base.invokelatest(pb.activate)
        Base.invokelatest(pb.is_active) || error("PureBLAS did not become active")
        BLAS.set_num_threads(1)
        println("PUREBLAS_THREADS\t", Base.invokelatest(pb.get_num_threads))
    end
    println("NTHREADS\t", Threads.nthreads())
    ck = load_mace_checkpoint(checkpoint)
    models = Dict(T => MACEModel{T}(ck) for T in (Float64, Float32))
    for (k, (_, T, nchunks, P, C, Z)) in enumerate(threaded_cases())
        m = models[T]
        nl = neighbor_list(P, C, m.r_max)
        n = nl.n_atoms
        ws = MACEWorkspace{T}(m, n, nl.n_edges; nchunks, elements = Z)
        r = mace_energy_forces!(MACEResult{T}(n), ws, m, P, C, Z, nl, ThreadedChunks())
        r2 = mace_energy_forces!(MACEResult{T}(n), ws, m, P, C, Z, nl, ThreadedChunks())
        same = all(f -> isequal(getfield(r, f), getfield(r2, f)), fieldnames(MACEResult))
        # The runtime occasionally adds an entry to its method cache when a thread first runs a
        # function in a task (up to 192 bytes seen); the least of five evaluations leaves it out.
        bytes = minimum(_ -> @allocated(mace_energy_forces!(r2, ws, m, P, C, Z, nl, ThreadedChunks())), 1:5)
        re = mace_energy!(MACEResult{T}(n), ws, m, P, C, Z, nl, ThreadedChunks())
        bytes_energy = minimum(_ -> @allocated(mace_energy!(re, ws, m, P, C, Z, nl, ThreadedChunks())), 1:5)
        close(ws)
        println(
            "CASE\t", k, "\t", hexbits(r.energy), "\t", hexbits(r.node_energies), "\t", hexbits(r.forces), "\t",
            hexbits(r.virial), "\t", hexbits(r.stress), "\t", hexbits(re.energy), "\t", same, "\t", bytes, "\t", bytes_energy
        )
    end
    m = models[Float64]
    structures = reference_structures()
    for s in structures
        r = threaded_forces(m, Matrix(s.positions'), Matrix(s.cell'), s.atomic_numbers, REFERENCE_CHUNKS)
        println("LAMMPS\t", s.name, "\t", hexbits(r.energy), "\t", hexbits(r.forces), "\t", hexbits(r.virial))
    end
    for (name, atoms) in FD_ATOMS
        s = structures[findfirst(s -> s.name == name, structures)]
        P, C, Z = Matrix(s.positions'), Matrix(s.cell'), s.atomic_numbers
        F = threaded_forces(m, P, C, Z, REFERENCE_CHUNKS).forces
        for a in atoms, d in 1:3
            Pp = copy(P); Pp[d, a] += FD_STEP
            Pm = copy(P); Pm[d, a] -= FD_STEP
            Ep = threaded_forces(m, Pp, C, Z, REFERENCE_CHUNKS).energy
            Em = threaded_forces(m, Pm, C, Z, REFERENCE_CHUNKS).energy
            println("FD\t", name, "\t", a, "\t", d, "\t", hexbits(F[d, a]), "\t", hexbits(Ep), "\t", hexbits(Em))
        end
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    child_main(ARGS[1], get(ARGS, 2, "openblas"))
end
