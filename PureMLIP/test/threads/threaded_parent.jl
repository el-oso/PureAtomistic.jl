# The checks of the output of `threaded_child.jl`, shared by the items that run it with OpenBLAS and
# with PureBLAS. The serial evaluations here use the BLAS of the test process, which must be the
# child's. `lammps` is the LAMMPS-MACE reference `mpa0_f64` as `LammpsReference.load` returns it,
# and `structure` is `LammpsReference.structure`.
using Test, LinearAlgebra, PureMLIP

# The bytes the first threaded evaluation of a new workspace allocates, for `mace_energy_forces!` and
# `mace_energy!`, measured on cu_bench(2) with 4 threads (see docs/src/model.md). With OpenBLAS
# they are the same in both precisions and both entry points, and the same in 3 fresh processes on
# Julia 1.12.7 and 1.13.3 on Apple arm64 (macOS); the exact figures are the gate. With PureBLAS the
# retained scratch of each worker is the same every time, but the total of an evaluation with
# `mace_energy_forces!` varies between processes (Float64, 4 chunks: 23.4 to 37.6 MB), so a
# range gates: the lower bound shows the scratch was allocated, the upper bound is about 1.4 times
# the largest figure seen (a 1.5-fold margin for `mace_energy!`, whose figures did not vary).
const FIRST_CALL_OPENBLAS = Dict(2 => 5664, 4 => 7808, 8 => 12432)
const FIRST_CALL_PUREBLAS_MIN = 512 * 1024
const FIRST_CALL_PUREBLAS_MAX = Dict(
    (Float64, :forces) => 48 * 1024^2, (Float32, :forces) => 24 * 1024^2,
    (Float64, :energy) => 12 * 1024^2, (Float32, :energy) => 6 * 1024^2,
)

function check_threaded_child(lines, ck, lammps, structure; pureblas::Bool = false)
    @test any(==("NTHREADS\t4"), lines)
    rows(tag) = [f for f in (split(l, '\t') for l in lines) if f[1] == tag]
    cases = threaded_cases()
    case_rows = Dict(parse(Int, f[2]) => f for f in rows("CASE"))
    @test sort(collect(keys(case_rows))) == eachindex(cases)
    models = Dict(T => MACEModel{T}(ck) for T in (Float64, Float32))
    for (k, (name, T, nchunks, P, C, Z)) in enumerate(cases)
        f = case_rows[k]
        m = models[T]
        nl = neighbor_list(P, C, m.r_max)
        n = nl.n_atoms
        r = mace_energy_forces!(MACEResult{T}(n), MACEWorkspace{T}(m, n, nl.n_edges; nchunks, elements = Z), m, P, C, Z, nl)
        @testset "$name, $T, $nchunks chunks" begin
            @test frombits(T, f[3]) == r.energy
            @test frombits(T, f[4], n) == r.node_energies
            @test frombits(T, f[5], (3, n)) == r.forces
            @test frombits(T, f[6], (3, 3)) == r.virial
            @test frombits(T, f[7], (3, 3)) == r.stress
            @test frombits(T, f[8]) == r.energy           # mace_energy! with ThreadedChunks()
            @test f[9] == "true"                           # a second threaded evaluation, same bits
            # The workers of the workspace started in the first evaluation; later evaluations
            # hand them their phases through channels and allocate nothing.
            @test parse(Int, f[10]) == 0                   # mace_energy_forces!
            @test parse(Int, f[11]) == 0                   # mace_energy!
            if name == "one atom"
                @test nl.n_edges == 0
                @test r.energy == m.e0[findfirst(==(29), m.z_table)]   # no edges: E0 exactly
                @test r.forces == zeros(T, 3, 1) && r.virial == zeros(T, 3, 3)
            else
                ref = mace_energy_forces!(MACEResult{T}(n), MACEWorkspace{T}(m, n, nl.n_edges; elements = Z), m, P, C, Z, nl)
                # Rounding bound 10·eps(T)·N relative to the largest magnitude of each quantity,
                # with N = 1280 the largest inner dimension of any matrix product in the model.
                # Measured over these cases: at most 12.3·eps(T) in Float64 (35 chunks) and
                # 5.3·eps(T) in Float32.
                tol = 10 * eps(T) * 1280
                @test abs(r.energy - ref.energy) <= tol * abs(ref.energy)
                @test maximum(abs, r.node_energies - ref.node_energies) <= tol * maximum(abs, ref.node_energies)
                @test maximum(abs, r.forces - ref.forces) <= tol * maximum(abs, ref.forces)
                @test maximum(abs, r.virial - ref.virial) <= tol * maximum(abs, ref.virial)
            end
        end
    end
    first_rows = rows("FIRST")
    @test length(first_rows) == 2 * length(FIRST_CALL_CHUNKS)
    for f in first_rows
        T, nchunks = f[2] == "Float64" ? Float64 : Float32, parse(Int, f[3])
        @testset "first evaluation, $T, $nchunks chunks" begin
            for (entry, bytes) in ((:forces, parse(Int, f[4])), (:energy, parse(Int, f[5])))
                if pureblas
                    @test FIRST_CALL_PUREBLAS_MIN <= bytes <= FIRST_CALL_PUREBLAS_MAX[(T, entry)]
                else
                    @test bytes == FIRST_CALL_OPENBLAS[nchunks]
                end
            end
        end
    end
    # The LAMMPS-MACE references, with the targets of "forces and virial match LAMMPS-MACE".
    lammps_rows = rows("LAMMPS")
    @test length(lammps_rows) == length(lammps["structures"])
    for f in lammps_rows
        s = structure(lammps, f[2])
        n = length(s.atomic_numbers)
        @testset "LAMMPS $(f[2])" begin
            @test abs(frombits(Float64, f[3]) - s.energy) <= 1.0e-6
            @test maximum(abs, frombits(Float64, f[4], (3, n)) - s.forces; init = 0.0) <= 1.0e-6
            @test maximum(abs, frombits(Float64, f[5], (3, 3)) - s.virial) <= 1.0e-6
        end
    end
    # Forces against central differences of the threaded energy. Measured margin: the largest
    # |F + ΔE/2h| over these nine components is 5.2e-7 eV/Å
    # (multi_element atom 2, x; the others are below 1e-8); tol_fd is 19 times that.
    fd_rows = rows("FD")
    @test length(fd_rows) == sum(((_, atoms),) -> 3 * length(atoms), FD_ATOMS)
    tol_fd = 1.0e-5
    for f in fd_rows
        F, Ep, Em = frombits(Float64, f[5]), frombits(Float64, f[6]), frombits(Float64, f[7])
        @test abs(F + (Ep - Em) / (2 * FD_STEP)) <= tol_fd
    end
    return nothing
end
