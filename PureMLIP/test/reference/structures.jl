using LinearAlgebra

# Deterministic reference structures. `cell` rows are lattice vectors in LAMMPS'
# restricted triclinic form (a along x, b in the xy plane); `positions` has one row
# per atom.

const FCC_BASIS = [0 0 0; 0.5 0.5 0; 0.5 0 0.5; 0 0.5 0.5]

# Fractional-unit fcc sites of an N×N×N block of cubic cells, i slowest.
function fcc_sites(a, N)
    sites = Matrix{Float64}(undef, 4N^3, 3)
    n = 0
    for i in 0:(N - 1), j in 0:(N - 1), k in 0:(N - 1), b in 1:4
        n += 1
        sites[n, :] = a .* ([i, j, k] .+ FCC_BASIS[b, :])
    end
    return sites
end

# Displacement amplitude·sin(freq·(3n+d) + phase) in Cartesian component d = 1..3 of atom n.
function displace!(pos, amplitude, freq, phase = 0.0)
    for n in axes(pos, 1), d in 1:3
        pos[n, d] += amplitude * sin(freq * (3n + d) + phase)
    end
    return pos
end

function cu_bench(N::Integer)
    a = 3.615
    pos = displace!(fcc_sites(a, N), 0.05, 1.3)
    return (; cell = Matrix(a * N * I(3)) .* 1.0, positions = pos, atomic_numbers = fill(29, size(pos, 1)))
end

function reference_structures()
    out = @NamedTuple{name::String, cell::Matrix{Float64}, positions::Matrix{Float64}, atomic_numbers::Vector{Int}}[]

    a = 3.615
    pos = displace!(fcc_sites(a, 2)[2:end, :], 0.05, 1.3)
    push!(out, (; name = "cu_vacancy", cell = Matrix(2a * I(3)) .* 1.0, positions = pos, atomic_numbers = fill(29, 31)))

    a = 3.56
    pos = displace!(fcc_sites(a, 2), 0.04, 0.7, 0.3)
    Z = [(24, 27, 28)[mod(5s + s ÷ 3, 3) + 1] for s in 1:32]
    push!(out, (; name = "crconi_alloy", cell = Matrix(2a * I(3)) .* 1.0, positions = pos, atomic_numbers = Z))

    cell = [4.2 0 0; 1.3 4.0 0; 0.9 1.1 4.4]
    pos = displace!(FCC_BASIS * cell, 0.03, 2.1)
    push!(out, (; name = "triclinic_ni3al", cell = Matrix{Float64}(cell), positions = pos, atomic_numbers = [28, 28, 13, 28]))

    pos = [5 5 5; 6 5 5; 5 6.25 5; 3.8 4.4 5.6; 7.5 7.5 2.5; 2.5 7.5 7.5; 7.6 2.4 7.4; 2.4 2.6 2.4]
    push!(out, (; name = "multi_element", cell = Matrix(10.0 * I(3)), positions = Matrix{Float64}(pos), atomic_numbers = [6, 1, 8, 7, 26, 29, 40, 74]))

    push!(out, (; name = "isolated_cu", cell = Matrix(20.0 * I(3)), positions = fill(10.0, 1, 3), atomic_numbers = [29]))
    return out
end
