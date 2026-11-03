# Shared setup of the relaxation benchmarks: the MACE-MPA-0 model, the two 256-site Cu cells and a
# potential wrapper that counts force evaluations.
using PureAtomistic, PureMLIP, LinearAlgebra, Random, SHA, Downloads
using PureAtomistic.Systems: System, fcc_cell, remove_atom, natoms
import PureAtomistic.Potentials as PP

# MACE-MPA-0 in PUREMLIP_CHECKPOINT_DIR, downloaded on first use and checked by SHA-256.
function mpa0_path()
    file = "mace-mpa-0-medium.model"
    url = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/" * file
    sha = "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"
    dir = get(ENV, "PUREMLIP_CHECKPOINT_DIR", joinpath(first(DEPOT_PATH), "scratchspaces", "PureAtomistic-checkpoints"))
    mkpath(dir)
    path = joinpath(dir, file)
    isfile(path) || (Downloads.download(url, path * ".part"); mv(path * ".part", path; force = true))
    bytes2hex(open(sha256, path)) == sha || error("SHA-256 mismatch for $path")
    return path
end

# The model's lattice constant to six digits (parabola fit of the 4-atom cell; see the test suite).
const A0 = 3.640391

# 4 × 4 × 4 fcc cells (256 sites) with the atom nearest (2a0, 2a0, 2a0) removed.
function vacancy_cell()
    perfect = fcc_cell(A0, 4, :Cu)
    v = argmin(j -> norm(perfect.positions[:, j] - fill(2A0, 3)), 1:natoms(perfect))
    return remove_atom(perfect, v)
end

# 256 atoms in a cell strained by diag(1.03, 1.00, 0.98), every coordinate displaced by a seeded
# normal deviate of 0.05 Å. The strained lattice is the minimum of the fixed cell.
function strained_cell()
    sys = fcc_cell(A0, 4, :Cu)
    D = Diagonal([1.03, 1.0, 0.98])
    sys.cell .= D * sys.cell
    sys.positions .= D * sys.positions .+ 0.05 .* randn(Xoshiro(1), 3, natoms(sys))
    return sys
end

const CELLS = (vacancy = vacancy_cell, strained = strained_cell)

# A potential that forwards to `inner`, counts evaluations and calls `hook()` after each one.
mutable struct Counting{P <: PP.AbstractPotential, R, H} <: PP.AbstractPotential
    const inner::P
    const result::R
    const hook::H
    evals::Int
end
Counting(inner; hook = () -> nothing) = Counting(inner, inner.result, hook, 0)
PP.n_atoms(p::Counting) = PP.n_atoms(p.inner)
PP.cutoff(p::Counting) = PP.cutoff(p.inner)
PP.prepare!(p::Counting, x::AbstractMatrix, c::AbstractMatrix) = PP.prepare!(p.inner, x, c)
function PP.energy_forces!(p::Counting, x::AbstractMatrix, c::AbstractMatrix)
    PP.energy_forces!(p.inner, x, c)
    p.evals += 1
    p.hook()
    return nothing
end

mace(model, sys) = MACEPotential(model, sys.positions, sys.cell, sys.species)

# Largest atomic force of a fresh evaluation at the positions of `sys`.
function fresh_fmax(model, sys)
    p = mace(model, sys)
    prepare!(p, sys.positions, sys.cell)
    energy_forces!(p, sys.positions, sys.cell)
    return maximum(norm, eachcol(p.result.forces)), p.result.energy
end

host_cpu() = strip(read(`sysctl -n machdep.cpu.brand_string`, String))
