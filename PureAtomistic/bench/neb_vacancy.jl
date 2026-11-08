# The migration barrier of a vacancy in Cu with MACE-MPA-0 by the climbing-image nudged elastic
# band (`neb`), in the 255-atom cell of the vacancy campaign (4 × 4 × 4 fcc cells, the atom
# nearest (2a, 2a, 2a) removed), on the CPU with Julia threads.
#
#   PUREMLIP_CHECKPOINT_DIR=<dir> julia -t 6,0 --gcthreads=1,0 --project=PureAtomistic/bench PureAtomistic/bench/neb_vacancy.jl \
#       lattice <T_K>          the zero-pressure lattice constant of the vacancy cell at T
#       neb <a_Å> <images> [fmax]   the barrier at lattice constant a with <images> moving images
#
# lattice  Float32 (the campaign's precision), Maxwell-Boltzmann velocities at T, 20 ps of Bussi
#          (τ 100 fs) and Berendsen (P₀ 0, τ_P 500 fs, β_T 1/140 GPa⁻¹) with dt 2 fs; the volume is
#          sampled every step and the mean over the last 10 ps gives a = (V / 64)^(1/3), with the
#          standard error from the integrated autocorrelation time.
# neb      Float64. The endpoints are the relaxed vacancy (fmax 1e-3 eV/Å, default relaxer) and the
#          same cell with the neighbor at a/2 · (1, 1, 0) from the vacant site moved onto it, then
#          relaxed. The band starts as the straight line between them and runs with the climbing
#          image, spring 0.1 eV/Å² and FIRE with its defaults, to fmax (default 0.01 eV/Å). The atom farthest
#          from the vacant site (minimum image) is fixed in both relaxations and in the band, so the
#          band does not drift. E_m is the climbing image's energy minus the first endpoint's.
#          The unrelaxed straight-line midpoint (the hopping atom halfway, every other atom on its
#          lattice site) is evaluated as well.
#
# Results are appended to PureAtomistic/bench/results/neb_vacancy.json. Chunks: PUREATOMISTIC_NCHUNKS
# (default: the number of Julia threads). PUREATOMISTIC_CPULOCK names the shared lock script, held
# for the whole run.
using JSON, Statistics, Dates, Printf, Random
include(joinpath(@__DIR__, "cells.jl"))
include(joinpath(@__DIR__, "autocorrelation.jl"))
using PureAtomistic.Systems: maxwell_boltzmann!, cell_volume
using PureMLIP: MACEWorkspace, mace_energy_forces!, ThreadedChunks
BLAS.set_num_threads(1)

const RESULTS = joinpath(@__DIR__, "results", "neb_vacancy.json")
const NCHUNKS = parse(Int, get(ENV, "PUREATOMISTIC_NCHUNKS", string(Threads.nthreads())))
const LOCK_SCRIPT = get(ENV, "PUREATOMISTIC_CPULOCK", "")
lock_acquire() = isempty(LOCK_SCRIPT) ? "" : strip(read(`$LOCK_SCRIPT acquire`, String))
lock_release(tok) = isempty(LOCK_SCRIPT) || run(`$LOCK_SCRIPT release $tok`)

function git_commit()
    hash = strip(read(setenv(`git rev-parse --short HEAD`; dir = @__DIR__), String))
    changed = read(setenv(`git status --porcelain -- bench src ext ":(exclude)bench/results"`; dir = dirname(@__DIR__)), String)
    return isempty(strip(changed)) ? hash : hash * "-dirty"
end

stamp() = Dict{String, Any}(
    "commit" => git_commit(), "date" => string(Dates.today()), "julia" => string(VERSION),
    "julia_threads" => Threads.nthreads(), "nchunks" => NCHUNKS, "blas_threads" => BLAS.get_num_threads(),
    "cpu" => host_cpu(), "potential" => "MACE-MPA-0",
)

function save(update)
    d = isfile(RESULTS) ? JSON.parsefile(RESULTS) : Dict{String, Any}()
    update(d)
    open(io -> JSON.print(io, d, 1), RESULTS, "w")
    return nothing
end

# A MACEPotential evaluated in `nchunks` receiver chunks on Julia threads; the results depend on
# `nchunks` through rounding only.
mutable struct ThreadedMACE{T, P <: MACEPotential{T}, R} <: PP.AbstractStressPotential
    const inner::P
    const result::R
    const nchunks::Int
    regrows::Int
end
function chunk!(inner::MACEPotential{T}, nchunks) where {T}
    inner.ws = MACEWorkspace{T}(inner.model, PP.n_atoms(inner), length(inner.nl.senders); elements = inner.atomic_numbers, nchunks)
    return nothing
end
function ThreadedMACE(model, positions, cell, species, nchunks)
    inner = MACEPotential(model, positions, cell, species)
    chunk!(inner, nchunks)
    return ThreadedMACE(inner, inner.result, nchunks, inner.n_regrows)
end
PP.n_atoms(p::ThreadedMACE) = PP.n_atoms(p.inner)
PP.cutoff(p::ThreadedMACE) = PP.cutoff(p.inner)
PP.stress(p::ThreadedMACE) = PP.stress(p.inner)
function PP.prepare!(p::ThreadedMACE, x::AbstractMatrix, c::AbstractMatrix)
    PP.prepare!(p.inner, x, c)
    # A regrow rebuilds the single-chunk workspace; restore the chunked one.
    if p.inner.n_regrows != p.regrows
        chunk!(p.inner, p.nchunks)
        p.regrows = p.inner.n_regrows
    end
    return nothing
end
function PP.energy_forces!(p::ThreadedMACE, x::AbstractMatrix, c::AbstractMatrix)
    i = p.inner
    mace_energy_forces!(i.result, i.ws, i.model, x, c, i.atomic_numbers, i.nl, ThreadedChunks())
    return nothing
end

function host_vacancy(::Type{T}, a) where {T}
    perfect = fcc_cell(T(a), 4, :Cu)
    v = argmin(j -> norm(perfect.positions[:, j] - fill(2a, 3)), 1:natoms(perfect))
    return perfect, v, remove_atom(perfect, v)
end

function run_lattice(Tset)
    T = Float32
    _, _, sys = host_vacancy(T, A0)
    seed = 20261108
    maxwell_boltzmann!(sys, Tset, Xoshiro(seed))
    model = MACEModel{T}(load_mace_checkpoint(mpa0_path()))
    pot = ThreadedMACE(model, sys.positions, sys.cell, sys.species, NCHUNKS)
    dt = 2.0
    nsteps = 10_000
    sim = Simulation(
        sys, pot, VelocityVerlet(dt); thermostat = BussiThermostat(Tset, 100.0),
        barostat = BerendsenBarostat(0.0, 500.0, 1 / 140), rng = Xoshiro(seed + 1)
    )
    V = zeros(nsteps)
    tok = lock_acquire()
    t0 = time()
    for s in 1:nsteps
        step!(sim)
        V[s] = cell_volume(sys.cell)
        s % 1000 == 0 && (@printf("step %d: V/atom = %.4f Å³\n", s, V[s] / natoms(sys)); flush(stdout))
    end
    wall = time() - t0
    lock_release(tok)
    tail = V[(nsteps ÷ 2 + 1):end]
    err = correlated_errors(tail)
    Vm = mean(tail)
    a = cbrt(Vm / 64)
    a_se = a * err.mean_error / (3Vm)
    @printf("T = %.0f K: a = %.5f ± %.5f Å (V/atom %.4f Å³), %.0f s\n", Tset, a, a_se, Vm / natoms(sys), wall)
    save() do d
        runs = get!(d, "lattice", Any[])
        push!(
            runs, merge(
                stamp(), Dict(
                    "temperature_K" => Tset, "precision" => "Float32", "natoms" => natoms(sys), "dt_fs" => dt, "steps" => nsteps,
                    "thermostat" => "Bussi tau 100 fs", "barostat" => "Berendsen P0 0, tau_P 500 fs, beta_T 1/140 per GPa",
                    "seed" => seed, "averaged_steps" => length(tail), "mean_volume_A3" => Vm,
                    "volume_mean_error_A3" => err.mean_error, "tau_samples" => err.tau_samples,
                    "lattice_constant_A" => a, "lattice_constant_se_A" => a_se, "wall_seconds" => wall,
                    "regrowths" => pot.inner.n_regrows,
                )
            )
        )
    end
    return nothing
end

# Minimum-image distance of `x` from `y` in a cubic cell of edge L.
mic(x, y, L) = norm(@. (x - y) - L * round((x - y) / L))

function run_neb(a, M, fmax)
    T = Float64
    perfect, v, host = host_vacancy(T, a)
    site = perfect.positions[:, v]
    L = 4a
    N = natoms(host)
    hop = findfirst(k -> norm(host.positions[:, k] - site - a / 2 * [1, 1, 0]) < 1.0e-8, 1:N)
    far = argmax(k -> mic(host.positions[:, k], site, L), 1:N)
    model = MACEModel{T}(load_mace_checkpoint(mpa0_path()))
    pot = ThreadedMACE(model, host.positions, host.cell, host.species, NCHUNKS)
    energy(x) = (prepare!(pot, x, host.cell); energy_forces!(pot, x, host.cell); pot.result.energy)
    tok = lock_acquire()
    t0 = time()
    # The unrelaxed straight-line midpoint against the unrelaxed vacancy.
    mid = copy(host.positions)
    mid[:, hop] = (host.positions[:, hop] + site) / 2
    E_unrelaxed = energy(mid) - energy(host.positions)
    initial = deepcopy(host)
    r_i = relax!(initial, pot; fmax = 1.0e-3, fixed = [far])
    final = deepcopy(host)
    final.positions[:, hop] = site
    r_f = relax!(final, pot; fmax = 1.0e-3, fixed = [far])
    (r_i.converged && r_f.converged) || error("an endpoint did not relax")
    t_relax = time() - t0
    @printf("a = %.4f: endpoints relaxed in %d and %d steps, E_f - E_i = %.2e eV; unrelaxed midpoint %.4f eV\n", a, r_i.steps, r_f.steps, r_f.energy - r_i.energy, E_unrelaxed)
    flush(stdout)
    X0, X1 = initial.positions, final.positions
    images = [X0 + t * (X1 - X0) for t in range(0, 1; length = M + 2)]
    spring = 0.1
    t1 = time()
    r = neb(images, host.cell, pot; spring, fmax, maxsteps = 3000, fixed = [far])
    t_neb = time() - t1
    lock_release(tok)
    Em = r.energies[r.climbing] - r.energies[1]
    c = r.images[r.climbing]
    @printf("a = %.4f, %d images: converged %s in %d steps (%.0f s), E_m = %.4f eV, climbing image %d, hop atom %.4f Å from the midpoint\n", a, M, r.converged, r.steps, t_neb, Em, r.climbing, norm(c[:, hop] - (X0[:, hop] + X1[:, hop]) / 2))
    save() do d
        runs = get!(d, "neb", Any[])
        push!(
            runs, merge(
                stamp(), Dict(
                    "lattice_constant_A" => a, "precision" => "Float64", "natoms" => N, "moving_images" => M,
                    "climbing" => true, "spring_eV_per_A2" => spring, "fmax_eV_per_A" => fmax, "optimizer" => "FIRE (defaults)",
                    "endpoint_fmax_eV_per_A" => 1.0e-3, "endpoint_relaxer" => "TrustRegionLBFGS (relax! default)",
                    "fixed_atom" => far, "hop_atom" => hop, "vacant_site_A" => site,
                    "endpoint_relax_steps" => [r_i.steps, r_f.steps], "endpoint_energies_eV" => [r_i.energy, r_f.energy],
                    "unrelaxed_midpoint_barrier_eV" => E_unrelaxed,
                    "converged" => r.converged, "steps" => r.steps, "force_evaluations" => (r.steps + 1) * M + 2,
                    "max_force_eV_per_A" => r.max_force, "climbing_image" => r.climbing,
                    "energies_eV" => r.energies, "E_m_eV" => Em,
                    "reverse_barrier_eV" => r.energies[r.climbing] - r.energies[end],
                    "image_distances_A" => [norm(r.images[j + 1] - r.images[j]) for j in 1:(M + 1)],
                    "hop_atom_path_A" => [x[:, hop] for x in r.images],
                    "climbing_hop_atom_from_midpoint_A" => norm(c[:, hop] - (X0[:, hop] + X1[:, hop]) / 2),
                    "wall_seconds_relax" => t_relax, "wall_seconds_neb" => t_neb,
                )
            )
        )
    end
    return nothing
end

mode = ARGS[1]
if mode == "lattice"
    run_lattice(parse(Float64, ARGS[2]))
elseif mode == "neb"
    run_neb(parse(Float64, ARGS[2]), parse(Int, ARGS[3]), length(ARGS) > 3 ? parse(Float64, ARGS[4]) : 0.01)
else
    error("mode must be lattice or neb")
end
