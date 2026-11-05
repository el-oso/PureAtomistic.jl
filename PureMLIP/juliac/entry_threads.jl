# juliac entry point: evaluate forces on the tiny checkpoint with two threaded chunks.
#   mace_threads <mode> <path to the tiny checkpoint written by PureMLIP.tiny_mace_checkpoint()>
# with <mode> one of
#   close      evaluate, print, close the workspace, exit
#   open       evaluate, print, exit with the workspace still open
#   dropped    evaluate, drop the workspace, wait for its finalizer to stop the workers
using PureMLIP

const POSITIONS = [0.0 1.4 0.2; 0.0 0.1 1.3; 0.0 0.3 0.1]
const CELL = [6.5 0.0 0.0; 0.4 6.2 0.0; 0.3 0.2 6.8]
const NUMBERS = [29, 1, 29]
const NCHUNKS = 2

hexbits(x::Float64) = string(reinterpret(UInt64, x); base = 16)

function print_result(r::MACEResult{Float64})
    print(Core.stdout, "energy=", hexbits(r.energy), "\n")
    print(Core.stdout, "forces=")
    for x in r.forces
        print(Core.stdout, hexbits(x), ",")
    end
    print(Core.stdout, "\n")
    return nothing
end

function new_workspace(model::MACEModel{Float64}, nl::NeighborList{Float64})
    return MACEWorkspace{Float64}(model, nl.n_atoms, nl.n_edges; nchunks = NCHUNKS, elements = NUMBERS)
end

function evaluate!(ws::MACEWorkspace{Float64}, model::MACEModel{Float64}, nl::NeighborList{Float64})
    r = mace_energy_forces!(MACEResult{Float64}(nl.n_atoms), ws, model, POSITIONS, CELL, NUMBERS, nl, ThreadedChunks())
    print_result(r)
    return nothing
end

# Evaluates on a workspace that goes out of scope, and returns the worker tasks.
@noinline function evaluate_and_drop(model::MACEModel{Float64}, nl::NeighborList{Float64})
    ws = new_workspace(model, nl)
    evaluate!(ws, model, nl)
    return copy(ws.workers.pool.tasks)
end

function workers_stopped(tasks::Vector{Task})
    deadline = time() + 30
    while !all(istaskdone, tasks) && time() < deadline
        GC.gc(true)
        sleep(0.05)
    end
    return all(istaskdone, tasks)
end

function (@main)(args::Vector{String})::Cint
    if length(args) != 2 || !(args[1] in ("close", "open", "dropped"))
        print(Core.stderr, "usage: mace_threads close|open|dropped <checkpoint>\n")
        return 1
    end
    print(Core.stdout, "threads=", string(Threads.nthreads()), "\n")
    model = MACEModel{Float64}(load_mace_checkpoint(args[2]))
    nl = neighbor_list(POSITIONS, CELL, model.r_max)
    if args[1] == "dropped"
        tasks = evaluate_and_drop(model, nl)
        print(Core.stdout, "workers_stopped=", string(workers_stopped(tasks)), "\n")
    else
        ws = new_workspace(model, nl)
        evaluate!(ws, model, nl)
        args[1] == "close" && close(ws)
    end
    return 0
end
