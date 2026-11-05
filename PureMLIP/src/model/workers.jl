# Long-lived tasks that run the chunks of a parallel loop, reused from one loop to the next.
#
# A job type `J` defines `_chunk!(job::J, phase::Int, c::Int)`, which runs chunk `c` of phase
# `phase` (a positive integer) of `job`. `_run_chunks!(workers, job, phase, n)` runs chunks `1` to
# `n - 1` on workers 1 to `n - 1` and chunk `n` on the calling task.
function _chunk! end

# The state the workers share. Worker `k` waits on `start[k]` for a phase, runs chunk `k` of
# `job[1]` and answers on `done[k]`: `true` when the chunk returned, `false` when it threw, after
# which the worker ends with that error. Phase 0 stops a worker. `job` holds the job only while
# a loop runs, so idle workers keep nothing of it alive.
struct _WorkerPool{J}
    start::Vector{Channel{Int}}
    done::Vector{Channel{Bool}}
    tasks::Vector{Task}
    job::Vector{J}
end

# The workers of one owner, started by the first `_run_chunks!` that needs them. Idle workers wait
# on their channels and run nothing. They hold the pool but not this object, so when it becomes
# unreachable its finalizer, registered when the first worker starts, stops them; `close` stops
# them at once. A worker that ended, by `close` or by an error, is started again by the next
# `_run_chunks!` that needs it.
mutable struct ChunkWorkers{J}
    const pool::_WorkerPool{J}
    ChunkWorkers{J}() where {J} = new{J}(_WorkerPool{J}(Channel{Int}[], Channel{Bool}[], Task[], J[]))
end

# A finalizer must not wait for another task, which stopping the workers does, so it hands the
# stop to a new task.
function _stop_when_unreachable(w::ChunkWorkers)
    Threads.@spawn _stop_workers!(w.pool)
    return nothing
end

Base.close(w::ChunkWorkers) = _stop_workers!(w.pool)

# Stops every running worker and waits until it has ended. A worker that failed has already
# ended; its error was rethrown by the loop it failed in.
function _stop_workers!(pool::_WorkerPool)
    for k in eachindex(pool.tasks)
        istaskdone(pool.tasks[k]) || put!(pool.start[k], 0)
    end
    for t in pool.tasks
        istaskfailed(t) || wait(t)
    end
    return nothing
end

function _worker_loop!(pool::_WorkerPool, k::Int)
    start, done = pool.start[k], pool.done[k]
    phase = take!(start)
    while !iszero(phase)
        try
            _worker_chunk!(pool, phase, k)
        catch
            put!(done, false)
            rethrow()
        end
        put!(done, true)
        phase = take!(start)
    end
    return nothing
end

# Out of line, so that no variable of the waiting loop still refers to the job.
@noinline function _worker_chunk!(pool::_WorkerPool, phase::Int, k::Int)
    _chunk!(pool.job[1], phase, k)
    return nothing
end

# Starts workers 1 to `m` where they are not running.
function _start_workers!(w::ChunkWorkers, m::Int)
    pool = w.pool
    isempty(pool.tasks) && m > 0 && finalizer(_stop_when_unreachable, w)
    for k in 1:m
        if k > length(pool.tasks)
            push!(pool.start, Channel{Int}(1))
            push!(pool.done, Channel{Bool}(1))
            push!(pool.tasks, Threads.@spawn _worker_loop!(pool, k))
        elseif istaskdone(pool.tasks[k])
            pool.tasks[k] = Threads.@spawn _worker_loop!(pool, k)
        end
    end
    return nothing
end

# Runs chunks `1` to `n - 1` of `phase` of `job` on the workers and chunk `n` on the calling
# task. It returns when every chunk has finished, also when one of them throws: the error of the
# calling task's chunk is rethrown as it is, and that of a worker's chunk (the first by chunk
# number) as a `TaskFailedException`.
function _run_chunks!(w::ChunkWorkers{J}, job::J, phase::Int, n::Int) where {J}
    phase > 0 || throw(ArgumentError(lazy"model: a chunk phase must be positive, got $phase"))
    pool = w.pool
    _start_workers!(w, n - 1)
    push!(pool.job, job)
    for k in 1:(n - 1)
        put!(pool.start[k], phase)
    end
    failed = 0
    try
        _chunk!(job, phase, n)
    finally
        for k in 1:(n - 1)
            if !take!(pool.done[k]) && iszero(failed)
                failed = k
            end
        end
        empty!(pool.job)
    end
    iszero(failed) || wait(pool.tasks[failed])
    return nothing
end
