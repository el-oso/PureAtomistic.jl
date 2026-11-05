# Long-lived tasks that run the chunks of a parallel loop, reused from one loop to the next.
#
# A job type `J` defines `_chunk!(job::J, phase::Int, c::Int)`, which runs chunk `c` of phase
# `phase` (a positive integer) of `job`. `_run_chunks!(workers, job, phase, n)` runs chunks `1` to
# `n - 1` on workers 1 to `n - 1` and chunk `n` on the calling task.
function _chunk! end

# The state the workers share. Worker `k` waits on `start[k]` for a phase, runs chunk `k` of
# `job[1]` and answers on `done[k]`: `true` when the chunk returned, `false` when it threw, after
# which the worker ends with that error; `failed[k]` holds that answer until the worker is
# replaced. Phase 0 stops a worker. `job` holds the job only while a loop runs, so idle workers
# keep nothing of it alive.
struct _WorkerPool{J}
    start::Vector{Channel{Int}}
    done::Vector{Channel{Bool}}
    tasks::Vector{Task}
    failed::Vector{Bool}
    job::Vector{J}
end

# The workers of one owner, started by the first `_run_chunks!` that needs them. Idle workers wait
# on their channels and run nothing. They hold the pool but not this object, so when it becomes
# unreachable its finalizer, registered when the first worker starts, stops them; `close` stops
# them at once. A worker that ended, by `close` or by an error, is started again by the next
# `_run_chunks!` that needs it. `busy` is set while `_run_chunks!` or `close` runs: the workers
# serve one loop at a time, and a second loop or a `close` during a loop would take or withhold
# the answers on `done` that the first one waits for.
mutable struct ChunkWorkers{J}
    const pool::_WorkerPool{J}
    @atomic busy::Bool
    ChunkWorkers{J}() where {J} = new{J}(_WorkerPool{J}(Channel{Int}[], Channel{Bool}[], Task[], Bool[], J[]), false)
end

# Marks `w` busy, or throws when it already is.
function _acquire!(w::ChunkWorkers)
    (@atomicswap w.busy = true) &&
        throw(ArgumentError("model: the workspace is in use by another evaluation or close; a workspace serves one evaluation at a time"))
    return nothing
end

_release!(w::ChunkWorkers) = (@atomic w.busy = false; nothing)

# A finalizer must not wait for another task, which stopping the workers does, so it hands the
# stop to a new task.
function _stop_when_unreachable(w::ChunkWorkers)
    Threads.@spawn _stop_workers!(w.pool)
    return nothing
end

function Base.close(w::ChunkWorkers)
    _acquire!(w)
    try
        _stop_workers!(w.pool)
    finally
        _release!(w)
    end
    return nothing
end

# Stops every running worker and waits until it has ended.
function _stop_workers!(pool::_WorkerPool)
    for k in eachindex(pool.tasks)
        istaskdone(pool.tasks[k]) || put!(pool.start[k], 0)
    end
    for t in pool.tasks
        wait(t)
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
            push!(pool.failed, false)
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
# number) as a `TaskFailedException`. A worker whose chunk threw is replaced before the call
# returns, so the pool keeps no failed task and nothing its error refers to.
function _run_chunks!(w::ChunkWorkers{J}, job::J, phase::Int, n::Int) where {J}
    phase > 0 || throw(ArgumentError(lazy"model: a chunk phase must be positive, got $phase"))
    _acquire!(w)
    pool = w.pool
    failure = nothing
    try
        isempty(pool.job) || throw(ArgumentError("model: the chunk workers still hold the job of another loop"))
        _start_workers!(w, n - 1)
        try
            _run_started_chunks!(pool, job, phase, n)
        finally
            failure = _replace_failed_workers!(pool)
        end
    finally
        _release!(w)
    end
    isnothing(failure) || throw(TaskFailedException(failure))
    return nothing
end

# The loop of `_run_chunks!` on running workers. Every chunk handed to a worker is waited for, also
# when the calling task's chunk throws; `pool.failed[k]` records whether worker `k`'s chunk threw.
function _run_started_chunks!(pool::_WorkerPool{J}, job::J, phase::Int, n::Int) where {J}
    started = 0
    try
        push!(pool.job, job)
        for k in 1:(n - 1)
            put!(pool.start[k], phase)
            started = k
        end
        _chunk!(job, phase, n)
    finally
        for k in 1:started
            pool.failed[k] = !take!(pool.done[k])
        end
        empty!(pool.job)
    end
    return nothing
end

# Waits for each worker whose chunk threw to end and starts a new worker in its place. Returns the
# ended task of the first, or `nothing` when no chunk threw.
function _replace_failed_workers!(pool::_WorkerPool)
    first_failed = nothing
    for k in eachindex(pool.tasks, pool.failed)
        pool.failed[k] || continue
        t = pool.tasks[k]
        # `wait` rethrows the chunk's error, which the caller reports from the returned task.
        try
            wait(t)
        catch
        end
        isnothing(first_failed) && (first_failed = t)
        pool.failed[k] = false
        pool.tasks[k] = Threads.@spawn _worker_loop!(pool, k)
    end
    return first_failed
end
