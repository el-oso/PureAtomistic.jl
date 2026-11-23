# The loop form of `substitute`. Every series of the unrolled form (the nilpotent parts hᵢ, the
# monomials and the partial sums) has its own mask; here each one is held in a scratch array: hᵢ and
# the monomials in the layout of their own mask, the partial sum in the layout of the mask of the
# result, where the entries outside the mask of a partial sum are zero. Each step performs the
# arithmetic of the corresponding unrolled statement on the entries of its masks, in the same
# order.

# For each variable i of E, the mask of the nilpotent part hᵢ = u[i] - value(u[i]), or -1 when
# variable i does not enter: u[i] is a plain number, or E does not depend on variable i.
function _substitute_masks(N, M, u)
    return ntuple(N) do i
        P = u.parameters[i]
        P <: Taylor && isodd(M >> (i - 1)) ? P.parameters[3] : -1
    end
end

# The steps of the loop form of `substitute`, computed when the method is generated, or `nothing`
# when the result is the constant term alone. Fields:
#   h      (i, offset) for each hᵢ: copy hᵢ to the scratch entries offset + 1:offset + length;
#   steps  (1, out, a, b, base, n, 0, 0): the product of the monomials at offsets a and b, of n
#          coefficients, with the pairs of `voff[base + p]:voff[base + p + 1] - 1` (`via`, `vib`)
#          for coefficient p, written at offset out; or (2, m, j, D, base, n, A, first): the term
#          _scale_from(monomial, E.c[j], D) of the monomial at offset m, whose coefficient q goes to
#          position `cmap[base + q]` of the partial sum, added to the partial sum of mask A unless
#          `first` is 1;
#   wdeg, wsupp  the total degree and the mask of the variables of each multi-index of the result;
#   S      the scratch entries for hᵢ and the monomials (the partial sum follows them);
#   Mall, Lall  the mask and the length of the result.
function _substitute_plan(N, K, M, N2, umasks)
    len(mask) = binomial(count_ones(mask) + K, K)
    mmask = Int[]
    ops = Tuple[]
    hid = map(1:N) do i
        umasks[i] < 0 && return 0
        push!(mmask, umasks[i])
        push!(ops, (:h, length(mmask), i))
        return length(mmask)
    end
    # The monomials of the unrolled form, by the same recursion.
    memo = Dict{NTuple{N, Int}, Int}()
    function mono(α)
        haskey(memo, α) && return memo[α]
        i = findlast(>(0), α)
        power = ntuple(k -> k == i ? α[i] : 0, N)
        if α == power
            α[i] == 1 && return memo[α] = hid[i]
            a = mono(power .- ntuple(k -> k == i, N))
            b = hid[i]
        else
            a = mono(α .- power)
            b = mono(power)
        end
        push!(mmask, mmask[a] | mmask[b])
        push!(ops, (:p, length(mmask), a, b))
        return memo[α] = length(mmask)
    end
    for (j, α) in enumerate(stored_indices(Val(N), Val(K), Val(M)))
        j == 1 && continue
        any(i -> α[i] > 0 && iszero(hid[i]), 1:N) && continue
        push!(ops, (:t, mono(α), j, sum(α)))
    end
    terms = filter(op -> op[1] === :t, ops)
    isempty(terms) && return nothing
    Mall = reduce(|, (mmask[op[2]] for op in terms))
    idxw = stored_indices(Val(N2), Val(K), Val(Mall))
    posw = Dict(α => p for (p, α) in enumerate(idxw))

    # Scratch offsets: a monomial's entries are reused, by one of the same length, after its last use.
    operands(op) = op[1] === :p ? (op[3], op[4]) : op[1] === :t ? (op[2],) : ()
    last = Dict{Int, Int}()
    for (s, op) in enumerate(ops), id in operands(op)
        last[id] = s
    end
    free = Dict{Int, Vector{Int}}()
    top = 0
    offset = Dict{Int, Int}()
    function alloc(id)
        n = len(mmask[id])
        pool = get!(free, n, Int[])
        if isempty(pool)
            offset[id] = top
            top += n
        else
            offset[id] = pop!(pool)
        end
        return offset[id]
    end
    release(id, s) = get(last, id, 0) <= s && push!(free[len(mmask[id])], offset[id])

    voff, via, vib = Int[], Int[], Int[]
    variants = Dict{Tuple{Int, Int}, Int}()
    function variant(Ma, Mb)
        return get!(variants, (Ma, Mb)) do
            base = length(voff)
            idx1 = stored_indices(Val(N2), Val(K), Val(Ma))
            pos2 = Dict(γ => k for (k, γ) in enumerate(stored_indices(Val(N2), Val(K), Val(Mb))))
            push!(voff, length(via) + 1)
            for α in stored_indices(Val(N2), Val(K), Val(Ma | Mb))
                for (i, β) in enumerate(idx1)
                    k = all(β .<= α) ? get(pos2, α .- β, 0) : 0
                    iszero(k) || (push!(via, i); push!(vib, k))
                end
                push!(voff, length(via) + 1)
            end
            base
        end
    end
    cmap = Int[]
    classes = Dict{Int, Int}()
    class(B) = get!(classes, B) do
        base = length(cmap)
        append!(cmap, (posw[α] for α in stored_indices(Val(N2), Val(K), Val(B))))
        base
    end

    h = Tuple{Int, Int}[]
    steps = NTuple{8, Int}[]
    A = 0
    for (s, op) in enumerate(ops)
        if op[1] === :h
            push!(h, (op[3], alloc(op[2])))
            release(op[2], s)
        elseif op[1] === :p
            id, a, b = op[2], op[3], op[4]
            out = alloc(id)
            push!(steps, (1, out, offset[a], offset[b], variant(mmask[a], mmask[b]), len(mmask[id]), 0, 0))
            release(a, s)
            a == b || release(b, s)
            release(id, s)
        else
            m, j, D = op[2], op[3], op[4]
            B = mmask[m]
            push!(steps, (2, offset[m], j, D, class(B), len(B), A, iszero(A) ? 1 : 0))
            A |= B
            release(m, s)
        end
    end
    # An unused table holds one entry, so that no table is the empty tuple.
    _table(v) = _int32_tuple(isempty(v) ? (0,) : v)
    wdeg = _int32_tuple(sum.(idxw))
    wsupp = Tuple(reduce(|, (α[i] > 0 ? 1 << (i - 1) : 0 for i in 1:N2); init = 0) for α in idxw)
    return (;
        h = Tuple(h), steps = Tuple(steps), voff = _table(voff), via = _table(via), vib = _table(vib),
        cmap = _table(cmap), wdeg, wsupp, S = top, Mall, Lall = length(idxw),
    )
end

# The body of `substitute`: the loop form `_substitute_loop` when `loop` is true and the result or
# `E` has more than `_UNROLL_MAX` coefficients, otherwise the unrolled form.
function _substitute_body(N, K, M, L, u, loop)
    # The number of new variables, from the Taylor elements of `u`; they must agree.
    news = unique(P.parameters[1] for P in u.parameters if P <: Taylor)
    length(news) == 1 ||
        return :(throw(ArgumentError($("PureTaylor: substitute needs at least one Taylor number in u, all over the same number of variables; got $(Tuple(news))"))))
    N2 = only(news)
    if loop
        plan = _substitute_plan(N, K, M, N2, _substitute_masks(N, M, u))
        if !isnothing(plan) && (_use_loop(L) || _use_loop(plan.Lall))
            return Expr(:block, Expr(:meta, :noinline), :(_substitute_loop(E, u)))
        end
    end
    body = Expr(:block, Expr(:meta, :inline))
    # h[i]: the nilpotent part of u[i], or nothing for a plain number (its increment is zero).
    h = map(1:N) do i
        u.parameters[i] <: Taylor && isodd(M >> (i - 1)) || return nothing
        s = Symbol(:h, i)
        push!(body.args, :($s = u[$i] - value(u[$i])))
        return s
    end
    # Each monomial Π hᵢ^αᵢ is built once: a power of one variable from the next lower power, any
    # other monomial as the monomial without its last variable times that variable's power.
    monomials = Dict{NTuple{N, Int}, Symbol}()
    function monomial(α)
        haskey(monomials, α) && return monomials[α]
        i = findlast(>(0), α)
        power = ntuple(k -> k == i ? α[i] : 0, N)
        if α == power
            α[i] == 1 && return monomials[α] = h[i]
            rhs = :($(monomial(power .- ntuple(k -> k == i, N))) * $(h[i]))
        else
            rhs = :($(monomial(α .- power)) * $(monomial(power)))
        end
        s = Symbol(:m_, join(α, '_'))
        push!(body.args, :($s = $rhs))
        return monomials[α] = s
    end
    # The sum, one fresh name per partial sum so that each has a single type.
    acc = nothing
    for (j, α) in enumerate(stored_indices(Val(N), Val(K), Val(M)))
        j == 1 && continue
        any(i -> α[i] > 0 && isnothing(h[i]), 1:N) && continue
        term = :(_scale_from($(monomial(α)), E.c[$j], Val($(sum(α)))))
        s = Symbol(:s_, j)
        push!(body.args, :($s = $(isnothing(acc) ? term : :($acc + $term))))
        acc = s
    end
    push!(body.args, isnothing(acc) ? :(constant(Val($N2), Val(K), E.c[1])) : :($acc + E.c[1]))
    return body
end

"""
    substitute(E::Taylor{N,K}, u)

Change of variables. `E` is a series whose `N` variables are increments of `N` quantities; `u[i]` is the
`i`-th quantity as a `Taylor{N2,K}` over new variables, or a plain number for a quantity that does not
change. At least one `u[i]` is a `Taylor`, and all of them have the same `N2`. The result is
`Σ_α c_α · Π_i (u[i] - value(u[i]))^α_i` over the coefficients `c_α` of `E`, a `Taylor{N2,K}` over
the new variables. The values of `u` are not used: `E` is taken to be expanded around them. When variable
`i` of `E` was seeded with a scale `s` (`variable(…, x, s)`), it is the increment divided by `s`, so
pass `u[i] / s`.
"""
@generated function substitute(
        E::Taylor{N, K, M, T, L}, u::Tuple{Vararg{Union{Taylor{<:Any, K, <:Any, T}, T}, N}}
    ) where {N, K, M, T, L}
    return _substitute_body(N, K, M, L, u, true)
end

# The unrolled form at any size; the tests compare it with the loop form.
@generated function _substitute_unrolled(
        E::Taylor{N, K, M, T, L}, u::Tuple{Vararg{Union{Taylor{<:Any, K, <:Any, T}, T}, N}}
    ) where {N, K, M, T, L}
    return _substitute_body(N, K, M, L, u, false)
end

# The unrolled body of `_scale_from`.
function _scale_from_unrolled_body(N, K, M, D)
    terms = map(enumerate(stored_indices(Val(N), Val(K), Val(M)))) do (j, α)
        sum(α) < D ? :(zero(T)) : :(m.c[$j] * r)
    end
    return quote
        $(Expr(:meta, :inline))
        Taylor{N, K, M, T, L}($(Expr(:tuple, terms...)))
    end
end

# r·m for a series m with no terms below degree D, such as a monomial of degree D in increments:
# the coefficients below degree D are zero without a product, so an infinite r (a coefficient whose
# value is beyond the range of T) does not reach them as 0·Inf. Unrolled up to `_UNROLL_MAX`
# coefficients, `_scale_from_loop` above.
@generated function _scale_from(m::Taylor{N, K, M, T, L}, r::T, ::Val{D}) where {N, K, M, T, L, D}
    _use_loop(L) && return :(_scale_from_loop(m, r, $D))
    return _scale_from_unrolled_body(N, K, M, D)
end

# The unrolled form at any size; the tests compare it with the loop form.
@generated _scale_from_unrolled(m::Taylor{N, K, M, T, L}, r::T, ::Val{D}) where {N, K, M, T, L, D} =
    _scale_from_unrolled_body(N, K, M, D)

# `_scale_from` by a loop over the degrees of the stored multi-indices.
@noinline function _scale_from_loop(m::Taylor{N, K, M, T, L}, r::T, D::Int) where {N, K, M, T, L}
    degree = _degrees(Val(N), Val(K), Val(M))
    c = _loop_tuple(T, Val(L)) do j
        degree[j] < D ? zero(T) : m.c[j] * r
    end
    return Taylor{N, K, M, T, L}(c)
end

@generated function _substitute_loop(E::Taylor{N, K, M, T, L}, u::Tuple) where {N, K, M, T, L}
    N2 = only(unique(P.parameters[1] for P in u.parameters if P <: Taylor))
    plan = _substitute_plan(N, K, M, N2, _substitute_masks(N, M, u))
    hoff = ntuple(i -> something(get(Dict(plan.h), i, nothing), -1), N)
    return quote
        $(Expr(:meta, :noinline))
        c = _substitute_run(
            E, u, $hoff, $(plan.steps), $(plan.voff), $(plan.via), $(plan.vib), $(plan.cmap), $(plan.wdeg),
            $(plan.wsupp), Val($(plan.S)), Val($(plan.Lall)),
        )
        return Taylor{$N2, K, $(plan.Mall), T, $(plan.Lall)}(c) + E.c[1]
    end
end

# The coefficients of the partial sum after the last step of `_substitute_plan`.
function _substitute_run(
        E::Taylor{N, K, M, T}, u, hoff, steps, voff, via, vib, cmap, wdeg, wsupp, ::Val{S}, ::Val{Lall}
    ) where {N, K, M, T, S, Lall}
    return _with_scratch(T, Val(S + Lall)) do s
        for p in 1:Lall
            _store!(s, S + p, zero(T))
        end
        _substitute_h!(s, hoff, u)
        _substitute_steps!(s, E, steps, voff, via, vib, cmap, wdeg, wsupp, S)
        return _scratch_tuple(s, T, S, Val(Lall))
    end
end

# hᵢ = uᵢ - value(uᵢ) for each i with an offset ≥ 0, written to the scratch entries offset + 1:offset +
# length; the tuples are walked head first, so each uᵢ keeps its own type.
_substitute_h!(s, ::Tuple{}, ::Tuple{}) = nothing
function _substitute_h!(s, hoff::Tuple, u::Tuple)
    _substitute_h1!(s, first(hoff), first(u))
    return _substitute_h!(s, Base.tail(hoff), Base.tail(u))
end
_substitute_h1!(s, offset::Int, ::Real) = nothing
function _substitute_h1!(s, offset::Int, ui::Taylor{N, K, M, T, L}) where {N, K, M, T, L}
    offset < 0 && return nothing
    for q in 1:L
        _store!(s, offset + q, isone(q) ? ui.c[1] - value(ui) : ui.c[q])
    end
    return nothing
end

# The products and the terms of `_substitute_plan`, in order, on the scratch array `s` whose partial
# sum starts after entry `acc`.
function _substitute_steps!(s, E::Taylor{N, K, M, T}, steps, voff, via, vib, cmap, wdeg, wsupp, acc::Int) where {N, K, M, T}
    for st in steps
        if isone(st[1])
            out, oa, ob, base, n = st[2], st[3], st[4], st[5], st[6]
            for p in 1:n
                v = zero(T)
                first = true
                for q in voff[base + p]:(voff[base + p + 1] - 1)
                    x, y = _load(s, oa + via[q]), _load(s, ob + vib[q])
                    v = first ? x * y : _fma(x, y, v)
                    first = false
                end
                _store!(s, out + p, v)
            end
        else
            om, j, D, base, n, A, first = st[2], st[3], st[4], st[5], st[6], st[7], st[8]
            r = E.c[j]
            for q in 1:n
                p = cmap[base + q]
                t = wdeg[p] < D ? zero(T) : _load(s, om + q) * r
                inA = !isone(first) && iszero(wsupp[p] & ~A)
                _store!(s, acc + p, inA ? _load(s, acc + p) + t : t)
            end
        end
    end
    return nothing
end
