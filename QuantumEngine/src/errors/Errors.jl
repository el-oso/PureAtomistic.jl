"""
    QuantumEngine.Errors

The error types of QuantumEngine and the record of a lenient-mode warning. Each `showerror` method
prints the message after a prefix that names its kind (`input:`, `basis:`, `scf:`, …). Messages are
stored as `LazyString`, so an interpolated `lazy"..."` message is not rendered until it is shown.
"""
module Errors

using StrictMode: @strict_exempt, @strict_policy

"""
    QuantumEngineError <: Exception

The supertype of every error QuantumEngine throws besides `ArgumentError`.
"""
abstract type QuantumEngineError <: Exception end

"""
    InputError(source, line, msg)

An input file `source` is invalid at line `line`; `line == 0` when the error has no line (a
structured input, or the file as a whole). Prints `input: <source>:<line>: <msg>`, or
`input: <source>: <msg>` for line 0.
"""
struct InputError <: QuantumEngineError
    source::String
    line::Int
    msg::LazyString
    InputError(source::AbstractString, line::Integer, msg::AbstractString) = new(source, line, LazyString(msg))
end

"""
    BasisError(msg)

A basis set cannot be read or has no shells for an element. Prints `basis: <msg>`.
"""
struct BasisError <: QuantumEngineError
    msg::LazyString
    BasisError(msg::AbstractString) = new(LazyString(msg))
end

"""
    SCFNotConverged(msg)

A self-consistent field iteration did not converge. Prints `scf: <msg>`.
"""
struct SCFNotConverged <: QuantumEngineError
    msg::LazyString
    SCFNotConverged(msg::AbstractString) = new(LazyString(msg))
end

"""
    OptimizationFailed(msg)

A geometry optimization did not converge or cannot continue. Prints `optimization: <msg>`.
"""
struct OptimizationFailed <: QuantumEngineError
    msg::LazyString
    OptimizationFailed(msg::AbstractString) = new(LazyString(msg))
end

"""
    NotImplemented(section, msg)

A requested feature is not built. `section` names the specification section that describes it,
and `msg` the request and, where relevant, the oracle behavior it replaces. Prints
`not implemented (<section>): <msg>`.

It is a different binding from TypeContracts' trait type of the same name.
"""
struct NotImplemented <: QuantumEngineError
    section::String
    msg::LazyString
    NotImplemented(section::AbstractString, msg::AbstractString) = new(section, LazyString(msg))
end

"""
    FunctionalUnavailable(msg)

A requested exchange-correlation functional is not available. Prints `functional: <msg>`.
"""
struct FunctionalUnavailable <: QuantumEngineError
    msg::LazyString
    FunctionalUnavailable(msg::AbstractString) = new(LazyString(msg))
end

"""
    NumericalError(msg)

A computation produced a non-finite or otherwise unusable value. Prints `numerical: <msg>`.
"""
struct NumericalError <: QuantumEngineError
    msg::LazyString
    NumericalError(msg::AbstractString) = new(LazyString(msg))
end

"""
    InputWarning(source, line, case, msg)

The record of one input that lenient mode accepted where strict mode throws an `InputError`:
the file `source`, the line `line` (0 when there is none), the `case` that names the accepted
behavior, and the message. It is not an exception; `show` prints it as the `InputError` with the
same source, line and message is shown: `input: <source>:<line>: <msg>`, or
`input: <source>: <msg>` for line 0.
"""
struct InputWarning
    source::String
    line::Int
    case::Symbol
    msg::LazyString
    InputWarning(source::AbstractString, line::Integer, case::Symbol, msg::AbstractString) =
        new(source, line, case, LazyString(msg))
end

# `<source>:<line>: ` or, for line 0, `<source>: `.
function _write_location(io::Base.IO, source::String, line::Int)
    write(io, source)
    if line != 0
        write(io, ":")
        print(io, line)
    end
    write(io, ": ")
    return nothing
end

# Prints a message one part at a time. Printing the `LazyString` itself materializes it through a
# call on each part's runtime type, which neither type-infers nor trims; here each part is tested
# against the types the messages interpolate (a `LazyString` part is the message a constructor
# wrapped with `LazyString(msg)`) and printed with a call on that concrete type.
# Reading a part of the abstract `parts` tuple boxes it, so the rendering functions are exempt
# from `:noboxing`; they run only on the error path.
function _print_message(io::Base.IO, m::LazyString)
    parts = m.parts
    for i in eachindex(parts)
        p = parts[i]
        if p isa String
            write(io, p)
        elseif p isa LazyString
            _print_message(io, p)
        elseif p isa SubString{String}
            write(io, p)
        elseif p isa Symbol
            print(io, p)
        elseif p isa Char
            print(io, p)
        elseif p isa Bool
            print(io, p)
        elseif p isa Int
            print(io, p)
        elseif p isa Float64
            print(io, p)
        else
            throw(ArgumentError("errors: a message part is not a String, LazyString, SubString, Symbol, Char, Bool, Int or Float64"))
        end
    end
    return nothing
end

# The methods below write one piece per call: a varargs `print(io, a, b, …)` boxes its arguments.
function Base.showerror(io::Base.IO, e::InputError)
    write(io, "input: ")
    _write_location(io, e.source, e.line)
    _print_message(io, e.msg)
    return nothing
end

function Base.showerror(io::Base.IO, e::BasisError)
    write(io, "basis: ")
    _print_message(io, e.msg)
    return nothing
end

function Base.showerror(io::Base.IO, e::SCFNotConverged)
    write(io, "scf: ")
    _print_message(io, e.msg)
    return nothing
end

function Base.showerror(io::Base.IO, e::OptimizationFailed)
    write(io, "optimization: ")
    _print_message(io, e.msg)
    return nothing
end

function Base.showerror(io::Base.IO, e::NotImplemented)
    write(io, "not implemented (")
    write(io, e.section)
    write(io, "): ")
    _print_message(io, e.msg)
    return nothing
end

function Base.showerror(io::Base.IO, e::FunctionalUnavailable)
    write(io, "functional: ")
    _print_message(io, e.msg)
    return nothing
end

function Base.showerror(io::Base.IO, e::NumericalError)
    write(io, "numerical: ")
    _print_message(io, e.msg)
    return nothing
end

function Base.show(io::Base.IO, w::InputWarning)
    write(io, "input: ")
    _write_location(io, w.source, w.line)
    _print_message(io, w.msg)
    return nothing
end

export QuantumEngineError, InputError, BasisError, SCFNotConverged, OptimizationFailed,
    NotImplemented, FunctionalUnavailable, NumericalError, InputWarning

@strict_exempt :noalloc :noboxing showerror reason = "entry point: renders a LazyString message, whose parts are an abstract Tuple"
@strict_exempt :noalloc :noboxing types = (Base.IO, InputWarning) show reason = "entry point: renders a LazyString message, whose parts are an abstract Tuple"
@strict_exempt :noalloc _write_location reason = "entry point: writes the location of the `showerror` and `show` messages"
@strict_exempt :noalloc :noboxing _print_message reason = "entry point: renders the LazyString message for showerror and show"
@strict_exempt :noalloc InputError reason = "entry point: error constructor"
@strict_exempt :noalloc BasisError reason = "entry point: error constructor"
@strict_exempt :noalloc SCFNotConverged reason = "entry point: error constructor"
@strict_exempt :noalloc OptimizationFailed reason = "entry point: error constructor"
@strict_exempt :noalloc NotImplemented reason = "entry point: error constructor"
@strict_exempt :noalloc FunctionalUnavailable reason = "entry point: error constructor"
@strict_exempt :noalloc NumericalError reason = "entry point: error constructor"
@strict_exempt :noalloc InputWarning reason = "entry point: warning constructor"

# Last statement: the policy checks the methods defined above it.
@strict_policy (:typestable, :noalloc)

end
