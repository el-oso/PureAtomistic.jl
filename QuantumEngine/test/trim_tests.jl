# The rendering of every error type and of `InputWarning` into an `IOBuffer`, which the juliac entry
# prints.
@testitem "TrimCheck: showerror of the error types and show of InputWarning" tags = [:checks] begin
    using TrimCheck
    @validate(
        init = begin
            using QuantumEngine
        end,
        Base.showerror(IOBuffer, InputError),
        Base.showerror(IOBuffer, BasisError),
        Base.showerror(IOBuffer, SCFNotConverged),
        Base.showerror(IOBuffer, OptimizationFailed),
        Base.showerror(IOBuffer, NotImplemented),
        Base.showerror(IOBuffer, FunctionalUnavailable),
        Base.showerror(IOBuffer, NumericalError),
        Base.show(IOBuffer, InputWarning),
    )
end
