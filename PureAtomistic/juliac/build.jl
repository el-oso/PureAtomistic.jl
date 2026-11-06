#!/usr/bin/env julia
# Build the `md_run` executable with juliac --trim. Run: `julia juliac/build.jl`.
# The executable lands in `juliac/build/bin` with its libraries alongside.
# The build driver is JuliaC.jl, resolved from PureMLIP's `juliac/toolenv` (Julia 1.13); the
# package being built is named explicitly with `--project=$ROOT`.

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTDIR = joinpath(@__DIR__, "build")
mkpath(OUTDIR)

const TOOLENV = normpath(joinpath(ROOT, "..", "PureMLIP", "juliac", "toolenv"))
const ENTRY = joinpath(@__DIR__, "entry.jl")

const JCFLAGS = `--output-exe md_run --bundle $OUTDIR --experimental --trim=safe --verbose`
@info "juliac: JuliaC.jl from $TOOLENV"
run(`$(Base.julia_cmd()) --startup-file=no --project=$TOOLENV -e "using Pkg; Pkg.instantiate()"`)
run(`$(Base.julia_cmd()) --startup-file=no --project=$TOOLENV -e "using JuliaC; JuliaC.main(ARGS)" -- --project=$ROOT $JCFLAGS $ENTRY`)
@info "built" OUTDIR
