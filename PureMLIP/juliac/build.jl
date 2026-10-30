#!/usr/bin/env julia
# Build the `mace_inspect` executable with juliac --trim. Run: `julia juliac/build.jl`.
# The executable lands in `juliac/build/bin` with its libraries alongside.
# The tool environment (`juliac/toolenv`) requires Julia 1.13, while the package itself
# supports Julia 1.12.
#
# The driver is the registered package JuliaC.jl, resolved from `juliac/toolenv` (the
# environment keeps no tracked Manifest). It runs in its own environment, so the package
# being built is named explicitly with `--project=$ROOT`.

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTDIR = joinpath(@__DIR__, "build")
mkpath(OUTDIR)

const TOOLENV = joinpath(@__DIR__, "toolenv")
const ENTRY = joinpath(@__DIR__, "entry.jl")

const JCFLAGS = `--output-exe mace_inspect --bundle $OUTDIR --experimental --trim=safe --verbose`
@info "juliac: JuliaC.jl from $TOOLENV"
run(`$(Base.julia_cmd()) --startup-file=no --project=$TOOLENV -e "using Pkg; Pkg.instantiate()"`)
run(`$(Base.julia_cmd()) --startup-file=no --project=$TOOLENV -e "using JuliaC; JuliaC.main(ARGS)" -- --project=$ROOT $JCFLAGS $ENTRY`)
@info "built" OUTDIR
