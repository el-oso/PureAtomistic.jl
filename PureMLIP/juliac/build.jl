#!/usr/bin/env julia
# Build an executable with juliac --trim. `julia juliac/build.jl` builds `mace_inspect` from
# `entry.jl`; `julia juliac/build.jl <name> <entry file in juliac/>` builds the executable `name`.
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
const EXE, ENTRY = length(ARGS) == 2 ? (ARGS[1], joinpath(@__DIR__, ARGS[2])) : ("mace_inspect", joinpath(@__DIR__, "entry.jl"))

const JCFLAGS = `--output-exe $EXE --bundle $OUTDIR --experimental --trim=safe --verbose`
@info "juliac: JuliaC.jl from $TOOLENV"
run(`$(Base.julia_cmd()) --startup-file=no --project=$TOOLENV -e "using Pkg; Pkg.instantiate()"`)
run(`$(Base.julia_cmd()) --startup-file=no --project=$TOOLENV -e "using JuliaC; JuliaC.main(ARGS)" -- --project=$ROOT $JCFLAGS $ENTRY`)
@info "built" OUTDIR
