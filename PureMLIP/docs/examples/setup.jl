# Shared by the example scripts: the checkpoint download, fcc structures and the media directory.
using PureMLIP, LinearAlgebra

# `checkpoint_path(:mpa0)` downloads mace-mpa-0-medium.model once, into
# PUREMLIP_CHECKPOINT_DIR, and checks its SHA-256 (the same code the tests use).
include(pkgdir(PureMLIP, "test", "reference", "checkpoints.jl"))

# Positions (3 × 4N³) and cell (columns are lattice vectors) of an N×N×N block of
# conventional fcc cells with lattice constant `a`, in Å.
function fcc(a, N)
    basis = ([0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5])
    positions = reduce(hcat, [a .* ([i, j, k] .+ b) for i in 0:(N - 1) for j in 0:(N - 1) for k in 0:(N - 1) for b in basis])
    return positions, Matrix(a * N * I(3)) .* 1.0
end

# Media files go next to the generated page: the docs build runs each script with the page's
# directory as the working directory; a script run on its own writes to that same directory.
const STANDALONE = endswith(dirname(abspath(PROGRAM_FILE)), joinpath("docs", "examples"))
media(name) = STANDALONE ? joinpath(mkpath(pkgdir(PureMLIP, "docs", "src", "examples")), name) : name
