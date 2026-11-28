# The basis search path and the lookup of a basis name's file.

"""
    basis_search_paths(; env = ENV, exe_dir = nothing) -> Vector{String}

The directories searched for basis files, in order:

1. `"basis_sets"`, `"../basis_sets"` and `"../../basis_sets"`, relative to the working directory
   at the time of the lookup;
2. `env["QUANTUM_ENGINE_BASIS_DIR"]`, when it is set and not empty;
3. when `exe_dir` is given, `basis_sets`, `../basis_sets` and `../../basis_sets` under `exe_dir`;
4. the package's `data/basis_sets` directory, when the package directory is known.

A `basis_sets` directory in the working directory therefore shadows the package data.
"""
function basis_search_paths(; env::AbstractDict = ENV, exe_dir::Union{Nothing, AbstractString} = nothing)
    paths = String["basis_sets", "../basis_sets", "../../basis_sets"]
    dir = get(env, "QUANTUM_ENGINE_BASIS_DIR", "")
    isempty(dir) || push!(paths, String(dir))
    if exe_dir !== nothing
        push!(paths, joinpath(exe_dir, "basis_sets"))
        push!(paths, joinpath(exe_dir, "..", "basis_sets"))
        push!(paths, joinpath(exe_dir, "..", "..", "basis_sets"))
    end
    d = pkgdir(@__MODULE__)
    d === nothing || push!(paths, joinpath(d, "data", "basis_sets"))
    return paths
end

"""
    locate_basis_file(name::AbstractString; paths = basis_search_paths()) -> String

The absolute path of the file of the basis name `name`: for each stem of `candidate_stems(name)`
in turn, each directory of `paths` in turn, the first `<dir>/<stem>.gbs` that exists. The file is
not read. A name containing `/` or `\\` throws a `BasisError`, on every platform, and so does a name
with no file, listing the stems and directories tried.
"""
function locate_basis_file(name::AbstractString; paths::AbstractVector{<:AbstractString} = basis_search_paths())
    for b in codeunits(name)
        if b == UInt8('/') || b == UInt8('\\')
            throw(BasisError(lazy"basis name \"$(String(name))\" contains a path separator; put the file in a directory on the search path or set QUANTUM_ENGINE_BASIS_DIR"))
        end
    end
    stems = candidate_stems(name)
    for stem in stems, dir in paths
        file = joinpath(dir, stem * ".gbs")
        isfile(file) && return abspath(file)
    end
    tried = _comma_list(stems)
    searched = _comma_list(paths)
    throw(BasisError(lazy"no file for \"$(String(name))\" (stems $tried; searched: $searched)"))
end

# The strings of `v` separated by `", "`, written with `write` calls on the concrete element type:
# `Base.join` fails the `:noboxing` scan.
function _comma_list(v::AbstractVector{<:AbstractString})
    io = IOBuffer()
    for (k, s) in enumerate(v)
        k > 1 && write(io, ", ")
        write(io, s)
    end
    return String(take!(io))
end
