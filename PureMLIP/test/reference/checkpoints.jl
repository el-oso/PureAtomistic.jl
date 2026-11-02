using SHA, Downloads

# File name, release URL and SHA-256 of each published checkpoint.
const CHECKPOINT_SOURCES = Dict(
    :mp0b3 => (
        "mace-mp-0b3-medium.model",
        "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
        "2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012",
    ),
    :mpa0 => (
        "mace-mpa-0-medium.model",
        "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
        "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638",
    ),
)

# Path of the checkpoint `name` in PUREMLIP_CHECKPOINT_DIR (default: a scratch directory of the
# first depot), downloaded on first use and verified by SHA-256. A file with the wrong digest is
# deleted.
function checkpoint_path(name::Symbol)
    file, url, sha = CHECKPOINT_SOURCES[name]
    dir = get(ENV, "PUREMLIP_CHECKPOINT_DIR", joinpath(first(DEPOT_PATH), "scratchspaces", "PureMLIP-checkpoints"))
    mkpath(dir)
    path = joinpath(dir, file)
    if !isfile(path)
        Downloads.download(url, path * ".part")
        mv(path * ".part", path; force = true)
    end
    digest = bytes2hex(open(sha256, path))
    if digest != sha
        rm(path)
        error("SHA-256 of $path is $digest, expected $sha; the file was deleted")
    end
    return path
end
