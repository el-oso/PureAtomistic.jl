using Documenter, DocumenterVitepress, Literate, PureMLIP

# The example scripts in `examples/` run here: Literate executes each one with `src/examples/`
# as the working directory, writes its page there, and the script writes its media next to the
# page. An error in a script stops the build.
const EXAMPLES = ["load_model", "energy_forces", "vacancy", "lattice_constant", "energy_workspace", "pureblas", "tensors"]
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
mkpath(EXAMPLES_OUT)
for name in EXAMPLES
    Literate.markdown(joinpath(@__DIR__, "examples", name * ".jl"), EXAMPLES_OUT; execute = true, credit = false)
end

makedocs(;
    sitename = "PureMLIP.jl",
    authors = "el_oso",
    modules = [PureMLIP],
    repo = Documenter.Remotes.GitHub("el-oso", "PureAtomistic.jl"),
    warnonly = true,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/el-oso/PureAtomistic.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    draft = false,
    source = "src",
    build = "build",
    pages = [
        "Home" => "index.md",
        "Equivariant math" => "equivariant.md",
        "MACE model" => "model.md",
        "Examples" => ["gallery.md"; ["examples/$name.md" for name in EXAMPLES]],
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/PureAtomistic.jl",
    devbranch = "main",
    push_preview = true,
)
