using Documenter, DocumenterVitepress, PureMLIP

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
    pages = ["Home" => "index.md", "Equivariant math" => "equivariant.md", "MACE model" => "model.md"],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/PureAtomistic.jl",
    devbranch = "main",
    push_preview = true,
)
