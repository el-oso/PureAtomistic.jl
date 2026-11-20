using Documenter, DocumenterVitepress, PureTaylor

# The site lives in the `PureTaylor/` folder of the gh-pages branch, next to the sites of the other
# packages. `deploy_url` gives VitePress the matching base path, and `dirname` in `deploydocs` the
# matching folder; the two must agree.
makedocs(;
    sitename = "PureTaylor",
    authors = "el_oso",
    modules = [PureTaylor],
    repo = Documenter.Remotes.GitHub("el-oso", "PureAtomistic.jl"),
    warnonly = true,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/el-oso/PureAtomistic.jl",
        devbranch = "main",
        devurl = "dev",
        deploy_url = "https://el-oso.github.io/PureAtomistic.jl/PureTaylor",
    ),
    draft = false,
    source = "src",
    build = "build",
    pages = ["Home" => "index.md", "API" => "api.md"],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/PureAtomistic.jl",
    devbranch = "main",
    push_preview = true,
    dirname = "PureTaylor",
)
