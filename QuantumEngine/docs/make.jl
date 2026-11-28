using Documenter, DocumenterVitepress, QuantumEngine

# The site lives in the `QuantumEngine/` folder of the gh-pages branch, next to the sites of the
# other packages. `deploy_url` gives VitePress the matching base path, and `dirname` in
# `deploydocs` the matching folder; the two must agree.
makedocs(;
    sitename = "QuantumEngine",
    authors = "el_oso",
    modules = [QuantumEngine],
    repo = Documenter.Remotes.GitHub("el-oso", "PureAtomistic.jl"),
    warnonly = true,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/el-oso/PureAtomistic.jl",
        devbranch = "main",
        devurl = "dev",
        deploy_url = "https://el-oso.github.io/PureAtomistic.jl/QuantumEngine",
    ),
    draft = false,
    source = "src",
    build = "build",
    pages = [
        "Home" => "index.md",
        "Input files" => "input.md",
        "Basis sets" => "basis.md",
        "API" => "api.md",
        "Dependencies" => "dependencies.md",
        "Parity with the oracle" => "parity.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/PureAtomistic.jl",
    devbranch = "main",
    push_preview = true,
    dirname = "QuantumEngine",
)
