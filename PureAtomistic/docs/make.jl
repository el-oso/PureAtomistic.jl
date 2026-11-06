using Documenter, DocumenterVitepress, Literate, PureAtomistic

# The example scripts in `examples/` read the stored records in `bench/results/` and plot them with
# CairoMakie; they run no dynamics. Literate executes each one with `src/examples/` as the working
# directory and writes its page there; the script writes its figures next to the page. An error in
# a script stops the build. The animations in `src/assets/` are rendered ahead of time by
# `render_animations.jl`.
const EXAMPLES = ["nve_conservation", "thermostat_check", "vacancy_hop_md"]
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
mkpath(EXAMPLES_OUT)
for name in EXAMPLES
    Literate.markdown(joinpath(@__DIR__, "examples", name * ".jl"), EXAMPLES_OUT; execute = true, credit = false)
end

# The site lives in the `PureAtomistic/` folder of the gh-pages branch, next to PureMLIP's site at
# its root. `deploy_url` gives VitePress the matching base path, and `dirname` in `deploydocs` the
# matching folder; the two must agree.
makedocs(;
    sitename = "PureAtomistic",
    authors = "el_oso",
    modules = [PureAtomistic],
    repo = Documenter.Remotes.GitHub("el-oso", "PureAtomistic.jl"),
    warnonly = true,
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/el-oso/PureAtomistic.jl",
        devbranch = "main",
        devurl = "dev",
        deploy_url = "https://el-oso.github.io/PureAtomistic.jl/PureAtomistic",
    ),
    draft = false,
    source = "src",
    build = "build",
    pages = [
        "Home" => "index.md",
        "Examples" => ["gallery.md"; ["examples/$name.md" for name in EXAMPLES]],
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/PureAtomistic.jl",
    devbranch = "main",
    push_preview = true,
    dirname = "PureAtomistic",
)
