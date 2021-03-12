using Documenter, BayesianExperiments

makedocs(
    sitename="BayesianExperiments.jl",
    modules=[BayesianExperiments],
    authors = "Jiangtao Fu",
    doctest = false,
    pages = [
        "Home" => "index.md",
        "Getting Started" => "basic_examples.md",
        "API" => "api.md",
        "Tutorials" => [ 
            "tutorials/sequential_experiment_two_models.md",
            "tutorials/fixed_vs_sequentail_type_s_error.md"
        ]
    ]
)

devurl = "dev"
deploydocs(
    repo="github.com/rakutentech/BayesianExperiments.jl.git",
    devbranch = "main",
    branch = "gh-pages",
    devurl=devurl,
    versions = ["stable" => "v^", "v#.#", "dev" => devurl]
)