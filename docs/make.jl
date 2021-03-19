using Documenter, BayesianExperiments

makedocs(
    sitename="BayesianExperiments.jl",
    modules=[BayesianExperiments],
    authors = "Jiangtao Fu",
    doctest = false,
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "examples_conjugate.md", 
            "examples_bayesfactor.md"],
        "API" => "api.md",
        "Tutorials" => [ 
            "tutorials/sequential_testing_conjugate_models.md",
            "tutorials/type_s_error.md",
            "tutorials/bayes_factor_optional_stopping.md"
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