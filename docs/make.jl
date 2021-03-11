using Documenter, BayesianExperiments

makedocs(
    modules=[BayesianExperiments],
    sitename="BayesianExperiments.jl",
    authors = "Jiangtao Fu",
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
