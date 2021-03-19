using Documenter, BayesianExperiments

sourcepath = joinpath(@__DIR__, "src")

# cleanup
filename = joinpath(sourcepath, "tutorials/bayes_factor_optional_stopping.md")
filecontent = open(filename, "r") do io
        read(io, String)
    end

filecontent = replace(filecontent, r"\e\[[0-9]+m"=>"")
open(filename, "w") do io
    write(io, filecontent)
end

makedocs(
    sitename="BayesianExperiments.jl",
    modules=[BayesianExperiments],
    authors = "Jiangtao Fu",
    doctest = false,
    pages = [
        "Home" => "index.md",
        "Getting Started" => "examples.md",
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