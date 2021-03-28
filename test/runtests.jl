using Test
using Random
using Distributions: Bernoulli, LogNormal, Poisson
using Statistics:std

using BayesianExperiments
include("utils.jl")

include("util.jl")
include("data.jl")
include("experiment.jl")
include("conjugate.jl")
include("bayesfactor.jl")
include("simulation.jl")