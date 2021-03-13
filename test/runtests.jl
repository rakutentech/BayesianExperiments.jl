using Test
using Random
using Distributions: Bernoulli, LogNormal
using Statistics:std

using BayesianExperiments
include("utils.jl")

include("util.jl")
include("experiment.jl")
include("model.jl")
include("simulation.jl")