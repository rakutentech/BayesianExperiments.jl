"""
    PosteriorSample

Samples from the poseteriors distribution.
"""
abstract type PosteriorSample end
abstract type SinglePosteriorSample <: PosteriorSample end
abstract type MultiplePosteriorSample <: PosteriorSample end

struct BernoulliPosteriorSample <: SinglePosteriorSample 
    θ::Vector{Real}
end 

struct ExponentialPosteriorSample <: SinglePosteriorSample 
    θ::Vector{Real} # scale, inverse of rate λ in wikipedia
end

struct PoissonPosteriorSample <: SinglePosteriorSample
    λ::Vector{Real} 
end

struct NormalPosteriorSample <: MultiplePosteriorSample 
    μ::Vector{Real}
    σ²::Vector{Real}
end

struct LogNormalPosteriorSample <: MultiplePosteriorSample 
    μ_logx::Vector{Real}
    σ²_logx::Vector{Real}
    μ_x::Vector{Real}
    σ²_x::Vector{Real}
end

function mean(postsample::PosteriorSample, parameter::Symbol)
    return Dict(parameter => mean(getfield(postsample, parameter)))
end
mean(postsample::BernoulliPosteriorSample) = mean(postsample, :θ)
mean(postsample::ExponentialPosteriorSample) = mean(postsample, :θ)

function mean(postsample::MultiplePosteriorSample) 
    result = Dict{Symbol,Real}()
    for field in fieldnames(typeof(postsample))
        result[field] = mean(getfield(postsample, field))
    end
    return result
end

"""
    ProbabilisticModel

`ProbabilisticModel` is a a model of the parameters that we are interested in.
The model is defined by its prior distribution and likelihood function.
"""
abstract type ProbabilisticModel end
