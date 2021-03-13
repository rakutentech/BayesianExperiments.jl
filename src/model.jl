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

"""
    ConjugateModel <: ProbabilisticModel

`ConjugateModel` is a `ProbabilisticModel` with a conjugate prior of the 
corresponding likelihood function.
"""
abstract type ConjugateModel <: ProbabilisticModel end

Base.show(io::IO, model::ConjugateModel) = print(io, 
    typeof(model), ": ", split(string(model.dist), ".", limit=2)[2])

mean(model::ConjugateModel) = mean(model.dist)


"""
    BernoulliModel <: ConjugateModel

Bernoulli likelihood with Beta distribution as the conjugate prior.

```julia
BernoulliModel(α, β)              # construct a BernoulliModel

update!(model, stats)             # update model with statistics from data
samplepost(model, numsamples)    # sampling from the posterior distribution
samplestats(model, numsamples)   # sampling statistics from the data generating distribution
```

"""
mutable struct BernoulliModel <: ConjugateModel
    dist::Beta 
    function BernoulliModel(α, β)
        return new(Beta(α, β))
    end
end

defaultparams(::BernoulliModel) = [:θ]

function update!(model::BernoulliModel, stats::BernoulliStatistics)
    numsuccesses = stats.s 
    numtrials = stats.n
    α = model.dist.α + numsuccesses
    β = model.dist.β + numtrials - numsuccesses
    model.dist = Beta(α, β)
    return nothing
end

"""
    ExponentialModel <: ConjugateModel

Exponential likelihood with Gamma distribution as the conjugate prior.

```julia
ExponentialModel(α, β)            # construct a ExponentialModel

update!(model, stats)             # update model with statistics from data
samplepost(model, numsamples)    # sampling from the posterior distribution
samplestats(model, numsamples)   # sampling statistics from the data generating distribution
```
"""
mutable struct ExponentialModel <: ConjugateModel
    dist::Gamma
    function ExponentialModel(α, θ)
        return new(Gamma(α, θ))
    end
end

defaultparams(::ExponentialModel) = [:θ]

function update!(model::ExponentialModel, stats::ExponentialStatistics)
    n = stats.n
    x̄ = stats.x̄
    α = model.dist.α + n
    θ = model.dist.θ / (1 + model.dist.θ * n * x̄)
    model.dist = Gamma(α, θ)
    return nothing
end

"""
    NormalModel <: ConjugateModel    

Normal likelihood and Normal Inverse Gamma distribution as the 
conjugate prior.

## Parameters

- μ: mean of normal distribution
- v:  scale variance of Normal 
- α:  shape of Gamma distribution
- θ:  scale of Gamma distribution

```julia
NormalModel(μ, v, α, θ)           # construct a NormalModel

update!(model, stats)             # update model with statistics from data
samplepost(model, numsamples)    # sampling from the posterior distribution
samplestats(model, numsamples)   # sampling statistics from the data generating distribution
```

## References

- The update rule for Normal distribution is based on this [lecture notes](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter9.pdf).
"""
mutable struct NormalModel{S} <: ConjugateModel
    dist::S
    function NormalModel{Normal}(μ::Real, σ::Real)
        T = promote(typeof(μ), typeof(σ))
        return new(Normal{T}(μ, σ))
    end
    function NormalModel{NormalInverseGamma}(μ::Real, v::Real, α::Real, θ::Real)
        return new(NormalInverseGamma(μ, v, α, θ))
    end
end

NormalModel(μ::T, v::T, α::T, θ::T) where T <: Real = NormalModel{NormalInverseGamma}(μ, v, α, θ)
NormalModel(μ::T, σ::T) where T <: Real = NormalModel{Normal}(μ, σ)

defaultparams(::NormalModel{NormalInverseGamma}) = [:μ]

"""
    LogNormalModel(μ, v, α, θ)

A model with Normal likelihood and Normal Inverse distribution with log transformed data.
Notice `LogNormal` in `Distributions.jl` takes mean and standard deviation of \$\\log(x)\$ 
instead of \$x\$ as the input parameters.

```julia
LogNormalModel(μ, v, α, θ) # construct a LogNormalModel

lognormalparams(μ_logx, σ²_logx) # convert normal parameters to log-normal parameters
update!(model, stats)              # update model with statistics from data
samplepost(model, numsamples)     # sampling from the posterior distribution
samplestats(model, numsamples)    # sampling statistics from the data generating distributio
```

"""
mutable struct LogNormalModel{S} <: ConjugateModel
    dist::S
    function LogNormalModel{NormalInverseGamma}(μ::Real, v::Real, α::Real, θ::Real)
        return new(NormalInverseGamma(μ, v, α, θ))
    end
end

LogNormalModel(μ::T, v::T, α::T, θ::T) where T <: Real = LogNormalModel{NormalInverseGamma}(μ, v, α, θ)

function lognormalparams(μ_logx, σ²_logx) 
    μ_x  = @. exp(μ_logx + σ²_logx / 2)
    σ²_x = @. (exp(σ²_logx) - 1) * exp(2 * μ_logx + σ²_logx)
    return (μ_x, σ²_x)
end

defaultparams(::LogNormalModel{NormalInverseGamma}) = [:μ_x]

function update!(model::Union{NormalModel{NormalInverseGamma},LogNormalModel{NormalInverseGamma}}, 
        stats::Union{NormalStatistics,LogNormalStatistics})
    n, x̄, sdx = getall(stats)
    ΣΔx² = sdx^2 * (n - 1)

    μ0 = model.dist.μ
    v0 = model.dist.v
    α0 = model.dist.α
    θ0 = model.dist.θ
    inv_v0 = 1.0 / v0
    inv_v = inv_v0 + n

    μ = (inv_v0 * μ0 + n * x̄) / inv_v
    v = 1 / inv_v
    α = α0 + n / 2
    θ = θ0 + 0.5 * (ΣΔx² + (n * inv_v0) * (x̄ - μ0)^2 * v)
    model.dist = NormalInverseGamma(μ, v, α, θ)
    return nothing
end

abstract type ChainOperator end

struct MultiplyOperator <: ChainOperator end
(m::MultiplyOperator)(a, b) = .*(a, b)

"""
    ChainedModel <: ProbabilisticModel

`ChainedModel` is a combination of `ConjugateModel`s chained by the specified operator.
It can be used to model a multiple step process.
"""
struct ChainedModel <: ProbabilisticModel
    models::Vector{ConjugateModel}
    operators::Vector{ChainOperator}
    function ChainedModel(models::Vector{ConjugateModel}, operators)
        length(models) - 1 == length(operators) || error("need to specify (number of model - 1) chaining operators")
        return new(models, operators)
    end
end

function ChainedModel(models::Vector{ConjugateModel})
    operators = [MultiplyOperator() for _ in 1:(length(models) -1)]
    return ChainedModel(models, operators)
end

function update!(chainedmodel::ChainedModel, listofstats::Vector{T}) where T <: ModelStatistics
    @assert(length(chainedmodel.models) == length(listofstats), 
            "Number of model should be equal to number of statistics.")
    for (model, stats) = zip(chainedmodel.models, listofstats)
        update!(model, stats)
    end
    return nothing
end


function defaultparams(chainedmodel::ChainedModel) 
    params = Symbol[]
    for model in chainedmodel.models
        push!(params, defaultparams(model)[1])
    end
    return params
end

"""
    samplepost(model, numsamples)

Sample from the posterior distribution of the model.
"""
function samplepost(model::BernoulliModel, numsamples::Int)
    return BernoulliPosteriorSample(rand(model.dist, numsamples))
end

function samplepost(model::ExponentialModel, numsamples::Int)
    # Gamma distribution generates the rate (λ) of Exponential distribution
    # convert it to scale θ to make it consistent with Distributions.jl
    λs = rand(model.dist, numsamples)
    θs = 1 ./ λs
    return ExponentialPosteriorSample(θs)
end

function samplepost(model::NormalModel{NormalInverseGamma}, numsamples::Int) 
    μ, σ² = rand(model.dist, numsamples)
    return NormalPosteriorSample(μ, σ²)
end

function samplepost(model::LogNormalModel{NormalInverseGamma}, numsamples::Int)
    # sample for normal means and variances
    μ_logx, σ²_logx = rand(model.dist, numsamples)
    μ_x, σ²_x = lognormalparams(μ_logx, σ²_logx)
    return LogNormalPosteriorSample(μ_logx, σ²_logx, μ_x, σ²_x)
end

function samplepost(model::T, parameter::Symbol, numsamples::Int) where T <: ConjugateModel
    samples = samplepost(model, numsamples)
    return getfield(samples, parameter)
end

function samplepost(model::T, parameters::Vector{Symbol}, numsamples::Int)  where T <: ConjugateModel
    parameter = toparameter(parameters)
    return samplepost(model, parameter, numsamples)
end

function samplepost(models::Vector{T}, parameters::Vector{Symbol}, numsamples::Int) where T <: ProbabilisticModel
    samples = Array{Float64,2}(undef, numsamples, length(models))
    for (modelindex, model) in enumerate(models)
        sample = samplepost(model, parameters, numsamples)
        samples[:, modelindex] = sample
    end
    return samples
end

function samplepost(chainedmodel::ChainedModel, parameters::Vector{Symbol}, numsamples::Int)
    length(parameters) == length(chainedmodel.models) || 
        throw(ArgumentError("Number of parameters must be equal to number of chained models"))
    models = chainedmodel.models
    operators = chainedmodel.operators
    post_samples = [samplepost(model, parameter, numsamples) for (model, parameter) in zip(models, parameters)]
    chainedsample = post_samples[1]
    for i = 2:length(post_samples)
        chainedsample = operators[i - 1](chainedsample, post_samples[i]) 
    end
    return chainedsample
end

"""
    samplestats(model, dist, numsamples)

Sample from the distribution of the data generating process, and 
calculate the corresponding statistics for the model.
"""
function samplestats(::BernoulliModel, dist::Bernoulli, numsamples::Integer)
    return BernoulliStatistics(rand(dist, numsamples))
end

function samplestats(::ExponentialModel, dist::Exponential, numsamples::Integer)
    return ExponentialStatistics(rand(dist, numsamples))
end

function samplestats(::NormalModel{NormalInverseGamma}, dist::Normal, numsamples::Integer)
    return NormalStatistics(rand(dist, numsamples))
end

function samplestats(::LogNormalModel{NormalInverseGamma}, dist::LogNormal, numsamples::Integer)
    return LogNormalStatistics(rand(dist, numsamples))
end

function samplestats(chainedmodel::ChainedModel, dists::Vector{T}, listofnumsamples::Vector{Int}) where T
    listofstats = Vector{ModelStatistics}()
    for (model, dist, numsamples) in zip(chainedmodel.models, dists, listofnumsamples)
        push!(listofstats, samplestats(model, dist, numsamples))
    end
    return listofstats
end
