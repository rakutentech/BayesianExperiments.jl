"""
    Experiment

An experiment has the models to compare, a stopping rule to make decision. 
"""
abstract type Experiment end

"""
    ExperimentABN{T,n} <: Experiment

An experiment with stopping rule of type `T` and `n` models. Models must have the same
`ProbabilisticModel` type.
"""
mutable struct ExperimentABN{T,n} <: Experiment
    models::Dict{String,ProbabilisticModel}
    modelnames::Vector{String}
    rule::T
    winner::Union{String,Nothing}

    function ExperimentABN{T,n}(models, rule::T; modelnames=nothing) where {T <: StoppingRule,n}
        length(models) >= 2 || throw(ArgumentError("Number of models needs to be greater than 1."))
        length(models) == n || throw(ArgumentError("Number of models needs to be equal to n."))
        
        if modelnames === nothing
            modelnames = ["control"]
            for i in 1:(length(models) - 1)
                push!(modelnames, "variant $i")
            end
        end
        @assert(length(models) == length(modelnames), 
                "Number of model names is different from number of models.")
        return new(Dict(zip(modelnames, models)), modelnames, rule, nothing)
    end
end

function ExperimentABN(models, rule::T; modelnames=nothing) where T <: StoppingRule
    return ExperimentABN{T,length(models)}(models, rule, modelnames=modelnames)
end

"""
    ExperimentAB(models, rule; modelnames=nothing) where T <: StoppingRule

An experiment of `ExperimentABN` with two models.
"""
function ExperimentAB(models, rule::T; modelnames=nothing) where T <: StoppingRule
    return ExperimentABN{T,2}(models, rule, modelnames=modelnames)
end

function Base.show(io::IO, experiment::ExperimentABN)
    modelinfo = ""
    for (i, modelname) in enumerate(experiment.modelnames)
        modelinfo = string(modelinfo, "  \"", modelname, "\": ", experiment.models[modelname])
        if i < length(experiment.modelnames)
            modelinfo = string(modelinfo, "\n")
        end
    end
    print(io, typeof(experiment), "\n", modelinfo)
end

function update!(experiment::ExperimentABN, 
                 listofstats::Union{Vector{T},Vector{Vector{T}}}) where T <: ModelStatistics
    for (i, stats) in enumerate(listofstats)
        modelname = experiment.modelnames[i]
        update!(experiment, modelname, stats)
    end
    return nothing
end

function update!(experiment::ExperimentABN, 
    modelname::String, stats::Union{T,Vector{T}}) where T <: ModelStatistics
    model = experiment.models[modelname]
    update!(model, stats)
    return nothing
end


function defaultparams(experiment::ExperimentABN)
    model1 = [model for (modelname, model) in experiment.models][1]
    return defaultparams(model1)
end

"""
    ExperimentBF{M} <: Experiment

`BayesFactorExperiment` is an experiment using a Bayes Factor between the 
null and alternative hypothesis as the stopping rule.

# Constructors

    ExperimentBF(model, p0, rule; kwargs...)

## Arguments

- `model::M`: Prior of effect size of alternative hypothesis 
- `p0::Float64`: Probablity of null hypothesis
- `rejection::Bool`: Decision to reject the null hypothesis or not
- `rule::BayesFactorThresh`: Stopping rule using Bayes Factor as the threshold

## Keywords

- `stats`: Statistics for calculating the bayes factor. Default is `nothing`.
- `names`: Names of the hypotheses. Default is `["null", "alternative"]`.

"""
mutable struct ExperimentBF{M<:BayesFactorModel} <: Experiment
    model::M
    p0::Float64
    winner::Union{String, Nothing}
    rule::BayesFactorThresh
    stats::Union{NormalStatistics, TwoNormalStatistics, Nothing}
    names::Vector{String} 

    function ExperimentBF(;
        model, rule, p0=0.5, stats=nothing, names=["null", "alternative"])
        return new{typeof(model)}(model, p0, nothing, rule, stats, names)
    end
end

function update!(experiment::ExperimentBF, 
    stats::Union{NormalStatistics, TwoNormalStatistics})
    if experiment.stats === nothing
        experiment.stats = stats
    else
        experiment.stats = update!(experiment.stats, stats)
    end
    return nothing
end

function bayesfactor(experiment::ExperimentBF{NormalEffectSize})
    experiment.stats !== nothing || error("Experiment statistics is not initialized.")
    return bayesfactor(experiment.model, experiment.stats)
end

function bayesfactor(experiment::ExperimentBF{StudentTEffectSize})
    experiment.stats !== nothing || error("Experiment statistics is not initialized.")
    tstats = StudentTStatistics(experiment.stats)
    return bayesfactor(experiment.model, tstats)
end

"""
    expectedloss(modelA, modelB; lossfunc, numsamples)

Approximating the expected loss for choosing model A over model B.

    expectedlosses(experiment::ExperimentABN; lossfunc, numsamples)

Approximating the expected loss for all models in the experiment. 
For A/B/N experiment, the expected loss for each model is the maximum of the losses 
by comparing that model to all the other models.
"""
function expectedloss(modelA::T, modelB::T, parameters; 
    lossfunc=upliftloss, numsamples=10_000) where T <: ProbabilisticModel
    sampleA = samplepost(modelA, parameters, numsamples)
    sampleB = samplepost(modelB, parameters, numsamples)
    length(sampleA) == length(sampleB) || 
        throw(ArgumentError("Approximating expected loss requires equal length."))
    losschooseA = expectedloss(sampleA, sampleB, lossfunc, numsamples) 
    return losschooseA
end

function expectedloss(modelA::T, modelB::T; lossfunc=upliftloss, 
        numsamples=10_000) where T <: ProbabilisticModel
    parameters = defaultparams(modelA) 
    return expectedloss(modelA, modelB, parameters, lossfunc=lossfunc, numsamples=numsamples)
end

function expectedloss(sampleA::Vector{T}, sampleB::Vector{T}, lossfunc, numsamples) where T<:Real
    return sum(lossfunc(sampleA, sampleB)) / numsamples
end

function expectedlosses(experiment::ExperimentABN{ExpectedLossThresh,n}, 
    parameters::Vector{Symbol}; lossfunc=upliftloss, numsamples=10_000) where n
    modelnames = experiment.modelnames
    expectedlosses = Vector{Float64}()
    # TODO: Need a test to make sure the max loss is selected.
    for (refmodelindex, refmodelname) in enumerate(modelnames)
        refmodel = experiment.models[refmodelname]
        maxloss = -Inf
        for compmodelname in modelnames
            if refmodelname == compmodelname
                continue
            end
            compmodel = experiment.models[compmodelname]
            loss = expectedloss(
                refmodel, compmodel, parameters; lossfunc=lossfunc, numsamples=numsamples)
            if loss > maxloss
                maxloss = loss
            end
        end
        push!(expectedlosses, maxloss)
    end
    return (modelnames, expectedlosses)
end

function expectedlosses(experiment::ExperimentABN{ExpectedLossThresh,n}; 
    lossfunc=upliftloss, numsamples=10_000) where n
    parameters = defaultparams(experiment) 
    return expectedlosses(experiment, parameters, lossfunc=lossfunc, numsamples=numsamples)
end

function probbeatall(experiment::ExperimentABN{ProbabilityBeatAllThresh,n}, 
    parameters::Vector{Symbol}; numsamples=10_000) where n
    modelnames = experiment.modelnames
    models = [experiment.models[modelname] for modelname in modelnames]
    samples = samplepost(models, parameters, numsamples)
    probsbeatall = Vector{Float64}(undef, length(modelnames))
    for refmodelindex = 1:length(modelnames)
        refsample = samples[:, refmodelindex]
        numbeatsall = refsample .- samples |>
            x -> mapslices(x -> x .>= 0, x, dims=1) |>
            x -> sum(x, dims=2) |>
            x -> sum(x .== length(modelnames))
        probbeatsall = numbeatsall / numsamples
        probsbeatall[refmodelindex] = probbeatsall
    end
    return (modelnames, probsbeatall)
end

function probbeatall(experiment::ExperimentABN{ProbabilityBeatAllThresh,n}; numsamples=10_000) where n
    parameters = defaultparams(experiment)
    return probbeatall(experiment, parameters, numsamples=numsamples)
end

# Stopping rule check functions
function checkrule(stoppingrule::ExpectedLossThresh, decisionvalue) 
    return decisionvalue < stoppingrule.threshold
end

function checkrule(stoppingrule::ProbabilityBeatAllThresh, decisionvalue)
    return decisionvalue > stoppingrule.threshold
end

"""
    metrics(experiment, parameters, numsamples)

Returns the winner's index and key metrics of the experiment.
"""
function metrics(experiment::ExperimentABN{ExpectedLossThresh,n}, 
    parameters::Vector{Symbol}; numsamples=10_000) where n
    modelnames, explosses = expectedlosses(experiment, parameters, numsamples=numsamples)
    minindex = argmin(explosses)
    return (minindex, explosses)
end

function metrics(experiment::ExperimentABN{ProbabilityBeatAllThresh,n}, 
    parameters::Vector{Symbol}; numsamples=10_000) where n
    modelnames, probabilities = probbeatall(experiment, parameters, numsamples=numsamples)
    maxindex = argmax(probabilities)
    return (maxindex, probabilities)
end

function metrics(experiment::ExperimentABN; numsamples=10_000)
    parameters = defaultparams(experiment)
    return metrics(experiment, parameters, numsamples=numsamples)
end

function metrics(experiment::ExperimentBF)
    (experiment.stats !== nothing && experiment.stats.n > 0) || 
        error("The experiment has no data.")
    return bayesfactor(experiment)
end

"""
    decide!(experiment; numsamples=10_000)

Make decision based on an experiment result and its stopping rule.
"""
function decide!(experiment::ExperimentABN, parameters::Vector{Symbol}; numsamples=10_000)
    winnerindex, winnermetric = metrics(experiment, parameters::Vector{Symbol}, numsamples=numsamples)
    experiment.winner = nothing
    if checkrule(experiment.rule, winnermetric[winnerindex])
        experiment.winner = experiment.modelnames[winnerindex] 
    end
    return experiment.winner
end

function decide!(experiment::ExperimentABN; numsamples=10_000)
    parameters = defaultparams(experiment)
    return decide!(experiment, parameters, numsamples=numsamples)
end

function decide!(experiment::ExperimentBF)
    modelnames = experiment.modelnames
    bayesfactor = metrics(experiment)
    threshold = experiment.rule.threshold
    if bayesfactor > threshold 
        experiment.winner = modelnames[2] 
    elseif bayesfactor < 1/threshold
        experiment.winner = modelnames[1]
    else
        experiment.winner = nothing 
    end
    return experiment.winner
end
