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
        new(Dict(zip(modelnames, models)), modelnames, rule, nothing)
    end
end

function ExperimentABN(models, rule::T; modelnames=nothing) where T <: StoppingRule
    ExperimentABN{T,length(models)}(models, rule, modelnames=modelnames)
end

"""
    ExperimentAB(models, rule; modelnames=nothing) where T <: StoppingRule

An experiment of `ExperimentABN` with two models.
"""
function ExperimentAB(models, rule::T; modelnames=nothing) where T <: StoppingRule
    ExperimentABN{T,2}(models, rule, modelnames=modelnames)
end

function Base.show(io::IO, experiment::Experiment)
    modelinfo = ""
    for (i, modelname) in enumerate(experiment.modelnames)
        modelinfo = string(modelinfo, "  \"", modelname, "\": ", experiment.models[modelname])
        if i < length(experiment.modelnames)
            modelinfo = string(modelinfo, "\n")
        end
    end
    print(io, typeof(experiment), "\n", modelinfo)
end

function update!(experiment::Experiment, 
                 listofstats::Union{Vector{T},Vector{Vector{T}}}) where T <: ModelStatistics
    for (i, stats) in enumerate(listofstats)
        modelname = experiment.modelnames[i]
        update!(experiment, modelname, stats)
    end
    nothing
end

function update!(experiment::Experiment, 
    modelname::String, stats::Union{T,Vector{T}}) where T <: ModelStatistics
    model = experiment.models[modelname]
    update!(model, stats)
    nothing
end

function calcexpectedloss(sampleA, sampleB, lossfunc, numsamples)
    sum(lossfunc(sampleA, sampleB)) / numsamples
end

function getdefaultparams(experiment::Experiment)
    model1 = [model for (modelname, model) in experiment.models][1]
    getdefaultparams(model1)
end

"""
    apprexpectedloss(modelA, modelB; lossfunc, numsamples)

Approximating the expected loss for choosing model A over model B.

    apprexpectedlosses(experiment::ExperimentABN; lossfunc, numsamples)

Approximating the expected loss for all models in the experiment. 
For A/B/N experiment, the expected loss for each model is the maximum of the losses 
by comparing that model to all the other models.
"""
function apprexpectedloss(modelA::T, modelB::T, parameters; 
    lossfunc=upliftloss, numsamples=10_000) where T <: ProbabilisticModel
    sampleA = sample_post(modelA, parameters, numsamples)
    sampleB = sample_post(modelB, parameters, numsamples)
    length(sampleA) == length(sampleB) || 
        throw(ArgumentError("Approximating expected loss requires equal length."))
    losschooseA = calcexpectedloss(sampleA, sampleB, lossfunc, numsamples) 
    losschooseA
end

function apprexpectedloss(modelA::T, modelB::T; lossfunc=upliftloss, 
        numsamples=10_000) where T <: ProbabilisticModel
    parameters = getdefaultparams(modelA) 
    apprexpectedloss(modelA, modelB, parameters, lossfunc=lossfunc, numsamples=numsamples)
end


function apprexpectedlosses(experiment::ExperimentABN{ExpectedLossThresh,n}, 
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
            loss = apprexpectedloss(
                refmodel, compmodel, parameters; lossfunc=lossfunc, numsamples=numsamples)
            if loss > maxloss
                maxloss = loss
            end
        end
        push!(expectedlosses, maxloss)
    end
    (modelnames, expectedlosses)
end

function apprexpectedlosses(experiment::ExperimentABN{ExpectedLossThresh,n}; 
    lossfunc=upliftloss, numsamples=10_000) where n
    parameters = getdefaultparams(experiment) 
    apprexpectedlosses(experiment, parameters, lossfunc=lossfunc, numsamples=numsamples)
end

function apprprobbeatall(experiment::ExperimentABN{ProbabilityBeatAllThresh,n}, 
    parameters::Vector{Symbol}; numsamples=10_000) where n
    modelnames = experiment.modelnames
    models = [experiment.models[modelname] for modelname in modelnames]
    samples = sample_post(models, parameters, numsamples)
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
    (modelnames, probsbeatall)
end

function apprprobbeatall(experiment::ExperimentABN{ProbabilityBeatAllThresh,n}; numsamples=10_000) where n
    parameters = getdefaultparams(experiment)
    apprprobbeatall(experiment, parameters, numsamples=numsamples)
end

# Stopping rule check functions
checkrule(stoppingrule::ExpectedLossThresh, decisionvalue) = 
    decisionvalue < stoppingrule.threshold

checkrule(stoppingrule::ProbabilityBeatAllThresh, decisionvalue) =
    decisionvalue > stoppingrule.threshold

"""
    calculatemetrics(experiment, parameters, numsamples)

Returns the winner's index and metrics.
"""
function calculatemetrics(experiment::ExperimentABN{ExpectedLossThresh,n}, 
    parameters::Vector{Symbol}; numsamples=10_000) where n
    modelnames, expectedlosses = apprexpectedlosses(experiment, parameters, numsamples=numsamples)
    minindex = argmin(expectedlosses)
    minindex, expectedlosses
end

function calculatemetrics(experiment::ExperimentABN{ProbabilityBeatAllThresh,n}, 
    parameters::Vector{Symbol}; numsamples=10_000) where n
    modelnames, probabilities = apprprobbeatall(experiment, parameters, numsamples=numsamples)
    maxindex = argmax(probabilities)
    maxindex, probabilities
end

function calculatemetrics(experiment::Experiment; numsamples=10_000)
    parameters = getdefaultparams(experiment)
    calculatemetrics(experiment, parameters, numsamples=numsamples)
end

function selectwinner!(experiment::Experiment, parameters::Vector{Symbol}; numsamples=10_000)
    winnerindex, winnermetric = calculatemetrics(experiment, parameters::Vector{Symbol}, numsamples=numsamples)
    experiment.winner = nothing
    if checkrule(experiment.rule, winnermetric[winnerindex])
        experiment.winner = experiment.modelnames[winnerindex] 
    end
    experiment.winner
end

function selectwinner!(experiment::Experiment; numsamples=10_000)
    parameters = getdefaultparams(experiment)
    selectwinner!(experiment, parameters, numsamples=numsamples)
end
