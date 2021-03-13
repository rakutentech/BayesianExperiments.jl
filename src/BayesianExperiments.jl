module BayesianExperiments

import Statistics: mean, rand, std
using Base
using Distributions: 
    pdf, 
    Beta, Gamma, LogNormal, Bernoulli, Exponential, InverseGamma, Normal,
    ContinuousUnivariateDistribution, UnivariateDistribution

include("util.jl")
include("data.jl")
include("distribution.jl")
include("model.jl")
include("rule.jl")
include("experiment.jl")
include("simulation.jl")

export Experiment,
    ExperimentABN,
    ExperimentAB,
    BayesFactorExperiment,

    Bernoulli,
    Exponential,
    Normal,
    LogNormal,
    
    ProbabilisticModel, 
    ConjugateModel,
    BernoulliModel,
    BernoulliStatistics, 
    BernoulliPosteriorSample,

    ExponentialModel,
    ExponentialStatistics,
    ExponentialPosteriorSample,

    NormalModel,
    NormalStatistics,
    NormalPosteriorSample,

    LogNormalModel,
    LogNormalStatistics,
    LogNormalPosteriorSample,

    ChainedModel, 
    ChainOperator,

    update!,
    decide!, 

    ExpectedLossThresh, 
    ProbabilityBeatAllThresh,
    BayesFactorThresh,

    expectedloss,
    expectedlosses, 
    probbeatall,
    rand, 
    mean,
    samplepost,
    samplestats,
    metrics,
    lognormalparams,
    upliftloss,

    Simulation,
    updateonce!,
    DataGeneratingDistibutions,
    convert,
    runonce,
    runsequential,
    unnest
end # module
