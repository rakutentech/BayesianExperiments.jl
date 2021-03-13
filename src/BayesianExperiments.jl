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

# external dependencies
export
    # Distributions.jl
    Bernoulli,
    Exponential,
    Normal,
    LogNormal

# package exports
export 
    # shared methods
    update!,
    rand, 
    mean,
    convert,

    # experiment.jl
    Experiment,
    ExperimentABN,
    ExperimentAB,
    BayesFactorExperiment,

    expectedloss,
    expectedlosses, 
    probbeatall,

    decide!, 
    upliftloss,
    metrics,
        
    # model.jl
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

    samplepost,
    samplestats,
    lognormalparams,

    # rule.jl
    ExpectedLossThresh, 
    ProbabilityBeatAllThresh,
    BayesFactorThresh,

    # simulation.jl
    Simulation,
    DataGeneratingDistibutions,

    runonce,
    runsequential,
    updateonce!,

    # util.jl
    unnest

end # module
