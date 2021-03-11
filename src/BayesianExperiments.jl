module BayesianExperiments

import Base:show
import Base.convert
import Statistics: mean, rand, std
using Distributions: 
    Beta, Gamma, LogNormal, Bernoulli, Exponential, 
    ContinuousUnivariateDistribution, InverseGamma, Normal,
    UnivariateDistribution

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

    ChainOperator,
    ChainedModel, 

    update!,
    selectwinner!, 

    ExpectedLossThresh, 
    ProbabilityBeatAllThresh,

    apprexpectedloss,
    apprexpectedlosses, 
    apprprobbeatall,
    rand, 
    mean,
    sample_post,
    sample_stats,
    calculatemetrics,
    tolognormalparams,
    upliftloss,
    calcexpectedloss,

    Simulation,
    updateonce!,
    DataGeneratingDistibutions,
    convert,
    runonce,
    runsequential,
    unnest

end # module
