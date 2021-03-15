module BayesianExperiments

import Statistics: mean, rand, std
using Base
using StaticArrays:SVector
using Distributions: 
    pdf, 
    Beta, Gamma, LogNormal, Bernoulli, Exponential, InverseGamma, Normal,
    ContinuousUnivariateDistribution, UnivariateDistribution

include("util.jl")
include("data.jl")
include("distribution.jl")
include("models/conjugate.jl")
include("models/bayesfactor.jl")
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

    # data.jl
    merge,

    # experiment.jl
    Experiment,
    ExperimentABN,
    ExperimentAB,
    ExperimentBF,

    expectedloss,
    expectedlosses, 
    probbeatall,
    bayesfactor,

    decide!, 
    upliftloss,
    metrics,
        
    # model.jl
    ProbabilisticModel, 
    ConjugateModel,

    ConjugateBernoulli,
    BernoulliStatistics, 
    BernoulliPosteriorSample,

    ConjugateExponential,
    ExponentialStatistics,
    ExponentialPosteriorSample,

    ConjugateNormal,
    NormalStatistics,
    NormalPosteriorSample,

    ConjugateLogNormal,
    LogNormalStatistics,
    LogNormalPosteriorSample,

    ChainedModel, 
    ChainOperator,

    TwoSampleStatistics,
    EffectSizeModel,

    samplepost,
    samplestats,
    lognormalparams,
    effectsize,

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
    catbyrow,
    zstat,
    effsamplesize,
    pooledsd

end # module
