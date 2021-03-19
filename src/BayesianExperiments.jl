module BayesianExperiments

import Statistics: mean, rand, std
using Base
using StaticArrays:SVector
using Distributions: 
    pdf, 
    Beta, Gamma, LogNormal, Bernoulli, Exponential, InverseGamma, Normal,
    ContinuousUnivariateDistribution, UnivariateDistribution
using QuadGK

include("util.jl")
include("data.jl")
include("distribution.jl")
include("models/common.jl")
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
    BernoulliStatistics, 
    ExponentialStatistics,
    NormalStatistics,
    LogNormalStatistics,
    TwoNormalStatistics,
    StudentTStatistics,

    merge,
    tstat,
    tstatpooled,
    tstatwelch,

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
        
    # common.jl
    ProbabilisticModel, 

    # conjugate.jl
    ConjugateModel,

    ConjugateBernoulli,
    BernoulliPosteriorSample,

    ConjugateExponential,
    ExponentialPosteriorSample,

    ConjugateNormal,
    NormalPosteriorSample,

    ConjugateLogNormal,
    LogNormalPosteriorSample,

    ChainedModel, 
    ChainOperator,
    MultiplyOperator,

    samplepost,
    samplestats,
    lognormalparams,

    # bayesfactor.jl 
    NormalEffectSize,
    StudentTEffectSize,

    effectsize,

    # rule.jl
    ExpectedLossThresh, 
    ProbabilityBeatAllThresh,
    OneSidedBFThresh,
    TwoSidedBFThresh,

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
