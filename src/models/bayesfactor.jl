abstract type BayesFactorModel <: ProbabilisticModel end

"""
EffectSizeModel <: ProbabilisticModel

A standard effect size model has two hypotheses: ``H_1``(null) an ``H_2``(alternative):

1. ``H_1``: ``\\mu = m_0``
2. ``H_2``: ``\\mu ≠ m_0``

with the population mean ``\\mu`` and pre-specified standard deviation ``\\sigma``. We want to test whether 
``\\mu`` is equal to ``m_0`` or not.

The prior of the standard effect size is 

``
\\delta | H_2 \\sim \\text{Normal}(0, n_0)
``

where ``\\delta`` is the standard effect size and ``n_0`` can be considered as a prior sample size. 
The standard effect size ``\\delta`` is defined as 

``
\\delta = \\frac{\\mu - m_0}{\\sigma}.
``

In practice, the standard deviations are unknown but in large sample scenario we assume 
they are known and use their estimates.

## Fileds

- `μ0`: mean of null hypothesis
- `n0`: Prior sample size. `1/n0` is the prior standard deviation.

## Methods

```julia
bayesfactor(model, stats)    # calculate Bayes factor from one group statistics
bayesfactor(model, twostats) # calculate Bayes factor from two group's statistics
```

## References

1. [Chapter 5 hypothesis Testing with Normal Populations](
https://statswithr.github.io/book/hypothesis-testing-with-normal-populations.html) 
in *An Introduction to Bayesian Thinking*.
2. Deng, Alex, Jiannan Lu, and Shouyuan Chen. "Continuous monitoring of A/B tests without pain: 
Optional stopping in Bayesian testing." 2016 IEEE international conference on data science 
and advanced analytics (DSAA). IEEE, 2016.
"""
struct NormalEffectSize <: BayesFactorModel
    μ0::Float64
    σ0::Float64
end

function bayesfactor(model::NormalEffectSize, stats::NormalStatistics)
    n = stats.n 
    σ0 = model.σ0
    δ = effectsize(stats, μ0=model.μ0)
    bf21 = pdf(Normal(0, sqrt(σ0^2+1/n)), δ)/pdf(Normal(0, sqrt(1/n)), δ) 
    return bf21
end

function bayesfactor(model::NormalEffectSize, twostats::TwoNormalStatistics)
    stats = merge(twostats)
    return bayesfactor(model, stats)
end

struct StudentTModel <: BayesFactorModel
    r::Real
    rtol::Real
end

StudentTModel(;r=0.707, rtol=1e-8) = StudentTModel(r, rtol)

function bayesfactor(model::StudentTModel, stats::StudentTStatistics)
    t = stats.t
    v = stats.dof
    n = stats.n

    r = model.r
    rtol = model.rtol

    numerator = (1 + t^2/v)^(-(v+1)/2)
    denominator, _ = quadgk(
        g -> (1+n*g*r^2)^(-0.5) * (1+t^2/((1+n*g*r^2)*v))^(-(v+1)/2) 
            * (2π)^(-0.5) * g^(-1.5) * exp(-1/(2*g)),
        0, Inf,
        rtol=rtol)

    # we take invserse to get b_1_0, instead of b_0_1
    return denominator/numerator
end