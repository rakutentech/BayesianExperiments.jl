"""
    upliftloss(a, b)

The lost uplift when we select "a" but "b" is actually better. 
Margin is the minimum lift level of a comparing to b.
"""
upliftloss(a, b) = @. max(b - a, 0.)

"""
    StoppingRule

`StoppingRule` is a decision rule that specifies the metric and its the threshold, 
along with the way how we can make the decision and stop the experiment.
"""

abstract type StoppingRule end

"""
    ExpectedLossThresh <: StoppingRule

The experiment has a winning model if the model has the smallest posterior expected loss, 
and its expected loss value is below the threshold.

## References

- [Definition of the *expected loss* on Wikipedia](https://en.wikipedia.org/wiki/Bayes_estimator#Posterior_median_and_other_quantiles)
"""
struct ExpectedLossThresh <: StoppingRule 
    threshold::Float64
end

"""
    ProbabilityBeatAllThresh <: StoppingRule

The experiment has a winning model if probability of that model's posterior samples 
is larger than the alternative models is above the threshold.
"""
struct ProbabilityBeatAllThresh <: StoppingRule
    threshold::Float64
end

"""
    BayesFactorThresh <: StoppingRule

The bayes factor itself is interpretable as the comparative evidence of data 
under the two competing hypotheses. Higher bayes factor, as defined ``\text{BF}_{10}``
favours the alternative hypothesis. 

In practice, a threshold can be used to make decision in bayes factor experiment.
The experiment will stop when 

1. ``\text{BF}_{10}`` > threshold
2. ``\text{BF}_{10}`` < 1/threshold

In (1), the bayes factor of alternative over null is above the threshold,
we can accept the alternative hypothesis. In (2), the bayes factor is below the
inverse of the threshold, we can accept the null hypothesis. Otherwise, we don't 
have enough evidence to accept any of these hypotheses.
"""
struct BayesFactorThresh <: StoppingRule
    threshold::Float64
end