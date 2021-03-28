# BayesianExperiments.jl

This is the documentation of `BayesianExperiments.jl`, a library for conducting Bayesian AB testing in Julia.

Current features include:

- Hypothesis testing with Bayes factor. Support the effect size model with Normal distribution prior and JZS prior.
- Bayesian decision making with conjugate prior models. Support expected loss and probability to beat all as the stopping rule.
- Flexible experiment design for both fixed horizon experiments and sequential test experiment.
- Efficient simulation tools to support power analysis and sensitivity analysis.

## Installation

You can install a stable version of BayesianExperiments by running the command in the Julia REPL:

```julia
julia> ] add BayesianExperiments
```

## Quick Start

Here's a simple example showing how to use the package:

```julia
using BayesianExperiments

# Generate sample data
n = 1000
dataA = rand(Bernoulli(0.15), n)
dataB = rand(Bernoulli(0.16), n)

# Define the models
modelA = ConjugateBernoulli(1, 1)
modelB = ConjugateBernoulli(1, 1)

# Choose the stopping rule that we will use for making decision
stoppingrule = ExpectedLossThresh(0.0002)

# Setup the experiment by specifying models and the stopping rule
experiment = ExperimentAB([modelA, modelB], stoppingrule)

# Calculate the statistics from our sample data
statsA = BetaStatistics(dataA)
statsB = BetaStatistics(dataB)

# Update the models in the experiment with the newly created statistics
update!(experiment, [statsA, statsB])

# Calculate the metric (expected loss in this case) of each model 
winner_index, expected_losses = metrics(experiment)
```

## Contributing

We welcome contributions to this project and discussion about its contents. Please open an issue or pull request on this repository to propose a change.
