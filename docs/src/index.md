# Home

This is the documentation of `BayesianExperiments.jl`, a library for conducting Bayesian AB testing in Julia.

Current features include:

- Conjugate prior models for distributions including Bernoulli, Normal, LogNormal, Exponential, etc.
- Basic models can be chained to model multiple steps process.
- Various stopping rules support: expected loss, probability to beat all.
- Support multiple experiment design including fixed horizon experiments, sequential experiment and online learning.
- Efficient Simulation tools to support power analysis.

## Installation

You can install a stable version of BayesianExperiments by running the command in the Julia REPL:

```julia
julia> ] add BayesianExperiments
```

## Contributing

We welcome contributions to this project and discussion about its contents. Please open an issue or pull request on this repository to propose a change.

## Quick Start

Here's a simple example showing how to use the package:

```julia
using BayesianExperiments

# Generate sample data
n = 1000
dataA = rand(Bernoulli(0.15), n)
dataB = rand(Bernoulli(0.16), n)

# Define the models
modelA = BernoulliModel(1, 1)
modelB = BernoulliModel(1, 1)

# Choose the stopping rule that we will use for making decision
stoppingrule = ExpectedLossThresh(0.0002)

# Setup the experiment by specifying models and the stopping rule
experiment = ExperimentAB([modelA, modelB], stoppingrule)

# Calculate the statistics from our sample data
statsA = BernoulliStatistics(dataA)
statsB = BernoulliStatistics(dataB)

# Update the models in the experiment with the newly created statistics
update!(experiment, [statsA, statsB])

# Calculate the metric (expected loss in this case) of each model 
winner_index, expected_losses = calculatemetrics(experiment)
```
