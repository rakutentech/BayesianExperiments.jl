## Example: Two Models

A basic example showing an experiment with two models.

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
winner_index, expected_losses = metrics(experiment)

# Or, we can directly find the winning model in the experiment 
winner = decide!(experiment)
```

## Example: Three Models

Similar to the two models cases, but now we have three models.

```julia
using BayesianExperiments

# Generate sample data
n = 1000
dataA = rand(Bernoulli(0.150), n)
dataB = rand(Bernoulli(0.145), n)
dataC = rand(Bernoulli(0.152), n)

# Define the models
modelA = BernoulliModel(1, 1)
modelB = BernoulliModel(1, 1)
modelC = BernoulliModel(1, 1)

# Choose the stopping rule
stoppingrule = ProbabilityBeatAllThresh(0.99)

# Setup the experiment
experiment = ExperimentAB([modelA, modelB, modelC], stoppingrule)

# Calculate the statistics from our sample data
statsA = BernoulliStatistics(dataA)
statsB = BernoulliStatistics(dataB)
statsC = BernoulliStatistics(dataC)

# Update the model in the experiment with the newly created statistics
update!(experiment, [statsA, statsB, statsC])

# Calculate the metric (expected loss in this case) of each model 
winner_index, expected_losses = metrics(experiment)

# Or, we can directly find the winning model in the experiment 
winner = decide!(experiment)
```

## Example: Chained Models

We have a chained model with BernoulliModel and LogNormalModel. A common use case is
when we want to use revenue per visitor as the metric. We need to model distributions of both the conversion rate and revenue.  

```julia
using BayesianExperiments

# Generate sample data
n = 1000
dataA1 = rand(Bernoulli(0.050), n)
dataA2 = rand(LogNormal(1.0, 1.0), n)
dataB1 = rand(Bernoulli(0.055), n)
dataB2 = rand(LogNormal(1.0, 1.0), n)

# Calculate the statistics from our sample data
statsA1 = BernoulliStatistics(dataA1)
statsA2 = LogNormalStatistics(dataA2)
statsB1 = BernoulliStatistics(dataB1)
statsB2 = LogNormalStatistics(dataB2)

# Setup the experiment
modelA = ChainedModel(
    [BernoulliModel(1, 1), LogNormalModel(0.0, 1.0, 0.001, 0.001)],
    [op_multiply]
)
modelB = ChainedModel(
    [BernoulliModel(1, 1), LogNormalModel(0.0, 1.0, 0.001, 0.001)],
    [op_multiply]
)

# Choose the stopping rule
stoppingrule = ExpectedLossThresh(0.001)

# Setup the experiment
experiment = ExperimentAB([modelA, modelB], stoppingrule)

# Update the model in the experiment with the newly created statistics
update!(experiment, [[statsA1, statsA2], [statsB1, statsB2]])

# Calculate the metric (expected loss in this case) of each model 
winner_index, expected_losses = metrics(experiment)

# Or, we can directly find the winning model in the experiment 
winner = decide!(experiment))
```

## Example: Power Analysis

```julia
# Choose the underlying data generating distributions
datagendists = [Bernoulli(0.2), Bernoulli(0.25)]

# Choose the parameters of the simulation
# Number of observations in each group in each step
onestepsizes = [1000, 1000]

# Maximum number of steps 
maxsteps = 30

# Minimum number of steps to run the simulation before 
# we apply the stopping rule
minsteps = 5

# Setup the experiment with models and stopping rule
modelA = BernoulliModel(1, 1)
modelB = BernoulliModel(1, 1)
stoppingrule = ProbabilityBeatAllThresh(0.99)
experiment = ExperimentAB([modelA, modelB], stoppingrule)

# Setup the simulation
simulation = Simulation(
    experiment=experiment,
    datagendists=datagendists,
    maxsteps=maxsteps,
    onestepsizes=[10000, 10000],
    minsteps=minsteps
)

# Run the simulation
numsteps, winners, _ = runsequential(
    simulation, numsamples=1000, numsims=50)

# Calculate the number of winning times for model B ("variant 1")
sum(winners .== "variant 1") ==  2 
```
