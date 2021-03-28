# Examples

```@contents
Pages = ["examples.md"]
```

## Examples: Conjugate Models

### Example: Two Models

A basic example showing an experiment with two models.

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

# Or, we can directly find the winning model in the experiment 
winner, _ = decide!(experiment)
```

### Example: Three Models

Similar to the two models cases, but now we have three models.

```julia
using BayesianExperiments

# Generate sample data
n = 1000
dataA = rand(Bernoulli(0.150), n)
dataB = rand(Bernoulli(0.145), n)
dataC = rand(Bernoulli(0.180), n)

# Define the models
modelA = ConjugateBernoulli(1, 1)
modelB = ConjugateBernoulli(1, 1)
modelC = ConjugateBernoulli(1, 1)

# Choose the stopping rule
stoppingrule = ProbabilityBeatAllThresh(0.99)

# Setup the experiment
experiment = ExperimentABN([modelA, modelB, modelC], stoppingrule)

# Calculate the statistics from our sample data
statsA = BetaStatistics(dataA)
statsB = BetaStatistics(dataB)
statsC = BetaStatistics(dataC)

# Update the model in the experiment with the newly created statistics
update!(experiment, [statsA, statsB, statsC])

# Calculate the metric (expected loss in this case) of each model 
winner_index, expected_losses = metrics(experiment)

# Or, we can directly find the winning model in the experiment 
winner, _ = decide!(experiment)
```

### Example: Chained Models

We have a chained model with ConjugateBernoulli and ConjugateLogNormal. A common use case is
when we want to use revenue per visitor as the metric. We need to model distributions of both the conversion rate and revenue.  

```julia
using BayesianExperiments

# Generate sample data
n = 1000
dataA1 = rand(Bernoulli(0.050), n)
dataA2 = rand(LogNormal(1.0, 1.0), n)
dataB1 = rand(Bernoulli(0.060), n)
dataB2 = rand(LogNormal(1.0, 1.0), n)

# Calculate the statistics from our sample data
statsA1 = BetaStatistics(dataA1)
statsA2 = LogNormalStatistics(dataA2)
statsB1 = BetaStatistics(dataB1)
statsB2 = LogNormalStatistics(dataB2)

# Setup the experiment
modelA = ChainedModel(
    [ConjugateBernoulli(1, 1), ConjugateLogNormal(0.0, 1.0, 0.001, 0.001)],
    [MultiplyOperator()]
)
modelB = ChainedModel(
    [ConjugateBernoulli(1, 1), ConjugateLogNormal(0.0, 1.0, 0.001, 0.001)],
    [MultiplyOperator()]
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
winner, _ = decide!(experiment)
```

### Example: Power Analysis

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
modelA = ConjugateBernoulli(1, 1)
modelB = ConjugateBernoulli(1, 1)
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

## Examples: Bayes Factor Models

### Example: One group

This example uses the sleep data in R:

```julia
# we have one group of sample
sleepdata = [-1.2, -2.4, -1.3, -1.3, 0.0, -1.0, -1.8, -0.8, -4.6, -1.4]

# calculate statistics
normalstats = NormalStatistics(sleepdata)

# setup the prior model for the effect size
model = StudentTEffectSize()

# calculate the t-statistics
tstats = StudentTStatistics(normalstats)

# we can calculate the posterior odds (equal to bayes factor with
# equal prior odds)
bf = bayesfactor(model, stats)

# we can setup the experiment after spcifying our stopping rule
stoppingrule = TwoSidedBFThresh(9)
experiment = ExperimentBF(model=model, rule=stoppingrule)

# update the experiment with the new data
update!(experiment, normalstats)

# make decision on the experiment
winner, _ = decide!(experiment)
```

### Example: Two groups

This example uses the `Chicken Weights by Feed Type` in `R`.

```julia
horsebean=[179, 160, 136, 227, 217, 168, 108, 124, 143, 140]
linseed=[309, 229, 181, 141, 260, 203, 148, 169, 213, 257, 244, 271]

# Calculate normal statistics from data in each group
stats1 = NormalStatistics(horsebean)
stats2 = NormalStatistics(linseed)

# Here we assume equal variance
tstat = StudentTStatistics(TwoNormalStatistics(stats1, stats2))

# setup the prior model for the effect size
# we can change the standard devation of the prior of effect size 
model = StudentTEffectSize(r=sqrt(2)/2)

# we can calculate the posterior odds (equal to bayes factor with
# equal prior odds)
bf = bayesfactor(model, tstat)

# we can setup the experiment after specifying our stopping rule
stoppingrule = TwoSidedBFThresh(9)
experiment = ExperimentBF(model=model, rule=stoppingrule)

# update the experiment with the new data
update!(experiment, normalstats)

# make decision on the experiment
winner, _ = decide!(experiment)
```
