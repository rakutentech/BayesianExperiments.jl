# Examples: Bayes Factor Models

## Example: One group

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
decide!(experiment)
```

## Example: Two groups

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
decide!(experiment)
```
