# Bayes Factor Experiment with Optional Stopping


```julia
# Setup notebook running environment.
# Please Be Patient: it might take a long time to 
# precompile these packages the first time you run 
# this notebook in your local environment.
import Pkg
Pkg.activate(".")
Pkg.instantiate();

using ProgressMeter: @showprogress
using DataFrames

using PrettyTables
using Random
using Query
using StatsPlots
using StatsPlots.PlotMeasures
using Plots

using BayesianExperiments

# number of columns in a dataframe to show 
ENV["COLUMNS"] = 200;
```

Optional stopping refers to the practice of peeking at data and make decision whether or not to continue an experiment. Such practice is usually prohibited in the frequentist AB testing framework. By using simulation-based result, **Rouder (2014)**[2] showed that a Bayes factor experiment with optional stopping can be valid with proper interpretation of the Bayesian quantities. 

This notebook follows the examples in **Schönbrodt et al. (2016)**[1] to conduct the error analysis of Bayes factor based experiment with optional stopping.

The simulation will be conducted by following steps:

1. Choose a threshold of Bayes factor for decision making. For example, if the threshold is set to 10, when a Bayes factor of $\text{BF}_{10}$ is larger than 10, or less than 1/10, we decide we have collected enough evidence and stop the experiment.
2. Choose a prior distribituion for the effect size under $H_1$. We will use the `StudentTEffectSize` model in the package. You can check the definition of `NormalEffectSize` model from the docstring by typing `?NormalEffectSize`.
3. Run a minimum number of steps (20 as the same in the paper), increase the sample size. Compute the bayes factor at each step.
4. As soon as the bayes factor value reached or exceeded the one of the thresholds as set in (1), or the maximum number of steps is reached, we will stop the experiment.

Some constants used in the simulation:

* Number of simulations: 5000
* Minimum number of steps: 20

The simulation function can be quickly created based on our package:


```julia
function simulate(δ, n, σ0; r=0.707, thresh=9, minsample=20)
    # we will use two-sided decision rule for bayes factor
    rule = TwoSidedBFThresh(thresh)
    
    # the prior distribution of effect size,
    # r is the standard deviation
    model = StudentTEffectSize(r=r)
    
    # setup the experiment
    experiment = ExperimentBF(model=model, rule=rule)
    
    # create a sample with size n, the effect size is 
    # specified as δ
    xs = rand(Normal(δ, 1), n)
    
    i = 0
    # specify the stopping condition
    while (i < n) & (experiment.winner === nothing)
        i += 1
        
        # if minimum number of sample is not reached, 
        # keep collecting data
        if i < minsample
            continue
        end
        
        stats = NormalStatistics(xs[1:i])
        experiment.stats = stats
        decide!(experiment)
    end
    experiment
end

# df table print helper
printtable(df) = pretty_table(df, tf=tf_markdown, nosubheader=true, header_crayon=Crayon(bold=:false))
```




    printtable (generic function with 1 method)



## Case when alternative $\delta = 0$

When alternative $\delta > 0$, the error rate relates to the false positive rate. 


```julia
#deltas = collect(range(0, 1.5, step=0.2));
delta = 0.0
rs = [0.707, 1.0, 1.414];
threshs = [3, 5, 7, 10];
totalnum = length(rs)*length(threshs);

paramsgrid = reshape(collect(Base.Iterators.product(rs, threshs)), (totalnum, 1));
paramsgrid = [(r=r, thresh=thresh) for (r, thresh) in paramsgrid];
@show length(paramsgrid);
```

    length(paramsgrid) = 12



```julia
n =  1000
ns = 5000
minsample = 20

sim_result1 = DataFrame(
    delta=Float64[], 
    r=Float64[], 
    thresh=Float64[], 
    num_sim=Int64[], 
    num_null=Int64[], 
    num_alt=Int64[],
    err_rate=Float64[], 
    avg_sample_size=Int64[])

@showprogress for params in paramsgrid
    delta = 0
    r = params.r
    thresh = params.thresh
    winners = []
    samplesizes = []
    for _ in 1:ns
        experiment = simulate(delta, n, r, thresh=thresh, minsample=minsample)
        push!(winners, experiment.winner)
        push!(samplesizes, experiment.stats.n)
    end
    
    num_null = sum(winners .== "null")
    num_alt = sum(winners .== "alternative")
    
    err_rate = num_alt/ns
    avg_sample_size = mean(samplesizes)
    push!(sim_result1, (delta, r, thresh, ns, num_null, num_alt, err_rate, convert(Int64, round(avg_sample_size))))
end
```

    [32mProgress: 100%|█████████████████████████████████████████| Time: 0:01:00[39m



```julia
printtable(sim_result1)
```

    |[22m delta [0m|[22m     r [0m|[22m thresh [0m|[22m num_sim [0m|[22m num_null [0m|[22m num_alt [0m|[22m err_rate [0m|[22m avg_sample_size [0m|
    |-------|-------|--------|---------|----------|---------|----------|-----------------|
    |   0.0 | 0.707 |    3.0 |    5000 |     4680 |     320 |    0.064 |              24 |
    |   0.0 |   1.0 |    3.0 |    5000 |     4688 |     312 |   0.0624 |              24 |
    |   0.0 | 1.414 |    3.0 |    5000 |     4680 |     320 |    0.064 |              24 |
    |   0.0 | 0.707 |    5.0 |    5000 |     4731 |     268 |   0.0536 |              49 |
    |   0.0 |   1.0 |    5.0 |    5000 |     4734 |     266 |   0.0532 |              49 |
    |   0.0 | 1.414 |    5.0 |    5000 |     4725 |     275 |    0.055 |              49 |
    |   0.0 | 0.707 |    7.0 |    5000 |     4777 |     220 |    0.044 |             100 |
    |   0.0 |   1.0 |    7.0 |    5000 |     4742 |     247 |   0.0494 |              98 |
    |   0.0 | 1.414 |    7.0 |    5000 |     4753 |     238 |   0.0476 |             100 |
    |   0.0 | 0.707 |   10.0 |    5000 |     4744 |     191 |   0.0382 |             205 |
    |   0.0 |   1.0 |   10.0 |    5000 |     4762 |     184 |   0.0368 |             206 |
    |   0.0 | 1.414 |   10.0 |    5000 |     4751 |     183 |   0.0366 |             211 |


## Case when alternative $\delta > 0$

We create a grid of combinations of all parameters.


```julia
deltas = collect(range(0.1, 1.0, step=0.2));
rs = [0.707, 1.0, 1.414];
threshs = [3, 5, 7, 10];
totalnum = length(deltas)*length(rs)*length(threshs);

paramsgrid = reshape(collect(Base.Iterators.product(deltas, rs, threshs)), (totalnum, 1));
paramsgrid = [(delta=delta, r=r, thresh=thresh) for (delta, r, thresh) in paramsgrid]
@show length(paramsgrid);
@show paramsgrid[1:5];
```

    length(paramsgrid) = 60
    paramsgrid[1:5] = NamedTuple{(:delta, :r, :thresh),Tuple{Float64,Float64,Int64}}[(delta = 0.1, r = 0.707, thresh = 3), (delta = 0.3, r = 0.707, thresh = 3), (delta = 0.5, r = 0.707, thresh = 3), (delta = 0.7, r = 0.707, thresh = 3), (delta = 0.9, r = 0.707, thresh = 3)]


The simulation is similar to the $\delta=0$ case. When alternative $\delta > 0$, the error rate relates to the false negative evidence.


```julia
n =  1000
ns = 5000
minsample = 20

sim_result2 = DataFrame(
    delta=Float64[], 
    r=Float64[], 
    thresh=Float64[], 
    num_sim=Int64[], 
    num_null=Int64[], 
    num_alt=Int64[],
    err_rate=Float64[], 
    avg_sample_size=Int64[])

@showprogress for params in paramsgrid
    delta=params.delta
    r = params.r
    thresh = params.thresh
    winners = []
    samplesizes = []
    for _ in 1:ns
        experiment = simulate(delta, n, r, thresh=thresh, minsample=minsample)
        push!(winners, experiment.winner)
        push!(samplesizes, experiment.stats.n)
    end
    
    num_null = sum(winners .== "null")
    num_alt = sum(winners .== "alternative")
    err_rate = 1-num_alt/ns
    avg_sample_size = mean(samplesizes)
    push!(sim_result2, (delta, r, thresh, ns, num_null, num_alt, 
            err_rate, convert(Int64, round(avg_sample_size))))
end
```

    [32mProgress: 100%|█████████████████████████████████████████| Time: 0:02:46[39m


Simulation result when $\delta=0.5$


```julia
sim_result2 |>
    @filter(_.delta==0.5)|>
    @orderby(_.delta) |> @thenby(_.r) |>
    df -> printtable(DataFrame(df))
```

    |[22m delta [0m|[22m     r [0m|[22m thresh [0m|[22m num_sim [0m|[22m num_null [0m|[22m num_alt [0m|[22m err_rate [0m|[22m avg_sample_size [0m|
    |-------|-------|--------|---------|----------|---------|----------|-----------------|
    |   0.5 | 0.707 |    3.0 |    5000 |      727 |    4273 |   0.1454 |              26 |
    |   0.5 | 0.707 |    5.0 |    5000 |       83 |    4917 |   0.0166 |              33 |
    |   0.5 | 0.707 |    7.0 |    5000 |        1 |    4999 |   0.0002 |              37 |
    |   0.5 | 0.707 |   10.0 |    5000 |        0 |    5000 |      0.0 |              39 |
    |   0.5 |   1.0 |    3.0 |    5000 |      759 |    4241 |   0.1518 |              26 |
    |   0.5 |   1.0 |    5.0 |    5000 |       69 |    4931 |   0.0138 |              34 |
    |   0.5 |   1.0 |    7.0 |    5000 |        2 |    4998 |   0.0004 |              37 |
    |   0.5 |   1.0 |   10.0 |    5000 |        0 |    5000 |      0.0 |              39 |
    |   0.5 | 1.414 |    3.0 |    5000 |      723 |    4277 |   0.1446 |              26 |
    |   0.5 | 1.414 |    5.0 |    5000 |       97 |    4903 |   0.0194 |              33 |
    |   0.5 | 1.414 |    7.0 |    5000 |        0 |    5000 |      0.0 |              36 |
    |   0.5 | 1.414 |   10.0 |    5000 |        0 |    5000 |      0.0 |              40 |


## Evaluate the simulation result with Type I & II Error and FDR

As pointed out by [2], we can evaluate the simulation result from the perspective of false discovery rate. Here we assume there is a 50-50 chance that the data is from either the null model or alternative model. 

We can merge the two simulations results by the prior standard deviation $r$ and threshold of bayes factor. In the merged dataframe, each row represents a simulation with the 5000 samples from the null model and 5000 samples from the alternative model with the corresponding parameters ($r$, $thresh$, $\delta_1$).


```julia
sim_result = leftjoin(sim_result1, sim_result2, 
    on=[:r, :thresh, :num_sim],
    renamecols= "_0" => "_1"
);

sim_result.num_dis = sim_result.num_alt_0 + sim_result.num_alt_1;
sim_result.num_false_dis = sim_result.num_alt_0;
sim_result.fdr = sim_result.num_false_dis ./ sim_result.num_dis;

sim_result.type1_error = sim_result.num_alt_0 ./ sim_result.num_sim;
sim_result.type2_error = 1 .- sim_result.num_alt_1 ./ sim_result.num_sim;
sim_result.power = sim_result.num_alt_1 ./ sim_result.num_sim;
sim_result.avg_sample_size = (sim_result.avg_sample_size_0 + sim_result.avg_sample_size_1) ./ 2

sim_result = sim_result |>
    df -> select(df, [:delta_1, :r, :thresh, :num_sim, :num_alt_0, :num_alt_1, :type1_error,  
                      :power, :fdr, :avg_sample_size]);
```

Examples from merged dataframe:


```julia
sim_result |>
    @filter(((_.delta_1 == 0.1) .& (_.r == 0.707)) .|
            ((_.delta_1 == 0.1) .& (_.r == 1.0))  .|
            ((_.delta_1 == 0.3) .& (_.r == 0.707)) .|
            ((_.delta_1 == 0.3) .& (_.r == 1.0)) .|
            ((_.delta_1 == 0.3) .& (_.r == 1.414))) |>
    @orderby(_.delta_1) |> @thenby(_.r) |> @thenby(_.thresh) |>
    df -> printtable(DataFrame(df))
```

    |[22m delta_1 [0m|[22m     r [0m|[22m thresh [0m|[22m num_sim [0m|[22m num_alt_0 [0m|[22m num_alt_1 [0m|[22m type1_error [0m|[22m  power [0m|[22m       fdr [0m|[22m avg_sample_size [0m|
    |---------|-------|--------|---------|-----------|-----------|-------------|--------|-----------|-----------------|
    |     0.1 | 0.707 |    3.0 |    5000 |       320 |       518 |       0.064 | 0.1036 |  0.381862 |            24.5 |
    |     0.1 | 0.707 |    5.0 |    5000 |       268 |       846 |      0.0536 | 0.1692 |  0.240575 |            55.0 |
    |     0.1 | 0.707 |    7.0 |    5000 |       220 |      1387 |       0.044 | 0.2774 |  0.136901 |           126.0 |
    |     0.1 | 0.707 |   10.0 |    5000 |       191 |      1999 |      0.0382 | 0.3998 | 0.0872146 |           274.5 |
    |     0.1 |   1.0 |    3.0 |    5000 |       312 |       553 |      0.0624 | 0.1106 |  0.360694 |            24.5 |
    |     0.1 |   1.0 |    5.0 |    5000 |       266 |       839 |      0.0532 | 0.1678 |  0.240724 |            55.0 |
    |     0.1 |   1.0 |    7.0 |    5000 |       247 |      1301 |      0.0494 | 0.2602 |  0.159561 |           122.0 |
    |     0.1 |   1.0 |   10.0 |    5000 |       184 |      2064 |      0.0368 | 0.4128 | 0.0818505 |           274.5 |
    |     0.3 | 0.707 |    3.0 |    5000 |       320 |      2400 |       0.064 |   0.48 |  0.117647 |            26.0 |
    |     0.3 | 0.707 |    5.0 |    5000 |       268 |      3908 |      0.0536 | 0.7816 | 0.0641762 |            53.0 |
    |     0.3 | 0.707 |    7.0 |    5000 |       220 |      4767 |       0.044 | 0.9534 | 0.0441147 |            91.5 |
    |     0.3 | 0.707 |   10.0 |    5000 |       191 |      4990 |      0.0382 |  0.998 | 0.0368655 |           152.0 |
    |     0.3 |   1.0 |    3.0 |    5000 |       312 |      2386 |      0.0624 | 0.4772 |  0.115641 |            26.0 |
    |     0.3 |   1.0 |    5.0 |    5000 |       266 |      3951 |      0.0532 | 0.7902 |  0.063078 |            53.5 |
    |     0.3 |   1.0 |    7.0 |    5000 |       247 |      4752 |      0.0494 | 0.9504 | 0.0494099 |            90.5 |
    |     0.3 |   1.0 |   10.0 |    5000 |       184 |      4988 |      0.0368 | 0.9976 | 0.0355762 |           152.5 |
    |     0.3 | 1.414 |    3.0 |    5000 |       320 |      2414 |       0.064 | 0.4828 |  0.117045 |            26.0 |
    |     0.3 | 1.414 |    5.0 |    5000 |       275 |      3936 |       0.055 | 0.7872 | 0.0653052 |            53.0 |
    |     0.3 | 1.414 |    7.0 |    5000 |       238 |      4757 |      0.0476 | 0.9514 | 0.0476476 |            92.5 |
    |     0.3 | 1.414 |   10.0 |    5000 |       183 |      4993 |      0.0366 | 0.9986 | 0.0353555 |           155.0 |


## Visualizations

### Type I Error

Some observations from the visualization below: 

1. Higher thresholds will lower the Type I error rate.
2. Standard deviation of prior of effect size ($r$). For lower value thresholds, lower $r$ value will increase the Type I error. However, as the threshold increases, the $r$ value seems to have smaller impact on the Type I error.


```julia
# create labels for visualizations
r_labels = hcat(["r=$val" for val in rs]...);
thresh_labels = hcat(["thresh=$(Int(val))" for val in threshs]...);
delta_labels = hcat(["\\delta=$(val)" for val in deltas]...);
```


```julia
p0 = sim_result |>
    @filter(_.delta_1==0.1) |>
    @df plot(:thresh, [:type1_error], 
             group=(:r), 
             label=r_labels,
             title="Bayes Factor Threshold vs Type I Error",
             xlabel="BF Threshold",
             ylabel="Type I Error",
             legend=true)
```




    
![svg](bayes_factor_optional_stopping_files/bayes_factor_optional_stopping_26_0.svg)
    



### Impact of Effect Size When $\delta_1 > 0$

When the effect size gets larger, the power will increase and the FDR will decrease. 


```julia
p1 = sim_result |> 
    @filter(_.r==0.707) |> 
    @df plot(:delta_1, [:power], 
        group=(:thresh), 
        label=thresh_labels, 
        xlabel="Effect Size \\delta",
        ylabel="Power");

p2 = sim_result |> 
    @filter(_.r==0.707) |> 
    @df plot(:delta_1, [:fdr], 
        group=(:thresh), 
        label=thresh_labels, 
        xlabel="Effect Size \\delta",
        ylabel="False Discovery Rate");

plot(p1, p2, size=(800, 400), layout=(1, 2), left_margin=[15mm 0mm]) 
```




    
![svg](bayes_factor_optional_stopping_files/bayes_factor_optional_stopping_29_0.svg)
    



### Impact of Bayes Factor Thresholds

When Bayes Factor threshold increases, the power also increase. This is because as the power increases, the chance we will falsely select the null hypothesis decreases.

The plots below shows the relationship between Bayes factor thresholds and power for different effect sizes.


```julia
function plot_thresh_vs_power(df, delta; xlabel="BF Threshold")
    return df |> @df plot(
        plot(:thresh, [:power], group=(:r), label=r_labels, xlabel=xlabel, ylabel="Power"),
        plot(:thresh, [:fdr], group=(:r), label=r_labels, xlabel=xlabel, ylabel="FDR"),
        title ="\\delta_{1} = $delta")
end

p1 = sim_result |>
    @filter(_.delta_1==0.1) |>
    df -> plot_thresh_vs_power(df, 0.1, xlabel="");

p2 = sim_result |>
    @filter(_.delta_1==0.3) |>
    df -> plot_thresh_vs_power(df, 0.3, xlabel="");

p3 = sim_result |>
    @filter(_.delta_1==0.7) |>
    df -> plot_thresh_vs_power(df, 0.7);

plot(p1, p2, p3, layout=(3, 1), size=(800, 800))
```




    
![svg](bayes_factor_optional_stopping_files/bayes_factor_optional_stopping_32_0.svg)
    



### Average Sample Sizes vs. Bayes Factor Thresholds

The average sample sizes needed to stop the experiment and make decision. As Bayes factor threshold gets larger, the expected sample sizes also get larger.


```julia
sim_result |> 
    @filter(_.r==0.707) |>
    @df plot(:thresh, [:avg_sample_size], 
            group=(:delta_1), legend=:topleft, label=delta_labels,
            xlabel="Bayes Factor Thresholds", ylabel="Average Sample Size")
```




    
![svg](bayes_factor_optional_stopping_files/bayes_factor_optional_stopping_35_0.svg)
    



## References

1. Schönbrodt, Felix D., Eric-Jan Wagenmakers, Michael Zehetleitner, and Marco Perugini. "Sequential hypothesis testing with Bayes factors: Efficiently testing mean differences." Psychological methods 22, no. 2 (2017): 322.
2. Deng, Alex, Jiannan Lu, and Shouyuan Chen. "Continuous monitoring of A/B tests without pain: Optional stopping in Bayesian testing." In 2016 IEEE international conference on data science and advanced analytics (DSAA), pp. 243-252. IEEE, 2016.
3. Rouder, Jeffrey N. "Optional stopping: No problem for Bayesians." Psychonomic bulletin & review 21, no. 2 (2014): 301-308.
