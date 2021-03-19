# Bayes Factor Experiment with Optional Stopping


```julia
using ProgressMeter: @showprogress
using DataFrames

using Plots
using Random
using BayesianExperiments

# number of columns in a dataframe to show 
ENV["COLUMNS"] = 200;
```

Optional stopping refers to the practice of peeking at data and make decision whether or not to continue an experiment. Such practice is usually prohibited in the frequentist AB testing framework. By using simulation-based result, *Rouder (2014)*[2] showed that a Bayes factor experiment with optional stopping can be valid with proper interpretation of the Bayesian quantities. 

This notebook follows the examples in *Schönbrodt et al. (2016)*[1] to conduct the error analysis of Bayes factor based experiment with optional stopping.

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

```




    simulate (generic function with 1 method)



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
sim_result1
```




<table class="data-frame"><thead><tr><th></th><th>delta</th><th>r</th><th>thresh</th><th>num_sim</th><th>num_null</th><th>num_alt</th><th>err_rate</th><th>avg_sample_size</th></tr><tr><th></th><th>Float64</th><th>Float64</th><th>Float64</th><th>Int64</th><th>Int64</th><th>Int64</th><th>Float64</th><th>Int64</th></tr></thead><tbody><p>12 rows × 8 columns</p><tr><th>1</th><td>0.0</td><td>0.707</td><td>3.0</td><td>5000</td><td>4666</td><td>334</td><td>0.0668</td><td>24</td></tr><tr><th>2</th><td>0.0</td><td>1.0</td><td>3.0</td><td>5000</td><td>4691</td><td>309</td><td>0.0618</td><td>24</td></tr><tr><th>3</th><td>0.0</td><td>1.414</td><td>3.0</td><td>5000</td><td>4714</td><td>286</td><td>0.0572</td><td>24</td></tr><tr><th>4</th><td>0.0</td><td>0.707</td><td>5.0</td><td>5000</td><td>4727</td><td>273</td><td>0.0546</td><td>48</td></tr><tr><th>5</th><td>0.0</td><td>1.0</td><td>5.0</td><td>5000</td><td>4743</td><td>256</td><td>0.0512</td><td>48</td></tr><tr><th>6</th><td>0.0</td><td>1.414</td><td>5.0</td><td>5000</td><td>4724</td><td>276</td><td>0.0552</td><td>47</td></tr><tr><th>7</th><td>0.0</td><td>0.707</td><td>7.0</td><td>5000</td><td>4750</td><td>244</td><td>0.0488</td><td>99</td></tr><tr><th>8</th><td>0.0</td><td>1.0</td><td>7.0</td><td>5000</td><td>4779</td><td>215</td><td>0.043</td><td>100</td></tr><tr><th>9</th><td>0.0</td><td>1.414</td><td>7.0</td><td>5000</td><td>4733</td><td>259</td><td>0.0518</td><td>102</td></tr><tr><th>10</th><td>0.0</td><td>0.707</td><td>10.0</td><td>5000</td><td>4755</td><td>181</td><td>0.0362</td><td>207</td></tr><tr><th>11</th><td>0.0</td><td>1.0</td><td>10.0</td><td>5000</td><td>4776</td><td>168</td><td>0.0336</td><td>205</td></tr><tr><th>12</th><td>0.0</td><td>1.414</td><td>10.0</td><td>5000</td><td>4746</td><td>190</td><td>0.038</td><td>208</td></tr></tbody></table>



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

    [32mProgress: 100%|█████████████████████████████████████████| Time: 0:02:35[39m


Simulation result when $\delta=0.5$


```julia
sim_result2 |>
    df -> filter(x->x.delta==0.5, df)|>
    df -> sort(df, [:delta, :r])
```




<table class="data-frame"><thead><tr><th></th><th>delta</th><th>r</th><th>thresh</th><th>num_sim</th><th>num_null</th><th>num_alt</th><th>err_rate</th><th>avg_sample_size</th></tr><tr><th></th><th>Float64</th><th>Float64</th><th>Float64</th><th>Int64</th><th>Int64</th><th>Int64</th><th>Float64</th><th>Int64</th></tr></thead><tbody><p>12 rows × 8 columns</p><tr><th>1</th><td>0.5</td><td>0.707</td><td>3.0</td><td>5000</td><td>785</td><td>4215</td><td>0.157</td><td>26</td></tr><tr><th>2</th><td>0.5</td><td>0.707</td><td>5.0</td><td>5000</td><td>86</td><td>4914</td><td>0.0172</td><td>33</td></tr><tr><th>3</th><td>0.5</td><td>0.707</td><td>7.0</td><td>5000</td><td>1</td><td>4999</td><td>0.0002</td><td>37</td></tr><tr><th>4</th><td>0.5</td><td>0.707</td><td>10.0</td><td>5000</td><td>0</td><td>5000</td><td>0.0</td><td>40</td></tr><tr><th>5</th><td>0.5</td><td>1.0</td><td>3.0</td><td>5000</td><td>731</td><td>4269</td><td>0.1462</td><td>26</td></tr><tr><th>6</th><td>0.5</td><td>1.0</td><td>5.0</td><td>5000</td><td>81</td><td>4919</td><td>0.0162</td><td>33</td></tr><tr><th>7</th><td>0.5</td><td>1.0</td><td>7.0</td><td>5000</td><td>1</td><td>4999</td><td>0.0002</td><td>36</td></tr><tr><th>8</th><td>0.5</td><td>1.0</td><td>10.0</td><td>5000</td><td>0</td><td>5000</td><td>0.0</td><td>40</td></tr><tr><th>9</th><td>0.5</td><td>1.414</td><td>3.0</td><td>5000</td><td>744</td><td>4256</td><td>0.1488</td><td>25</td></tr><tr><th>10</th><td>0.5</td><td>1.414</td><td>5.0</td><td>5000</td><td>86</td><td>4914</td><td>0.0172</td><td>34</td></tr><tr><th>11</th><td>0.5</td><td>1.414</td><td>7.0</td><td>5000</td><td>2</td><td>4998</td><td>0.0004</td><td>37</td></tr><tr><th>12</th><td>0.5</td><td>1.414</td><td>10.0</td><td>5000</td><td>0</td><td>5000</td><td>0.0</td><td>39</td></tr></tbody></table>



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
#sim_result.type2_error = 1 .- sim_result.num_alt_1 ./ sim_result.num_sim;
sim_result.power = sim_result.num_alt_1 ./ sim_result.num_sim;

sim_result = sim_result |>
    df -> select(df, [:delta_1, :r, :thresh, :num_sim, :num_null_0, :num_alt_0, 
        :num_null_1, :num_alt_1, :type1_error, :power, :fdr]);
```


```julia
sim_result.num_dis = sim_result.num_alt_0 + sim_result.num_alt_1;
sim_result.num_false_dis = sim_result.num_alt_0;
sim_result.fdr = sim_result.num_false_dis ./ sim_result.num_dis;

sim_result.type1_error = sim_result.num_alt_0 ./ sim_result.num_sim;
sim_result.type2_error = 1 .- sim_result.num_alt_1 ./ sim_result.num_sim;
```


```julia
sim_result = sim_result |>
    df -> select(df, [:delta_1, :r, :thresh, :num_sim, :num_null_0, :num_alt_0, 
        :num_null_1, :num_alt_1, :type1_error, :power, :fdr]);
```

Examples from merged dataframe:


```julia
sim_result |>
    df -> filter(
        x -> ((x.delta_1 == 0.1) .& (x.r == 0.707)) .|
             ((x.delta_1 == 0.1) .& (x.r == 1.0)) .|
             ((x.delta_1 == 0.3) .& (x.r == 1.0))
            , df) |>
    df -> sort(df, [:delta_1, :r, :thresh])
```




<table class="data-frame"><thead><tr><th></th><th>delta_1</th><th>r</th><th>thresh</th><th>num_sim</th><th>num_null_0</th><th>num_alt_0</th><th>num_null_1</th><th>num_alt_1</th><th>type1_error</th><th>power</th><th>fdr</th></tr><tr><th></th><th>Float64?</th><th>Float64</th><th>Float64</th><th>Int64</th><th>Int64</th><th>Int64</th><th>Int64?</th><th>Int64?</th><th>Float64</th><th>Float64</th><th>Float64</th></tr></thead><tbody><p>12 rows × 11 columns</p><tr><th>1</th><td>0.1</td><td>0.707</td><td>3.0</td><td>5000</td><td>4666</td><td>334</td><td>4451</td><td>549</td><td>0.0668</td><td>0.1098</td><td>0.378256</td></tr><tr><th>2</th><td>0.1</td><td>0.707</td><td>5.0</td><td>5000</td><td>4727</td><td>273</td><td>4197</td><td>799</td><td>0.0546</td><td>0.1598</td><td>0.254664</td></tr><tr><th>3</th><td>0.1</td><td>0.707</td><td>7.0</td><td>5000</td><td>4750</td><td>244</td><td>3595</td><td>1346</td><td>0.0488</td><td>0.2692</td><td>0.153459</td></tr><tr><th>4</th><td>0.1</td><td>0.707</td><td>10.0</td><td>5000</td><td>4755</td><td>181</td><td>2533</td><td>2042</td><td>0.0362</td><td>0.4084</td><td>0.0814215</td></tr><tr><th>5</th><td>0.1</td><td>1.0</td><td>3.0</td><td>5000</td><td>4691</td><td>309</td><td>4453</td><td>547</td><td>0.0618</td><td>0.1094</td><td>0.360981</td></tr><tr><th>6</th><td>0.1</td><td>1.0</td><td>5.0</td><td>5000</td><td>4743</td><td>256</td><td>4192</td><td>807</td><td>0.0512</td><td>0.1614</td><td>0.240828</td></tr><tr><th>7</th><td>0.1</td><td>1.0</td><td>7.0</td><td>5000</td><td>4779</td><td>215</td><td>3649</td><td>1301</td><td>0.043</td><td>0.2602</td><td>0.141821</td></tr><tr><th>8</th><td>0.1</td><td>1.0</td><td>10.0</td><td>5000</td><td>4776</td><td>168</td><td>2569</td><td>2053</td><td>0.0336</td><td>0.4106</td><td>0.0756416</td></tr><tr><th>9</th><td>0.3</td><td>1.0</td><td>3.0</td><td>5000</td><td>4691</td><td>309</td><td>2584</td><td>2416</td><td>0.0618</td><td>0.4832</td><td>0.113394</td></tr><tr><th>10</th><td>0.3</td><td>1.0</td><td>5.0</td><td>5000</td><td>4743</td><td>256</td><td>1136</td><td>3864</td><td>0.0512</td><td>0.7728</td><td>0.0621359</td></tr><tr><th>11</th><td>0.3</td><td>1.0</td><td>7.0</td><td>5000</td><td>4779</td><td>215</td><td>254</td><td>4746</td><td>0.043</td><td>0.9492</td><td>0.043338</td></tr><tr><th>12</th><td>0.3</td><td>1.0</td><td>10.0</td><td>5000</td><td>4776</td><td>168</td><td>8</td><td>4992</td><td>0.0336</td><td>0.9984</td><td>0.0325581</td></tr></tbody></table>



## References

1. Schönbrodt, Felix D., Eric-Jan Wagenmakers, Michael Zehetleitner, and Marco Perugini. "Sequential hypothesis testing with Bayes factors: Efficiently testing mean differences." Psychological methods 22, no. 2 (2017): 322.
2. Deng, Alex, Jiannan Lu, and Shouyuan Chen. "Continuous monitoring of A/B tests without pain: Optional stopping in Bayesian testing." In 2016 IEEE international conference on data science and advanced analytics (DSAA), pp. 243-252. IEEE, 2016.
3. Rouder, Jeffrey N. "Optional stopping: No problem for Bayesians." Psychonomic bulletin & review 21, no. 2 (2014): 301-308.
