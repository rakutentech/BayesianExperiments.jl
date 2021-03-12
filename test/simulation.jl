using Random

using BayesianExperiments

@testset "Continous Power Analysis" begin
    @testset "Bernoulli model with probability to beat all with same distribution" begin
        Random.seed!(1234)
        truedists = [Bernoulli(0.2), Bernoulli(0.2)]

        modelA = BernoulliModel(1, 1)
        modelB = BernoulliModel(1, 1)
        stoppingrule = ProbabilityBeatAllThresh(0.99)
        experiment = ExperimentAB([modelA, modelB], stoppingrule)

        simulation = Simulation(
            experiment=experiment,
            datagendists=truedists,
            maxsteps=30,
            onestepsizes=[10000, 10000],
            minsteps=5
        )

        numsteps, winners, _ = runsequential(simulation, numsamples=1000, numsims=50)
        @test sum(winners .== "control") == 3 
        @test sum(winners .== "variant 1") ==  2 
        @test sum(winners .== "nothing") == 45
    end

    @testset "Bernoulli model with expected loss with same distribution" begin
        Random.seed!(1234)
        truedists = [Bernoulli(0.2), Bernoulli(0.2)]
        modelA = BernoulliModel(1, 1)
        modelB = BernoulliModel(1, 1)
        stoppingrule = ExpectedLossThresh(1e-3)
        experiment = ExperimentAB([modelA, modelB], stoppingrule)

        simulation = Simulation(
            experiment=experiment,
            datagendists=truedists,
            maxsteps=30,
            onestepsizes=[1000, 1000],
            minsteps=5
        )

        numsteps, winners, _ = runsequential(simulation, numsamples=5000, numsims=50)
        @test sum(winners .== "control") == 27 
        @test sum(winners .== "variant 1") ==  22 
        @test sum(winners .== "nothing") == 1 
    end

    @testset "ChainedModel with expected loss with same distribution" begin
        Random.seed!(1234)
        truedists = [
            [Bernoulli(0.1), LogNormal(6.0, 0.8)],
            [Bernoulli(0.1), LogNormal(6.0, 0.8)]
        ]

        modelA = ChainedModel(
            [BernoulliModel(1, 1), LogNormalModel(0.0, 1.0, 0.001, 0.001)],
            [op_multiply]
        )
        modelB = ChainedModel(
            [BernoulliModel(1, 1), LogNormalModel(0.0, 1.0, 0.001, 0.001)],
            [op_multiply]
        )

        stoppingrule = ExpectedLossThresh(0.1)
        experiment = ExperimentAB([modelA, modelB], stoppingrule)

        simulation = Simulation(
            experiment=experiment,
            datagendists=truedists,
            maxsteps=30,
            onestepsizes=[[2000, 400], [1000, 200]]
        )

        numsteps, winners, _ = runsequential(simulation, numsamples=1000, numsims=20)
        @test sum(winners .== "control") ==  5
        @test sum(winners .== "variant 1") == 6 
        @test sum(winners .== "nothing") ==  9
    end

end