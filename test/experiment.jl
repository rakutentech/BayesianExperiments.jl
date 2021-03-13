using BayesianExperiments
include("utils.jl")

@testset "Expected Loss" begin

    @testset "calculate expected loss" begin
        Random.seed!(1234)
        n = 5
        sampleA = [1, 1, 1, 0, 0]
        sampleB = [1, 0, 0, 1, 0]
        @test expectedloss(sampleA, sampleB, upliftloss, 5) == 0.2
        @test expectedloss(sampleB, sampleA, upliftloss, 5) == 0.4
    end

    @testset "Approximate expected loss from posteriors" begin
        Random.seed!(1234)
        n = 10000
        modelA = BernoulliModel(50, 50)
        modelB = BernoulliModel(60, 40)

        exploss = expectedloss(modelA, modelB, [:θ], lossfunc=upliftloss, numsamples=n)
        @test isapprox(exploss, 0.1, atol=0.01)

        exploss = expectedloss(modelB, modelA, [:θ], lossfunc=upliftloss, numsamples=n)
        @test isapprox(exploss, 0.0, atol=0.01)
    end

    @testset "Approximate expected loss from experiment" begin
        Random.seed!(1234)
        n = 10000
        modelA = BernoulliModel(50, 50)
        modelB = BernoulliModel(60, 40)

        exploss = expectedloss(modelA, modelB, lossfunc=upliftloss, numsamples=n)
        @test isapprox(exploss, 0.1, atol=0.01)

        stoppingrule = ExpectedLossThresh(0.001)
        experiment = ExperimentAB([modelA, modelB], stoppingrule)

        (modelnames, explosses) = expectedlosses(experiment, lossfunc=upliftloss, numsamples=n)

        @test modelnames == ["control", "variant 1"]
        @test isapprox(explosses[1], 0.1, atol=0.01)
        @test isapprox(explosses[2], 0.0, atol=0.01)
    end

end

@testset "ExperimentAB" begin
    @testset "ChainedModel with Bernoulli and LogNormal" begin
        Random.seed!(1234)
        eachrunnum = 100000
        truedistA1 = Bernoulli(0.05)
        truedistA2 = LogNormal(1.0, 1.0)
        truedistB1 = Bernoulli(0.05 * 1.05)
        truedistB2 = LogNormal(1.0, 1.0)

        numsims = 50000

        dataA1 = rand(truedistA1, eachrunnum)
        dataA2 = rand(truedistA2, eachrunnum)
        dataB1 = rand(truedistB1, eachrunnum)
        dataB2 = rand(truedistB2, eachrunnum)

        statsA1 = BernoulliStatistics(dataA1)
        statsA2 = LogNormalStatistics(dataA2)
        statsB1 = BernoulliStatistics(dataB1)
        statsB2 = LogNormalStatistics(dataB2)

        modelA = ChainedModel(
            [BernoulliModel(1, 1), LogNormalModel(0.0, 1.0, 0.001, 0.001)],
        )
        modelB = ChainedModel(
            [BernoulliModel(1, 1), LogNormalModel(0.0, 1.0, 0.001, 0.001)],
        )

        stoppingrule = ExpectedLossThresh(0.001)
        experiment = ExperimentAB([modelA, modelB], stoppingrule)
        update!(experiment, [[statsA1, statsA2], [statsB1, statsB2]])

        modelnames, explosses = expectedlosses(experiment)

        @testset "Control Group" begin
            model1_list = experiment.models["control"].models
            @test model1_list[1].dist.α == 4972
            @test model1_list[1].dist.β == 95030
            @test model1_list[2].dist.μ ≈ 1.0 
            @test model1_list[2].dist.v ≈ 0 
            @test model1_list[2].dist.α ≈ 50000
            @test isapprox(model1_list[2].dist.θ, 49373, atol=1.0)
        end

        @testset "Expected loss" begin
            @test modelnames == ["control", "variant 1"]
            @test explosses[1] ≈ 0.016
            @test explosses[2] ≈ 0
        end

        @test decide!(experiment) == "variant 1"
    end
end

@testset "ExperimentBF" begin

    @testset "Normal Statistics batch update" begin
        Random.seed!(12)
        data1 = rand(Normal(0.01, 1), 100_000)
        data2 = rand(Normal(0.03, 2), 100_000);
        s1 = NormalStatistics(data1)
        s2 = NormalStatistics(data2)
        stats1 = update!(s1, s2)
        stats2 = NormalStatistics([data1;data2])

        @test stats1.n == stats2.n
        @test isapprox(stats1.meanx, stats2.meanx, atol=0.0001)
        @test isapprox(stats1.sdx, stats2.sdx, atol=0.0001)
    end

end