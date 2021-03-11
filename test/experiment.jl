using BayesianExperiments
include("utils.jl")

@testset "Expected Loss" begin

    @testset "calculate expected loss" begin
        Random.seed!(1234)
        n = 5
        sampleA = [1, 1, 1, 0, 0]
        sampleB = [1, 0, 0, 1, 0]
        @test calcexpectedloss(sampleA, sampleB, upliftloss, 5) == 0.2
        @test calcexpectedloss(sampleB, sampleA, upliftloss, 5) == 0.4
    end

    @testset "Approximate expected loss from posteriors" begin
        Random.seed!(1234)
        n = 10000
        modelA = BernoulliModel(50, 50)
        modelB = BernoulliModel(60, 40)

        exploss = apprexpectedloss(modelA, modelB, [:θ], lossfunc=upliftloss, numsamples=n)
        @test isapprox(exploss, 0.1, atol=0.01)

        exploss = apprexpectedloss(modelB, modelA, [:θ], lossfunc=upliftloss, numsamples=n)
        @test isapprox(exploss, 0.0, atol=0.01)
    end

    @testset "Approximate expected loss from experiment" begin
        Random.seed!(1234)
        n = 10000
        modelA = BernoulliModel(50, 50)
        modelB = BernoulliModel(60, 40)

        exploss = apprexpectedloss(modelA, modelB, lossfunc=upliftloss, numsamples=n)
        @test isapprox(exploss, 0.1, atol=0.01)

        stoppingrule = ExpectedLossThresh(0.001)
        experiment = ExperimentAB([modelA, modelB], stoppingrule)

        (modelnames, expectedlosses) = apprexpectedlosses(experiment, lossfunc=upliftloss, numsamples=n)

        @test modelnames == ["control", "variant 1"]
        @test isapprox(expectedlosses[1], 0.1, atol=0.01)
        @test isapprox(expectedlosses[2], 0.0, atol=0.01)
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
            [ChainOperator.multiply]
        )
        modelB = ChainedModel(
            [BernoulliModel(1, 1), LogNormalModel(0.0, 1.0, 0.001, 0.001)],
            [ChainOperator.multiply]
        )

        stoppingrule = ExpectedLossThresh(0.001)
        experiment = ExperimentAB([modelA, modelB], stoppingrule)
        update!(experiment, [[statsA1, statsA2], [statsB1, statsB2]])

        modelnames, expected_losses = apprexpectedlosses(experiment)

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
            @test expected_losses[1] ≈ 0.016
            @test expected_losses[2] ≈ 0
        end

        @test selectwinner!(experiment) == "variant 1"
    end
end