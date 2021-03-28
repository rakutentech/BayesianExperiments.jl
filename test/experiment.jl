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
        modelA = ConjugateBernoulli(50, 50)
        modelB = ConjugateBernoulli(60, 40)

        exploss = expectedloss(modelA, modelB, [:θ], lossfunc=upliftloss, numsamples=n)
        @test isapprox(exploss, 0.1, atol=0.01)

        exploss = expectedloss(modelB, modelA, [:θ], lossfunc=upliftloss, numsamples=n)
        @test isapprox(exploss, 0.0, atol=0.01)
    end

    @testset "Approximate expected loss from experiment" begin
        Random.seed!(1234)
        n = 10000
        modelA = ConjugateBernoulli(50, 50)
        modelB = ConjugateBernoulli(60, 40)

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
            [ConjugateBernoulli(1, 1), ConjugateLogNormal(0.0, 1.0, 0.001, 0.001)],
        )
        modelB = ChainedModel(
            [ConjugateBernoulli(1, 1), ConjugateLogNormal(0.0, 1.0, 0.001, 0.001)],
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

        winner, _ = decide!(experiment)
        @test winner == "variant 1"
    end


end

@testset "Number of models" begin
    # Generate sample data
    n = 1000
    dataA = rand(Bernoulli(0.150), n)
    dataB = rand(Bernoulli(0.145), n)
    dataC = rand(Bernoulli(0.180), n)

    # Define the models
    modelA = ConjugateBernoulli(1, 1)
    modelB = ConjugateBernoulli(1, 1)
    modelC = ConjugateBernoulli(1, 1)

    stoppingrule = ProbabilityBeatAllThresh(0.99)
    @test_throws ArgumentError("Number of models needs to be equal to 2.") ExperimentAB([modelA, modelB, modelC], stoppingrule)
end


@testset "ExperiemntBF{NormalEffectSize}" begin
    @testset "Bayes factor calculation" begin
        # the example is taken from 
        # https://statswithr.github.io/book/hypothesis-testing-with-normal-populations.html
        n0 = 32.7^2
        m0 = 0.5
        x̄ = 0.500177
        σ = 0.5
        n = 1.0449e8;
        σ0 = 1/sqrt(n0)

        model = NormalEffectSize(m0, σ0)
        stats = NormalStatistics(n=1.0449e8, meanx=x̄, sdx=σ)
        bf10 = bayesfactor(model, stats)
        @test isapprox(bf10, 2.2303, rtol=0.001)
    end

    @testset "Two samples with equal size and sd" begin
        n0 = 32.7^2
        m0 = 0.5 # null hypothesis: difference is 0.5 
        σ = 0.5
        n = 1.0449e8;
        σ0 = 1/sqrt(n0)

        # 2*n so that the effective sample size will be `n`
        stats1 = NormalStatistics(n=2*n, meanx=1.000177, sdx=σ)
        stats2 = NormalStatistics(n=2*n, meanx=0.5, sdx=σ)
        stats = TwoNormalStatistics(stats1, stats2)

        model = NormalEffectSize(m0, σ0)
        bf10 = bayesfactor(model, stats)
        @test isapprox(bf10, 2.2303, rtol=0.001)
    end
end

@testset "ExperiemntBF{StudentTEffectSize}" begin
    @testset "Two-Sided null wins, thresh=5" begin
        normalstats = TwoNormalStatistics(
            NormalStatistics(meanx=28.8, sdx=13.5, n=133),
            NormalStatistics(meanx=30.6, sdx=14.3, n=867)
        )
        model = StudentTEffectSize(r=1.0)
        stats = StudentTStatistics(normalstats)
        bf10_model = bayesfactor(model, stats)

        thresh=5
        stoppingrule = TwoSidedBFThresh(thresh)
        experiment = ExperimentBF(model=model, rule=stoppingrule)
        update!(experiment, normalstats)
        bf10_exp = bayesfactor(experiment)

        @test bf10_model == bf10_exp
        @test isapprox(bf10_exp, 1/5.45, rtol=0.01)

        winner, _ = decide!(experiment)
        @test winner == "null"
    end

    @testset "Two-Sided alternative wins, thresh=5" begin
        normalstats = TwoNormalStatistics(
            NormalStatistics(meanx=28.8, sdx=13.5, n=133),
            NormalStatistics(meanx=33.6, sdx=14.3, n=867)
        )
        model = StudentTEffectSize(r=1.0)
        stats = StudentTStatistics(normalstats)
        bf10_model = bayesfactor(model, stats)

        thresh=5
        stoppingrule = TwoSidedBFThresh(thresh)
        experiment = ExperimentBF(model=model, rule=stoppingrule)
        update!(experiment, normalstats)
        bf10_exp = bayesfactor(experiment)

        @test bf10_model == bf10_exp
        @test isapprox(bf10_exp, 46.6077, rtol=0.01)

        winner, _ = decide!(experiment)
        @test winner == "alternative"
    end

    @testset "Two-Sided no winners, thresh=0.01" begin
        normalstats = TwoNormalStatistics(
            NormalStatistics(meanx=28.8, sdx=13.5, n=133),
            NormalStatistics(meanx=30.6, sdx=14.3, n=867)
        )
        model = StudentTEffectSize(r=1.0)
        stats = StudentTStatistics(normalstats)
        bf10_model = bayesfactor(model, stats)

        thresh=20
        stoppingrule = TwoSidedBFThresh(thresh)
        experiment = ExperimentBF(model=model, rule=stoppingrule)
        update!(experiment, normalstats)
        bf10_exp = bayesfactor(experiment)

        @test bf10_model == bf10_exp
        @test isapprox(bf10_exp, 1/5.45, rtol=0.01)

        winner, _ = decide!(experiment)
        @test winner === nothing 
    end

    @testset "One-Sided no winner, thresh=5" begin
        normalstats = TwoNormalStatistics(
            NormalStatistics(meanx=28.8, sdx=13.5, n=133),
            NormalStatistics(meanx=30.6, sdx=14.3, n=867)
        )
        model = StudentTEffectSize(r=1.0)
        stats = StudentTStatistics(normalstats)
        bf10_model = bayesfactor(model, stats)

        thresh=5
        stoppingrule = OneSidedBFThresh(thresh)
        experiment = ExperimentBF(model=model, rule=stoppingrule)
        update!(experiment, normalstats)
        bf10_exp = bayesfactor(experiment)

        @test bf10_model == bf10_exp
        @test isapprox(bf10_exp, 1/5.45, rtol=0.01)

        winner, _ = decide!(experiment)
        @test winner === nothing 
    end

    @testset "One-Sided alternative wins, thresh=5" begin
        normalstats = TwoNormalStatistics(
            NormalStatistics(meanx=28.8, sdx=13.5, n=133),
            NormalStatistics(meanx=33.6, sdx=14.3, n=867)
        )
        model = StudentTEffectSize(r=1.0)
        stats = StudentTStatistics(normalstats)
        bf10_model = bayesfactor(model, stats)

        thresh=5
        stoppingrule = OneSidedBFThresh(thresh)
        experiment = ExperimentBF(model=model, rule=stoppingrule)
        update!(experiment, normalstats)
        bf10_exp = bayesfactor(experiment)

        @test bf10_model == bf10_exp
        @test isapprox(bf10_exp, 46.6077, rtol=0.01)

        winner, _ = decide!(experiment)
        @test winner == "alternative"
    end
end